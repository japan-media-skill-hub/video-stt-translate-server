"""translate 阶段执行器。"""

from __future__ import annotations

from pathlib import Path
from queue import Queue
from typing import Any, Callable
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from datetime import datetime, timezone

import requests

from whisper_stt_service.executor.atomic import (
    StaleWriteRejected,
    atomic_write_text,
    atomic_write_text_fenced,
)
from whisper_stt_service.executor.common import (
    ExecutorCancelledError,
    SrtEntry,
    TranslateBatch,
    TranslateSettings,
    _build_translate_messages,
    _dump_srt,
    _emit_progress,
    _extract_json_object,
    _load_llm_config,
    _parse_srt,
    _sha256_file,
    build_translate_batches,
    resolve_translate_settings,
)
from whisper_stt_service.service.metrics import (
    MetricsState,
    record_translate_batch,
    record_translate_retry,
)


logger = logging.getLogger(__name__)

# 检查点结构版本：结构不兼容时旧检查点会被丢弃并全量重算。
CHECKPOINT_VERSION = 1


def _load_llm_profiles(
    config, overrides: dict[str, str] | None = None
) -> list[dict[str, str]]:
    """加载可用 LLM 配置列表：主配置 + 可选回退配置。

    `overrides` 为**仅内存传递**的连接覆盖（`base_url`/`api_key`/`model`），
    仅作用于主配置段 `[llm]`；回退配置段保持原样。未提供的键回退 config.ini 值。
    该机制让脚本模式 CLI 能兼容 `--base-url/--api-key/--model`，
    而无需把真实凭据写入任何临时配置文件。
    """

    safe_overrides: dict[str, str] = {
        key: str(value).strip()
        for key, value in (overrides or {}).items()
        if key in {"base_url", "api_key", "model"} and str(value).strip()
    }

    profiles: list[dict[str, str]] = []

    def _append_profile(section: str) -> None:
        """把某个配置段作为候选 LLM；base_url/api_key/model 任一缺失即跳过。"""

        base_url = config.get(section, "base_url", fallback="").strip()
        api_key = config.get(section, "api_key", fallback="").strip()
        model = config.get(section, "model", fallback="").strip()
        if section == "llm" and safe_overrides:
            # 命令行覆盖只作用于主配置段，优先级高于 config.ini。
            base_url = safe_overrides.get("base_url", base_url)
            api_key = safe_overrides.get("api_key", api_key)
            model = safe_overrides.get("model", model)
        if not base_url or not api_key or not model:
            return
        profiles.append(
            {
                "section": section,
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
            }
        )

    if config.has_section("llm"):
        _append_profile("llm")

    for section in ("llm_fallback", "llm_backup", "llm_secondary"):
        if not config.has_section(section):
            continue
        enabled = config.getboolean(section, "enabled", fallback=True)
        if not enabled:
            continue
        _append_profile(section)

    # 去重：同 base/model/api_key 的重复配置仅保留第一条。
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for p in profiles:
        key = (p["base_url"], p["model"], p["api_key"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def _pick_llm_profile(profiles: list[dict[str, str]], attempt: int) -> dict[str, str]:
    """按重试轮次选择 LLM：首轮主模型，后续优先回退模型。"""

    if not profiles:
        raise RuntimeError("no_llm_profiles")
    if len(profiles) == 1:
        return profiles[0]
    if attempt <= 1:
        return profiles[0]
    return profiles[1]


class TranslateCancelledError(ExecutorCancelledError):
    """翻译被上游取消（失租/停机/timeout）：中止且不得写回结果。

    继承公共的 `ExecutorCancelledError`，runtime 可用统一类型判断取消。
    """


class TranslateNonRetryableError(RuntimeError):
    """4xx 等不可重试错误：直接失败，避免无意义计费。"""


class TranslatePartialIdsError(ValueError):
    """模型返回的 id 集合与请求不完全一致（缺失或越界）。"""

    def __init__(
        self,
        missing: list[int],
        unexpected: list[int],
        partial: dict[int, str] | None = None,
    ) -> None:
        """记录缺失与多余的 id，并保留已成功解析的译文子集。

        `partial` 必须随异常一起带出：否则调用方拿不到"已成功的那部分"，
        C.5 的"先接受已有、只补缺失"就会退化为整批重发并把已得译文丢掉。
        """

        self.missing = missing
        self.unexpected = unexpected
        self.partial: dict[int, str] = dict(partial or {})
        super().__init__(
            f"batch translation ids mismatch, missing={missing[:10]} "
            f"unexpected={unexpected[:10]}"
        )


def _call_translate_api(
    post_func: Callable[..., requests.Response],
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout_sec: int,
    batch: list[SrtEntry],
    idempotency_key: str | None = None,
) -> dict[int, str]:
    """调用 LLM 翻译单个批次并返回 id -> 中文文本映射。

    返回项按 id 子集接受（仅接受本批出现的 id）；缺失或越界 id 抛
    `TranslatePartialIdsError`，由调用方决定重试或降级保留原文。
    传入 `idempotency_key` 时附加 `Idempotency-Key` 头，便于上游去重计费。
    """

    messages = _build_translate_messages(batch)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    response = post_func(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "temperature": 0.2,
            "messages": messages,
        },
        timeout=timeout_sec,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    parsed = _extract_json_object(content)
    if not isinstance(parsed, list):
        raise ValueError("model output is not a JSON array")

    expected = {e.index for e in batch}
    mapping: dict[int, str] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        text_zh = item.get("text_zh")
        # 仅接受本批 id，避免模型串批写入错误字幕。
        if (
            isinstance(item_id, int)
            and item_id in expected
            and isinstance(text_zh, str)
            and text_zh.strip()
        ):
            mapping[item_id] = text_zh.strip()
    missing = sorted(expected - set(mapping.keys()))
    unexpected = sorted(
        {
            item.get("id")
            for item in parsed
            if isinstance(item, dict)
            and isinstance(item.get("id"), int)
            and item["id"] not in expected
        }
    )
    if missing or unexpected:
        # C.5：把"已成功解析的子集"一并抛出，调用方据此只补缺失 id 并保留已得译文。
        raise TranslatePartialIdsError(
            missing=missing, unexpected=unexpected, partial=mapping
        )
    return mapping


def run_translate(
    input_ja_srt: Path,
    output_zh_srt: Path,
    config_path: Path,
    timeout_sec: int,
    *,
    input_video_path: Path,
    copy_back: str | None = None,
    chunk_minutes: int | None = None,
    retry: int | None = None,
    progress_queue: Queue[dict[str, Any]] | None = None,
    task_id: str | None = None,
    worker_id: str | None = None,
    job_config: dict[str, Any] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    post_func: Callable[..., requests.Response] | None = None,
    job_id: str | None = None,
    metrics: MetricsState | None = None,
    checkpoint_path: Path | None = None,
    defer_copy_back: bool = False,
    checkpoint_publish: Callable[[Path, str], bool] | None = None,
    llm_overrides: dict[str, str] | None = None,
) -> None:
    """直接在 service 内部执行翻译，并把进度写入队列。

    批次以受限线程池并发执行（并发上限 = `parallel`，默认 2），结果按批次
    序号与原文顺序聚合写回；每批成功后立即原子落检查点，重跑只补未完成批次。

    `job_config` 为 task 的 `job_config.translate` 段（优先级最高）；
    `cancel_check` 供 runtime 在失租/停机时中止翻译：返回 True 后不再派发新批次、
    不再发布进度或写入产物，并抛出 `TranslateCancelledError` 交由上层放弃结果写回；
    `post_func` 用于测试注入 fake HTTP 客户端。
    `job_id` 用于构造幂等键；未传时回退 `task_id`（兼容旧调用），推荐 runtime 显式传入，
    这样同一 job 的多次 task 重试（attempt 变化）不会改变幂等键、不会重复计费。
    `metrics` 为进程指标句柄；未传时跳过指标记录（不影响翻译）。

    产物级 fencing（P0）相关参数：
    `checkpoint_path` 指定检查点位置；默认仍为 `<output>.progress.json`，
    runtime 传入 canonical 稳定路径即可跨 attempt 续跑（staging 输出每次 attempt 不同）。
    `defer_copy_back=True` 时不回写视频目录，由 runtime 在 promotion 成功后统一负责。
    `checkpoint_publish` 为租约 fenced 的检查点发布回调 `(path, text) -> bool`：
    回调在 BEGIN 事务中确认 `lease_owner`/`attempt` 后执行原子写；返回 True 表示
    发布已全权完成（helper 不再锁外重写），返回 False 表示本写入方已 stale，
    此时期望本次 attempt 放弃产物（抛 `StaleWriteRejected`）。
    `llm_overrides` 为**仅内存传递**的 LLM 连接覆盖（`base_url`/`api_key`/`model`），
    供脚本模式 CLI 兼容 `--base-url/--api-key/--model` 而**不把任何凭据写入临时文件**；
    键缺失时回退 config.ini 的 `[llm]` 值。
    """

    config = _load_llm_config(config_path)
    settings = resolve_translate_settings(
        config,
        job_config,
        chunk_minutes=chunk_minutes,
        retry=retry,
        timeout_sec=timeout_sec,
        copy_back=copy_back,
    )
    llm_profiles = _load_llm_profiles(config, llm_overrides)
    if not llm_profiles:
        raise RuntimeError("missing llm config: require [llm] base_url/api_key/model")

    # 真正生效的配置全部来自 settings（job_config > [translation] > 默认值）。
    effective_retry = settings.retry
    effective_timeout_sec = settings.timeout_sec
    request_interval = settings.request_interval
    parallel = settings.parallel

    source_text = input_ja_srt.read_text(encoding="utf-8")
    entries = _parse_srt(source_text)
    if not entries:
        raise RuntimeError(f"no valid srt entries found: {input_ja_srt}")

    # 源内容摘要用于检查点失效判定与幂等键，避免误复用过期进度。
    source_sha256 = _sha256_file(input_ja_srt)
    batches = build_translate_batches(
        entries,
        window_minutes=settings.chunk_minutes,
        batch_size=settings.batch_size,
        max_chars=settings.max_chars,
    )

    # 历史注释保留，现为临时文件+原子替换/源hash检查点，绝不删正式旧产物
    # 先清理目标字幕与翻译进度副产物，保障可恢复可重跑。
    # 检查点位置：runtime 可传 canonical 稳定路径，跨 attempt 续跑；
    # 默认保持旧行为（<output>.progress.json），对现有调用完全兼容。
    if checkpoint_path is not None:
        progress_artifact = Path(checkpoint_path)
    else:
        progress_artifact = output_zh_srt.with_suffix(
            output_zh_srt.suffix + ".progress.json"
        )

    _emit_progress(
        progress_queue,
        stage="translate",
        percent=0.0,
        message="translate_started",
        task_id=task_id,
        worker_id=worker_id,
    )

    logger.info(
        "translate started: task_id=%s batches=%d entries=%d parallel=%d "
        "batch_size=%d max_chars=%d chunk_minutes=%d retry=%d checkpoint=%s",
        task_id,
        len(batches),
        len(entries),
        parallel,
        settings.batch_size,
        settings.max_chars,
        settings.chunk_minutes,
        effective_retry,
        settings.checkpoint_enabled,
    )

    runner = _TranslateRunner(
        settings=settings,
        llm_profiles=llm_profiles,
        batches=batches,
        entries=entries,
        progress_artifact=progress_artifact,
        output_zh_srt=output_zh_srt,
        source_sha256=source_sha256,
        job_id=(job_id or task_id or ""),
        deadline_at=time.perf_counter() + effective_timeout_sec,
        request_interval=request_interval,
        parallel=parallel,
        progress_queue=progress_queue,
        task_id=task_id,
        worker_id=worker_id,
        cancel_check=cancel_check,
        post_func=post_func,
        metrics=metrics,
        checkpoint_publish=checkpoint_publish,
    )
    translations = runner.execute()

    # 产物级 fencing：写正式 zh.srt 前再确认一次取消信号，失租后不得覆盖正式产物。
    _raise_if_cancelled(cancel_check)

    # 仅在全量翻译成功后覆盖 zh.srt；原子写保证不留半截字幕。
    _dump_srt(entries, translations, output_zh_srt)

    if defer_copy_back:
        # runtime 使用 staging 输出 + 事务 promotion：executor 不得提前回写正式目录。
        logger.info(
            "copy_back deferred to runtime promotion: task_id=%s output=%s",
            task_id,
            output_zh_srt,
        )
    else:
        configured_copy_back = (settings.copy_back or "").strip()
        copy_back_dir = _resolve_copy_back_dir(
            configured_copy_back,
            config_path=config_path,
            input_video_path=input_video_path,
        )
        _copy_back_subtitles(input_ja_srt, output_zh_srt, copy_back_dir)

    # 任务成功完成后保留检查点 progress_ttl_sec，之后由清理逻辑删除；
    # 期间若任务被重试可直接复用，无需重翻。
    if settings.checkpoint_enabled:
        _touch_checkpoint_retention(progress_artifact, settings.progress_ttl_sec)

    _emit_progress(
        progress_queue,
        stage="translate",
        percent=100.0,
        message="translate_done",
        task_id=task_id,
        worker_id=worker_id,
    )


# 挂起在途批次的最长等待：中止后不再等待新批次完成，交由上层重试。
_INFLIGHT_JOIN_TIMEOUT_SEC = 0.5


def _error_class(exc: Exception) -> str:
    """把异常归类为可审计的 error_class（用于重试日志与成本核对）。"""

    if isinstance(exc, requests.Timeout):
        return "timeout"
    if isinstance(exc, requests.ConnectionError):
        return "connection"
    if isinstance(exc, requests.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(status, int):
            return f"http_{status}"
        return "http"
    if isinstance(exc, TranslatePartialIdsError):
        return "ids_mismatch"
    if isinstance(exc, (ValueError, KeyError, IndexError, json.JSONDecodeError)):
        return "parse"
    if isinstance(exc, (TimeoutError,)):
        return "timeout"
    return exc.__class__.__name__.lower()


def _is_retryable(exc: Exception) -> bool:
    """判定异常是否值得重试：仅连接错误、超时、5xx、JSON 解析失败、id 不匹配。"""

    if isinstance(exc, (ExecutorCancelledError, TranslateCancelledError)):
        return False
    if isinstance(exc, TranslateNonRetryableError):
        return False
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if not isinstance(status, int):
            return True
        # SPEC C.6：只重试 5xx；4xx（含 429 限流）一律不重试，避免无意义计费。
        return status >= 500
    if isinstance(exc, TranslatePartialIdsError):
        return True
    if isinstance(exc, (ValueError, KeyError, IndexError, json.JSONDecodeError)):
        return True
    return False


def _is_non_retryable_http(exc: Exception) -> bool:
    """是否为 4xx 客户端错误（含 429 限流；一旦命中立即失败，不进入重试）。"""

    if isinstance(exc, requests.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(status, int) and 400 <= status < 500:
            return True
    return False


def _raise_if_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    """产物级取消检查：写正式产出前确认本次 attempt 仍拥有执行权。"""

    if cancel_check is not None and cancel_check():
        raise TranslateCancelledError("translate_cancelled_before_publish")


def _sub_batch_id(batch_id: str, ids: list[int]) -> str:
    """计算本次请求的 sub-batch 标识。

    完整批次（请求集合等于原批）沿用 SPEC 的 `<window>/<part>` 作为 batch_id；
    仅补缺失 id 时，payload 已变化，必须使用带缺失 id 集合的独立后缀
    （`<batch_id>#missing-1,2`），否则同一幂等键会命中上游对旧 payload 的缓存响应。
    """

    missing_token = ",".join(str(i) for i in ids)
    return f"{batch_id}#missing-{missing_token}"


def _build_idempotency_key(
    job_id: str, sub_batch_id: str, source_sha256: str
) -> str:
    """生成幂等键 sha256(job_id|sub_batch_id|source_sha256)，便于上游去重计费。

    同一 payload 的重复尝试（完整批次重试 / 缺失子集重试）key 保持稳定；
    payload 一旦变化（补缺失），`sub_batch_id` 随之变化，key 也不同。
    """

    payload = f"{job_id}|{sub_batch_id}|{source_sha256}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_checkpoint_payload(
    *,
    source_sha256: str,
    settings: TranslateSettings,
    batch_states: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """构造检查点文件内容（C.4 约定结构 + inflight 标记）。"""

    return {
        "version": CHECKPOINT_VERSION,
        "source_sha256": source_sha256,
        "chunk_minutes": settings.chunk_minutes,
        "max_chars": settings.max_chars,
        "batch_size": settings.batch_size,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "batches": {k: dict(v) for k, v in batch_states.items()},
    }


def _load_checkpoint(
    path: Path,
    *,
    source_sha256: str,
    settings: TranslateSettings,
) -> dict[str, dict[str, Any]]:
    """读取检查点并按“源内容 + 切分配置”校验；不匹配时返回空进度。

    复用条件：version 受支持、source_sha256 与当前 ja.srt 一致、
    chunk_minutes/max_chars/batch_size 与本次生效配置一致。任一不符即全量重算。
    """

    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(
            "translate checkpoint unreadable, full recompute: path=%s error=%s",
            path,
            exc,
        )
        return {}
    if not isinstance(payload, dict):
        return {}

    reasons: list[str] = []
    if payload.get("version") != CHECKPOINT_VERSION:
        reasons.append(f"version={payload.get('version')}")
    if payload.get("source_sha256") != source_sha256:
        reasons.append("source_changed")
    for key in ("chunk_minutes", "max_chars", "batch_size"):
        if key in payload and payload.get(key) != getattr(settings, key):
            reasons.append(f"{key}_changed")
    batches = payload.get("batches")
    if not isinstance(batches, dict):
        reasons.append("batches_missing")

    if reasons:
        # 源文件或配置变更时不得复用旧进度，否则会写回错位字幕。
        logger.info(
            "translate checkpoint ignored, full recompute: path=%s reasons=%s",
            path,
            ",".join(reasons),
        )
        return {}

    restored: dict[str, dict[str, Any]] = {}
    for batch_id, state in batches.items():
        if not isinstance(batch_id, str) or not isinstance(state, dict):
            continue
        if not state.get("done"):
            continue
        texts = state.get("translations")
        # 只认"有实际译文内容"的批次：仅有 done=true 而译文为空/不全的检查点必须重算。
        if not isinstance(texts, dict) or not texts:
            logger.info(
                "translate checkpoint batch skipped (no translations): batch_id=%s",
                batch_id,
            )
            continue
        try:
            parsed_texts = {
                int(k): str(v)
                for k, v in texts.items()
                if str(v).strip()
            }
        except (TypeError, ValueError) as exc:
            logger.warning(
                "translate checkpoint batch skipped (bad ids): batch_id=%s error=%s",
                batch_id,
                exc,
            )
            continue
        recorded_ids = {
            int(i) for i in (state.get("ids") or []) if isinstance(i, int)
        }
        # 记录的 id 必须在译文里有对应内容，否则视为不完整批次并重算。
        if not recorded_ids or not recorded_ids.issubset(parsed_texts.keys()):
            logger.info(
                "translate checkpoint batch skipped (incomplete ids): batch_id=%s "
                "recorded=%d restored=%d",
                batch_id,
                len(recorded_ids),
                len(parsed_texts),
            )
            continue
        restored[batch_id] = {
            "done": True,
            "ids": sorted(recorded_ids),
            "translations": parsed_texts,
        }
    if restored:
        logger.info(
            "translate checkpoint reused: path=%s done_batches=%d",
            path,
            len(restored),
        )
    return restored


def _save_checkpoint(
    path: Path,
    payload: dict[str, Any],
    *,
    publish: Callable[[Path, str], bool] | None = None,
) -> None:
    """原子写检查点：写入过程被中断也不会留下半截 JSON。

    传入 `publish` 时走 fenced 写入（runtime 在事务内确认租约与 attempt）；
    回调返回 True 表示它已在事务内全权落盘，本函数不再锁外重写；回调返回 False
    即抛 `StaleWriteRejected`，stale worker 的检查点不会落盘。
    """

    atomic_write_text_fenced(
        path,
        json.dumps(payload, ensure_ascii=False, indent=None, sort_keys=True),
        publish,
    )


def _remove_checkpoint(path: Path) -> None:
    """删除检查点文件；失败仅告警，不影响翻译主流程。"""

    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("translate checkpoint cleanup failed: path=%s error=%s", path, exc)


def _touch_checkpoint_retention(path: Path, ttl_sec: int) -> None:
    """任务成功后保留检查点 ttl_sec：刷新 mtime 供 TTL 清理逻辑判定过期。"""

    if not path.is_file():
        return
    try:
        os.utime(path, None)
    except OSError as exc:
        logger.warning(
            "translate checkpoint retention touch failed: path=%s error=%s", path, exc
        )
        return
    logger.info(
        "translate checkpoint retained: path=%s ttl_sec=%d", path, ttl_sec
    )


class _DispatchThrottle:
    """批次派发节流：保证相邻两次请求发出间隔不小于 request_interval。

    仅当同时有并发批次时才需要等待，因此用 Lock + 时间戳实现即可；
    request_interval <= 0 时完全关闭（D2：不引入额外令牌桶）。
    """

    def __init__(self, interval_sec: float) -> None:
        """记录派发间隔与上次派发时间点。"""

        self._interval = max(float(interval_sec), 0.0)
        self._lock = threading.Lock()
        self._last_at: float | None = None

    def wait_turn(self, cancel_check: Callable[[], bool] | None = None) -> None:
        """阻塞直到可以派发下一批；期间按小步长检查取消信号。"""

        if self._interval <= 0:
            return
        while True:
            with self._lock:
                now = time.perf_counter()
                if self._last_at is None:
                    self._last_at = now
                    return
                remaining = self._interval - (now - self._last_at)
                if remaining <= 0:
                    self._last_at = now
                    return
            if cancel_check is not None and cancel_check():
                raise TranslateCancelledError("translate_cancelled")
            time.sleep(min(remaining, 0.05))


class _ThreadLocalSession:
    """按线程创建 requests.Session，并统一设置 trust_env = not disable_proxy。"""

    def __init__(self, disable_proxy: bool) -> None:
        """保存代理开关，延迟到首次使用时再创建 Session。"""

        self._disable_proxy = bool(disable_proxy)
        self._local = threading.local()

    def post(self, *args: Any, **kwargs: Any) -> requests.Response:
        """使用当前线程独占的 Session 发起请求。"""

        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            # disable_proxy=True 时忽略环境变量中的代理；False 时遵守环境代理。
            session.trust_env = not self._disable_proxy
            self._local.session = session
        return session.post(*args, **kwargs)

    def close(self) -> None:
        """关闭当前线程持有的 Session（未创建时无操作）。"""

        session = getattr(self._local, "session", None)
        if session is not None:
            session.close()
            self._local.session = None


class _TranslateRunner:
    """受限并发翻译执行器：限流派发、重试、检查点与取消检查集中在这里。"""

    def __init__(
        self,
        *,
        settings: TranslateSettings,
        llm_profiles: list[dict[str, str]],
        batches: list[TranslateBatch],
        entries: list[SrtEntry],
        progress_artifact: Path,
        output_zh_srt: Path,
        source_sha256: str,
        job_id: str,
        deadline_at: float,
        request_interval: float,
        parallel: int,
        progress_queue: Queue[dict[str, Any]] | None,
        task_id: str | None,
        worker_id: str | None,
        cancel_check: Callable[[], bool] | None,
        post_func: Callable[..., requests.Response] | None,
        metrics: MetricsState | None = None,
        checkpoint_publish: Callable[[Path, str], bool] | None = None,
    ) -> None:
        """保存本次翻译的全部生效配置与线程安全共享状态。

        `metrics` 为可选 Prometheus 句柄（SPEC G 批次/重试指标）；为 None 时
        不记录任何指标，保证脚本模式与纯单测无 side effect。
        """

        self.settings = settings
        self.llm_profiles = llm_profiles
        self.batches = batches
        self.entries = entries
        self.progress_artifact = progress_artifact
        self.output_zh_srt = output_zh_srt
        self.source_sha256 = source_sha256
        self.job_id = job_id
        self.deadline_at = deadline_at
        self.parallel = max(int(parallel), 1)
        self.progress_queue = progress_queue
        self.task_id = task_id
        self.worker_id = worker_id
        self.cancel_check = cancel_check
        self._post_func = post_func
        self._metrics = metrics
        self._checkpoint_publish = checkpoint_publish

        # 保留请求节流行为，避免对上游接口形成瞬时冲击。
        self._throttle = _DispatchThrottle(request_interval)
        self._session = _ThreadLocalSession(settings.disable_proxy)

        # 保护 translations / batch_states / 失败状态 / 进度发布。
        self._lock = threading.Lock()
        self._translations: dict[int, str] = {}
        self._batch_states: dict[str, dict[str, Any]] = {}
        self._completed_batches = 0
        self._first_error: BaseException | None = None
        # _aborted：本次运行已决定失败，不再派发新批次（在途批次仍可落检查点）。
        self._aborted = False
        # _cancelled：失租/停机取消，连在途批次也不得写盘。
        self._cancelled = False
        self._inflight_requests = 0

    # ------------------------------------------------------------------ 公共入口

    def execute(self) -> dict[int, str]:
        """执行全部批次并返回 id -> 译文映射；异常时抛出首个失败原因。"""

        if self.settings.checkpoint_enabled:
            self._batch_states = _load_checkpoint(
                self.progress_artifact,
                source_sha256=self.source_sha256,
                settings=self.settings,
            )
            for state in self._batch_states.values():
                self._translations.update(state.get("translations", {}))
            # 只统计 done 的批次；inflight（上次中断）批次仍需重新请求。
            self._completed_batches = sum(
                1 for b in self.batches if self._is_done(b.batch_id)
            )

        pending = [b for b in self.batches if not self._is_done(b.batch_id)]

        if self._completed_batches:
            self._publish_progress(message="translate_resumed")

        if not pending:
            return dict(self._translations)

        max_workers = min(self.parallel, len(pending))
        logger.info(
            "translate dispatch: task_id=%s pending_batches=%d completed_batches=%d "
            "workers=%d",
            self.task_id,
            len(pending),
            self._completed_batches,
            max_workers,
        )

        executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="translate"
        )
        futures = {}
        try:
            for batch in pending:
                futures[executor.submit(self._run_batch, batch)] = batch
            # 失败后立即停止派发：尚未开始的 future 会被取消。
            done, not_done = wait(futures, return_when=FIRST_EXCEPTION)
            for future in done:
                if future.cancelled():
                    continue
                exc = future.exception()
                if exc is not None:
                    self._mark_aborted()
                    for pending_future in not_done:
                        pending_future.cancel()
                    break
            if self._first_error is None:
                # 等待在途批次收尾（正常路径下此时已全部结束）。
                wait(not_done, timeout=_INFLIGHT_JOIN_TIMEOUT_SEC)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            self._session.close()

        if self._first_error is not None:
            raise self._first_error

        missing = [b.batch_id for b in self.batches if not self._is_done(b.batch_id)]
        if missing:
            raise RuntimeError(
                f"translate_incomplete_batches: {missing[:10]} "
                f"(total={len(missing)})"
            )
        return dict(self._translations)

    # ------------------------------------------------------------------ 批次执行

    def _is_done(self, batch_id: str) -> bool:
        """判断批次是否已在检查点中完成。"""

        state = self._batch_states.get(batch_id)
        return bool(state and state.get("done"))

    def _check_cancel(self) -> None:
        """取消信号检查点：失租/停机后尽快中止，不再写结果。"""

        if self.cancel_check is not None and self.cancel_check():
            raise TranslateCancelledError("translate_cancelled")
        if time.perf_counter() > self.deadline_at:
            raise TranslateCancelledError("translate_timeout")

    def _mark_aborted(self) -> None:
        """标记中止：后续线程不再派发新批次。

        注意这里**不**阻止在途成功批次写检查点（C.1 失败收尾契约）；
        需要完全禁写时使用 `_mark_cancelled`。
        """

        with self._lock:
            self._aborted = True

    def _mark_cancelled(self) -> None:
        """标记取消（失租/停机）：在途批次也不得写检查点或发布进度。"""

        with self._lock:
            self._aborted = True
            self._cancelled = True

    def _run_batch(self, batch: TranslateBatch) -> None:
        """在工作线程中执行单个批次；异常统一记录到首个错误槽位。

        取消异常先登记为首个错误再上抛：`execute()` 据此停止派发并放弃结果；
        被取消而未开始的批次由 `cancel_futures` 处理，不会进入这里。
        """

        if self._aborted:
            # 已中止：直接放弃，避免占用线程继续请求上游。
            return
        try:
            self._translate_batch(batch)
        except BaseException as exc:  # noqa: BLE001
            with self._lock:
                if self._first_error is None:
                    self._first_error = exc
                self._aborted = True
                # 取消（失租/停机）必须连带禁写；普通批次失败只停止派发。
                if isinstance(exc, (ExecutorCancelledError, TranslateCancelledError)):
                    self._cancelled = True
            raise

    def _translate_batch(self, batch: TranslateBatch) -> None:
        """批次内重试循环：成功即原子落检查点，部分缺失按 C.5 降级。"""

        last_err: Exception | None = None
        # 标记最后失败是否属于“仅部分 id 缺失”：只有这种情况才允许降级保留原文。
        partial_only = False
        pending_entries = list(batch.entries)
        accepted: dict[int, str] = {}

        for attempt in range(1, self.settings.retry + 1):
            self._check_cancel()
            if not pending_entries:
                break
            llm_profile = _pick_llm_profile(self.llm_profiles, attempt)
            # 幂等键：完整批次沿用 SPEC 的 batch_id；补缺失时用带 id 集合的子批次 id，
            # 避免 payload 变化后仍命中上游对旧响应/旧计费的缓存。
            sub_batch_id = (
                batch.batch_id
                if not accepted
                else _sub_batch_id(batch.batch_id, pending_entries)
            )
            idempotency_key = _build_idempotency_key(
                self.job_id, sub_batch_id, self.source_sha256
            )
            if self.settings.checkpoint_enabled:
                # 请求发出前先标记 inflight：超时/崩溃后重跑可识别该批。
                self._record_inflight(batch, pending_entries, attempt)
            attempt_started = time.perf_counter()
            try:
                mapped = self._invoke_api(
                    llm_profile=llm_profile,
                    batch_entries=pending_entries,
                    idempotency_key=idempotency_key,
                    attempt=attempt,
                    batch_id=batch.batch_id,
                )
            except Exception as exc:  # noqa: BLE001
                elapsed_ms = int((time.perf_counter() - attempt_started) * 1000)
                error_class = _error_class(exc)
                if isinstance(exc, (ExecutorCancelledError, TranslateCancelledError)):
                    # 取消不是失败：直接上抛，绝不参与重试或降级。
                    raise
                if isinstance(exc, TranslatePartialIdsError) and exc.partial:
                    # C.5 真实路径：先收下已返回的译文，再只对缺失 id 用小批次补。
                    accepted.update(exc.partial)
                    pending_entries = [
                        e for e in pending_entries if e.index not in accepted
                    ]
                    partial_only = True
                    last_err = exc
                    if not pending_entries:
                        break
                    logger.warning(
                        "translate_partial: job_id=%s batch_id=%s attempt=%d "
                        "accepted=%d missing=%d retrying_missing_subset=true",
                        self.job_id,
                        batch.batch_id,
                        attempt,
                        len(exc.partial),
                        len(pending_entries),
                    )
                    if self.settings.strict_ids:
                        raise TranslateNonRetryableError(
                            "translate batch failed (strict_ids): missing="
                            f"{[e.index for e in pending_entries][:10]}"
                        ) from exc
                    if attempt < self.settings.retry:
                        self._sleep_before_retry(attempt)
                    continue
                if _is_non_retryable_http(exc) or not _is_retryable(exc):
                    # 4xx 等确定性错误立即失败，避免重复计费。
                    logger.error(
                        "translate batch failed: job_id=%s batch_id=%s attempt=%d "
                        "error_class=%s elapsed_ms=%d retryable=false",
                        self.job_id,
                        batch.batch_id,
                        attempt,
                        error_class,
                        elapsed_ms,
                    )
                    raise TranslateNonRetryableError(
                        f"translate batch failed (non-retryable): "
                        f"{llm_profile.get('section', 'llm')}:{llm_profile['model']}: {exc}"
                    ) from exc
                # SPEC G：每次进入重试路径即计一次 retry 指标。
                record_translate_retry(self._metrics)
                logger.warning(
                    "translate batch retry: job_id=%s batch_id=%s attempt=%d/%d "
                    "error_class=%s elapsed_ms=%d",
                    self.job_id,
                    batch.batch_id,
                    attempt,
                    self.settings.retry,
                    error_class,
                    elapsed_ms,
                )
                last_err = exc
                # 连接/超时/5xx/解析失败属于整批失败，不得降级为原文；
                # 但 id 不匹配（含"补缺失后又少返回"）仍属可降级的部分缺失。
                partial_only = isinstance(exc, TranslatePartialIdsError)
                if attempt < self.settings.retry:
                    self._sleep_before_retry(attempt)
                continue

            elapsed_ms = int((time.perf_counter() - attempt_started) * 1000)
            logger.info(
                "translate batch succeeded: job_id=%s batch_id=%s attempt=%d "
                "elapsed_ms=%d entries=%d",
                self.job_id,
                batch.batch_id,
                attempt,
                elapsed_ms,
                len(mapped),
            )
            accepted.update(mapped)
            pending_entries = [
                e for e in pending_entries if e.index not in accepted
            ]

            if pending_entries:
                # 部分 id 缺失：先接受已返回项，仅对缺失 id 用更小批次重试。
                logger.warning(
                    "translate_partial: job_id=%s batch_id=%s attempt=%d missing=%d "
                    "retrying_missing_subset=true",
                    self.job_id,
                    batch.batch_id,
                    attempt,
                    len(pending_entries),
                )
                last_err = TranslatePartialIdsError(
                    missing=[e.index for e in pending_entries], unexpected=[]
                )
                partial_only = True
                if self.settings.strict_ids:
                    raise TranslateNonRetryableError(
                        "translate batch failed (strict_ids): missing="
                        f"{[e.index for e in pending_entries][:10]}"
                    )
                continue

        if pending_entries and self.settings.strict_ids:
            raise TranslateNonRetryableError(
                f"translate batch failed (strict_ids): {last_err}"
            )

        if pending_entries and not partial_only:
            # 整批失败（连接/超时/5xx/解析）耗尽重试：交给任务重试，不降级。
            # SPEC G：失败批次同样计数，便于用 finished/claimed 差值发现卡死。
            record_translate_batch(self._metrics, "failed")
            raise RuntimeError(
                f"translate batch failed: {batch.batch_id}: {last_err}"
            )

        if pending_entries:
            # 仅“部分 id 缺失”耗尽了重试：默认保留日文原文写入 zh.srt，绝不丢字幕行。
            missing_ids = [e.index for e in pending_entries]
            record_translate_batch(self._metrics, "partial")
            logger.warning(
                "translate_partial: job_id=%s batch_id=%s missing=%d ids=%s "
                "fallback=keep_source_text",
                self.job_id,
                batch.batch_id,
                len(missing_ids),
                missing_ids[:20],
            )
            for entry in pending_entries:
                accepted[entry.index] = entry.text
        else:
            record_translate_batch(self._metrics, "succeeded")

        self._commit_batch(batch, accepted)

    def _sleep_before_retry(self, attempt: int) -> None:
        """重试退避：以 request_interval 为基准并在小步长中检查取消信号。"""

        backoff = min(max(self.settings.request_interval, 0.0) * 2.0, 5.0)
        if backoff <= 0:
            return
        deadline = time.perf_counter() + backoff
        while time.perf_counter() < deadline:
            self._check_cancel()
            time.sleep(min(0.05, max(deadline - time.perf_counter(), 0.0)))

    def _invoke_api(
        self,
        *,
        llm_profile: dict[str, str],
        batch_entries: list[SrtEntry],
        idempotency_key: str,
        attempt: int,
        batch_id: str,
    ) -> dict[int, str]:
        """在计数保护下发起一次上游请求；返回 id -> 译文映射。"""

        post = self._post_func if self._post_func is not None else self._session.post
        remaining = max(self.deadline_at - time.perf_counter(), 0.0)
        request_timeout = int(min(self.settings.timeout_sec, remaining) or 1)
        with self._lock:
            if self._aborted:
                raise TranslateCancelledError("translate_aborted")
            self._inflight_requests += 1
        try:
            self._throttle.wait_turn(self.cancel_check)
            return _call_translate_api(
                post,
                base_url=llm_profile["base_url"],
                api_key=llm_profile["api_key"],
                model=llm_profile["model"],
                timeout_sec=request_timeout,
                batch=batch_entries,
                idempotency_key=idempotency_key,
            )
        finally:
            with self._lock:
                self._inflight_requests -= 1

    # ------------------------------------------------------------------ 共享状态

    def _record_inflight(
        self,
        batch: TranslateBatch,
        pending_entries: list[SrtEntry],
        attempt: int,
    ) -> None:
        """请求发出前把批次标记为 inflight 并原子落检查点。"""

        with self._lock:
            if self._aborted:
                return
            previous = self._batch_states.get(batch.batch_id) or {}
            self._batch_states[batch.batch_id] = {
                "done": False,
                "inflight": True,
                "attempt": attempt,
                "ids": [e.index for e in pending_entries],
                "translations": previous.get("translations", {}),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            self._write_checkpoint_locked()

    def _commit_batch(self, batch: TranslateBatch, accepted: dict[int, str]) -> None:
        """提交批次结果：更新内存映射并原子落检查点。

        C.1 失败收尾契约：某批失败后只是**不再派发新批次**，已经成功的在途批次
        仍必须写入检查点，否则重试会重复请求上游并重复计费。
        真正的取消（失租/停机，`_cancelled`）才禁止写盘。
        """

        with self._lock:
            if self._cancelled:
                raise TranslateCancelledError("translate_cancelled")
            self._translations.update(accepted)
            self._batch_states[batch.batch_id] = {
                "done": True,
                "inflight": False,
                "ids": [e.index for e in batch.entries],
                "translations": {str(k): v for k, v in sorted(accepted.items())},
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            self._completed_batches += 1
            self._write_checkpoint_locked()
            percent = (
                self._completed_batches / max(len(self.batches), 1) * 100.0
            )
        self._publish_progress(percent=percent, message="translate_running")

    def _write_checkpoint_locked(self) -> None:
        """在持有锁时写检查点（调用方须已持锁）。"""

        if not self.settings.checkpoint_enabled:
            return
        payload = _build_checkpoint_payload(
            source_sha256=self.source_sha256,
            settings=self.settings,
            batch_states=self._batch_states,
        )
        try:
            _save_checkpoint(
                self.progress_artifact, payload, publish=self._checkpoint_publish
            )
        except StaleWriteRejected:
            # fenced 写入被拒 = 本 worker 已失租：立即中止，不得继续产出。
            logger.warning(
                "translate checkpoint rejected (stale worker): path=%s task_id=%s",
                self.progress_artifact,
                self.task_id,
            )
            # fenced 拒绝等价于失租：视为取消，禁止任何后续写盘。
            self._cancelled = True
            self._aborted = True
            raise TranslateCancelledError("translate_fenced_stale") from None
        except OSError as exc:
            # 检查点写失败不应中断翻译，但要告警：该批会重复请求上游。
            logger.warning(
                "translate checkpoint write failed: path=%s error=%s",
                self.progress_artifact,
                exc,
            )

    def _publish_progress(
        self, *, percent: float | None = None, message: str = "translate_running"
    ) -> None:
        """发布进度事件；中止后不再发布。"""

        with self._lock:
            if self._aborted:
                return
            if percent is None:
                percent = (
                    self._completed_batches / max(len(self.batches), 1) * 100.0
                )
        _emit_progress(
            self.progress_queue,
            stage="translate",
            percent=percent,
            message=message,
            task_id=self.task_id,
            worker_id=self.worker_id,
        )


def _resolve_copy_back_dir(
    configured: str,
    *,
    config_path: Path,
    input_video_path: Path,
) -> Path:
    """解析 copy_back 目标目录：默认回写到输入视频所在目录。"""

    lowered = configured.strip().lower()
    if lowered in {"", "__video_dir__", "video_dir", "input_video_dir"}:
        return input_video_path.parent

    target = Path(configured).expanduser()
    if target.is_absolute():
        return target
    return (config_path.parent / target).resolve()


def _copy_back_subtitles(
    input_ja_srt: Path, output_zh_srt: Path, target_dir: Path
) -> None:
    """把翻译阶段最终 ja/zh 字幕复制到回写目录。"""

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "copy_back skipped: cannot prepare target directory: dir=%s error=%s",
            target_dir,
            exc,
        )
        return

    for source in (input_ja_srt, output_zh_srt):
        target = target_dir / source.name
        try:
            # 目标若为符号链接则拒绝写入，避免把内容写到白名单之外的路径。
            if target.is_symlink():
                logger.warning(
                    "copy_back skipped: target is a symlink: target=%s", target
                )
                continue
            if source.resolve() == target.resolve():
                continue
            # 原子回写：先读源内容再原子替换目标，中断不会留下半截字幕。
            atomic_write_text(target, source.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "copy_back failed: source=%s target=%s error=%s",
                source,
                target,
                exc,
            )
