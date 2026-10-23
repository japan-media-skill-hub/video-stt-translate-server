#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import datetime as dt
import json
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

# 脚本可从任意 cwd 启动（worktree / 主仓库 / CI）：显式把仓库根注入 sys.path，
# 保证能导入统一的 service 执行引擎 whisper_stt_service.executor.translate。
_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))


@dataclass
class SrtEntry:
    index: int
    timestamp: str
    text: str


@dataclass
class BatchResult:
    mapping: dict[int, str]
    attempts_used: int


def parse_srt(content: str) -> list[SrtEntry]:
    blocks = [
        b.strip() for b in content.replace("\r\n", "\n").split("\n\n") if b.strip()
    ]
    entries: list[SrtEntry] = []
    for block in blocks:
        lines = block.split("\n")
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        ts = lines[1].strip()
        text = "\n".join(lines[2:]).strip()
        entries.append(SrtEntry(index=idx, timestamp=ts, text=text))
    return entries


def dump_srt(
    entries: list[SrtEntry], translations: dict[int, str], out_path: Path
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            text_zh = translations.get(entry.index, entry.text)
            f.write(f"{entry.index}\n")
            f.write(f"{entry.timestamp}\n")
            f.write(f"{text_zh}\n\n")


def _timestamp_to_seconds(ts: str) -> float:
    hhmmss, millis = ts.split(",", 1)
    h, m, s = hhmmss.split(":", 2)
    return int(h) * 3600 + int(m) * 60 + int(s) + int(millis) / 1000.0


def _entry_start_seconds(timestamp: str) -> float:
    try:
        start_text = timestamp.split("-->", 1)[0].strip()
        return _timestamp_to_seconds(start_text)
    except Exception:
        return 0.0


def _split_entries_by_time_window(
    entries: list[SrtEntry], window_minutes: int = 30
) -> list[list[SrtEntry]]:
    window_sec = max(window_minutes, 1) * 60
    grouped: dict[int, list[SrtEntry]] = {}
    for entry in entries:
        bucket = int(_entry_start_seconds(entry.timestamp) // window_sec)
        grouped.setdefault(bucket, []).append(entry)
    return [grouped[k] for k in sorted(grouped)]


def _build_translate_messages(batch: list[SrtEntry]) -> list[dict[str, str]]:
    payload_items = [{"id": e.index, "text": e.text} for e in batch]
    system_prompt = (
        "You are a professional subtitle translator. Translate Japanese subtitles into natural, concise "
        "Simplified Chinese. Use context and plot continuity to infer omitted subjects, keep naming and tone "
        "consistent across this whole chunk, and avoid literal word-by-word translation. Intelligently clean "
        "disfluencies and meaningless repetitions (such as long runs of interjections or stutters like "
        "'ああああ', 'えええ', 'ううう'), compressing them into natural Chinese or removing them when they "
        "carry no semantic meaning, while preserving the speaker's intent and emotion. Think carefully, but "
        "output ONLY JSON."
    )
    user_prompt = (
        "Translate the following JSON array from Japanese to Simplified Chinese subtitles. "
        "Return ONLY a JSON array with objects in the form: "
        '{"id": <same id>, "text_zh": "..."}. Keep the same ids and same item count. '
        "For repetitive filler words, perform intelligent denoising and output fluent subtitles.\n\n"
        f"{json.dumps(payload_items, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def extract_json_object(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if (
            len(lines) >= 3
            and lines[0].startswith("```")
            and lines[-1].startswith("```")
        ):
            text = "\n".join(lines[1:-1]).strip()
    if "```" in text:
        chunks = text.split("```")
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("[") or chunk.startswith("{"):
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    pass
    if text.startswith("json"):
        text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        return json.loads(text[arr_start : arr_end + 1])

    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        return json.loads(text[obj_start : obj_end + 1])
    raise RuntimeError("unable to parse JSON response")


def _format_hms(seconds: float) -> str:
    s = int(max(seconds, 0.0))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _render_bar(progress: float, width: int = 28) -> str:
    p = max(0.0, min(progress, 1.0))
    done = int(width * p)
    return "[" + "#" * done + "-" * (width - done) + "]"


def _progress_line(
    prefix: str, progress: float, elapsed: float, eta: float | None
) -> str:
    if eta is None:
        eta_text = "--:--:--"
        finish_text = "--:--:--"
    else:
        eta_text = _format_hms(eta)
        finish_text = (dt.datetime.now() + dt.timedelta(seconds=eta)).strftime(
            "%H:%M:%S"
        )
    bar = _render_bar(progress)
    return (
        f"{prefix} {bar} {progress * 100.0:6.2f}% elapsed={_format_hms(elapsed)} "
        f"eta={eta_text} finish={finish_text}"
    )


def _call_api(
    post_func: Any,
    base_url: str,
    api_key: str,
    model: str,
    batch: list[SrtEntry],
    timeout: int,
) -> dict[int, str]:
    messages = _build_translate_messages(batch)

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "temperature": 0.2,
        "messages": messages,
    }
    resp = post_func(url, headers=headers, json=data, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    parsed = extract_json_object(content)
    if not isinstance(parsed, list):
        raise ValueError("model output is not a JSON array")

    mapping: dict[int, str] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        text_zh = item.get("text_zh")
        if isinstance(item_id, int) and isinstance(text_zh, str) and text_zh.strip():
            mapping[item_id] = text_zh.strip()

    expected = {e.index for e in batch}
    if set(mapping.keys()) != expected:
        missing = sorted(expected - set(mapping.keys()))
        raise ValueError(f"batch translation ids mismatch, missing={missing[:10]}")
    return mapping


def translate_batch_with_retry(
    batch_no: int,
    total_batches: int,
    post_func: Any,
    base_url: str,
    api_key: str,
    model: str,
    batch: list[SrtEntry],
    timeout: int,
    retry: int,
) -> BatchResult:
    last_err: Exception | None = None
    for attempt in range(1, retry + 1):
        try:
            mapped = _call_api(post_func, base_url, api_key, model, batch, timeout)
            return BatchResult(mapping=mapped, attempts_used=attempt)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(
                f"RETRY batch={batch_no}/{total_batches} attempt={attempt}/{retry} "
                f"pending={len(batch)} error_type={type(exc).__name__}",
                flush=True,
            )
            time.sleep(min(2.0 * attempt, 8.0))
    raise RuntimeError(f"batch={batch_no}/{total_batches} failed: {last_err}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-translate Japanese SRT to Chinese"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config.ini path (defaults to repo_root/config.ini)",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input Japanese .srt")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output Chinese .srt"
    )
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chunk-minutes", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--retry", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=None)
    parser.add_argument("--parallel", type=int, default=None)
    parser.add_argument("--request-interval", type=float, default=None)
    parser.add_argument(
        "--disable-proxy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Disable proxy usage for API requests",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="unified",
        choices=["unified", "legacy"],
        help=(
            "Translation engine: 'unified' (default) reuses the service executor "
            "with full hardening; 'legacy' keeps the old standalone loop for debugging"
        ),
    )
    return parser.parse_args()


def _load_config(config_path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    if config_path.is_file():
        config.read(config_path, encoding="utf-8")
    return config


def _cfg_str(
    cli_value: str | None,
    config: configparser.ConfigParser,
    section: str,
    key: str,
    fallback: str,
) -> str:
    if cli_value is not None:
        return cli_value
    return config.get(section, key, fallback=fallback)


def _cfg_int(
    cli_value: int | None,
    config: configparser.ConfigParser,
    section: str,
    key: str,
    fallback: int,
) -> int:
    if cli_value is not None:
        return cli_value
    return config.getint(section, key, fallback=fallback)


def _cfg_float(
    cli_value: float | None,
    config: configparser.ConfigParser,
    section: str,
    key: str,
    fallback: float,
) -> float:
    if cli_value is not None:
        return cli_value
    return config.getfloat(section, key, fallback=fallback)


def _cfg_bool(
    cli_value: bool | None,
    config: configparser.ConfigParser,
    section: str,
    key: str,
    fallback: bool,
) -> bool:
    if cli_value is not None:
        return cli_value
    return config.getboolean(section, key, fallback=fallback)


def _invalidate_legacy_checkpoint(progress_path: Path) -> bool:
    """安全失效旧的 CLI 检查点（结构为 {id: text}，无 source 校验）。

    旧结构无法证明产物来自当前 ja.srt，继续复用会导致"换了视频却沿用旧译文"。
    因此只要检测到旧结构就删除文件并返回 True，强制本次全量重算；
    已经是统一引擎结构（含 version/source_sha256/batches）的文件保持不动，
    交给引擎自身的 source_sha256 + 切分配置校验决定是否复用。
    """

    if not progress_path.is_file():
        return False
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # 损坏的文件同样无法安全复用。
        payload = None
    is_engine_format = isinstance(payload, dict) and {
        "version",
        "source_sha256",
        "batches",
    } <= set(payload.keys())
    if is_engine_format:
        return False
    try:
        progress_path.unlink(missing_ok=True)
    except OSError as exc:
        # 删不掉也不能复用旧结构：直接报错，避免静默沿用旧译文。
        message = "cannot invalidate legacy checkpoint " + str(progress_path)
        raise RuntimeError(message) from exc
    print(
        "CHECKPOINT legacy_format_invalidated path=" + str(progress_path),
        flush=True,
    )
    return True


def _run_unified_engine(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    config_path: Path,
    config: configparser.ConfigParser,
) -> int:
    """把 CLI 编排委托给统一执行引擎（与服务端同一份实现）。

    这是 CLI 的默认路径：并发上限、三重切分（batch_size/max_chars）、
    真检查点（source_sha256 校验）、原子写、部分 id 保留原文、
    4xx 不重试、Idempotency-Key 全部沿服务端 `run_translate`，
    不再维护第二份翻译逻辑。
    """

    from whisper_stt_service.executor.common import _sha256_file
    from whisper_stt_service.executor.translate import run_translate

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(output_path.suffix + ".progress.json")
    _invalidate_legacy_checkpoint(progress_path)

    chunk_minutes = _cfg_int(
        args.chunk_minutes, config, "translation", "chunk_minutes", 30
    )
    timeout = _cfg_int(args.timeout, config, "translation", "timeout", 180)
    retry = _cfg_int(args.retry, config, "translation", "retry", 4)
    # D2：脚本模式默认并发同样为 2（仍可通过 --parallel 或 [translation] parallel 覆盖）。
    parallel = _cfg_int(args.parallel, config, "translation", "parallel", 2)
    batch_size = _cfg_int(args.batch_size, config, "translation", "batch_size", 200)
    request_interval = _cfg_float(
        args.request_interval, config, "translation", "request_interval", 1.0
    )
    disable_proxy = _cfg_bool(
        args.disable_proxy, config, "translation", "disable_proxy", True
    )

    # --base-url/--api-key/--model 仅内存传递，绝不落任何临时配置文件。
    llm_overrides: dict[str, str] = {}
    if args.base_url is not None:
        llm_overrides["base_url"] = str(args.base_url)
    if args.api_key is not None:
        llm_overrides["api_key"] = str(args.api_key)
    if args.model is not None:
        llm_overrides["model"] = str(args.model)

    source_digest = _sha256_file(args.input)
    cli_job_id = "cli-" + args.input.stem + "-" + source_digest[:12]

    print(
        "SUMMARY engine=unified input=" + str(args.input)
        + " output=" + str(output_path)
        + " chunk_minutes=" + str(chunk_minutes)
        + " batch_size=" + str(batch_size)
        + " max_parallel=" + str(parallel)
        + " retry=" + str(retry),
        flush=True,
    )

    run_translate(
        args.input,
        output_path,
        config_path=config_path,
        timeout_sec=timeout,
        # CLI 没有视频上下文：由 defer_copy_back=True 显式关闭回写，避免污染任意目录。
        input_video_path=output_path.parent,
        defer_copy_back=True,
        checkpoint_path=progress_path,
        job_config={
            "parallel": parallel,
            "batch_size": batch_size,
            "chunk_minutes": chunk_minutes,
            "timeout": timeout,
            "retry": retry,
            "request_interval": request_interval,
            "disable_proxy": disable_proxy,
        },
        llm_overrides=llm_overrides or None,
        job_id=cli_job_id,
        task_id=cli_job_id,
    )

    print("SUMMARY output=" + str(output_path), flush=True)
    print("SUMMARY progress=" + str(progress_path), flush=True)
    return 0


def _run_legacy_engine(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    config_path: Path,
    config: configparser.ConfigParser,
) -> int:
    """旧版 CLI 翻译实现（逐窗切分 + 自建线程池 + 自建重试）。

    保留原因：排障时可用 `--engine legacy` 对照旧行为，且其辅助函数
    （`_call_api` / `translate_batch_with_retry` / `_split_entries_by_time_window`）
    仍被既有测试与脚本引用。**默认不再使用**：CLI 默认走统一引擎
    `_run_unified_engine`，以获得 4xx 不重试、部分 id 降级、原子写、
    带 source_sha256 校验的检查点等加固行为。
    """
    base_url = _cfg_str(
        args.base_url,
        config,
        "llm",
        "base_url",
        "https://www.right.codes/codex/v1",
    )
    api_key = _cfg_str(args.api_key, config, "llm", "api_key", "").strip()
    if not api_key:
        raise RuntimeError(
            "missing api key: provide --api-key or set [llm] api_key in config.ini"
        )
    model = _cfg_str(args.model, config, "llm", "model", "gpt-5.4-mini")
    chunk_minutes = _cfg_int(
        args.chunk_minutes,
        config,
        "translation",
        "chunk_minutes",
        30,
    )
    timeout = _cfg_int(args.timeout, config, "translation", "timeout", 180)
    retry = _cfg_int(args.retry, config, "translation", "retry", 4)
    sleep = _cfg_float(args.sleep, config, "translation", "sleep", 0.2)
    # D2：脚本模式默认并发同样改为 2（仍可通过 --parallel 或 [translation] parallel 覆盖）。
    parallel = _cfg_int(args.parallel, config, "translation", "parallel", 2)
    request_interval = _cfg_float(
        args.request_interval,
        config,
        "translation",
        "request_interval",
        1.0,
    )
    disable_proxy = _cfg_bool(
        args.disable_proxy,
        config,
        "translation",
        "disable_proxy",
        True,
    )

    source = args.input.read_text(encoding="utf-8")
    entries = parse_srt(source)
    if not entries:
        raise RuntimeError(f"no valid srt entries found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.output.with_suffix(args.output.suffix + ".progress.json")
    translations: dict[int, str] = {}
    if progress_path.is_file():
        existing = json.loads(progress_path.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            translations = {int(k): str(v) for k, v in existing.items()}

    total_entries = len(entries)
    all_batches = _split_entries_by_time_window(entries, window_minutes=chunk_minutes)
    pending: list[tuple[int, list[SrtEntry]]] = []
    for batch_no, batch in enumerate(all_batches, start=1):
        need = [e for e in batch if e.index not in translations]
        if need:
            pending.append((batch_no, need))

    effective_workers = max(1, min(parallel, len(pending))) if pending else 1

    print(
        f"SUMMARY total_entries={total_entries} already_translated={len(translations)} "
        f"total_batches={len(all_batches)} pending_batches={len(pending)} "
        f"chunk_minutes={chunk_minutes} workers={effective_workers} max_parallel={parallel} "
        f"request_interval={request_interval:.2f}s",
        flush=True,
    )

    if disable_proxy:
        session = requests.Session()
        session.trust_env = False
        post_func = session.post
    else:
        post_func = requests.post

    started_at = time.perf_counter()
    completed_batches = 0
    retry_total = 0
    future_meta: dict[Future[BatchResult], tuple[int, list[SrtEntry]]] = {}

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        for i, (batch_no, batch) in enumerate(pending, start=1):
            print(
                f"DISPATCH batch={batch_no}/{len(all_batches)} pending={len(batch)} "
                f"dispatch={i}/{len(pending)}",
                flush=True,
            )
            fut = executor.submit(
                translate_batch_with_retry,
                batch_no,
                len(all_batches),
                post_func,
                base_url,
                api_key,
                model,
                batch,
                timeout,
                retry,
            )
            future_meta[fut] = (batch_no, batch)
            if i < len(pending):
                time.sleep(max(request_interval, 0.0))

        for fut in as_completed(future_meta):
            batch_no, batch = future_meta[fut]
            result = fut.result()
            translations.update(result.mapping)
            progress_path.write_text(
                json.dumps(
                    {str(k): v for k, v in translations.items()}, ensure_ascii=False
                ),
                encoding="utf-8",
            )
            completed_batches += 1
            retry_total += max(result.attempts_used - 1, 0)
            elapsed = time.perf_counter() - started_at
            progress = completed_batches / max(len(pending), 1)
            speed = completed_batches / max(elapsed, 1e-6)
            eta = (len(pending) - completed_batches) / max(speed, 1e-6)
            print(
                _progress_line("TRANSLATE", progress, elapsed, eta)
                + f" batch={batch_no}/{len(all_batches)} translated={len(batch)} "
                f"entries={len(translations)}/{total_entries} retries_total={retry_total}",
                flush=True,
            )
            time.sleep(sleep)

    dump_srt(entries, translations, args.output)
    total_elapsed = time.perf_counter() - started_at
    print(_progress_line("TRANSLATE", 1.0, total_elapsed, 0.0), flush=True)
    print(f"SUMMARY output={args.output}", flush=True)
    print(f"SUMMARY progress={progress_path}", flush=True)
    return 0


def main() -> int:
    """CLI 入口：默认走统一执行引擎，可用 `--engine legacy` 回退旧实现。

    默认（unified）与服务端同一份翻译实现，因此并发上限、三重切分、
    带 source_sha256 校验的检查点、原子写、部分 id 降级、4xx 不重试、
    Idempotency-Key 等加固行为对脚本模式同样生效。
    """

    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config is not None else repo_root / "config.ini"
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    config = _load_config(config_path)

    engine = str(getattr(args, "engine", "unified") or "unified").strip().lower()
    if engine == "legacy":
        print("SUMMARY engine=legacy (deprecated path)", flush=True)
        return _run_legacy_engine(
            args, repo_root=repo_root, config_path=config_path, config=config
        )
    return _run_unified_engine(
        args, repo_root=repo_root, config_path=config_path, config=config
    )


if __name__ == "__main__":
    raise SystemExit(main())
