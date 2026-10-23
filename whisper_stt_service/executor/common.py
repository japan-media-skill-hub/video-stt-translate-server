"""执行器共享数据结构与通用工具。"""

from __future__ import annotations

import configparser
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import Full, Queue
from typing import Any

from whisper_stt_service.executor.atomic import atomic_write_text


logger = logging.getLogger(__name__)

# 缺失 ASR 输入音频产物时的错误码（E 契约：不得静默回退原视频）。
MISSING_AUDIO_ARTIFACT_CODE = "missing_audio_artifact"


class ExecutorCancelledError(RuntimeError):
    """执行器被上游取消（失租/停机）：必须尽快中止，且不得发布结果或回写产物。

    各执行器统一使用本类型，runtime 可依据同一类型放弃结果写回；
    `cancel_check()` 回调返回 True 时执行器即抛出该异常。
    """

    code = "executor_cancelled"


class MissingAudioArtifactError(RuntimeError):
    """ASR 输入 wav 缺失或为 0 字节（错误码 `missing_audio_artifact`）。"""

    code = MISSING_AUDIO_ARTIFACT_CODE

    def __init__(self, path: str | Path) -> None:
        """记录缺失的音频产物路径，便于排障。"""

        self.path = str(path)
        super().__init__(f"{MISSING_AUDIO_ARTIFACT_CODE}: {self.path}")


@dataclass
class SrtEntry:
    """SRT 条目结构。"""

    index: int
    timestamp: str
    text: str


def _timestamp_to_seconds(ts: str) -> float:
    """将 `HH:MM:SS,mmm` 转为秒；格式不合法时抛 ValueError。"""

    hhmmss, millis = ts.split(",", 1)
    h, m, s = hhmmss.split(":", 2)
    return int(h) * 3600 + int(m) * 60 + int(s) + int(millis) / 1000.0


def _entry_start_seconds(timestamp: str) -> float:
    """解析 SRT 时间轴起点秒数，失败时退化为 0。"""

    try:
        start_text = timestamp.split("-->", 1)[0].strip()
        return _timestamp_to_seconds(start_text)
    except Exception:
        return 0.0


def _emit_progress(
    progress_queue: Queue[dict[str, Any]] | None,
    *,
    stage: str,
    percent: float,
    message: str,
    task_id: str | None,
    worker_id: str | None,
) -> None:
    """把进度写入内存队列；队列满时丢弃一个旧事件后重试。"""

    if progress_queue is None:
        return
    event = {
        "task_id": task_id or "",
        "stage": stage,
        "percent": max(0.0, min(percent, 100.0)),
        "message": message,
        "worker_id": worker_id or "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        progress_queue.put_nowait(event)
    except Full:
        try:
            progress_queue.get_nowait()
        except Exception:
            pass
        try:
            progress_queue.put_nowait(event)
        except Exception:
            pass


def _probe_duration(path: Path) -> float | None:
    """用 ffprobe 获取媒体时长，失败时返回 None。"""

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def preclean_output(path: Path) -> None:
    """执行前清理：组合多种策略，尽可能释放目标输出路径。"""

    failures: list[str] = []

    def _record_failure(step: str, exc: Exception) -> None:
        failures.append(f"{step}: {exc.__class__.__name__}: {exc}")

    def _run_silent_rm(target: Path, recursive: bool = False) -> int:
        args = ["rm", "-f"]
        if recursive:
            args.append("-r")
        args.extend(["--", str(target)])
        proc = subprocess.run(
            args,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc.returncode

    def _path_lexists(target: Path) -> bool:
        try:
            return os.path.lexists(target)
        except Exception:
            return target.exists()

    # SMB 挂载下可能出现 “ls 可见但路径操作报不存在” 的残留目录项。
    # 先执行 rm，再走 Python 与系统调用多轮清理，双保险降低 ffmpeg 输出冲突概率。
    for attempt in range(1, 6):
        # 第一层：直接走 rm，兼容网络文件系统上“可见但 stat 异常”的目录项。
        try:
            rc = _run_silent_rm(path, recursive=True)
            if rc != 0:
                failures.append(f"rm: returncode={rc}")
        except Exception as exc:
            _record_failure("rm", exc)

        # 第二层：尝试修正权限后删除，兼容只读位阻塞。
        try:
            path.chmod(0o666)
        except Exception as exc:
            _record_failure("chmod", exc)

        # 第三层：Python 侧按类型删除，覆盖文件/目录/符号链接。
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        except Exception as exc:
            _record_failure("python_unlink", exc)

        # 第四层：直接系统调用兜底。
        try:
            os.remove(path)
        except Exception as exc:
            _record_failure("os.remove", exc)
        try:
            os.rmdir(path)
        except Exception as exc:
            _record_failure("os.rmdir", exc)

        # 第五层：路径仍被占用时，先改名让原目标路径可复用，再后台清理墓碑文件。
        if _path_lexists(path):
            tombstone = path.with_name(f".{path.name}.preclean.{time.time_ns()}")
            try:
                os.replace(path, tombstone)
                try:
                    rc = _run_silent_rm(tombstone, recursive=True)
                    if rc != 0:
                        failures.append(f"rm_tombstone: returncode={rc}")
                except Exception as exc:
                    _record_failure("rm_tombstone", exc)
                try:
                    if tombstone.is_dir() and not tombstone.is_symlink():
                        shutil.rmtree(tombstone, ignore_errors=True)
                    else:
                        tombstone.unlink(missing_ok=True)
                except Exception as exc:
                    _record_failure("python_unlink_tombstone", exc)
            except Exception as exc:
                _record_failure("os.replace_tombstone", exc)

        if not _path_lexists(path):
            return
        if attempt == 5:
            logger.error(
                "preclean_output failed, target remains: path=%s failures=%s",
                path,
                " | ".join(failures) if failures else "unknown",
            )
        time.sleep(0.05)

    # 清理失败不应阻断后续流程，让 ffmpeg 自行报告可复现错误。


DEFAULT_TRANSLATE_BATCH_SIZE = 200
DEFAULT_TRANSLATE_MAX_CHARS = 12000


@dataclass
class TranslateBatch:
    """翻译批次：携带稳定 batch_id（window_no/part_no）与成员条目。"""

    batch_id: str
    window_no: int
    part_no: int
    entries: list[SrtEntry] = field(default_factory=list)

    @property
    def source_chars(self) -> int:
        """批次内原文总字符数。"""

        return sum(len(entry.text) for entry in self.entries)

    @property
    def ids(self) -> list[int]:
        """批次内字幕 id 列表（保持原文顺序）。"""

        return [entry.index for entry in self.entries]


def build_translate_batches(
    entries: list[SrtEntry],
    *,
    window_minutes: int = 30,
    batch_size: int = DEFAULT_TRANSLATE_BATCH_SIZE,
    max_chars: int = DEFAULT_TRANSLATE_MAX_CHARS,
) -> list[TranslateBatch]:
    """按“时间窗 + 条目数 + 字符数”三重上限切分翻译批次。

    先按 `window_minutes` 时间窗分组，再在窗口内按 `batch_size` 与 `max_chars`
    二次切分；三个上限同时生效、取先到者。窗口内顺序保持原文顺序，
    batch_id 固定为 `window_no/part_no`（从 0 与 1 起编号，与完成顺序无关），
    因此同一输入总能得到同一组 batch_id，检查点可安全复用。
    单条字幕超过 `max_chars` 时单独成批，绝不截断原文。
    """

    window_sec = max(window_minutes, 1) * 60
    safe_batch_size = max(int(batch_size), 1)
    safe_max_chars = max(int(max_chars), 1)

    # 先按时间窗分组，保持每个窗口内条目原有顺序。
    grouped: dict[int, list[SrtEntry]] = {}
    for entry in entries:
        bucket = int(_entry_start_seconds(entry.timestamp) // window_sec)
        grouped.setdefault(bucket, []).append(entry)

    batches: list[TranslateBatch] = []
    for window_index, window_no in enumerate(sorted(grouped), start=0):
        part_no = 0
        current: list[SrtEntry] = []
        current_chars = 0

        def _flush(part_entries: list[SrtEntry], part_index: int) -> None:
            """把当前累积条目落成一个带稳定 batch_id 的批次。"""

            batches.append(
                TranslateBatch(
                    batch_id=f"{window_index}/{part_index}",
                    window_no=window_index,
                    part_no=part_index,
                    entries=part_entries,
                )
            )

        for entry in grouped[window_no]:
            entry_chars = len(entry.text)
            # 单条超长字幕单独成批，不得截断原文。
            if entry_chars > safe_max_chars:
                if current:
                    _flush(current, part_no)
                    part_no += 1
                    current, current_chars = [], 0
                _flush([entry], part_no)
                part_no += 1
                continue

            exceeds_size = len(current) >= safe_batch_size
            exceeds_chars = current and current_chars + entry_chars > safe_max_chars
            if current and (exceeds_size or exceeds_chars):
                _flush(current, part_no)
                part_no += 1
                current, current_chars = [], 0

            current.append(entry)
            current_chars += entry_chars

        if current:
            _flush(current, part_no)

    return batches


def _split_entries_by_time_window(
    entries: list[SrtEntry], window_minutes: int = 30
) -> list[list[SrtEntry]]:
    """按字幕起始时间切分窗口，默认每 30 分钟一个翻译批次。

    兼容旧调用方：仅按时间窗切分，不做条目/字符二次切分。
    """

    window_sec = max(window_minutes, 1) * 60
    grouped: dict[int, list[SrtEntry]] = {}
    for entry in entries:
        bucket = int(_entry_start_seconds(entry.timestamp) // window_sec)
        grouped.setdefault(bucket, []).append(entry)
    return [grouped[k] for k in sorted(grouped)]


def _sha256_text(text: str) -> str:
    """计算文本的 SHA-256 十六进制摘要（检查点与幂等键复用）。"""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    """计算文件内容的 SHA-256 摘要；文件不存在时抛 FileNotFoundError。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_translate_messages(batch: list[SrtEntry]) -> list[dict[str, str]]:
    """构造翻译消息，强调剧情上下文推理和全段一致性。"""

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


def _parse_srt(content: str) -> list[SrtEntry]:
    """解析 SRT 文本为结构化条目。"""

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
        entries.append(
            SrtEntry(
                index=idx, timestamp=lines[1].strip(), text="\n".join(lines[2:]).strip()
            )
        )
    return entries


def _dump_srt(
    entries: list[SrtEntry], translations: dict[int, str], out_path: Path
) -> None:
    """按原时间轴写回翻译后的 SRT 文件。

    写入采用原子替换，失败不会留下半成品；缺失译文的条目保留日文原文，
    绝不丢弃字幕行。
    """

    parts: list[str] = []
    for entry in entries:
        text_zh = translations.get(entry.index, entry.text)
        parts.append(f"{entry.index}\n{entry.timestamp}\n{text_zh}\n\n")
    # 先拼装完整文本再原子落盘：进程中断不会留下截断的 zh.srt。
    atomic_write_text(out_path, "".join(parts))


def _extract_json_object(text: str) -> Any:
    """从模型返回文本中提取 JSON 对象/数组。"""

    payload = text.strip()
    if payload.startswith("```") and payload.endswith("```"):
        lines = payload.splitlines()
        if len(lines) >= 3:
            payload = "\n".join(lines[1:-1]).strip()
    if payload.startswith("json"):
        payload = payload[4:].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass
    arr_start = payload.find("[")
    arr_end = payload.rfind("]")
    if arr_start != -1 and arr_end > arr_start:
        return json.loads(payload[arr_start : arr_end + 1])
    obj_start = payload.find("{")
    obj_end = payload.rfind("}")
    if obj_start != -1 and obj_end > obj_start:
        return json.loads(payload[obj_start : obj_end + 1])
    raise RuntimeError("unable to parse JSON response")


def _load_llm_config(config_path: Path) -> configparser.ConfigParser:
    """加载 config.ini（若不存在则返回空配置）。"""

    cfg = configparser.ConfigParser()
    if config_path.is_file():
        cfg.read(config_path, encoding="utf-8")
    return cfg


@dataclass
class TranslateSettings:
    """translate 阶段生效配置（已按 job_config > [translation] > 默认值合并）。"""

    parallel: int = 2
    batch_size: int = DEFAULT_TRANSLATE_BATCH_SIZE
    max_chars: int = DEFAULT_TRANSLATE_MAX_CHARS
    chunk_minutes: int = 30
    timeout_sec: int = 240
    retry: int = 4
    request_interval: float = 1.0
    disable_proxy: bool = True
    copy_back: str = "__video_dir__"
    checkpoint_enabled: bool = True
    strict_ids: bool = False
    progress_ttl_sec: int = 604800


def _coerce_bool(value: Any, default: bool) -> bool:
    """把配置值宽松地转换为布尔值，无法识别时回退默认值。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "y", "t"}:
            return True
        if lowered in {"0", "false", "no", "off", "n", "f"}:
            return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    """把配置值宽松地转换为整数，无法识别时回退默认值。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    """把配置值宽松地转换为浮点数，无法识别时回退默认值。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_translation_section(
    config: configparser.ConfigParser, key: str, *, kind: str
) -> Any:
    """从 [translation] 段读取单个键；段/键缺失或非法时返回 None。"""

    if not config.has_section("translation"):
        return None
    readers: dict[str, Any] = {
        "int": config.getint,
        "float": config.getfloat,
        "bool": config.getboolean,
        "str": config.get,
    }
    try:
        value = readers[kind]("translation", key, fallback=None)
    except (ValueError, configparser.Error) as exc:
        # 配置值非法时告警并回退默认值，而不是让整个翻译阶段失败。
        logger.warning(
            "translation config ignored: key=%s kind=%s error=%s", key, kind, exc
        )
        return None
    if isinstance(value, str):
        return value.strip()
    return value


def resolve_translate_settings(
    config: configparser.ConfigParser,
    job_config: dict[str, Any] | None = None,
    *,
    chunk_minutes: int | None = None,
    retry: int | None = None,
    timeout_sec: int | None = None,
    copy_back: str | None = None,
    progress_ttl_sec: int | None = None,
) -> TranslateSettings:
    """解析 translate 阶段全部生效配置。

    优先级：显式关键字/job_config（对应 task 的 job_config.translate）
    > `[translation]` 配置段 > 代码默认值。旧调用方的
    `chunk_minutes`/`retry`/`timeout_sec`/`copy_back` 关键字仍生效，且作为
    job_config 的同级覆盖，保证向后兼容。新增键（`max_chars`、
    `checkpoint_enabled`、`strict_ids`、`progress_ttl_sec` 等）缺失时取其默认值，
    因此旧配置可直接加载。
    """

    overrides: dict[str, Any] = dict(job_config or {})
    if chunk_minutes is not None:
        overrides["chunk_minutes"] = chunk_minutes
    if retry is not None:
        overrides["retry"] = retry
    if timeout_sec is not None:
        overrides["timeout_sec"] = timeout_sec
    if copy_back is not None and str(copy_back).strip():
        overrides["copy_back"] = copy_back
    if progress_ttl_sec is not None:
        overrides["progress_ttl_sec"] = progress_ttl_sec

    defaults = TranslateSettings()
    resolved: dict[str, Any] = {}
    for key in (
        "parallel",
        "batch_size",
        "max_chars",
        "chunk_minutes",
        "timeout_sec",
        "retry",
        "request_interval",
        "disable_proxy",
        "copy_back",
        "checkpoint_enabled",
        "strict_ids",
        "progress_ttl_sec",
    ):
        current = getattr(defaults, key)
        if isinstance(current, bool):
            kind = "bool"
        elif isinstance(current, int):
            kind = "int"
        elif isinstance(current, float):
            kind = "float"
        else:
            kind = "str"

        # 逐级回退：job_config/关键字覆盖 > [translation] > 代码默认值。
        value: Any = overrides.get(key)
        if value is None:
            value = _read_translation_section(config, key, kind=kind)
        if value is None:
            value = current

        if kind == "bool":
            resolved[key] = _coerce_bool(value, current)
        elif kind == "int":
            resolved[key] = _coerce_int(value, current)
        elif kind == "float":
            resolved[key] = _coerce_float(value, current)
        else:
            resolved[key] = str(value).strip() or str(current)

    # 数值下限收敛：并发至少 1、条目/字符上限至少 1，节流间隔不得为负。
    resolved["parallel"] = max(int(resolved["parallel"]), 1)
    resolved["batch_size"] = max(int(resolved["batch_size"]), 1)
    resolved["max_chars"] = max(int(resolved["max_chars"]), 1)
    resolved["chunk_minutes"] = max(int(resolved["chunk_minutes"]), 1)
    resolved["timeout_sec"] = max(int(resolved["timeout_sec"]), 1)
    resolved["retry"] = max(int(resolved["retry"]), 1)
    resolved["request_interval"] = max(float(resolved["request_interval"]), 0.0)
    resolved["progress_ttl_sec"] = max(int(resolved["progress_ttl_sec"]), 0)
    return TranslateSettings(**resolved)
