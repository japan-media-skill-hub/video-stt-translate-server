"""执行器共享数据结构与通用工具。"""

from __future__ import annotations

import configparser
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import Full, Queue
from typing import Any, Iterable, Iterator

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


# 字幕单段最长时长（秒）：faster-whisper 的段级时间预测在长静音处会把 end
# 预测得极远（实测 A/D 影片出现 886.7s、489.9s），单段跨度过大在播放器里表现为
# "字幕长时间不切换"。这不是后处理造成的，只能在下游写字幕前兜底收紧。
DEFAULT_MAX_SEGMENT_SEC = 30.0

# 单段最短时长（秒）：end <= start 的零长度/倒挂段给一个可播放的最小时长。
MIN_SEGMENT_SEC = 0.3

# 时间戳相等判定容差：清洗前后差异小于该值视为"未调整"（1ms 是 SRT 精度）。
_TIMESTAMP_EPSILON = 1e-6


def coerce_max_segment_sec(
    value: Any, default: float = DEFAULT_MAX_SEGMENT_SEC
) -> float:
    """归一化「单段最长时长」配置：非数值/非有限/<=0 一律回退默认值。

    `effective_config` 里的值可能来自任务配置字符串（例如 "30" 或 "abc"），
    这里统一转成可直接比较的浮点，绝不因为配置写错而把字幕切坏或抛异常。
    """

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed) or parsed <= 0:
        return float(default)
    return parsed


def _clean_segment_bounds(
    *,
    start: float,
    end: float,
    prev_end: float,
    next_start: float | None,
    max_segment_sec: float,
) -> tuple[float, float]:
    """按清洗规则算出一段的 (start, end)；只动时间，不碰文本。

    规则顺序即语义（顺序不可交换）：
    1. `start` 不早于上一条 `end`，且不为负 → 乱序/重叠段被推后，保证单调；
    2. `end <= start`（零长度/倒挂）→ `start + MIN_SEGMENT_SEC`；
    3. 单段超过 `max_segment_sec` → 截到 `start + max_segment_sec`；
    4. lookahead 收紧：`end = min(end, 下一条 start)`，最后一条不收紧；
    5. 收紧后仍 `end <= start` → 再补最小 0.3s（下一条会被规则 1 推后，不会重叠）。
    """

    cleaned_start = max(float(start), 0.0)
    if cleaned_start < prev_end:
        cleaned_start = prev_end
    cleaned_end = float(end)
    if cleaned_end <= cleaned_start:
        cleaned_end = cleaned_start + MIN_SEGMENT_SEC
    if cleaned_end - cleaned_start > max_segment_sec:
        cleaned_end = cleaned_start + max_segment_sec
    if next_start is not None and cleaned_start < next_start < cleaned_end:
        # 下一句已经开口：本句不得越过它的起点（最后一条没有 next_start）。
        cleaned_end = next_start
    if cleaned_end <= cleaned_start:
        cleaned_end = cleaned_start + MIN_SEGMENT_SEC
    return cleaned_start, cleaned_end


def _log_timestamp_cleanup_summary(
    stats: dict[str, Any] | None,
    *,
    stage: str,
    task_id: str | None = None,
) -> None:
    """记录一条字幕时间戳清洗汇总，便于观测"到底有没有超长/错乱段"。

    `adjusted > 0`（确实改了时间）时用 WARNING：这是真实影片里出现过 886.7s 超长段
    的信号，必须能在日志里被看到；没改动时降到 INFO，避免每部正常影片都刷一条告警
    把真正的 WARNING 淹没。`stats` 为空（循环未跑完/提前异常）时直接返回。
    """

    if not stats:
        return
    level = logger.warning if int(stats.get("adjusted", 0)) > 0 else logger.info
    level(
        "%s timestamps sanitized: task_id=%s total=%d adjusted=%d "
        "max_original=%.1fs max_output=%.1fs max_segment_sec=%.1fs",
        stage,
        task_id,
        int(stats.get("total", 0)),
        int(stats.get("adjusted", 0)),
        float(stats.get("max_original_sec", 0.0)),
        float(stats.get("max_output_sec", 0.0)),
        float(stats.get("max_segment_sec", DEFAULT_MAX_SEGMENT_SEC)),
    )


def clean_segment_timeline(
    segments: Iterable[tuple[str, float, float]],
    *,
    max_segment_sec: float = DEFAULT_MAX_SEGMENT_SEC,
    stats: dict[str, Any] | None = None,
) -> Iterator[tuple[str, float, float]]:
    """把 ASR 段级时间戳清洗成「单调不重叠 + 单段不超长」的字幕时间轴。

    输入/输出都是 `(text, start_sec, end_sec)`：**文本原样透传，绝不增删条目**，
    调用方负责先把空文本段过滤掉（空段没有字幕行，不该参与 lookahead）。

    清洗规则见 `_clean_segment_bounds`；lookahead 用「下一条的 start」收紧上一段，
    因此本生成器会缓冲一条：只有读到下一条（或流结束）才产出上一段。

    `stats` 可选传入 dict，耗尽后填入汇总（供调用方记录一条 WARNING）：
    `total`/`adjusted`/`max_original_sec`/`max_output_sec`/`max_segment_sec`。
    """

    limit = coerce_max_segment_sec(max_segment_sec)
    total = 0
    adjusted = 0
    max_original = 0.0
    max_output = 0.0
    prev_end = 0.0
    pending: tuple[str, float, float] | None = None

    def _finalize(
        item: tuple[str, float, float], next_start: float | None
    ) -> tuple[str, float, float]:
        """把一条缓冲段清洗后返回，并累计统计量。"""

        nonlocal adjusted, max_original, max_output, prev_end
        text, start, end = item
        cleaned_start, cleaned_end = _clean_segment_bounds(
            start=start,
            end=end,
            prev_end=prev_end,
            next_start=next_start,
            max_segment_sec=limit,
        )
        if (
            abs(cleaned_start - start) > _TIMESTAMP_EPSILON
            or abs(cleaned_end - end) > _TIMESTAMP_EPSILON
        ):
            adjusted += 1
        # 原始时长按字面记录（含倒挂段的负值归零），仅用于观测与告警。
        max_original = max(max_original, max(end - start, 0.0))
        max_output = max(max_output, max(cleaned_end - cleaned_start, 0.0))
        prev_end = cleaned_end
        return text, cleaned_start, cleaned_end

    for text, raw_start, raw_end in segments:
        current = (text, float(raw_start), float(raw_end))
        total += 1
        if pending is not None:
            yield _finalize(pending, current[1])
        pending = current
    if pending is not None:
        yield _finalize(pending, None)

    if stats is not None:
        stats.update(
            {
                "total": total,
                "adjusted": adjusted,
                "max_original_sec": max_original,
                "max_output_sec": max_output,
                "max_segment_sec": limit,
            }
        )


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
# 单次请求的输出 token 上限：必须显式下发，否则上游默认上限会在"大批量 + 长译文"
# 时把 JSON 截断，表现为 parse 错误后重试空转（2026-11-01 实测单批 200 条曾耗 303s 仍解析失败）。
DEFAULT_TRANSLATE_MAX_TOKENS = 16000
# 关思考：显式下发 reasoning.enabled=false。当前 OpenRouter + deepseek-v3.2 路由的
# reasoning_tokens 恒为 0（推理写在正文），该参数对这类路由无可测收益；但对后续
# 会走独立思考 token 的系列模型是必要开关，故默认关闭思考并保留配置。
DEFAULT_TRANSLATE_REASONING_ENABLED = False


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
    """构造翻译消息，强调剧情上下文推理和全段一致性。

    输出契约（修复"整段后移一位"）：从「对象数组 [{id, text_zh}]」改为「以 id 为
    key 的 JSON 对象 {'<id>': '<中文>'}」。数组契约下模型偶发把一条日文拆成
    两条、又丢掉最后一条时，返回的 id 仍可能是连续的 84..107，本地只校验「id 集合
    相等」就发现不了，整段译文后移一位。改成 id 作为 key 后，译文与 id 的绑定由
    解析层直接按 key 建立，不再依赖数组位置，行数也不再需要和长度对齐。
    """

    payload_items = [{"id": e.index, "text": e.text} for e in batch]
    item_count = len(payload_items)
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
        "Translate the following JSON array of Japanese subtitles into Simplified Chinese subtitles. "
        f"The input contains exactly {item_count} subtitle items, each with a numeric \"id\". "
        "Return ONLY ONE JSON object that maps every input id, as a string key, to that single "
        'item\'s Chinese translation string, i.e. {"<id>": "<Chinese translation>", ...}. '
        "Hard constraints on the output object:\n"
        f"- it MUST have exactly one key for EVERY input id ({item_count} keys in total, "
        "no more, no fewer) and no other keys;\n"
        "- never merge two input items into one entry, never split one input item into several "
        "entries, and never add, drop or renumber ids: the number of keys MUST equal the number "
        "of input items;\n"
        "- every value is one single string (no nested objects/arrays), and it must translate only "
        "the text of its own id.\n"
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


class JsonExtractionError(ValueError):
    """上游响应里确实不存在任何可用的 JSON 值（parse 类错误，可重试）。

    继承 `ValueError` 是为了与 `json.loads` 抛出的 `JSONDecodeError` 保持同一
    语义：`translate._error_class()` 会把它归为 `parse`，
    `translate._is_retryable()` 判定为可重试——不再因为"文本里连一个括号都没有"
    就退化成不可重试错误、当场放弃任务（旧实现正是如此）。
    """


# 顶层 JSON 值的结构化解码器：`raw_decode(s, idx)` 从 idx 处解码**一个** JSON 值并
# 返回 (值, 结束偏移)，因此可以只吃下第一个 JSON 值、完全忽略其后的多余内容。
_JSON_DECODER = json.JSONDecoder()

# 候选起点尝试上限：畸形输出（例如成千上万个未闭合的 `{`）会让"逐个 `{`/`[` 试
# 解码"退化成 O(n²)；正常响应里的候选起点只有个位数，256 足够兜住真实文本。
_MAX_JSON_CANDIDATE_ATTEMPTS = 256


def _iter_json_values(payload: str) -> Iterator[tuple[int, Any]]:
    """按出现顺序产出 payload 中的顶层 JSON 值及其起始偏移。

    只在 `{` / `[` 处尝试解码：
    - 解码成功即跳到该值末尾继续扫描（内层对象不会被当成独立值）；
    - 解码失败（例如正文里恰好出现 `{`）则前进一个字符再试。
    这样正文里的普通数字/汉字/单词不会被误判成 JSON 值。
    """

    index = 0
    attempts = 0
    length = len(payload)
    while index < length and attempts < _MAX_JSON_CANDIDATE_ATTEMPTS:
        if payload[index] not in "{[":
            index += 1
            continue
        attempts += 1
        try:
            value, end = _JSON_DECODER.raw_decode(payload, index)
        except ValueError:
            # JSONDecodeError 是 ValueError 的子类：该起点不是合法 JSON，继续往后找。
            index += 1
            continue
        yield index, value
        index = end


def _looks_like_id_keyed_payload(value: Any) -> bool:
    """判断值是否是「以 id 为 key 的 JSON 对象」（新版翻译输出契约）。

    只做形状判定：非空对象、且**每个** key 都能 `int()` 成整数。JSON 对象的 key
    永远是字符串（`{"84": "..."}`），这里同时接受字符串与整数 key，便于单测直接
    构造 Python dict。任一 key 不是数字即判定失败——避免把 `{"note": "ok"}` 这类
    元数据对象当成译文载荷。
    """

    if not isinstance(value, dict) or not value:
        return False
    for key in value:
        if isinstance(key, bool) or not isinstance(key, (str, int)):
            return False
        try:
            int(str(key).strip())
        except (TypeError, ValueError):
            return False
    return True


def _looks_like_expected_payload(
    value: Any,
    required_keys: tuple[str, ...] | None,
    *,
    accept_id_keyed: bool = False,
) -> bool:
    """判断某个 JSON 值是否"像想要的载荷"（决定是否跳过它继续看下一个值）。

    未指定 `required_keys` 时只排除 None/空容器；指定后要求对象本身、或数组里
    至少一个元素命中其中一个字段名——用于跨过上游偶尔先吐出来的空对象/元数据
    对象（例如先 `{"note": "ok"}` 再给真正的数组）。

    `accept_id_keyed=True` 时额外把「以 id 为 key 的对象」视为可用载荷：新契约
    下模型返回 `{"84": "..."}`，其 key 不是 `id`/`text_zh` 这类字段名，若不加这条
    判定，前面若先出现一个元数据对象就会把它当载荷返回。
    """

    if accept_id_keyed and _looks_like_id_keyed_payload(value):
        return True
    if required_keys is None:
        if isinstance(value, (dict, list)):
            return bool(value)
        return True
    if isinstance(value, dict):
        return any(key in value for key in required_keys)
    if isinstance(value, list):
        return any(
            isinstance(item, dict) and any(key in item for key in required_keys)
            for item in value
        )
    return False


def _extract_json_object(
    text: str,
    *,
    required_keys: tuple[str, ...] | None = None,
    accept_id_keyed: bool = False,
) -> Any:
    """从模型返回文本中提取 JSON 对象/数组。

    P2 修复（线上真实反例）：上游返回"JSON 值 + 尾部多余内容"——两个 JSON 值
    直接拼接，或 JSON 后跟说明文字（说明里还带括号）——旧实现"从首个 `[`/`{`
    截到最后一个 `]`/`}` 再整体 loads"会把尾部一起塞进 `json.loads()`，于是
    稳定抛 `Extra data: line 1 column 31 (char 30)`，同一批次重试全部 parse
    失败、整个任务失败。

    现在的取值顺序：
    1. 先按原样整段解析（保留代码围栏 / `json` 前缀的旧处理），命中即返回；
    2. 否则用 `json.JSONDecoder().raw_decode()` 从每一处 `{` / `[` 起始解码，
       返回**第一个可用**的 JSON 值并忽略其后的全部内容；
    3. `required_keys` 用来判定"首个值是否可用"（缺字段/为空 = 不可用，例如
       上游先吐出来的元数据对象）：此时才继续看后面出现的下一个 JSON 值；
    4. 没有任何候选满足 `required_keys` 时退回第一个解码出的值——解析层不替
       上层判定业务形状（数组/对象由调用方自己校验），不把可救的响应判死；
    5. 一个 JSON 值都解不出来时才抛 `JsonExtractionError`（可重试的 parse
       错误），绝不静默返回空结果，也绝不丢批次。

    "取首个、忽略尾部"是刻意选择：模型有时会在答案后面附格式示例
    （如 `[{"id": 1, "text_zh": "..."}]`），合并会把示例当成译文写进字幕。
    首个值缺 id 时不会丢整批：调用方 C.5 会先接受已返回的子集、只补缺失 id，
    最后仍缺的照旧保留日文原文，绝不丢字幕行。

    `accept_id_keyed=True` 时，除了 `required_keys` 命中的对象/数组，**以 id 为 key
    的对象**（新契约 `{"84": "..."}`）同样算"可用载荷"，见
    `_looks_like_expected_payload`。`required_keys` 与它的判定是"或"关系。
    """

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

    candidates = list(_iter_json_values(payload))
    if not candidates:
        # 明确失败（可重试的 parse 类错误）：绝不返回空结果让上层当成"翻译完了"。
        raise JsonExtractionError("unable to extract JSON value from response")
    for _offset, value in candidates:
        if _looks_like_expected_payload(
            value, required_keys, accept_id_keyed=accept_id_keyed
        ):
            return value
    # 形状都不符合预期：退回首个值，交由上层按既有契约判定（不在这里吞掉）。
    return candidates[0][1]


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
    # 请求体透传：输出上限与是否允许思考（见上方常量说明）。
    max_tokens: int = DEFAULT_TRANSLATE_MAX_TOKENS
    reasoning_enabled: bool = DEFAULT_TRANSLATE_REASONING_ENABLED


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

    HTTP 单批超时（`timeout_sec` 字段）的来源固定为：
    `job_config["timeout"]` > 旧关键字 `timeout_sec` > `[translation] timeout`
    > 代码默认值。`job_config["timeout_sec"]` 是**任务级**超时（runtime 用它做
    总 deadline，来源是 `[timeouts]` 或显式任务覆盖），绝不能再被解读成 HTTP
    超时，否则任务超时与单批超时会互相污染（P1 回归）。
    """

    overrides: dict[str, Any] = dict(job_config or {})
    if chunk_minutes is not None:
        overrides["chunk_minutes"] = chunk_minutes
    if retry is not None:
        overrides["retry"] = retry
    # 旧关键字 timeout_sec 只作为 HTTP 超时的兼容回退（见下方 http_timeout_override）；
    # 不写入 overrides，避免它被当成任务级键继续向下游扩散。
    http_timeout_override = overrides.get("timeout")
    if http_timeout_override is None:
        http_timeout_override = timeout_sec
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
        "max_tokens",
        "reasoning_enabled",
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
        if key == "timeout_sec":
            # HTTP 超时专用：job_config["timeout_sec"] 属于任务级 deadline，
            # 这里显式忽略它，只认 timeout 键/旧关键字；配置段里对应的键名是
            # `[translation] timeout`（不是 timeout_sec）。
            value = http_timeout_override
            if value is None:
                value = _read_translation_section(config, "timeout", kind="int")
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
    # max_tokens=0 表示"不下发该字段"（沿用上游默认），负值同样归零。
    resolved["max_tokens"] = max(int(resolved["max_tokens"]), 0)
    return TranslateSettings(**resolved)
