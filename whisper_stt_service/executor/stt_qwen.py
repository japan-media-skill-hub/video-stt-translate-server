"""stt_qwen 阶段执行器：FireRedVAD 定点切片 + 本地 Qwen3-ASR 批处理。

与既有 `stt`（faster-whisper）/`stt_whisperx` **并列**的第三个 ASR 阶段，产物契约
完全一致：`artifacts/<job_id>/asr/<stem>.ja.srt`，因此 `translate` 无需任何改动。

流水线（每一步的取舍都有实测依据，见
`docs/superpowers/specs/2026-11-01-stt-qwen-execution-spec.md`）：

    读 wav → FireRedVAD 一次前向（拿到 100Hz 帧概率 + 语音段）
      → pick_cuts（纯函数，见 qwen_boundary.py）定点切片
      → 按批喂本地 Qwen3-ASR-1.7B（+ ForcedAligner 出字级时间戳）
      → 字级结果聚合成 SRT 行并叠加片偏移
      → clean_segment_timeline 清洗（与 stt.py/stt_whisperx.py 同一函数）
      → 原子写盘

必须保住的行为（SPEC §6.2，任何一条退化都算缺陷）：

1. **增量落盘**：每批成功即重写目标 SRT，异常路径 `finally` 也写，绝不白烧算力；
2. **空对齐兜底**：`time_stamps is None` → 用片边界出一行并计数（历史实测崩过一次，13 分钟算力报废）；
3. **空文本片**：只计数不出行；
4. **四类计数**（`fallback`/`hard_cut`/`align_missing`/`cleanup_adjusted`）全部进日志与快照，禁止静默降级；
5. **GPU 落点校验 + 显存门槛**：不是"加载成功"就算过，必须确认参数真的在 cuda 上；
6. **按片长收紧 `max_new_tokens`**：退化生成会一路打到上限（实测 64/512/2048 token → 3.73s/21.69s/85.9s
   线性相关），上限必须与片长匹配而不是固定 2048。
"""

from __future__ import annotations

import logging
import math
import re
import time
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Iterable, Sequence
import wave

from whisper_stt_service.executor.atomic import atomic_write_text
from whisper_stt_service.executor.common import (
    DEFAULT_MAX_SEGMENT_SEC,
    ExecutorCancelledError,
    MissingAudioArtifactError,
    _emit_progress,
    _log_timestamp_cleanup_summary,
    clean_segment_timeline,
    coerce_max_segment_sec,
)
from whisper_stt_service.executor.qwen_boundary import (
    MAX_CHUNK_SECONDS,
    MIN_TAIL_SECONDS,
    BoundaryParams,
    QwenBoundaryConfigError,
    merge_short_tail,
    pick_cuts,
)
# 复用同包内既有实现，避免第三份复制（stt.py:40 已有一份历史副本，此处不再新增漂移点）：
#   _format_timestamp   —— SRT 时间戳格式化
#   _check_cancel       —— 协作取消检查点
#   _ensure_audio_artifact —— wav 产物存在且非空（错误码 missing_audio_artifact）
from whisper_stt_service.executor.stt_whisperx import (
    _check_cancel,
    _ensure_audio_artifact,
    _format_timestamp,
)
from whisper_stt_service.service.model_cache import get_model_cache


logger = logging.getLogger(__name__)

# 协作取消回调类型：返回 True 表示"本任务已被回收/停机"，执行器必须尽快中止。
CancelCheck = Callable[[], bool]

# FireRedVAD 帧移：100Hz（每帧 10ms）。切片吸附与语音占比都按该栅格换算。
_VAD_FRAME_SEC = 0.01

# `max_new_tokens` 的收敛区间：下限保证短片的正常输出不被截断，上限沿用包内默认。
_MIN_MAX_NEW_TOKENS = 256
_MAX_MAX_NEW_TOKENS = 2048

# 句末标点：命中即可断行（同时受字数/时长上限约束）。
_SENTENCE_END = "。！？!?…"
# 次级标点：仅在超过软预算时才允许在此断行。
_CLAUSE_END = "、,，"

# SRT 行聚合预算：单行最多 20 字 / 6 秒（与原型一致，实测可读性最好）。
_LINE_MAX_CHARS = 20
_LINE_MAX_SEC = 6.0

# "有实际内容"的字符：假名 / 汉字 / 拉丁字母 / 数字。
# 只用来判定**纯标点行**（如单独一个「。」）——静音片被模型输出成句号时会产生
# 这种行，对齐器又会给它一整片的时长，表现为"30 秒只有一个句号"的噪声字幕。
# 2026-11-01 真片验收实测：2049 行里有 6 行属于此类（SAN 4 / CAWB 2 / ABF 0），
# 且 SAN 的 4 行正好落在 `chunk_zero_speech=5` 的那几个静音片上（间隔恰为 70s 上界）。
# 判定为噪声的行会**丢弃并计数**（`punct_only_lines`），绝不静默处理。
_CONTENT_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]")


def _has_content(text: str) -> bool:
    """文本里是否至少有一个"实义字符"（假名/汉字/字母/数字）。"""

    return _CONTENT_CHAR_RE.search(str(text or "")) is not None

# `time_stamps=None` 时的兜底策略。目前只支持"退回片边界出单行"：
# 另一种做法（丢掉这一段）会静默丢内容，直接不允许配置。
_ALIGN_FALLBACK_CHUNK_BOUNDARY = "chunk_boundary"

# ISO 639-1 / 常见别名 → Qwen3-ASR 规范语言名（包内只认规范名，'ja' 会被判不支持）。
# 来源：`qwen_asr.inference.utils.SUPPORTED_LANGUAGES`（30 种）。
_QWEN_LANGUAGE_ALIASES: dict[str, str] = {
    "zh": "Chinese",
    "cmn": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "jp": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "tl": "Filipino",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}

# 规范语言名（小写）→ 规范名：让配置里直接写 "Japanese" 也能通过。
_QWEN_LANGUAGE_NAMES: dict[str, str] = {
    name.lower(): name for name in set(_QWEN_LANGUAGE_ALIASES.values())
}

# dtype 字符串 → torch dtype 名（真实对象在加载时惰性取，避免模块导入期依赖 torch）。
_DTYPE_NAMES = ("bfloat16", "float16", "float32")


def _resolve_torch_dtype(name: str) -> Any:
    """把配置里的 dtype 字符串解析成 torch dtype；非法值 fail-fast。"""

    key = str(name or "").strip().lower()
    if key not in _DTYPE_NAMES:
        raise QwenBoundaryConfigError(f"unsupported_dtype: {name}")
    import torch  # type: ignore

    return getattr(torch, key)


def resolve_qwen_language(language: str) -> str:
    """把服务里的语言标识解析成 Qwen3-ASR 规范语言名。

    包内 `validate_language` 只认规范名（`normalize_language_name("ja")` 会得到
    "Ja" 并直接判不支持），因此这里必须显式映射；映射不到时 fail-fast，
    绝不静默退化成"自动检测"（那会悄悄改变识别语言）。
    """

    raw = str(language or "").strip()
    if not raw:
        raise QwenBoundaryConfigError("language_empty")
    lowered = raw.lower()
    if lowered in _QWEN_LANGUAGE_ALIASES:
        return _QWEN_LANGUAGE_ALIASES[lowered]
    if lowered in _QWEN_LANGUAGE_NAMES:
        return _QWEN_LANGUAGE_NAMES[lowered]
    raise QwenBoundaryConfigError(f"unsupported_language: {raw}")


def _read_wav_16k_mono(path: Path) -> tuple[Any, int]:
    """读 16k mono pcm_s16le wav，返回 `(float32[-1,1] 波形, 采样率)`。

    输入来自 `extract` 阶段的固定产物（SPEC B.2-3：16k mono pcm_s16le），
    因此这里不做重采样：格式不符直接失败，避免"悄悄换了采样率"。
    """

    import numpy as np  # type: ignore

    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        raw = handle.readframes(handle.getnframes())
    if width != 2:
        raise RuntimeError(f"stt_qwen_expect_pcm_s16le: sampwidth={width}")
    if sample_rate != 16000:
        raise RuntimeError(f"stt_qwen_expect_16k: sample_rate={sample_rate}")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sample_rate


def _group_lines(
    items: Iterable[Any],
    max_chars: int = _LINE_MAX_CHARS,
    max_sec: float = _LINE_MAX_SEC,
    *,
    max_end: float | None = None,
) -> list[tuple[float, float, str]]:
    """把字级对齐结果聚合成 SRT 行：句末标点优先断行，其次受字数/时长上限约束。

    返回片内相对时间的 `(start, end, text)`；调用方负责叠加片偏移。

    `max_end` 是**片内时长**：对齐模型偶发给出远超片长的 end（历史 886.7s 级
    逃逸时间戳），若不在这里夹住，下面"保证单调"的兜底会把后续行的 start
    推到片长之外，最终写出**起点超过音频时长**的字幕（单测已覆盖这条）。
    因此每个字级结果的 start/end 都先夹到 `[0, max_end]`。
    """

    lines: list[tuple[float, float, str]] = []
    buf: list[str] = []
    start: float | None = None
    end: float | None = None
    for item in items:
        text = str(getattr(item, "text", "") or "")
        if text == "":
            continue
        item_start = float(item.start_time)
        item_end = float(item.end_time)
        if max_end is not None:
            item_start = min(max(item_start, 0.0), float(max_end))
            item_end = min(max(item_end, 0.0), float(max_end))
        if start is None:
            start = item_start
        end = item_end
        buf.append(text)
        joined = "".join(buf)
        hit_end = joined.rstrip()[-1:] in _SENTENCE_END and len(joined) >= 4
        over = len(joined) >= max_chars or (end - (start or 0.0)) >= max_sec
        soft = joined.rstrip()[-1:] in _CLAUSE_END and len(joined) >= max_chars * 0.6
        if hit_end or over or soft:
            lines.append((start, end, joined.strip()))
            buf, start, end = [], None, None
    if buf and start is not None and end is not None:
        lines.append((start, end, "".join(buf).strip()))
    # 兜底：保证行内时间戳单调且时长为正（对齐模型偶发给出倒挂区间）。
    fixed: list[tuple[float, float, str]] = []
    last_end = 0.0
    for start_sec, end_sec, text in lines:
        start_sec = max(start_sec, last_end)
        end_sec = max(end_sec, start_sec + 0.2)
        fixed.append((start_sec, end_sec, text))
        last_end = end_sec
    return fixed


def _render_srt(lines: Sequence[tuple[float, float, str]]) -> str:
    """把 `(start, end, text)` 列表渲染成 SRT 文本。"""

    parts: list[str] = []
    for index, (start_sec, end_sec, text) in enumerate(lines, start=1):
        parts.append(
            f"{index}\n{_format_timestamp(start_sec)} --> "
            f"{_format_timestamp(end_sec)}\n{text}\n\n"
        )
    return "".join(parts)


def _speech_ratio(probs: Any, start_sec: float, end_sec: float, threshold: float) -> float:
    """片内语音帧占比（只用于观测，不参与切片决策）。"""

    total = int(len(probs))
    first = max(int(start_sec / _VAD_FRAME_SEC), 0)
    last = min(int(end_sec / _VAD_FRAME_SEC), total)
    if last <= first:
        return 0.0
    return float((probs[first:last] >= threshold).mean())


def build_stt_qwen_effective_config(
    *,
    language: str,
    timeout_sec: int,
    max_retries: int,
    model_dir: str,
    aligner_dir: str,
    vad_model_dir: str,
    device: str,
    dtype: str,
    batch_size: int,
    params: BoundaryParams,
    tokens_per_sec: float,
    max_segment_sec: float,
    vad_use_gpu: bool,
    vad_speech_threshold: float,
    min_free_vram_gb: float,
    min_coverage: float,
    strict_output: bool,
    align_missing_fallback: str,
) -> dict[str, Any]:
    """构造 stt_qwen 生效参数快照，便于日志审计与问题排查（SPEC §6.4）。"""

    return {
        "language": language,
        "timeout_sec": int(timeout_sec),
        "max_retries": int(max_retries),
        "model_dir": model_dir,
        "aligner_dir": aligner_dir,
        "vad_model_dir": vad_model_dir,
        "device": device,
        "dtype": dtype,
        "batch_size": max(int(batch_size), 1),
        "nominal_seconds": float(params.nominal_seconds),
        "delta_back_seconds": float(params.delta_back_seconds),
        "delta_forward_seconds": float(params.delta_forward_seconds),
        "fallback_forward_seconds": float(params.fallback_forward_seconds),
        "min_chunk_seconds": float(params.min_chunk_seconds),
        "hard_cut_on_miss": bool(params.hard_cut_on_miss),
        "max_chunk_seconds": params.max_chunk_seconds,
        "absolute_max_chunk_seconds": MAX_CHUNK_SECONDS,
        "min_tail_seconds": MIN_TAIL_SECONDS,
        "tokens_per_sec": float(tokens_per_sec),
        "max_segment_sec": float(max_segment_sec),
        "vad_use_gpu": bool(vad_use_gpu),
        "vad_speech_threshold": float(vad_speech_threshold),
        "min_free_vram_gb": float(min_free_vram_gb),
        "min_coverage": float(min_coverage),
        "strict_output": bool(strict_output),
        "align_missing_fallback": align_missing_fallback,
    }


def _check_paths(
    *, model_dir: Path, aligner_dir: Path, vad_model_dir: Path
) -> None:
    """三个本地模型目录必须存在（运行期禁止联网下载，缺失即 fail-fast）。"""

    if not model_dir.is_dir():
        raise FileNotFoundError(f"stt_qwen_model_missing: {model_dir}")
    if not aligner_dir.is_dir():
        raise FileNotFoundError(f"stt_qwen_aligner_missing: {aligner_dir}")
    if not vad_model_dir.is_dir():
        raise FileNotFoundError(f"stt_qwen_vad_missing: {vad_model_dir}")


def _resolve_device(device: str, *, require_cuda: bool, min_free_vram_gb: float) -> str:
    """校验设备可用性与显存门槛，返回实际生效的设备字符串。

    - `device` 不是 cuda 系 → 原样返回（CPU 调试路径）；
    - cuda 不可用：`require_cuda=True`（默认）直接 `stt_qwen_cuda_unavailable`；
      `require_cuda=False` 才显式降级到 CPU 并留 WARNING——绝不做无声降级；
    - 显存门槛：加载前先看空闲显存（实测峰值 9.6–11.9GB），不足即
      `stt_qwen_insufficient_vram`，别等 OOM 崩在半个片中间。
    """

    import torch  # type: ignore

    resolved = str(device or "").strip() or "cuda:0"
    if not resolved.startswith("cuda"):
        return resolved
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError(
                f"stt_qwen_cuda_unavailable: device={resolved} but CUDA is not available"
            )
        logger.warning(
            "stt_qwen cuda unavailable, falling back to cpu "
            "(require_cuda=false): device=%s",
            resolved,
        )
        return "cpu"
    if float(min_free_vram_gb) > 0:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024**3)
        if free_gb < float(min_free_vram_gb):
            raise RuntimeError(
                "stt_qwen_insufficient_vram: "
                f"free={free_gb:.2f}GiB total={total_bytes / (1024**3):.2f}GiB "
                f"required={float(min_free_vram_gb):.2f}GiB"
            )
    return resolved


def _assert_on_device(model: Any, device: str) -> None:
    """确认模型参数真的落在目标设备上（SPEC §6.2-5）。

    "加载没报错"不等于"跑在 GPU 上"：`device_map` 写错时 transformers 会静默把
    部分层放到 CPU，推理慢几十倍但结果照出。这里显式遍历参数集合，发现任何
    非目标设备的落点立即失败。
    """

    inner = getattr(model, "model", model)
    devices = {str(param.device) for _name, param in inner.named_parameters()}
    expected = {str(device)}
    # `cuda` 与 `cuda:0` 是同一落点的两种写法，比较时统一成 `cuda:<idx>` 形式。
    normalized = {
        item if ":" in item else f"{item}:0" for item in (devices | expected)
    }
    if len(normalized) != 1:
        raise RuntimeError(f"stt_qwen_not_on_gpu: expected={sorted(expected)} actual={sorted(devices)}")


def run_stt_qwen(
    input_video: Path | None = None,
    output_ja_srt: Path | None = None,
    language: str = "ja",
    timeout_sec: int = 5400,
    *,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    model_dir: str,
    aligner_dir: str,
    vad_model_dir: str,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    require_cuda: bool = True,
    min_free_vram_gb: float = 12.0,
    vad_use_gpu: bool = True,
    vad_speech_threshold: float = 0.4,
    nominal_seconds: float = 30.0,
    delta_back_seconds: float = 20.0,
    delta_forward_seconds: float = 20.0,
    fallback_forward_seconds: float = 40.0,
    min_chunk_seconds: float = 10.0,
    hard_cut_on_miss: bool = True,
    batch_size: int = 10,
    tokens_per_sec: float = 16.0,
    max_segment_sec: float = DEFAULT_MAX_SEGMENT_SEC,
    min_coverage: float = 0.98,
    strict_output: bool = False,
    align_missing_fallback: str = _ALIGN_FALLBACK_CHUNK_BOUNDARY,
    progress_every: int = 1,
    progress_queue: Queue[dict[str, Any]] | None = None,
    task_id: str | None = None,
    worker_id: str | None = None,
    require_audio_artifact: bool = False,
    max_retries: int = 0,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    """在 service 内部执行 stt_qwen 阶段，返回生效配置快照。

    `input_video`/`output_ja_srt` 为历史参数名，同时接受 io 契约别名
    `input_path`/`output_path`（别名优先，与 `run_stt_whisperx` 一致）。
    `require_audio_artifact=True` 时强制校验输入 wav 存在且非空，否则抛
    `MissingAudioArtifactError`（`missing_audio_artifact`），绝不回退原视频。
    `cancel_check` 为协作取消回调（失租/停机），返回 True 时中止且不发布产物。
    """

    resolved_in = input_path if input_path is not None else input_video
    resolved_out = output_path if output_path is not None else output_ja_srt
    if resolved_in is None or resolved_out is None:
        raise ValueError("stt_qwen_io_paths_missing")
    input_wav = Path(resolved_in)
    output_srt = Path(resolved_out)
    output_srt.parent.mkdir(parents=True, exist_ok=True)

    if require_audio_artifact:
        _ensure_audio_artifact(input_wav)

    # ---- 参数校验（fail-fast，全部错误码进 error 字段）----
    params = BoundaryParams(
        nominal_seconds=nominal_seconds,
        delta_back_seconds=delta_back_seconds,
        delta_forward_seconds=delta_forward_seconds,
        fallback_forward_seconds=fallback_forward_seconds,
        min_chunk_seconds=min_chunk_seconds,
        hard_cut_on_miss=bool(hard_cut_on_miss),
    )
    params.validate()
    safe_batch_size = max(int(batch_size), 1)
    safe_tokens_per_sec = float(tokens_per_sec)
    if safe_tokens_per_sec <= 0:
        raise QwenBoundaryConfigError("tokens_per_sec_must_be_positive")
    resolved_max_segment_sec = coerce_max_segment_sec(max_segment_sec)
    resolved_language = resolve_qwen_language(language)
    resolved_dtype = str(dtype).strip().lower()
    if resolved_dtype not in _DTYPE_NAMES:
        raise QwenBoundaryConfigError(f"unsupported_dtype: {dtype}")
    fallback_mode = str(align_missing_fallback or "").strip()
    if fallback_mode != _ALIGN_FALLBACK_CHUNK_BOUNDARY:
        raise QwenBoundaryConfigError(
            f"unsupported_align_missing_fallback: {align_missing_fallback}"
        )

    model_path = Path(model_dir)
    aligner_path = Path(aligner_dir)
    vad_path = Path(vad_model_dir)
    _check_paths(model_dir=model_path, aligner_dir=aligner_path, vad_model_dir=vad_path)

    # 显存门槛必须先区分"冷启动"与"命中缓存"（2026-11-01 验收实测缺陷）：
    # 上一部片跑完后模型仍被 model_cache 持有，`torch.cuda.mem_get_info()` 会把
    # 缓存分配器持有的块算成**已用**，看到 free≈0；此时再要求"空闲 ≥ 12GB"
    # 会让第 2/3 部片连续以 stt_qwen_insufficient_vram 失败（实测 3 次重试全废）。
    # 命中缓存 = 显存已经在进程里，不需要新的空闲额度，因此跳过门槛并**记录跳过**。
    bundle_kwargs = dict(
        model_dir=str(model_path),
        aligner_dir=str(aligner_path),
        vad_model_dir=str(vad_path),
        device=str(device),
        dtype=str(dtype).strip().lower(),
        max_new_tokens=_effective_max_new_tokens(
            params.max_chunk_seconds, safe_tokens_per_sec
        ),
        batch_size=safe_batch_size,
        vad_use_gpu=bool(vad_use_gpu),
        vad_speech_threshold=float(vad_speech_threshold),
    )
    bundle_cached = bool(get_model_cache().has_qwen_asr_bundle(**bundle_kwargs))
    # 设备门槛：加载前先挡住，并把真实生效的设备写进快照。
    resolved_device = _resolve_device(
        device,
        require_cuda=bool(require_cuda),
        min_free_vram_gb=0.0 if bundle_cached else float(min_free_vram_gb),
    )
    if bundle_cached:
        logger.info(
            "stt_qwen vram gate skipped: task_id=%s reason=bundle_cached "
            "min_free_vram_gb=%.2f",
            task_id,
            float(min_free_vram_gb),
        )

    effective_config = build_stt_qwen_effective_config(
        language=resolved_language,
        timeout_sec=int(timeout_sec),
        max_retries=int(max_retries),
        model_dir=str(model_path),
        aligner_dir=str(aligner_path),
        vad_model_dir=str(vad_path),
        device=resolved_device,
        dtype=resolved_dtype,
        batch_size=safe_batch_size,
        params=params,
        tokens_per_sec=safe_tokens_per_sec,
        max_segment_sec=resolved_max_segment_sec,
        vad_use_gpu=bool(vad_use_gpu),
        vad_speech_threshold=float(vad_speech_threshold),
        min_free_vram_gb=float(min_free_vram_gb),
        min_coverage=float(min_coverage),
        strict_output=bool(strict_output),
        align_missing_fallback=fallback_mode,
    )
    # 显存门槛是否因"命中缓存"被跳过：必须留痕，不许变成不可见的行为差异。
    effective_config["vram_gate_skipped"] = bundle_cached

    samples, sample_rate = _read_wav_16k_mono(input_wav)
    duration = float(len(samples)) / float(sample_rate)
    if duration <= 0:
        raise RuntimeError("stt_qwen_empty_audio")
    effective_config["audio_duration_sec"] = round(duration, 3)

    started = time.perf_counter()
    _emit_progress(
        progress_queue,
        stage="stt_qwen",
        percent=0.0,
        message="stt_qwen_started",
        task_id=task_id,
        worker_id=worker_id,
    )

    # 当前已算出的行（增量落盘与最终清洗都基于它）。
    lines: list[tuple[float, float, str]] = []
    # 四类计数：任何一类都不能是"隐藏值"，最终全部写进快照与日志。
    align_missing_chunks = 0
    empty_chunks = 0
    failed_chunks = 0
    # 纯标点行（静音片被输出成「。」等）：丢弃但必须计数，不许静默。
    punct_only_lines = 0
    tail_merged_sec = 0.0
    batch_times: list[float] = []
    total_chunks = 0
    batches_total = 0

    def _flush_srt() -> None:
        """增量落盘：把当前已完成的行原子写成 SRT（崩溃也不丢已算内容）。"""

        if not lines:
            return
        atomic_write_text(output_srt, _render_srt(lines))

    lease = None
    try:
        # ---- VAD + 切片 + 批量 ASR 全部在"同卡并发许可"内，绝不放行并发 ----
        lease = get_model_cache().get_qwen_asr_bundle(
            model_dir=str(model_path),
            aligner_dir=str(aligner_path),
            vad_model_dir=str(vad_path),
            device=resolved_device,
            dtype=resolved_dtype,
            max_new_tokens=_effective_max_new_tokens(
                params.max_chunk_seconds, safe_tokens_per_sec
            ),
            batch_size=safe_batch_size,
            vad_use_gpu=bool(vad_use_gpu),
            vad_speech_threshold=float(vad_speech_threshold),
        )
        effective_config["model_cache_hit"] = bool(lease.cached)
        effective_config["model_cache_key"] = lease.key
        asr_model, vad_model = lease.resources
        _assert_on_device(asr_model, resolved_device)

        lease.acquire_transcribe_slot(cancel_check=cancel_check)
        try:
            _check_cancel(cancel_check)
            _reset_vram_peak(resolved_device)
            probs, speech_ends, vad_sec = _run_vad(
                vad_model, input_wav, duration=duration
            )
            effective_config["vad_seconds"] = round(vad_sec, 3)
            effective_config["vad_speech_segments"] = len(speech_ends)
            effective_config["vad_speech_frames_ratio"] = round(
                float((probs >= float(vad_speech_threshold)).mean()), 4
            )

            cuts, meta = pick_cuts(
                duration,
                params.nominal_seconds,
                params.delta_back_seconds,
                params.delta_forward_seconds,
                params.fallback_forward_seconds,
                params.min_chunk_seconds,
                params.hard_cut_on_miss,
                speech_ends,
            )
            cuts, tail_merged_sec = merge_short_tail(cuts, duration)
            _check_cancel(cancel_check)

            chunks: list[tuple[float, float]] = []
            bounds = [0.0] + list(cuts) + [duration]
            for start_sec, end_sec in zip(bounds[:-1], bounds[1:]):
                if end_sec - start_sec < MIN_TAIL_SECONDS:
                    # 理论上不会发生（尾片已被 merge_short_tail 处理），真发生了也必须计数。
                    tail_merged_sec += end_sec - start_sec
                    continue
                chunks.append((float(start_sec), float(end_sec)))
            total_chunks = len(chunks)
            if total_chunks == 0:
                raise RuntimeError("stt_qwen_no_chunks")

            ratios = [
                _speech_ratio(probs, start_sec, end_sec, float(vad_speech_threshold))
                for start_sec, end_sec in chunks
            ]
            # 逐片元数据补齐可观测字段（切点处的 VAD 概率 + 片内语音占比）。
            for item, (start_sec, end_sec), ratio in zip(meta, chunks, ratios):
                frame = min(int(float(item["cut"]) / _VAD_FRAME_SEC), len(probs) - 1)
                item["vad_prob"] = round(float(probs[frame]), 6) if len(probs) else None
                item["start"] = round(start_sec, 3)
                item["end"] = round(end_sec, 3)
                item["speech_ratio"] = round(ratio, 4)
            policy_counts = {policy: 0 for policy in ("vad_offset", "vad_offset_extended", "hard_cut")}
            for item in meta:
                policy_counts[str(item["policy"])] = (
                    policy_counts.get(str(item["policy"]), 0) + 1
                )
            effective_config.update(
                {
                    "chunks_total": total_chunks,
                    "cut_policy_counts": policy_counts,
                    "chunk_seconds_min": round(min(b - a for a, b in chunks), 3),
                    "chunk_seconds_max": round(max(b - a for a, b in chunks), 3),
                    "chunk_seconds_mean": round(
                        sum(b - a for a, b in chunks) / total_chunks, 3
                    ),
                    "chunk_zero_speech": int(
                        sum(1 for ratio in ratios if ratio <= 0.0)
                    ),
                    "tail_merged_seconds": round(tail_merged_sec, 6),
                }
            )
            # 硬切/fallback 必须留痕：超过绝对上限的片根本不该存在，出现即告警。
            if policy_counts.get("hard_cut", 0) > 0:
                logger.warning(
                    "stt_qwen hard cut applied: task_id=%s chunks=%s ratio=%.4f "
                    "nominal=%.1fs fallback_forward=%.1fs",
                    task_id,
                    policy_counts["hard_cut"],
                    policy_counts["hard_cut"] / total_chunks,
                    params.nominal_seconds,
                    params.fallback_forward_seconds,
                )
            if policy_counts.get("vad_offset_extended", 0) > 0:
                logger.info(
                    "stt_qwen fallback window used: task_id=%s chunks=%s",
                    task_id,
                    policy_counts["vad_offset_extended"],
                )
            logger.info(
                "stt_qwen cut plan: task_id=%s duration=%.2fs chunks=%s "
                "min=%.2fs max=%.2fs mean=%.2fs policies=%s zero_speech=%s",
                task_id,
                duration,
                total_chunks,
                effective_config["chunk_seconds_min"],
                effective_config["chunk_seconds_max"],
                effective_config["chunk_seconds_mean"],
                policy_counts,
                effective_config["chunk_zero_speech"],
            )

            max_new_tokens = _effective_max_new_tokens(
                params.max_chunk_seconds, safe_tokens_per_sec
            )
            effective_config["max_new_tokens"] = max_new_tokens
            batches_total = (total_chunks + safe_batch_size - 1) // safe_batch_size
            effective_config["batches_total"] = batches_total

            # 片内音频按需切片（不在磁盘上落中间文件）。
            slices: list[Any] = [
                samples[int(start_sec * sample_rate) : int(end_sec * sample_rate)]
                for start_sec, end_sec in chunks
            ]

            inference_started = time.perf_counter()
            try:
                for batch_index in range(0, total_chunks, safe_batch_size):
                    _check_cancel(cancel_check)
                    group_indexes = list(
                        range(batch_index, min(batch_index + safe_batch_size, total_chunks))
                    )
                    group = [chunks[i] for i in group_indexes]
                    batch_started = time.perf_counter()
                    results = asr_model.transcribe(
                        audio=[(slices[i], sample_rate) for i in group_indexes],
                        language=resolved_language,
                        return_time_stamps=True,
                    )
                    batch_times.append(round(time.perf_counter() - batch_started, 2))
                    offset_lines, stats = _collect_batch_lines(
                        group, results, duration=duration
                    )
                    lines.extend(offset_lines)
                    align_missing_chunks += stats["align_missing"]
                    empty_chunks += stats["empty"]
                    failed_chunks += stats["failed"]
                    punct_only_lines += stats["punct_only"]
                    # 每批成功即重写 SRT：崩溃/取消时已算出的内容不丢。
                    _flush_srt()
                    done = min(batch_index + safe_batch_size, total_chunks)
                    if progress_every <= 1 or (batch_index // safe_batch_size) % progress_every == 0:
                        _emit_progress(
                            progress_queue,
                            stage="stt_qwen",
                            # VAD/切片约占 5%，其余按批线性推进。
                            percent=5.0 + 95.0 * done / total_chunks,
                            message="stt_qwen_running",
                            task_id=task_id,
                            worker_id=worker_id,
                        )
                    logger.info(
                        "stt_qwen batch done: task_id=%s batch=%s/%s sec=%.2f "
                        "lines=%s align_missing=%s empty=%s",
                        task_id,
                        batch_index // safe_batch_size + 1,
                        batches_total,
                        batch_times[-1],
                        len(lines),
                        align_missing_chunks,
                        empty_chunks,
                    )
            finally:
                # 异常/取消路径也先把已完成的部分写盘（历史教训：13 分钟算力报废）。
                _flush_srt()
            effective_config["inference_seconds"] = round(
                time.perf_counter() - inference_started, 3
            )
            effective_config["batch_seconds"] = batch_times
            # 峰值显存必须进快照（SPEC §9.3 判据之一）：本轮档实测 9.6~10.4GB，
            # 目标档 batch 13 曾到 11.9GB（torch 口径）。按任务重置峰值再读，
            # 否则长驻服务里读到的是历史最高水位，无法回答"这次跑了多少"。
            effective_config["peak_vram_gb"] = _read_vram_peak(resolved_device)
        finally:
            lease.release_transcribe_slot()
    finally:
        if lease is not None:
            lease.release()

    # ---- 结果校验 + 清洗 + 最终落盘 ----
    if not lines:
        raise RuntimeError("stt_qwen_empty_srt")
    last_end = max(end_sec for _start, end_sec, _text in lines)
    coverage = last_end / duration if duration > 0 else 0.0
    effective_config["lines_total"] = len(lines)
    effective_config["chars_total"] = sum(len(text) for _s, _e, text in lines)
    effective_config["last_end_sec"] = round(last_end, 3)
    effective_config["coverage"] = round(coverage, 4)
    effective_config["align_missing_chunks"] = align_missing_chunks
    effective_config["empty_chunks"] = empty_chunks
    effective_config["failed_chunks"] = failed_chunks
    effective_config["punct_only_lines"] = punct_only_lines
    if align_missing_chunks > 0:
        # 空对齐是"降级但没丢内容"：必须留一条可追溯的告警。
        logger.warning(
            "stt_qwen align missing fallback used: task_id=%s chunks=%s fallback=%s",
            task_id,
            align_missing_chunks,
            fallback_mode,
        )
    if empty_chunks > 0:
        logger.warning(
            "stt_qwen empty chunks: task_id=%s chunks=%s (未产出字幕行)",
            task_id,
            empty_chunks,
        )
    if punct_only_lines > 0:
        # 典型来源：静音片被模型输出成「。」，对齐器再给它一整片时长。
        # 这种行没有信息量，丢弃；但必须留下可追溯的计数与告警。
        logger.warning(
            "stt_qwen punctuation-only lines dropped: task_id=%s lines=%s",
            task_id,
            punct_only_lines,
        )
    if coverage < float(min_coverage):
        effective_config["error"] = "stt_qwen_output_incomplete"
        logger.warning(
            "stt_qwen output incomplete: task_id=%s coverage=%.4f last_end=%.2fs "
            "duration=%.2fs threshold=%.4f",
            task_id,
            coverage,
            last_end,
            duration,
            float(min_coverage),
        )
        if strict_output:
            raise RuntimeError(
                f"stt_qwen_output_incomplete: coverage={coverage:.4f} "
                f"threshold={float(min_coverage):.4f}"
            )

    _check_cancel(cancel_check)
    if time.perf_counter() - started > int(timeout_sec):
        raise TimeoutError("stt_qwen_timeout")

    timestamp_stats: dict[str, Any] = {}
    # 注意 `clean_segment_timeline` 的入参/出参顺序是 `(text, start, end)`，
    # 而本执行器内部统一用 `(start, end, text)`——两次转换都要显式写出来，
    # 顺序写反会静默把文本当成时间戳（单测已覆盖这条）。
    cleaned = [
        (start_sec, end_sec, text)
        for text, start_sec, end_sec in clean_segment_timeline(
            (
                (text, start_sec, min(end_sec, duration))
                for start_sec, end_sec, text in lines
            ),
            max_segment_sec=resolved_max_segment_sec,
            stats=timestamp_stats,
        )
    ]
    atomic_write_text(output_srt, _render_srt(cleaned))
    effective_config["timestamp_cleanup"] = dict(timestamp_stats)
    # 清洗汇总：adjusted>0 说明确实存在超长/错乱段，用 WARNING 让它在日志里可见。
    _log_timestamp_cleanup_summary(timestamp_stats, stage="stt_qwen", task_id=task_id)

    effective_config["rtf"] = round(
        float(effective_config.get("inference_seconds", 0.0)) / duration, 6
    )
    _emit_progress(
        progress_queue,
        stage="stt_qwen",
        percent=100.0,
        message="stt_qwen_done",
        task_id=task_id,
        worker_id=worker_id,
    )
    logger.info(
        "stt_qwen finished: task_id=%s duration=%.2fs chunks=%s batches=%s "
        "lines=%s chars=%s coverage=%.4f rtf=%.4f hard_cut=%s fallback=%s "
        "align_missing=%s empty=%s punct_only=%s cleanup_adjusted=%s",
        task_id,
        duration,
        total_chunks,
        batches_total,
        effective_config["lines_total"],
        effective_config["chars_total"],
        effective_config["coverage"],
        effective_config["rtf"],
        effective_config.get("cut_policy_counts", {}).get("hard_cut", 0),
        effective_config.get("cut_policy_counts", {}).get("vad_offset_extended", 0),
        align_missing_chunks,
        empty_chunks,
        punct_only_lines,
        int(timestamp_stats.get("adjusted", 0)),
    )
    return effective_config


def _reset_vram_peak(device: str) -> None:
    """把 torch 的显存峰值统计归零（仅 cuda；CPU 路径是 no-op）。"""

    if not str(device).startswith("cuda"):
        return
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception as exc:  # noqa: BLE001 - 观测失败不能影响主流程
        logger.warning("stt_qwen reset_peak_memory_stats failed: error=%s", exc)


def _read_vram_peak(device: str) -> float | None:
    """读取本任务期间的 torch 显存峰值（GB）；CPU 路径返回 None。"""

    if not str(device).startswith("cuda"):
        return None
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / (1024**3), 2)
    except Exception as exc:  # noqa: BLE001 - 观测失败不能影响主流程
        logger.warning("stt_qwen max_memory_allocated failed: error=%s", exc)
        return None


def _effective_max_new_tokens(max_chunk_sec: float, tokens_per_sec: float) -> int:
    """按最长片收紧 `max_new_tokens`：`clamp(ceil(max_chunk_sec × tokens_per_sec), 256, 2048)`。

    退化生成会一路打到 `max_new_tokens`，上限直接决定耗时（实测 64/512/2048 →
    3.73s/21.69s/85.9s，约 17ms/token 线性）。固定 2048 会让一次退化烧掉 86 秒。
    """

    budget = math.ceil(max(float(max_chunk_sec), 0.0) * float(tokens_per_sec))
    return int(min(max(budget, _MIN_MAX_NEW_TOKENS), _MAX_MAX_NEW_TOKENS))


def _run_vad(vad_model: Any, wav_path: Path, *, duration: float) -> tuple[Any, list[float], float]:
    """一次前向拿到帧概率与语音段（避免重复提特征），返回 `(probs, 语音段结束点, 耗时秒)`。"""

    import numpy as np  # type: ignore
    import torch  # type: ignore

    vad_started = time.perf_counter()
    # do_postprocess=False：只取原始帧概率，后处理由下面显式调用，
    # 这样同一份 probs 既能算片内语音占比，也能产出语音段边界（单一事实源）。
    _result, probs = vad_model.detect(str(wav_path), do_postprocess=False)
    probs_np = probs.detach().cpu().numpy().reshape(-1).astype(np.float32)
    decisions = vad_model.vad_postprocessor.process(probs_np.tolist())
    segments = vad_model.vad_postprocessor.decision_to_segment(
        decisions, len(probs_np) * _VAD_FRAME_SEC
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    speech_ends = [
        float(end) for _start, end in segments if 0.0 < float(end) <= float(duration) + 1e-6
    ]
    return probs_np, speech_ends, time.perf_counter() - vad_started


def _collect_batch_lines(
    group: Sequence[tuple[float, float]],
    results: Sequence[Any],
    *,
    duration: float,
) -> tuple[list[tuple[float, float, str]], dict[str, int]]:
    """把一批模型结果转成全局时间轴的 SRT 行，并统计三类异常片。

    - `empty`：模型返回空文本（无字幕行，但必须计数）；
    - `align_missing`：有文本但 `time_stamps` 为空 → 用片边界出单行并计数（不丢内容）；
    - `failed`：结果条数少于请求片数（上游异常）→ 只计数告警（无文本可兜底，不编造内容）。
    """

    lines: list[tuple[float, float, str]] = []
    stats = {"empty": 0, "align_missing": 0, "failed": 0, "punct_only": 0}
    returned = list(results) if results is not None else []
    for index, (start_sec, end_sec) in enumerate(group):
        if index >= len(returned):
            # 结果比请求少：该片连文本都没有，无内容可兜底——绝不编造字幕行，
            # 但必须计数告警（静默少一片就是静默丢内容）。
            stats["failed"] += 1
            continue
        result = returned[index]
        text = str(getattr(result, "text", "") or "").strip()
        if not text:
            stats["empty"] += 1
            continue
        raw_stamps = getattr(result, "time_stamps", None)
        stamps = list(raw_stamps) if raw_stamps else []
        if not stamps:
            # 对齐缺失：退回片边界单行（保留内容），并计数告警。
            # 纯标点文本在这里也要判一次：否则会写出"整片时长只有一个句号"的噪声行。
            if not _has_content(text):
                stats["punct_only"] += 1
                continue
            stats["align_missing"] += 1
            lines.append((float(start_sec), min(float(end_sec), duration), text))
            continue
        # 片内时长作为字级时间戳的硬上界：防止逃逸时间戳把后续行推出音频之外。
        chunk_len = max(float(end_sec) - float(start_sec), 0.0)
        for line_start, line_end, line_text in _group_lines(stamps, max_end=chunk_len):
            if not _has_content(line_text):
                # 纯标点行（典型：静音片被输出成「。」）：不产出字幕，但必须计数。
                stats["punct_only"] += 1
                continue
            lines.append(
                (
                    min(max(float(start_sec) + line_start, 0.0), duration),
                    min(float(start_sec) + line_end, duration),
                    line_text,
                )
            )
    return lines, stats
