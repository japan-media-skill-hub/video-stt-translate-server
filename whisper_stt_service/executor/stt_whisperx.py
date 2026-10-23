"""stt_whisperx 阶段执行器。"""

from __future__ import annotations

from pathlib import Path
from queue import Queue
from typing import Any, Callable
import importlib
import logging
import os
import tempfile
import time

from whisper_stt_service.executor.common import (
    ExecutorCancelledError,
    MissingAudioArtifactError,
    _emit_progress,
    _probe_duration,
)
from whisper_stt_service.service.model_cache import get_model_cache


logger = logging.getLogger(__name__)

# 协作取消回调类型：返回 True 表示“本任务已被回收/停机”，执行器必须尽快中止。
CancelCheck = Callable[[], bool]

# WhisperX ASR 选项：与既有实现保持一致（兼容新版 faster-whisper 的必填字段）。
_WHISPERX_ASR_OPTIONS: dict[str, Any] = {
    # 兼容新版本 faster-whisper 的 TranscriptionOptions 必填字段。
    "max_new_tokens": None,
    "clip_timestamps": "0",
    "hallucination_silence_threshold": None,
    "hotwords": None,
}


def _format_timestamp(seconds: float) -> str:
    """秒转 SRT 时间戳格式。"""

    total_millis = int(max(seconds, 0.0) * 1000)
    hours = total_millis // 3_600_000
    minutes = (total_millis % 3_600_000) // 60_000
    secs = (total_millis % 60_000) // 1000
    millis = total_millis % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _resolve_runtime(device_arg: str, compute_arg: str) -> tuple[str, str]:
    """根据 auto 配置解析运行设备与精度。"""

    device = device_arg
    if device_arg == "auto":
        try:
            import torch  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    compute = compute_arg
    if compute_arg == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def _normalize_io_paths(
    *,
    input_video: Path | str | None,
    output_ja_srt: Path | str | None,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> tuple[Path, Path]:
    """把 `input_video/output_ja_srt` 与 io 契约别名 `input_path/output_path` 归一化。

    runtime 按 B.3 的 `io` 段传 `input_path/output_path`；CLI 脚本仍用
    `input_video/output_ja_srt`。别名存在时优先，缺失时回退历史参数名；
    两者都缺失视为调用错误（`stt_whisperx_io_paths_missing`）。
    """

    resolved_in = input_path if input_path is not None else input_video
    resolved_out = output_path if output_path is not None else output_ja_srt
    if resolved_in is None or resolved_out is None:
        raise ValueError("stt_whisperx_io_paths_missing")
    return Path(resolved_in), Path(resolved_out)


def _ensure_audio_artifact(input_path: Path) -> None:
    """校验 ASR 输入音频产物存在且非空；不满足时抛 MissingAudioArtifactError。

    SPEC E（D1）：ASR 只吃 extract 产出的 16k mono wav；缺失/0 字节直接失败，
    不静默回退视频，错误码 `missing_audio_artifact`。
    """

    try:
        size = input_path.stat().st_size
    except OSError as exc:
        raise MissingAudioArtifactError(input_path) from exc
    if size <= 0:
        raise MissingAudioArtifactError(input_path)


def _check_cancel(cancel_check: CancelCheck | None) -> None:
    """协作取消检查点：回调返回 True（或回调自身抛异常）时抛出取消异常。"""

    if cancel_check is None:
        return
    try:
        cancelled = bool(cancel_check())
    except ExecutorCancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        # 取消回调依赖上游状态查询（如租约续期），查询失败按“已取消”保守处理。
        raise ExecutorCancelledError(f"cancel_check_failed: {exc}") from exc
    if cancelled:
        raise ExecutorCancelledError("cancelled_by_upstream")


def _render_whisperx_srt(segments: list[dict[str, Any]]) -> tuple[str, int, float]:
    """把 WhisperX 段列表渲染成 SRT 文本，返回 (文本, 条数, 最后结束秒)。

    先渲染到内存，再一次性原子落盘：任何异常/取消都不会留下半成品字幕。
    """

    lines: list[str] = []
    count = 0
    last_end = 0.0
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start_sec = float(seg.get("start", 0.0))
        end_sec = float(seg.get("end", start_sec))
        count += 1
        last_end = end_sec
        lines.append(f"{count}\n")
        lines.append(
            f"{_format_timestamp(start_sec)} --> {_format_timestamp(end_sec)}\n"
        )
        lines.append(f"{text}\n\n")
    return "".join(lines), count, last_end


def _resolve_align_model_path(align_model_root: str, language: str) -> Path:
    """按语言解析 alignment 模型目录，优先 `<root>/<language>`。"""

    root = Path(align_model_root)
    language_dir = root / language
    if language_dir.exists():
        return language_dir
    return root


def _resolve_vad_model_file(vad_config_path: Path) -> Path:
    """解析本地 VAD config，并提取 segmentation 的 pytorch_model.bin 路径。"""

    import yaml

    raw = yaml.safe_load(vad_config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"invalid_vad_config_yaml: {vad_config_path}")

    pipeline = raw.get("pipeline")
    if not isinstance(pipeline, dict):
        raise RuntimeError(f"invalid_vad_config_pipeline: {vad_config_path}")
    params = pipeline.get("params")
    if not isinstance(params, dict):
        raise RuntimeError(f"invalid_vad_config_params: {vad_config_path}")

    segmentation = params.get("segmentation")
    if not isinstance(segmentation, str) or not segmentation.strip():
        raise RuntimeError(f"missing_vad_segmentation_path: {vad_config_path}")

    seg_path = Path(segmentation.strip())
    if not seg_path.is_absolute():
        seg_path = (vad_config_path.parent / seg_path).resolve()

    if seg_path.is_dir():
        model_fp = seg_path / "pytorch_model.bin"
    else:
        model_fp = seg_path

    if not model_fp.is_file():
        raise FileNotFoundError(f"vad_segmentation_model_not_found: {model_fp}")
    return model_fp


def build_stt_whisperx_effective_config(
    *,
    model: str,
    language: str,
    timeout_sec: int,
    max_retries: int,
    device: str,
    compute_type: str,
    batch_size: int,
    vad_config_path: str,
    align_model_root: str,
    align_enabled: bool,
    vad_backend: str,
    vad_onset: float,
    vad_offset: float,
    local_files_only: bool,
) -> dict[str, Any]:
    """构造 stt_whisperx 生效参数快照，便于日志审计与问题排查。"""

    resolved_device, resolved_compute = _resolve_runtime(device, compute_type)
    return {
        "timeout_sec": int(timeout_sec),
        "max_retries": int(max_retries),
        "model": model,
        "language": language,
        "device": device,
        "compute_type": compute_type,
        "resolved_device": resolved_device,
        "resolved_compute_type": resolved_compute,
        "batch_size": max(int(batch_size), 1),
        "vad_config_path": vad_config_path,
        "align_model_root": align_model_root,
        "align_enabled": bool(align_enabled),
        "vad_backend": vad_backend,
        "vad_onset": min(max(float(vad_onset), 0.01), 0.99),
        "vad_offset": min(max(float(vad_offset), 0.01), 0.99),
        "local_files_only": bool(local_files_only),
    }


def run_stt_whisperx(
    input_video: Path | None = None,
    output_ja_srt: Path | None = None,
    language: str = "ja",
    timeout_sec: int = 7200,
    *,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    model: str,
    device: str,
    compute_type: str,
    batch_size: int,
    vad_config_path: str,
    align_model_root: str,
    align_enabled: bool,
    vad_backend: str,
    vad_onset: float,
    vad_offset: float,
    local_files_only: bool,
    progress_every: int = 25,
    progress_queue: Queue[dict[str, Any]] | None = None,
    task_id: str | None = None,
    worker_id: str | None = None,
    require_audio_artifact: bool = False,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    """在 service 内部执行 WhisperX 路径（VAD 切段 + batched ASR + 可选对齐）。

    兼容性：`input_video`/`output_ja_srt` 参数名沿用历史，同时接受 B.3 的 io 契约别名
    `input_path`/`output_path`（runtime 按 io 段调用；两者同时给出时以别名为准）。
    `require_audio_artifact=True` 时强制校验输入存在且非空，否则抛
    `MissingAudioArtifactError`（`missing_audio_artifact`），绝不回退视频。
    `cancel_check` 为协作取消回调（失租/停机），返回 True 时中止且不发布产物。
    """

    input_video, output_ja_srt = _normalize_io_paths(
        input_video=input_video,
        output_ja_srt=output_ja_srt,
        input_path=input_path,
        output_path=output_path,
    )
    # 【兼容注释】旧行为会先 preclean_output(output_ja_srt) 删除正式字幕；现在改为
    # “先渲染 + 临时文件 + 成功后 os.replace”（C.7），取消/异常时旧 ja.srt 保持原样。
    output_ja_srt.parent.mkdir(parents=True, exist_ok=True)

    if require_audio_artifact:
        _ensure_audio_artifact(Path(input_video))

    effective_config = build_stt_whisperx_effective_config(
        model=model,
        language=language,
        timeout_sec=timeout_sec,
        max_retries=0,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        vad_config_path=vad_config_path,
        align_model_root=align_model_root,
        align_enabled=align_enabled,
        vad_backend=vad_backend,
        vad_onset=vad_onset,
        vad_offset=vad_offset,
        local_files_only=local_files_only,
    )

    model_path = Path(str(effective_config["model"]))
    if not model_path.exists():
        raise FileNotFoundError(f"whisper_model_not_found: {model_path}")

    vad_cfg = Path(str(effective_config["vad_config_path"]))
    if not vad_cfg.is_file():
        raise FileNotFoundError(f"vad_config_not_found: {vad_cfg}")
    vad_model_fp = _resolve_vad_model_file(vad_cfg)

    align_model_path = _resolve_align_model_path(
        str(effective_config["align_model_root"]),
        str(effective_config["language"]),
    )
    if bool(effective_config["align_enabled"]) and not align_model_path.exists():
        raise FileNotFoundError(f"align_model_not_found: {align_model_path}")

    if not bool(effective_config["local_files_only"]):
        raise RuntimeError("stt_whisperx_requires_local_files_only")

    vad_method = str(effective_config["vad_backend"]).strip().lower()
    vad_options = {
        "vad_onset": float(effective_config["vad_onset"]),
        "vad_offset": float(effective_config["vad_offset"]),
    }
    # SPEC D：ASR + VAD + align 三件套按 key 缓存，同 key 只冷启动一次。
    # 传入未解析的 device/compute_type（auto 也可），缓存内部按 auto 语义解析，
    # 保证 key 与执行器实际使用的运行设备一致。
    lease = get_model_cache().get_whisperx_bundle(
        str(model_path),
        device=str(effective_config["device"]),
        compute_type=str(effective_config["compute_type"]),
        language=str(effective_config["language"]),
        vad_config_path=str(vad_cfg),
        vad_backend=vad_method,
        vad_onset=vad_options["vad_onset"],
        vad_offset=vad_options["vad_offset"],
        vad_model_fp=str(vad_model_fp),
        align_model_path=str(align_model_path),
        align_enabled=bool(effective_config["align_enabled"]),
        asr_options=dict(_WHISPERX_ASR_OPTIONS),
        vad_options=vad_options,
    )
    effective_config["model_cache_hit"] = bool(lease.cached)
    effective_config["model_cache_key"] = lease.key

    media_duration = _probe_duration(input_video)
    started = time.perf_counter()
    _emit_progress(
        progress_queue,
        stage="stt_whisperx",
        percent=0.0,
        message="stt_whisperx_started",
        task_id=task_id,
        worker_id=worker_id,
    )

    try:
        asr, _vad_model, align_model, align_meta = lease.resources
        whisperx = importlib.import_module("whisperx")
        _check_cancel(cancel_check)

        # 同卡并发受 model_parallelism 信号量约束：ASR 与 align 共用一次许可。
        # 同卡排队期间响应取消（drain/失租），绝不超时放行。
        lease.acquire_transcribe_slot(cancel_check=cancel_check)
        try:
            audio = whisperx.load_audio(str(input_video))
            result = asr.transcribe(
                audio,
                batch_size=int(effective_config["batch_size"]),
                language=str(effective_config["language"]),
            )

            segments = result.get("segments", [])
            if bool(effective_config["align_enabled"]):
                try:
                    aligned = whisperx.align(
                        segments,
                        align_model,
                        align_meta,
                        audio,
                        str(effective_config["resolved_device"]),
                        return_char_alignments=False,
                    )
                except AttributeError as exc:
                    # transformers 新版本下 processor.sampling_rate 可能不存在。
                    # 这里不再走 preprocess=False 绕过路径，避免静默改变对齐行为。
                    if "sampling_rate" not in str(exc):
                        raise
                    raise RuntimeError(
                        "whisperx_alignment_api_mismatch: please use official whisperx release and compatible transformers version"
                    ) from exc
                segments = aligned.get("segments", [])
        finally:
            lease.release_transcribe_slot()

        # 长任务的时间/取消检查放在重计算之后、发布之前，至少一次。
        _check_cancel(cancel_check)
        if time.perf_counter() - started > timeout_sec:
            raise TimeoutError("stt_whisperx_timeout")

        rendered, count, last_end = _render_whisperx_srt(list(segments))
        if count == 0:
            raise RuntimeError("stt_whisperx_empty_srt")

        # C.7：字幕先写临时文件再 os.replace，保证取消/异常时不留半成品。
        handle = None
        temp_path: Path | None = None
        try:
            output_ja_srt.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="",
                dir=str(output_ja_srt.parent),
                prefix=f".whisperx-{os.getpid()}-",
                suffix=f".{output_ja_srt.name}.tmp",
                delete=False,
            )
            temp_path = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            handle = None
            _check_cancel(cancel_check)
            os.replace(temp_path, output_ja_srt)
            temp_path = None
        finally:
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning(
                        "whisperx temp cleanup failed: path=%s error=%s",
                        temp_path,
                        exc,
                    )

        if media_duration and media_duration > 0:
            _emit_progress(
                progress_queue,
                stage="stt_whisperx",
                percent=min(last_end / media_duration, 1.0) * 100.0,
                message="stt_whisperx_running",
                task_id=task_id,
                worker_id=worker_id,
            )
    finally:
        lease.release()

    _emit_progress(
        progress_queue,
        stage="stt_whisperx",
        percent=100.0,
        message="stt_whisperx_done",
        task_id=task_id,
        worker_id=worker_id,
    )
    return effective_config