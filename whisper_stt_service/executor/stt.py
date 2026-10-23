"""stt 阶段执行器。"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Iterator
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
from whisper_stt_service.service.model_cache import (
    get_model_cache,
    make_cache_key,
    normalize_vad_params,
)


logger = logging.getLogger(__name__)

# 协作取消回调类型：返回 True 表示“本任务已被回收/停机”，执行器必须尽快中止。
CancelCheck = Callable[[], bool]

# 同卡排队告警阈值：等待模型信号量超过该值时记录 WARNING，便于观察 model_parallelism。
MODEL_SIGNAL_WARN_WAIT_MS = 5000.0


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


def build_stt_effective_config(
    *,
    model: str,
    language: str,
    timeout_sec: int,
    max_retries: int,
    device: str,
    compute_type: str,
    batch_size: int,
    beam_size: int,
    best_of: int,
    patience: float,
    condition_on_previous_text: bool,
    vad_filter: bool,
    vad_threshold: float,
    vad_min_speech_duration_ms: int,
    vad_max_speech_duration_s: float,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    no_speech_threshold: float,
    compression_ratio_threshold: float,
    log_prob_threshold: float,
    hallucination_silence_threshold: float,
    initial_prompt: str,
    hotwords: str,
) -> dict[str, Any]:
    """构造 STT 实际生效参数快照，便于日志审计与问题排查。"""

    resolved_device, resolved_compute = _resolve_runtime(device, compute_type)
    normalized_batch_size = max(batch_size, 1)
    use_batched_pipeline = (
        resolved_device == "cuda"
        and normalized_batch_size > 1
        and not condition_on_previous_text
    )
    return {
        "timeout_sec": int(timeout_sec),
        "max_retries": int(max_retries),
        "model": model,
        "language": language,
        "device": device,
        "compute_type": compute_type,
        "resolved_device": resolved_device,
        "resolved_compute_type": resolved_compute,
        "batch_size": normalized_batch_size,
        "use_batched_pipeline": use_batched_pipeline,
        "beam_size": max(beam_size, 1),
        "best_of": max(best_of, 1),
        "patience": max(patience, 0.1),
        "condition_on_previous_text": condition_on_previous_text,
        "vad_filter": vad_filter,
        "vad_threshold": min(max(vad_threshold, 0.01), 0.99),
        "vad_min_speech_duration_ms": max(vad_min_speech_duration_ms, 50),
        "vad_max_speech_duration_s": max(vad_max_speech_duration_s, 1.0),
        "vad_min_silence_duration_ms": max(vad_min_silence_duration_ms, 50),
        "vad_speech_pad_ms": max(vad_speech_pad_ms, 0),
        "no_speech_threshold": min(max(no_speech_threshold, 0.01), 0.99),
        "compression_ratio_threshold": max(compression_ratio_threshold, 0.1),
        "log_prob_threshold": log_prob_threshold,
        "hallucination_silence_threshold": max(hallucination_silence_threshold, 0.0),
        "initial_prompt": initial_prompt.strip(),
        "hotwords": hotwords.strip(),
    }


def _normalize_io_paths(
    *,
    input_video: Path | str | None,
    output_ja_srt: Path | str | None,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> tuple[Path, Path]:
    """把 `input_video/output_ja_srt` 与 io 契约别名 `input_path/output_path` 归一化。

    runtime 按 B.3 的 `io` 段传 `input_path/output_path`；脚本模式与既有调用仍用
    `input_video/output_ja_srt`。别名存在时优先，缺失时回退历史参数名；
    两者都缺失视为调用错误（`stt_io_paths_missing`）。
    """

    resolved_in = input_path if input_path is not None else input_video
    resolved_out = output_path if output_path is not None else output_ja_srt
    if resolved_in is None or resolved_out is None:
        raise ValueError("stt_io_paths_missing")
    return Path(resolved_in), Path(resolved_out)


def _ensure_audio_artifact(input_path: Path) -> None:
    """校验 ASR 输入音频产物存在且非空；不满足时抛 MissingAudioArtifactError。

    SPEC E（D1）：ASR 只吃 extract 产出的 16k mono wav。这里**绝不回退到视频**：
    缺失或 0 字节直接失败，错误码 `missing_audio_artifact`，让上游恢复/重试路径可见。
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
        # 取消回调依赖上游状态查询（如租约续期），查询失败按“已取消”保守处理，
        # 避免失去租约的 worker 继续写结果。
        raise ExecutorCancelledError(f"cancel_check_failed: {exc}") from exc
    if cancelled:
        raise ExecutorCancelledError("cancelled_by_upstream")


def _segment_check_interval(progress_every: int, hard_limit: int = 50) -> int:
    """计算 STT 循环内取消检查的间隔：取 `progress_every` 与硬性上限的较小值。"""

    return max(min(max(int(progress_every), 1), max(int(hard_limit), 1)), 1)


def _build_stt_vad_params(
    *,
    vad_filter: bool,
    vad_threshold: float,
    vad_min_speech_duration_ms: int,
    vad_max_speech_duration_s: float,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
) -> dict[str, Any]:
    """收集参与缓存 key 的 STT VAD 参数。

    只要 VAD 参数不同，faster-whisper 的转写结果就可能不同；把它们全部纳入
    缓存 key，避免不同 job 覆盖 VAD 后仍复用旧实例。
    """

    return {
        "vad_filter": bool(vad_filter),
        "vad_threshold": float(vad_threshold),
        "vad_min_speech_duration_ms": int(vad_min_speech_duration_ms),
        "vad_max_speech_duration_s": float(vad_max_speech_duration_s),
        "vad_min_silence_duration_ms": int(vad_min_silence_duration_ms),
        "vad_speech_pad_ms": int(vad_speech_pad_ms),
    }


def _open_srt_temp(output_ja_srt: Path) -> tuple[Any, Path]:
    """在输出同目录创建临时字幕文件，保证后续 os.replace 处于同一文件系统。"""

    output_ja_srt.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=str(output_ja_srt.parent),
        prefix=f".stt-{os.getpid()}-",
        suffix=f".{output_ja_srt.name}.tmp",
        delete=False,
    )
    return handle, Path(handle.name)


def _discard_temp(temp_path: Path) -> None:
    """删除临时字幕文件；删除失败只告警，不影响原始异常向上传播。"""

    try:
        temp_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("stt temp cleanup failed: path=%s error=%s", temp_path, exc)


@contextmanager
def _model_slot(lease: Any, *, cancel_check: CancelCheck | None = None) -> Iterator[float]:
    """在模型实例的并发转写许可内执行一段逻辑（model_parallelism）。

    SPEC D 的关键点：faster-whisper 的 `transcribe()` 返回的是**惰性生成器**，真正的
    推理发生在迭代 `segments` 时。因此许可必须覆盖“取到 generator → 迭代完所有段”
    的整段区间，而不是只包住 `transcribe()` 调用本身（否则并发限制形同虚设）。

    取不到许可时排队等待（绝不超时放行），排队期间响应 `cancel_check`；无论正常
    结束、异常还是取消，`finally` 都会归还许可（且只归还本租约实际持有的那一份）。
    """

    # SPEC D：取不到同卡许可就一直排队；drain/失租时通过 cancel_check 中止排队。
    waited_ms = lease.acquire_transcribe_slot(cancel_check=cancel_check)
    if 0.0 < waited_ms and waited_ms >= MODEL_SIGNAL_WARN_WAIT_MS:
        logger.warning(
            "model_transcribe_wait key=%s waited_ms=%.1f（同卡排队）",
            lease.key,
            waited_ms,
        )
    try:
        yield waited_ms
    finally:
        lease.release_transcribe_slot()


def run_stt(
    input_video: Path | None = None,
    output_ja_srt: Path | None = None,
    language: str = "ja",
    timeout_sec: int = 7200,
    *,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    model: str = "models/faster-whisper-small",
    device: str = "auto",
    compute_type: str = "auto",
    batch_size: int = 8,
    beam_size: int = 5,
    best_of: int = 5,
    patience: float = 1.0,
    condition_on_previous_text: bool = False,
    vad_filter: bool = True,
    vad_threshold: float = 0.45,
    vad_min_speech_duration_ms: int = 200,
    vad_max_speech_duration_s: float = 18.0,
    vad_min_silence_duration_ms: int = 700,
    vad_speech_pad_ms: int = 300,
    no_speech_threshold: float = 0.6,
    compression_ratio_threshold: float = 2.2,
    log_prob_threshold: float = -1.0,
    hallucination_silence_threshold: float = 1.5,
    initial_prompt: str = "",
    hotwords: str = "",
    progress_every: int = 25,
    progress_queue: Queue[dict[str, Any]] | None = None,
    task_id: str | None = None,
    worker_id: str | None = None,
    require_audio_artifact: bool = False,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    """直接在 service 内部执行 STT，不再 `python` 调 `python`。

    兼容性：`input_video`/`output_ja_srt` 参数名沿用历史（脚本模式仍可能直接传视频），
    同时接受 B.3 的 io 契约别名 `input_path`/`output_path`（runtime 按 io 段调用）；
    两者同时给出时以 `input_path`/`output_path` 为准。
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

    # 历史注释保留，现为临时文件+原子替换/源hash检查点，绝不删正式旧产物
    # 每次都删除旧字幕，确保结果来自本次运行。
    # 【兼容注释】旧行为在这里每次都 preclean_output(output_ja_srt) 先删掉旧字幕；
    # 现在改为“临时文件 + 成功后 os.replace”（C.7），因此不再删除正式产物：
    # 取消/异常/超时时旧 ja.srt 保持原样，绝不会留下半成品。
    output_ja_srt.parent.mkdir(parents=True, exist_ok=True)

    if require_audio_artifact:
        _ensure_audio_artifact(Path(input_video))

    effective_config = build_stt_effective_config(
        model=model,
        language=language,
        timeout_sec=timeout_sec,
        max_retries=0,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        beam_size=beam_size,
        best_of=best_of,
        patience=patience,
        condition_on_previous_text=condition_on_previous_text,
        vad_filter=vad_filter,
        vad_threshold=vad_threshold,
        vad_min_speech_duration_ms=vad_min_speech_duration_ms,
        vad_max_speech_duration_s=vad_max_speech_duration_s,
        vad_min_silence_duration_ms=vad_min_silence_duration_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
        no_speech_threshold=no_speech_threshold,
        compression_ratio_threshold=compression_ratio_threshold,
        log_prob_threshold=log_prob_threshold,
        hallucination_silence_threshold=hallucination_silence_threshold,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
    )
    resolved_device = str(effective_config["resolved_device"])
    resolved_compute = str(effective_config["resolved_compute_type"])
    # SPEC D：模型实例走进程内缓存（同 key 只冷启动一次），并纳入 VAD/语言维度。
    vad_params = _build_stt_vad_params(
        vad_filter=vad_filter,
        vad_threshold=float(effective_config["vad_threshold"]),
        vad_min_speech_duration_ms=int(effective_config["vad_min_speech_duration_ms"]),
        vad_max_speech_duration_s=float(effective_config["vad_max_speech_duration_s"]),
        vad_min_silence_duration_ms=int(
            effective_config["vad_min_silence_duration_ms"]
        ),
        vad_speech_pad_ms=int(effective_config["vad_speech_pad_ms"]),
    )
    effective_config["model_cache_key"] = make_cache_key(
        "whisper_rt",
        model=str(model),
        device=resolved_device,
        compute=resolved_compute,
        batched=bool(effective_config["use_batched_pipeline"]),
        vad=normalize_vad_params(vad_params),
        language=str(effective_config["language"]),
    )
    model_cache = get_model_cache()
    effective_config["model_cache_hit"] = None

    media_duration = _probe_duration(input_video)
    started = time.perf_counter()
    _check_cancel(cancel_check)
    transcribe_kwargs: dict[str, Any] = {
        "language": str(effective_config["language"]),
        "beam_size": int(effective_config["beam_size"]),
        "best_of": int(effective_config["best_of"]),
        "patience": float(effective_config["patience"]),
        "condition_on_previous_text": condition_on_previous_text,
        "vad_filter": vad_filter,
        "vad_parameters": {
            "threshold": float(effective_config["vad_threshold"]),
            "min_speech_duration_ms": int(
                effective_config["vad_min_speech_duration_ms"]
            ),
            "max_speech_duration_s": float(
                effective_config["vad_max_speech_duration_s"]
            ),
            "min_silence_duration_ms": int(
                effective_config["vad_min_silence_duration_ms"]
            ),
            "speech_pad_ms": int(effective_config["vad_speech_pad_ms"]),
        },
        "no_speech_threshold": float(effective_config["no_speech_threshold"]),
        "compression_ratio_threshold": float(
            effective_config["compression_ratio_threshold"]
        ),
        "log_prob_threshold": float(effective_config["log_prob_threshold"]),
        "hallucination_silence_threshold": float(
            effective_config["hallucination_silence_threshold"]
        ),
    }
    if str(effective_config["initial_prompt"]):
        transcribe_kwargs["initial_prompt"] = str(effective_config["initial_prompt"])
    if str(effective_config["hotwords"]):
        transcribe_kwargs["hotwords"] = str(effective_config["hotwords"])

    # 惰性导入，避免服务仅做队列查询时就加载大模型依赖。
    # 使用模型是惰性生成器：必须保持 lease 直到循环结束才释放。
    lease = model_cache.get_whisper_model(
        model,
        device,
        compute_type,
        vad_params=vad_params,
        language=str(effective_config["language"]),
        use_batched_pipeline=bool(effective_config["use_batched_pipeline"]),
    )
    check_interval = _segment_check_interval(progress_every)
    handle = None
    temp_path: Path | None = None
    try:
        workload = lease.value
        if isinstance(workload, tuple):
            model_runtime, batched_runner = workload
        else:
            model_runtime, batched_runner = workload, None
        effective_config["model_cache_hit"] = bool(lease.cached)

        handle, temp_path = _open_srt_temp(output_ja_srt)
        _emit_progress(
            progress_queue,
            stage="stt",
            percent=0.0,
            message="stt_started",
            task_id=task_id,
            worker_id=worker_id,
        )

        count = 0
        last_end = 0.0
        # 许可覆盖“创建 generator + 迭代完全部段”：faster-whisper 的推理是惰性的，
        # 只在 for 循环里才真正跑模型（见 _model_slot 的说明）。
        with _model_slot(lease, cancel_check=cancel_check) as waited_ms:
            if waited_ms >= MODEL_SIGNAL_WARN_WAIT_MS and waited_ms > 0:
                logger.info(
                    "stt starts after model slot wait: key=%s waited_ms=%.1f",
                    lease.key,
                    waited_ms,
                )
            if batched_runner is not None:
                batched_kwargs = dict(transcribe_kwargs)
                batched_kwargs["batch_size"] = int(effective_config["batch_size"])
                batched_kwargs["without_timestamps"] = False
                batched_kwargs.pop("condition_on_previous_text", None)
                segments, _ = batched_runner.transcribe(
                    str(input_video), **batched_kwargs
                )
            else:
                segments, _ = model_runtime.transcribe(
                    str(input_video), **transcribe_kwargs
                )
            for seg_no, seg in enumerate(segments, start=1):
                text = seg.text.strip()
                if not text:
                    continue
                count += 1
                last_end = float(seg.end)
                handle.write(f"{count}\n")
                handle.write(
                    f"{_format_timestamp(seg.start)} --> {_format_timestamp(seg.end)}\n"
                )
                handle.write(f"{text}\n\n")

                if (
                    seg_no % max(progress_every, 1) == 0
                    and media_duration
                    and media_duration > 0
                ):
                    _emit_progress(
                        progress_queue,
                        stage="stt",
                        percent=min(last_end / media_duration, 1.0) * 100.0,
                        message="stt_running",
                        task_id=task_id,
                        worker_id=worker_id,
                    )

                # 取消检查点：每 min(progress_every, 50) 段一次，失租后尽快停止并丢弃临时文件。
                if seg_no % check_interval == 0:
                    _check_cancel(cancel_check)

                if time.perf_counter() - started > timeout_sec:
                    raise TimeoutError("stt_timeout")

        # 发布前最后确认一次：取消/超时都不得把半成品变成正式产物。
        _check_cancel(cancel_check)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.replace(temp_path, output_ja_srt)
        temp_path = None
    except BaseException:
        # 任何异常（含取消）都不发布产物：关闭并删除临时文件，旧 srt 保持原样。
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        if temp_path is not None:
            _discard_temp(temp_path)
        raise
    finally:
        lease.release()

    _emit_progress(
        progress_queue,
        stage="stt",
        percent=100.0,
        message="stt_done",
        task_id=task_id,
        worker_id=worker_id,
    )
    return effective_config