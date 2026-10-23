"""extract 阶段执行器。"""

from __future__ import annotations

import time
from pathlib import Path
from queue import Queue
from typing import Any, Callable
import logging
import os
import subprocess

from whisper_stt_service.executor.common import (
    ExecutorCancelledError,
    _emit_progress,
    _probe_duration,
)


logger = logging.getLogger(__name__)

# 协作取消回调类型：返回 True 表示“本任务已被回收/停机”，执行器必须尽快中止。
CancelCheck = Callable[[], bool]


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
        raise ExecutorCancelledError("cancelled_by_extract")


def _temp_wav_path(output_wav: Path) -> Path:
    """生成与目标同目录的临时 wav 路径，保证后续 os.replace 原子可见。

    临时文件必须以 `.wav` 结尾：ffmpeg 依据输出文件**扩展名**推断容器格式，早期实现
    用 `.part` 后缀会让 ffmpeg 因“无法识别容器”直接失败（真实媒体冒烟暴露的 bug）。
    """

    return output_wav.with_name(
        f".extract-{os.getpid()}-{time.time_ns()}.{output_wav.name}"
    )


def run_extract(
    input_video: Path,
    output_wav: Path,
    timeout_sec: int,
    *,
    progress_queue: Queue[dict[str, Any]] | None = None,
    task_id: str | None = None,
    worker_id: str | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    """调用 ffmpeg 抽取 16k 单声道 wav，并把进度写入队列。

    返回 `{"effective_config": ...}` 快照（含输入/输出路径与契约参数），runtime 会把
    它写入 task_finished 日志，便于核对“ASR 究竟吃了哪个 wav”。

    SPEC E：ASR 只消费本阶段产出的 `audio/<stem>.16k.wav`。ffmpeg 先写同目录临时文件，
    成功后才 `os.replace` 成正式产物，避免 ASR 读到半截 wav；取消/失败/超时时删除临时
    文件并保留旧产物。`cancel_check` 为协作取消回调（失租/停机），每条 ffmpeg 进度行
    检查一次。
    """

    # 启动前清理，避免断点产物影响重复执行。
    # 【兼容注释】旧行为在这里 preclean_output(output_wav) 会先删掉正式产物；现在改为
    # “只写临时文件 + 成功后 os.replace”，因此**不再删除正式产物**：任务失败/取消时
    # 旧 wav 保持原样，ASR 永远不会读到半截文件。
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    temp_wav = _temp_wav_path(output_wav)

    duration = _probe_duration(input_video)
    # 契约参数固定 16k/mono/pcm_s16le（D1）：显式记录，避免日志里只能靠猜。
    effective_config: dict[str, Any] = {
        "input_video": str(input_video),
        "output_wav": str(output_wav),
        "sample_rate": 16000,
        "channels": 1,
        "codec": "pcm_s16le",
        "container": "wav",
        "timeout_sec": int(timeout_sec),
        "source_duration_sec": duration,
    }
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        # 显式声明容器，避免临时文件名/扩展名变化导致 ffmpeg 推断失败。
        "-f",
        "wav",
        "-progress",
        "pipe:1",
        "-nostats",
        str(temp_wav),
    ]
    _emit_progress(
        progress_queue,
        stage="extract",
        percent=0.0,
        message="extract_started",
        task_id=task_id,
        worker_id=worker_id,
    )

    def _discard_temp() -> None:
        """删除未完成的临时 wav；失败只告警，不影响异常向上传播。"""

        try:
            temp_wav.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("extract temp cleanup failed: path=%s error=%s", temp_wav, exc)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        # ffmpeg 可能持续写 stderr；若无人消费 PIPE 会导致子进程阻塞。
        # 合并到 stdout，避免在长视频抽取阶段卡死在 99% 附近。
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    started = time.perf_counter()
    assert proc.stdout is not None
    try:
        for raw in proc.stdout:
            # 每条进度行都是一个协作取消检查点（SPEC A.1）。
            _check_cancel(cancel_check)
            line = raw.strip()
            if not line.startswith("out_time_ms="):
                continue
            if duration is None or duration <= 0:
                continue
            try:
                out_sec = float(line.split("=", 1)[1].strip()) / 1_000_000.0
            except ValueError:
                continue
            _emit_progress(
                progress_queue,
                stage="extract",
                percent=min(out_sec / duration, 1.0) * 100.0,
                message="extract_running",
                task_id=task_id,
                worker_id=worker_id,
            )
            if time.perf_counter() - started > timeout_sec:
                proc.kill()
                raise TimeoutError("extract_timeout")

        ret = proc.wait(timeout=max(timeout_sec, 1))
        if ret != 0:
            raise RuntimeError("ffmpeg extraction failed")

        # 发布前再确认一次取消，避免失租的 worker 把 wav 变成正式产物。
        _check_cancel(cancel_check)
        try:
            if temp_wav.stat().st_size <= 0:
                raise RuntimeError("extract_empty_wav")
        except OSError as exc:
            raise RuntimeError(f"extract_wav_missing: {temp_wav}") from exc
        os.replace(temp_wav, output_wav)
    except BaseException:
        # 取消/超时/失败：终止 ffmpeg 并清理临时文件，旧产物保持原样。
        try:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
        except Exception as exc:  # noqa: BLE001
            logger.warning("extract ffmpeg kill failed: error=%s", exc)
        _discard_temp()
        raise

    _emit_progress(
        progress_queue,
        stage="extract",
        percent=100.0,
        message="extract_done",
        task_id=task_id,
        worker_id=worker_id,
    )
    return {"effective_config": effective_config}
