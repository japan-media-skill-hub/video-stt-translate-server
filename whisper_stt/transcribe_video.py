#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import time
import uuid
from pathlib import Path

from faster_whisper import WhisperModel

# 与 service 端一致：ASR 只吃 16k mono wav，缺失/0 字节即失败。
MISSING_AUDIO_ARTIFACT_CODE = "missing_audio_artifact"


class MissingAudioArtifactError(RuntimeError):
    """ASR 输入 wav 缺失或为 0 字节（错误码 `missing_audio_artifact`）。"""

    code = MISSING_AUDIO_ARTIFACT_CODE

    def __init__(self, path: Path | str) -> None:
        """记录缺失的音频产物路径，便于排障。"""

        self.path = str(path)
        super().__init__(f"{MISSING_AUDIO_ARTIFACT_CODE}: {self.path}")


def atomic_write_text(path: Path, text: str) -> Path:
    """原子写入文本：同目录临时文件 + flush + fsync + os.replace。

    写入失败时删除临时文件并原样抛出，避免留下半截字幕。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".tmp-{os.getpid()}-{uuid.uuid4().hex}.{path.name}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return path


def ensure_wav_artifact(wav_path: Path) -> Path:
    """校验产物是存在且非空的 wav 文件；否则抛 MissingAudioArtifactError。

    脚本模式同样遵守 "ASR 只吃 wav" 契约：即使调用方关闭预抽，
    也只会把输入当作 wav 直接交给模型，绝不把 mp4 等视频交给 faster-whisper。
    除大小检查外还校验 RIFF/WAVE 头，避免"文件存在但其实是视频"漏过契约。
    """

    if not wav_path.is_file():
        raise MissingAudioArtifactError(wav_path)
    try:
        size = wav_path.stat().st_size
    except OSError as exc:
        raise MissingAudioArtifactError(wav_path) from exc
    if size <= 0:
        raise MissingAudioArtifactError(wav_path)
    try:
        with wav_path.open("rb") as handle:
            header = handle.read(12)
    except OSError as exc:
        raise MissingAudioArtifactError(wav_path) from exc
    # 合法 wav：前 4 字节 "RIFF"，第 8-12 字节 "WAVE"。
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise MissingAudioArtifactError(wav_path)
    return wav_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a full video to Japanese SRT using faster-whisper"
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Input video/audio file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/faster-whisper-small",
        help="Local model path or model name",
    )
    parser.add_argument("--language", type=str, default="ja", help="Language code")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="auto",
        help="CTranslate2 compute type",
    )
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size")
    parser.add_argument(
        "--preextract-wav",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Extract input to 16k mono wav before transcription (default: on). "
            "ASR only ever consumes wav: with --no-preextract-wav the --input "
            "must already be a non-empty wav file, otherwise the run fails with "
            "missing_audio_artifact instead of feeding a video to the model."
        ),
    )
    parser.add_argument(
        "--temp-audio",
        type=Path,
        default=None,
        help="Extracted 16k mono wav path (also used as the wav input when pre-extraction is off)",
    )
    parser.add_argument(
        "--keep-temp-audio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep extracted wav file",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N subtitle segments",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output .srt file path"
    )
    return parser.parse_args()


def _format_timestamp(seconds: float) -> str:
    total_millis = int(max(seconds, 0.0) * 1000)
    hours = total_millis // 3_600_000
    minutes = (total_millis % 3_600_000) // 60_000
    secs = (total_millis % 60_000) // 1000
    millis = total_millis % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


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


def _probe_duration(path: Path) -> float | None:
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


def _resolve_runtime(device_arg: str, compute_arg: str) -> tuple[str, str]:
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


def _print_progress(
    prefix: str, progress: float, elapsed: float, eta: float | None
) -> None:
    if eta is None:
        eta_text = "--:--:--"
        finish_text = "--:--:--"
    else:
        eta_text = _format_hms(eta)
        finish_text = (dt.datetime.now() + dt.timedelta(seconds=eta)).strftime(
            "%H:%M:%S"
        )

    bar = _render_bar(progress)
    pct = progress * 100.0
    print(
        f"{prefix} {bar} {pct:6.2f}% elapsed={_format_hms(elapsed)} eta={eta_text} finish={finish_text}",
        flush=True,
    )


def _extract_audio_with_progress(
    input_path: Path, output_wav: Path, source_duration: float | None
) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-progress",
        "pipe:1",
        "-nostats",
        str(output_wav),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    started = time.perf_counter()
    last_emit = 0.0
    extracted_sec = 0.0

    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.strip()
        if line.startswith("out_time_ms="):
            val = line.split("=", 1)[1].strip()
            try:
                extracted_sec = float(val) / 1_000_000.0
            except ValueError:
                continue
            now = time.perf_counter()
            if now - last_emit >= 1.2:
                elapsed = now - started
                if source_duration and source_duration > 0:
                    progress = min(extracted_sec / source_duration, 1.0)
                    if extracted_sec >= 1.0 and elapsed >= 1.0:
                        speed = extracted_sec / elapsed
                        eta = (source_duration - extracted_sec) / max(speed, 1e-6)
                    else:
                        eta = None
                else:
                    progress = 0.0
                    eta = None
                _print_progress("EXTRACT", progress, elapsed, eta)
                last_emit = now

    ret = proc.wait()
    if ret != 0:
        stderr_text = ""
        if proc.stderr is not None:
            stderr_text = proc.stderr.read().strip()
        raise RuntimeError(f"ffmpeg extraction failed: {stderr_text}")

    elapsed = time.perf_counter() - started
    _print_progress("EXTRACT", 1.0, elapsed, 0.0)


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    input_path = args.input if args.input.is_absolute() else repo_root / args.input
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    if not input_path.is_file():
        raise FileNotFoundError(f"input file not found: {input_path}")

    model_ref = args.model
    maybe_model_path = Path(model_ref)
    if not maybe_model_path.is_absolute():
        repo_model = repo_root / maybe_model_path
        if repo_model.exists():
            model_ref = str(repo_model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_duration = _probe_duration(input_path)
    device, compute_type = _resolve_runtime(args.device, args.compute_type)

    temp_audio = None
    if args.preextract_wav:
        temp_audio = args.temp_audio
        if temp_audio is None:
            temp_audio = output_path.with_suffix(".tmp.16k.wav")
        if not temp_audio.is_absolute():
            temp_audio = repo_root / temp_audio
        _extract_audio_with_progress(input_path, temp_audio, source_duration)
        audio_input = temp_audio
    else:
        # 关闭预抽时仍然只喂 wav：--temp-audio 指向已存在的 wav，否则用 --input。
        # 绝不把原视频直接交给 faster-whisper（SPEC E：ASR 只吃 wav）。
        audio_input = args.temp_audio or input_path
        if not audio_input.is_absolute():
            audio_input = repo_root / audio_input

    ensure_wav_artifact(audio_input)

    print(f"input={input_path}", flush=True)
    print(f"audio_input={audio_input}", flush=True)
    print(f"output={output_path}", flush=True)
    print(f"model={model_ref}", flush=True)
    print(f"runtime_device={device}", flush=True)
    print(f"runtime_compute_type={compute_type}", flush=True)

    model = WhisperModel(
        model_ref,
        device=device,
        compute_type=compute_type,
        local_files_only=True,
    )

    transcribe_duration = _probe_duration(audio_input) or source_duration
    transcribe_started = time.perf_counter()
    segments, info = model.transcribe(
        str(audio_input),
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=True,
    )

    count = 0
    last_emit = 0.0
    last_end = 0.0
    # 先累积到内存，循环结束后一次性原子替换输出文件：
    # 中断时不会留下半截 ja.srt，也不会破坏已有产物。
    srt_parts: list[str] = []
    for seg_no, seg in enumerate(segments, start=1):
        text = seg.text.strip()
        if not text:
            continue
        count += 1
        last_end = float(seg.end)
        srt_parts.append(f"{count}\n")
        srt_parts.append(
            f"{_format_timestamp(seg.start)} --> {_format_timestamp(seg.end)}\n"
        )
        srt_parts.append(f"{text}\n\n")

        now = time.perf_counter()
        should_emit = (
            seg_no % max(args.progress_every, 1) == 0 or now - last_emit >= 2.0
        )
        if should_emit:
            elapsed = now - transcribe_started
            if transcribe_duration and transcribe_duration > 0:
                progress = min(last_end / transcribe_duration, 1.0)
                speed = last_end / max(elapsed, 1e-6)
                eta = (transcribe_duration - last_end) / max(speed, 1e-6)
            else:
                progress = 0.0
                eta = None
            _print_progress("TRANSCRIBE", progress, elapsed, eta)
            last_emit = now

    # 循环结束再原子落盘：中途中断不会留下半截字幕。
    atomic_write_text(output_path, "".join(srt_parts))

    elapsed = time.perf_counter() - transcribe_started
    _print_progress("TRANSCRIBE", 1.0, elapsed, 0.0)

    if temp_audio is not None and not args.keep_temp_audio and temp_audio.exists():
        temp_audio.unlink()

    print(f"segments={count}", flush=True)
    print(f"detected_language={info.language}", flush=True)
    print(f"language_probability={info.language_probability:.4f}", flush=True)
    print(
        f"audio_duration_sec={transcribe_duration if transcribe_duration else 0.0:.3f}",
        flush=True,
    )
    print(f"subtitle_last_end_sec={last_end:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
