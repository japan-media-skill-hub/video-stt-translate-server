#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 脚本可从任意 cwd 启动（worktree / 主仓库 / CI）：显式把仓库根与脚本目录注入
# sys.path，既保证 whisper_stt_service 可导入，也保证能复用同目录的 wav 契约实现。
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _path in (str(_REPO_ROOT), str(_SCRIPT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from whisper_stt_service.executor.stt_whisperx import run_stt_whisperx  # noqa: E402

# 与 transcribe_video.py 复用同一份 wav 契约实现（脚本模式两条路径行为一致）。
from transcribe_video import (  # noqa: E402
    _extract_audio_with_progress,
    _probe_duration,
    ensure_wav_artifact,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a full video to Japanese SRT using WhisperX"
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Input video/audio file"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output .srt file path"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Local whisper model path"
    )
    parser.add_argument("--language", type=str, default="ja", help="Language code")
    parser.add_argument("--device", type=str, default="auto", help="Inference device")
    parser.add_argument(
        "--compute-type", type=str, default="auto", help="CTranslate2 compute type"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="WhisperX batch size"
    )
    parser.add_argument(
        "--vad-config-path",
        type=Path,
        required=True,
        help="Local pyannote VAD config.yaml path",
    )
    parser.add_argument(
        "--align-model-root",
        type=Path,
        required=True,
        help="Local alignment model root (contains ja/zh directories)",
    )
    parser.add_argument(
        "--align-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable alignment",
    )
    parser.add_argument(
        "--vad-backend",
        type=str,
        default="pyannote",
        help="VAD backend name",
    )
    parser.add_argument("--vad-onset", type=float, default=0.5, help="VAD onset")
    parser.add_argument("--vad-offset", type=float, default=0.363, help="VAD offset")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disallow any runtime model downloads",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=7200,
        help="Execution timeout in seconds",
    )
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    input_path = args.input if args.input.is_absolute() else repo_root / args.input
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    vad_cfg = (
        args.vad_config_path
        if args.vad_config_path.is_absolute()
        else repo_root / args.vad_config_path
    )
    align_root = (
        args.align_model_root
        if args.align_model_root.is_absolute()
        else repo_root / args.align_model_root
    )
    model_path = Path(args.model)
    model_ref = str(
        model_path if model_path.is_absolute() else (repo_root / model_path)
    )

    if not input_path.is_file():
        raise FileNotFoundError(f"input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # SPEC E（D1）：ASR 只吃 16k mono wav。脚本模式在这里强制预抽或校验 wav，
    # 绝不把原视频直接交给 run_stt_whisperx。
    temp_audio: Path | None = None
    if args.preextract_wav:
        temp_audio = args.temp_audio
        if temp_audio is None:
            temp_audio = output_path.with_suffix(".tmp.16k.wav")
        if not temp_audio.is_absolute():
            temp_audio = repo_root / temp_audio
        _extract_audio_with_progress(input_path, temp_audio, _probe_duration(input_path))
        audio_input = temp_audio
    else:
        # 关闭预抽时仍然只喂 wav：--temp-audio 指向已存在的 wav，否则用 --input。
        audio_input = args.temp_audio or input_path
        if not audio_input.is_absolute():
            audio_input = repo_root / audio_input
        # 契约优先：视频输入必须在调用 ASR/外部工具（ffprobe 等）之前就报
        # missing_audio_artifact，与 transcribe_video.py 保持同一顺序。
        ensure_wav_artifact(audio_input)

    # 预抽分支在此统一校验新生成的 wav；关闭预抽分支已提前校验（幂等、开销极小）。
    ensure_wav_artifact(audio_input)

    started = time.perf_counter()
    try:
        effective = run_stt_whisperx(
            input_video=audio_input,
            output_ja_srt=output_path,
            language=args.language,
            timeout_sec=max(args.timeout_sec, 1),
            model=model_ref,
            device=args.device,
            compute_type=args.compute_type,
            batch_size=max(args.batch_size, 1),
            vad_config_path=str(vad_cfg),
            align_model_root=str(align_root),
            align_enabled=bool(args.align_enabled),
            vad_backend=args.vad_backend,
            vad_onset=args.vad_onset,
            vad_offset=args.vad_offset,
            local_files_only=bool(args.local_files_only),
        )
    finally:
        # 预抽出来的临时 wav 默认用完即删；--temp-audio 由调用方提供的路径不动。
        if (
            temp_audio is not None
            and args.temp_audio is None
            and temp_audio.exists()
        ):
            temp_audio.unlink()
    elapsed = time.perf_counter() - started

    print(f"input={input_path}", flush=True)
    print(f"audio_input={audio_input}", flush=True)
    print(f"output={output_path}", flush=True)
    print(f"elapsed_sec={elapsed:.2f}", flush=True)
    print(f"effective_config={json.dumps(effective, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
