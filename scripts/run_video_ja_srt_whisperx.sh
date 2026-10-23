#!/usr/bin/env bash
set -euo pipefail

# scripts/ 位于 <repo>/whisper-stt/scripts，因此仓库根（whisper-stt 检出目录）为其上一级。
SCRIPT_ORIGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_ORIGIN_DIR}/.." && pwd)"
# SPEC：只使用主仓库共享 .venv；worktree 部署时自动解析主仓库（支持 WHISPER_STT_ROOT）。
# shellcheck source=./_venv.sh
source "${SCRIPT_ORIGIN_DIR}/_venv.sh"

INPUT_VIDEO="${INPUT_VIDEO:-/srv/media/jellyfin/JELLYFIN/JAV/atmp/MIDA-574-U/489155.com@MIDA-574-U.mp4}"
OUTPUT_SRT="${OUTPUT_SRT:-${REPO_ROOT}/output/489155.com@MIDA-574-U.whisperx.ja.srt}"
MODEL_REF="${MODEL_REF:-/home/base/repo/video-stt-whisper-server/models/faster-whisper-large-v3-turbo}"
DEVICE="${DEVICE:-auto}"
COMPUTE_TYPE="${COMPUTE_TYPE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-16}"
VAD_CONFIG_PATH="${VAD_CONFIG_PATH:-/home/base/repo/video-stt-whisper-server/models/whisperx/vad/pyannote/config.yaml}"
ALIGN_MODEL_ROOT="${ALIGN_MODEL_ROOT:-/home/base/repo/video-stt-whisper-server/models/whisperx/align}"
ALIGN_ENABLED="${ALIGN_ENABLED:-true}"
# SPEC E（D1）：ASR 只吃 16k mono wav，因此默认务必预抽 wav。
# PREEXTRACT_WAV=false 时脚本不会把视频喂给模型：必须同时用 TEMP_AUDIO
# 指定一个已存在的 wav（否则 transcribe_video_whisperx.py 报 missing_audio_artifact）。
PREEXTRACT_WAV="${PREEXTRACT_WAV:-true}"
TEMP_AUDIO="${TEMP_AUDIO:-}"

ALIGN_FLAG="--align-enabled"
if [[ "${ALIGN_ENABLED}" != "true" ]]; then
  ALIGN_FLAG="--no-align-enabled"
fi

PREEXTRACT_FLAG="--preextract-wav"
if [[ "${PREEXTRACT_WAV}" != "true" ]]; then
  PREEXTRACT_FLAG="--no-preextract-wav"
fi

TEMP_AUDIO_ARGS=()
if [[ -n "${TEMP_AUDIO}" ]]; then
  TEMP_AUDIO_ARGS=(--temp-audio "${TEMP_AUDIO}")
fi

"${VENV_PYTHON}" "${REPO_ROOT}/whisper_stt/transcribe_video_whisperx.py" \
  --input "${INPUT_VIDEO}" \
  --output "${OUTPUT_SRT}" \
  --model "${MODEL_REF}" \
  --language "ja" \
  --device "${DEVICE}" \
  --compute-type "${COMPUTE_TYPE}" \
  --batch-size "${BATCH_SIZE}" \
  --vad-config-path "${VAD_CONFIG_PATH}" \
  --align-model-root "${ALIGN_MODEL_ROOT}" \
  "${ALIGN_FLAG}" \
  "${PREEXTRACT_FLAG}" \
  ${TEMP_AUDIO_ARGS[@]+"${TEMP_AUDIO_ARGS[@]}"} \
  --local-files-only
