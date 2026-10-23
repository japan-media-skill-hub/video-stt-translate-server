#!/usr/bin/env bash
set -euo pipefail

# scripts/ 位于 <repo>/whisper-stt/scripts，因此仓库根（whisper-stt 检出目录）为其上一级。
SCRIPT_ORIGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_ORIGIN_DIR}/.." && pwd)"
# SPEC：只使用主仓库共享 .venv；worktree 部署时自动解析主仓库（支持 WHISPER_STT_ROOT）。
# shellcheck source=./_venv.sh
source "${SCRIPT_ORIGIN_DIR}/_venv.sh"

INPUT_VIDEO="${INPUT_VIDEO:-/srv/media/jellyfin/JELLYFIN/JAV/atmp/115_downloads/PFES-115/489155.com@PFES-115.mp4}"
OUTPUT_SRT="${OUTPUT_SRT:-${REPO_ROOT}/whisper-stt/output/489155.com@PFES-115.ja.srt}"
MODEL_REF="${MODEL_REF:-models/faster-whisper-small}"
DEVICE="${DEVICE:-auto}"
COMPUTE_TYPE="${COMPUTE_TYPE:-auto}"
# SPEC E（D1）：ASR 只吃 16k mono wav，因此默认务必预抽 wav。
# PREEXTRACT_WAV=false 时脚本不会把视频喂给模型：必须同时用 TEMP_AUDIO
# 指定一个已存在的 wav（否则 transcribe_video.py 报 missing_audio_artifact）。
PREEXTRACT_WAV="${PREEXTRACT_WAV:-true}"
KEEP_TEMP_AUDIO="${KEEP_TEMP_AUDIO:-false}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
TEMP_AUDIO="${TEMP_AUDIO:-}"

PREEXTRACT_FLAG="--preextract-wav"
if [[ "${PREEXTRACT_WAV}" != "true" ]]; then
  PREEXTRACT_FLAG="--no-preextract-wav"
fi

TEMP_AUDIO_ARGS=()
if [[ -n "${TEMP_AUDIO}" ]]; then
  TEMP_AUDIO_ARGS=(--temp-audio "${TEMP_AUDIO}")
fi

KEEP_TEMP_FLAG="--no-keep-temp-audio"
if [[ "${KEEP_TEMP_AUDIO}" == "true" ]]; then
  KEEP_TEMP_FLAG="--keep-temp-audio"
fi

"${VENV_PYTHON}" "${REPO_ROOT}/whisper_stt/transcribe_video.py" \
  --input "${INPUT_VIDEO}" \
  --output "${OUTPUT_SRT}" \
  --model "${MODEL_REF}" \
  --language "ja" \
  --device "${DEVICE}" \
  --compute-type "${COMPUTE_TYPE}" \
  "${PREEXTRACT_FLAG}" \
  ${TEMP_AUDIO_ARGS[@]+"${TEMP_AUDIO_ARGS[@]}"} \
  "${KEEP_TEMP_FLAG}" \
  --progress-every "${PROGRESS_EVERY}"
