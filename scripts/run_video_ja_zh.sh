#!/usr/bin/env bash
set -euo pipefail

# scripts/ 位于 <repo>/whisper-stt/scripts，因此仓库根（whisper-stt 检出目录）为其上一级。
SCRIPT_ORIGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_ORIGIN_DIR}/.." && pwd)"
# SPEC：只使用主仓库共享 .venv；worktree 部署时自动解析主仓库（支持 WHISPER_STT_ROOT）。
# shellcheck source=./_venv.sh
source "${SCRIPT_ORIGIN_DIR}/_venv.sh"

INPUT_VIDEO="${INPUT_VIDEO:-}"
if [[ -z "${INPUT_VIDEO}" ]]; then
  echo "INPUT_VIDEO is required" >&2
  exit 1
fi

VIDEO_BASENAME="$(basename "${INPUT_VIDEO}")"
VIDEO_STEM="${VIDEO_BASENAME%.*}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output}"
OUTPUT_JA="${OUTPUT_JA:-${OUTPUT_DIR}/${VIDEO_STEM}.ja.srt}"
OUTPUT_ZH="${OUTPUT_ZH:-${OUTPUT_DIR}/${VIDEO_STEM}.zh.srt}"
# 配置位于主仓库根目录（worktree 下也应读主仓库配置）。
CONFIG_PATH="${CONFIG_PATH:-${MAIN_REPO_ROOT}/config.ini}"
# D2：翻译并发上限默认 2（与 [translation] parallel 一致），可用 PARALLEL 覆盖。
PARALLEL="${PARALLEL:-2}"
# SPEC E（D1）：ASR 只吃 wav；PREEXTRACT_WAV / TEMP_AUDIO 透传给 ASR 脚本。
PREEXTRACT_WAV="${PREEXTRACT_WAV:-true}"
TEMP_AUDIO="${TEMP_AUDIO:-}"

mkdir -p "${OUTPUT_DIR}"

INPUT_VIDEO="${INPUT_VIDEO}" \
OUTPUT_SRT="${OUTPUT_JA}" \
PREEXTRACT_WAV="${PREEXTRACT_WAV}" \
TEMP_AUDIO="${TEMP_AUDIO}" \
bash "${REPO_ROOT}/scripts/run_video_ja_srt.sh"

"${VENV_PYTHON}" "${REPO_ROOT}/whisper_stt/translate_srt_ja_to_zh.py" \
  --input "${OUTPUT_JA}" \
  --output "${OUTPUT_ZH}" \
  --config "${CONFIG_PATH}" \
  --parallel "${PARALLEL}"
