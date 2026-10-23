#!/usr/bin/env bash
# 按 [security] bind_host / bind_port 启动服务（SPEC H）。
#
# - 监听地址来自配置文件，缺省回落到安全的 127.0.0.1:18000；
# - 使用主项目 venv 的 Python（子仓库/worktree 不得自建 venv）；
# - 通过 uvicorn.Server + main.bind_server 让 drain 完成后触发 should_exit；
# - CLI 可覆盖：--host / --port / --config / --reload / --print-config / --help。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 向上查找主项目 .venv（兼容子仓库与 .worktrees/<name> 工作树），
# 遵守 AGENTS：子仓库/worktree 不得自建 venv，只能共享主项目 venv。
find_main_venv_python() {
  local dir="${REPO_ROOT}"
  while [[ "${dir}" != "/" ]]; do
    if [[ -x "${dir}/.venv/bin/python" ]]; then
      echo "${dir}/.venv/bin/python"
      return 0
    fi
    dir="$(cd "${dir}/.." && pwd)"
  done
  return 1
}

PYTHON_BIN="${PYTHON_BIN:-$(find_main_venv_python || true)}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python not found: ${PYTHON_BIN}" >&2
  echo "必须使用主仓库 .venv: /home/base/repo/video-stt-whisper-server/.venv/bin/python" >&2
  exit 1
fi

# 只设置“缺省”配置来源：命令行传了 --config 时**不要**预导出，否则会覆盖用户
# 选择的文件（bootstrap 只认 WHISPER_STT_CONFIG）。--config 的最终注入由 launcher
# 在 import main/build_app 之前完成，保证监听/鉴权/DB/roots 同源。
has_config_arg="0"
for arg in "$@"; do
  if [[ "${arg}" == "--config" || "${arg}" == --config=* ]]; then
    has_config_arg="1"
    break
  fi
done
if [[ "${has_config_arg}" == "0" ]]; then
  export WHISPER_STT_CONFIG="${WHISPER_STT_CONFIG:-${REPO_ROOT}/config.ini}"
fi

# 切换到仓库根目录，保证相对配置路径（db_path/log_root）与历史行为一致。
# 注意：--config 的相对路径在切换目录**之后**解析，launcher 会再绝对化一次。
cd "${REPO_ROOT}"

exec "${PYTHON_BIN}" -m whisper_stt_service.service.launcher "$@"
