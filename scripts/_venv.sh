#!/usr/bin/env bash
# 共享 venv 解析：遵守 SPEC "只允许主仓库 .venv，子仓库/worktree 不得创建独立 venv"。
#
# 使用方式（在每个脚本顶部 source）：
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_venv.sh"
# 之后可用：
#   VENV_PYTHON  —— 主仓库 venv 的 python 解释器绝对路径
#   MAIN_REPO_ROOT —— 主仓库根目录
#
# 解析顺序（任一命中即用）：
#   1. 环境变量 WHISPER_STT_ROOT（显式指定主仓库根目录，最高优先级）
#   2. git 公共目录推导：worktree 下 --git-common-dir 指向主仓库的 .git/...，
#      因此不受 "worktree 内不存在 .venv" 影响
#   3. 从工作树根目录逐级向上查找 .venv/bin/python（跳过 .worktrees 层级之外的误判）
# 全部失败时报错退出，绝不回退到 worktree 内的 .venv（SPEC 禁止）。

_venv_found_root=""

_venv_python_is_usable() {
  local candidate="${1:-}"
  [[ -n "${candidate}" && -x "${candidate}" ]]
}

# 1) 显式环境变量
if [[ -n "${WHISPER_STT_ROOT:-}" ]]; then
  if _venv_python_is_usable "${WHISPER_STT_ROOT}/.venv/bin/python"; then
    _venv_found_root="${WHISPER_STT_ROOT}"
  fi
fi

# 2) git 公共目录：worktree 场景下指向主仓库 .git/modules/<name> 或 .git
if [[ -z "${_venv_found_root}" ]]; then
  _venv_common_dir="$(git -C "${SCRIPT_ORIGIN_DIR:-.}" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
  if [[ -n "${_venv_common_dir}" ]]; then
    _venv_candidate_root=""
    # 主仓库直接使用 .git 时：common-dir = <main>/.git
    if [[ "${_venv_common_dir}" == *"/.git/modules/"* ]]; then
      _venv_candidate_root="${_venv_common_dir%%/.git/modules/*}"
    elif [[ "${_venv_common_dir}" == *"/.git" ]]; then
      _venv_candidate_root="${_venv_common_dir%/.git}"
    fi
    if _venv_python_is_usable "${_venv_candidate_root}/.venv/bin/python"; then
      _venv_found_root="${_venv_candidate_root}"
    fi
  fi
fi

# 3) 从脚本所在目录向上逐级查找 .venv/bin/python
if [[ -z "${_venv_found_root}" ]]; then
  _venv_probe="${SCRIPT_ORIGIN_DIR:-.}"
  while [[ "${_venv_probe}" != "/" && -n "${_venv_probe}" ]]; do
    if _venv_python_is_usable "${_venv_probe}/.venv/bin/python"; then
      _venv_found_root="${_venv_probe}"
      break
    fi
    _venv_probe="$(dirname "${_venv_probe}")"
  done
fi

if [[ -z "${_venv_found_root}" ]]; then
  {
    echo "错误：未能找到主仓库共享 venv（SPEC 要求使用主仓库 .venv，禁止在 worktree 内创建独立 venv）。" >&2
    echo "请设置 WHISPER_STT_ROOT=/path/to/video-stt-whisper-server 后重试。" >&2
  } >&2
  exit 1
fi

MAIN_REPO_ROOT="${_venv_found_root}"
VENV_PYTHON="${_venv_found_root}/.venv/bin/python"
unset _venv_found_root _venv_common_dir _venv_candidate_root _venv_probe
