#!/usr/bin/env bash
# Secret 扫描（SPEC H）：阻止真实 provider token 与密钥文件入库。
#
# 用法：
#   bash secret_scan.sh            # 扫描仓库中所有"纳入版本跟踪"的文件（CI 用法）
#   bash secret_scan.sh FILE...    # 只扫描给定文件（自测用）
#
# 检测模式（均要求词边界，避免把合法标识符误判为密钥）：
#   sk-[A-Za-z0-9]{20,}      OpenAI 风格
#   hf_[A-Za-z0-9]{20,}      Hugging Face
#   ghp_[A-Za-z0-9]{20,}     GitHub PAT
#   AIza[A-Za-z0-9_-]{30,}   Google API key
#
# 为什么需要词边界：任务 ID / 文件名里会出现 task-<长字母数字> 这样的串，
# 其中 "ta" + "sk-…" 片段会被无边界正则误判为 OpenAI key（实测命中
# docs/superpowers/specs/2026-04-05-readable-job-task-id-design.md 的 task 示例）。
# 现在要求前缀前必须是行首或非 [A-Za-z0-9_] 的字符；真实密钥通常出现在
# 引号、等号、冒号、空白之后，因此仍然会被检出（见 secret_scan_selftest.sh）。
#
# 输出只包含"文件:行号"，不回显命中内容，避免把密钥写进 CI 日志。
set -euo pipefail

# 路径本身属于密钥文件的，一律不得入库。
forbidden_paths=$(git ls-files | grep -Ei '(^|/)(hf-token\.md|\.env|.*\.token)$' || true)
if [ -n "$forbidden_paths" ]; then
  echo "密钥文件不得入库:"
  echo "$forbidden_paths"
  exit 1
fi

PATTERN='(^|[^A-Za-z0-9_])(sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{30,})'

if [ "$#" -gt 0 ]; then
  matches=$(grep -nEI -- "$PATTERN" "$@" || true)
else
  matches=$(git grep -nEI -- "$PATTERN" -- . || true)
fi

if [ -n "$matches" ]; then
  echo "检测到疑似真实密钥（仅显示位置）:"
  echo "$matches" | cut -d: -f1,2
  exit 1
fi

echo "secret scan passed"
