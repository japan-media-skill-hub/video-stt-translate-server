#!/usr/bin/env bash
# Secret 扫描器自测：覆盖"真实形态必须命中"与"合法 task ID 不得误报"两类样本。
#
# 样本通过 printf 拼接构造，仓库里不存在密钥字面量，因此本文件本身不会让 secret_scan 失败。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCANNER="${SCRIPT_DIR}/secret_scan.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail=0

# expect_hit <说明> <内容>：真实密钥形态，必须命中
expect_hit() {
  local label="$1" content="$2"
  printf '%s\n' "$content" > "$WORK/case.txt"
  if bash "$SCANNER" "$WORK/case.txt" >/dev/null 2>&1; then
    echo "FAIL (expected hit): $label"
    fail=1
  else
    echo "ok  (hit):          $label"
  fi
}

# expect_clean <说明> <内容>：合法内容，不得命中
expect_clean() {
  local label="$1" content="$2"
  printf '%s\n' "$content" > "$WORK/case.txt"
  if bash "$SCANNER" "$WORK/case.txt" >/dev/null 2>&1; then
    echo "ok  (no hit):       $label"
  else
    echo "FAIL (unexpected hit): $label"
    fail=1
  fi
}

# --- 正例：真实 key 形态（含常见上下文：引号 / 等号 / 冒号 / 行首） ---
long_a=$(printf 'A%.0s' {1..40})
long_b=$(printf 'b%.0s' {1..32})
long_c=$(printf 'C%.0s' {1..34})
long_d=$(printf 'd%.0s' {1..36})
long_e=$(printf 'E%.0s' {1..35})
long_f=$(printf 'f%.0s' {1..30})
expect_hit "openai key in quotes"     "api_key = \"sk-${long_a}\""
expect_hit "openai key at line start" "sk-${long_b}"
expect_hit "hf token after colon"     "HF_TOKEN: hf_${long_c}"
expect_hit "github pat"               "token=\"ghp_${long_d}\""
expect_hit "google api key"           "key: AIza${long_e}"
expect_hit "json style openai key"    "{\"api_key\":\"sk-${long_f}\"}"

# --- 负例：合法 task ID / 文档示例（"ta" + "sk-" 片段不得误报） ---
task_long=$(printf 'a%.0s' {1..28})
job_long=$(printf 'b%.0s' {1..24})
task_long2=$(printf 'c%.0s' {1..40})
expect_clean "long task id doc"   "task-${task_long} job-${job_long}"
expect_clean "task id in title"   "task-${task_long2}-suffix"
expect_clean "placeholder token"  "api_token = <CHANGE_ME_STRONG_TOKEN>"
expect_clean "ordinary words"     "risk-management disks-rolled task-sketch glossary"

if [ "$fail" -ne 0 ]; then
  echo "secret scanner self-test failed"
  exit 1
fi
echo "secret scanner self-test passed"
