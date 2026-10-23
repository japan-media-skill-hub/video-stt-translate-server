# video-stt-translate-server v0.5.0

面向批量电影翻译的本地流水线系统：将 `ffmpeg` 媒体处理、`Whisper/WhisperX` 语音转写、LLM 字幕翻译整合为一个可编排 DAG。

[English](./README.md) | [简体中文](./README.zh-CN.md)

> **发布状态（2026-10-24）：v0.5.0 为 release candidate，尚未最终验收、尚未打 tag。**
> 首次远端 CI（[run 39981535390](https://github.com/japan-media-skill-hub/video-stt-translate-server/actions/runs/39981535390)，
> commit `bba96a0`）在提交锁校验处失败：`uv.lock` 仍引用开发机专属 TUNA 镜像。锁已归一为公共 PyPI，
> 等待重跑，目前没有远端绿灯。真实 2 小时 / GPU 验收与供应商侧 token 轮换仍无证据，最终 P1 修复仍在进行。
> 详见[验证状态](#验证状态)。

## 项目定位

本项目的目标是把电影翻译链路中的三个核心能力统一到一个可批处理、可恢复、可观测的执行系统中：

- `extract`：用 `ffmpeg` 抽取/预处理音频（16 kHz 单声道 `pcm_s16le` wav）。
- `stt` 或 `stt_whisperx`：用 Faster-Whisper / WhisperX 完成日语字幕转写。
- `translate`：用大模型把日语字幕翻译为中文字幕。

服务层通过 DAG 组织任务依赖关系，支持批量视频流水线式处理，并通过 `job_id` / `task_id` 与阶段日志实现可追踪运行。

## 功能概览（v0.5.0）

- 支持脚本模式与服务模式，覆盖单视频到批量队列任务。
- **wav 契约**：ASR（`stt` / `stt_whisperx`）只消费 `extract` 产出的 16k 单声道 wav；wav 缺失或为 0 字节时任务失败并报 `missing_audio_artifact`。
- **产物归属唯一**：`artifacts/<job_id>/{audio,asr,translate}/`，每阶段只有一个写者。
- **入队即校验 DAG 语义**：必须恰好一个 ASR 阶段、ASR 依赖闭包必须含 `extract`、`translate` 依赖闭包必须含该 ASR 阶段。
- **任务生命周期闭环（待最终验收）**：租约心跳、过期回收（reaper）、结果写回 fencing、上游失败级联（`blocked`）、崩溃恢复消耗重试预算、有界 drain 停机。
- **重跑与取消接口**：`POST /jobs/{job_id}/retry`（仅 failed）与 `POST /jobs/{job_id}/cancel`（未开始的任务）。
- **翻译链路加固（待最终验收）**：并发上限默认 2（`parallel`）、按批检查点续跑（`*.progress.json` + `source_sha256`）、原子写、缺 id 保留日文原文、4xx 不重试、请求带 `Idempotency-Key`；检查点 fencing 仍在修复。
- **模型缓存**：进程内 LRU 模型缓存 + 每模型并发信号量（`model_cache_size`、`model_parallelism`）。
- **安全默认值（fail-closed）**：占位 token 拒绝启动、token 常量时间比较、视频路径/扩展名白名单、脚本模式 copy_back 的路径/符号链接逃逸校验（服务模式 sink 尚未接通）、docs 默认关闭、默认只监听 `127.0.0.1`。
- **可观测性**：`request_id` 贯穿的结构化日志、`GET /healthz`、`GET /readyz`、`GET /metrics`（Prometheus）、队列最近 5 分钟吞吐。
- WhisperX 可作为独立阶段接入（`stt_whisperx`：本地 VAD + batched ASR + 可选对齐），VAD 默认参数更激进（`vad_onset=0.35`、`vad_offset=0.2`）。
- 脚本模式支持字幕产物回写（`[translation] copy_back`），输出 `.ja.srt` 与 `.zh.srt`；服务模式 copy_back 尚未接到真实 sink（见“验证状态”）。
- 支持可读任务 ID、任务归档、失败重试与中断续跑。
- 支持小时级 artifact 自动清理（按已完成 job 清理目录，降低磁盘占用）。
- 提供端到端验证脚本，支持轮询、超时、退出原因记录（`E2E_EXIT ...`）。

## 目录说明

- `whisper_stt_service/`：REST 服务、队列 worker、DAG 调度。
- `whisper_stt/`：本地脚本（转写、翻译）。
- `tests/unit`：单元测试（默认测试集）。
- `tests/manual`：真实媒体手工用例（需显式开启，见[测试](#测试)）。
- `tests/e2e`：E2E 驱动脚本。

## 运行前要求

- Python `>=3.10`
- 系统可用 `ffmpeg` / `ffprobe`
- 本地可访问 Faster-Whisper 模型目录（`runtime.model_path` 或 `WHISPER_STT_MODEL`）
- LLM 翻译配置可用（`llm.base_url` / `llm.api_key` / `llm.model`）

支持两种硬件运行方式：

- **纯 CPU 模式**：可运行整条流程，适合功能验证和低并发任务；目前仅短片验证，未做真实 2 小时验收。
- **CUDA 加速模式（NVIDIA GPU）**：在 Whisper/WhisperX 阶段显著提速，适合大批量生产任务；当前机器 NVML 被 OS 阻断，
  显存峰值尚未验证。

WhisperX 长时任务建议：`workers.stt_whisperx_workers <= 2`，可显著降低 CUDA OOM 风险。

推荐模型版本：

- `faster-whisper-large-v2`
- `faster-whisper-large-v3`
- `faster-whisper-large-v3-turbo`

## Python 环境（严格）

所有 Python 执行必须使用主仓库共享环境：

```text
/home/base/repo/video-stt-whisper-server/.venv
```

- 子仓库（`whisper-stt/`）与 `.worktrees/` 下的任何工作树都**不得**创建自己的 `.venv`。
- 请始终使用绝对路径调用解释器，例如
  `/home/base/repo/video-stt-whisper-server/.venv/bin/python` 或
  `/home/base/repo/video-stt-whisper-server/.venv/bin/uvicorn`。
- 若使用 `uv`，先固定项目环境，避免 uv 在工作树里自动建 venv：
  `export UV_PROJECT_ENVIRONMENT=/home/base/repo/video-stt-whisper-server/.venv`。
  新增依赖请用
  `uv pip install --python /home/base/repo/video-stt-whisper-server/.venv/bin/python <包名>`，
  **不要**在工作树里直接 `uv sync`（会替换显卡依赖栈）。

## 安装依赖

`dev` 分组以 CPU 为目标，CI 也只安装它；但"不装 `gpu` 组"**不等于**"没有 GPU 依赖"：
`whisperx` 传递依赖 `pyannote-audio -> torch`，而本仓库在 `[tool.uv.sources]` 里把
`torch`/`torchaudio` 钉在 `pytorch-cu128` 索引上。因此 CI 会把 torch 源临时改为 PyTorch
**CPU** 索引（见 [CI](#ci)），以 **CPU-only** 方式执行、不安装 CUDA 运行时。

```bash
uv sync --group dev          # dev 分组（本机 / CI）
```

如需 GPU 依赖（仅限本机）：

```bash
uv sync --group dev --group gpu
```

`uv.lock` 已纳入版本跟踪，覆盖完整解析图（含 `gpu` 组）。普通包走显式公共 PyPI 默认源（`https://pypi.org/simple`，wheel 走 `files.pythonhosted.org`）；`torch` / `torchaudio` 仍由显式 `pytorch-cu128` 索引提供。

## 配置策略

1. `config.example.ini` 作为模板保留在仓库。
2. `config.ini` 作为本地运行配置（默认不提交）。
3. 服务启动时若 `config.ini` 不存在，会自动由 `config.example.ini` 生成。
4. 若缺少必填项，日志会输出 `section.option` 形式的缺失键。
5. 可在 `[security] api_token` 中配置全局 API 访问令牌（请求头 `X-API-Token`）。
6. 可通过 `[runtime] artifact_cleanup_*` 配置开启小时级 artifact 自动清理。

### 安全默认值（fail-closed）

| 配置键 | 默认 | 语义 |
|---|---|---|
| `[security] api_token` | 空 | 空 = 鉴权关闭（启动告警） |
| `[security] require_token` | `true` | true 且 token 为空 = **拒绝启动** |
| `[security] allow_placeholder_token` | `false` | 占位 token（`<...>`、示例值）拒绝启动 |
| `[security] docs_enabled` | `false` | 关闭时 `/docs`、`/redoc`、`/openapi.json` 返回 404 |
| `[security] allowed_video_roots` | 空 | 逗号分隔；视频路径 resolve 后必须位于其下 |
| `[security] allowed_copy_back_roots` | 空 | 空 = 仅允许视频所在目录 |
| `[security] bind_host` / `bind_port` | `127.0.0.1` / `18000` | 启动脚本据此监听 |

> **示例配置故意无法直接启动**：请先设置真实 token（建议 ≥16 字符；代码只拒绝空/占位 token，短 token 仅告警），
> 或显式 `require_token = false` 自行承担风险。
> **公网部署必须前置反向代理 + TLS**，不要直接把服务绑到 `0.0.0.0`。

### 关键配置（v0.5.0）

```ini
[translation]
parallel = 2                 # 并发上限；旧配置里的 16 会被读成 16，请改成 2
batch_size = 200             # 每请求条目上限
max_chars = 12000            # 每请求字符上限
chunk_minutes = 20           # 每个请求组的时间窗
request_interval = 1.0       # 批次派发最小间隔，0 = 关闭
retry = 10
timeout = 240
disable_proxy = true
copy_back = __video_dir__
checkpoint_enabled = true    # 按批检查点续跑
strict_ids = false           # true 时缺 id 即整批失败

[timeouts]
lease_timeout_sec = 600
drain_timeout_sec = 300      # 有界优雅停机

[runtime]
recovery_counts_as_attempt = true   # 启动恢复消耗重试预算
model_cache_size = 2                # 进程内 LRU 模型缓存（0 = 关闭，仅调试）
model_parallelism = 1               # 每个模型实例的并发转写上限

[observability]
log_level = INFO
log_jsonl = false
metrics_enabled = true
```

配置优先级：`job_config.translate` > `[translation]` > 代码默认值。

上表即 `config.example.ini` 的默认值，且与代码内默认值一致：
`parallel=2`、`batch_size=200`、`max_chars=12000`、`chunk_minutes=20`、`timeout=240`、`retry=10`。
本机 `config.ini` 可能覆盖这些值，请以实际生效配置为准，不要假设一定是默认值。

## 三种运行模式

### 1) Python 脚本模式（适合单视频/调参）

```bash
bash scripts/run_video_ja_srt.sh
bash scripts/run_video_ja_zh.sh
```

或直接调用（始终走主仓库 `.venv`）：

```bash
/home/base/repo/video-stt-whisper-server/.venv/bin/python whisper_stt/transcribe_video.py --help
/home/base/repo/video-stt-whisper-server/.venv/bin/python whisper_stt/transcribe_video_whisperx.py --help
/home/base/repo/video-stt-whisper-server/.venv/bin/python whisper_stt/translate_srt_ja_to_zh.py --help
```

### 2) 服务模式（适合批量任务）

推荐用启动器（自动读取 `[security] bind_host` / `bind_port`，并使用主仓库 `.venv`）：

```bash
cd /home/base/repo/video-stt-whisper-server/.worktrees/reliability-hardening
bash scripts/run_service.sh                 # 按 [security] bind_host / bind_port 监听
bash scripts/run_service.sh --config /abs/path/config.ini --print-config
```

等价的直接调用（请保持 `--host 127.0.0.1`，不要把服务直接绑到 `0.0.0.0`）：

```bash
WHISPER_STT_CONFIG=/abs/path/config.ini \
  /home/base/repo/video-stt-whisper-server/.venv/bin/uvicorn \
  whisper_stt_service.main:app --host 127.0.0.1 --port 18000
```

鉴权开启时，所有 API 请求（含 `/healthz`、`/readyz`、`/metrics`）都需要携带：

```bash
-H "X-API-Token: <your_api_token>"
```

优雅停机（停止领取新任务，等待在途任务完成后自动退出；受 `drain_timeout_sec` 约束）：

```bash
curl -X POST "http://127.0.0.1:18000/admin/shutdown" \
  -H "Content-Type: application/json" \
  -H "X-API-Token: <your_api_token>" \
  -d '{"reason":"manual_shutdown"}'
```

重跑 failed job / 取消排队 job：

```bash
curl -X POST "http://127.0.0.1:18000/jobs/<job_id>/retry" -H "X-API-Token: <token>"
curl -X POST "http://127.0.0.1:18000/jobs/<job_id>/cancel" -H "X-API-Token: <token>"
```

### 3) E2E 模式（真实链路验证）

E2E 驱动会自动拉起服务、提交任务、轮询状态并持续记录监控日志，适合验证完整电影翻译流水线。

推荐命令（后台持续运行，含日志落盘）：

```bash
nohup /home/base/repo/video-stt-whisper-server/.venv/bin/python tests/e2e/run_e2e_explicit_dag_flow.py \
  --run-mode until_done \
  --video-paths tests/e2e/video_paths.txt \
  --poll-sec 15 \
  --deadline-sec 43200 \
  --monitor-log tmp/e2e/explicit_dag_monitor.log \
  --server-log tmp/e2e/explicit_dag_server.log \
  > tmp/e2e/explicit_dag_nohup.log 2>&1 < /dev/null &
echo $! > tmp/e2e/explicit_dag.pid
```

监控方式：

```bash
tail -f tmp/e2e/explicit_dag_monitor.log
tail -f tmp/e2e/explicit_dag_nohup.log
```

分轮停机验证：

```bash
bash /home/base/repo/video-stt-whisper-server/scripts/run_shutdown_round.sh round1
bash /home/base/repo/video-stt-whisper-server/scripts/run_shutdown_round.sh round2
```

典型监控日志片段（节选）：

```text
=== E2E Round 1 @ 2026-04-06T04:46:12.725612+00:00 ===
jobs_done=0/4; job_status=(queued=2, running=2)
queue=extract:q=0,c=0,s=19,f=0 | stt_whisperx:q=2,c=2,s=15,f=0 | translate:q=4,c=0,s=15,f=0
task_logs_root=/.../tmp/logs
```

## 测试

默认测试（跳过真实媒体与 E2E；不依赖 ffmpeg、模型、GPU 与网络）：

```bash
cd /home/base/repo/video-stt-whisper-server/.worktrees/reliability-hardening
/home/base/repo/video-stt-whisper-server/.venv/bin/python -m pytest -q
```

pytest 配置里 `addopts = -m "not manual and not e2e"`，因此 `tests/manual` 与 `tests/e2e` 默认不会执行。

真实媒体手工用例（需要本地模型权重与可用的 LLM 接口）：

```bash
WHISPER_STT_REAL_MEDIA=1 \
  /home/base/repo/video-stt-whisper-server/.venv/bin/python -m pytest \
  tests/manual/test_real_media.py -m manual -v
```

- 开关为环境变量 `WHISPER_STT_REAL_MEDIA`，模块同时标记为 `manual`。
- 素材位置可用 `WHISPER_STT_REAL_OUTPUT_DIR`、`WHISPER_STT_REAL_VIDEO`、`WHISPER_STT_MODEL`、`WHISPER_STT_CONFIG` 覆盖。
- ASR 用例会先跑 `extract`，再对产出的 `*.16k.wav` 转写（v0.5.0 wav 契约）。

## CI

`.github/workflows/ci.yml` 在每次 push / pull request 上运行。workflow 已推送到 origin，但**尚无远端
CI 绿灯**：首次运行（[run 39981535390](https://github.com/japan-media-skill-hub/video-stt-translate-server/actions/runs/39981535390)，
commit `bba96a0`）在提交锁校验处失败，原因是 `uv.lock` 仍引用开发机专属 TUNA 镜像。锁已归一为公共 PyPI，
等待重跑；本地检查不能替代最终要求的两次远端 CI。

- `secret-scan`：`.github/scripts/secret_scan.sh` 在纳入跟踪的文件含真实 `sk-…` / `hf_…` / `ghp_…` / `AIza…` 令牌，
  或提交了 `hf-token.md` 一类密钥文件时直接失败。令牌必须落在词边界上，因此 `task-<长ID>` 这类标识符不会误报；
  `secret_scan_selftest.sh` 同时验证正例（引号内 / `=`、`:` 后的真实 key）与该负例。
- `pytest (python 3.10, cpu)`：先在干净 runner 环境校验提交锁（`uv lock --check --python 3.10`；
  显式公共 PyPI 默认源、无宿主机 TUNA 配置）→ 执行 `ci-cpu-sources.py`（把 torch/torchaudio 源改为
  PyTorch CPU 索引，仅 CI）→ `uv lock --python 3.10` → `uv sync --group dev`（不装 `gpu` 组）→
  先跑 CPU 子集，再跑完整测试集 → 语法门禁（`python -m compileall`，仅跟踪文件）→ 用
  `config.example.ini` 在临时目录装配应用冒烟。

只设 `UV_TORCH_BACKEND=cpu` **不够**：显式的 `[tool.uv.sources]` 会覆盖 torch backend 设置，
所以 CI 直接把索引改成 `https://download.pytorch.org/whl/cpu` 并去掉 `+cu128` 本地版本约束。
在干净临时副本中已实测：改写后 `uv sync --group dev --dry-run` 解析为 `torch==2.8.0+cpu` / `torchaudio==2.8.0+cpu`，
`nvidia-*` 包数量为 **0**。

> `ruff` 只作为 dev 组的本地工具，**不**作为 CI 门禁：存量代码仍有 pyflakes 告警，修复它们需要与本版本无关的大范围改动。

## Roadmap

### 已完成

- [x] `v0.1.0`：REST API + 后台 worker + 基础流水线调度。
- [x] `v0.2.0`：DAG 规划、任务配置快照、归档接口、字幕回写、STT 参数配置化。
- [x] `v0.3.0`：WhisperX 阶段、显式 DAG E2E、确定性退出日志、中断续跑增强。
- [x] `v0.4.0`：全局 API 鉴权、安全 drain 停机、分轮停机 E2E、小时级 artifact 清理调度。

### Release candidate（v0.5.0 — 尚未验收）

- [ ] `v0.5.0`：可靠性与翻译链路加固——任务生命周期闭环（租约心跳/回收/fencing/级联/有界停机）、入队即校验 DAG 语义、ASR wav 契约、retry/cancel 接口、翻译并发上限 2 + 按批检查点续跑、产物原子写、模型缓存与并发上限、安全 fail-closed 默认值 + 路径白名单、结构化日志 + 健康检查/指标、CI + 锁文件 + 密钥治理
  - 故意不勾选：最终验收尚未完成，远端 CI / v0.5.0 tag / 真实 2 小时 + GPU 证据均未取得。

### 相比 v0.4.0 的增量

- **任务生命周期（目标行为，待最终验收）**：租约回收与 fencing；上游失败后下游置 `blocked`；启动恢复消耗重试预算；新增 `POST /jobs/{id}/retry` 与 `POST /jobs/{id}/cancel`
- **产物级 fencing（暂存 + promotion）**：每个 attempt 只写自己的私有暂存文件
  （`<name>.staging-<attempt>-<id>`），只有经 `ArtifactStaging.promote()` 才 `os.replace` 为正式产物；
  落地前在 `BEGIN IMMEDIATE` 事务内确认任务仍由同一 worker / 同一 attempt 持有
  （`with_task_lease` / `publish_task_artifacts`）。设计目标是旧 attempt 无法覆盖新 attempt 的产物，被拒的暂存会被清理；
  租约身份 / stop 回滚等边界仍在修复（见“验证状态”）。
- **canonical 翻译检查点**：runtime 把稳定的 canonical 检查点路径传给翻译器，而字幕本体走暂存；
  重试时可从未完成批次续跑，检查点与产物一起 promotion 落地。
  已接入 runtime，但仍**待最终验收**：检查点所有权 / 租约身份等 fencing 边界仍在修复。
- **正确性**：`/jobs/by-path` 恢复 200；无 ASR 阶段 / 两个 ASR 阶段 / 缺 `extract` 依赖 / `translate` 无 ASR 依赖的 DAG 在入队时即被拒绝
- **wav 契约**：ASR 读取 `artifacts/<job_id>/audio/<stem>.16k.wav`；wav 缺失报 `missing_audio_artifact`
- **翻译**：`parallel` 默认 2（原 16）、新增 `max_chars`/`checkpoint_enabled`/`strict_ids`；按批检查点续跑（fencing 边界修复中）；4xx 不重试；`Idempotency-Key`；`zh.srt` 原子写
- **资源**：进程内模型缓存 + 信号量（`model_cache_size`、`model_parallelism`）
- **安全**：`require_token`、`allow_placeholder_token`、`docs_enabled`、`allowed_video_roots`、`allowed_copy_back_roots`、`bind_host/bind_port`；token 常量时间比较
- **可观测性**：`/healthz`、`/readyz`、`/metrics`、request-id 日志、队列 5 分钟吞吐
- **交付（未完成）**：新增 CI workflow（首次远端运行在锁校验失败，原因是隐式宿主机 TUNA 镜像；锁已归一公共 PyPI，等待重跑）、`uv.lock` 纳入跟踪、真实媒体用例迁到 `tests/manual`、删除 `docs/hf-token.md`、删除空 `.venv`；源码整合、两次远端 CI 绿灯与 v0.5.0 tag 仍由发布父代理统一执行

### 规划中

- MCP 服务化能力与标准协议接入。
- Agent Skill 集成与稳定调用方案。
- Web 控制台（提交、监控、排障）。
- Docker / Compose 可复现部署。
- 基于现有指标的告警与看板。
- 多用户权限、并发限制与资源配额治理。

## 验证状态（2026-10-24）

以下本地检查均使用主 `.venv`；未在该 venv 执行 `uv sync`，未安装 GPU 包，本次复核未发送付费 LLM 请求。

| 门禁 | 状态 | 证据 / 剩余缺口 |
|---|---|---|
| secret 扫描 + 自测 | 本地通过 | `secret_scan.sh` 与 `secret_scan_selftest.sh` exit 0 |
| child 锁走公共 PyPI | 本地通过 | TUNA 0；201 个 `pypi.org/simple` source；2489 个 `files.pythonhosted.org` URL；`torch/torchaudio` 仍为 `2.8.0+cu128`（4 处） |
| root 锁走公共 PyPI | 本地通过 | TUNA 0；140 个 `pypi.org/simple` source；2199 个 `files.pythonhosted.org` URL；143 packages；check exit 0 |
| 干净环境锁校验 | 本地通过 | 空 XDG、无 `UV_INDEX`/`UV_DEFAULT_INDEX`：child `uv lock --check --python 3.10` 与 root check 均 exit 0 |
| 锁图等价性 | 本地通过 | child 144 / root 143 packages；版本、sdist+wheel 哈希、依赖图未变（仅 source URL/size 元数据归一） |
| CI CPU torch 覆盖 | 本地通过 | 临时副本 lock/dry-run 解析为 `torch/torchaudio==2.8.0+cpu`，待安装 `nvidia-*` 为 0 |
| 启动冒烟（显式 token + 缺失模型） | 本地通过 | 错误 token 401；queue/health 200；`/readyz` 503；未创建 model 目录 |
| 全量 pytest | 本地通过 | 527 passed / 5 deselected，37.73s（2026-10-23 23:29） |
| 远端 CI | **首次失败，待重跑** | run 39981535390 @ `bba96a0` 在 `uv lock --check` 失败（锁含宿主机 TUNA 镜像）；锁已归一公共 PyPI，尚无绿灯 |
| 真实 2 小时 E2E | **未验证** | `/srv/media/...` 4 条路径都不存在；`/srv/media/jellyfin` 为空；只有 30s/180s 样本 |
| GPU 显存峰值 | **未验证** | `nvidia-smi` 退出 255，`NVML GPU access blocked by OS` |
| 供应商侧 token 轮换 | **未完成 / 无证据** | 属于外部操作；本地删除/忽略不等于撤销 |
| v0.5.0 最终验收 | **待完成** | 服务模式 copy_back sink、租约身份/fencing、promotion 异常仍在修复 |

## 已知问题

- ruff 基线（`E9` + `F`）在存量模块上仍有告警；CI 只做语法门禁。
- `pyproject.toml` 为主依赖引入了 `prometheus-client`（`/metrics` 需要）；若运行环境缺少该包，`/metrics` 应优雅降级而不能拖垮应用。
- **v0.5.0 尚未验收**：服务模式 copy_back 没有真实 sink；租约身份 / stop 回滚 / 检查点 fencing /
  promotion 异常等 P1 仍在修复。不能用本地测试数量代替最终验收。
- 真实 2 小时样本端到端验收（各阶段墙钟、请求数、显存峰值、`kill -9` 续跑）**尚未验证**。
  2026-10-23 复核：`tests/e2e/video_paths.txt` 的 4 条 `/srv/media/jellyfin/...` 路径仍全部不存在，
  `/srv/media/jellyfin` 为空；本机只有 30s / 180s 样本；`nvidia-smi` 退出 255 并报
  `NVML GPU access blocked by OS`。
- 曾经落盘的 HF token 与 LLM API key 需要**人工轮换**（HF 统一改用 `HF_TOKEN` 环境变量，不再使用 `docs/hf-token.md`）。
  当前没有 provider 端撤销证据，不能仅凭本地文件清理就标记轮换完成。

## Contributing

欢迎 Issue 与 PR，提交前建议执行：

```bash
/home/base/repo/video-stt-whisper-server/.venv/bin/python -m pytest -q
```

## License

本项目采用 MIT 许可证，详见 `LICENSE`。
