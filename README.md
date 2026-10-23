# video-stt-translate-server v0.5.0

A local movie translation pipeline that combines `ffmpeg`, `Whisper/WhisperX`, and LLM subtitle translation in one DAG-based system.

[English](./README.md) | [简体中文](./README.zh-CN.md)

> **Release status (2026-10-24): v0.5.0 is a release candidate, not yet accepted or tagged.**
> The first remote CI run ([39981535390](https://github.com/japan-media-skill-hub/video-stt-translate-server/actions/runs/39981535390),
> commit `bba96a0`) failed at the committed-lock check because `uv.lock` still referenced a host-only
> TUNA mirror. The lock is now normalized to public PyPI and a rerun is pending; there is no green remote
> run yet. Real 2-hour/GPU acceptance and provider-side token rotation remain unverified, and final P1
> fixes are still in progress. See [Verification status](#verification-status).

## Project purpose

This project unifies three core steps for end-to-end movie translation into one batch-ready pipeline:

- `extract`: audio extraction and preprocessing via `ffmpeg` (16 kHz mono `pcm_s16le` wav)
- `stt` or `stt_whisperx`: Japanese subtitle transcription via Faster-Whisper / WhisperX
- `translate`: LLM-based Japanese-to-Chinese subtitle translation

Instead of isolated scripts, the service runs these stages as a DAG so jobs are traceable, restartable, and scalable for bulk video processing.

## Key capabilities (v0.5.0)

- Script mode and service mode for single-video to large-batch workflows
- Queue workers with default DAG and explicit DAG execution
- **wav contract**: ASR (`stt` / `stt_whisperx`) consumes only the 16k mono wav produced by `extract`; a missing/empty wav fails the task with `missing_audio_artifact`
- **Per-stage artifact ownership**: `artifacts/<job_id>/{audio,asr,translate}/`, one writer per stage
- **DAG semantic validation at enqueue**: exactly one ASR stage, ASR must depend on `extract`, `translate` must depend on the ASR stage
- **Task lifecycle closure (pending final acceptance)**: lease heartbeat, expired-lease reaper, fencing on result write-back, upstream failure cascade (`blocked`), startup recovery that consumes the retry budget, bounded drain shutdown
- **Retry / cancel APIs**: `POST /jobs/{job_id}/retry` (failed jobs) and `POST /jobs/{job_id}/cancel` (not yet started)
- **Translation hardening (pending final acceptance)**: concurrency capped at 2 by default (`parallel`), per-batch checkpoint resume (`*.progress.json` + `source_sha256`), atomic writes, missing-id fallback to the Japanese source, 4xx never retried, `Idempotency-Key` on every request; checkpoint fencing is still under fix
- **Model cache**: in-process LRU model cache + per-model concurrency semaphore (`model_cache_size`, `model_parallelism`)
- **Security defaults (fail-closed)**: placeholder tokens refuse startup, API token compared in constant time, video path / extension allow-lists, copy_back path/symlink-escape validation for the script-mode sink (the service-mode sink is still pending), docs disabled by default, binds to `127.0.0.1` by default
- **Observability**: structured logging with `request_id`, `GET /healthz`, `GET /readyz`, `GET /metrics` (Prometheus), queue throughput over the last 5 minutes
- WhisperX stage support (`stt_whisperx`) for local VAD + batched ASR + optional alignment, with aggressive VAD defaults (`vad_onset=0.35`, `vad_offset=0.2`)
- Script-mode subtitle copy-back (`[translation] copy_back`) to source video directories (`.ja.srt` and `.zh.srt`); service-mode copy-back is not wired to a real sink yet (see Verification status)
- Hourly artifact cleanup scheduler for completed jobs to control disk usage
- E2E drivers with deterministic exit reason logging (`E2E_EXIT ...`)

## Project layout

- `whisper_stt_service/`: REST APIs, queue workers, DAG scheduling
- `whisper_stt/`: standalone script workflows
- `tests/unit`: unit tests (default test run)
- `tests/manual`: real-media manual tests (opt-in, see [Tests](#tests))
- `tests/e2e`: E2E drivers

## Prerequisites

- Python `>=3.10`
- `ffmpeg` and `ffprobe` in `PATH`
- Local Faster-Whisper model path (`runtime.model_path` or `WHISPER_STT_MODEL`)
- LLM config (`llm.base_url`, `llm.api_key`, `llm.model`)

Hardware/runtime options:

- **CPU-only mode**: supported for pipeline execution; validated only on short local samples, not on the real
  2-hour acceptance run
- **CUDA acceleration (NVIDIA GPU)**: recommended for Whisper/WhisperX throughput; GPU memory peaks are not verified
  on the current machine (NVML blocked)

WhisperX recommendation for long-running GPU jobs: `workers.stt_whisperx_workers <= 2` to reduce OOM risk.

Recommended model variants:

- `faster-whisper-large-v2`
- `faster-whisper-large-v3`
- `faster-whisper-large-v3-turbo`

## Python environment (strict)

All Python execution MUST use the shared environment of the main repository:

```text
/home/base/repo/video-stt-whisper-server/.venv
```

- The child repository (`whisper-stt/`) and any worktree under `.worktrees/` MUST NOT create their own `.venv`.
- Always call the interpreter by absolute path, e.g.
  `/home/base/repo/video-stt-whisper-server/.venv/bin/python` or
  `/home/base/repo/video-stt-whisper-server/.venv/bin/uvicorn`.
- If you use `uv`, pin the project environment first so uv never creates a per-worktree venv:
  `export UV_PROJECT_ENVIRONMENT=/home/base/repo/video-stt-whisper-server/.venv`.
  To add a package, use
  `uv pip install --python /home/base/repo/video-stt-whisper-server/.venv/bin/python <pkg>`
  instead of running `uv sync` inside the worktree (a full sync would replace the GPU stack).

## Install with uv

The `dev` group is CPU-oriented and CI installs only it — but "no `gpu` group" is **not** the same as
"no GPU dependency": `whisperx` transitively requires `pyannote-audio -> torch`, and this repo pins
`torch`/`torchaudio` to the `pytorch-cu128` index in `[tool.uv.sources]`. CI therefore rewrites the torch
sources to the PyTorch **CPU** index (see [CI](#ci)) and runs **CPU-only** without CUDA runtimes.

```bash
uv sync --group dev          # dev group (local machines, CI)
```

Optional GPU dependencies (local machines only):

```bash
uv sync --group dev --group gpu
```

`uv.lock` is tracked in this repository and resolves the full graph, including the `gpu` group.
Normal packages resolve from the explicit public PyPI default index (`https://pypi.org/simple`, wheels from
`files.pythonhosted.org`); `torch` / `torchaudio` remain pinned to the explicit `pytorch-cu128` index.

## Configuration

1. Keep `config.example.ini` tracked in repo.
2. Use `config.ini` for local runtime values (git-ignored).
3. If `config.ini` is missing, the service auto-generates it from `config.example.ini`.
4. Missing required keys are logged as `section.option`.
5. Set `[security] api_token` to enforce global API access control via `X-API-Token`.
6. Runtime supports hourly artifact cleanup via `[runtime] artifact_cleanup_* options`.

### Security defaults (fail-closed)

| Key | Default | Meaning |
|---|---|---|
| `[security] api_token` | empty | empty = auth disabled (startup warning) |
| `[security] require_token` | `true` | `true` + empty token = **startup refused** |
| `[security] allow_placeholder_token` | `false` | placeholder tokens (`<...>`, example values) refuse startup |
| `[security] docs_enabled` | `false` | `/docs`, `/redoc`, `/openapi.json` return 404 when disabled |
| `[security] allowed_video_roots` | empty | comma-separated; video path must resolve under one of them |
| `[security] allowed_copy_back_roots` | empty | empty = only the video's own directory |
| `[security] bind_host` / `bind_port` | `127.0.0.1` / `18000` | listened on by the launch scripts |

> **The example config cannot start as-is on purpose.** Set a real token (>= 16 characters is
> recommended; the code rejects empty/placeholder tokens and only warns on short ones) or explicitly
> set `require_token = false` and own the risk.
> **Public deployments must sit behind a reverse proxy with TLS**; never bind to `0.0.0.0` directly.

### Key configuration (v0.5.0)

```ini
[translation]
parallel = 2                 # concurrency cap; the old 16 is still read as 16 -> change it to 2
batch_size = 200             # entries per request
max_chars = 12000            # characters per request
chunk_minutes = 20           # time window per request group
request_interval = 1.0       # minimum dispatch interval between batches, 0 = off
retry = 10
timeout = 240
disable_proxy = true
copy_back = __video_dir__
checkpoint_enabled = true    # per-batch checkpoint resume
strict_ids = false           # true = a missing id fails the whole batch

[timeouts]
lease_timeout_sec = 600
drain_timeout_sec = 300      # bounded graceful shutdown

[runtime]
recovery_counts_as_attempt = true   # startup recovery consumes the retry budget
model_cache_size = 2                # in-process LRU model cache (0 = disabled, debug only)
model_parallelism = 1               # concurrent transcriptions per model instance

[observability]
log_level = INFO
log_jsonl = false
metrics_enabled = true
```

Values shown are the defaults written to `config.example.ini` and match the code defaults:
`parallel=2`, `batch_size=200`, `max_chars=12000`, `chunk_minutes=20`, `timeout=240`, `retry=10`.
Your local `config.ini` may override them; inspect the effective config instead of assuming defaults.

Priority: `job_config.translate` > `[translation]` > code defaults.

## Three run modes

### 1) Python script mode

```bash
bash scripts/run_video_ja_srt.sh
bash scripts/run_video_ja_zh.sh
```

Or run scripts directly (always through the main `.venv`):

```bash
/home/base/repo/video-stt-whisper-server/.venv/bin/python whisper_stt/transcribe_video.py --help
/home/base/repo/video-stt-whisper-server/.venv/bin/python whisper_stt/transcribe_video_whisperx.py --help
/home/base/repo/video-stt-whisper-server/.venv/bin/python whisper_stt/translate_srt_ja_to_zh.py --help
```

### 2) Service mode

Preferred: the launcher reads `[security] bind_host` / `bind_port` and uses the main `.venv`:

```bash
cd /home/base/repo/video-stt-whisper-server/.worktrees/reliability-hardening
bash scripts/run_service.sh                 # honours [security] bind_host / bind_port
bash scripts/run_service.sh --config /abs/path/config.ini --print-config
```

Equivalent direct invocation (keep `--host 127.0.0.1` — never bind `0.0.0.0` without a reverse proxy + TLS):

```bash
WHISPER_STT_CONFIG=/abs/path/config.ini \
  /home/base/repo/video-stt-whisper-server/.venv/bin/uvicorn \
  whisper_stt_service.main:app --host 127.0.0.1 --port 18000
```

All API requests (including `/healthz`, `/readyz` and `/metrics`) must include the token when auth is enabled:

```bash
-H "X-API-Token: <your_api_token>"
```

Graceful drain shutdown (stop claiming new tasks, finish in-flight, then exit; bounded by `drain_timeout_sec`):

```bash
curl -X POST "http://127.0.0.1:18000/admin/shutdown" \
  -H "Content-Type: application/json" \
  -H "X-API-Token: <your_api_token>" \
  -d '{"reason":"manual_shutdown"}'
```

Retry a failed job / cancel a queued one:

```bash
curl -X POST "http://127.0.0.1:18000/jobs/<job_id>/retry" -H "X-API-Token: <token>"
curl -X POST "http://127.0.0.1:18000/jobs/<job_id>/cancel" -H "X-API-Token: <token>"
```

### 3) E2E mode

The E2E runner starts the service, submits jobs through REST APIs, polls status, and writes monitor/server logs for full pipeline verification.

Recommended background command with logging:

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

Monitor progress:

```bash
tail -f tmp/e2e/explicit_dag_monitor.log
tail -f tmp/e2e/explicit_dag_nohup.log
```

Round-based shutdown verification:

```bash
bash /home/base/repo/video-stt-whisper-server/scripts/run_shutdown_round.sh round1
bash /home/base/repo/video-stt-whisper-server/scripts/run_shutdown_round.sh round2
```

Example monitor excerpt:

```text
=== E2E Round 1 @ 2026-04-06T04:46:12.725612+00:00 ===
jobs_done=0/4; job_status=(queued=2, running=2)
queue=extract:q=0,c=0,s=19,f=0 | stt_whisperx:q=2,c=2,s=15,f=0 | translate:q=4,c=0,s=15,f=0
task_logs_root=/.../tmp/logs
```

## Tests

Default run (skips real media and E2E; no ffmpeg, model, GPU or network required):

```bash
cd /home/base/repo/video-stt-whisper-server/.worktrees/reliability-hardening
/home/base/repo/video-stt-whisper-server/.venv/bin/python -m pytest -q
```

The pytest config sets `addopts = -m "not manual and not e2e"`, so `tests/manual` and `tests/e2e`
drivers never run by default.

Real-media manual tests (real local model weights + a live LLM endpoint):

```bash
WHISPER_STT_REAL_MEDIA=1 \
  /home/base/repo/video-stt-whisper-server/.venv/bin/python -m pytest \
  tests/manual/test_real_media.py -m manual -v
```

- The switch is the `WHISPER_STT_REAL_MEDIA` environment variable; the module is also marked `manual`.
- Media location can be overridden with `WHISPER_STT_REAL_OUTPUT_DIR`, `WHISPER_STT_REAL_VIDEO`,
  `WHISPER_STT_MODEL` and `WHISPER_STT_CONFIG`.
- The ASR cases first run `extract` and transcribe the produced `*.16k.wav` (the v0.5.0 wav contract).

## CI

`.github/workflows/ci.yml` runs on every push / pull request. The workflow is on origin, but **no remote
CI run is green yet**: the first run ([39981535390](https://github.com/japan-media-skill-hub/video-stt-translate-server/actions/runs/39981535390),
commit `bba96a0`) failed at the committed-lock check because `uv.lock` still referenced a host-only TUNA
mirror. The lock has been normalized to public PyPI and a rerun is pending; local checks below are not a
substitute for the required two green remote runs.

- `secret-scan`: `.github/scripts/secret_scan.sh` fails if a tracked file contains a real
  `sk-…` / `hf_…` / `ghp_…` / `AIza…` token or a key file such as `hf-token.md` is committed.
  Tokens must start at a word boundary, so identifiers such as `task-<long-id>` are not false positives;
  `secret_scan_selftest.sh` checks both true positives (keys in quotes/after `=` or `:`) and that negative case.
- `pytest (python 3.10, cpu)`: verifies the committed lock (`uv lock --check --python 3.10`) in a clean
  runner environment (explicit public PyPI default index; no host TUNA config) → applies `ci-cpu-sources.py`
  (torch/torchaudio → PyTorch CPU index, CI only) → `uv lock` → `uv sync --group dev` (no `gpu` group) →
  CPU subset of unit tests → full suite → syntax gate (`python -m compileall` on tracked files) → boots the
  app from `config.example.ini` in a temp dir.

`UV_TORCH_BACKEND=cpu` alone is **not** enough: explicit `[tool.uv.sources]` win over the torch backend, so CI
rewrites the index to `https://download.pytorch.org/whl/cpu` and drops the `+cu128` local version. Verified in a
clean temp copy: `uv sync --group dev --dry-run` then resolves `torch==2.8.0+cpu` / `torchaudio==2.8.0+cpu`
with **zero** `nvidia-*` packages.

> `ruff` ships in the `dev` group for local use but is **not** a CI gate: the baseline code base still has
> pyflakes findings that would require unrelated refactors, which is out of scope for this release.

## Roadmap

### Completed

- [x] `v0.1.0`: REST service baseline and pipeline scheduling
- [x] `v0.2.0`: DAG planning, archive API, copy-back, and runtime knobs
- [x] `v0.3.0`: WhisperX stage, explicit DAG E2E, deterministic E2E exits, interrupted-run recovery
- [x] `v0.4.0`: global API token auth, graceful drain shutdown, round-based shutdown E2E, and hourly artifact cleanup scheduler

### Release candidate (v0.5.0 — not yet accepted)

- [ ] `v0.5.0`: reliability & translation hardening — task lifecycle closure (lease heartbeat/reaper/fencing/cascade/bounded drain), enqueue-time DAG semantics, wav contract for ASR, retry/cancel APIs, translation concurrency cap 2 with per-batch checkpoint resume, atomic artifact writes, model cache + parallelism limits, security fail-closed defaults with path allow-lists, structured logging + health/readiness/metrics, CI + lockfile + secret hygiene
  - Unchecked on purpose: final acceptance is pending, and remote CI / v0.5.0 tag / real 2-hour + GPU evidence are not available yet.

### Delta since v0.4.0

- **Task lifecycle (target behavior, pending final acceptance)**: lease reaper and fencing; downstream tasks become `blocked` when an upstream task fails; startup recovery consumes the retry budget; `POST /jobs/{id}/retry` and `POST /jobs/{id}/cancel`
- **Artifact-level fencing (staging + promotion)**: each attempt writes to its private staging file
  (`<name>.staging-<attempt>-<id>`); it becomes the final artifact only through `ArtifactStaging.promote()`,
  which `os.replace`es it after the repo confirms — inside one `BEGIN IMMEDIATE` transaction — that the task is
  still held by the same worker/attempt (`with_task_lease` / `publish_task_artifacts`). The intended guarantee
  is that a stale attempt cannot overwrite a newer attempt's artifact; rejected attempts are cleaned up.
  Lease identity/stop-rollback edge cases are still under fix (see Verification status).
- **Canonical translation checkpoint**: the runtime passes the stable canonical checkpoint path to the
  translator while the SRT itself is staged, so a retried attempt resumes from batches already committed
  instead of re-translating; the checkpoint is only promoted together with the artifact.
  Wired into the runtime, but still **pending final acceptance**: checkpoint ownership / lease-identity
  fencing edge cases remain under fix.
- **Correctness**: `/jobs/by-path` returns 200 again; DAG is rejected at enqueue when it has no ASR stage, two ASR stages, no `extract` dependency or `translate` without an ASR dependency
- **wav contract**: ASR reads `artifacts/<job_id>/audio/<stem>.16k.wav`; missing wav → `missing_audio_artifact`
- **Translation**: `parallel` default 2 (was 16), `max_chars`, `checkpoint_enabled`, `strict_ids`; per-batch checkpoint resume (fencing edge case under fix); 4xx never retried; `Idempotency-Key`; atomic `zh.srt`
- **Resources**: in-process model cache + semaphore (`model_cache_size`, `model_parallelism`)
- **Security**: `require_token`, `allow_placeholder_token`, `docs_enabled`, `allowed_video_roots`, `allowed_copy_back_roots`, `bind_host/bind_port`; constant-time token comparison
- **Observability**: `/healthz`, `/readyz`, `/metrics`, request-id logging, 5-minute queue throughput
- **Delivery (not complete)**: CI workflow added (first remote run failed at the lock check due an implicit host TUNA mirror; lock normalized to public PyPI, rerun pending), tracked `uv.lock`, real-media tests moved to `tests/manual`, `docs/hf-token.md` removed, empty `.venv` removed; source merge, two green remote CI runs and v0.5.0 tag are still owned by the release parent

### Planned

- MCP integration and protocol interoperability
- Agent-skill integration for stable automation
- Web console for submit/monitor/troubleshoot
- Docker/Compose reproducible deployment
- Alerting on top of the existing metrics (log/metric dashboards)
- Multi-user permission and quota control

## Verification status (2026-10-24)

All local checks below used the strict shared `.venv`; no `uv sync` was run in it and no GPU packages were
installed. No paid LLM request was sent during this verification pass.

| Gate | Status | Evidence / remaining gap |
|---|---|---|
| Secret scanner + self-test | passed locally | `secret_scan.sh` and `secret_scan_selftest.sh` exit 0 |
| Child lock on public PyPI | passed locally | TUNA 0; 201 `pypi.org/simple` sources; 2489 `files.pythonhosted.org` URLs; `torch/torchaudio` still `2.8.0+cu128` (4 refs) |
| Root lock on public PyPI | passed locally | TUNA 0; 140 `pypi.org/simple` sources; 2199 `files.pythonhosted.org` URLs; 143 packages; check exit 0 |
| Clean-env lock check | passed locally | empty XDG, no `UV_INDEX`/`UV_DEFAULT_INDEX`: child `uv lock --check --python 3.10` and root check exit 0 |
| Lock graph equivalence | passed locally | child 144 and root 143 packages; versions/sdist+wheel hashes/dependency graph unchanged (only source URLs/size metadata normalized) |
| CI CPU torch override | passed locally | temp-copy lock/dry-run resolves `torch/torchaudio==2.8.0+cpu`, zero `nvidia-*` installs |
| Bootstrap smoke (explicit token + missing model) | passed locally | wrong token 401; queue/health 200; `/readyz` 503; model dir not created |
| Full pytest suite | passed locally | 527 passed, 5 deselected in 37.73s (2026-10-23 23:29) |
| Remote CI | **first run failed; rerun pending** | run 39981535390 @ `bba96a0` failed at `uv lock --check` because the lock used a host-only TUNA mirror; lock normalized to public PyPI, no green run yet |
| Real 2-hour E2E | **not verified** | 4 configured `/srv/media/...` paths missing; `/srv/media/jellyfin` empty; only 30s/180s samples |
| GPU memory peak | **not verified** | `nvidia-smi` exit 255, `NVML GPU access blocked by OS` |
| Provider-side token rotation | **not done / no evidence** | operator action; local deletion/ignore is not revocation |
| Final v0.5.0 acceptance | **pending** | service-mode copy-back sink, lease identity/fencing, promotion-error handling still under fix |

## Known issues

- The ruff baseline (`E9` + `F`) still reports findings on pre-existing modules; CI only enforces a syntax gate.
- `pyproject.toml` keeps `prometheus-client` in the main dependencies for `/metrics`; if the package is missing at runtime, `/metrics` must degrade gracefully instead of breaking the app.
- **v0.5.0 is not accepted yet**: service-mode copy-back has no real sink, and lease identity / stop-rollback /
  checkpoint-fencing / promotion-error P1s are still under fix. Do not cite the local test count as final acceptance.
- End-to-end acceptance on a real 2-hour sample (wall-clock per stage, request counts, GPU memory peaks,
  `kill -9` resume) is **not yet verified**. Re-checked 2026-10-23: the four paths in
  `tests/e2e/video_paths.txt` under `/srv/media/jellyfin/...` are all missing, `/srv/media/jellyfin` is empty,
  and only 30s / 180s local samples exist; `nvidia-smi` exits 255 with `NVML GPU access blocked by OS`.
- The real HF token and the LLM API key that were once stored on disk must be **rotated by the operator**
  (use `HF_TOKEN` env var for Hugging Face instead of `docs/hf-token.md`). No provider-side revocation
  evidence exists, so token rotation must not be marked complete based on local file cleanup.

## Contributing

Issues and pull requests are welcome. Before submitting:

```bash
/home/base/repo/video-stt-whisper-server/.venv/bin/python -m pytest -q
```

## License

Licensed under MIT. See `LICENSE`.
