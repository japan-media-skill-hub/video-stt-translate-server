"""批次1整合：runtime 的 IO 契约、生效配置快照、检查点 TTL 与指标接入测试。"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from whisper_stt_service.config import (
    RetrySettings,
    RuntimeSettings,
    SecuritySettings,
    Settings,
    SttSettings,
    SttWhisperxSettings,
    TimeoutSettings,
    WorkerSettings,
)
from whisper_stt_service.db import Database
from whisper_stt_service.progress import ProgressStore
from whisper_stt_service.repository import JobRepository, TaskExecutionContext
from whisper_stt_service.service import runtime as runtime_module
from whisper_stt_service.service.runtime import WorkerRuntime


def _settings(*, progress_ttl_sec: int = 300) -> Settings:
    """构造最小配置（无 worker 线程、关闭 artifact 清理线程）。"""

    return Settings(
        workers=WorkerSettings(
            extract_workers=0,
            stt_workers=0,
            stt_whisperx_workers=0,
            translate_workers=0,
            scheduler_interval_sec=60,
            poll_interval_sec=1,
        ),
        timeouts=TimeoutSettings(
            extract_timeout_sec=120,
            stt_timeout_sec=3600,
            stt_whisperx_timeout_sec=3600,
            translate_timeout_sec=3600,
            lease_timeout_sec=60,
            drain_timeout_sec=300,
        ),
        retry=RetrySettings(
            extract_max_retries=2,
            stt_max_retries=2,
            stt_whisperx_max_retries=2,
            translate_max_retries=2,
        ),
        runtime=RuntimeSettings(
            db_path=Path("/tmp/test.db"),
            progress_ttl_sec=progress_ttl_sec,
            log_root=Path("/tmp/logs"),
            artifact_root=Path("/tmp/artifacts"),
            artifact_cleanup_enabled=False,
            artifact_cleanup_interval_sec=3600,
            artifact_cleanup_statuses=("succeeded",),
            model_path=Path("/tmp/model"),
        ),
        stt=SttSettings(
            device="auto",
            compute_type="auto",
            batch_size=8,
            beam_size=3,
            best_of=3,
            patience=1.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_threshold=0.45,
            vad_min_speech_duration_ms=200,
            vad_max_speech_duration_s=18.0,
            vad_min_silence_duration_ms=700,
            vad_speech_pad_ms=300,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.2,
            log_prob_threshold=-1.0,
            hallucination_silence_threshold=1.5,
            initial_prompt="",
            hotwords="",
        ),
        stt_whisperx=SttWhisperxSettings(
            model="/tmp/model",
            device="auto",
            compute_type="auto",
            batch_size=8,
            vad_config_path=Path("/tmp/vad/config.yaml"),
            align_model_root=Path("/tmp/align"),
            align_enabled=True,
            vad_backend="pyannote",
            vad_onset=0.35,
            vad_offset=0.2,
            local_files_only=True,
        ),
        security=SecuritySettings(api_token=""),
    )


def _repo(tmp_path: Path, *, artifact_root: Path | None = None) -> JobRepository:
    """构造指向 tmp 的仓储。"""

    db = Database(tmp_path / "q.db")
    db.init_schema()
    return JobRepository(
        db,
        log_root=tmp_path / "logs",
        artifact_root=artifact_root or (tmp_path / "artifacts"),
    )


def _runtime(
    repo: JobRepository,
    tmp_path: Path,
    *,
    metrics=None,
    checkpoint_ttl_sec: int | None = None,
) -> WorkerRuntime:
    """构造不启动线程的运行时。"""

    return WorkerRuntime(
        repo=repo,
        progress_store=ProgressStore(300),
        settings=_settings(),
        config_path=tmp_path / "config.ini",
        model_path="/tmp/model",
        exit_callback=lambda: None,
        heartbeat_interval_sec=0.05,
        metrics=metrics,
        checkpoint_ttl_sec=checkpoint_ttl_sec,
    )


def _ctx(
    *, task_config: dict, output_ja: str, output_zh: str, video: str
) -> TaskExecutionContext:
    """构造执行上下文（不建库）。"""

    return TaskExecutionContext(
        task_id="task-ctx-1",
        job_id="job-ctx-1",
        stage="stt",
        video_path=video,
        source_language="ja",
        output_ja_path=output_ja,
        output_zh_path=output_zh,
        timeout_sec=600,
        attempt=0,
        max_retries=2,
        log_dir="/tmp/logs/job-ctx-1",
        log_file="/tmp/logs/job-ctx-1/task.log",
        task_config=task_config,
    )


def test_io_paths_are_taken_from_task_config(tmp_path: Path) -> None:
    """B.3：extract/ASR/translate 的输入输出优先使用 task_config.io。"""

    runtime = _runtime(_repo(tmp_path), tmp_path)
    ctx = _ctx(
        task_config={
            "stage": "stt",
            "effective_config": {},
            "io": {
                "input_path": "/art/job-x/audio/a.16k.wav",
                "output_path": "/art/job-x/asr/a.ja.srt",
            },
        },
        output_ja="/art/job-x/a.ja.srt",
        output_zh="/art/job-x/a.zh.srt",
        video="/videos/a.mp4",
    )

    assert runtime._resolve_asr_input(ctx) == Path("/art/job-x/audio/a.16k.wav")
    assert runtime._resolve_asr_output(ctx) == Path("/art/job-x/asr/a.ja.srt")
    assert runtime._resolve_translate_input(ctx) == Path("/art/job-x/audio/a.16k.wav")
    assert runtime._resolve_translate_output(ctx) == Path("/art/job-x/asr/a.ja.srt")


def test_extract_output_prefers_io_and_falls_back_to_wav_rule(tmp_path: Path) -> None:
    """B.3：extract 输出读 io.output_path；旧 job 回退到 ja.srt 同目录 wav。"""

    runtime = _runtime(_repo(tmp_path), tmp_path)
    new_ctx = _ctx(
        task_config={"io": {"output_path": "/art/job-x/audio/a.16k.wav"}},
        output_ja="/art/job-x/asr/a.ja.srt",
        output_zh="/art/job-x/translate/a.zh.srt",
        video="/videos/a.mp4",
    )
    assert runtime._resolve_extract_output(new_ctx) == Path("/art/job-x/audio/a.16k.wav")

    legacy_ctx = _ctx(
        task_config={"effective_config": {}},
        output_ja="/art/job-y/a.ja.srt",
        output_zh="/art/job-y/a.zh.srt",
        video="/videos/a.mp4",
    )
    assert runtime._resolve_extract_output(legacy_ctx) == Path("/art/job-y/a.wav")


def test_legacy_asr_input_falls_back_to_wav_not_video(tmp_path: Path) -> None:
    """D1：旧 job 没有 io 段时，ASR 输入仍是 wav 规则，绝不回退原视频。"""

    runtime = _runtime(_repo(tmp_path), tmp_path)
    ctx = _ctx(
        task_config={"effective_config": {}},
        output_ja="/art/job-y/a.ja.srt",
        output_zh="/art/job-z/a.zh.srt",
        video="/videos/a.mp4",
    )

    resolved = runtime._resolve_asr_input(ctx)

    assert resolved == Path("/art/job-y/a.wav")
    assert resolved != Path(ctx.video_path)


def test_task_started_snapshot_uses_effective_config(tmp_path: Path, monkeypatch) -> None:
    """G：task_started 快照取该 task 的 effective_config（job 覆盖生效）。"""

    runtime = _runtime(_repo(tmp_path), tmp_path)
    captured: dict[str, object] = {}

    def _fake_build(**kwargs):
        captured.update(kwargs)
        return {"batch_size": kwargs["batch_size"], "device": kwargs["device"]}

    monkeypatch.setattr(runtime_module, "build_stt_effective_config", _fake_build)
    ctx = _ctx(
        task_config={"effective_config": {"batch_size": 16, "device": "cpu"}},
        output_ja="/art/a.ja.srt",
        output_zh="/art/a.zh.srt",
        video="/videos/a.mp4",
    )

    extra = runtime._build_task_started_extra(
        ctx=ctx, stage="stt", task_cfg={"batch_size": 16, "device": "cpu"}
    )

    assert captured["batch_size"] == 16
    assert captured["device"] == "cpu"
    assert extra["effective_config"] == {"batch_size": 16, "device": "cpu"}


def _prepared_succeeded_job(
    repo: JobRepository, tmp_path: Path, *, job_id: str, checkpoint_mtime: float
) -> Path:
    """构造一个 succeeded job 的 artifact 目录，并放入带 mtime 的检查点。"""

    job_dir = (tmp_path / "artifacts") / job_id
    (job_dir / "translate").mkdir(parents=True, exist_ok=True)
    checkpoint = job_dir / "translate" / "a.zh.srt.progress.json"
    checkpoint.write_text("{}", encoding="utf-8")
    os.utime(checkpoint, (checkpoint_mtime, checkpoint_mtime))
    return job_dir


def test_artifact_cleanup_defers_job_dir_while_checkpoint_unexpired(
    tmp_path: Path,
) -> None:
    """C.4：成功 job 存在未过期检查点时，整个目录延迟清理。"""

    repo = _repo(tmp_path)
    runtime = _runtime(repo, tmp_path, checkpoint_ttl_sec=3600)
    job_id = repo.enqueue("/tmp/ttl.mp4", "ja").job_id
    now = datetime.now(timezone.utc)
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE jobs SET status='succeeded', finished_at=? WHERE job_id=?",
            (now.isoformat(), job_id),
        )
        conn.execute("UPDATE tasks SET status='succeeded' WHERE job_id=?", (job_id,))
    job_dir = _prepared_succeeded_job(
        repo, tmp_path, job_id=job_id, checkpoint_mtime=now.timestamp()
    )

    runtime._run_artifact_cleanup_once()

    assert job_dir.exists()
    assert (job_dir / "translate" / "a.zh.srt.progress.json").exists()


def test_artifact_cleanup_removes_job_dir_after_checkpoint_ttl(tmp_path: Path) -> None:
    """C.4：检查点 TTL 过期后，成功 job 整目录被清理。"""

    repo = _repo(tmp_path)
    runtime = _runtime(repo, tmp_path, checkpoint_ttl_sec=60)
    job_id = repo.enqueue("/tmp/ttl-expired.mp4", "ja").job_id
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE jobs SET status='succeeded', finished_at=? WHERE job_id=?",
            (old.isoformat(), job_id),
        )
        conn.execute("UPDATE tasks SET status='succeeded' WHERE job_id=?", (job_id,))
    job_dir = _prepared_succeeded_job(
        repo, tmp_path, job_id=job_id, checkpoint_mtime=old.timestamp()
    )

    runtime._run_artifact_cleanup_once()

    assert not job_dir.exists()


def test_artifact_cleanup_ttl_zero_deletes_immediately(tmp_path: Path) -> None:
    """C.4：TTL=0（配置或 job 覆盖）表示立即可清。"""

    repo = _repo(tmp_path)
    runtime = _runtime(repo, tmp_path, checkpoint_ttl_sec=0)
    job_id = repo.enqueue("/tmp/ttl-zero.mp4", "ja").job_id
    now = datetime.now(timezone.utc)
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE jobs SET status='succeeded', finished_at=? WHERE job_id=?",
            (now.isoformat(), job_id),
        )
        conn.execute("UPDATE tasks SET status='succeeded' WHERE job_id=?", (job_id,))
    job_dir = _prepared_succeeded_job(
        repo, tmp_path, job_id=job_id, checkpoint_mtime=now.timestamp()
    )

    runtime._run_artifact_cleanup_once()

    assert not job_dir.exists()


def test_failed_job_checkpoint_is_never_cleaned(tmp_path: Path) -> None:
    """C.4：失败 job 始终保留（检查点可被 retry 复用），不参与清理。"""

    repo = _repo(tmp_path)
    runtime = _runtime(repo, tmp_path, checkpoint_ttl_sec=0)
    job_id = repo.enqueue("/tmp/failed-ttl.mp4", "ja").job_id
    old = datetime.now(timezone.utc) - timedelta(days=30)
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE jobs SET status='failed', finished_at=? WHERE job_id=?",
            (old.isoformat(), job_id),
        )
        conn.execute("UPDATE tasks SET status='failed' WHERE job_id=?", (job_id,))
    job_dir = _prepared_succeeded_job(
        repo, tmp_path, job_id=job_id, checkpoint_mtime=old.timestamp()
    )

    runtime._run_artifact_cleanup_once()

    assert job_dir.exists()


class _FakeMetric:
    """最小指标替身：记录 labels().inc() 调用。"""

    def __init__(self) -> None:
        """初始化调用记录。"""

        self.calls: list[tuple[dict, float]] = []

    def labels(self, **kwargs):
        """返回可链式调用的记录代理。"""

        metric = self

        class _Bound:
            def inc(self, amount: float = 1.0) -> None:
                """记录计数增量。"""

                metric.calls.append((kwargs, amount))

            def observe(self, amount: float) -> None:
                """记录直方图观测值。"""

                metric.calls.append((kwargs, amount))

        return _Bound()


class _FakeMetrics:
    """MetricsState 替身，仅提供 runtime 会用到的三项指标。"""

    def __init__(self) -> None:
        """创建三项计数/直方图替身。"""

        self.tasks_claimed_total = _FakeMetric()
        self.tasks_finished_total = _FakeMetric()
        self.task_duration_seconds = _FakeMetric()


def test_claim_and_failure_record_metrics(tmp_path: Path, monkeypatch) -> None:
    """G：领取计数 + 失败完成计数/耗时写入注入的 metrics。"""

    repo = _repo(tmp_path)
    metrics = _FakeMetrics()
    runtime = _runtime(repo, tmp_path, metrics=metrics)
    repo.enqueue("/tmp/metrics.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None

    def _boom(*args, **kwargs):
        raise RuntimeError("ffmpeg_not_found")

    monkeypatch.setattr(runtime_module, "run_extract", _boom)
    runtime._execute_claimed(claimed, "w1", "extract")

    assert metrics.tasks_claimed_total.calls == [({"stage": "extract"}, 1.0)]
    finished = [
        c for c in metrics.tasks_finished_total.calls if c[0]["stage"] == "extract"
    ]
    assert finished and finished[0][0].get("result") in {"queued", "failed"}
    assert metrics.task_duration_seconds.calls


def test_job_level_ttl_override_wins(tmp_path: Path) -> None:
    """C.4：job 覆盖的 progress_ttl_sec 优先于注入的全局 TTL。"""

    repo = _repo(tmp_path)
    runtime = _runtime(repo, tmp_path, checkpoint_ttl_sec=0)
    job_id = repo.enqueue("/tmp/ttl-override.mp4", "ja").job_id
    now = datetime.now(timezone.utc)
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE jobs SET status='succeeded', finished_at=? WHERE job_id=?",
            (now.isoformat(), job_id),
        )
        conn.execute(
            "UPDATE tasks SET status='succeeded', task_config_json=? "
            "WHERE job_id=? AND stage='translate'",
            (json.dumps({"effective_config": {"progress_ttl_sec": 86400}}), job_id),
        )
    job_dir = _prepared_succeeded_job(
        repo, tmp_path, job_id=job_id, checkpoint_mtime=now.timestamp()
    )

    runtime._run_artifact_cleanup_once()

    assert job_dir.exists()
