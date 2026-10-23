"""WorkerRuntime 停机状态判断测试。"""

from __future__ import annotations

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
from whisper_stt_service.repository import JobRepository
from whisper_stt_service.service.runtime import FencedLease, WorkerRuntime


def _settings() -> Settings:
    """构造 runtime 逻辑测试用配置。"""

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
        ),
        retry=RetrySettings(
            extract_max_retries=2,
            stt_max_retries=2,
            stt_whisperx_max_retries=2,
            translate_max_retries=2,
        ),
        runtime=RuntimeSettings(
            db_path=Path("/tmp/test.db"),
            progress_ttl_sec=300,
            log_root=Path("/tmp/logs"),
            artifact_root=Path("/tmp/artifacts"),
            artifact_cleanup_enabled=True,
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


def test_shutdown_status_can_exit_when_no_claimed(tmp_path: Path) -> None:
    """drain 后若无 claimed/inflight，应立即满足退出条件。"""

    db = Database(tmp_path / "q.db")
    db.init_schema()
    repo = JobRepository(db)
    runtime = WorkerRuntime(
        repo=repo,
        progress_store=ProgressStore(300),
        settings=_settings(),
        config_path=tmp_path / "config.ini",
        model_path="/tmp/model",
    )

    status = runtime.request_shutdown("unit_test")
    assert status["drain_requested"] is True
    assert status["can_exit"] is True


def test_shutdown_status_treats_foreign_claimed_as_advisory_only(
    tmp_path: Path,
) -> None:
    """A.6：全局 claimed 仅告警，其他进程/僵尸的 claimed 不应阻塞退出。

    历史断言描述(已改为本进程无在途可退出)：
    存在 claimed 任务时，drain 不应提前判定可退出。
    原断言依据全局 claimed_count，已被 A.6 废弃；本用例仍验证新规范：
    非本进程在途的 claimed 只作为告警指标，不阻塞退出。
    """

    db = Database(tmp_path / "q.db")
    db.init_schema()
    repo = JobRepository(db)
    _ = repo.enqueue("/tmp/demo.mp4", "ja")
    claimed = repo.claim_next(stage="extract", worker_id="w1", lease_timeout_sec=60)
    assert claimed is not None

    runtime = WorkerRuntime(
        repo=repo,
        progress_store=ProgressStore(300),
        settings=_settings(),
        config_path=tmp_path / "config.ini",
        model_path="/tmp/model",
    )

    status = runtime.request_shutdown("unit_test")
    assert status["drain_requested"] is True
    # 僵尸 claimed 仍出现在告警指标里……
    assert status["claimed_count"] >= 1
    # ……但它不属于本进程在途任务，drain 可以正常退出（不再永久挂住）。
    assert status["inflight"] == []
    assert status["can_exit"] is True


def test_shutdown_status_blocks_exit_when_local_inflight_exists(
    tmp_path: Path,
) -> None:
    """A.6：本进程仍有在途任务时，drain 不应提前判定可退出。"""

    db = Database(tmp_path / "q.db")
    db.init_schema()
    repo = JobRepository(db)
    runtime = WorkerRuntime(
        repo=repo,
        progress_store=ProgressStore(300),
        settings=_settings(),
        config_path=tmp_path / "config.ini",
        model_path="/tmp/model",
    )
    lease = FencedLease(
        repo=repo,
        task_id="task-local-1",
        job_id="job-local-1",
        stage="extract",
        worker_id="extract-w1",
        attempt=0,
        lease_timeout_sec=60,
        heartbeat_interval_sec=5.0,
    )
    with runtime._state_lock:
        runtime._inflight_leases["task-local-1"] = lease

    status = runtime.request_shutdown("unit_test")
    assert status["drain_requested"] is True
    assert status["inflight"] == ["task-local-1"]
    assert status["inflight_task_ids"] == ["task-local-1"]
    assert status["can_exit"] is False
