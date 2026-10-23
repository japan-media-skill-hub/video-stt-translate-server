"""批次1：运行时租约心跳、协作式取消与有界停机测试。"""

from __future__ import annotations

import json
import time
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
from whisper_stt_service.service import runtime as runtime_module
from whisper_stt_service.service.runtime import FencedLease, WorkerRuntime


def _settings(
    *,
    lease_timeout_sec: int = 60,
    drain_timeout_sec: int = 300,
    recovery_counts_as_attempt: bool = True,
) -> Settings:
    """构造运行时逻辑测试用配置（无 worker 线程）。"""

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
            lease_timeout_sec=lease_timeout_sec,
            drain_timeout_sec=drain_timeout_sec,
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
            artifact_cleanup_enabled=False,
            artifact_cleanup_interval_sec=3600,
            artifact_cleanup_statuses=("succeeded",),
            model_path=Path("/tmp/model"),
            recovery_counts_as_attempt=recovery_counts_as_attempt,
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


def _repo(tmp_path: Path) -> JobRepository:
    """构造临时库仓储。"""

    db = Database(tmp_path / "q.db")
    db.init_schema()
    return JobRepository(db, log_root=tmp_path / "logs", artifact_root=tmp_path / "artifacts")


def _task_log_file(repo: JobRepository, task_id: str) -> str:
    """读取任务的 task.log 路径（get_job_detail 不返回该列）。"""

    with repo.db.connect() as conn:
        row = conn.execute(
            "SELECT log_file FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
    assert row is not None
    return str(row["log_file"])


def _read_events(log_file: str) -> list[dict]:
    """读取 task.log jsonl 事件列表。"""

    path = Path(log_file)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text("utf-8").splitlines()
        if line.strip()
    ]


def _runtime(
    repo: JobRepository,
    tmp_path: Path,
    *,
    settings: Settings | None = None,
    heartbeat_interval_sec: float = 0.05,
    exit_callback=None,
) -> WorkerRuntime:
    """构造不启动线程的运行时实例。"""

    return WorkerRuntime(
        repo=repo,
        progress_store=ProgressStore(300),
        settings=settings or _settings(),
        config_path=tmp_path / "config.ini",
        model_path="/tmp/model",
        exit_callback=exit_callback,
        heartbeat_interval_sec=heartbeat_interval_sec,
    )


def test_heartbeat_interval_derived_from_lease_with_floor() -> None:
    """A.1：默认心跳间隔为 max(lease/3, 10s)，防止短租约写放大。"""

    short = WorkerRuntime(
        repo=None,  # type: ignore[arg-type]
        progress_store=None,  # type: ignore[arg-type]
        settings=_settings(lease_timeout_sec=60),
        config_path=Path("/tmp/config.ini"),
        model_path="/tmp/model",
    )
    # 60/3=20 > 10，下限只在更短租约时生效。
    assert short.heartbeat_interval_sec == 20.0
    tiny = WorkerRuntime(
        repo=None,  # type: ignore[arg-type]
        progress_store=None,  # type: ignore[arg-type]
        settings=_settings(lease_timeout_sec=6),
        config_path=Path("/tmp/config.ini"),
        model_path="/tmp/model",
    )
    assert tiny.heartbeat_interval_sec == 10.0

    long = WorkerRuntime(
        repo=None,  # type: ignore[arg-type]
        progress_store=None,  # type: ignore[arg-type]
        settings=_settings(lease_timeout_sec=600),
        config_path=Path("/tmp/config.ini"),
        model_path="/tmp/model",
    )
    assert long.heartbeat_interval_sec == 200.0


def test_heartbeat_keeps_long_blocking_execution_alive(tmp_path: Path) -> None:
    """A.1：执行线程长时间阻塞时，带外心跳线程仍持续续租。"""

    repo = _repo(tmp_path)
    repo.enqueue("/tmp/long.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 1)
    assert claimed is not None
    lease = FencedLease(
        repo=repo,
        task_id=claimed.task_id,
        job_id=claimed.job_id,
        stage=claimed.stage,
        worker_id="w1",
        attempt=claimed.attempt,
        lease_timeout_sec=1,
        heartbeat_interval_sec=0.05,
    )
    lease.start()
    try:
        # 模拟 0.5s 的“长执行”：不做任何协作检查，租约由心跳维持。
        time.sleep(0.5)
        assert lease.lost is False
        with repo.db.connect() as conn:
            row = conn.execute(
                "SELECT status, lease_owner, lease_expires_at FROM tasks WHERE task_id=?",
                (claimed.task_id,),
            ).fetchone()
        assert row["status"] == "claimed"
        assert row["lease_owner"] == "w1"
        assert repo.reclaim_expired_leases()["reclaimed"] == 0
    finally:
        lease.stop()


def test_heartbeat_detects_reclaim_and_cancel_check_returns_true(tmp_path: Path) -> None:
    """A.1：租约被回收后心跳失败，cancel_check 立即返回 True。"""

    repo = _repo(tmp_path)
    runtime = _runtime(repo, tmp_path)
    repo.enqueue("/tmp/reclaimed.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None
    lease = FencedLease(
        repo=repo,
        task_id=claimed.task_id,
        job_id=claimed.job_id,
        stage=claimed.stage,
        worker_id="w1",
        attempt=claimed.attempt,
        lease_timeout_sec=60,
        heartbeat_interval_sec=0.05,
    )
    cancel_check = runtime._make_cancel_check(lease)
    lease.start()
    try:
        assert cancel_check() is False
        # 人为回收：租约过期后由 reaper 收回给 w2。
        past = "2000-01-01T00:00:00+00:00"
        with repo.db.tx() as conn:
            conn.execute(
                "UPDATE tasks SET lease_expires_at=? WHERE task_id=?",
                (past, claimed.task_id),
            )
        repo.reclaim_expired_leases()
        deadline = time.monotonic() + 5
        while not cancel_check() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert cancel_check() is True
        assert lease.lost is True
        # 丢失原因以 reaper 的 fencing 拒绝为准；心跳事件是异步写日志，
        # 这里断言“已标记丢失 + 原因来自 fencing/时间片”。
        assert "lease_lost" in (lease.lost_reason or "")
        assert lease.lost_reason in {
            "lease_lost:fencing_rejected",
            "lease_lost:time_slice_check",
        }
    finally:
        lease.stop()


def test_cancel_check_detects_foreign_takeover_via_time_slice(tmp_path: Path) -> None:
    """A.1：时间片检查发现任务已被他人接管时，cancel_check 返回 True。"""

    repo = _repo(tmp_path)
    runtime = _runtime(repo, tmp_path)
    repo.enqueue("/tmp/takeover.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None
    lease = FencedLease(
        repo=repo,
        task_id=claimed.task_id,
        job_id=claimed.job_id,
        stage=claimed.stage,
        worker_id="w1",
        attempt=claimed.attempt,
        lease_timeout_sec=60,
        heartbeat_interval_sec=60.0,
    )
    cancel_check = runtime._make_cancel_check(lease)
    assert cancel_check() is False

    # 模拟他人 retry：任务回到 queued，再由 w2 领取。
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE tasks SET status='queued', lease_owner=NULL WHERE task_id=?",
            (claimed.task_id,),
        )
    assert repo.claim_next("extract", "w2", 60) is not None

    # 节流保护下第一次调用可能仍返回 False，推进时间片后必然为 True。
    lease._last_time_slice_check = 0.0
    assert cancel_check() is True
    assert "time_slice_check" in (lease.lost_reason or "")


def test_worker_task_failure_cascades_and_fenced_stale_writeback_is_logged(
    tmp_path: Path, monkeypatch
) -> None:
    """A.3/A.4：执行器抛错 -> 失败级联；失租后的写回被拒绝且记录事件。"""

    repo = _repo(tmp_path)
    runtime = _runtime(repo, tmp_path)
    created = repo.enqueue("/tmp/exec-fail.mp4", "ja")

    def _boom(*args, **kwargs):
        raise RuntimeError("ffmpeg_not_found")

    monkeypatch.setattr(runtime_module, "run_extract", _boom)
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None
    runtime._execute_claimed(claimed, "w1", "extract")

    detail = repo.get_job_detail(created.job_id)
    # 第一次失败仍在重试预算内：任务回到 queued。
    tasks = {t["stage"]: t for t in detail["tasks"]}
    assert tasks["extract"]["status"] == "queued"
    assert tasks["extract"]["attempt"] == 1
    events = _read_events(_task_log_file(repo, claimed.task_id))
    assert [e["event"] for e in events if e["event"] in {"task_started", "task_failed"}] == [
        "task_started",
        "task_failed",
    ]

    # 失租后（租约被他人接管）执行器才返回：写回被 fencing 拒绝。
    claimed2 = repo.claim_next("extract", "w2", 60)
    assert claimed2 is not None
    lease = FencedLease(
        repo=repo,
        task_id=claimed2.task_id,
        job_id=claimed2.job_id,
        stage=claimed2.stage,
        worker_id="w2",
        attempt=claimed2.attempt,
        lease_timeout_sec=60,
        heartbeat_interval_sec=60.0,
    )
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE tasks SET lease_owner='w9' WHERE task_id=?", (claimed2.task_id,)
        )
    ctx = repo.get_task_execution_context(claimed2.task_id)
    runtime._record_fenced_stale(ctx, lease, "succeeded", None)
    events = [
        json.loads(line)
        for line in Path(ctx.log_file).read_text("utf-8").splitlines()
        if line.strip()
    ]
    assert events[-1]["event"] == "fenced_stale_result"
    assert events[-1]["result"] == "succeeded"
    with repo.db.connect() as conn:
        row = conn.execute(
            "SELECT status FROM tasks WHERE task_id=?", (claimed2.task_id,)
        ).fetchone()
    assert row["status"] == "claimed"


def test_shutdown_status_reports_deadline_inflight_and_blocked(tmp_path: Path) -> None:
    """A.6：停机状态包含 drain_deadline_at / inflight / blocked 字段。"""

    repo = _repo(tmp_path)
    runtime = _runtime(repo, tmp_path, settings=_settings(drain_timeout_sec=120))

    before = runtime.shutdown_status()
    assert before["drain_requested"] is False
    assert before["drain_deadline_at"] is None
    assert before["inflight"] == []
    assert before["blocked"] == 0

    status = runtime.request_shutdown("unit_test")
    assert status["drain_deadline_at"] is not None
    assert 100 <= float(status["drain_remaining_sec"]) <= 121
    assert status["can_exit"] is True


def test_drain_timeout_forces_rollback_of_local_claimed(tmp_path: Path) -> None:
    """A.6：drain 超时后强制回滚本进程 claimed、请求中止并退出。"""

    repo = _repo(tmp_path)
    exits: list[str] = []
    runtime = _runtime(
        repo, tmp_path, exit_callback=lambda: exits.append("exit")
    )
    created = repo.enqueue("/tmp/drain-timeout.mp4", "ja")
    claimed = repo.claim_next("extract", "extract-w1", 60)
    assert claimed is not None
    lease = FencedLease(
        repo=repo,
        task_id=claimed.task_id,
        job_id=claimed.job_id,
        stage=claimed.stage,
        worker_id="extract-w1",
        attempt=claimed.attempt,
        lease_timeout_sec=60,
        heartbeat_interval_sec=60.0,
    )
    with runtime._state_lock:
        runtime._inflight_leases[claimed.task_id] = lease

    # 超时：模拟 drain 期限已过。
    runtime.request_shutdown("unit_test")
    with runtime._state_lock:
        runtime._drain_deadline_monotonic = 0.0
    assert runtime.shutdown_status()["drain_timed_out"] is True
    assert runtime.shutdown_status()["can_exit"] is True

    runtime._force_drain_timeout()

    with repo.db.connect() as conn:
        row = conn.execute(
            "SELECT status, lease_owner FROM tasks WHERE task_id=?", (claimed.task_id,)
        ).fetchone()
    assert row["status"] == "queued"
    assert row["lease_owner"] is None
    assert lease.lost is True
    assert repo.get_job_status(created.job_id) == "queued"

    runtime._trigger_process_exit()
    assert exits == ["exit"]
    assert runtime.shutdown_status()["exit_triggered"] is True


def test_exit_callback_failure_falls_back_to_sigterm(
    tmp_path: Path, monkeypatch
) -> None:
    """A.6：注入回调异常时回退 SIGTERM，且不会在测试里真的杀掉进程。"""

    repo = _repo(tmp_path)
    killed: list[tuple] = []

    def _bad_callback() -> None:
        raise RuntimeError("server_not_bound")

    runtime = _runtime(repo, tmp_path, exit_callback=_bad_callback)
    monkeypatch.setattr(
        runtime_module.os, "kill", lambda pid, sig: killed.append((pid, sig))
    )

    runtime._trigger_process_exit()

    assert killed and killed[0][1] == runtime_module.signal.SIGTERM


def test_start_reports_recovery_stats(tmp_path: Path) -> None:
    """A.5：启动日志/接口可读到恢复条数与超预算失败条数。"""

    repo = _repo(tmp_path)
    repo.enqueue("/tmp/start-recovery.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None
    runtime = _runtime(
        repo, tmp_path, settings=_settings(recovery_counts_as_attempt=False)
    )

    recovered = runtime.start()
    try:
        assert recovered == 1
        assert runtime.startup_stats() == {"recovered": 1, "failed": 0}
    finally:
        runtime.stop(timeout_sec=2)
