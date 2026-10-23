"""批次1：产物级 fencing（attempt 隔离暂存 + SQL 锁下 promotion）回归测试。"""

from __future__ import annotations

import json
from pathlib import Path

from whisper_stt_service.db import Database
from whisper_stt_service.progress import ProgressStore
from whisper_stt_service.repository import JobRepository
from whisper_stt_service.service.runtime import (
    ArtifactStaging,
    FencedLease,
    WorkerRuntime,
)

from tests.unit.test_runtime_integration import _settings


def _repo(tmp_path: Path) -> JobRepository:
    """构造指向 tmp 的仓储（artifact 根在 tmp 内）。"""

    db = Database(tmp_path / "q.db")
    db.init_schema()
    return JobRepository(
        db, log_root=tmp_path / "logs", artifact_root=tmp_path / "artifacts"
    )


def _runtime(repo: JobRepository, tmp_path: Path) -> WorkerRuntime:
    """构造不启动线程的运行时。"""

    return WorkerRuntime(
        repo=repo,
        progress_store=ProgressStore(300),
        settings=_settings(),
        config_path=tmp_path / "config.ini",
        model_path="/tmp/model",
        exit_callback=lambda: None,
        heartbeat_interval_sec=0.05,
        checkpoint_ttl_sec=3600,
    )


def _lease(repo: JobRepository, claimed, worker_id: str) -> FencedLease:
    """按领取结果构造租约对象（连接已存在，不启动心跳）。"""

    return FencedLease(
        repo=repo,
        task_id=claimed.task_id,
        job_id=claimed.job_id,
        stage=claimed.stage,
        worker_id=worker_id,
        attempt=claimed.attempt,
        lease_timeout_sec=60,
        heartbeat_interval_sec=60.0,
    )


def _write_staged(staging: ArtifactStaging, content: str) -> None:
    """模拟执行器只在私有暂存路径写出产物。"""

    staging.staged_path.parent.mkdir(parents=True, exist_ok=True)
    staging.staged_path.write_text(content, encoding="utf-8")


def test_staging_path_is_attempt_scoped_and_same_directory(tmp_path: Path) -> None:
    """暂存路径与正式目标同目录，且含 attempt/worker 唯一标记。"""

    final = tmp_path / "artifacts" / "job-1" / "asr" / "a.ja.srt"
    first = ArtifactStaging(
        final_path=final, work_dir=tmp_path / "logs" / "j1" / "t1", attempt=0, worker_id="w1"
    )
    second = ArtifactStaging(
        final_path=final, work_dir=tmp_path / "logs" / "j1" / "t1", attempt=1, worker_id="w2"
    )

    assert first.staged_path.parent == final.parent
    assert first.staged_path != second.staged_path
    assert first.staged_path.name.startswith(f".{final.name}.staging-0-")
    assert second.staged_path.name.startswith(f".{final.name}.staging-1-")


def test_staging_path_is_stable_per_instance(tmp_path: Path) -> None:
    """同一实例多次读取暂存路径必须一致（nonce 在 __init__ 时固定）。"""

    staging = ArtifactStaging(
        final_path=tmp_path / "artifacts" / "job-1" / "asr" / "a.ja.srt",
        work_dir=tmp_path / "logs",
        attempt=0,
        worker_id="w1",
    )

    assert staging.staged_path == staging.staged_path
    assert staging.staged_checkpoint_path == staging.staged_checkpoint_path


def test_same_task_attempt_different_owner_gets_distinct_staging(
    tmp_path: Path,
) -> None:
    """回归：同 task 同 attempt、不同 owner 的暂存路径必须不同。

    即使 `recovery_counts_as_attempt=false` 让重启后的 attempt 不递增，
    旧 worker 也不能写进新 worker 的暂存文件。
    """

    final = tmp_path / "artifacts" / "job-1" / "audio" / "a.16k.wav"
    old_worker = ArtifactStaging(
        final_path=final, work_dir=tmp_path / "logs" / "t1", attempt=0, worker_id="extract-w1"
    )
    new_worker = ArtifactStaging(
        final_path=final, work_dir=tmp_path / "logs" / "t1", attempt=0, worker_id="extract-w2"
    )
    same_worker_other_instance = ArtifactStaging(
        final_path=final, work_dir=tmp_path / "logs" / "t1", attempt=0, worker_id="extract-w1"
    )

    assert old_worker.staged_path != new_worker.staged_path
    assert old_worker.staged_checkpoint_path != new_worker.staged_checkpoint_path
    # 同一 worker 的不同实例（例如进程重启）也必须隔离。
    assert old_worker.staged_path != same_worker_other_instance.staged_path


def test_stale_attempt_cannot_overwrite_published_artifact(tmp_path: Path) -> None:
    """核心回归：旧 attempt 挂起 -> 被回收 -> 新 attempt 发布 -> 旧 attempt 醒来。

    旧 attempt 只能写自己的暂存文件；正式产物必须仍是新 attempt 的内容。
    """

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/stale-race.mp4", "ja")
    first = repo.claim_next("extract", "w1", 60)
    assert first is not None
    older = _runtime(repo, tmp_path)
    old_staging = ArtifactStaging(
        final_path=Path(repo.get_job_detail(created.job_id)["tasks"][0]["task_config"]["io"]["output_path"]),
        work_dir=tmp_path / "logs" / "old",
        attempt=first.attempt,
        worker_id="w1",
    )
    stale_lease = _lease(repo, first, "w1")

    # 旧 attempt 先“挂起”：租约过期被回收，新 worker 领取 attempt=1。
    past = "2000-01-01T00:00:00+00:00"
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE tasks SET lease_expires_at=? WHERE task_id=?", (past, first.task_id)
        )
    repo.reclaim_expired_leases()
    second = repo.claim_next("extract", "w2", 60)
    assert second is not None and second.attempt == 1
    new_staging = ArtifactStaging(
        final_path=old_staging.final_path,
        work_dir=tmp_path / "logs" / "new",
        attempt=second.attempt,
        worker_id="w2",
    )

    # 新 attempt 正常发布。
    _write_staged(new_staging, "content-from-new-attempt")
    published, status = older._publish_stage_outputs(
        repo.get_task_execution_context(second.task_id),
        _lease(repo, second, "w2"),
        "extract",
        new_staging,
    )
    assert status == "succeeded"
    assert old_staging.final_path.read_text(encoding="utf-8") == "content-from-new-attempt"

    # 旧 attempt 现在才醒来写产物并尝试发布：只能写私有暂存，正式产物不变。
    _write_staged(old_staging, "content-from-stale-attempt")
    old_ctx = repo.get_task_execution_context(first.task_id)
    published_stale, stale_status = older._publish_stage_outputs(
        old_ctx, stale_lease, "extract", old_staging
    )

    assert stale_status == "fenced_stale_result"
    assert published_stale == []
    assert old_staging.final_path.read_text(encoding="utf-8") == "content-from-new-attempt"
    # 旧暂存不会被误当作正式产物；任务状态仍是新 attempt 的 succeeded。
    with repo.db.connect() as conn:
        row = conn.execute(
            "SELECT status, attempt FROM tasks WHERE task_id=?", (first.task_id,)
        ).fetchone()
    assert (row["status"], row["attempt"]) == ("succeeded", 1)


def test_failed_attempt_keeps_previous_published_artifact(tmp_path: Path) -> None:
    """新 attempt 失败时，旧正式产物保持不变（不发布半成品）。"""

    repo = _repo(tmp_path)
    repo.enqueue("/tmp/keep-old.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None
    runtime = _runtime(repo, tmp_path)
    ctx = repo.get_task_execution_context(claimed.task_id)
    final = Path(ctx.task_config["io"]["output_path"])
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_text("previous-good-content", encoding="utf-8")

    staging = ArtifactStaging(
        final_path=final, work_dir=tmp_path / "logs", attempt=0, worker_id="w1"
    )
    _write_staged(staging, "half-written")
    runtime._mark_failed_with_log(ctx, _lease(repo, claimed, "w1"), "boom")

    assert final.read_text(encoding="utf-8") == "previous-good-content"
    assert staging.staged_path.exists()


def test_publish_without_claim_is_rejected_and_touches_nothing(tmp_path: Path) -> None:
    """未领取（非 claimed）时 promotion 被拒绝，正式路径不被创建/修改。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/not-claimed.mp4", "ja")
    runtime = _runtime(repo, tmp_path)
    detail = repo.get_job_detail(created.job_id)
    extract_task = detail["tasks"][0]
    ctx = repo.get_task_execution_context(extract_task["task_id"])
    final = Path(ctx.task_config["io"]["output_path"])
    staging = ArtifactStaging(
        final_path=final, work_dir=tmp_path / "logs", attempt=0, worker_id="w1"
    )
    _write_staged(staging, "should-not-publish")
    lease = FencedLease(
        repo=repo,
        task_id=ctx.task_id,
        job_id=ctx.job_id,
        stage="extract",
        worker_id="w1",
        attempt=0,
        lease_timeout_sec=60,
        heartbeat_interval_sec=60.0,
    )

    published, status = runtime._publish_stage_outputs(ctx, lease, "extract", staging)

    assert (published, status) == ([], "fenced_stale_result")
    assert not final.exists()


def test_task_log_has_eleven_fields_and_request_id(tmp_path: Path) -> None:
    """G：task 日志固定 11 字段，request_id 来自 metadata，extra 不能覆盖。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/log-fields.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None
    runtime = _runtime(repo, tmp_path)
    ctx = repo.get_task_execution_context(claimed.task_id)
    ctx.task_config = dict(ctx.task_config)
    ctx.task_config["metadata"] = {"request_id": "req-abc"}

    runtime._write_task_log(
        log_file=ctx.log_file,
        job_id=ctx.job_id,
        task_id=ctx.task_id,
        stage=ctx.stage,
        worker_id="w1",
        event="task_started",
        result="running",
        ctx=ctx,
        attempt=0,
        extra={"request_id": "should-not-override", "custom": 1},
    )

    payload = json.loads(Path(ctx.log_file).read_text("utf-8").strip().splitlines()[-1])
    expected = {
        "timestamp",
        "request_id",
        "job_id",
        "task_id",
        "stage",
        "worker_id",
        "event",
        "result",
        "attempt",
        "effective_config",
        "error",
    }
    assert expected.issubset(payload.keys())
    assert payload["request_id"] == "req-abc"
    assert payload["custom"] == 1
    assert created.job_id == payload["job_id"]


def test_task_log_request_id_is_null_without_metadata(tmp_path: Path) -> None:
    """G：没有 metadata.request_id 时写 null，而不是空串。"""

    repo = _repo(tmp_path)
    repo.enqueue("/tmp/log-null.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None
    runtime = _runtime(repo, tmp_path)
    ctx = repo.get_task_execution_context(claimed.task_id)

    runtime._write_task_log(
        log_file=ctx.log_file,
        job_id=ctx.job_id,
        task_id=ctx.task_id,
        stage=ctx.stage,
        worker_id="w1",
        event="lease_lost",
        result="aborting",
        ctx=ctx,
        attempt=0,
        error="lease_lost",
    )

    payload = json.loads(Path(ctx.log_file).read_text("utf-8").strip().splitlines()[-1])
    assert payload["request_id"] is None
    assert payload["error"] == "lease_lost"
