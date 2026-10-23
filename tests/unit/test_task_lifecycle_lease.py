"""批次1：租约心跳、过期回收、写回 fencing 与失败级联测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from whisper_stt_service.db import Database
from whisper_stt_service.repository import JobRepository


def _repo(tmp_path: Path, *, log_root: Path | None = None) -> JobRepository:
    """构造临时库仓储，日志目录落在 tmp_path 下。"""

    db = Database(tmp_path / "q.db")
    db.init_schema()
    return JobRepository(db, log_root=log_root or (tmp_path / "logs"))


def _expire_lease(repo: JobRepository, task_id: str, seconds: int = 5) -> None:
    """把任务租约改成已过期，模拟 worker 崩溃/挂起未续租。"""

    past = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE tasks SET lease_expires_at=? WHERE task_id=?", (past, task_id)
        )


def _task_row(repo: JobRepository, task_id: str) -> dict:
    """读取任务的原始行，便于断言状态/attempt/租约字段。"""

    with repo.db.connect() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    assert row is not None
    return dict(row)


def _task_log_file(repo: JobRepository, task_id: str) -> str:
    """读取任务日志路径（get_job_detail 不返回 log_file 列）。"""

    with repo.db.connect() as conn:
        row = conn.execute(
            "SELECT log_file FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
    assert row is not None
    return str(row["log_file"])


def _task_log_events(log_file: str) -> list[dict]:
    """解析 task.log 的 jsonl 事件。"""

    path = Path(log_file)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_expired_lease_is_reclaimed_and_attempt_increments(tmp_path: Path) -> None:
    """A.2：过期租约被回收为 queued，attempt+1 并写 lease_reclaimed 事件。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/reap.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None and claimed.attempt == 0
    _expire_lease(repo, claimed.task_id)

    stats = repo.reclaim_expired_leases()

    assert stats["reclaimed"] == 1
    row = _task_row(repo, claimed.task_id)
    assert row["status"] == "queued"
    assert row["attempt"] == 1
    assert row["last_error"] == "lease_expired"
    assert row["lease_owner"] is None

    events = [
        e for e in _task_log_events(row["log_file"]) if e["event"] == "lease_reclaimed"
    ]
    assert len(events) == 1
    assert events[0]["worker_id"] == "w1"
    assert events[0]["attempt"] == 1

    # 回收后应能被其他 worker 再次领取。
    reclaimed = repo.claim_next("extract", "w2", 60)
    assert reclaimed is not None
    assert reclaimed.task_id == claimed.task_id
    assert reclaimed.attempt == 1
    assert repo.get_job_status(created.job_id) == "running"


def test_reclaim_exceeding_retry_budget_marks_failed_and_cascades(
    tmp_path: Path,
) -> None:
    """A.2：回收后超预算置 failed（lease_expired_retry_exhausted）并级联 blocked。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/reap-exhausted.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None
    with repo.db.tx() as conn:
        conn.execute("UPDATE tasks SET attempt=2 WHERE task_id=?", (claimed.task_id,))
    _expire_lease(repo, claimed.task_id)

    stats = repo.reclaim_expired_leases()

    assert stats["failed"] == 1
    row = _task_row(repo, claimed.task_id)
    assert row["status"] == "failed"
    assert row["last_error"] == "lease_expired_retry_exhausted"

    detail = repo.get_job_detail(created.job_id)
    assert detail is not None
    statuses = {t["stage"]: t["status"] for t in detail["tasks"]}
    assert statuses["stt"] == "blocked"
    assert statuses["translate"] == "blocked"
    assert detail["status"] == "failed"


def test_heartbeat_renews_only_own_lease(tmp_path: Path) -> None:
    """A.1：心跳只续自己的租约，其他 worker/attempt 续租被拒绝。"""

    repo = _repo(tmp_path)
    repo.enqueue("/tmp/heartbeat.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None

    assert (
        repo.heartbeat_task(
            claimed.task_id, worker_id="w1", attempt=0, lease_timeout_sec=120
        )
        is True
    )
    assert (
        repo.heartbeat_task(
            claimed.task_id, worker_id="w2", attempt=0, lease_timeout_sec=120
        )
        is False
    )
    assert (
        repo.heartbeat_task(
            claimed.task_id, worker_id="w1", attempt=1, lease_timeout_sec=120
        )
        is False
    )
    # 续租后租约不再过期，回收不会误伤。
    row = _task_row(repo, claimed.task_id)
    assert row["status"] == "claimed"
    assert row["lease_owner"] == "w1"
    assert repo.reclaim_expired_leases()["reclaimed"] == 0


def test_fenced_success_rejected_for_stale_attempt(tmp_path: Path) -> None:
    """A.3：旧 attempt 的成功写回被 fencing 拒绝，不改状态也不刷新 job。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/fencing.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None
    _expire_lease(repo, claimed.task_id)
    repo.reclaim_expired_leases()
    # 新 worker 领取了 attempt=1 的租约。
    second = repo.claim_next("extract", "w2", 60)
    assert second is not None and second.attempt == 1

    marked = repo.mark_task_succeeded(
        claimed.task_id, worker_id="w1", attempt=claimed.attempt
    )

    assert marked is False
    row = _task_row(repo, claimed.task_id)
    assert row["status"] == "claimed"
    assert row["lease_owner"] == "w2"
    assert row["attempt"] == 1
    assert repo.get_job_status(created.job_id) == "running"


def test_fenced_failure_rejected_for_stale_attempt(tmp_path: Path) -> None:
    """A.3：旧 attempt 的失败写回同样被拒绝，不会消耗新 attempt 预算。"""

    repo = _repo(tmp_path)
    repo.enqueue("/tmp/fencing-fail.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None
    _expire_lease(repo, claimed.task_id)
    repo.reclaim_expired_leases()
    second = repo.claim_next("extract", "w2", 60)
    assert second is not None

    status = repo.mark_task_failed(
        claimed.task_id, "boom", worker_id="w1", attempt=claimed.attempt
    )

    assert status == "fenced_stale_result"
    row = _task_row(repo, claimed.task_id)
    assert row["status"] == "claimed"
    assert row["attempt"] == 1


def test_failure_cascades_blocked_downstream_and_fails_job(tmp_path: Path) -> None:
    """A.4：上游进入 failed 时，下游被置 blocked 且 job 聚合为 failed。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/cascade.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None
    # attempt 已到 2：本次失败即超预算（max_retries=2 -> attempt+1=3）。
    with repo.db.tx() as conn:
        conn.execute("UPDATE tasks SET attempt=2 WHERE task_id=?", (claimed.task_id,))

    status = repo.mark_task_failed(
        claimed.task_id, "ffmpeg_failed", worker_id="w1", attempt=2
    )
    assert status == "failed"

    detail = repo.get_job_detail(created.job_id)
    assert detail is not None
    tasks = {t["stage"]: t for t in detail["tasks"]}
    assert tasks["extract"]["status"] == "failed"
    assert tasks["stt"]["status"] == "blocked"
    assert tasks["translate"]["status"] == "blocked"
    assert tasks["stt"]["last_error"] == f"upstream_failed:{claimed.task_id}"
    assert detail["status"] == "failed"

    # blocked 任务不会被领取（不是“幽灵排队”）。
    assert repo.claim_next("stt", "w1", 60) is None
    summary = repo.queue_summary()
    assert summary["stages"]["stt"]["blocked"] == 1


def test_retry_budget_exhausted_blocks_intermediary_downstream(tmp_path: Path) -> None:
    """A.4：中间阶段失败时，间接下游同样被级联置 blocked。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/cascade2.mp4", "ja")
    extract = repo.claim_next("extract", "w1", 60)
    assert extract is not None
    assert (
        repo.mark_task_succeeded(
            extract.task_id, worker_id="w1", attempt=extract.attempt
        )
        is True
    )
    stt = repo.claim_next("stt", "w1", 60)
    assert stt is not None
    with repo.db.tx() as conn:
        conn.execute("UPDATE tasks SET attempt=2 WHERE task_id=?", (stt.task_id,))
    assert (
        repo.mark_task_failed(stt.task_id, "asr_crash", worker_id="w1", attempt=2)
        == "failed"
    )

    detail = repo.get_job_detail(created.job_id)
    assert detail is not None
    tasks = {t["stage"]: t for t in detail["tasks"]}
    assert tasks["translate"]["status"] == "blocked"
    assert tasks["translate"]["last_error"] == f"upstream_failed:{stt.task_id}"
    assert detail["status"] == "failed"


def test_recovery_counts_as_attempt_and_fails_when_exhausted(tmp_path: Path) -> None:
    """A.5：启动恢复消耗重试预算；超预算直接 failed 并级联 blocked。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/recovery.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None

    stats = repo.recover_claimed_to_queued(counts_as_attempt=True)

    assert stats == {"recovered": 1, "failed": 0}
    row = _task_row(repo, claimed.task_id)
    assert row["status"] == "queued"
    assert row["attempt"] == 1

    # 再次领取后就地崩溃 -> 第二次恢复超预算。
    with repo.db.tx() as conn:
        conn.execute("UPDATE tasks SET attempt=2 WHERE task_id=?", (claimed.task_id,))
    repo.claim_next("extract", "w2", 60)
    stats2 = repo.recover_claimed_to_queued(counts_as_attempt=True)
    assert stats2 == {"recovered": 0, "failed": 1}

    detail = repo.get_job_detail(created.job_id)
    assert detail is not None
    tasks = {t["stage"]: t for t in detail["tasks"]}
    assert tasks["extract"]["status"] == "failed"
    assert tasks["stt"]["status"] == "blocked"
    assert detail["status"] == "failed"


def test_recovery_can_skip_attempt_counting_for_debug(tmp_path: Path) -> None:
    """A.5：recovery_counts_as_attempt=False 时保持旧行为（attempt 不变）。"""

    repo = _repo(tmp_path)
    repo.enqueue("/tmp/recovery-legacy.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None

    stats = repo.recover_claimed_to_queued(counts_as_attempt=False)

    assert stats == {"recovered": 1, "failed": 0}
    row = _task_row(repo, claimed.task_id)
    assert row["status"] == "queued"
    assert row["attempt"] == 0


def test_recovered_upstream_unblocks_downstream(tmp_path: Path) -> None:
    """A.2/A.5：上游仍有重试预算被重新排队时，下游 blocked 应被撤销。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/unblock.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None
    with repo.db.tx() as conn:
        conn.execute("UPDATE tasks SET attempt=2 WHERE task_id=?", (claimed.task_id,))
    _expire_lease(repo, claimed.task_id)
    assert repo.reclaim_expired_leases()["failed"] == 1

    detail = repo.get_job_detail(created.job_id)
    assert detail is not None
    statuses = {t["stage"]: t["status"] for t in detail["tasks"]}
    assert statuses["stt"] == "blocked"

    # retry 之后下游不再是 blocked（同时验证 retry 的状态机）。
    result = repo.retry_job(created.job_id)
    assert result["retried_tasks"] == 3
    detail = repo.get_job_detail(created.job_id)
    assert detail is not None
    assert {t["status"] for t in detail["tasks"]} == {"queued"}
    assert detail["status"] == "queued"


def test_retry_only_allows_failed_job(tmp_path: Path) -> None:
    """A.7：retry 只接受 failed job；queued/运行中抛 retry_not_allowed。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/retry-gate.mp4", "ja")
    with pytest.raises(ValueError, match="retry_not_allowed"):
        repo.retry_job(created.job_id)
    with pytest.raises(ValueError, match="job_not_found"):
        repo.retry_job("job-missing")


def test_retry_resets_blocked_tasks_and_clears_errors(tmp_path: Path) -> None:
    """A.7：retry 把 failed/blocked 重置为 queued、attempt=0、清 last_error。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/retry-ok.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None
    with repo.db.tx() as conn:
        conn.execute("UPDATE tasks SET attempt=2 WHERE task_id=?", (claimed.task_id,))
    repo.mark_task_failed(
        claimed.task_id, "ffmpeg_failed", worker_id="w1", attempt=2
    )
    assert repo.get_job_status(created.job_id) == "failed"

    result = repo.retry_job(created.job_id)

    assert result == {"job_id": created.job_id, "retried_tasks": 3, "status": "queued"}
    detail = repo.get_job_detail(created.job_id)
    assert detail is not None
    for task in detail["tasks"]:
        assert task["status"] == "queued"
        assert task["attempt"] == 0
        assert task["last_error"] is None
        assert task["finished_at"] is None
        events = _task_log_events(_task_log_file(repo, task["task_id"]))
        assert any(e["event"] == "task_retry_requested" for e in events)

    # 重跑后可正常领取。
    again = repo.claim_next("extract", "w3", 60)
    assert again is not None
    assert again.task_id == claimed.task_id


def test_retry_rejects_historical_invalid_dag(tmp_path: Path) -> None:
    """A.7：历史 job 的 DAG 无法被当前版本解析时 retry 返回 409。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/retry-bad-dag.mp4", "ja")
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE jobs SET status='failed', dag_json=? WHERE job_id=?",
            ('{"version": 1, "stages": "broken"}', created.job_id),
        )

    with pytest.raises(ValueError, match="retry_job_dag_invalid"):
        repo.retry_job(created.job_id)


def test_cancel_only_allows_not_started_job(tmp_path: Path) -> None:
    """A.7：cancel 只允许全 queued 的 job；已开始/已结束返回 cancel_not_allowed。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/cancel-gate.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None
    with pytest.raises(ValueError, match="cancel_not_allowed"):
        repo.cancel_job(created.job_id)


def test_cancel_marks_tasks_cancelled_and_frees_path(tmp_path: Path) -> None:
    """A.7：cancel 把任务置 cancelled、job 结束，并释放同路径入队占用。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/cancel-ok.mp4", "ja")

    result = repo.cancel_job(created.job_id)

    assert result["cancelled_tasks"] == 3
    detail = repo.get_job_detail(created.job_id)
    assert detail is not None
    assert {t["status"] for t in detail["tasks"]} == {"cancelled"}
    # SPEC 6：jobs.status 取值不变（queued/running/succeeded/failed）。
    assert detail["status"] in {"queued", "running", "succeeded", "failed"}
    assert detail["finished_at"] is not None

    # 路径占用被释放：再次入队应新建 job，而不是幂等返回已取消的 job。
    again = repo.enqueue("/tmp/cancel-ok.mp4", "ja")
    assert again.job_id != created.job_id
    assert again.message == "created"
