"""SPEC A.3 产物级 fencing：publish_task_artifacts 与 with_task_lease。

全部使用真实临时目录与真实文件，验证“旧 attempt 绝不覆盖新 attempt”，
而不是只 mock SQL。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from whisper_stt_service.repo.database import Database
from whisper_stt_service.repo.job_repository import JobRepository


def _repo(tmp_path: Path) -> JobRepository:
    """构造独立仓储。"""

    db = Database(tmp_path / "q.db")
    db.init_schema()
    return JobRepository(db, log_root=tmp_path / "logs", artifact_root=tmp_path / "artifacts")


def _claimed(tmp_path: Path, worker: str = "w1") -> tuple[JobRepository, str, int, str]:
    """入队并领取 extract 任务，返回 (repo, task_id, attempt, job_id)。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/v.mp4", "ja")
    claimed = repo.claim_next("extract", worker, 60)
    assert claimed is not None
    return repo, claimed.task_id, claimed.attempt, created.job_id


def _steal_lease(repo: JobRepository, task_id: str, new_worker: str) -> int:
    """模拟租约过期回收后由新 worker 重新领取，返回新 attempt。"""

    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE tasks SET status='queued', lease_owner=NULL, lease_expires_at=NULL WHERE task_id=?",
            (task_id,),
        )
    claimed = repo.claim_next("extract", new_worker, 60)
    assert claimed is not None and claimed.task_id == task_id
    return claimed.attempt


# ------------------------------------------------- publish_task_artifacts


def test_publish_moves_staged_to_final_and_marks_succeeded(tmp_path: Path) -> None:
    """正常路径：staged 被 rename 到正式路径，任务置 succeeded。"""

    repo, task_id, attempt, job_id = _claimed(tmp_path)
    staged = tmp_path / "private" / "a.ja.srt"
    staged.parent.mkdir(parents=True)
    staged.write_text("NEW-SUBS", encoding="utf-8")
    final = tmp_path / "out" / "a.ja.srt"

    ok = repo.publish_task_artifacts(
        task_id, worker_id="w1", attempt=attempt, artifacts=[(staged, final)]
    )
    assert ok is True
    assert final.read_text(encoding="utf-8") == "NEW-SUBS"
    assert not staged.exists()

    status = {t["task_id"]: t for t in repo.get_job_detail(job_id)["tasks"]}[task_id]
    assert status["status"] == "succeeded"


def test_stale_worker_cannot_overwrite_new_artifact(tmp_path: Path) -> None:
    """核心回归：租约被新 worker 接管后，旧 worker 发布必须失败且不动文件。"""

    repo, task_id, old_attempt, _ = _claimed(tmp_path, "w1")
    new_attempt = _steal_lease(repo, task_id, "w2")

    final = tmp_path / "out" / "a.ja.srt"
    final.parent.mkdir(parents=True)
    final.write_text("NEW-VERSION", encoding="utf-8")

    stale_staged = tmp_path / "old.staged"
    stale_staged.write_text("OLD-VERSION", encoding="utf-8")
    ok = repo.publish_task_artifacts(
        task_id,
        worker_id="w1",
        attempt=old_attempt,
        artifacts=[(stale_staged, final)],
    )
    assert ok is False
    # 正式产物必须仍是新 attempt 的内容，且旧 staging 未被消费。
    assert final.read_text(encoding="utf-8") == "NEW-VERSION"
    assert stale_staged.exists()

    # 新 worker 发布成功，覆盖为它自己的内容。
    new_staged = tmp_path / "new.staged"
    new_staged.write_text("W2-RESULT", encoding="utf-8")
    assert repo.publish_task_artifacts(
        task_id, worker_id="w2", attempt=new_attempt, artifacts=[(new_staged, final)]
    )
    assert final.read_text(encoding="utf-8") == "W2-RESULT"


def test_publish_rejects_expired_lease(tmp_path: Path) -> None:
    """租约过期（即使 owner/attempt 未变）也不得发布。"""

    repo, task_id, attempt, _ = _claimed(tmp_path)
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE tasks SET lease_expires_at=? WHERE task_id=?",
            ("2000-01-01T00:00:00+00:00", task_id),
        )
    staged = tmp_path / "s.staged"
    staged.write_text("X", encoding="utf-8")
    final = tmp_path / "f.srt"
    assert (
        repo.publish_task_artifacts(
            task_id, worker_id="w1", attempt=attempt, artifacts=[(staged, final)]
        )
        is False
    )
    assert not final.exists()


def test_publish_wrong_attempt_is_rejected(tmp_path: Path) -> None:
    """attempt 不匹配（即使 owner 一致）也不得发布。"""

    repo, task_id, attempt, _ = _claimed(tmp_path)
    staged = tmp_path / "s.staged"
    staged.write_text("X", encoding="utf-8")
    final = tmp_path / "f.srt"
    assert (
        repo.publish_task_artifacts(
            task_id, worker_id="w1", attempt=attempt + 1, artifacts=[(staged, final)]
        )
        is False
    )
    assert not final.exists()


def test_publish_multiple_artifacts_in_order(tmp_path: Path) -> None:
    """多产物按调用方顺序 rename，全部到位后置 succeeded。"""

    repo, task_id, attempt, _ = _claimed(tmp_path)
    staged_a = tmp_path / "a.staged"
    staged_b = tmp_path / "b.staged"
    staged_a.write_text("A", encoding="utf-8")
    staged_b.write_text("B", encoding="utf-8")
    final_a = tmp_path / "out" / "a.ja.srt"
    final_b = tmp_path / "out" / "a.zh.srt"

    assert repo.publish_task_artifacts(
        task_id,
        worker_id="w1",
        attempt=attempt,
        artifacts=[(staged_a, final_a), (staged_b, final_b)],
    )
    assert final_a.read_text(encoding="utf-8") == "A"
    assert final_b.read_text(encoding="utf-8") == "B"


def test_publish_missing_staged_raises_without_success(tmp_path: Path) -> None:
    """staged 缺失时抛错，任务不得被置 succeeded。"""

    repo, task_id, attempt, job_id = _claimed(tmp_path)
    with pytest.raises(FileNotFoundError):
        repo.publish_task_artifacts(
            task_id,
            worker_id="w1",
            attempt=attempt,
            artifacts=[(tmp_path / "missing.staged", tmp_path / "f.srt")],
        )
    status = {t["task_id"]: t for t in repo.get_job_detail(job_id)["tasks"]}[task_id]
    assert status["status"] == "claimed"


def test_publish_marks_job_succeeded_when_all_tasks_done(tmp_path: Path) -> None:
    """发布最后一个任务后 job 聚合为 succeeded。"""

    db = Database(tmp_path / "q.db")
    db.init_schema()
    repo = JobRepository(db, log_root=tmp_path / "logs", artifact_root=tmp_path / "a")
    dag = {
        "version": 1,
        "stages": [
            {"stage": "extract", "depends_on": []},
            {"stage": "stt", "depends_on": ["extract"]},
        ],
    }
    created = repo.enqueue("/tmp/v.mp4", "ja", dag=dag)

    for stage in ("extract", "stt"):
        claimed = repo.claim_next(stage, "w1", 60)
        assert claimed is not None
        staged = tmp_path / f"{stage}.staged"
        staged.write_text(stage, encoding="utf-8")
        assert repo.publish_task_artifacts(
            claimed.task_id,
            worker_id="w1",
            attempt=claimed.attempt,
            artifacts=[(staged, tmp_path / "out" / f"{stage}.out")],
        )
    assert repo.get_job_detail(created.job_id)["status"] == "succeeded"



# ------------------------------------------------ 租约时段与 rename 跨期


def test_publish_freezes_now_and_does_not_recheck_expiry_later(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rename 耗时跨过租约到期点：不得在二次校验里重新读时钟判过期。

    契约：租约过期只在**事务开始时**用冻结的 now 校验一次；之后（rename 期间）
    即使真实时间已越过 expires，也必须完成发布并置 succeeded，
    否则会出现“文件已发布但任务未成功”的不一致。

    实现约束：本方法只读一次时钟（用于冻结 now）；若实现改成在二次校验时
    “重新取当前时间”，本用例会失败——这正是我们要防的回归。
    """

    import whisper_stt_service.repo.job_repository as jr_module

    repo, task_id, attempt, job_id = _claimed(tmp_path)

    base = datetime.now(timezone.utc)
    expires = (base + timedelta(seconds=2)).isoformat()
    with repo.db.tx() as conn:
        conn.execute("UPDATE tasks SET lease_expires_at=? WHERE task_id=?", (expires, task_id))

    final = tmp_path / "out" / "a.ja.srt"
    staged = tmp_path / "a.staged"
    staged.write_text("NEW", encoding="utf-8")

    observed: list[str] = []

    def fake_now() -> str:
        # 模拟真实时间：每次被调用都前进 5 秒。
        value = base + timedelta(seconds=5 * len(observed))
        observed.append(value.isoformat())
        return value.isoformat()

    monkeypatch.setattr(jr_module, "_now", fake_now)

    assert repo.publish_task_artifacts(
        task_id, worker_id="w1", attempt=attempt, artifacts=[(staged, final)]
    )
    assert final.read_text(encoding="utf-8") == "NEW"
    with repo.db.connect() as conn:
        status = conn.execute(
            "SELECT status FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()["status"]
    assert status == "succeeded"

    # 冻结语义：**fencing/状态更新路径**只读一次时钟（事务开始冻结的 now 一路传到
    # 二次校验与 job 刷新）。事务之后的 task 日志事件会另取时间戳，那不属于 fencing
    # 路径，也不影响正确性，因此扣除这 1 次。
    assert len(observed) == 2, f"unexpected clock reads: {observed}"


def test_publish_rejects_when_expired_at_transaction_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """租约在事务开始时刻就已过期：仍必须拒绝且不写文件。"""

    import whisper_stt_service.repo.job_repository as jr_module

    repo, task_id, attempt, _ = _claimed(tmp_path)
    monkeypatch.setattr(
        jr_module,
        "_now",
        lambda: datetime.now(timezone.utc).isoformat(),
    )
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE tasks SET lease_expires_at=? WHERE task_id=?",
            ("2000-01-01T00:00:00+00:00", task_id),
        )
    staged = tmp_path / "s.staged"
    staged.write_text("X", encoding="utf-8")
    final = tmp_path / "f.srt"
    assert repo.publish_task_artifacts(
        task_id, worker_id="w1", attempt=attempt, artifacts=[(staged, final)]
    ) is False
    assert not final.exists()

# ------------------------------------------------------- with_task_lease


def test_with_task_lease_runs_action_inside_valid_lease(tmp_path: Path) -> None:
    """租约有效时 action 被执行且任务状态不变。"""

    repo, task_id, attempt, job_id = _claimed(tmp_path)
    checkpoint = tmp_path / "ckpt.json"
    calls: list[str] = []

    def action() -> None:
        calls.append("ran")
        checkpoint.write_text('{"batches": {"0/0": {"done": true}}}', encoding="utf-8")

    assert repo.with_task_lease(
        task_id, worker_id="w1", attempt=attempt, action=action
    )
    assert calls == ["ran"]
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["batches"]["0/0"]["done"]

    status = {t["task_id"]: t for t in repo.get_job_detail(job_id)["tasks"]}[task_id]
    assert status["status"] == "claimed"


def test_stale_attempt_cannot_overwrite_checkpoint(tmp_path: Path) -> None:
    """核心回归：旧 attempt 不得覆盖新 attempt 的检查点（避免丢批次）。"""

    repo, task_id, old_attempt, _ = _claimed(tmp_path, "w1")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text('{"batches": {"0/0": {"done": true}}}', encoding="utf-8")

    new_attempt = _steal_lease(repo, task_id, "w2")
    # 新 attempt 追加批次 0/1。
    assert repo.with_task_lease(
        task_id,
        worker_id="w2",
        attempt=new_attempt,
        action=lambda: checkpoint.write_text(
            '{"batches": {"0/0": {"done": true}, "0/1": {"done": true}}}',
            encoding="utf-8",
        ),
    )

    # 旧 attempt 试图回写更旧的检查点：必须被拒且内容不变。
    ran: list[int] = []
    ok = repo.with_task_lease(
        task_id,
        worker_id="w1",
        attempt=old_attempt,
        action=lambda: (ran.append(1), checkpoint.write_text('{"batches": {}}', encoding="utf-8")),
    )
    assert ok is False
    assert ran == []
    batches = json.loads(checkpoint.read_text(encoding="utf-8"))["batches"]
    assert set(batches) == {"0/0", "0/1"}


def test_with_task_lease_rejects_expired_lease(tmp_path: Path) -> None:
    """租约过期时不执行 action。"""

    repo, task_id, attempt, _ = _claimed(tmp_path)
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE tasks SET lease_expires_at=? WHERE task_id=?",
            ("2000-01-01T00:00:00+00:00", task_id),
        )
    ran: list[int] = []
    assert not repo.with_task_lease(
        task_id, worker_id="w1", attempt=attempt, action=lambda: ran.append(1)
    )
    assert ran == []


def test_with_task_lease_propagates_action_errors(tmp_path: Path) -> None:
    """action 抛异常时向上抛出且事务回滚（不吞错）。"""

    repo, task_id, attempt, _ = _claimed(tmp_path)

    def boom() -> None:
        raise OSError("disk full")

    with pytest.raises(OSError):
        repo.with_task_lease(task_id, worker_id="w1", attempt=attempt, action=boom)


def test_with_task_lease_handles_unknown_task(tmp_path: Path) -> None:
    """任务不存在时返回 False。"""

    repo = _repo(tmp_path)
    assert not repo.with_task_lease(
        "task-missing", worker_id="w1", attempt=0, action=lambda: None
    )


def test_repository_does_not_import_executor_atomic() -> None:
    """repo 不得 import executor.atomic（会触发包循环），原子写由 runtime 注入。"""

    import ast

    from whisper_stt_service.repo import job_repository as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any("executor" in name for name in imported)
