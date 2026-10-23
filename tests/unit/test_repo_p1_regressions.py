"""P1/P2 回归：网络盘 WAL fail-fast 与损坏依赖载荷的事务内级联。

两个缺陷都是“表面成功、实际危险”的类型，因此测试必须断言**真实状态**，
而不是只断言日志或返回值。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from whisper_stt_service.repo.database import Database, JournalModeError
from whisper_stt_service.repo.filesystem import FilesystemInfo
from whisper_stt_service.repo.job_repository import JobRepository


def _wal_db(path: Path) -> Path:
    """建一个已处于 WAL 模式的真实数据库文件。"""

    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t(x)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.close()
    return path


# ------------------------------------------------- F: 网络盘 WAL fail-fast


def test_network_fs_with_concurrent_reader_fails_fast_not_spoof(tmp_path: Path) -> None:
    """并发读事务让 DELETE 切换失败时，必须抛错而不是谎报 delete。

    缺陷复现：旧实现吞掉 sqlite 异常并把 mode 设为 "delete"，但库里实际仍是
    WAL——网络盘上以 WAL 运行是危险状态。
    """

    path = _wal_db(tmp_path / "net.db")
    reader = sqlite3.connect(path, isolation_level=None)
    reader.execute("BEGIN")
    reader.execute("SELECT * FROM t").fetchall()
    try:
        db = Database(path)
        db._filesystem = FilesystemInfo("nfs", "/mnt", True, False, "proc_mounts")
        with pytest.raises(JournalModeError):
            db.configure_journal_mode()
        # 失败时不得把 delete 写入内部状态当成功。
        assert db.journal_mode is None
    finally:
        reader.execute("COMMIT")
        reader.close()


def test_network_fs_clean_fallback_reports_verified_delete(tmp_path: Path) -> None:
    """无并发时可正常回退，且回读结果确实是 delete。"""

    path = _wal_db(tmp_path / "net2.db")
    db = Database(path)
    db._filesystem = FilesystemInfo("cifs", "/mnt", True, False, "proc_mounts")
    assert db.configure_journal_mode() == "delete"
    with db.connect() as conn:
        actual = conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
    assert actual == "delete"
    assert db.journal_mode == actual


def test_reported_mode_always_matches_actual_pragma(tmp_path: Path) -> None:
    """任何成功路径下，报告的 journal_mode 必须等于数据库真实值。"""

    path = _wal_db(tmp_path / "local.db")
    db = Database(path)
    db._filesystem = FilesystemInfo("ext4", "/", False, False, "proc_mounts")
    reported = db.configure_journal_mode()
    with db.connect() as conn:
        actual = conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
    assert reported == actual


# --------------------------------------- A: 损坏依赖载荷必须事务内级联


def _repo(tmp_path: Path) -> JobRepository:
    """构造独立仓储。"""

    db = Database(tmp_path / "q.db")
    db.init_schema()
    return JobRepository(db, log_root=tmp_path / "logs", artifact_root=tmp_path / "artifacts")


def _corrupt_dependency(repo: JobRepository, task_id: str) -> None:
    """把任务的 depends_on_task_id 改成无法解析的 JSON。"""

    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE tasks SET depends_on_task_id=? WHERE task_id=?",
            ("[invalid", task_id),
        )


def test_corrupt_dependency_cascades_to_blocked_in_same_tx(tmp_path: Path) -> None:
    """extract 依赖载荷损坏 → extract failed 且下游同事务置 blocked。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/v.mp4", "ja")
    tasks = {t["stage"]: t for t in repo.get_job_detail(created.job_id)["tasks"]}
    _corrupt_dependency(repo, tasks["extract"]["task_id"])

    # 该任务被判定失败，因此本轮不领取任何任务。
    assert repo.claim_next("extract", "w1", 60) is None

    detail = repo.get_job_detail(created.job_id)
    by_stage = {t["stage"]: t for t in detail["tasks"]}
    assert by_stage["extract"]["status"] == "failed"
    assert by_stage["extract"]["last_error"] == "invalid_dependency_payload"
    # 关键断言：下游不得残留 queued（幽灵排队）。
    assert by_stage["stt"]["status"] == "blocked"
    assert by_stage["translate"]["status"] == "blocked"
    assert by_stage["stt"]["last_error"].startswith("upstream_failed:")
    assert by_stage["translate"]["last_error"].startswith("upstream_failed:")
    assert detail["status"] == "failed"


def test_corrupt_dependency_does_not_leave_claimable_downstream(tmp_path: Path) -> None:
    """级联后下游不可被领取（否则会形成永久排队）。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/v.mp4", "ja")
    tasks = {t["stage"]: t for t in repo.get_job_detail(created.job_id)["tasks"]}
    _corrupt_dependency(repo, tasks["extract"]["task_id"])
    repo.claim_next("extract", "w1", 60)

    assert repo.claim_next("stt", "w1", 60) is None
    assert repo.claim_next("translate", "w1", 60) is None


def test_corrupt_dependency_cascade_is_atomic(tmp_path: Path) -> None:
    """级联与置 failed 在同一事务内：不会出现“可领取中间态”。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/v.mp4", "ja")
    tasks = {t["stage"]: t for t in repo.get_job_detail(created.job_id)["tasks"]}
    _corrupt_dependency(repo, tasks["extract"]["task_id"])

    # 在 claim_next 内部观察：其事务提交后不应有任何 queued 下游。
    repo.claim_next("extract", "w1", 60)
    with repo.db.connect() as conn:
        rows = conn.execute(
            "SELECT stage, status FROM tasks WHERE job_id=? ORDER BY stage",
            (created.job_id,),
        ).fetchall()
    statuses = {r["stage"]: r["status"] for r in rows}
    assert "queued" not in statuses.values()


def test_corrupt_dependency_on_downstream_stage_only_blocks_itself(tmp_path: Path) -> None:
    """仅下游自身依赖损坏时，不应误伤无关任务。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/v.mp4", "ja")
    tasks = {t["stage"]: t for t in repo.get_job_detail(created.job_id)["tasks"]}
    _corrupt_dependency(repo, tasks["translate"]["task_id"])

    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None and claimed.task_id == tasks["extract"]["task_id"]
    assert repo.claim_next("translate", "w1", 60) is None

    detail = repo.get_job_detail(created.job_id)
    by_stage = {t["stage"]: t for t in detail["tasks"]}
    assert by_stage["translate"]["status"] == "failed"
    assert by_stage["stt"]["status"] == "queued"
