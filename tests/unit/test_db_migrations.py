"""SPEC F 数据库迁移与连接策略测试。

覆盖证据：
- schema_migrations 版本表与 001/002 登记；
- 迁移真原子（失败时 DDL/DML 全部回滚、版本不登记）；
- 幂等（重复 init 不重复应用、迁移体可重复执行）；
- 老库（只登记 version=1 但没有实体/缺列）自动修复；
- 迁移失败 fail-fast（异常向上抛出）；
- 网络挂载识别 -> 跳过 WAL 并回退 DELETE；WAL 成功路径；临时 FS 告警；
- busy_timeout=30000、foreign_keys=ON、synchronous 策略；
- EXPLAIN QUERY PLAN 启动日志探针。
"""

from __future__ import annotations

import ast
import logging
import sqlite3
from pathlib import Path

import pytest

from whisper_stt_service.repo import database as database_module
from whisper_stt_service.repo.database import Database
from whisper_stt_service.repo.filesystem import (
    FilesystemInfo,
    detect_filesystem,
    parse_mounts,
)
from whisper_stt_service.repo.migrations import (
    LATEST_SCHEMA_VERSION,
    MIGRATIONS,
    m001_initial,
    m002_lease_heartbeat_indexes,
)


def _tables(db: Database) -> set[str]:
    """返回库中全部表名。"""

    with db.connect() as conn:
        return {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }


def _indexes(db: Database) -> set[str]:
    """返回库中全部索引名。"""

    with db.connect() as conn:
        return {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }


def _versions(db: Database) -> list[int]:
    """返回 schema_migrations 中已登记的版本号（升序）。"""

    with db.connect() as conn:
        return [
            int(r["version"])
            for r in conn.execute("SELECT version FROM schema_migrations ORDER BY version")
        ]


# ------------------------------------------------------------ 版本表与登记


def test_init_schema_registers_versions(tmp_path: Path) -> None:
    """全新库应登记 001..LATEST，并建出全部基准对象。"""

    db = Database(tmp_path / "new.db")
    db.init_schema()

    assert _versions(db) == sorted(MIGRATIONS)
    assert LATEST_SCHEMA_VERSION == max(MIGRATIONS)
    tables = _tables(db)
    assert {"jobs", "tasks", "jobs_archive", "tasks_archive", "schema_migrations"} <= tables
    assert {"idx_tasks_lease_owner_attempt", "idx_tasks_job_status"} <= _indexes(db)


def test_init_schema_is_idempotent(tmp_path: Path) -> None:
    """重复 init_schema 不重复应用迁移，也不报错。"""

    path = tmp_path / "idem.db"
    db = Database(path)
    db.init_schema()
    first = _versions(db)

    again = Database(path)
    assert again._run_migrations() == []
    again.init_schema()
    assert _versions(again) == first


def test_migration_functions_are_rerunnable(tmp_path: Path) -> None:
    """迁移体本身幂等：在已建好的库上再执行一次应成功且无副作用。"""

    db = Database(tmp_path / "rerun.db")
    db.init_schema()
    with db.tx() as conn:
        m001_initial.apply(conn)
        m002_lease_heartbeat_indexes.apply(conn)
    assert _versions(db) == sorted(MIGRATIONS)


# ---------------------------------------------------------------- 真原子性


def test_failed_migration_rolls_back_all_statements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """迁移中途失败必须整体回滚：不留下表/索引，也不登记版本。"""

    db = Database(tmp_path / "atomic.db")
    db.configure_journal_mode()

    def broken_001(conn: sqlite3.Connection) -> None:
        # 先成功执行 001 的真实语句，再制造一个必然失败的语句。
        m001_initial.apply(conn)
        conn.execute("CREATE TABLE should_be_rolled_back (x INTEGER)")
        conn.execute("INSERT INTO no_such_table VALUES (1)")

    monkeypatch.setitem(database_module._MIGRATIONS, 1, broken_001)
    with pytest.raises(sqlite3.Error):
        db._run_migrations()

    # DDL 与版本登记都必须回滚（executescript 的隐式 COMMIT 会让这里失败）。
    assert _tables(db) == {"schema_migrations"}
    assert _versions(db) == []


def test_no_executescript_in_migration_path() -> None:
    """迁移实现不得真的调用 executescript（会隐式提交，破坏原子性）。

    用 AST 检查“调用点”而不是文本匹配：docstring 里说明该禁令是允许且必要的。
    """

    for module in (m001_initial, m002_lease_heartbeat_indexes, database_module):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        calls = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        assert "executescript" not in calls, f"{module.__name__} must not call executescript"


def test_executescript_in_migration_would_leak_partial_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """反证用例：若迁移体使用 executescript，失败后会残留半成品。

    这条用例固定 `_run_migrations` 的“每版本独立事务”契约——它证明
    `test_failed_migration_rolls_back_all_statements` 的断言不是空转：
    换成 executescript 版本时，残留表会真实存在。
    """

    db = Database(tmp_path / "hazard.db")
    db.configure_journal_mode()

    def executescript_then_fail(conn: sqlite3.Connection) -> None:
        conn.executescript("CREATE TABLE half_done (x INTEGER)")
        conn.execute("INSERT INTO missing_table VALUES (1)")

    monkeypatch.setitem(database_module._MIGRATIONS, 1, executescript_then_fail)
    with pytest.raises(sqlite3.Error):
        db._run_migrations()
    # 这正是禁止 executescript 的原因：隐式 COMMIT 让半成品留在了库里。
    assert "half_done" in _tables(db)


def test_second_migration_failure_keeps_first_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """002 失败时回滚 002，但已提交的 001 保持生效（可重启续跑）。"""

    db = Database(tmp_path / "partial.db")

    def broken_002(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tmp_partial ON tasks(stage)")
        conn.execute("INSERT INTO missing_table VALUES (1)")

    monkeypatch.setitem(database_module._MIGRATIONS, 2, broken_002)
    with pytest.raises(sqlite3.Error):
        db._run_migrations()

    assert _versions(db) == [1]
    indexes = _indexes(db)
    assert "idx_tmp_partial" not in indexes
    assert "idx_tasks_job_status" not in indexes


def test_migration_failure_is_fail_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """init_schema 中迁移失败必须向上抛出（bootstrap 据此拒绝启动）。"""

    def boom(conn: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setitem(database_module._MIGRATIONS, 1, boom)
    db = Database(tmp_path / "failfast.db")
    with pytest.raises(sqlite3.OperationalError):
        db.init_schema()


# ------------------------------------------------------------------ 老库


def test_legacy_database_without_tables_is_upgraded(tmp_path: Path) -> None:
    """0.4.0 之前的老库（无 schema_migrations）应被补齐基线列与归档表。"""

    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY, video_path TEXT NOT NULL,
            source_language TEXT NOT NULL, status TEXT NOT NULL,
            output_ja_path TEXT NOT NULL, output_zh_path TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL, finished_at TEXT
        );
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, stage TEXT NOT NULL,
            status TEXT NOT NULL, depends_on_task_id TEXT,
            attempt INTEGER NOT NULL DEFAULT 0, max_retries INTEGER NOT NULL,
            timeout_sec INTEGER NOT NULL, lease_owner TEXT, lease_expires_at TEXT,
            claimed_at TEXT, started_at TEXT, finished_at TEXT, last_error TEXT,
            log_dir TEXT NOT NULL, log_file TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()

    db = Database(path)
    db.init_schema()

    assert _versions(db) == sorted(MIGRATIONS)
    with db.connect() as c:
        jobs_cols = {r["name"] for r in c.execute("PRAGMA table_info(jobs)")}
        tasks_cols = {r["name"] for r in c.execute("PRAGMA table_info(tasks)")}
    assert {"dag_json", "job_config_json"} <= jobs_cols
    assert "task_config_json" in tasks_cols
    assert {"jobs_archive", "tasks_archive"} <= _tables(db)


def test_version_row_without_objects_is_repaired(tmp_path: Path) -> None:
    """旧实现只写 version=1 却不建表/补列：必须检测实体缺失并重跑 001。"""

    path = tmp_path / "stub.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY, video_path TEXT NOT NULL,
            source_language TEXT NOT NULL, status TEXT NOT NULL,
            output_ja_path TEXT NOT NULL, output_zh_path TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL, finished_at TEXT
        );
        CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT);
        INSERT INTO schema_migrations(version, applied_at) VALUES (1, 'old');
        """
    )
    conn.commit()
    conn.close()

    db = Database(path)
    db.init_schema()

    assert {"dag_json", "job_config_json"} <= {
        r["name"] for r in db.connect().execute("PRAGMA table_info(jobs)")
    }
    assert {"jobs", "tasks", "jobs_archive", "tasks_archive"} <= _tables(db)
    assert _versions(db) == sorted(MIGRATIONS)


# --------------------------------------------------------- journal/PRAGMA


def _capture_warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """捕获 database 模块的 WARNING 文案。

    不用 caplog：应用 import 时可能已安装 dictConfig（propagate=False），
    导致 caplog 收不到记录；直接替换 logger.warning，与日志配置解耦。
    """

    messages: list[str] = []

    def _fake_warning(msg, *args, **kwargs) -> None:
        try:
            messages.append(msg % args if args else str(msg))
        except (TypeError, ValueError):
            messages.append(str(msg))

    monkeypatch.setattr(database_module.LOGGER, "warning", _fake_warning)
    return messages


def test_network_filesystem_skips_wal_and_falls_back_to_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """识别为网络挂载时不尝试 WAL，直接 DELETE 并告警（不是等 SQLite 拒绝）。"""

    db = Database(tmp_path / "net.db")
    db._filesystem = FilesystemInfo(
        fs_type="nfs",
        mount_point=str(tmp_path),
        is_network=True,
        is_ephemeral=False,
        source="proc_mounts",
    )
    messages = _capture_warnings(monkeypatch)
    mode = db.configure_journal_mode()

    assert mode == "delete"
    assert db.journal_mode == "delete"
    assert "network_filesystem:nfs" in " ".join(messages)
    with db.connect() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"


def test_wal_enabled_on_local_filesystem(tmp_path: Path) -> None:
    """本地磁盘正常启用 WAL。"""

    db = Database(tmp_path / "local.db")
    db._filesystem = FilesystemInfo("ext4", "/", False, False, "proc_mounts")
    assert db.configure_journal_mode() == "wal"
    with db.connect() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_wal_disabled_by_config_uses_delete(tmp_path: Path) -> None:
    """wal_enabled=False 时使用 DELETE，并说明是配置关闭。"""

    db = Database(tmp_path / "cfg.db", wal_enabled=False)
    assert db.configure_journal_mode() == "delete"
    assert db.journal_mode == "delete"


def test_ephemeral_filesystem_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tmpfs/overlay 允许 WAL 但要告警数据可能丢失。"""

    db = Database(tmp_path / "eph.db")
    db._filesystem = FilesystemInfo("tmpfs", "/dev/shm", False, True, "proc_mounts")
    messages = _capture_warnings(monkeypatch)
    assert db.configure_journal_mode() == "wal"
    assert "ephemeral filesystem" in " ".join(messages)


def test_connection_pragmas(tmp_path: Path) -> None:
    """每个连接都必须 busy_timeout=30000、foreign_keys=ON，WAL 下 synchronous=NORMAL。"""

    db = Database(tmp_path / "pragma.db")
    db._filesystem = FilesystemInfo("ext4", "/", False, False, "proc_mounts")
    db.configure_journal_mode()
    with db.connect() as conn:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL


def test_explain_query_plan_probes_return_plans(tmp_path: Path) -> None:
    """启动日志用的 EXPLAIN QUERY PLAN 探针应能返回计划文本。"""

    db = Database(tmp_path / "plan.db")
    db.init_schema()
    plans = db.check_query_plans()
    assert set(plans) == {"claim_next", "reap_expired_leases", "lease_heartbeat"}
    assert all(isinstance(value, str) for value in plans.values())
    assert any(plans.values())


# ------------------------------------------------------ 文件系统探测纯函数


def test_parse_mounts_handles_escapes() -> None:
    """/proc/mounts 中的八进制转义应被还原。"""

    text = "a /mnt/with\\040space ext4 rw 0 0\nb /mnt cifs rw 0 0\n"
    entries = parse_mounts(text)
    assert entries[0][1] == "/mnt/with space"
    assert entries[1][2] == "cifs"


def test_detect_filesystem_longest_prefix_wins() -> None:
    """嵌套挂载时按最长前缀匹配，避免把 /mnt/fast 误判成 /mnt 的 nfs。"""

    mounts = [
        ("srv:/", "/mnt", "nfs"),
        ("/dev/disk1", "/mnt/fast", "ext4"),
    ]
    info = detect_filesystem("/mnt/fast/db.sqlite3", mounts=mounts)
    assert info.fs_type == "ext4"
    assert info.is_network is False
    assert info.mount_point == "/mnt/fast"

    net = detect_filesystem("/mnt/other/db.sqlite3", mounts=mounts)
    assert net.is_network is True
    assert net.fs_type == "nfs"


def test_detect_filesystem_unknown_when_mounts_unreadable(tmp_path: Path) -> None:
    """读不到挂载表时返回 unknown，且不把它当作网络盘（不阻断启动）。"""

    missing = tmp_path / "no-mounts"
    info = detect_filesystem(tmp_path / "x.db", mounts_path=missing)
    assert info.source == "unknown"
    assert info.is_network is False

    db = Database(tmp_path / "unk.db", mounts_path=missing)
    assert db.configure_journal_mode() == "wal"
