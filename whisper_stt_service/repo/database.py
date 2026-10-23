"""SQLite 访问与事务封装模块。"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from whisper_stt_service.repo.filesystem import FilesystemInfo, detect_filesystem
from whisper_stt_service.repo.migrations import LATEST_SCHEMA_VERSION, MIGRATIONS


LOGGER = logging.getLogger(__name__)

# 连接级 PRAGMA：WAL 由 init_schema 探测后决定，其余为固定值。
_BUSY_TIMEOUT_MS = 30000

# 迁移注册表：版本号 -> 迁移函数。新增变更一律追加新版本，不改旧版本。
# 迁移实现已外置到 whisper_stt_service/repo/migrations/，此处仅保留类型别名以兼容旧引用。
MigrationFn = Callable[[sqlite3.Connection], None]


class JournalModeError(RuntimeError):
    """journal_mode 无法安全确定时抛出（fail-fast，由启动流程拒绝启动）。

    典型场景：数据库位于网络文件系统，按 SPEC F 必须回退 DELETE，但实际回读
    SQLite 的 `PRAGMA journal_mode` 仍为 `wal`（例如并发读事务导致切换返回
    SQLITE_BUSY）。此时**不允许**继续以 WAL 运行，也**不允许**谎报已回退。
    """


def _utc_now() -> str:
    """返回当前 UTC 时间的 ISO8601 字符串（迁移审计用）。"""

    return datetime.now(timezone.utc).isoformat()


# 基线版本：schema_migrations 中该版本表示“基准表已存在”。
# 0.4.0 及更早版本只写了 version=1 却没有任何 001 迁移实现，因此这里不能仅凭
# 版本行就跳过 001——必须再做一次基准对象核对（见 _schema_base_is_complete）。
_BASE_SCHEMA_VERSION = 1

# 001 基准 schema 必须存在的对象（用于识别“只登记版本、没有实体”的旧库）。
_BASE_SCHEMA_OBJECTS: tuple[tuple[str, str], ...] = (
    ("table", "jobs"),
    ("table", "tasks"),
    ("table", "jobs_archive"),
    ("table", "tasks_archive"),
)

# 老库（0.4.0 之前）缺失、需要 001 补齐的列。
_BASE_SCHEMA_COLUMNS: tuple[tuple[str, str], ...] = (
    ("jobs", "dag_json"),
    ("jobs", "job_config_json"),
    ("tasks", "task_config_json"),
)

# 迁移注册表实现在 whisper_stt_service/repo/migrations/（001 基准 schema、002 索引等），
# 启动时按版本升序执行，已应用版本记录在 schema_migrations。保留别名供旧引用继续可用。
_MIGRATIONS: dict[int, MigrationFn] = MIGRATIONS


class Database:
    """提供 SQLite 连接、建表和事务上下文能力。"""

    def __init__(
        self,
        db_path: Path,
        *,
        wal_enabled: bool = True,
        mounts_path: Path | str | None = None,
    ) -> None:
        """记录数据库文件路径。

        同时记录 WAL 开关（默认开启）。
        `mounts_path` 仅用于测试注入自定义挂载表；缺省读取 `/proc/self/mounts`。
        """

        self.db_path = db_path
        # journal_mode 会持久化到数据库文件，因此启动探测一次后全局复用，
        # 避免每条连接都执行 PRAGMA（只读查询连接也保持一致）。
        self._journal_mode: str | None = None
        self._wal_requested = bool(wal_enabled)
        # 文件系统探测结果：启动时计算一次，供 WAL 决策与启动日志复用。
        self._filesystem: FilesystemInfo | None = None
        self._mounts_path = mounts_path

    def connect(self) -> sqlite3.Connection:
        """创建具备 Row 访问能力的连接对象。"""

        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        # 迁移/并发写场景：默认隔离级别会隐式开启事务并与 BEGIN IMMEDIATE 冲突，
        # 这里改为自动提交，事务边界完全由 tx()/_run_migrations 显式控制。
        conn.isolation_level = None
        if self._journal_mode is not None:
            self._apply_pragmas(conn)
        return conn

    def _apply_pragmas(self, conn: sqlite3.Connection) -> None:
        """按探测结果设置连接级 PRAGMA（busy_timeout/synchronous/journal_mode）。"""

        conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        if self._journal_mode == "wal":
            # WAL 下 NORMAL 在断电场景仍保证一致性，写入显著更快。
            conn.execute("PRAGMA synchronous = NORMAL")
        else:
            conn.execute("PRAGMA synchronous = FULL")

    @property
    def filesystem(self) -> FilesystemInfo:
        """返回数据库所在文件系统的探测结果（启动时计算一次并缓存）。"""

        if self._filesystem is None:
            self._filesystem = detect_filesystem(
                self.db_path, mounts_path=self._mounts_path
            )
        return self._filesystem

    def configure_journal_mode(self) -> str:
        """探测并启用 WAL；网络盘等场景回退 DELETE 并告警。

        返回实际生效的 journal_mode（小写）。幂等：重复调用只处理一次。

        判定顺序（SPEC F）：
        1. 静态识别文件系统：命中网络/分布式挂载 -> **不尝试 WAL**，直接 DELETE 并 WARN
           （这类挂载往往能“成功”切到 WAL 但在写竞争下损坏库，不能只等 SQLite 拒绝）。
        2. 否则尝试 `PRAGMA journal_mode = WAL`；结果不是 wal 或抛错 -> 回退 DELETE 并 WARN。
        3. 临时文件系统（tmpfs/overlay）允许 WAL，但额外 WARN，提示重启后可能丢数据。
        """

        if self._journal_mode is not None:
            return self._journal_mode
        filesystem = self.filesystem
        if self._wal_requested and not filesystem.is_network:
            self._enable_wal()
        else:
            self._disable_wal(filesystem)
        return str(self._journal_mode)

    def _enable_wal(self) -> None:
        """尝试把数据库切到 WAL，并记录回退原因。"""

        filesystem = self.filesystem
        with self.connect() as conn:
            mode = "delete"
            if self._wal_requested:
                try:
                    row = conn.execute("PRAGMA journal_mode = WAL").fetchone()
                    mode = str(row[0]).lower() if row is not None else "delete"
                except sqlite3.DatabaseError as exc:
                    LOGGER.warning(
                        "enable WAL failed, fallback to DELETE: db_path=%s error=%s",
                        self.db_path,
                        exc,
                    )
                    mode = "delete"
                if mode != "wal":
                    LOGGER.warning(
                        "journal_mode is not WAL, fallback to DELETE semantics: db_path=%s mode=%s",
                        self.db_path,
                        mode,
                    )
                    try:
                        conn.execute("PRAGMA journal_mode = DELETE")
                        mode = "delete"
                    except sqlite3.DatabaseError as exc:
                        LOGGER.warning(
                            "fallback DELETE failed: db_path=%s error=%s",
                            self.db_path,
                            exc,
                        )
        self._journal_mode = mode
        if mode == "wal" and filesystem.is_ephemeral:
            LOGGER.warning(
                "WAL enabled on ephemeral filesystem (%s), data may be lost after reboot: db_path=%s mount_point=%s",
                filesystem.fs_type,
                self.db_path,
                filesystem.mount_point,
            )

    def _disable_wal(self, filesystem: FilesystemInfo) -> None:
        """不启用 WAL：显式回退 DELETE 并说明原因（网络挂载或配置关闭）。

        网络挂载直接跳过 `PRAGMA journal_mode = WAL`，避免在不可靠的共享内存/锁语义上
        跑一段时间后才出问题；顺手把持久化的 journal_mode 改回 DELETE，确保后续连接一致。

        fail-fast（SPEC F）：切换后**回读** SQLite 的实际 journal_mode 并以其为准。
        若切换失败或回读结果不是 DELETE（网络盘残留 WAL 属危险状态），抛
        :class:`JournalModeError` 让启动失败，而不是把 `delete` 当成既成事实记录。
        """

        reasons: list[str] = []
        if not self._wal_requested:
            reasons.append("wal_disabled_by_config")
        if filesystem.is_network:
            reasons.append(
                f"network_filesystem:{filesystem.fs_type or 'unknown'}"
            )
        reason = ",".join(reasons) or "wal_not_requested"
        try:
            with self.connect() as conn:
                conn.execute("PRAGMA journal_mode = DELETE")
                # 以 SQLite 的**实际回读值**为准，绝不按“意图”假定成功：
                # 切换在有并发读事务时会以 SQLITE_BUSY 失败，此时库里仍是 WAL。
                row = conn.execute("PRAGMA journal_mode").fetchone()
                mode = str(row[0]).lower() if row is not None else ""
        except sqlite3.DatabaseError as exc:
            raise JournalModeError(
                f"cannot switch journal_mode to DELETE on network filesystem: "
                f"db_path={self.db_path} reason={reason} error={exc}"
            ) from exc

        if mode != "delete":
            # 网络盘上残留 WAL 是危险的：宁可拒绝启动，也不能谎报已回退。
            raise JournalModeError(
                f"journal_mode still {mode!r} after DELETE fallback; refusing to run WAL "
                f"on network filesystem: db_path={self.db_path} reason={reason}"
            )

        LOGGER.warning(
            "WAL disabled, using DELETE journal (verified): db_path=%s reason=%s fs_type=%s mount_point=%s",
            self.db_path,
            reason,
            filesystem.fs_type or "unknown",
            filesystem.mount_point or "unknown",
        )
        self._journal_mode = mode

    @property
    def journal_mode(self) -> str | None:
        """返回已探测的 journal_mode；尚未探测时为 None。"""

        return self._journal_mode

    def check_query_plans(self) -> dict[str, str]:
        """用 EXPLAIN QUERY PLAN 记录关键查询的执行计划（启动日志用）。

        仅用于观测索引是否命中，不影响业务逻辑；查询失败时返回空 dict。
        """

        probes: dict[str, tuple[str, tuple[object, ...]]] = {
            "claim_next": (
                "EXPLAIN QUERY PLAN SELECT task_id FROM tasks WHERE stage=? AND status='queued' ORDER BY created_at ASC, task_id ASC",
                ("extract",),
            ),
            "reap_expired_leases": (
                "EXPLAIN QUERY PLAN SELECT task_id FROM tasks WHERE status='claimed' AND lease_expires_at < ?",
                ("1970-01-01T00:00:00+00:00",),
            ),
            "lease_heartbeat": (
                "EXPLAIN QUERY PLAN UPDATE tasks SET lease_expires_at=? WHERE task_id=? AND status='claimed' AND lease_owner=? AND attempt=?",
                ("x", "x", "x", 0),
            ),
        }
        plans: dict[str, str] = {}
        try:
            with self.connect() as conn:
                for name, (sql, params) in probes.items():
                    rows = conn.execute(sql, params).fetchall()
                    plans[name] = " | ".join(
                        str(r["detail"]) for r in rows if "detail" in r.keys()
                    )
        except sqlite3.Error as exc:
            LOGGER.warning("explain query plan failed: db_path=%s error=%s", self.db_path, exc)
            return {}
        return plans

    def init_schema(self) -> None:
        """初始化 jobs 与 tasks 两张核心表。

        旧两表描述保留。当前实现已扩展为初始化数据库 schema：探测 journal_mode
        后执行版本化迁移，并包含 schema_migrations 与归档等扩展表/列。

        流程：
        1. `configure_journal_mode`：先识别网络挂载，再尝试 WAL / 回退 DELETE。
        2. `_run_migrations`：按版本升序执行 001/002/...，每个版本一个独立事务。

        迁移失败会抛出异常（fail-fast），由 bootstrap 拒绝启动；测试/脚本里的
        调用方同理会看到异常，而不是半应用状态。
        """

        # 先完成 journal_mode 探测，后续所有连接（含迁移事务）行为一致。
        self.configure_journal_mode()
        # 建表与补列已全部外置到 repo/migrations/001_initial.py，按版本执行。
        self._run_migrations()

    def _run_migrations(self) -> list[int]:
        """按版本顺序执行未应用的迁移，返回本次应用的版本号列表。

        原子性约定：
        - 每个版本在**独立的** `BEGIN IMMEDIATE` 事务中执行，迁移函数内的所有 DDL/DML
          与 `schema_migrations` 版本登记一起提交；任一步失败即整体回滚，不会留下
          “表建了但版本没记”或“版本记了但表没建”的半成品。
        - 迁移函数禁止使用 `executescript`（其隐式 COMMIT 会破坏上述原子性）。
        - 单一版本的失败不影响已经提交的更早版本（它们保持幂等，重启后继续）。

        失败即抛出（fail-fast，由启动流程拒绝启动）。迁移只允许“新增索引/表/可空列”，
        保证老库可平滑升级。
        """

        self._ensure_schema_migrations_table()
        self._repair_base_schema_if_needed()

        applied: list[int] = []
        for version in sorted(_MIGRATIONS):
            if self._is_migration_applied(version):
                continue
            migration = _MIGRATIONS[version]
            try:
                with self.tx() as conn:
                    migration(conn)
                    conn.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES(?, ?)",
                        (version, _utc_now()),
                    )
            except sqlite3.Error as exc:
                LOGGER.error(
                    "schema migration failed (rolled back): version=%s db_path=%s error=%s",
                    version,
                    self.db_path,
                    exc,
                )
                raise
            applied.append(version)
            LOGGER.info(
                "schema migration applied: version=%s db_path=%s", version, self.db_path
            )
        LOGGER.info(
            "schema ready: version=%s applied=%s db_path=%s",
            LATEST_SCHEMA_VERSION,
            applied,
            self.db_path,
        )
        return applied

    def _ensure_schema_migrations_table(self) -> None:
        """确保版本表存在（老库首次升级时补建）。"""

        with self.tx() as conn:
            # 版本表本身极简，必须能在任何历史库上直接补建，且不能写成 executescript。
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT)"
            )

    def _is_migration_applied(self, version: int) -> bool:
        """判断某个版本是否已经登记在 schema_migrations。"""

        with self.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM schema_migrations WHERE version=? LIMIT 1", (version,)
            ).fetchone()
        return row is not None

    def _base_schema_is_complete(self) -> bool:
        """核对 001 基准 schema 的实体对象是否齐全。

        老版本（0.4.0 及更早）会写入 version=1 却没有任何 001 迁移实现；如果只看版本行
        就会跳过补齐，留下缺列的库。这里以实际对象为主证据。
        """

        with self.connect() as conn:
            for kind, name in _BASE_SCHEMA_OBJECTS:
                row = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type=? AND name=? LIMIT 1",
                    (kind, name),
                ).fetchone()
                if row is None:
                    return False
            for table, column in _BASE_SCHEMA_COLUMNS:
                columns = {
                    r["name"] for r in conn.execute(f"PRAGMA table_info({table})")
                }
                if column not in columns:
                    return False
        return True

    def _repair_base_schema_if_needed(self) -> None:
        """版本行显示 001 已完成但实体缺失时，重新执行 001 修复老库。

        修复本身也是幂等的（IF NOT EXISTS / 缺列才 ALTER），且在一个事务内完成；
        若修复失败会整体回滚并抛出，保持 fail-fast 语义。
        """

        if not self._is_migration_applied(_BASE_SCHEMA_VERSION):
            return
        if self._base_schema_is_complete():
            return
        LOGGER.warning(
            "schema_migrations says version=%s but base objects are missing, re-applying 001: db_path=%s",
            _BASE_SCHEMA_VERSION,
            self.db_path,
        )
        migration = _MIGRATIONS[_BASE_SCHEMA_VERSION]
        try:
            with self.tx() as conn:
                migration(conn)
        except sqlite3.Error as exc:
            LOGGER.error(
                "base schema repair failed (rolled back): version=%s db_path=%s error=%s",
                _BASE_SCHEMA_VERSION,
                self.db_path,
                exc,
            )
            raise

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table: str,
        column_sql: str,
        column_name: str,
    ) -> None:
        """在旧库缺列时执行增量补齐。"""

        columns = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column_name in columns:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_sql}")

    @contextmanager
    def tx(self):
        """开启 `BEGIN IMMEDIATE` 事务并自动提交/回滚。"""

        conn = self.connect()
        try:
            # 立即获取写锁，避免并发领取时出现竞争写入冲突。
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            # 任何异常均回滚，保证调用方看到一致状态。
            conn.rollback()
            raise
        finally:
            # 无论成功或失败，最终都释放连接资源。
            conn.close()
