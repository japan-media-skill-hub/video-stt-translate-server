"""版本化 schema 迁移注册表。

设计约束（SPEC F）：

- 每个迁移是**纯函数** `(conn) -> None`，只负责一个版本的 DDL/DML，不自行开启事务。
- 事务由 :meth:`Database._run_migrations` 用 `BEGIN IMMEDIATE` 统一包裹，迁移体**禁止**
  使用 `executescript`（会隐式 COMMIT，破坏原子性）。
- 迁移只允许“新增表/索引/可空列”，保证老库可平滑升级；已发布的版本号不得修改。
"""

from __future__ import annotations

import sqlite3
from typing import Callable

from whisper_stt_service.repo.migrations import m001_initial, m002_lease_heartbeat_indexes

# 迁移函数签名：接收一个已在事务内的连接，执行该版本的全部变更。
MigrationFn = Callable[[sqlite3.Connection], None]

# 迁移注册表：版本号 -> 迁移函数。启动时按版本升序执行，已应用版本记录在 schema_migrations。
MIGRATIONS: dict[int, MigrationFn] = {
    1: m001_initial.apply,
    2: m002_lease_heartbeat_indexes.apply,
}

# 当前代码期望的最高 schema 版本（启动日志/健康检查可对照）。
LATEST_SCHEMA_VERSION = max(MIGRATIONS)

__all__ = ["LATEST_SCHEMA_VERSION", "MIGRATIONS", "MigrationFn"]
