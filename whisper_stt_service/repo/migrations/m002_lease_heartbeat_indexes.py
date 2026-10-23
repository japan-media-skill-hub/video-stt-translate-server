"""002：补齐租约心跳/回收与 task 级查询的索引。

迁移由原 `Database._migration_002_lease_heartbeat_indexes` 等价外置而来，索引定义
与原注释含义保持不变：

- idx_tasks_lease_owner_attempt：心跳与写回 fencing 按 (lease_owner, attempt) 定位任务。
- idx_tasks_job_status：job 状态聚合与 failed/blocked 任务重跑扫描。

只新增索引，幂等（IF NOT EXISTS），可安全重复执行。
"""

from __future__ import annotations

import sqlite3

_INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_tasks_lease_owner_attempt ON tasks(lease_owner, attempt)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_job_status ON tasks(job_id, status)",
)


def apply(conn: sqlite3.Connection) -> None:
    """执行 002：建立租约心跳与 job 状态索引。

    调用方必须已经把 `conn` 置于事务中；本函数自身不 BEGIN/COMMIT/executescript。
    """

    for statement in _INDEX_STATEMENTS:
        conn.execute(statement)
