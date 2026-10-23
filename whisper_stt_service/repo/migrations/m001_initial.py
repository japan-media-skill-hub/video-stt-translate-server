"""001：基准 schema（jobs/tasks 建档、DAG 与归档表、历史补列）。

本迁移是把原 `Database.init_schema` 中内联的建表/补列逻辑**等价外置**而来：

- 语句全部逐条 `conn.execute`，不使用 `executescript`——后者会在事务中途隐式
  COMMIT，导致迁移失败时无法整体回滚。
- 建表保持 `IF NOT EXISTS`、补列保持“先查 PRAGMA table_info”，因此对全新库和
  已有的老库都幂等：老库（0.4.0 之前的版本）只会被补齐缺失的 `dag_json`/
  `job_config_json`/`task_config_json` 列与归档表。
"""

from __future__ import annotations

import sqlite3

# 基准表的建表与索引语句（顺序与历史实现保持一致，避免老库出现语义差异）。
_BASE_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        video_path TEXT NOT NULL,
        source_language TEXT NOT NULL,
        status TEXT NOT NULL,
        output_ja_path TEXT NOT NULL,
        output_zh_path TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        finished_at TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_video_created ON jobs(video_path, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS tasks (
        task_id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL,
        stage TEXT NOT NULL,
        status TEXT NOT NULL,
        depends_on_task_id TEXT,
        attempt INTEGER NOT NULL DEFAULT 0,
        max_retries INTEGER NOT NULL,
        timeout_sec INTEGER NOT NULL,
        lease_owner TEXT,
        lease_expires_at TEXT,
        claimed_at TEXT,
        started_at TEXT,
        finished_at TEXT,
        last_error TEXT,
        log_dir TEXT NOT NULL,
        log_file TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tasks_stage_status_created ON tasks(stage, status, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_job_stage ON tasks(job_id, stage)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_status_lease_exp ON tasks(status, lease_expires_at)",
)

# 历史增量列：(表名, 列定义, 列名)。用 PRAGMA 判断缺失后再 ALTER，老库同样适用。
_LEGACY_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("jobs", "dag_json TEXT", "dag_json"),
    ("jobs", "job_config_json TEXT", "job_config_json"),
    ("tasks", "task_config_json TEXT", "task_config_json"),
)

# 归档表的建表与索引语句。
_ARCHIVE_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS jobs_archive (
        job_id TEXT PRIMARY KEY,
        video_path TEXT NOT NULL,
        source_language TEXT NOT NULL,
        status TEXT NOT NULL,
        output_ja_path TEXT NOT NULL,
        output_zh_path TEXT NOT NULL,
        dag_json TEXT,
        job_config_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        finished_at TEXT,
        archived_at TEXT NOT NULL,
        archive_reason TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_archive_video_created ON jobs_archive(video_path, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_archive_status_created ON jobs_archive(status, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS tasks_archive (
        task_id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL,
        stage TEXT NOT NULL,
        status TEXT NOT NULL,
        depends_on_task_id TEXT,
        task_config_json TEXT,
        attempt INTEGER NOT NULL DEFAULT 0,
        max_retries INTEGER NOT NULL,
        timeout_sec INTEGER NOT NULL,
        lease_owner TEXT,
        lease_expires_at TEXT,
        claimed_at TEXT,
        started_at TEXT,
        finished_at TEXT,
        last_error TEXT,
        log_dir TEXT NOT NULL,
        log_file TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        archived_at TEXT NOT NULL,
        archive_reason TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tasks_archive_job_stage ON tasks_archive(job_id, stage)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_archive_stage_status_created ON tasks_archive(stage, status, created_at)",
)


def apply(conn: sqlite3.Connection) -> None:
    """执行 001：建立基准表、补齐历史列、建立归档表。

    调用方必须已经把 `conn` 置于事务中；本函数自身不 BEGIN/COMMIT/executescript。
    """

    for statement in _BASE_SCHEMA_STATEMENTS:
        conn.execute(statement)

    for table, column_sql, column_name in _LEGACY_COLUMNS:
        _ensure_column(conn, table, column_sql, column_name)

    for statement in _ARCHIVE_SCHEMA_STATEMENTS:
        conn.execute(statement)


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column_sql: str,
    column_name: str,
) -> None:
    """在旧库缺列时执行增量补齐（与 Database._ensure_column 等价，供迁移内自用）。"""

    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column_name in columns:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_sql}")
