"""队列与 worker 概览路由。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import sqlite3

from fastapi import APIRouter, Request

from whisper_stt_service.api.dependencies import get_repo, get_runtime
from whisper_stt_service.core.stages import SUPPORTED_STAGES
from whisper_stt_service.service.metrics import THROUGHPUT_WINDOW_SEC, compute_throughput


LOGGER = logging.getLogger(__name__)

router = APIRouter()


def _throughput_rows_from_repo(
    repo, *, window_sec: int, now: datetime
) -> list[dict] | None:
    """读取窗口内已终态任务（stage/status/finished_at）。

    优先调用仓储新增方法 `list_finished_tasks_since`（建议由仓储代理补充，
    签名 `(since_iso: str) -> list[dict]`）；仓储尚未提供时退回直接查询
    `tasks` 表——`finished_at` 是既有列，不需要任何 schema 变更。
    查询失败返回 None，由调用方标记 throughput 不可用，绝不 500。
    """

    since_iso = (now - timedelta(seconds=max(int(window_sec), 1))).isoformat()
    repo_reader = getattr(repo, "list_finished_tasks_since", None)
    if callable(repo_reader):
        try:
            return [
                {"stage": row["stage"], "status": row["status"], "finished_at": row["finished_at"]}
                for row in repo_reader(since_iso) or []
            ]
        except (sqlite3.Error, KeyError, TypeError) as exc:
            LOGGER.warning("throughput reader failed, fallback to direct query: error=%s", exc)

    db = getattr(repo, "db", None)
    if db is None:
        return None
    try:
        with db.connect() as conn:
            rows = conn.execute(
                """
                SELECT stage, status, finished_at
                  FROM tasks
                 WHERE finished_at IS NOT NULL
                   AND finished_at >= ?
                   AND status IN ('succeeded', 'failed', 'blocked', 'cancelled')
                """,
                (since_iso,),
            ).fetchall()
        return [
            {"stage": row["stage"], "status": row["status"], "finished_at": row["finished_at"]}
            for row in rows
        ]
    except sqlite3.Error as exc:
        LOGGER.warning("throughput query failed: error=%s", exc)
        return None


@router.get("/queue/summary")
def queue_summary(request: Request):
    """返回队列摘要、worker 活动视图与配置快照。"""

    repo = get_repo(request)
    summary = repo.queue_summary()
    runtime = get_runtime(request)
    workers = getattr(request.app.state, "worker_view", {})
    if runtime is not None:
        workers = runtime.active_workers()
    summary["workers"] = workers
    summary["shutdown"] = runtime.shutdown_status() if runtime is not None else {}

    # SPEC G：throughput = 最近 5 分钟完成数/阶段（不再返回空 dict）。
    now = datetime.now(timezone.utc)
    stages = sorted(
        set(SUPPORTED_STAGES)
        | set((summary.get("stages") or {}).keys())
    )
    rows = _throughput_rows_from_repo(repo, window_sec=THROUGHPUT_WINDOW_SEC, now=now)
    if rows is None:
        summary["throughput"] = {"available": False, "window_sec": THROUGHPUT_WINDOW_SEC}
    else:
        per_stage = compute_throughput(rows, stages=stages, now=now)
        summary["throughput"] = {
            "available": True,
            "window_sec": THROUGHPUT_WINDOW_SEC,
            "stages": per_stage,
        }
    return summary
