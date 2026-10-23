"""批次1 API：by-path 路由顺序修复与 retry/cancel 状态机测试。"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from whisper_stt_service.api import create_app
from whisper_stt_service.db import Database
from whisper_stt_service.repository import JobRepository


def _client(tmp_path: Path) -> tuple[TestClient, JobRepository]:
    """构造使用临时库的测试客户端与仓储。"""

    db = Database(tmp_path / "q.db")
    db.init_schema()
    repo = JobRepository(db, log_root=tmp_path / "logs", artifact_root=tmp_path / "artifacts")
    return TestClient(create_app(repo=repo)), repo


def test_by_path_returns_latest_job(tmp_path: Path) -> None:
    """B.1：POST /jobs 后 GET /jobs/by-path 必须 200 且 job_id 一致。"""

    client, repo = _client(tmp_path)
    created = repo.enqueue(video_path="/tmp/by-path.mp4", language="ja")

    resp = client.get("/jobs/by-path", params={"video_path": "/tmp/by-path.mp4"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["video_path"] == "/tmp/by-path.mp4"
    assert body["item"]["job_id"] == created.job_id


def test_by_path_unknown_video_returns_null_item(tmp_path: Path) -> None:
    """B.1：未知路径返回 200 + item=None（不是 404 job_not_found）。"""

    client, _repo = _client(tmp_path)

    resp = client.get("/jobs/by-path", params={"video_path": "/tmp/none.mp4"})

    assert resp.status_code == 200
    assert resp.json()["item"] is None


def test_retry_endpoint_state_machine(tmp_path: Path) -> None:
    """A.7：queued job retry=409；failed job retry=200 并回到 queued。"""

    client, repo = _client(tmp_path)
    created = repo.enqueue(video_path="/tmp/retry-api.mp4", language="ja")

    not_allowed = client.post(f"/jobs/{created.job_id}/retry")
    assert not_allowed.status_code == 409
    assert not_allowed.json()["detail"] == "retry_not_allowed"

    # 制造 failed：领取后超预算失败。
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None
    with repo.db.tx() as conn:
        conn.execute("UPDATE tasks SET attempt=2 WHERE task_id=?", (claimed.task_id,))
    repo.mark_task_failed(claimed.task_id, "boom", worker_id="w1", attempt=2)
    assert client.get(f"/jobs/{created.job_id}").json()["status"] == "failed"

    ok = client.post(f"/jobs/{created.job_id}/retry")
    assert ok.status_code == 200
    assert ok.json()["retried_tasks"] == 3
    assert client.get(f"/jobs/{created.job_id}").json()["status"] == "queued"


def test_retry_endpoint_unknown_job_returns_404(tmp_path: Path) -> None:
    """A.7：不存在的 job 返回 404。"""

    client, _repo = _client(tmp_path)

    resp = client.post("/jobs/job-missing/retry")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "job_not_found"


def test_cancel_endpoint_state_machine(tmp_path: Path) -> None:
    """A.7：cancel 仅允许未开始的 job；成功后任务为 cancelled。"""

    client, repo = _client(tmp_path)
    created = repo.enqueue(video_path="/tmp/cancel-api.mp4", language="ja")

    resp = client.post(f"/jobs/{created.job_id}/cancel")

    assert resp.status_code == 200
    assert resp.json()["cancelled_tasks"] == 3
    detail = client.get(f"/jobs/{created.job_id}").json()
    assert {t["status"] for t in detail["tasks"]} == {"cancelled"}
    # SPEC 6：jobs.status 集合保持不变。
    assert detail["status"] in {"queued", "running", "succeeded", "failed"}

    # 已取消的 job 不能再次取消。
    assert client.post(f"/jobs/{created.job_id}/cancel").status_code == 409
    assert client.post("/jobs/job-missing/cancel").status_code == 404


def test_retry_rejects_historical_invalid_dag_with_409(tmp_path: Path) -> None:
    """A.7/SPEC 6：历史非法 DAG 的 retry 返回 409 + 原因。"""

    client, repo = _client(tmp_path)
    created = repo.enqueue(video_path="/tmp/retry-bad-dag-api.mp4", language="ja")
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE jobs SET status='failed', dag_json=? WHERE job_id=?",
            ('{"version": 1, "stages": "broken"}', created.job_id),
        )

    resp = client.post(f"/jobs/{created.job_id}/retry")

    assert resp.status_code == 409
    assert resp.json()["detail"].startswith("retry_job_dag_invalid")
