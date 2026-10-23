"""SPEC G：request_id 从真实 HTTP 请求贯穿 enqueue 与 task 日志。

关键点：不能只断言响应头回显——必须证明 request_id 真正落到了
task_config.metadata 与仓储侧写入的 task 日志里。
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from whisper_stt_service.api import create_app
from whisper_stt_service.repo.database import Database
from whisper_stt_service.repo.job_repository import JobRepository


def _build(tmp_path: Path) -> tuple[JobRepository, TestClient]:
    """构造独立 repo 与客户端（视频文件真实存在，满足 API 严格校验）。"""

    db = Database(tmp_path / "q.db")
    db.init_schema()
    repo = JobRepository(
        db, log_root=tmp_path / "logs", artifact_root=tmp_path / "artifacts"
    )
    return repo, TestClient(create_app(repo=repo))


def _video(tmp_path: Path) -> Path:
    """创建满足扩展名白名单的真实视频文件。"""

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"fake-video")
    return path


def _task_rows(repo: JobRepository, job_id: str) -> list[dict]:
    """读取任务的原始行（含 log_file，get_job_detail 不返回该列）。"""

    with repo.db.connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT task_id, stage, log_file, task_config_json FROM tasks WHERE job_id=? ORDER BY stage",
                (job_id,),
            ).fetchall()
        ]


def test_request_id_propagates_to_metadata_and_task_log(tmp_path: Path) -> None:
    """带 X-Request-ID 的真实请求：响应头、task metadata、task 日志三处一致。"""

    repo, client = _build(tmp_path)
    supplied = "req-e2e-0123456789abcdef"
    resp = client.post(
        "/jobs",
        json={"video_path": str(_video(tmp_path)), "language": "ja"},
        headers={"X-Request-ID": supplied},
    )
    assert resp.status_code == 200, resp.text
    # 1) 响应头回显。
    assert resp.headers.get("X-Request-ID") == supplied

    job_id = resp.json()["job_id"]
    rows = _task_rows(repo, job_id)
    assert rows

    # 2) task_config.metadata 固化 request_id。
    for row in rows:
        cfg = json.loads(row["task_config_json"])
        assert cfg["metadata"]["request_id"] == supplied

    # 3) 仓储侧写入的 task 日志带同一 request_id（而不仅是响应头）。
    for row in rows:
        repo._append_task_log_event(
            job_id=job_id,
            task_id=row["task_id"],
            stage=row["stage"],
            event="task_linked",
            result="queued",
        )
        payload = json.loads(
            Path(row["log_file"]).read_text(encoding="utf-8").splitlines()[-1]
        )
        assert payload["request_id"] == supplied


def test_generated_request_id_when_header_absent(tmp_path: Path) -> None:
    """未传 X-Request-ID 时由中间件生成，并同样落到 metadata。"""

    repo, client = _build(tmp_path)
    resp = client.post(
        "/jobs", json={"video_path": str(_video(tmp_path)), "language": "ja"}
    )
    assert resp.status_code == 200
    generated = resp.headers.get("X-Request-ID")
    assert generated

    rows = _task_rows(repo, resp.json()["job_id"])
    for row in rows:
        cfg = json.loads(row["task_config_json"])
        assert cfg["metadata"]["request_id"] == generated


def test_each_request_gets_distinct_request_id(tmp_path: Path) -> None:
    """两个不同文件的两个请求应得到各自的 request_id，不会串号。"""

    repo, client = _build(tmp_path)
    first_video = tmp_path / "a.mp4"
    second_video = tmp_path / "b.mp4"
    first_video.write_bytes(b"a")
    second_video.write_bytes(b"b")

    r1 = client.post("/jobs", json={"video_path": str(first_video), "language": "ja"})
    r2 = client.post("/jobs", json={"video_path": str(second_video), "language": "ja"})
    id1, id2 = r1.headers["X-Request-ID"], r2.headers["X-Request-ID"]
    assert id1 != id2

    for job_id, expected in ((r1.json()["job_id"], id1), (r2.json()["job_id"], id2)):
        cfg = json.loads(_task_rows(repo, job_id)[0]["task_config_json"])
        assert cfg["metadata"]["request_id"] == expected


def test_invalid_header_is_sanitized_and_still_propagates(tmp_path: Path) -> None:
    """非法 X-Request-ID 被中间件替换为新 id，且链路依然一致。"""

    repo, client = _build(tmp_path)
    resp = client.post(
        "/jobs",
        json={"video_path": str(_video(tmp_path)), "language": "ja"},
        headers={"X-Request-ID": "bad id with spaces\n"},
    )
    assert resp.status_code == 200
    effective = resp.headers.get("X-Request-ID")
    assert effective and effective != "bad id with spaces\n"

    cfg = json.loads(_task_rows(repo, resp.json()["job_id"])[0]["task_config_json"])
    assert cfg["metadata"]["request_id"] == effective
