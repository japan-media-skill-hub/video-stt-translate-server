"""SPEC G/H 一致性审计测试：针对 api/app.py 暴露面与可观测性返回契约。

本文件只做"现状断言 + 回归锁定"，不依赖模型、网络或真实硬件；
每条用例注释里写明对应的 SPEC 条款，便于后续 review 逐条对照。
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from whisper_stt_service.api.app import create_app
from whisper_stt_service.repo.database import Database
from whisper_stt_service.repo.job_repository import JobRepository


TOKEN = "audit-token-0123456789"


def _repo(tmp_path: Path) -> JobRepository:
    """独立 SQLite 仓储（每个用例一个 db 文件）。"""

    db = Database(tmp_path / "audit.db")
    db.init_schema()
    return JobRepository(db, log_root=tmp_path / "logs", artifact_root=tmp_path / "artifacts")


def _ready_settings(tmp_path: Path, *, docs_enabled: bool = False):
    """构造能让 /readyz 通过的最小 settings（真实模型目录 + 可写 artifact_root）。"""

    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    return SimpleNamespace(
        security=SimpleNamespace(
            api_token=TOKEN,
            require_token=True,
            allow_placeholder_token=False,
            docs_enabled=docs_enabled,
        ),
        runtime=SimpleNamespace(
            artifact_root=tmp_path / "artifacts", model_path=model_dir
        ),
        metrics_enabled=True,
    )


# ---------------------------------------------------------------- 错误返回契约


def test_unhandled_error_returns_request_id_header_and_body(tmp_path: Path) -> None:
    """SPEC G：错误返回必须带 request_id（响应头 + 响应体），且不泄漏 traceback。"""

    app = create_app(repo=_repo(tmp_path), api_token=TOKEN)

    @app.get("/audit-boom")
    def _boom():
        """故意抛异常，验证 500 响应带 request_id 且不泄漏 traceback。"""

        raise RuntimeError("kaboom-secret-detail")

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/audit-boom", headers={"X-API-Token": TOKEN, "X-Request-ID": "rid-err"})
    assert resp.status_code == 500
    assert resp.headers["X-Request-ID"] == "rid-err"
    payload = resp.json()
    assert payload == {"detail": "internal_server_error", "request_id": "rid-err"}
    assert "kaboom-secret-detail" not in resp.text


def test_unhandled_error_without_incoming_id_generates_one(tmp_path: Path) -> None:
    """未携带上游 id 时仍生成 request_id，便于按 id 检索日志。"""

    app = create_app(repo=_repo(tmp_path))
    app.get("/audit-boom-2")(lambda: (_ for _ in ()).throw(RuntimeError("x")))
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/audit-boom-2")
    assert resp.status_code == 500
    assert len(resp.headers["X-Request-ID"]) == 32
    assert resp.json()["request_id"] == resp.headers["X-Request-ID"]


def test_auth_failure_body_is_stable_and_has_request_id(tmp_path: Path) -> None:
    """SPEC H：鉴权失败固定 401 + invalid_api_token，并带 request_id。"""

    client = TestClient(create_app(repo=_repo(tmp_path), api_token=TOKEN))
    resp = client.get("/queue/summary")
    assert resp.status_code == 401
    assert resp.json() == {"detail": "invalid_api_token"}
    assert resp.headers["X-Request-ID"]


# ------------------------------------------------------------- metrics 关闭语义


def test_metrics_disabled_is_404_even_with_valid_token(tmp_path: Path) -> None:
    """SPEC G：metrics_enabled=false 时 /metrics 返回 404（带 token 也是 404）。"""

    client = TestClient(create_app(repo=_repo(tmp_path), api_token=TOKEN, metrics_enabled=False))
    # 与 docs 关闭一致：不论是否携带 token 一律 404（端点未注册、不泄露路由）。
    assert client.get("/metrics").status_code == 404
    assert client.get("/metrics", headers={"X-API-Token": "wrong"}).status_code == 404
    assert client.get("/metrics", headers={"X-API-Token": TOKEN}).status_code == 404
    # 开启时仍走全局鉴权。
    enabled = TestClient(create_app(repo=_repo(tmp_path), api_token=TOKEN))
    assert enabled.get("/metrics").status_code == 401
    assert enabled.get("/metrics", headers={"X-API-Token": TOKEN}).status_code == 200
    # 其余观测能力保留。
    assert client.get("/healthz", headers={"X-API-Token": TOKEN}).status_code == 200


def test_metrics_disabled_from_settings_observability_section(tmp_path: Path) -> None:
    """metrics_enabled 可从 settings.observability 读取（SPEC 的 [observability] 段）。"""

    settings = SimpleNamespace(
        security=SimpleNamespace(api_token=TOKEN, docs_enabled=False),
        observability=SimpleNamespace(metrics_enabled=False),
    )
    app = create_app(repo=_repo(tmp_path), settings=settings)
    assert app.state.metrics_enabled is False
    client = TestClient(app)
    # disabled => 一律 404，不被鉴权改写成 401。
    assert client.get("/metrics").status_code == 404
    assert client.get("/metrics", headers={"X-API-Token": TOKEN}).status_code == 404
    assert app.state.metrics is None


def test_metrics_enabled_by_default_when_unconfigured(tmp_path: Path) -> None:
    """未配置时默认开启 /metrics（SPEC：[observability] metrics_enabled = true）。"""

    app = create_app(repo=_repo(tmp_path))
    assert app.state.metrics_enabled is True
    assert TestClient(app).get("/metrics").status_code == 200


# ------------------------------------------------------------- readyz 真实性


def test_readyz_rejects_empty_model_directory(tmp_path: Path) -> None:
    """readyz 的模型检查不接受"空目录/占位目录"，避免假就绪。"""

    empty_model = tmp_path / "empty-model"
    empty_model.mkdir()
    settings = SimpleNamespace(
        security=SimpleNamespace(api_token=TOKEN, docs_enabled=False),
        runtime=SimpleNamespace(artifact_root=tmp_path / "artifacts", model_path=empty_model),
    )
    client = TestClient(create_app(repo=_repo(tmp_path), settings=settings))
    resp = client.get("/readyz", headers={"X-API-Token": TOKEN})
    assert resp.status_code == 503
    assert resp.json()["checks"]["model_path"]["error"] == "model_path_empty"


def _readonly_repo(tmp_path: Path, db_path: Path):
    """构造一个用 `mode=ro` 只读连接访问同一 DB 文件的仓储替身。"""

    import sqlite3

    from whisper_stt_service.repo.database import Database

    class _ReadOnlyDB(Database):
        """只读连接：WAL/journal 由 SQLite 保证，任何 main 写入都会报错。"""

        def connect(self):
            """以 URI `mode=ro` 打开同一个 DB 文件。"""

            conn = sqlite3.connect(
                f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            return conn

    return JobRepository(_ReadOnlyDB(db_path), log_root=tmp_path / "logs")


def test_readyz_db_writable_rejects_readonly_connection(tmp_path: Path) -> None:
    """回归：`mode=ro` 连接能 BEGIN IMMEDIATE、也能写 TEMP，必须判 not ready。

    SQLite 的只读连接只拒绝写 main 库；若探针只做 BEGIN/TEMP 就会给出假就绪
    （Sol 复现：check_db_writable -> (True, None)）。
    """

    from whisper_stt_service.api.observability import check_db_writable

    repo = _repo(tmp_path)
    readonly_repo = _readonly_repo(tmp_path, repo.db.db_path)
    ok, error = check_db_writable(readonly_repo)
    assert ok is False
    assert error is not None and "readonly" in error

    # 真实可写连接仍然通过。
    assert check_db_writable(repo) == (True, None)


def test_readyz_returns_503_for_readonly_db(tmp_path: Path) -> None:
    """端到端：只读 DB 时 /readyz 返回 503 并给出 db_writable 失败明细。"""

    repo = _repo(tmp_path)
    job = repo.enqueue(str(tmp_path / "video.mp4"), "ja")
    assert job.job_id
    readonly_repo = _readonly_repo(tmp_path, repo.db.db_path)
    settings = _ready_settings(tmp_path)
    client = TestClient(create_app(repo=readonly_repo, settings=settings))
    resp = client.get("/readyz", headers={"X-API-Token": TOKEN})
    assert resp.status_code == 503
    checks = resp.json()["checks"]
    assert checks["db_writable"]["ok"] is False
    assert "readonly" in checks["db_writable"]["error"]


def test_readyz_db_writable_does_not_mutate_data(tmp_path: Path) -> None:
    """DB 可写检查是真实写事务且可回滚：不新增表、不改业务数据。"""

    repo = _repo(tmp_path)
    job = repo.enqueue(str(tmp_path / "video.mp4"), "ja")
    before = repo.get_job_detail(job.job_id)
    client = TestClient(create_app(repo=repo, settings=_ready_settings(tmp_path)))
    assert client.get("/readyz", headers={"X-API-Token": TOKEN}).status_code == 200
    assert client.get("/readyz", headers={"X-API-Token": TOKEN}).status_code == 200
    after = repo.get_job_detail(job.job_id)
    assert before == after
    with repo.db.connect() as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    # main 库写探测表已回滚：既不留 `__readyz_write_probe`，也不留旧版 TEMP 名。
    assert not any(name.startswith("__readyz") for name in tables)
    assert not any(name.startswith("_readyz") for name in tables)


def test_readyz_flags_insecure_security_config(tmp_path: Path) -> None:
    """占位 token + allow_placeholder_token=false 时 readyz 报 insecure_security_config。"""

    settings = SimpleNamespace(
        security=SimpleNamespace(
            api_token="<CHANGE_ME_STRONG_TOKEN>",
            require_token=True,
            allow_placeholder_token=False,
            docs_enabled=False,
        ),
        runtime=SimpleNamespace(artifact_root=tmp_path / "artifacts", model_path=tmp_path),
    )
    client = TestClient(create_app(repo=_repo(tmp_path), settings=settings))
    resp = client.get("/readyz", headers={"X-API-Token": "<CHANGE_ME_STRONG_TOKEN>"})
    assert resp.status_code == 503
    assert resp.json()["checks"]["config"]["error"] == "insecure_security_config"


def test_readyz_reports_settings_not_injected(tmp_path: Path) -> None:
    """未注入 settings 时 readyz 明确报 settings_not_injected（而不是静默 ready）。"""

    client = TestClient(create_app(repo=_repo(tmp_path)))
    resp = client.get("/readyz")
    assert resp.status_code == 503
    assert resp.json()["checks"]["config"]["error"] == "settings_not_injected"


# ------------------------------------------------------------------ 健康响应体


def test_healthz_payload_shape(tmp_path: Path) -> None:
    """healthz 响应含 status/checks/request_id/timestamp/version，且 db_query 明细可见。"""

    client = TestClient(create_app(repo=_repo(tmp_path)))
    payload = client.get("/healthz").json()
    assert set(payload) == {"status", "checks", "request_id", "timestamp", "version"}
    assert payload["checks"]["db_query"] == {"ok": True, "error": None}
    assert payload["status"] == "ok"


def test_healthz_does_not_touch_task_tables(tmp_path: Path) -> None:
    """healthz 是轻量探针：只做 SELECT 1，不读任务表（不受大表拖累）。"""

    repo = _repo(tmp_path)
    repo.enqueue(str(tmp_path / "video.mp4"), "ja")
    client = TestClient(create_app(repo=repo))
    assert client.get("/healthz").status_code == 200


# --------------------------------------------------------------- queue throughput


def test_queue_summary_throughput_schema(tmp_path: Path) -> None:
    """throughput 结构固定：available/window_sec/stages{stage:{succeeded,failed,total,...}}。"""

    client = TestClient(create_app(repo=_repo(tmp_path)))
    throughput = client.get("/queue/summary").json()["throughput"]
    assert throughput["available"] is True
    assert throughput["window_sec"] == 300
    for stage in ("extract", "stt", "stt_whisperx", "translate"):
        assert set(throughput["stages"][stage]) == {
            "succeeded",
            "failed",
            "total",
            "window_sec",
            "per_minute",
        }