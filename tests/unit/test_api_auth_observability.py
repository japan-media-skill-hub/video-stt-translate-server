"""应用层暴露面鉴权测试（SPEC H 的 app.py 部分）。

覆盖：
- 历史 `api_token` 参数与新增 `settings` 参数的优先级；
- token 常量时间比较（core.security.tokens_match / hmac.compare_digest）；
- docs 关闭时 /docs、/redoc、/openapi.json 返回 404 且不被鉴权改写成 401；
- docs 开启时仍受 token 保护；
- /healthz、/readyz、/metrics 不享受任何鉴权豁免；
- 鉴权失败日志只含来源 IP 与 request_id，不含 token 明文。
"""

from __future__ import annotations

import hmac
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from whisper_stt_service.api import app as app_module
from whisper_stt_service.api.app import create_app, resolve_security_policy
from whisper_stt_service.core.security import tokens_match
from whisper_stt_service.repo.database import Database
from whisper_stt_service.repo.job_repository import JobRepository


GOOD_TOKEN = "good-token-0123456789"
DOC_PATHS = ("/docs", "/redoc", "/openapi.json")


def _repo(tmp_path) -> JobRepository:
    """构造独立 SQLite 仓储，避免多个 app 共用 ./tmp/api-test.db。"""

    db = Database(tmp_path / "app.db")
    db.init_schema()
    return JobRepository(db, log_root=tmp_path / "logs", artifact_root=tmp_path / "artifacts")


def _settings(
    *,
    token: str,
    docs_enabled: bool,
    require_token: bool = True,
    tmp_path: Path | None = None,
):
    """构造 SecuritySettings 兼容的 settings 替身（runtime 段供 readyz 使用）。"""

    runtime = None
    if tmp_path is not None:
        # 真实存在的模型目录与 artifact_root，保证 readyz 能返回 200。
        model_dir = tmp_path / "model"
        model_dir.mkdir(exist_ok=True)
        # 模型产物真实存在才算就绪（空目录会被 readyz 判 503）。
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        runtime = SimpleNamespace(
            artifact_root=tmp_path / "artifacts", model_path=model_dir
        )
    return SimpleNamespace(
        security=SimpleNamespace(
            api_token=token,
            require_token=require_token,
            allow_placeholder_token=False,
            docs_enabled=docs_enabled,
        ),
        runtime=runtime,
        metrics_enabled=True,
    )


# ------------------------------------------------------------------ 策略解析


def test_resolve_security_policy_prefers_settings() -> None:
    """settings.security 存在时优先于 api_token 参数。"""

    settings = SimpleNamespace(
        security=SimpleNamespace(
            api_token="settings-token",
            require_token=True,
            allow_placeholder_token=False,
            docs_enabled=True,
        )
    )
    policy = resolve_security_policy(settings, "legacy-token")
    assert policy["api_token"] == "settings-token"
    assert policy["docs_enabled"] is True
    assert policy["require_token"] is True
    assert policy["source"] == "settings"


def test_resolve_security_policy_legacy_api_token_keeps_compat() -> None:
    """不传 settings 时沿用 api_token 参数；空 token 表示鉴权关闭。"""

    policy = resolve_security_policy(None, "token-123")
    assert policy["api_token"] == "token-123"
    assert policy["source"] == "api_token"
    # 历史行为：不要求 token、不校验占位值。
    assert policy["require_token"] is False
    assert policy["allow_placeholder_token"] is True
    assert policy["docs_enabled"] is False

    empty = resolve_security_policy(None, None)
    assert empty["api_token"] == ""
    assert empty["source"] == "none"


# ------------------------------------------------------------------ 常量时间


def test_tokens_match_uses_constant_time_compare(monkeypatch) -> None:
    """鉴权比较必须走 hmac.compare_digest（打桩验证调用点真实生效）。"""

    calls: list[tuple[bytes, bytes]] = []
    real_compare = hmac.compare_digest

    def _spy(left, right):
        """记录 compare_digest 入参，再委托给真实实现。"""

        calls.append((left, right))
        return real_compare(left, right)

    monkeypatch.setattr(hmac, "compare_digest", _spy)
    assert tokens_match("abc", "abc") is True
    assert tokens_match("abc", "abd") is False
    # 空 token 一律不匹配，且不会短路成 True。
    assert tokens_match("", "") is False
    assert tokens_match(None, "abc") is False
    assert len(calls) == 2
    assert calls[0] == (b"abc", b"abc")


def test_auth_uses_tokens_match_helper(monkeypatch) -> None:
    """app 的鉴权守卫复用 core.security.tokens_match，不另写比较逻辑。"""

    seen: list[tuple[str | None, str | None]] = []

    def _spy(provided, expected):
        """记录每次比较的入参，再委托给原实现。"""

        seen.append((provided, expected))
        return provided == expected

    monkeypatch.setattr(app_module, "tokens_match", _spy)
    client = TestClient(create_app(api_token="token-123"))
    assert client.get("/jobs/x/progress", headers={"X-API-Token": "token-123"}).status_code == 404
    assert client.get("/jobs/x/progress", headers={"X-API-Token": "nope"}).status_code == 401
    assert seen == [("token-123", "token-123"), ("nope", "token-123")]


# ------------------------------------------------------------- 历史参数兼容


@pytest.mark.parametrize(
    "path,status",
    [
        ("/healthz", 200),
        ("/metrics", 200),
    ],
)
def test_no_token_means_auth_disabled(tmp_path, path: str, status: int) -> None:
    """未配置 token 时鉴权关闭（记录 WARN），服务照常可用。"""

    client = TestClient(create_app(repo=_repo(tmp_path)))
    assert client.get(path).status_code == status


def test_legacy_api_token_still_protects_business_routes(tmp_path) -> None:
    """只传 api_token 参数的旧调用方式行为不变。"""

    client = TestClient(create_app(repo=_repo(tmp_path), api_token="token-123"))
    assert client.get("/queue/summary").status_code == 401
    assert (
        client.get("/queue/summary", headers={"X-API-Token": "token-123"}).status_code
        == 200
    )


# --------------------------------------------------------------------- docs


def test_docs_disabled_returns_404_not_401(tmp_path) -> None:
    """docs 关闭时即使不带 token 也必须是 404（不能被鉴权提前改写成 401）。"""

    settings = _settings(token=GOOD_TOKEN, docs_enabled=False)
    client = TestClient(create_app(repo=_repo(tmp_path), settings=settings))
    for path in DOC_PATHS:
        assert client.get(path).status_code == 404
        # 带 token 也是 404：docs 已真正关闭，而不是"隐藏"。
        assert (
            client.get(path, headers={"X-API-Token": GOOD_TOKEN}).status_code == 404
        )
    assert client.get("/docs/oauth2-redirect").status_code == 404


def test_docs_enabled_requires_token(tmp_path) -> None:
    """docs 开启时同样受 token 保护：无 token 401，带 token 200。"""

    settings = _settings(token=GOOD_TOKEN, docs_enabled=True)
    client = TestClient(create_app(repo=_repo(tmp_path), settings=settings))
    for path in DOC_PATHS:
        assert client.get(path).status_code == 401
        assert (
            client.get(path, headers={"X-API-Token": GOOD_TOKEN}).status_code == 200
        )
    assert client.get("/docs/oauth2-redirect").status_code == 401


def test_docs_enabled_without_token_is_open(tmp_path) -> None:
    """鉴权关闭 + docs 开启：文档公开（安全默认值由 require_token 启动校验保证）。"""

    settings = SimpleNamespace(security=SimpleNamespace(api_token="", docs_enabled=True))
    client = TestClient(create_app(repo=_repo(tmp_path), settings=settings))
    assert client.get("/openapi.json").status_code == 200


# --------------------------------------------------- 观测接口没有鉴权豁免


def test_observability_endpoints_are_not_exempt_from_auth(tmp_path) -> None:
    """SPEC 全局安全：healthz/readyz/metrics 与业务接口同级鉴权。"""

    settings = _settings(
        token=GOOD_TOKEN, docs_enabled=False, require_token=True, tmp_path=tmp_path
    )
    client = TestClient(create_app(repo=_repo(tmp_path), settings=settings))
    for path in ("/healthz", "/readyz", "/metrics"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"X-API-Token": "wrong"}).status_code == 401
        assert client.get(path, headers={"X-API-Token": GOOD_TOKEN}).status_code == 200


# -------------------------------------------------------------------- 日志


def test_auth_failure_log_contains_ip_and_request_id_without_token(
    tmp_path, caplog
) -> None:
    """鉴权失败日志带来源 IP 与 request_id；绝不出现 token 明文。"""

    client = TestClient(create_app(repo=_repo(tmp_path), api_token="super-secret-token"))
    with caplog.at_level(
        logging.WARNING, logger="whisper_stt_service.api.app"
    ):
        resp = client.get(
            "/queue/summary",
            headers={"X-API-Token": "super-secret-token-but-wrong", "X-Request-ID": "rid-42"},
        )
    assert resp.status_code == 401
    messages = [record.getMessage() for record in caplog.records]
    assert any("api auth rejected" in message for message in messages)
    joined = "\n".join(messages)
    assert "rid-42" in joined
    assert "client_ip=" in joined
    assert "reason=token_mismatch" in joined
    assert "super-secret-token" not in joined


def test_request_id_is_echoed_and_propagated_to_logs(tmp_path, caplog) -> None:
    """request_id：透传 X-Request-ID、响应头回写、日志包含。"""

    client = TestClient(create_app(repo=_repo(tmp_path)))
    with caplog.at_level(logging.INFO, logger="whisper_stt_service"):
        resp = client.get("/queue/summary", headers={"X-Request-ID": "rid-abc-1"})
    assert resp.status_code == 200
    assert resp.headers["X-Request-ID"] == "rid-abc-1"
    assert any("rid-abc-1" in r.getMessage() for r in caplog.records)

    generated = client.get("/queue/summary")
    assert len(generated.headers["X-Request-ID"]) == 32


def test_unauthorized_response_also_carries_request_id(tmp_path) -> None:
    """401 响应同样带 request_id（request_id 中间件位于鉴权中间件外层）。"""

    client = TestClient(create_app(repo=_repo(tmp_path), api_token="token-123"))
    resp = client.get("/queue/summary", headers={"X-Request-ID": "rid-denied"})
    assert resp.status_code == 401
    assert resp.headers["X-Request-ID"] == "rid-denied"


def test_health_payload_never_leaks_token(tmp_path) -> None:
    """健康检查响应里不出现任何 token/密钥字段。"""

    settings = _settings(token=GOOD_TOKEN, docs_enabled=False)
    client = TestClient(create_app(repo=_repo(tmp_path), settings=settings))
    headers = {"X-API-Token": GOOD_TOKEN}
    for path in ("/healthz", "/readyz"):
        body = client.get(path, headers=headers).text
        assert GOOD_TOKEN not in body
        assert "token" not in body.lower()