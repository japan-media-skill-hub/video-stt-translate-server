"""配置与装配整合测试（SPEC H/G/D/C 的 config + bootstrap 部分）。

全部在 tmp_path 下用真实 config.ini + 真实 SQLite 跑 build_app，不加载模型、
不访问网络；需要模型的断言只检查路径/开关，不做推理。
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from whisper_stt_service.core.config import (
    ObservabilitySettings,
    SecuritySettings,
    Settings,
    TranslationSettings,
    load_settings,
    validate_startup_security,
)
from whisper_stt_service.core.security import SecurityError


GOOD_TOKEN = "integration-token-0123456789"


def _write_config(tmp_path: Path, *, model_path: Path | None = None, **overrides: str) -> Path:
    """写一份最小可启动配置，按需覆盖 section.key（形如 security.api_token）。"""

    model = model_path or (tmp_path / "model")
    model.mkdir(parents=True, exist_ok=True)
    if not any(model.iterdir()):
        (model / "config.json").write_text("{}", encoding="utf-8")
    values = {
        "security.api_token": GOOD_TOKEN,
        "security.allowed_video_roots": "",
        "security.allowed_copy_back_roots": "",
        "security.require_token": "true",
        "security.allow_placeholder_token": "false",
        "security.docs_enabled": "false",
        "observability.log_level": "WARNING",
        "observability.log_jsonl": "true",
        "observability.metrics_enabled": "true",
        "translation.parallel": "2",
        "translation.max_chars": "9000",
        "translation.checkpoint_enabled": "true",
        "translation.strict_ids": "false",
        "runtime.model_cache_size": "3",
        "runtime.model_parallelism": "2",
    }
    values.update(overrides)
    body = textwrap.dedent(
        f"""
        [llm]
        base_url = http://127.0.0.1:9/v1
        api_key = dummy-key
        model = dummy-model

        [security]
        api_token = {values["security.api_token"]}
        require_token = {values["security.require_token"]}
        allow_placeholder_token = {values["security.allow_placeholder_token"]}
        docs_enabled = {values["security.docs_enabled"]}
        allowed_video_roots = {values["security.allowed_video_roots"]}
        allowed_copy_back_roots = {values["security.allowed_copy_back_roots"]}

        [observability]
        log_level = {values["observability.log_level"]}
        log_jsonl = {values["observability.log_jsonl"]}
        metrics_enabled = {values["observability.metrics_enabled"]}

        [translation]
        parallel = {values["translation.parallel"]}
        max_chars = {values["translation.max_chars"]}
        chunk_minutes = 20
        retry = 10
        timeout = 240
        checkpoint_enabled = {values["translation.checkpoint_enabled"]}
        strict_ids = {values["translation.strict_ids"]}

        [workers]
        extract_workers = 1
        stt_workers = 1
        stt_whisperx_workers = 1
        translate_workers = 1

        [timeouts]
        extract_timeout_sec = 60
        stt_timeout_sec = 60
        stt_whisperx_timeout_sec = 60
        translate_timeout_sec = 60
        lease_timeout_sec = 60

        [retry]
        extract_max_retries = 0
        stt_max_retries = 0
        stt_whisperx_max_retries = 0
        translate_max_retries = 0

        [runtime]
        db_path = {tmp_path}/db/app.db
        log_root = {tmp_path}/logs
        artifact_root = {tmp_path}/artifacts
        model_path = {model}
        model_cache_size = {values["runtime.model_cache_size"]}
        model_parallelism = {values["runtime.model_parallelism"]}
        """
    ).strip()
    path = tmp_path / "config.ini"
    path.write_text(body + "\n", encoding="utf-8")
    return path


def _build_app(monkeypatch, tmp_path: Path, **overrides: str):
    """设置 WHISPER_STT_CONFIG 并调用 build_app（返回 app）。"""

    monkeypatch.setenv("WHISPER_STT_CONFIG", str(_write_config(tmp_path, **overrides)))
    monkeypatch.delenv("WHISPER_STT_MODEL", raising=False)
    from whisper_stt_service.service.bootstrap import build_app

    return build_app()


# --------------------------------------------------------------- 配置默认值


def test_security_defaults_are_fail_closed(tmp_path: Path) -> None:
    """SecuritySettings 缺省即 fail-closed：要求 token、拒绝占位、关闭 docs、只绑回环。"""

    security = SecuritySettings(api_token="")
    assert security.require_token is True
    assert security.allow_placeholder_token is False
    assert security.docs_enabled is False
    assert security.bind_host == "127.0.0.1"
    assert security.bind_port == 18000
    assert security.allowed_video_roots == ()


def test_observability_and_translation_defaults() -> None:
    """G/C 默认值：INFO、非 JSONL、metrics 开启；翻译并发默认 2。"""

    obs = ObservabilitySettings()
    assert (obs.log_level, obs.log_jsonl, obs.metrics_enabled) == ("INFO", False, True)
    translation = TranslationSettings()
    assert translation.parallel == 2
    assert translation.max_chars == 12000
    assert translation.checkpoint_enabled is True
    assert translation.strict_ids is False
    assert translation.progress_ttl_sec == 604800


def test_settings_dataclass_keeps_old_construction_compatible() -> None:
    """旧调用方只传原 7 段即可构造 Settings（新段有默认值）。"""

    from tests.unit.test_stt_effective_config_logging import _fake_settings

    settings = _fake_settings()
    assert isinstance(settings, Settings)
    assert settings.observability.metrics_enabled is True
    assert settings.translation.parallel == 2


# ------------------------------------------------------------ config.ini 解析


def test_load_settings_reads_all_spec_keys(tmp_path: Path) -> None:
    """[security]/[observability]/[translation]/[runtime] 的 SPEC 键全部被解析。"""

    settings = load_settings(_write_config(tmp_path))
    assert settings.security.api_token == GOOD_TOKEN
    assert settings.security.require_token is True
    assert settings.security.docs_enabled is False
    assert settings.observability.log_level == "WARNING"
    assert settings.observability.log_jsonl is True
    assert settings.observability.metrics_enabled is True
    assert settings.translation.parallel == 2
    assert settings.translation.max_chars == 9000
    assert settings.runtime.model_cache_size == 3
    assert settings.runtime.model_parallelism == 2


def test_load_settings_old_config_without_new_sections(tmp_path: Path) -> None:
    """旧配置缺 [security]/[observability] 新键时按安全默认值加载（不报错）。"""

    path = tmp_path / "old.ini"
    path.write_text(
        textwrap.dedent(
            """
            [llm]
            base_url = http://127.0.0.1:9/v1
            api_key = k
            model = m

            [security]
            api_token =

            [workers]
            extract_workers = 1
            stt_workers = 1
            stt_whisperx_workers = 1
            translate_workers = 1

            [timeouts]
            extract_timeout_sec = 60
            stt_timeout_sec = 60
            stt_whisperx_timeout_sec = 60
            translate_timeout_sec = 60
            lease_timeout_sec = 60

            [retry]
            extract_max_retries = 0
            stt_max_retries = 0
            stt_whisperx_max_retries = 0
            translate_max_retries = 0

            [runtime]
            db_path = ./tmp/x.db
            log_root = ./tmp/logs
            model_path = /abs/model
            """
        ),
        encoding="utf-8",
    )
    settings = load_settings(path)
    assert settings.security.require_token is True
    assert settings.security.docs_enabled is False
    assert settings.observability == ObservabilitySettings()
    assert settings.translation.parallel == 2
    assert settings.runtime.model_cache_size == 2
    assert settings.runtime.model_parallelism == 1


def test_bind_port_falls_back_on_invalid_value(tmp_path: Path) -> None:
    """非法 bind_port（非数字/越界）回退 18000，保证启动脚本可用。"""

    path = _write_config(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        "docs_enabled = false", "docs_enabled = false\nbind_port = not-a-port"
    )
    path.write_text(text, encoding="utf-8")
    assert load_settings(path).security.bind_port == 18000


# ------------------------------------------------------------ 启动期安全校验


def test_validate_startup_security_rejects_empty_token() -> None:
    """require_token=true 且 token 为空 → api_token_missing。"""

    settings = SimpleSecuritySettings(api_token="", require_token=True)
    with pytest.raises(SecurityError) as excinfo:
        validate_startup_security(settings)
    assert excinfo.value.code == "api_token_missing"


def test_validate_startup_security_rejects_placeholder() -> None:
    """默认 allow_placeholder_token=false → 占位 token 拒绝启动。"""

    settings = SimpleSecuritySettings(api_token="<CHANGE_ME_STRONG_TOKEN>")
    with pytest.raises(SecurityError) as excinfo:
        validate_startup_security(settings)
    assert excinfo.value.code == "api_token_placeholder"


def test_validate_startup_security_allows_explicit_open_mode() -> None:
    """require_token=false 且 token 为空 → 不报错（调用方显式承担风险）。"""

    settings = SimpleSecuritySettings(api_token="", require_token=False)
    validate_startup_security(settings)


class SimpleSecuritySettings(Settings):
    """只覆盖 security 的 Settings 替身（其余段复用安全默认构造）。"""

    def __init__(self, *, api_token: str, require_token: bool = True) -> None:
        """构造仅用于安全校验的最小 Settings。"""

        base = _base_settings()
        object.__setattr__(self, "__dict__", dict(base.__dict__))
        object.__setattr__(
            self,
            "security",
            SecuritySettings(
                api_token=api_token,
                require_token=require_token,
                allow_placeholder_token=False,
                docs_enabled=False,
            ),
        )


def _base_settings() -> Settings:
    """从最小 ini 构造一份完整 Settings（供安全校验测试改写 security）。"""

    from tests.unit.test_stt_effective_config_logging import _fake_settings

    return _fake_settings()


def test_build_app_refuses_to_start_with_placeholder_token(monkeypatch, tmp_path) -> None:
    """占位 token → build_app 直接抛 SecurityError，不会带着示例口令启动。"""

    monkeypatch.setenv(
        "WHISPER_STT_CONFIG",
        str(_write_config(tmp_path, **{"security.api_token": "<CHANGE_ME_STRONG_TOKEN>"})),
    )
    from whisper_stt_service.service.bootstrap import build_app

    with pytest.raises(SecurityError) as excinfo:
        build_app()
    assert excinfo.value.code == "api_token_placeholder"


def test_build_app_refuses_empty_token_when_required(monkeypatch, tmp_path) -> None:
    """require_token=true + 空 token → 拒绝启动（默认值不允许悄悄放开鉴权）。"""

    monkeypatch.setenv(
        "WHISPER_STT_CONFIG",
        str(_write_config(tmp_path, **{"security.api_token": "", "security.require_token": "true"})),
    )
    from whisper_stt_service.service.bootstrap import build_app

    with pytest.raises(SecurityError) as excinfo:
        build_app()
    assert excinfo.value.code == "api_token_missing"


def test_build_app_allows_explicit_no_auth(monkeypatch, tmp_path) -> None:
    """显式 require_token=false + 空 token 时允许启动（WARN），接口无鉴权。"""

    app = _build_app(
        monkeypatch,
        tmp_path,
        **{"security.api_token": "", "security.require_token": "false"},
    )
    from fastapi.testclient import TestClient

    client = TestClient(app)
    assert client.get("/healthz").status_code == 200
    assert app.state.security_policy["api_token"] == ""


# ------------------------------------------------------------ bootstrap 装配


def test_build_app_wires_settings_metrics_and_cache(monkeypatch, tmp_path) -> None:
    """build_app 把 settings 传给 create_app，并把 metrics/缓存配置接进运行时。"""

    from whisper_stt_service.service.model_cache import get_model_cache_stats

    app = _build_app(monkeypatch, tmp_path)
    assert app.state.settings is not None
    assert app.state.docs_enabled is False
    assert app.state.metrics_enabled is True
    assert app.state.metrics is not None
    stats = get_model_cache_stats()["config"]
    assert stats["cache_size"] == 3
    assert stats["parallelism"] == 2
    # 运行时持有同一 MetricsState 实例（可直接从 repo/属性读取，避免依赖私有名）。
    runtime_metrics = getattr(app.state.runtime, "metrics", None) or getattr(
        app.state.runtime, "_metrics", None
    )
    assert runtime_metrics is app.state.metrics


def test_build_app_applies_translation_defaults_to_stage_config(monkeypatch, tmp_path) -> None:
    """C：stage 默认值来自 [translation]（而不是写死的 30/4）。"""

    app = _build_app(monkeypatch, tmp_path, **{"translation.parallel": "3"})
    translate_defaults = app.state.repo.stage_effective_defaults["translate"]
    assert translate_defaults["parallel"] == 3
    assert translate_defaults["max_chars"] == 9000
    assert translate_defaults["checkpoint_enabled"] is True
    assert translate_defaults["timeout_sec"] == 240


def test_build_app_metrics_disabled_is_404(monkeypatch, tmp_path) -> None:
    """G：metrics_enabled=false → /metrics 404，健康检查仍可用。"""

    from fastapi.testclient import TestClient

    app = _build_app(
        monkeypatch, tmp_path, **{"observability.metrics_enabled": "false"}
    )
    service_log = tmp_path / "logs" / "service.log"
    # G：configure_logging 返回时 service.log 就必须已存在且可写，不能依赖
    # 后续是否产生日志记录（clean-CPU CI 上没有 tmpfs 的 WAL WARNING，
    # WARNING 等级下启动期可能一条落盘记录都没有）。
    assert service_log.exists()
    assert os.access(service_log, os.W_OK)
    client = TestClient(app)
    headers = {"X-API-Token": GOOD_TOKEN}
    assert app.state.metrics is None
    assert client.get("/metrics", headers=headers).status_code == 404
    assert client.get("/healthz", headers=headers).status_code == 200
    # 日志装配不受影响：service.log 已创建。
    assert (tmp_path / "logs" / "service.log").exists()


def test_configure_logging_creates_service_log_immediately(tmp_path: Path) -> None:
    """G：没有日志记录时，configure_logging 也必须立即创建可写的空 service.log。"""

    from whisper_stt_service.service.logging_config import (
        LOG_FILE_NAME,
        configure_logging,
    )

    log_dir = tmp_path / "logs"
    configure_logging(log_dir=log_dir, level="WARNING", jsonl=True)
    service_log = log_dir / LOG_FILE_NAME
    assert service_log.exists()
    # WARNING 等级下启动期没有落盘记录，文件应为空；且必须已可写。
    assert service_log.stat().st_size == 0
    with service_log.open("a", encoding="utf-8"):
        pass


def test_build_app_writes_jsonl_service_log(monkeypatch, tmp_path) -> None:
    """G：log_jsonl=true 时 service.log 为 JSON 行，字段含 request_id。"""

    import json
    import logging

    from fastapi.testclient import TestClient

    # 用 INFO 等级，保证访问日志（INFO）与请求日志都落盘。
    app = _build_app(monkeypatch, tmp_path, **{"observability.log_level": "INFO"})
    client = TestClient(app)
    # 用业务路径（/queue/summary）验证访问日志：健康探针刻意降级为 DEBUG 以免刷屏。
    client.get("/queue/summary", headers={"X-API-Token": GOOD_TOKEN, "X-Request-ID": "rid-log"})
    lines = (tmp_path / "logs" / "service.log").read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines if line.strip()]
    assert payloads, "service.log should not be empty"
    request_record = next(
        item for item in payloads if item["logger"] == "whisper_stt_service.api.observability"
    )
    # request_id 由 RequestMetadataMiddleware 写入 scope state、请求中间件绑定到
    # ContextVar，RequestIdFilter 在写日志时读取，必须与响应头一致。
    assert "rid-log" in request_record["message"]
    assert request_record["request_id"] == "rid-log"
    assert request_record["level"] == "INFO"


def test_build_app_docs_enabled_requires_token(monkeypatch, tmp_path) -> None:
    """H：docs 开启时 /docs 需要 token；关闭时 404（含带 token）。"""

    from fastapi.testclient import TestClient

    app = _build_app(monkeypatch, tmp_path, **{"security.docs_enabled": "true"})
    client = TestClient(app)
    assert client.get("/docs").status_code == 401
    assert client.get("/docs", headers={"X-API-Token": GOOD_TOKEN}).status_code == 200


def test_build_app_resolves_allowed_video_roots(monkeypatch, tmp_path) -> None:
    """H：allowed_video_roots 相对路径解析为绝对路径并接进 repo.security。"""

    app = _build_app(
        monkeypatch, tmp_path, **{"security.allowed_video_roots": "media,./other"}
    )
    roots = app.state.repo.security.allowed_video_roots
    assert [str(root) for root in roots] == [
        str((tmp_path / "media").resolve()),
        str((tmp_path / "other").resolve()),
    ]


def test_build_app_lifespan_clears_model_cache(monkeypatch, tmp_path) -> None:
    """D：lifespan 结束时清空模型缓存（进程退出释放）。"""

    from fastapi.testclient import TestClient

    from whisper_stt_service.service import model_cache as model_cache_module

    app = _build_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        # 进入 lifespan：runtime 已启动（worker 数 1，但无任务可领，不加载模型）。
        assert client.get("/healthz", headers={"X-API-Token": GOOD_TOKEN}).status_code == 200
        cache = model_cache_module.get_model_cache()
    # TestClient 退出即触发 lifespan finally：缓存被清空（无残留条目）。
    assert cache.stats()["entries"] == []
    assert model_cache_module._global_cache is not None