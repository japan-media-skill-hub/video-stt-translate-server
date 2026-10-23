"""FastAPI 应用构建。

入口契约（供 bootstrap / 生命周期代理接入）：
`create_app(repo=..., progress_store=..., runtime=..., settings=..., api_token=...,
metrics=..., metrics_enabled=...)`。

鉴权与文档暴露面（SPEC H、D4）：
- `api_token`：历史参数，保持兼容；token 为空 = 鉴权关闭（启动时记录 WARN）。
- `settings`：可选的安全设置来源（SecuritySettings/兼容对象，可含
  `require_token`/`allow_placeholder_token`/`docs_enabled`）。传入时优先于
  `api_token`，bootstrap 后续应改为只传 `settings`。
- 关闭 docs 时 /docs、/redoc、/openapi.json 一律 **404**（不得先被鉴权改成 401）；
  开启 docs 时这些路径仍然受 token 保护。
- healthz/readyz/metrics 不设任何鉴权豁免（SPEC 全局安全）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from whisper_stt_service.api.observability import (
    disabled_endpoint_404,
    install_error_handler,
    install_metadata_middleware,
    install_request_id_middleware,
    resolve_request_id,
    router as observability_router,
)
from whisper_stt_service.api.routes.jobs import router as jobs_router
from whisper_stt_service.api.routes.admin import router as admin_router
from whisper_stt_service.api.routes.metrics import router as metrics_router
from whisper_stt_service.api.routes.queue import router as queue_router
from whisper_stt_service.core.security import (
    SecurityError,
    is_placeholder_token,
    tokens_match,
    token_is_secure_enough,
    validate_startup_token,
)
from whisper_stt_service.core.progress import ProgressStore
from whisper_stt_service.repo.database import Database
from whisper_stt_service.repo.job_repository import JobRepository
from whisper_stt_service.service.metrics import MetricsState, build_metrics
from whisper_stt_service.service.runtime import WorkerRuntime


LOGGER = logging.getLogger(__name__)

# FastAPI 自动生成且由本服务自行接管可见性的文档端点。


def _setting(source: Any, name: str, default: Any = None) -> Any:
    """从 settings（或兼容对象）读取字段；dict 与对象两种形态都支持。"""

    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def resolve_security_policy(
    settings: Any | None,
    api_token: str | None,
) -> dict[str, Any]:
    """把两处 token 来源合并成一份鉴权策略。

    优先级：`settings.api_token`（含 `require_token`/`allow_placeholder_token`/
    `docs_enabled`）> `api_token` 参数 > 默认（无 token、关闭鉴权、docs 关闭）。
    返回值：
    `{"api_token", "require_token", "allow_placeholder_token", "docs_enabled", "source"}`。
    """

    security = _setting(settings, "security", None) if settings is not None else None
    token_source = security if security is not None else settings

    settings_token = _setting(token_source, "api_token", None)
    if settings_token is not None:
        api_token_value = str(settings_token).strip()
        require_token = bool(_setting(token_source, "require_token", False))
        allow_placeholder = bool(_setting(token_source, "allow_placeholder_token", False))
        docs_enabled = _setting(token_source, "docs_enabled", None)
        source = "settings"
    else:
        api_token_value = str(api_token or "").strip()
        # 历史 api_token 参数没有 require_token 语义：空 token 即关闭鉴权。
        require_token = False
        allow_placeholder = True
        docs_enabled = _setting(settings, "docs_enabled", None)
        source = "api_token" if api_token_value else "none"

    if docs_enabled is None:
        # 未显式配置时 docs 默认关闭（D4 安全默认值）。
        docs_enabled = False
    return {
        "api_token": api_token_value,
        "require_token": require_token,
        "allow_placeholder_token": allow_placeholder,
        "docs_enabled": bool(docs_enabled),
        "source": source,
    }


def missing_strong_token(security: Any) -> bool:
    """readyz 复用：判断安全配置是否不满足 fail-closed 要求。"""

    token = _setting(security, "api_token", "") or ""
    require_token = bool(_setting(security, "require_token", False))
    allow_placeholder = bool(_setting(security, "allow_placeholder_token", False))
    try:
        validate_startup_token(
            token,
            require_token=require_token,
            allow_placeholder_token=allow_placeholder,
        )
    except SecurityError:
        return True
    return False


def create_app(
    *,
    repo: JobRepository | None = None,
    progress_store: ProgressStore | None = None,
    runtime: WorkerRuntime | None = None,
    worker_view: dict | None = None,
    api_token: str | None = None,
    settings: Any | None = None,
    metrics: MetricsState | None = None,
    metrics_enabled: bool | None = None,
) -> FastAPI:
    """创建并返回 FastAPI 应用实例。

    新增参数：
    - `settings`：可选安全设置（优先于 `api_token`），同时挂到 app.state 供 readyz 使用。
    - `metrics`：bootstrap 预先构建的 `MetricsState`；缺省时按 repo 自动构建。
    - `metrics_enabled`：False 时不注册 /metrics（返回 404）；None 时读 settings。
    """

    if repo is None:
        # 兼容单元测试直接 create_app() 的调用方式。
        db_file = Path("./tmp/api-test.db")
        db_file.parent.mkdir(parents=True, exist_ok=True)
        db = Database(db_file)
        db.init_schema()
        repo = JobRepository(db)
    if progress_store is None:
        progress_store = ProgressStore(ttl_seconds=3600)

    policy = resolve_security_policy(settings, api_token)
    if metrics_enabled is None:
        configured = _setting(settings, "metrics_enabled", None)
        if configured is None:
            configured = _setting(_setting(settings, "observability", None), "metrics_enabled", None)
        # 默认开启：SPEC G 要求 metrics_enabled=false 才关闭。
        metrics_enabled = True if configured is None else bool(configured)
    docs_enabled = policy["docs_enabled"]

    app = FastAPI(
        title="whisper-stt-service",
        # 关闭 docs 时不给 schema：即使有人误调用也会得到 404 而不是 200。
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )
    app.state.repo = repo
    app.state.progress_store = progress_store
    app.state.runtime = runtime
    app.state.worker_view = worker_view or {}
    # 保留既有属性名：外部脚本/测试仍通过 app.state.api_token 读取当前口令。
    app.state.api_token = policy["api_token"]
    app.state.settings = settings
    app.state.security_policy = policy
    app.state.docs_enabled = docs_enabled
    app.state.metrics_enabled = bool(metrics_enabled)

    # 未传入 settings 时，把配置里的路径显式挂到 state，供 readyz 检查。
    if settings is not None:
        runtime_settings = _setting(settings, "runtime", None)
        if getattr(app.state, "artifact_root", None) is None:
            app.state.artifact_root = _setting(runtime_settings, "artifact_root", None)
        if getattr(app.state, "model_path", None) is None:
            app.state.model_path = _setting(runtime_settings, "model_path", None)

    if app.state.metrics_enabled:
        if metrics is None:
            metrics = build_metrics(
                repo_factory=lambda: getattr(app.state, "repo", None),
                stages=("extract", "stt", "stt_whisperx", "translate"),
            )
        app.state.metrics = metrics
    else:
        app.state.metrics = None

    def _log_auth_failure(request, request_id: str, reason: str) -> None:
        """记录鉴权失败：只记来源与 request_id，绝不记录 token 明文。"""

        client = getattr(request, "client", None)
        client_host = getattr(client, "host", None) or "unknown"
        LOGGER.warning(
            "api auth rejected: request_id=%s method=%s path=%s client_ip=%s reason=%s",
            request_id,
            request.method,
            request.url.path,
            client_host,
            reason,
            # request_id 在调用点固定进 LogRecord，避免 handler 跨上下文执行时丢失。
            extra={"request_id": request_id},
        )

    def _log_token_policy(policy_snapshot: dict[str, Any]) -> None:
        """启动期记录 token 安全状态（不打印 token 本身）。"""

        token_value = policy_snapshot["api_token"]
        if not token_value:
            LOGGER.warning(
                "api token auth is DISABLED (api_token empty); "
                "set [security] api_token to enable authentication"
            )
            return
        if is_placeholder_token(token_value):
            LOGGER.warning("api token looks like a placeholder value; replace it")
        elif not token_is_secure_enough(token_value):
            LOGGER.warning("api token is shorter than 16 chars; consider a stronger token")
        if not policy_snapshot["docs_enabled"]:
            LOGGER.info("docs endpoints disabled (/docs, /redoc, /openapi.json -> 404)")

    _log_token_policy(policy)

    # 鉴权中间件只在配置了 token 时注册：token 为空（鉴权关闭）时完全等价于
    # 旧行为，不会给任何请求（包括 /metrics、docs 404）附加额外的鉴权判定。
    #
    # 历史实现（HEAD 的 exempt_paths 集合）曾把 /docs、/openapi.json、/redoc、
    # /docs/oauth2-redirect 对 token 豁免；SPEC H 要求"docs 关闭时 404、开启时
    # 仍受 token 保护"，因此本次不再保留该豁免——docs 端点与其他接口同级鉴权。
    if policy["api_token"]:

        @app.middleware("http")
        async def _api_token_guard(request, call_next):
            """按配置启用全局 API Token 鉴权。

            鉴权使用常量时间比较，失败时记录来源 IP；这些是当前实现细节，
            不改变“按配置启用”的鉴权语义。
            """

            # request_id 中间件在内层已解析好；这里兜底保证独立可测。
            request_id = str(
                getattr(request.state, "request_id", "")
                or resolve_request_id(request.headers.get("X-Request-ID"))
            )
            # docs/metrics 在配置上关闭时先给出 404 语义（端点不存在），
            # 不能被鉴权改写成 401；开启时这些路径必须携带 token（不设豁免）。
            disabled_response = disabled_endpoint_404(request, request_id)
            if disabled_response is not None:
                return disabled_response

            provided = request.headers.get("X-API-Token", "").strip()
            if not tokens_match(provided, str(request.app.state.api_token).strip()):
                _log_auth_failure(
                    request,
                    request_id,
                    "missing_token" if not provided else "token_mismatch",
                )
                return JSONResponse(
                    status_code=401,
                    content={"detail": "invalid_api_token"},
                    headers={"X-Request-ID": request_id},
                )
            return await call_next(request)

    # 顺序（由内到外）：鉴权 -> request_id -> metadata。
    # - request_id 在鉴权外层：401 响应也能带 request_id。
    # - metadata 最外层：Starlette 路由匹配前就补 scope["app"]，使未匹配路径
    #   （如 metrics_enabled=false 的 /metrics）也能安全进入鉴权判定。
    install_request_id_middleware(app)
    install_metadata_middleware(app)
    # 未处理异常同样返回 request_id（500 响应可追踪）。
    install_error_handler(app)

    app.include_router(admin_router)
    app.include_router(jobs_router)
    app.include_router(queue_router)
    app.include_router(observability_router)
    if app.state.metrics_enabled:
        app.include_router(metrics_router)
    return app