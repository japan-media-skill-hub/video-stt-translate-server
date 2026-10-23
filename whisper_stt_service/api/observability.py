"""请求链路可观测性：request_id 中间件与健康检查路由（SPEC G）。

安全约束（SPEC H / 全局鉴权）：
- 这里**不**注册任何鉴权豁免，全部路由都走 `api/app.py` 的全局 token 鉴权。
  /healthz、/readyz、/metrics 与业务接口同级保护；运维探针需携带
  `X-API-Token`。唯一的公开例外是文档站点在 `docs_enabled=true` 且
  `public_docs=true` 的显式选择（由 app.py 处理）。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
import time
from typing import Any
import uuid

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from whisper_stt_service.service.logging_config import (
    bind_request_id,
    current_request_id,
    reset_request_id,
)


def _log_ctx() -> dict[str, str]:
    """构造带当前 request_id 的日志 extra。

    为什么用 extra 而不是 LogRecord 工厂/过滤器：记录工厂与过滤器都依赖
    logging 内部机制（pytest 捕获、跨任务 handler 执行会改写它们），而
    `extra` 在 **log 调用的那一刻**由本函数求值，随后以属性形式固定在
    LogRecord 上，经过任何 handler/进程都不会丢失。
    """

    return {"request_id": current_request_id() or "-"}


def _logger() -> logging.Logger:
    """延迟获取本模块 logger。

    app.py 的 `create_app` 可能在 `dictConfig` 之后（且跨任务上下文）才记日志；
    若在导入期就绑定 logger 实例，可能与 `configure_logging` 配置的实例脱节。
    统一在调用点取 logger，避免"日志静默丢失"。
    """

    return logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
# 透传的 request_id 会被写进日志：限制字符集与长度，防止日志注入 / 膨胀。
REQUEST_ID_MAX_LENGTH = 128
_SAFE_REQUEST_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789-_.:@"
)

router = APIRouter()


def sanitize_request_id(raw: str | None) -> str:
    """校验上游传入的 request_id；非法时返回空串，由调用方生成新 id。"""

    text = str(raw or "").strip()
    if not text or len(text) > REQUEST_ID_MAX_LENGTH:
        return ""
    if any(ch not in _SAFE_REQUEST_ID_CHARS for ch in text):
        return ""
    return text


def new_request_id() -> str:
    """生成 32 位十六进制 request_id（无连字符，便于日志 grep）。"""

    return uuid.uuid4().hex


def resolve_request_id(raw: str | None) -> str:
    """优先沿用上游 X-Request-ID，否则生成新的 request_id。"""

    return sanitize_request_id(raw) or new_request_id()


# 统一错误响应体不会回显内部异常文本，只暴露 request_id 便于运维对日志。
def disabled_endpoint_404(request: Request, request_id: str) -> JSONResponse | None:
    """对"配置上关闭"的端点直接返回 404（先于鉴权判定，不得被改写成 401）。

    规则（SPEC G/H：接口未注册即不存在）：
    - docs 关闭：/docs、/redoc、/openapi.json（含 /docs/*）→ 404；
      docs 开启：端点存在且**受 token 保护**（不豁免）。
    - metrics_enabled=false：/metrics → 404（不论是否携带 token）；
      metrics_enabled=true：端点存在且**受 token 保护**。
    返回 None 表示该路径不属于被关闭的端点，调用方继续正常鉴权流程。
    """

    path = request.url.path
    app_state = request.app.state
    if not getattr(app_state, "docs_enabled", False) and (
        path in DOC_PATHS or path.startswith("/docs/")
    ):
        reason = "docs_disabled"
    elif path == METRICS_PATH and not getattr(app_state, "metrics_enabled", True):
        reason = "metrics_disabled"
    else:
        return None
    _logger().info(
        "endpoint disabled: request_id=%s path=%s reason=%s",
        request_id or "-",
        path,
        reason,
        extra=_log_ctx(),
    )
    return JSONResponse(
        status_code=404,
        content={"detail": "not_found"},
        headers={REQUEST_ID_HEADER: request_id} if request_id else None,
    )


def install_error_handler(app: FastAPI) -> None:
    """注册兜底异常处理：500 响应也带上 request_id（错误返回可追踪）。

    Starlette 的 `ServerErrorMiddleware` 位于应用最外层，异常从中间件内抛出后
    会绕过 request_id 中间件的响应头回写；因此在注册期捕获异常类型并统一
    返回 JSON（含 request_id），既保留 traceback 日志，又让调用方拿到 id。
    """

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        """把未处理异常转成 500 JSON，并回写 X-Request-ID 响应头。"""

        request_id = str(getattr(request.state, "request_id", "") or "")
        # 走到这里说明 request_id 中间件已在末尾 reset，故以 request.state 为准。
        # 同时在此记录完整 traceback：与错误响应使用同一个 request_id，便于对日志。
        _logger().error(
            "unhandled exception: request_id=%s method=%s path=%s",
            request_id or "-",
            request.method,
            request.url.path,
            exc_info=exc,
            extra=_log_ctx(),
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "internal_server_error", "request_id": request_id},
            headers={REQUEST_ID_HEADER: request_id} if request_id else None,
        )


class RequestMetadataMiddleware:
    """纯 ASGI 中间件：补全 scope["app"] 与 request_id 后调用下游 app。

    为什么需要它：Starlette 只在**路由匹配成功后**才往 scope 写入 `app`；鉴权等
    纯 ASGI 中间件在匹配前就需要读 `request.app.state`。若不补全，未匹配路径
    （如 metrics_enabled=false 的 /metrics）会在鉴权分支抛 KeyError。

    构造签名必须保持 `(app, **kwargs)`：它会被 Starlette 以
    `cls(inner_app, **options)` 形式实例化，`self.app` 即下游 ASGI 应用；
    `__call__` 必须 `await self.app(scope, receive, send)`，绝不能再调用外层 app
    （曾因工厂闭包忽略 inner_app 造成无限递归）。
    """

    def __init__(self, app) -> None:
        """保存下游 ASGI 应用。"""

        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        """补全 http scope 元数据后透传调用下游应用。"""

        if scope.get("type") == "http":
            scope.setdefault("app", self.app)
            state = scope.setdefault("state", {})
            if "request_id" not in state:
                headers = {
                    key.decode("latin-1").lower(): value.decode("latin-1")
                    for key, value in scope.get("headers", [])
                }
                state["request_id"] = resolve_request_id(
                    headers.get(REQUEST_ID_HEADER.lower())
                )
        await self.app(scope, receive, send)


def install_metadata_middleware(app: FastAPI) -> None:
    """把 `RequestMetadataMiddleware` 装到中间件栈最外层。

    Starlette 的 `add_middleware` 是"后注册者在外层"，这里直接 `insert(0)`
    到栈顶，确保鉴权与 request_id 中间件都在它之内。
    """

    app.add_middleware(RequestMetadataMiddleware)


def install_request_id_middleware(app: FastAPI) -> None:
    """给 app 安装 request_id 中间件：绑定上下文、回写响应头、记录访问日志。

    同时覆盖业务接口与 `/metrics`、`/healthz` 等观测接口；鉴权中间件在
    `app.py` 中先注册（Starlette 后注册者在外层），因此 401 响应同样带
    request_id，且失败日志能对上来源。
    """

    @app.middleware("http")
    async def _request_id_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[object]],
    ):
        """为每个请求生成/透传 request_id，并写入响应头与访问日志。"""

        request_id = resolve_request_id(request.headers.get(REQUEST_ID_HEADER))
        token = bind_request_id(request_id)
        request.state.request_id = request_id
        started = time.perf_counter()
        # 注意：reset 必须放在 finally，但访问日志要在此之前写出——
        # 否则 ContextVar 已还原，日志里的 request_id 会退化成 "-"。
        # 因此这里用"受控 finally"：先记录，再 reset（异常路径记录失败日志后 reset）。
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _logger().exception(
                "request failed: request_id=%s method=%s path=%s elapsed_ms=%d",
                request_id,
                request.method,
                request.url.path,
                elapsed_ms,
                extra=_log_ctx(),
            )
            raise
        else:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            response.headers[REQUEST_ID_HEADER] = request_id
            # 观测接口探针调用频繁，默认只记 debug；业务接口记 info。
            logger = _logger()
            log_method = (
                logger.debug if request.url.path in OBSERVABILITY_PATHS else logger.info
            )
            log_method(
                "request completed: request_id=%s method=%s path=%s status=%d elapsed_ms=%d",
                request_id,
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
                extra=_log_ctx(),
            )
            return response
        finally:
            reset_request_id(token)


# 指标端点路径：metrics_enabled=false 时与 docs 一致，先于鉴权返回 404。
METRICS_PATH = "/metrics"

# 文档端点的具体路径集合（由 FastAPI 生成）；关闭 docs 时先于鉴权返回 404。
DOC_PATHS: frozenset[str] = frozenset(
    {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
)

# 高频探针路径：访问日志降级为 debug，避免 5s 一次的探针刷屏。
OBSERVABILITY_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})


def resolve_service_version() -> str:
    """读取服务版本号（优先安装元数据，回退 pyproject，再回退 unknown）。"""

    try:
        from importlib.metadata import version

        return version("video-stt-translate-server")
    except Exception:  # noqa: BLE001 - 版本号缺失不应影响健康检查
        pass
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        for line in pyproject_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("version"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return "unknown"


def _close_quietly(conn: Any) -> None:
    """关闭连接并吞掉关闭阶段的错误（探针失败时连接可能已半关闭）。"""

    if conn is None:
        return
    try:
        conn.close()
    except Exception:  # noqa: BLE001 - 关闭失败不影响健康结论
        pass


def _now_iso() -> str:
    """返回 UTC ISO 时间戳，用于健康检查响应。"""

    return datetime.now(timezone.utc).isoformat()


def check_db_query(repo: object | None) -> tuple[bool, str | None]:
    """执行 `SELECT 1` 验证 DB 可查询；返回 (ok, error)。"""

    if repo is None:
        return False, "repo_not_configured"
    db = getattr(repo, "db", None)
    if db is None:
        return False, "db_not_configured"
    try:
        with db.connect() as conn:
            row = conn.execute("SELECT 1").fetchone()
        if row is None or int(row[0]) != 1:
            return False, "unexpected_select_result"
    except sqlite3.Error as exc:
        return False, f"sqlite_error: {exc}"
    except Exception as exc:  # noqa: BLE001 - 任何异常都视为 not ready
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


# readyz 的 main DB 写探测表名：每次探测"创建→写入→回滚"，不留任何 schema。
READYZ_PROBE_TABLE = "__readyz_write_probe"


def check_db_writable(repo: object | None, probe_dir: Path | None = None) -> tuple[bool, str | None]:
    """验证 DB 可写：对 **main** 库执行一个真实写事务，随后回滚。

    为什么必须写 main 而不是只写 TEMP：SQLite 的 `mode=ro` 连接照样能
    `BEGIN IMMEDIATE`、也能创建/写入 TEMP 表（TEMP 落在临时存储），只有触碰
    main 库时才会报 `attempt to write a readonly database`。只写 TEMP 会给出
    假就绪，因此这里：

    1. `BEGIN IMMEDIATE` 取得写锁（不经过 `db.tx()`：它的正常退出是 commit，
       读写分离/只读挂载的库在空 commit 时可能照样成功）；
    2. 在 main 库内 `CREATE TABLE IF NOT EXISTS` 探测表并 `INSERT` 一行，
       证明连接对主库具备真实写能力；
    3. 一律 `ROLLBACK`，探测表与其数据都不会持久化（实测 DB 文件字节级不变）；
    4. 再在 DB 同目录创建并删除探测文件，覆盖"DB 文件可写但目录不可写"
       （SQLite 建 journal/WAL 需要写目录）。
    """

    db = getattr(repo, "db", None) if repo is not None else None
    if db is None:
        return False, "db_not_configured"
    db_path = Path(getattr(db, "db_path", ""))
    if not str(db_path):
        return False, "db_path_not_configured"

    conn = None
    try:
        conn = db.connect()
        conn.execute("BEGIN IMMEDIATE")
        # 写 main 库：只读连接（mode=ro / 只读挂载）会在这里失败，这正是我们要探测的。
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS main.{READYZ_PROBE_TABLE} (v INTEGER)"
        )
        conn.execute(f"INSERT INTO main.{READYZ_PROBE_TABLE} (v) VALUES (1)")
    except sqlite3.Error as exc:
        return False, f"sqlite_not_writable: {exc}"
    except Exception as exc:  # noqa: BLE001 - 探针必须给出结论，不能抛给 ASGI
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        # 无论成功失败都回滚并释放连接：结论在释放后不再依赖连接状态。
        if conn is not None:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            _close_quietly(conn)

    directory = probe_dir if probe_dir is not None else db_path.parent
    probe_path = directory / f".readyz-probe-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        # 探测文件落在 DB 同目录：目录不可写时负责建 journal 的 SQLite 同样会失败。
        directory.mkdir(parents=True, exist_ok=True)
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink()
    except OSError as exc:
        return False, f"db_dir_not_writable: {exc}"
    finally:
        try:
            probe_path.unlink(missing_ok=True)
        except OSError:
            pass
    return True, None


def check_directory_writable(path: Path | None, label: str) -> tuple[bool, str | None]:
    """验证目录存在（自动创建）且可写；用于 artifact_root / log_root。"""

    if path is None or not str(path):
        return False, f"{label}_not_configured"
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_path = path / f".{label}-probe-{os.getpid()}-{uuid.uuid4().hex}"
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink()
    except OSError as exc:
        return False, f"{label}_not_writable: {exc}"
    return True, None


# 模型目录至少要有一个非隐藏条目（config.json / model.bin / *.safetensors 等）
# 才算"已就位"；空目录 / 只有 .gitkeep 的占位目录给出 503，避免假就绪。
MODEL_ARTIFACT_EXTENSIONS: frozenset[str] = frozenset(
    {".bin", ".safetensors", ".json", ".onnx", ".pt", ".ckpt", ".model", ".xml"}
)


def check_model_path(path: Path | None) -> tuple[bool, str | None]:
    """验证模型路径确实已就位；不加载模型，不做任何推理。

    目录形态要求包含至少一个非隐藏条目、且至少命中一个已知模型产物扩展名
    （config.json/model.bin/*.safetensors 等）；文件形态要求非空。
    """

    if path is None or not str(path):
        return False, "model_path_not_configured"
    if not path.exists():
        return False, "model_path_missing"
    if path.is_file():
        try:
            if path.stat().st_size <= 0:
                return False, "model_path_empty"
        except OSError as exc:
            return False, f"model_path_unreadable: {exc}"
        return True, None
    if not path.is_dir():
        return False, "model_path_not_regular"
    try:
        entries = [
            entry
            for entry in path.iterdir()
            if not entry.name.startswith(".") and entry.is_file()
        ]
    except OSError as exc:
        return False, f"model_path_unreadable: {exc}"
    if not entries:
        return False, "model_path_empty"
    if not any(entry.suffix.lower() in MODEL_ARTIFACT_EXTENSIONS for entry in entries):
        return False, "model_path_no_artifacts"
    return True, None


def _settings_snapshot(request: Request) -> object | None:
    """读取 app.state.settings（bootstrap 注入），缺失时为 None。"""

    return getattr(request.app.state, "settings", None)


def _run_readiness_check(
    name: str, probe: Callable[[], tuple[bool, str | None]]
) -> dict[str, object]:
    """执行单个就绪检查，把意外异常收敛成 {ok: false, error}，避免 500。"""

    try:
        ok, error = probe()
    except Exception as exc:  # noqa: BLE001 - 探针必须给出结论，不能抛给 ASGI
        return {"ok": False, "error": f"{name}_check_failed: {type(exc).__name__}: {exc}"}
    return {"ok": bool(ok), "error": error}


def _readiness_checks(request: Request) -> dict[str, dict[str, object]]:
    """执行 readyz 的四类检查，返回 {name: {ok, error, ...}}。"""

    repo = getattr(request.app.state, "repo", None)
    settings = _settings_snapshot(request)
    checks: dict[str, dict[str, object]] = {}

    # 1) 配置完整：bootstrap 注入 settings 视为已装配；缺失时明确标记。
    if settings is None:
        checks["config"] = {"ok": False, "error": "settings_not_injected"}
    else:
        missing: list[str] = []
        security = getattr(settings, "security", None)
        if security is not None:
            # 与 api/app.py 的参考实现保持同一套"安全默认值"判定。
            from whisper_stt_service.api.app import missing_strong_token  # 延迟导入避免环

            if missing_strong_token(security):
                missing.append("security.api_token")
        if missing:
            checks["config"] = {
                "ok": False,
                "error": "insecure_security_config",
                "missing": missing,
            }
        else:
            checks["config"] = {"ok": True, "error": None}

    # 2) DB 可写
    checks["db_writable"] = _run_readiness_check(
        "db_writable", lambda: check_db_writable(repo)
    )

    # 3) 模型路径存在
    model_path = getattr(request.app.state, "model_path", None)
    if model_path is None and settings is not None:
        model_path = getattr(getattr(settings, "runtime", None), "model_path", None)
    checks["model_path"] = _run_readiness_check(
        "model_path", lambda: check_model_path(Path(model_path) if model_path else None)
    )
    checks["model_path"]["path"] = str(model_path) if model_path else None

    # 4) artifact_root 可写
    artifact_root = getattr(request.app.state, "artifact_root", None)
    if artifact_root is None and settings is not None:
        artifact_root = getattr(getattr(settings, "runtime", None), "artifact_root", None)
    checks["artifact_root"] = _run_readiness_check(
        "artifact_root",
        lambda: check_directory_writable(
            Path(artifact_root) if artifact_root else None, "artifact_root"
        ),
    )
    checks["artifact_root"]["path"] = str(artifact_root) if artifact_root else None
    return checks


@router.get("/healthz")
def healthz(request: Request):
    """存活探针：进程能响应 + DB 可执行 `SELECT 1` 即 200。"""

    repo = getattr(request.app.state, "repo", None)
    ok, error = check_db_query(repo)
    payload = {
        "status": "ok" if ok else "error",
        "checks": {"db_query": {"ok": ok, "error": error}},
        "request_id": getattr(request.state, "request_id", None),
        "timestamp": _now_iso(),
        "version": resolve_service_version(),
    }
    if not ok:
        return JSONResponse(status_code=503, content=payload)
    return payload


@router.get("/readyz")
def readyz(request: Request):
    """就绪探针：配置完整、DB 可写、模型路径存在、artifact_root 可写。"""

    checks = _readiness_checks(request)
    ready = all(bool(item.get("ok")) for item in checks.values())
    payload = {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
        "request_id": getattr(request.state, "request_id", None),
        "timestamp": _now_iso(),
        "version": resolve_service_version(),
    }
    if not ready:
        return JSONResponse(status_code=503, content=payload)
    return payload


def build_health_log_payload() -> str:
    """把健康检查结果转成单行 JSON（供 scripts/运维脚本 grep 使用）。"""

    return json.dumps({"timestamp": _now_iso()}, ensure_ascii=False)