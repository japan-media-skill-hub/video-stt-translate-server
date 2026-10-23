"""服务日志配置：控制台 + 轮转文件，可选 JSONL 结构化输出（SPEC G）。

职责范围：
- 用 `logging.config.dictConfig` 一次性配置应用 logger 与
  `uvicorn` / `uvicorn.error` / `uvicorn.access`。
- 等级取 `WHISPER_STT_LOG_LEVEL`（默认 INFO）；`WHISPER_STT_LOG_JSONL=1`
  时落盘的 `service.log` 每行一个 JSON 对象。
- 通过 `request_id_var` / `RequestIdFilter` 把 HTTP 请求 id 透传到日志，
  由 `api/observability.py` 的中间件负责绑定与回写响应头。

本模块不在导入时产生任何副作用：只有显式调用 `configure_logging()` 才会
改动全局 logging 状态，避免污染纯单元测试。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
from logging.config import dictConfig
import os
from pathlib import Path
import sys
from typing import Any, Iterator


def _logger() -> logging.Logger:
    """延迟获取本模块 logger。

    本模块会在 bootstrap 的 `configure_logging` 里被 `dictConfig` 接管；若在
    模块导入期就创建 logger，可能早于 dictConfig（存在 service.metrics ↔
    model_cache ↔ logging_config 的导入链），导致 logger 实例与配置不同步。
    因此统一通过本函数在调用点取 logger。
    """

    return logging.getLogger(__name__)


# 应用侧需要统一配置的业务 logger 前缀集合。
APP_LOGGER_NAMES: tuple[str, ...] = ("whisper_stt_service",)
# uvicorn 自带 logger：与业务日志共享同一套 handler 与格式。
UVICORN_LOGGER_NAMES: tuple[str, ...] = ("uvicorn", "uvicorn.error", "uvicorn.access")

LOG_FILE_NAME = "service.log"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 5
DEFAULT_TEXT_FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"

# 请求上下文变量：由 HTTP 中间件写入，请求结束（或 reset）后恢复。
# 不在 HTTP 请求内（例如 worker 线程）时为空串，由 formatter 兜底成 "-"。
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# 结构化字段的标准顺序（SPEC G：task 日志字段统一）。
STRUCTURED_FIELD_ORDER: tuple[str, ...] = (
    "timestamp",
    "level",
    "logger",
    "request_id",
    "job_id",
    "task_id",
    "stage",
    "worker_id",
    "event",
    "result",
    "attempt",
    "effective_config",
    "error",
    "message",
)

_TRUE_FLAGS = {"1", "true", "yes", "on"}
_FALSE_FLAGS = {"0", "false", "no", "off"}


def resolve_log_level(raw: object | None = None) -> str:
    """把 `WHISPER_STT_LOG_LEVEL` 解析为合法等级名，非法值回退 INFO。

    显式传入的 `raw` 优先于环境变量；空值视为未设置。
    """

    value = raw if raw is not None else os.getenv("WHISPER_STT_LOG_LEVEL", "")
    candidate = str(value).strip().upper()
    if not candidate:
        return DEFAULT_LOG_LEVEL
    level_value = getattr(logging, candidate, None)
    if isinstance(level_value, int):
        return candidate
    _logger().warning(
        "invalid log level %r, fallback to %s", str(value), DEFAULT_LOG_LEVEL
    )
    return DEFAULT_LOG_LEVEL


def env_flag(name: str, default: bool = False) -> bool:
    """解析形如 `1/true/yes/on` 的环境变量布尔值。"""

    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in _TRUE_FLAGS


def resolve_jsonl_enabled(
    *,
    jsonl: bool | None = None,
    log_format: str | None = None,
) -> bool:
    """解析是否启用 JSONL 落盘。

    优先级：显式 `jsonl` 参数 > `[observability] log_format` > 环境变量
    `WHISPER_STT_LOG_JSONL` > 默认 False（文本格式）。
    """

    if jsonl is not None:
        return bool(jsonl)
    fmt = str(log_format or "").strip().lower()
    if fmt in {"json", "jsonl"}:
        return True
    if fmt in {"text", "plain"}:
        return False
    return env_flag("WHISPER_STT_LOG_JSONL", default=False)


def current_request_id() -> str:
    """读取当前上下文的 request_id；请求外返回空串。"""

    return str(request_id_var.get() or "")


def bind_request_id(request_id: str):
    """把 request_id 绑定到当前上下文，返回可供 `reset_request_id` 使用的 token。"""

    return request_id_var.set(str(request_id or ""))


def reset_request_id(token) -> None:
    """恢复绑定前的 request_id，容忍跨上下文 reset 失败。"""

    try:
        request_id_var.reset(token)
    except (ValueError, RuntimeError):
        # token 来自其他上下文时 reset 会报错；直接归零，避免中间件抛异常。
        request_id_var.set("")


def _coerce_log_value(value: Any) -> Any:
    """把非 JSON 原生对象转成可序列化形式，保证一条日志不会写失败。"""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _coerce_log_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_log_value(v) for v in value]
    return repr(value)


def sanitize_text_message(text: Any) -> str:
    """清洗第三方（如 uvicorn access）写进消息里的裸引号。

    uvicorn.access 使用 `'"%s" %s %s'` 拼消息，会在文本格式里产生
    `""GET /healthz" 200"`；JSONL 下消息是字符串，不影响解析，但这里统一
    剥离成 `GET /healthz 200` 便于 grep。
    """

    message = str(text)
    if message.count('"') >= 2:
        message = message.replace('"', "")
    return message


def _resolve_record_request_id(record: logging.LogRecord) -> str:
    """计算一条日志实际应携带的 request_id。

    优先级与原因：
    1. 调用方通过 `extra={"request_id": ...}` 显式传入的值——HTTP 中间件都走这条
       路径，在 log 调用的同一上下文里求值，经过任何 handler / 进程都不会丢；
    2. 上下文中当前的 request_id（请求作用域内的业务日志自动继承）；
    3. `-`：请求外（worker 线程、启动期）的占位。
    """

    explicit = getattr(record, "request_id", "")
    if explicit:
        return str(explicit)
    return current_request_id() or "-"


class RequestIdFilter(logging.Filter):
    """把上下文中的 request_id 注入每条 LogRecord。"""

    def filter(self, record: logging.LogRecord) -> bool:
        """补齐 request_id 字段并返回 True（不做日志丢弃）。"""

        record.request_id = _resolve_record_request_id(record)
        return True


class TextFormatter(logging.Formatter):
    """文本格式：时间 等级 [request_id] logger: 消息。"""

    def format(self, record: logging.LogRecord) -> str:
        """渲染文本行，并在有异常信息时附加 traceback。"""

        record.request_id = _resolve_record_request_id(record)
        rendered = super().format(record)
        rendered = sanitize_text_message(rendered)
        if record.exc_info:
            exception_text = self.formatException(record.exc_info)
            rendered = f"{rendered}\n{exception_text}"
        return rendered


def order_structured_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """按 SPEC G 字段顺序重排字典：标准字段在前，其余按字母序追加。

    单独抽成函数，供任务日志（task.log）与 service.log 复用同一字段序。
    """

    ordered: dict[str, Any] = {}
    for key in STRUCTURED_FIELD_ORDER:
        if key in payload:
            ordered[key] = payload[key]
    for key in sorted(payload):
        if key not in ordered:
            ordered[key] = payload[key]
    return ordered


class JsonLineFormatter(logging.Formatter):
    """JSONL 格式：一行一个 JSON 对象。

    固定输出 SPEC G 约定字段（缺失为 null），并把调用方通过
    `extra={"job_id": ...}` 等方式附加的自定义字段合并到顶层。
    """

    # LogRecord 内建属性，序列化时需排除，避免把日志对象自身写进 JSON。
    _RESERVED_ATTRS = frozenset(
        {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "taskName",
            "message",
            "asctime",
            "request_id",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        """渲染单行 JSON：标准字段在前，扩展字段按字母序追加在后。"""

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": _resolve_record_request_id(record),
            "message": record.getMessage(),
            "error": None,
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self._RESERVED_ATTRS and not key.startswith("_")
        }
        for key, value in extras.items():
            # error 可能是结构化字段（字符串/字典），也可能是复杂对象；
            # 统一走 `_coerce_log_value`，保证 JSON 一定可序列化。
            payload[key] = _coerce_log_value(value)
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(
            order_structured_fields(payload), ensure_ascii=False, default=repr
        )


def build_logging_config(
    *,
    log_dir: Path | str | None,
    level: str | None = None,
    jsonl: bool = False,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    log_file_name: str = LOG_FILE_NAME,
    console: bool = True,
) -> dict[str, Any]:
    """构造 `dictConfig` 字典（纯函数，便于单测直接断言结构）。

    - 控制台固定使用人类可读文本（即使开启 JSONL，排障仍可直读终端）。
    - 文件 handler 使用 `RotatingFileHandler`：maxBytes 轮转、backupCount 保留份数。
    - `log_dir` 为 None 时只挂控制台 handler（测试 / CLI 场景）。
    """

    level_name = resolve_log_level(level)
    handlers: dict[str, dict[str, Any]] = {}
    handler_names: list[str] = []

    if console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "level": level_name,
            "formatter": "text",
            "filters": ["request_id"],
            "stream": "ext://sys.stderr",
        }
        handler_names.append("console")

    if log_dir is not None:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level_name,
            "formatter": "json" if jsonl else "text",
            "filters": ["request_id"],
            "filename": str(Path(log_dir) / log_file_name),
            "maxBytes": int(max_bytes),
            "backupCount": int(backup_count),
            "encoding": "utf-8",
            # 延迟打开：只有真正写入时才创建文件，避免空文件堆积。
            "delay": True,
        }
        handler_names.append("file")

    config: dict[str, Any] = {
        "version": 1,
        # 不禁用既有 logger：pytest/其他库的 handler 保持工作，只接管目标 logger。
        "disable_existing_loggers": False,
        "formatters": {
            "text": {"()": f"{__name__}.TextFormatter", "format": DEFAULT_TEXT_FORMAT},
            "json": {"()": f"{__name__}.JsonLineFormatter"},
        },
        "filters": {
            "request_id": {"()": f"{__name__}.RequestIdFilter"},
        },
        "handlers": handlers,
        "loggers": {},
        "root": {
            "level": level_name,
            "handlers": list(handler_names),
        },
    }
    for name in APP_LOGGER_NAMES + UVICORN_LOGGER_NAMES:
        config["loggers"][name] = {
            "level": level_name,
            "handlers": list(handler_names),
            # uvicorn 的 logger 默认未 propagate；统一关闭，避免重复输出。
            "propagate": False,
        }
    # dictConfig 会把"未出现在 loggers 里"的既存 logger 的 disabled 置为 True，
    # 从而吞掉宿主进程（pytest caplog 等）挂在这些 logger 上的 handler。
    # 这里把当前已存在的 logger 全部登记为"保持 propagate、不改输出"，
    # 宿主 handler 因此仍留在原处、原样收到日志（生产语义不做任何 pytest 特判）。
    for existing_name in list(logging.Logger.manager.loggerDict):
        if existing_name in config["loggers"]:
            continue
        config["loggers"][existing_name] = {"handlers": [], "propagate": True}

    return config


def configure_logging(
    *,
    log_dir: Path | str | None,
    level: str | None = None,
    jsonl: bool | None = None,
    log_format: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    console: bool = True,
) -> dict[str, Any]:
    """进程启动早期调用一次：配置控制台 + `log_dir/service.log` 轮转日志。

    返回实际应用的配置字典，便于 bootstrap 记录生效的日志文件与等级。
    `log_dir` 不存在时自动创建；创建失败时降级为仅控制台，不阻断启动。
    """

    file_dir: Path | None = None
    if log_dir is not None:
        candidate = Path(log_dir)
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            file_dir = candidate
        except OSError as exc:  # 只读盘 / 权限不足时不应阻断服务启动。
            _logger().error(
                "log dir not writable, fallback to console only: dir=%s error=%s",
                candidate,
                exc,
            )

    jsonl_enabled = resolve_jsonl_enabled(jsonl=jsonl, log_format=log_format)
    config = build_logging_config(
        log_dir=file_dir,
        level=level,
        jsonl=jsonl_enabled,
        max_bytes=max_bytes,
        backup_count=backup_count,
        console=console,
    )
    dictConfig(config)
    _logger().info(
        "logging configured: level=%s file=%s jsonl=%s max_bytes=%d backups=%d",
        resolve_log_level(level),
        str(file_dir / LOG_FILE_NAME) if file_dir is not None else "-",
        jsonl_enabled,
        int(max_bytes),
        int(backup_count),
    )
    return config


@contextmanager
def logging_config_scope(level: int | str = logging.DEBUG) -> Iterator[logging.Logger]:
    """测试辅助：临时把应用 logger 提到指定等级并挂一个 JSONL 捕获 handler。

    退出时恢复 logger 原有等级、handler 列表、propagate 标记与 disabled 标记，
    保证 `configure_logging` 造成的全局状态不会泄漏到其他用例。
    """
    logger = logging.getLogger(APP_LOGGER_NAMES[0])
    previous_level = logger.level
    previous_handlers = list(logger.handlers)
    previous_propagate = logger.propagate
    previous_disabled = logger.disabled

    captured = logging.StreamHandler(sys.stderr)
    captured.setFormatter(JsonLineFormatter())
    try:
        logger.setLevel(level)
        logger.propagate = False
        logger.handlers = [captured]
        yield logger
    finally:
        logger.handlers = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
        logger.disabled = previous_disabled