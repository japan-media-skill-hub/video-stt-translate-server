"""全局测试隔离：阻止 `configure_logging`/`dictConfig` 的 logger 状态泄漏。

背景：生产代码在启动时会执行 `logging.config.dictConfig`，它会就地改写
`whisper_stt_service` / `uvicorn*` / root 的 handlers、level、propagate 与
disabled，并给"既存 logger"补登记。任一用例触发它之后，后续用例的
`caplog`（依赖挂在 root 上的 LogCaptureHandler）就可能静默收不到日志，
表现为"单跑绿、组合跑红"的顺序依赖。

这里用**测试侧**的 autouse fixture 在用例前后各快照一次整棵 logger 树，
结束后整体还原；生产日志配置保持纯语义，不掺任何 pytest 专用逻辑。
"""

from __future__ import annotations

import logging

import pytest


# 需要还原的 logger 属性：handler 列表、等级、是否向上传播、是否被禁用。
_LOGGER_STATE_ATTRS = ("handlers", "level", "propagate", "disabled")


def _all_loggers() -> dict[str, logging.Logger]:
    """返回当前已实例化的全部 logger（不含 root）。

    `loggerDict` 里可能存放 PlaceHolder（父节点占位），这里只保留真正的 Logger。
    """

    return {
        name: logger
        for name, logger in logging.Logger.manager.loggerDict.items()
        if isinstance(logger, logging.Logger)
    }


def _snapshot() -> dict[str, dict[str, object]]:
    """快照 root 与全部 logger 的可变状态（含其 handler 自身的等级）。"""

    state: dict[str, dict[str, object]] = {}
    loggers: dict[str, logging.Logger] = {"": logging.getLogger()}
    loggers.update(_all_loggers())
    for name, logger in loggers.items():
        entry: dict[str, object] = {
            attr: (
                list(getattr(logger, attr))
                if attr == "handlers"
                else getattr(logger, attr)
            )
            for attr in _LOGGER_STATE_ATTRS
        }
        # dictConfig 会改写 handler 的 level（例如把外部采集器降级或升级）。
        entry["handler_levels"] = [
            (handler, handler.level) for handler in list(logger.handlers)
        ]
        state[name] = entry
    return state


def _handlers_in(snapshot: dict[str, dict[str, object]]) -> set:
    """收集快照中出现过的全部 handler 对象，用于区分"新装"与"原有"。"""

    return {
        handler
        for entry in snapshot.values()
        for handler, _level in entry["handler_levels"]  # type: ignore[union-attr]
    }


def _restore(snapshot: dict[str, dict[str, object]]) -> None:
    """把 logger 树还原到快照状态。

    - 快照中已有的 logger：handler 列表、level、propagate、disabled 全部还原；
    - 用例期间新建的 logger：解除禁用、恢复向上传播、清空挂载，避免影响邻居；
    - 快照中已有 handler 的 level 一并还原。
    """

    current: dict[str, logging.Logger] = {"": logging.getLogger()}
    current.update(_all_loggers())
    for name, logger in current.items():
        entry = snapshot.get(name)
        if entry is None:
            # 用例中新创建的 logger：恢复默认状态，避免影响后续用例。
            logger.disabled = False
            logger.propagate = True
            logger.handlers = []
            logger.setLevel(logging.NOTSET)
            continue
        for attr in _LOGGER_STATE_ATTRS:
            value = entry[attr]
            setattr(logger, attr, list(value) if attr == "handlers" else value)
        for handler, level in entry["handler_levels"]:  # type: ignore[union-attr]
            handler.setLevel(level)


def _close_new_handlers(before: dict[str, dict[str, object]]) -> None:
    """关闭用例新装上的 handler（多为临时目录里的文件 handler），释放句柄。"""

    known = _handlers_in(before)
    current: dict[str, logging.Logger] = {"": logging.getLogger()}
    current.update(_all_loggers())
    for logger in current.values():
        for handler in list(logger.handlers):
            if handler in known:
                continue
            try:
                handler.flush()
                handler.close()
            except Exception:  # noqa: BLE001 - 关闭失败不应影响用例结论
                pass


def _reset_request_id_context() -> None:
    """清空 request_id 的 ContextVar，避免上一个用例的请求 id 串到下一个用例。"""

    from whisper_stt_service.service.logging_config import request_id_var

    request_id_var.set("")


@pytest.fixture(autouse=True)
def _isolate_logging_state():
    """用例级日志隔离：前后还原整棵 logger 树，并清理请求上下文。"""

    before = _snapshot()
    _reset_request_id_context()
    try:
        yield
    finally:
        _reset_request_id_context()
        _close_new_handlers(before)
        _restore(before)
