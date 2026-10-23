"""服务层导出。

注意：本包在导入期**不得**急切导入 `runtime`/`bootstrap`。
导入链条 `executor → service.model_cache` 会先触发本包的 `__init__`，而 runtime 又反向
导入 executor，急切导出会形成循环导入（`ImportError` / 半初始化模块）。因此这里改为
模块级 `__getattr__` 惰性导出：首次访问属性时才导入对应模块，`from ... import X` 与
`import whisper_stt_service.service as svc; svc.X` 两种写法都兼容。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# 公开属性 → (模块路径, 模块内属性名)。只在被访问时才真正导入。
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "WorkerRuntime": ("whisper_stt_service.service.runtime", "WorkerRuntime"),
    "recover_claimed_to_queued": (
        "whisper_stt_service.service.runtime",
        "recover_claimed_to_queued",
    ),
}


def __getattr__(name: str) -> Any:
    """惰性解析包内导出：首次访问时导入目标模块并缓存到包命名空间。"""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr_name = target
    value = getattr(import_module(module_path), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """把惰性导出一并列进 dir()，便于交互式排查与自动化工具发现。"""

    return sorted(set(globals()) | set(_LAZY_EXPORTS) | {"build_app"})


def build_app():
    """延迟导入应用装配，避免包初始化阶段形成循环依赖。"""

    from whisper_stt_service.service.bootstrap import build_app as _build_app

    return _build_app()


__all__ = ["WorkerRuntime", "build_app", "recover_claimed_to_queued"]