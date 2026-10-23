"""服务启动器：把 [security] bind_host/bind_port 适配到 uvicorn。

设计目标（SPEC H / D4）：

- 监听地址不再写死在脚本里，而是从配置文件的 `[security]` 读取，
  缺省回落到安全的 `127.0.0.1:18000`；
- 使用 `uvicorn.Server` 而不是命令行字符串启动，从而能在启动后调用
  :func:`whisper_stt_service.main.bind_server` 把 server 绑定给运行时的
  退出回调，drain 完成后由运行时置 `server.should_exit`（SPEC A.6）；
- 不修改 `main.py`/`bootstrap.py`，只消费它们已有的公开接口。

本模块刻意不导入 FastAPI/uvicorn（除真正 `run()` 时），以便单元测试在
不启动服务的前提下校验配置解析逻辑。
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# 安全默认值：只监听回环地址，端口与 SPEC 一致。
DEFAULT_BIND_HOST = "127.0.0.1"
DEFAULT_BIND_PORT = 18000


@dataclass(frozen=True)
class BindConfig:
    """一次启动所用的监听配置。"""

    host: str
    port: int
    config_path: Path
    reload: bool = False

    def as_dict(self) -> dict[str, object]:
        """返回可 JSON 序列化的描述（供 --print-config 与日志使用）。"""

        return {
            "host": self.host,
            "port": self.port,
            "config_path": str(self.config_path),
            "reload": self.reload,
        }


def read_bind_settings(config_path: Path) -> tuple[str, int]:
    """从配置文件读取 `[security] bind_host/bind_port`，缺失或非法时回落默认值。

    只读 configparser，不依赖 core.config：`[security]` 的新键由 config owner
    逐步补齐，启动脚本必须在旧配置上也能工作（AGENTS：旧配置必须可加载）。
    """

    parser = configparser.ConfigParser()
    # 允许配置里出现 % 等字符，避免插值语法误伤。
    parser.read(str(config_path), encoding="utf-8")
    host = DEFAULT_BIND_HOST
    port = DEFAULT_BIND_PORT
    if parser.has_option("security", "bind_host"):
        value = parser.get("security", "bind_host", fallback="").strip()
        if value:
            host = value
    if parser.has_option("security", "bind_port"):
        raw = parser.get("security", "bind_port", fallback="").strip()
        try:
            candidate = int(raw)
        except (TypeError, ValueError):
            candidate = DEFAULT_BIND_PORT
        # 端口非法（越界/非正数）时同样回落到默认值，避免启动即失败。
        if 1 <= candidate <= 65535:
            port = candidate
    return host, port


def resolve_bind_config(
    *,
    config_path: Path | str | None = None,
    host: str | None = None,
    port: int | None = None,
    reload: bool = False,
) -> BindConfig:
    """解析最终监听配置：CLI 覆盖 > 配置文件 > 安全默认值。"""

    path = Path(config_path) if config_path else Path(
        os.getenv("WHISPER_STT_CONFIG", "config.ini")
    )
    path = path.expanduser()
    file_host, file_port = read_bind_settings(path)
    resolved_host = (host or "").strip() or file_host
    return BindConfig(
        host=resolved_host,
        port=int(port) if port is not None else file_port,
        config_path=path,
        reload=bool(reload),
    )


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 解析器（`--help` 只读，不触发任何启动副作用）。"""

    parser = argparse.ArgumentParser(
        prog="run_service.sh",
        description="按 [security] bind_host/bind_port 启动 whisper_stt_service",
    )
    parser.add_argument("--host", default=None, help="覆盖绑定地址")
    parser.add_argument("--port", type=int, default=None, help="覆盖绑定端口")
    parser.add_argument("--config", default=None, help="config.ini 路径")
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="只打印解析后的监听配置（JSON），不启动服务",
    )
    return parser


def pin_config_environment(bind: BindConfig) -> Path:
    """把选定的配置文件绝对路径钉进 `WHISPER_STT_CONFIG`，并返回该路径。

    重要性（单一配置源）：\`bootstrap._resolve_settings()\` 只认环境变量
    `WHISPER_STT_CONFIG`（缺省 `config.ini`）。若 `--config` 只用于解析绑定地址、
    不写回环境，就会出现“CLI 选中 A 文件、bootstrap 实际读 B 文件”的分裂，
    监听端口 / 鉴权 / DB 路径 / 白名单全部不同源。

    因此这里必须在 **import `whisper_stt_service.main`（其导入即执行 build_app）
    之前**完成注入。绝对路径化也避免依赖启动时的 cwd。
    """

    resolved = Path(bind.config_path).expanduser().resolve()
    os.environ["WHISPER_STT_CONFIG"] = str(resolved)
    return resolved


def _purge_cached_main() -> None:
    """移除可能已被缓存的 `whisper_stt_service.main` 及其 app。

    若本进程内 `main` 早已被 import（例如测试或嵌入式调用先加载了默认配置），
    其模块级 `app = build_app()` 会固化“旧配置”的实例，直接复用会造成配置错配。
    这里在切换配置后剔除缓存，强制下一次 import 用新环境变量重新构建。
    """

    sys.modules.pop("whisper_stt_service.main", None)

def run(bind: BindConfig) -> int:
    """用 uvicorn.Server 启动服务，并把 server 绑定给运行时退出回调。

    `main.bind_server` 已由 main.py 提供；reload 模式走 `uvicorn.run` 字符串
    入口（此时无法注入 Server 实例，drain 回调回退 SIGTERM）。

    配置同源：本函数第一件事就是把选定配置绝对路径写入 `WHISPER_STT_CONFIG`，
    并剔除可能缓存的 `main` 模块——顺序绝不能颠倒，因为
    `import whisper_stt_service.main` 会在导入期执行 `build_app()`。
    """

    # 1) 先钉住配置来源（必须早于任何 main/bootstrap 导入）。
    resolved_config = pin_config_environment(bind)
    _purge_cached_main()

    if bind.reload:
        import uvicorn

        uvicorn.run(
            "whisper_stt_service.main:app",
            host=bind.host,
            port=bind.port,
            reload=True,
        )
        return 0

    import uvicorn

    # 2) 之后才导入 main：此时 build_app 读到的就是选定文件。
    from whisper_stt_service.main import app, bind_server

    config = uvicorn.Config(
        app=app,
        host=bind.host,
        port=bind.port,
        log_level=os.getenv("WHISPER_STT_LOG_LEVEL", "info").lower(),
    )
    server = uvicorn.Server(config)
    # 必须在 run() 之前绑定：运行时的退出句柄在 lifespan 启动后才会被使用。
    bind_server(server)
    server.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    """启动器入口：解析参数并启动服务（`--print-config` 只打印配置）。"""

    args = build_parser().parse_args(argv)
    bind = resolve_bind_config(
        config_path=args.config,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    if args.print_config:
        print(json.dumps(bind.as_dict(), ensure_ascii=False))
        return 0
    return run(bind)


if __name__ == "__main__":  # pragma: no cover - 由脚本调用
    raise SystemExit(main())
