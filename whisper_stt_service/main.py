"""服务入口：装配 DB/Repository/Workers，并暴露 ASGI app。"""

from whisper_stt_service.service.bootstrap import bind_exit_target, build_app


# 在模块导入时创建 app，确保 Uvicorn 能直接发现并加载应用实例。
app = build_app()


def bind_server(server) -> None:
    """把 uvicorn Server 绑定给运行时退出回调（给自定义启动脚本使用）。

    以 `uvicorn whisper_stt_service.main:app` 方式启动时无法拿到 Server 实例，
    运行时会回退 SIGTERM；需要 should_exit 优雅退出的脚本请改用
    `uvicorn.Server(...)` 并调用本函数。
    """

    bind_exit_target(app, server)
