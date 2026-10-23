"""服务装配与生命周期绑定。"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI

from whisper_stt_service.api import create_app
from whisper_stt_service.core.config import (
    Settings,
    TranslationSettings,
    ensure_config_file,
    find_missing_required_fields,
    load_settings,
    validate_startup_security,
)
from whisper_stt_service.core.progress import ProgressStore
from whisper_stt_service.core.security import SecurityError
from whisper_stt_service.repo.database import Database
from whisper_stt_service.repo.job_repository import JobRepository
from whisper_stt_service.service.logging_config import configure_logging
from whisper_stt_service.service.metrics import MetricsState, build_metrics
from whisper_stt_service.service.model_cache import clear_model_cache, configure_model_cache
from whisper_stt_service.service.runtime import WorkerRuntime


LOGGER = logging.getLogger(__name__)


def _resolve_settings() -> tuple[Settings, Path]:
    """解析服务配置文件路径并加载配置。"""

    config_path = Path(os.getenv("WHISPER_STT_CONFIG", "config.ini")).resolve()
    default_example_path = (
        Path(__file__).resolve().parent.parent.parent / "config.example.ini"
    )
    example_path = config_path.with_name("config.example.ini")
    if not example_path.is_file():
        # 支持自定义 WHISPER_STT_CONFIG 路径时回退到仓库内默认模板。
        example_path = default_example_path
    if ensure_config_file(config_path=config_path, example_path=example_path):
        LOGGER.warning(
            "config file not found, created default from example: %s",
            config_path,
        )

    missing_fields = find_missing_required_fields(config_path)
    if missing_fields:
        detail = ", ".join(
            f"{section}.{option}"
            for section, options in missing_fields.items()
            for option in options
        )
        LOGGER.error("missing required config entries: %s", detail)

    settings = load_settings(config_path)
    return settings, config_path


def _resolve_runtime_path(base_path: Path, value: Path) -> Path:
    """把相对路径解释为基于配置文件目录的路径。"""

    if value.is_absolute():
        return value
    return (base_path.parent / value).resolve()


def _build_stage_effective_defaults(
    settings: Settings, config_path: Path
) -> dict[str, dict]:
    """集中构造各阶段默认配置，便于后续扩展更多阶段。"""

    return {
        "extract": {},
        "stt": {
            # 与 stt_whisperx 保持一致：模型路径可被 WHISPER_STT_MODEL 覆盖，
            # 否则两侧会读到不同模型（stt 段无 model 键而 whisperx 段有）。
            "model": os.getenv(
                "WHISPER_STT_MODEL",
                str(_resolve_runtime_path(config_path, settings.runtime.model_path)),
            ),
            "device": settings.stt.device,
            "compute_type": settings.stt.compute_type,
            "batch_size": settings.stt.batch_size,
            "beam_size": settings.stt.beam_size,
            "best_of": settings.stt.best_of,
            "patience": settings.stt.patience,
            "condition_on_previous_text": settings.stt.condition_on_previous_text,
            "vad_filter": settings.stt.vad_filter,
            "vad_threshold": settings.stt.vad_threshold,
            "vad_min_speech_duration_ms": settings.stt.vad_min_speech_duration_ms,
            "vad_max_speech_duration_s": settings.stt.vad_max_speech_duration_s,
            "vad_min_silence_duration_ms": settings.stt.vad_min_silence_duration_ms,
            "vad_speech_pad_ms": settings.stt.vad_speech_pad_ms,
            "no_speech_threshold": settings.stt.no_speech_threshold,
            "compression_ratio_threshold": settings.stt.compression_ratio_threshold,
            "log_prob_threshold": settings.stt.log_prob_threshold,
            "hallucination_silence_threshold": settings.stt.hallucination_silence_threshold,
            "initial_prompt": settings.stt.initial_prompt,
            "hotwords": settings.stt.hotwords,
        },
        "stt_whisperx": {
            "model": os.getenv(
                "WHISPER_STT_MODEL",
                str(_resolve_runtime_path(config_path, Path(settings.stt_whisperx.model))),
            ),
            "device": settings.stt_whisperx.device,
            "compute_type": settings.stt_whisperx.compute_type,
            "batch_size": settings.stt_whisperx.batch_size,
            "vad_config_path": str(
                _resolve_runtime_path(
                    config_path, settings.stt_whisperx.vad_config_path
                )
            ),
            "align_model_root": str(
                _resolve_runtime_path(
                    config_path, settings.stt_whisperx.align_model_root
                )
            ),
            "align_enabled": settings.stt_whisperx.align_enabled,
            "vad_backend": settings.stt_whisperx.vad_backend,
            "vad_onset": settings.stt_whisperx.vad_onset,
            "vad_offset": settings.stt_whisperx.vad_offset,
            "local_files_only": settings.stt_whisperx.local_files_only,
        },
        "translate": {
            # SPEC C.2：stage 默认值来自 [translation]（job_config 仍可覆盖）。
            **_translation_settings(settings).as_job_config(),
        },
    }


def _translation_settings(settings: Settings) -> TranslationSettings:
    """读取 [translation] 解析结果；老 Settings 实例缺少该段时回退默认值。"""

    return getattr(settings, "translation", None) or TranslationSettings()


def _resolve_security_roots(
    settings: Settings, config_path: Path
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """把 [security] 的路径白名单解析为绝对路径（相对路径基于配置文件目录）。

    SPEC H：`allowed_video_roots` 为空表示"不限制视频目录"，
    `allowed_copy_back_roots` 为空表示"仅允许视频所在目录"。这里只在非空时
    做解析，保持空值语义不变。
    """

    def _resolve_all(values: tuple[str, ...]) -> tuple[str, ...]:
        """逐项解析：绝对路径原样保留，相对路径基于配置文件目录展开。"""

        resolved: list[str] = []
        for raw in values:
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = config_path.parent / candidate
            resolved.append(str(candidate.resolve()))
        return tuple(resolved)

    return (
        _resolve_all(settings.security.allowed_video_roots),
        _resolve_all(settings.security.allowed_copy_back_roots),
    )


def _validate_security_or_fail(settings: Settings) -> None:
    """启动期安全校验（fail-closed）：不满足即拒绝启动并给出明确错误码。

    - require_token=true 且 token 为空 → api_token_missing；
    - allow_placeholder_token=false 且 token 为占位值 → api_token_placeholder；
    - 鉴权关闭（token 为空且 require_token=false）→ WARN，由调用方显式担责。
    错误信息只包含错误码与说明，绝不输出 token 内容。
    """

    try:
        validate_startup_security(settings)
    except SecurityError as exc:
        LOGGER.error(
            "refusing to start: insecure [security] config: code=%s detail=%s "
            "(set a real api_token or explicitly require_token=false)",
            exc.code,
            exc.detail or "-",
        )
        raise
    if not settings.security.api_token:
        LOGGER.warning(
            "refusing-to-authenticate: [security] api_token is empty and "
            "require_token=false; API is open to anyone who can reach it"
        )


def build_app() -> FastAPI:
    """创建完整应用并绑定生命周期。"""

    settings, config_path = _resolve_settings()
    security_video_roots, security_copy_back_roots = _resolve_security_roots(
        settings, config_path
    )
    db_path = _resolve_runtime_path(config_path, settings.runtime.db_path)
    log_root = _resolve_runtime_path(config_path, settings.runtime.log_root)
    artifact_root = _resolve_runtime_path(config_path, settings.runtime.artifact_root)

    # G：先装配日志，保证后续所有启动日志都落到 service.log；
    # 等级/JSONL/路径全部来自 [observability]（env WHISPER_STT_LOG_LEVEL 仍可覆盖）。
    observability = getattr(settings, "observability", None)
    configure_logging(
        log_dir=log_root,
        level=getattr(observability, "log_level", None),
        jsonl=getattr(observability, "log_jsonl", None),
    )

    # H：安全默认值 fail-closed —— 占位/缺失 token 直接拒绝启动。
    _validate_security_or_fail(settings)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    # F：按配置启用 WAL（网络盘等场景置 false 回退 DELETE）；探测结果会影响
    # 所有连接，因此必须在建表/迁移前完成。
    db = Database(
        db_path, wal_enabled=bool(getattr(settings.runtime, "sqlite_wal_enabled", True))
    )
    db.init_schema()
    LOGGER.info(
        "sqlite configured: db_path=%s journal_mode=%s", db_path, db.journal_mode
    )
    for name, plan in db.check_query_plans().items():
        LOGGER.info("sqlite query plan: %s -> %s", name, plan)
    repo = JobRepository(
        db,
        stage_max_retries={
            "extract": settings.retry.extract_max_retries,
            "stt": settings.retry.stt_max_retries,
            "stt_whisperx": settings.retry.stt_whisperx_max_retries,
            "translate": settings.retry.translate_max_retries,
        },
        stage_timeouts={
            "extract": settings.timeouts.extract_timeout_sec,
            "stt": settings.timeouts.stt_timeout_sec,
            "stt_whisperx": settings.timeouts.stt_whisperx_timeout_sec,
            "translate": settings.timeouts.translate_timeout_sec,
        },
        stage_effective_defaults=_build_stage_effective_defaults(settings, config_path),
        log_root=log_root,
        artifact_root=artifact_root,
        # H：路径白名单（为空时保持"不限视频 / copy_back 仅视频同目录"）。
        allowed_video_roots=security_video_roots or None,
        allowed_copy_back_roots=security_copy_back_roots or None,
    )
    progress_store = ProgressStore(settings.runtime.progress_ttl_sec)
    model_path_cfg = _resolve_runtime_path(config_path, settings.runtime.model_path)
    model_path = os.getenv("WHISPER_STT_MODEL", str(model_path_cfg))

    # G：/metrics 是否注册由配置决定；关闭时不构建指标对象（也不注册路由）。
    metrics_enabled = bool(getattr(observability, "metrics_enabled", True))
    metrics: MetricsState | None = None
    if metrics_enabled:
        metrics = build_metrics(
            repo_factory=lambda: repo,
            stages=("extract", "stt", "stt_whisperx", "translate"),
        )

    # D：模型缓存容量与并发上限（0 = 不缓存）；命中/未命中计入同一 MetricsState。
    cache_stats = configure_model_cache(
        cache_size=settings.runtime.model_cache_size,
        parallelism=settings.runtime.model_parallelism,
        metrics=metrics,
    )
    LOGGER.info("model cache configured: %s", cache_stats)
    # A.6：退出方式改为注入回调，bootstrap 在 lifespan 里把它接到 uvicorn 的
    # should_exit；未注入回调时运行时回退 SIGTERM。
    exit_handle = SimpleNamespace(target=None)

    def _request_exit() -> None:
        """退出回调：优先让 uvicorn 优雅退出，未就绪时回退 SIGTERM。"""

        server = exit_handle.target
        if server is None:
            raise RuntimeError("exit_target_not_bound")
        LOGGER.warning("drain complete, requesting server exit")
        server.should_exit = True

    runtime = WorkerRuntime(
        repo=repo,
        progress_store=progress_store,
        settings=settings,
        config_path=config_path,
        model_path=model_path,
        exit_callback=_request_exit,
        # G：运行时指标句柄与 /metrics、model_cache 使用同一实例。
        metrics=metrics,
        checkpoint_ttl_sec=getattr(
            getattr(settings, "translation", None), "progress_ttl_sec", None
        ),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """应用生命周期：启动运行时；退出时停止运行时并释放模型缓存（SPEC D）。"""

        runtime.start()
        LOGGER.info(
            "runtime started: startup_recovery=%s drain_timeout_sec=%s "
            "heartbeat_interval_sec=%.1f",
            runtime.startup_stats(),
            runtime.drain_timeout_sec,
            runtime.heartbeat_interval_sec,
        )
        try:
            yield
        finally:
            runtime.stop()
            # D：进程退出时清空模型缓存并释放（lifespan finally，保证 SIGTERM 也执行）。
            cleared = clear_model_cache(force=True)
            LOGGER.info("model cache cleared on shutdown: %s", cleared)

    app = create_app(
        repo=repo,
        progress_store=progress_store,
        runtime=runtime,
        # H/G：settings 统一承载 token、docs、metrics 开关与 readyz 所需的 runtime 路径。
        settings=settings,
        metrics=metrics,
        metrics_enabled=metrics_enabled,
    )
    app.router.lifespan_context = lifespan
    app.state.exit_handle = exit_handle
    return app


def bind_exit_target(app: Any, server: Any) -> None:
    """把 uvicorn Server 绑定为运行时退出回调目标（由 main 在启动后调用）。

    运行时只依赖注入回调，不直接持有 uvicorn 对象；此函数把 server 实例
    交给 bootstrap 内部的退出句柄，之后 drain 完成即置 server.should_exit。
    """

    handle = getattr(app.state, "exit_handle", None)
    if handle is None:
        LOGGER.warning("exit target binding skipped: exit_handle missing")
        return
    handle.target = server
    LOGGER.info("exit target bound: %s", type(server).__name__)