"""流水线服务配置加载模块。

该模块把 config.ini 转换成强类型 dataclass，避免业务层直接使用字符串键。
"""

from __future__ import annotations

import configparser
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkerSettings:
    """Worker 池与调度轮询配置。"""

    extract_workers: int
    stt_workers: int
    stt_whisperx_workers: int
    translate_workers: int
    scheduler_interval_sec: int
    poll_interval_sec: int


@dataclass(frozen=True)
class TimeoutSettings:
    """各阶段执行超时与领取租约超时配置。"""

    extract_timeout_sec: int
    stt_timeout_sec: int
    stt_whisperx_timeout_sec: int
    translate_timeout_sec: int
    lease_timeout_sec: int
    # A.6：drain 请求后等待在途任务结束的上限，超时强制回滚本进程 claimed 并退出。
    drain_timeout_sec: int = 300


@dataclass(frozen=True)
class RetrySettings:
    """各阶段最大重试次数配置。"""

    extract_max_retries: int
    stt_max_retries: int
    stt_whisperx_max_retries: int
    translate_max_retries: int


@dataclass(frozen=True)
class RuntimeSettings:
    """运行期路径与内存进度保留策略配置。"""

    db_path: Path
    progress_ttl_sec: int
    log_root: Path
    artifact_root: Path
    artifact_cleanup_enabled: bool
    artifact_cleanup_interval_sec: int
    artifact_cleanup_statuses: tuple[str, ...]
    model_path: Path
    # A.5：启动恢复是否消耗一次重试预算（false 仅用于调试，保持旧行为）。
    recovery_counts_as_attempt: bool = True
    # SPEC D：进程内模型缓存容量（0 = 不缓存，仅调试）与单模型并发上限。
    model_cache_size: int = 2
    model_parallelism: int = 1
    # F：SQLite 连接是否启用 WAL（网络盘等场景可关闭，回退 DELETE）。
    sqlite_wal_enabled: bool = True


@dataclass(frozen=True)
class SttSettings:
    """STT 阶段转写参数配置。"""

    device: str
    compute_type: str
    batch_size: int
    beam_size: int
    best_of: int
    patience: float
    condition_on_previous_text: bool
    vad_filter: bool
    vad_threshold: float
    vad_min_speech_duration_ms: int
    vad_max_speech_duration_s: float
    vad_min_silence_duration_ms: int
    vad_speech_pad_ms: int
    no_speech_threshold: float
    compression_ratio_threshold: float
    log_prob_threshold: float
    hallucination_silence_threshold: float
    initial_prompt: str
    hotwords: str


@dataclass(frozen=True)
class SttWhisperxSettings:
    """WhisperX STT 阶段转写参数配置。"""

    model: str
    device: str
    compute_type: str
    batch_size: int
    vad_config_path: Path
    align_model_root: Path
    align_enabled: bool
    vad_backend: str
    vad_onset: float
    vad_offset: float
    local_files_only: bool


@dataclass(frozen=True)
class SecuritySettings:
    """API 全局访问控制配置。

    同时覆盖暴露面参数（SPEC H / D4，全部可配置、默认安全）。

    - api_token：空 = 鉴权关闭（启动 WARN）。
    - require_token：True 且 token 为空 → 启动失败（fail-closed）。
    - allow_placeholder_token：False 时占位 token（含 `<>` 或示例值）→ 启动失败。
    - docs_enabled：关闭时 /docs、/redoc、/openapi.json 返回 404；开启仍受 token 保护。
    - allowed_video_roots / allowed_copy_back_roots：非空时路径必须 resolve 后位于其下。
    - bind_host / bind_port：供启动脚本读取（默认仅监听回环地址）。
    """

    api_token: str
    require_token: bool = True
    allow_placeholder_token: bool = False
    docs_enabled: bool = False
    allowed_video_roots: tuple[str, ...] = ()
    allowed_copy_back_roots: tuple[str, ...] = ()
    bind_host: str = "127.0.0.1"
    bind_port: int = 18000


@dataclass(frozen=True)
class ObservabilitySettings:
    """可观测性配置（SPEC G，对应 [observability] 段）。

    - log_level：应用与 uvicorn logger 等级，非法值回退 INFO。
    - log_jsonl：True 时落盘的 service.log 每行一个 JSON 对象。
    - metrics_enabled：False 时不注册 /metrics（返回 404），其余观测能力保留。
    """

    log_level: str = "INFO"
    log_jsonl: bool = False
    metrics_enabled: bool = True


@dataclass(frozen=True)
class TranslationSettings:
    """翻译链路配置（SPEC C / D2，对应 [translation] 段）。

    全部键都有默认值，旧配置缺键可直接加载；`parallel` 默认 2（D2 拍板）。
    `timeout_sec` 是服务计划内的单批 HTTP 超时，`retry` 是批次重试次数。
    """

    parallel: int = 2
    batch_size: int = 200
    max_chars: int = 12000
    chunk_minutes: int = 20
    timeout_sec: int = 240
    retry: int = 10
    request_interval: float = 1.0
    disable_proxy: bool = True
    copy_back: str = "__video_dir__"
    checkpoint_enabled: bool = True
    strict_ids: bool = False
    progress_ttl_sec: int = 604800

    def as_job_config(self) -> dict[str, object]:
        """转成 run_translate 的 job_config 形态（作为 stage 生效默认值）。"""

        return {
            "parallel": self.parallel,
            "batch_size": self.batch_size,
            "max_chars": self.max_chars,
            "chunk_minutes": self.chunk_minutes,
            "timeout_sec": self.timeout_sec,
            "retry": self.retry,
            "request_interval": self.request_interval,
            "disable_proxy": self.disable_proxy,
            "copy_back": self.copy_back,
            "checkpoint_enabled": self.checkpoint_enabled,
            "strict_ids": self.strict_ids,
            "progress_ttl_sec": self.progress_ttl_sec,
        }


@dataclass(frozen=True)
class Settings:
    """服务总配置对象（不可变）。"""

    workers: WorkerSettings
    timeouts: TimeoutSettings
    retry: RetrySettings
    runtime: RuntimeSettings
    stt: SttSettings
    stt_whisperx: SttWhisperxSettings
    security: SecuritySettings
    # 新增段带默认值：旧测试/旧调用方直接构造 Settings 时无需改动。
    observability: ObservabilitySettings = ObservabilitySettings()
    translation: TranslationSettings = TranslationSettings()


# 启动前必须存在的配置项（按 section/option 列举）。
REQUIRED_CONFIG_OPTIONS: dict[str, tuple[str, ...]] = {
    "workers": (
        "extract_workers",
        "stt_workers",
        "stt_whisperx_workers",
        "translate_workers",
    ),
    "timeouts": (
        "extract_timeout_sec",
        "stt_timeout_sec",
        "stt_whisperx_timeout_sec",
        "translate_timeout_sec",
        "lease_timeout_sec",
    ),
    "retry": (
        "extract_max_retries",
        "stt_max_retries",
        "stt_whisperx_max_retries",
        "translate_max_retries",
    ),
    "runtime": (
        "db_path",
        "log_root",
        "model_path",
    ),
    "llm": (
        "base_url",
        "api_key",
        "model",
    ),
    "security": ("api_token",),
}


def ensure_config_file(config_path: Path, example_path: Path) -> bool:
    """当目标配置不存在时，按 example 自动创建并返回 True。"""

    if config_path.exists():
        return False
    if not example_path.is_file():
        raise FileNotFoundError(f"config example not found: {example_path}")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(example_path, config_path)
    return True


def find_missing_required_fields(config_path: Path) -> dict[str, list[str]]:
    """检测配置文件缺失的 section/option，供启动日志输出。"""

    cp = configparser.ConfigParser()
    cp.read(config_path, encoding="utf-8")
    missing: dict[str, list[str]] = {}
    for section, options in REQUIRED_CONFIG_OPTIONS.items():
        if not cp.has_section(section):
            missing[section] = list(options)
            continue
        missing_options = [
            option for option in options if not cp.has_option(section, option)
        ]
        if missing_options:
            missing[section] = missing_options
    return missing


def load_settings(config_path: Path) -> Settings:
    """读取 config.ini 并返回强类型配置。

    对必填字段采用失败即报错策略，避免服务在配置不完整时继续启动。
    """

    # 先读取原始配置，再按 section 映射为明确的数据结构。
    cp = configparser.ConfigParser()
    cp.read(config_path, encoding="utf-8")
    workers = WorkerSettings(
        extract_workers=cp.getint("workers", "extract_workers"),
        stt_workers=cp.getint("workers", "stt_workers"),
        stt_whisperx_workers=cp.getint(
            "workers",
            "stt_whisperx_workers",
            fallback=cp.getint("workers", "stt_workers"),
        ),
        translate_workers=cp.getint("workers", "translate_workers"),
        scheduler_interval_sec=cp.getint(
            "workers", "scheduler_interval_sec", fallback=180
        ),
        poll_interval_sec=cp.getint("workers", "poll_interval_sec", fallback=1),
    )
    timeouts = TimeoutSettings(
        extract_timeout_sec=cp.getint("timeouts", "extract_timeout_sec"),
        stt_timeout_sec=cp.getint("timeouts", "stt_timeout_sec"),
        stt_whisperx_timeout_sec=cp.getint(
            "timeouts",
            "stt_whisperx_timeout_sec",
            fallback=cp.getint("timeouts", "stt_timeout_sec"),
        ),
        translate_timeout_sec=cp.getint("timeouts", "translate_timeout_sec"),
        lease_timeout_sec=cp.getint("timeouts", "lease_timeout_sec"),
        drain_timeout_sec=max(
            cp.getint("timeouts", "drain_timeout_sec", fallback=300), 1
        ),
    )
    retry = RetrySettings(
        extract_max_retries=cp.getint("retry", "extract_max_retries"),
        stt_max_retries=cp.getint("retry", "stt_max_retries"),
        stt_whisperx_max_retries=cp.getint(
            "retry",
            "stt_whisperx_max_retries",
            fallback=cp.getint("retry", "stt_max_retries"),
        ),
        translate_max_retries=cp.getint("retry", "translate_max_retries"),
    )
    runtime = RuntimeSettings(
        db_path=Path(cp.get("runtime", "db_path")),
        progress_ttl_sec=cp.getint("runtime", "progress_ttl_sec", fallback=3600),
        log_root=Path(cp.get("runtime", "log_root")),
        artifact_root=Path(
            cp.get(
                "runtime",
                "artifact_root",
                fallback=str(Path(cp.get("runtime", "log_root")).parent / "artifacts"),
            )
        ),
        artifact_cleanup_enabled=cp.getboolean(
            "runtime", "artifact_cleanup_enabled", fallback=True
        ),
        artifact_cleanup_interval_sec=max(
            cp.getint("runtime", "artifact_cleanup_interval_sec", fallback=3600), 60
        ),
        artifact_cleanup_statuses=tuple(
            sorted(
                {
                    token.strip()
                    for token in cp.get(
                        "runtime",
                        "artifact_cleanup_statuses",
                        fallback="succeeded",
                    ).split(",")
                    if token.strip()
                }
                or {"succeeded"}
            )
        ),
        model_path=Path(
            cp.get("runtime", "model_path", fallback="models/faster-whisper-small")
        ),
        recovery_counts_as_attempt=cp.getboolean(
            "runtime", "recovery_counts_as_attempt", fallback=True
        ),
        sqlite_wal_enabled=cp.getboolean(
            "runtime", "sqlite_wal_enabled", fallback=True
        ),
        # SPEC D：缓存容量 0 = 不缓存（仅调试）；并发上限至少 1。
        model_cache_size=max(cp.getint("runtime", "model_cache_size", fallback=2), 0),
        model_parallelism=max(cp.getint("runtime", "model_parallelism", fallback=1), 1),
    )
    stt = SttSettings(
        device=cp.get("stt", "device", fallback="auto"),
        compute_type=cp.get("stt", "compute_type", fallback="auto"),
        batch_size=max(cp.getint("stt", "batch_size", fallback=8), 1),
        beam_size=max(cp.getint("stt", "beam_size", fallback=5), 1),
        best_of=max(cp.getint("stt", "best_of", fallback=5), 1),
        patience=max(cp.getfloat("stt", "patience", fallback=1.0), 0.1),
        condition_on_previous_text=cp.getboolean(
            "stt", "condition_on_previous_text", fallback=False
        ),
        vad_filter=cp.getboolean("stt", "vad_filter", fallback=True),
        vad_threshold=min(
            max(cp.getfloat("stt", "vad_threshold", fallback=0.45), 0.01), 0.99
        ),
        vad_min_speech_duration_ms=max(
            cp.getint("stt", "vad_min_speech_duration_ms", fallback=200), 50
        ),
        vad_max_speech_duration_s=max(
            cp.getfloat("stt", "vad_max_speech_duration_s", fallback=18.0), 1.0
        ),
        vad_min_silence_duration_ms=max(
            cp.getint("stt", "vad_min_silence_duration_ms", fallback=700), 50
        ),
        vad_speech_pad_ms=max(cp.getint("stt", "vad_speech_pad_ms", fallback=300), 0),
        no_speech_threshold=min(
            max(cp.getfloat("stt", "no_speech_threshold", fallback=0.6), 0.01), 0.99
        ),
        compression_ratio_threshold=max(
            cp.getfloat("stt", "compression_ratio_threshold", fallback=2.2), 0.1
        ),
        log_prob_threshold=cp.getfloat("stt", "log_prob_threshold", fallback=-1.0),
        hallucination_silence_threshold=max(
            cp.getfloat("stt", "hallucination_silence_threshold", fallback=1.5), 0.0
        ),
        initial_prompt=cp.get("stt", "initial_prompt", fallback="").strip(),
        hotwords=cp.get("stt", "hotwords", fallback="").strip(),
    )
    stt_whisperx = SttWhisperxSettings(
        model=cp.get("stt_whisperx", "model", fallback=str(runtime.model_path)).strip(),
        device=cp.get("stt_whisperx", "device", fallback=stt.device),
        compute_type=cp.get("stt_whisperx", "compute_type", fallback=stt.compute_type),
        batch_size=max(
            cp.getint("stt_whisperx", "batch_size", fallback=stt.batch_size), 1
        ),
        vad_config_path=Path(
            cp.get(
                "stt_whisperx",
                "vad_config_path",
                fallback="models/whisperx/vad/pyannote/config.yaml",
            )
        ),
        align_model_root=Path(
            cp.get(
                "stt_whisperx",
                "align_model_root",
                fallback="models/whisperx/align",
            )
        ),
        align_enabled=cp.getboolean("stt_whisperx", "align_enabled", fallback=True),
        vad_backend=cp.get("stt_whisperx", "vad_backend", fallback="pyannote").strip(),
        vad_onset=min(
            max(cp.getfloat("stt_whisperx", "vad_onset", fallback=0.35), 0.01), 0.99
        ),
        vad_offset=min(
            max(cp.getfloat("stt_whisperx", "vad_offset", fallback=0.2), 0.01), 0.99
        ),
        local_files_only=cp.getboolean(
            "stt_whisperx", "local_files_only", fallback=True
        ),
    )
    security = _load_security_settings(cp)
    observability = _load_observability_settings(cp)
    translation = _load_translation_settings(cp)
    return Settings(
        workers=workers,
        timeouts=timeouts,
        retry=retry,
        runtime=runtime,
        stt=stt,
        stt_whisperx=stt_whisperx,
        security=security,
        observability=observability,
        translation=translation,
    )


def _split_path_list(raw: str | None) -> tuple[str, ...]:
    """把逗号分隔的路径白名单拆成去空白的元组（保持原有顺序）。"""

    return tuple(
        token.strip() for token in str(raw or "").split(",") if token.strip()
    )


def _load_security_settings(cp: configparser.ConfigParser) -> SecuritySettings:
    """解析 [security] 段；缺省值即安全默认值（SPEC H fail-closed）。

    - require_token 默认 true、allow_placeholder_token 默认 false、docs 默认关闭；
    - bind_host 默认 127.0.0.1（公网部署必须显式修改并前置反向代理）；
    - bind_port 非法时回退 18000。
    """

    port_raw = cp.get("security", "bind_port", fallback="18000").strip()
    try:
        bind_port = int(port_raw)
    except ValueError:
        bind_port = 18000
    if not (1 <= bind_port <= 65535):
        bind_port = 18000
    bind_host = cp.get("security", "bind_host", fallback="127.0.0.1").strip() or "127.0.0.1"
    return SecuritySettings(
        api_token=cp.get("security", "api_token", fallback="").strip(),
        require_token=cp.getboolean("security", "require_token", fallback=True),
        allow_placeholder_token=cp.getboolean(
            "security", "allow_placeholder_token", fallback=False
        ),
        docs_enabled=cp.getboolean("security", "docs_enabled", fallback=False),
        allowed_video_roots=_split_path_list(
            cp.get("security", "allowed_video_roots", fallback="")
        ),
        allowed_copy_back_roots=_split_path_list(
            cp.get("security", "allowed_copy_back_roots", fallback="")
        ),
        bind_host=bind_host,
        bind_port=bind_port,
    )


def _load_observability_settings(cp: configparser.ConfigParser) -> ObservabilitySettings:
    """解析 [observability] 段；缺段/缺键时使用安全且实用的默认值。"""

    level = cp.get("observability", "log_level", fallback="INFO").strip().upper() or "INFO"
    return ObservabilitySettings(
        log_level=level,
        log_jsonl=cp.getboolean("observability", "log_jsonl", fallback=False),
        metrics_enabled=cp.getboolean("observability", "metrics_enabled", fallback=True),
    )


def _load_translation_settings(cp: configparser.ConfigParser) -> TranslationSettings:
    """解析 [translation] 段（SPEC C 全键）；旧配置缺键即取默认值。

    注意：并行度默认 2（D2 决策），旧配置里显式写的 `parallel = 16` 仍会被读成
    16，需在发布说明中提示改为 2。
    """

    defaults = TranslationSettings()
    return TranslationSettings(
        parallel=max(cp.getint("translation", "parallel", fallback=defaults.parallel), 1),
        batch_size=max(
            cp.getint("translation", "batch_size", fallback=defaults.batch_size), 1
        ),
        max_chars=max(
            cp.getint("translation", "max_chars", fallback=defaults.max_chars), 1
        ),
        chunk_minutes=max(
            cp.getint("translation", "chunk_minutes", fallback=defaults.chunk_minutes), 1
        ),
        timeout_sec=max(
            cp.getint("translation", "timeout", fallback=defaults.timeout_sec), 1
        ),
        retry=max(cp.getint("translation", "retry", fallback=defaults.retry), 0),
        request_interval=max(
            cp.getfloat(
                "translation", "request_interval", fallback=defaults.request_interval
            ),
            0.0,
        ),
        disable_proxy=cp.getboolean(
            "translation", "disable_proxy", fallback=defaults.disable_proxy
        ),
        copy_back=(
            cp.get("translation", "copy_back", fallback=defaults.copy_back).strip()
            or defaults.copy_back
        ),
        checkpoint_enabled=cp.getboolean(
            "translation", "checkpoint_enabled", fallback=defaults.checkpoint_enabled
        ),
        strict_ids=cp.getboolean(
            "translation", "strict_ids", fallback=defaults.strict_ids
        ),
        progress_ttl_sec=max(
            cp.getint(
                "translation", "progress_ttl_sec", fallback=defaults.progress_ttl_sec
            ),
            0,
        ),
    )


def validate_startup_security(settings: Settings) -> None:
    """启动期安全校验（fail-closed）：不满足则抛 SecurityError，由 bootstrap 拒绝启动。

    规则（SPEC H）：
    - require_token=true（默认）且 token 为空 → `api_token_missing`；
    - allow_placeholder_token=false（默认）且 token 是占位值 → `api_token_placeholder`；
    - token 为空且 require_token=false → 不报错（调用方显式承担风险，bootstrap 记录 WARN）。
    """

    # 延迟导入：core.security 仅依赖标准库，但保持 config 模块导入期无外部耦合。
    from whisper_stt_service.core.security import validate_startup_token

    validate_startup_token(
        settings.security.api_token,
        require_token=settings.security.require_token,
        allow_placeholder_token=settings.security.allow_placeholder_token,
    )