"""兼容旧导入路径的 DAG 导出模块。"""

from whisper_stt_service.core.dag import (
    ASR_STAGES,
    REQUIRED_ASR_UPSTREAM_STAGE,
    SUPPORTED_DAG_VERSION,
    build_default_dag,
    normalize_and_validate_dag,
    normalize_and_validate_job_config,
)

__all__ = [
    "ASR_STAGES",
    "REQUIRED_ASR_UPSTREAM_STAGE",
    "SUPPORTED_DAG_VERSION",
    "build_default_dag",
    "normalize_and_validate_dag",
    "normalize_and_validate_job_config",
]
