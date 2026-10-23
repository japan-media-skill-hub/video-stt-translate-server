"""Prometheus 指标导出路由（SPEC G）。

`/metrics` 由 `create_app(metrics=...)` 传入的 `MetricsState` 驱动：
- `metrics_enabled=false` 时 app 不注册本路由，请求得到 404（而非 403/500）。
- 本路由不注册鉴权豁免，与其他接口一样受全局 token 保护（SPEC H 全局安全）。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from whisper_stt_service.service.metrics import PROMETHEUS_AVAILABLE, MetricsState


LOGGER = logging.getLogger(__name__)

router = APIRouter()

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# 以这些指标名存在与否判断 prometheus_client 是否真正可用：
# 服务自身指标必需；进程级默认指标（python_gc_*）用于判断默认 collector 是否注册。
REQUIRED_METRIC_PREFIXES: tuple[str, ...] = (
    "whisper_stt_tasks_claimed_total",
    "whisper_stt_tasks_finished_total",
    "whisper_stt_task_duration_seconds",
    "whisper_stt_queue_depth",
    "whisper_stt_translate_batches_total",
    "whisper_stt_translate_retries_total",
    "whisper_stt_model_cache_hits_total",
    "whisper_stt_model_cache_misses_total",
)
PROCESS_METRIC_NAME = "python_gc_objects_collected_total"


def ensure_prometheus_available() -> bool:
    """确认 prometheus_client 可用，并在需要时注册进程级默认指标。

    部分环境（多进程 fork 后）默认 collector 未注册；这里幂等补注册，
    保证 /metrics 同时暴露进程 CPU/内存/文件描述符等排障必需指标。
    """

    if not PROMETHEUS_AVAILABLE:
        return False
    try:
        from prometheus_client import REGISTRY, ProcessCollector

        if PROCESS_METRIC_NAME not in REGISTRY._names_to_collectors:  # noqa: SLF001
            REGISTRY.register(ProcessCollector())
    except Exception as exc:  # noqa: BLE001 - 进程指标缺失不影响业务指标
        LOGGER.warning("process collector registration skipped: error=%s", exc)
    return True


def missing_metrics(body: str) -> list[str]:
    """返回缺失的必需指标名（用于返回值契约自检与测试断言）。"""

    return [name for name in REQUIRED_METRIC_PREFIXES if name not in body]


@router.get("/metrics", response_class=PlainTextResponse)
def metrics_endpoint(request: Request):
    """返回 Prometheus 文本格式指标；未安装 prometheus_client 时返回 503。"""

    state: MetricsState | None = getattr(request.app.state, "metrics", None)
    if state is None or not ensure_prometheus_available():
        # 配置开启但依赖缺失：明确 503，避免探针把空 200 误判为健康。
        LOGGER.warning("metrics requested but prometheus_client is unavailable")
        return PlainTextResponse(
            "# prometheus_client is not installed; install the delivery dependency\n",
            status_code=503,
            media_type=CONTENT_TYPE,
        )
    return PlainTextResponse(state.render().decode("utf-8"), media_type=CONTENT_TYPE)