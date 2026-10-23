"""Prometheus 指标注册与采集（SPEC G）。

设计要点：
- 指标在模块级只注册一次，并挂在**独立** `CollectorRegistry` 上，避免
  多次 `create_app()`（单元测试会反复创建）触发 prometheus_client 的
  "Duplicated timeseries" 异常。
- 需要"采集时读 DB"的 `queue_depth` 用自定义 collector 在 `/metrics` 请求时
  采样；读库失败只记录错误并输出 0，绝不让 /metrics 整体 500。
- `prometheus_client` 属于可选交付依赖：导入失败时 `PROMETHEUS_AVAILABLE=False`，
  运行时通过 `_noop_*()` 降级为无副作用对象，其余观测能力不受影响。

指标清单（SPEC G）：
`whisper_stt_tasks_claimed_total{stage}`、`whisper_stt_tasks_finished_total{stage,result}`、
`whisper_stt_task_duration_seconds{stage}`、`whisper_stt_queue_depth{stage,status}`、
`whisper_stt_translate_batches_total{result}`、`whisper_stt_translate_retries_total`、
`whisper_stt_model_cache_hits_total` / `whisper_stt_model_cache_misses_total`。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Callable, Iterable

try:  # pragma: no cover - 依赖是否存在由交付批次决定
    from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
    from prometheus_client.core import GaugeMetricFamily

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - 交付代理会补依赖
    CollectorRegistry = None  # type: ignore[assignment]
    Counter = None  # type: ignore[assignment]
    Histogram = None  # type: ignore[assignment]
    GaugeMetricFamily = None  # type: ignore[assignment]

    PROMETHEUS_AVAILABLE = False


LOGGER = logging.getLogger(__name__)

# 服务自身指标统一使用该前缀；`generate_latest` 仍会附带进程级默认指标。
METRIC_PREFIX = "whisper_stt"

# 任务终态结果标签，与 tasks.status 中的终态一致。
FINISHED_RESULTS: tuple[str, ...] = ("succeeded", "failed", "blocked", "cancelled")

# /queue/summary 的 throughput 窗口；
# 与 SPEC G "最近 5 分钟完成数/阶段" 对齐。
THROUGHPUT_WINDOW_SEC = 300


def metric_name(suffix: str) -> str:
    """拼接带服务前缀的指标名（避免各调用点手写前缀出错）。"""

    return f"{METRIC_PREFIX}_{suffix}"


class _NoopMetric:
    """prometheus_client 缺失时的空指标对象：吞掉所有调用。"""

    def labels(self, *args: Any, **kwargs: Any) -> "_NoopMetric":
        """兼容 Counter/Histogram 的 labels 链式调用。"""

        return self

    def inc(self, amount: float = 1.0) -> None:
        """空实现。"""

    def observe(self, amount: float) -> None:
        """空实现。"""

    def set(self, value: float) -> None:
        """空实现。"""


NOOP = _NoopMetric()


def _noop_metric() -> _NoopMetric:
    """返回空指标对象（每次返回同一个实例，避免反复分配）。"""

    return NOOP


def _normalize_stage(stage: str | None) -> str:
    """把阶段名归一化为可作标签的字符串，空值记为 unknown。"""

    text = str(stage or "").strip()
    return text or "unknown"


class QueueDepthCollector:
    """采集时读取 DB 的队列深度 gauge。

    `repo_factory` 返回带 `queue_summary()` 的仓储对象；每次 `collect` 都重新
    读取，保证指标反映实时状态。异常只记 error 日志并输出 0，不影响其它指标。
    """

    def __init__(
        self,
        repo_factory: Callable[[], Any],
        *,
        stages: Iterable[str] = (),
        statuses: Iterable[str] = (),
    ) -> None:
        """记录仓储工厂与期望的 stage/status 组合。"""

        self._repo_factory = repo_factory
        self._stages = tuple(stages)
        self._statuses = tuple(statuses)

    def collect(self):
        """产出 `whisper_stt_queue_depth{stage,status}` 的当前值。"""

        family = GaugeMetricFamily(
            metric_name("queue_depth"),
            "Queued/claimed/failed task counts per stage, sampled from SQLite",
            labels=["stage", "status"],
        )
        try:
            repo = self._repo_factory()
            summary = repo.queue_summary() if repo is not None else {}
        except Exception as exc:  # noqa: BLE001 - 采集失败不能让 /metrics 500
            LOGGER.error("queue_depth collector failed: error=%s", exc)
            summary = {}
        stages_map = dict(summary.get("stages") or {})
        # 配置里声明的阶段即使当前为 0 也输出，便于告警规则稳定匹配。
        for stage in self._stages:
            stages_map.setdefault(stage, {})
        for stage, statuses in sorted(stages_map.items()):
            status_map = dict(statuses or {})
            for status in self._statuses:
                status_map.setdefault(status, 0)
            for status, count in sorted(status_map.items()):
                try:
                    value = float(count)
                except (TypeError, ValueError):
                    value = 0.0
                family.add_metric([_normalize_stage(stage), str(status)], value)
        yield family


class MetricsState:
    """一次指标注册产生的全部指标句柄。

    `registry` 可直接交给 `generate_latest`；`collector` 持有 SQLite 采样器，
    便于测试直接调用 `collect()` 验证读数。
    """

    def __init__(
        self,
        *,
        registry: Any,
        tasks_claimed_total: Any,
        tasks_finished_total: Any,
        task_duration_seconds: Any,
        translate_batches_total: Any,
        translate_retries_total: Any,
        model_cache_hits_total: Any,
        model_cache_misses_total: Any,
        queue_depth_collector: QueueDepthCollector | None,
    ) -> None:
        """保存指标句柄；仅由 `build_metrics()` 调用。"""

        self.registry = registry
        self.tasks_claimed_total = tasks_claimed_total
        self.tasks_finished_total = tasks_finished_total
        self.task_duration_seconds = task_duration_seconds
        self.translate_batches_total = translate_batches_total
        self.translate_retries_total = translate_retries_total
        self.model_cache_hits_total = model_cache_hits_total
        self.model_cache_misses_total = model_cache_misses_total
        self.queue_depth_collector = queue_depth_collector

    def render(self) -> bytes:
        """渲染 Prometheus 文本格式（`/metrics` 的响应体）。"""

        if not PROMETHEUS_AVAILABLE:
            return b""
        return generate_latest(self.registry)


def build_metrics(
    *,
    repo_factory: Callable[[], Any] | None = None,
    stages: Iterable[str] = (),
    statuses: Iterable[str] = ("queued", "claimed", "succeeded", "failed"),
) -> MetricsState:
    """在全新 registry 上注册一套指标，返回句柄。

    每次调用都创建独立的 `CollectorRegistry`，因此反复 `create_app()` 不会重复
    注册；旧 app 持有的 registry 也不会受影响。
    """

    if not PROMETHEUS_AVAILABLE:
        LOGGER.warning(
            "prometheus_client not installed; /metrics will be unavailable "
            "(install the delivery dependency to enable it)"
        )
        return MetricsState(
            registry=None,
            tasks_claimed_total=NOOP,
            tasks_finished_total=NOOP,
            task_duration_seconds=NOOP,
            translate_batches_total=NOOP,
            translate_retries_total=NOOP,
            model_cache_hits_total=NOOP,
            model_cache_misses_total=NOOP,
            queue_depth_collector=None,
        )

    registry = CollectorRegistry(auto_describe=False)
    tasks_claimed_total = Counter(
        metric_name("tasks_claimed_total"),
        "Tasks claimed by workers",
        ["stage"],
        registry=registry,
    )
    tasks_finished_total = Counter(
        metric_name("tasks_finished_total"),
        "Tasks finished per terminal result",
        ["stage", "result"],
        registry=registry,
    )
    task_duration_seconds = Histogram(
        metric_name("task_duration_seconds"),
        "Task execution wall-clock seconds per stage",
        ["stage"],
        # 1s ~ 4h：覆盖 extract/stt/translate 的实际量级差异。
        buckets=(1, 5, 15, 60, 300, 900, 1800, 3600, 7200, 14400),
        registry=registry,
    )
    translate_batches_total = Counter(
        metric_name("translate_batches_total"),
        "Translate batches finished per result",
        ["result"],
        registry=registry,
    )
    translate_retries_total = Counter(
        metric_name("translate_retries_total"),
        "Translate batch retry attempts",
        registry=registry,
    )
    model_cache_hits_total = Counter(
        metric_name("model_cache_hits_total"),
        "Model cache hits",
        registry=registry,
    )
    model_cache_misses_total = Counter(
        metric_name("model_cache_misses_total"),
        "Model cache misses (cold loads)",
        registry=registry,
    )

    collector: QueueDepthCollector | None = None
    if repo_factory is not None:
        collector = QueueDepthCollector(
            repo_factory, stages=stages, statuses=statuses
        )
        registry.register(collector)
    return MetricsState(
        registry=registry,
        tasks_claimed_total=tasks_claimed_total,
        tasks_finished_total=tasks_finished_total,
        task_duration_seconds=task_duration_seconds,
        translate_batches_total=translate_batches_total,
        translate_retries_total=translate_retries_total,
        model_cache_hits_total=model_cache_hits_total,
        model_cache_misses_total=model_cache_misses_total,
        queue_depth_collector=collector,
    )


def record_task_claimed(metrics: MetricsState | None, stage: str) -> None:
    """记录一次任务领取（worker 调 `claim_next` 成功后调用）。"""

    if metrics is None:
        return
    metrics.tasks_claimed_total.labels(stage=_normalize_stage(stage)).inc()


def record_task_finished(
    metrics: MetricsState | None,
    *,
    stage: str,
    result: str,
    duration_sec: float | None = None,
) -> None:
    """记录任务终态；`duration_sec` 非空时同时写耗时直方图。"""

    if metrics is None:
        return
    normalized_stage = _normalize_stage(stage)
    normalized_result = str(result or "unknown").strip() or "unknown"
    metrics.tasks_finished_total.labels(
        stage=normalized_stage, result=normalized_result
    ).inc()
    if duration_sec is not None:
        try:
            value = max(float(duration_sec), 0.0)
        except (TypeError, ValueError):
            value = 0.0
        metrics.task_duration_seconds.labels(stage=normalized_stage).observe(value)


def record_translate_batch(metrics: MetricsState | None, result: str) -> None:
    """记录翻译批次结果（`succeeded` / `failed` / `partial`）。"""

    if metrics is None:
        return
    normalized = str(result or "unknown").strip() or "unknown"
    metrics.translate_batches_total.labels(result=normalized).inc()


def record_translate_retry(metrics: MetricsState | None, count: int = 1) -> None:
    """记录翻译批次重试次数。"""

    if metrics is None:
        return
    metrics.translate_retries_total.inc(max(int(count), 0))


def record_model_cache(metrics: MetricsState | None, *, hit: bool, count: int = 1) -> None:
    """记录模型缓存命中/未命中（批次 3 的 model_cache 调用）。"""

    if metrics is None:
        return
    target = metrics.model_cache_hits_total if hit else metrics.model_cache_misses_total
    target.inc(max(int(count), 0))


def _parse_timestamp(raw: Any) -> datetime | None:
    """把 DB 里的 ISO 时间戳解析为 aware datetime，失败返回 None。"""

    text = str(raw or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def compute_throughput(
    rows: Iterable[Any],
    *,
    stages: Iterable[str],
    now: datetime | None = None,
    window_sec: int = THROUGHPUT_WINDOW_SEC,
) -> dict[str, dict[str, Any]]:
    """按阶段统计最近 `window_sec` 内完成（succeeded/failed）的任务数。

    `rows` 为若干行（支持 dict 或带 `__getitem__` 的 sqlite3.Row），必须
    包含 `stage` 与 `finished_at` 字段。返回结构：

    ```python
    {"extract": {"succeeded": 1, "failed": 0, "total": 1, "window_sec": 300, "per_minute": 0.2}, ...}
    ```
    """

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    window_start = current - timedelta(seconds=max(int(window_sec), 1))
    result: dict[str, dict[str, Any]] = {
        _normalize_stage(stage): {
            "succeeded": 0,
            "failed": 0,
            "total": 0,
            "window_sec": int(window_sec),
            "per_minute": 0.0,
        }
        for stage in stages
    }
    for row in rows:
        try:
            stage = _normalize_stage(row["stage"])
            finished_at = _parse_timestamp(row["finished_at"])
        except (KeyError, IndexError, TypeError):
            continue
        # 容差 1s：调用方 now 与写入 finished_at 的时间天然有微小先后差，
        # 不能用严格 "finished_at > current" 判定，否则会丢掉刚完成的任务。
        if finished_at is None or finished_at < window_start:
            continue
        if finished_at > current + timedelta(seconds=1):
            continue
        bucket = result.setdefault(
            stage,
            {
                "succeeded": 0,
                "failed": 0,
                "total": 0,
                "window_sec": int(window_sec),
                "per_minute": 0.0,
            },
        )
        bucket["total"] += 1
        # 只区分"成功/其他"两类，blocked/cancelled 归入 failed，保持口径简单。
        if _read_row_status(row) == "succeeded":
            bucket["succeeded"] += 1
        else:
            bucket["failed"] += 1
    minutes = max(int(window_sec), 1) / 60.0
    for bucket in result.values():
        bucket["per_minute"] = round(bucket["total"] / minutes, 4)
    return result


def _read_row_status(row: Any) -> str:
    """读取行的 status 字段，缺失时返回空串。"""

    try:
        return str(row["status"] or "")
    except (KeyError, IndexError, TypeError):
        return ""
