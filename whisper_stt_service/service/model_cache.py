"""进程内 ASR 模型缓存：LRU 限额、活跃引用保护与同卡并发信号量。

本模块是 SPEC 第二节 D 的落地实现，解决“每个 task 重新加载 GB 级模型导致显存随
worker 数线性叠加”的问题。设计要点：

- **key 覆盖全部影响模型实例的参数**：模型路径/名称、解析后的设备、compute type，
  faster-whisper 侧还包括 VAD 参数、语言、batched pipeline 开关；WhisperX 侧还包括
  VAD config、VAD backend/onset/offset 与 align 参数。任何一个参数不同都会得到不同
  实例，避免“参数串味”。
- **引用计数 + LRU 不逐出活跃引用**：`acquire_*` 返回的 lease 在 `release()` 之前
  计入活跃引用；超限逐出时只淘汰活跃引用为 0 的条目。所有引用都被占满时不逐出，
  记录一次 `over_capacity` 告警后临时超限（宁可多占一份显存，也不打断在途转写）。
- **model_cache_size=0 禁用缓存**：每次 acquire 都重新加载，release 后立即释放。
- **model_parallelism**：每个缓存条目自带 `BoundedSemaphore`，同一模型实例的并发
  转写数受控；取不到信号量时排队等待，不失败（超时后仍然放行并记录告警）。
- **可观测**：冷启动/命中写 `model_cache_miss/hit key=... elapsed_ms=...` 日志，并维护
  计数器供 `/metrics` 与 `/admin/model-cache` 读取。

所有加载器都是可注入的，单元测试用 fake loader 即可覆盖 LRU/信号量语义，不依赖
torch / faster-whisper / whisperx 是否已安装。
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import importlib
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Hashable, Iterator
import weakref

# metrics 参数“省略 vs 显式 None”的区分哨兵（_UNSET 表示调用方未提供该参数）。
_UNSET: Any = object()

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_CACHE_SIZE = 2
DEFAULT_MODEL_PARALLELISM = 1
# 同卡排队告警阈值：等待模型信号量超过该值只记一次 WARNING（不改变排队语义）。
MODEL_SIGNAL_WARN_WAIT_MS = 5000.0

# 兼容保留：旧配置沿用 acquire_timeout_sec 字段，新语义下它不再是“放弃限流”的阈值。
DEFAULT_MODEL_PREREQ_WAIT_SEC = 30.0

# 已知的 VAD 参数名（STT 阶段），用于缓存 key 归一化。
STT_VAD_PARAM_KEYS: tuple[str, ...] = (
    "vad_filter",
    "vad_threshold",
    "vad_min_speech_duration_ms",
    "vad_max_speech_duration_s",
    "vad_min_silence_duration_ms",
    "vad_speech_pad_ms",
)


def resolve_runtime_hint(device: str, compute_type: str) -> tuple[str, str]:
    """按 "auto" 语义解析运行设备与 compute type（与执行器 `_resolve_runtime` 等价）。

    本模块刻意不 import `executor.common`：导入链 `executor → service.model_cache` 会先
    触发 service 包初始化，反向导入 executor 会形成循环导入。设备探测实现保持在本模块
    内部，保证缓存 key 与执行器实际使用的 device/compute 完全一致。
    """

    device_value = str(device)
    if device_value == "auto":
        try:
            import torch  # type: ignore

            device_value = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device_value = "cpu"

    compute_value = str(compute_type)
    if compute_value == "auto":
        compute_value = "float16" if device_value == "cuda" else "int8"
    return device_value, compute_value


def make_cache_key(kind: str, **parts: Any) -> str:
    """把模型缓存维度拼装成稳定、可读的字符串 key（按传入顺序）。"""

    tokens = [f"kind={kind}"]
    for name, value in parts.items():
        tokens.append(f"{name}={_normalize_key_value(value)}")
    return "|".join(tokens)


def _normalize_key_value(value: Any) -> str:
    """归一化 key 片段：路径类统一转字符串，浮点保留可读精度，布尔转小写。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (int, str)):
        return str(value)
    if value is None:
        return "none"
    return str(value)


def normalize_vad_params(values: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """把 VAD 参数字典归一化为有序元组，保证同样参数得到同样 key。"""

    return tuple(
        (name, _normalize_key_value(values.get(name)))
        for name in STT_VAD_PARAM_KEYS
    )


def _cancelled_error(message: str) -> BaseException:
    """构造协作取消异常（惰性导入 `executor.common`，避免导入期循环依赖）。

    导入链 `executor → service.model_cache` 会先初始化本模块；若在模块顶层直接
    import `executor.common`，而 executor/__init__ 又未初始化完成，就会形成循环导入。
    因此这里在真正需要抛异常时才导入，保证包初始化顺序与运行期都安全。
    """

    from whisper_stt_service.executor.common import ExecutorCancelledError

    return ExecutorCancelledError(message)


def _hashable(value: Any) -> Hashable:
    """把任意 key 片段转成可哈希形式（dict/list 递归展开）。"""

    if isinstance(value, dict):
        return tuple(sorted((str(k), _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class _CacheEntry:
    """缓存条目：模型实例 + 活跃引用计数 + 加载耗时 + 并发信号量。"""

    key: Hashable
    display_key: str
    kind: str
    resources: tuple[Any, ...]
    created_at: float
    load_elapsed_ms: float
    signal: threading.BoundedSemaphore
    refs: int = 0
    inflight: int = 0
    hits: int = 0

    def describe(self) -> dict[str, Any]:
        """返回条目的只读快照，供状态查询与日志使用（不泄漏模型对象的 repr）。"""

        payload = {
            "key": self.display_key,
            "kind": self.kind,
            "refs": self.refs,
            "inflight": self.inflight,
            "hits": self.hits,
            "load_elapsed_ms": round(self.load_elapsed_ms, 3),
            "created_at": self.created_at,
            "resources": [type(item).__name__ for item in self.resources],
        }
        return payload


class ModelLease:
    """模型使用租约：上下文管理器，保证 release 与信号量归还只发生一次。

    用法::

        with model_cache.get_whisper_model(path, "cuda", "float16") as lease:
            model, transcribe_slot = lease.value, lease.transcribe_slot
            with transcribe_slot:
                model.transcribe(...)
    """

    def __init__(self, entry: _CacheEntry, cache: "ModelCache", *, cached: bool) -> None:
        """绑定缓存条目与所属缓存；`cached=False` 表示本次为禁缓存模式下的临时实例。"""

        self._entry = entry
        self._cache = cache
        self._cached = cached
        self._released = False
        self._lock = threading.Lock()
        # 并发转写许可的持有标记：保证 release 只归还“自己拿到的那一份”，且幂等。
        self._slot_held = False
        self._slot_lock = threading.Lock()

    @property
    def value(self) -> Any:
        """返回主资源；多资源场景（WhisperX bundle）返回完整元组。"""

        if len(self._entry.resources) == 1:
            return self._entry.resources[0]
        return self._entry.resources

    @property
    def resources(self) -> tuple[Any, ...]:
        """返回全部资源元组。"""

        return self._entry.resources

    @property
    def transcribe_slot(self) -> threading.BoundedSemaphore:
        """返回该模型实例的并发转写信号量（model_parallelism）。"""

        return self._entry.signal

    @contextmanager
    def transcribe_guard(self) -> Iterator[None]:
        """上下文管理器：进入时取并发许可，退出时归还（推荐写法）。"""

        self.acquire_transcribe_slot()
        try:
            yield
        finally:
            self.release_transcribe_slot()

    @property
    def key(self) -> str:
        """返回可读缓存 key。"""

        return self._entry.display_key

    @property
    def cached(self) -> bool:
        """本次是否复用了进程内缓存中的实例。"""

        return self._cached

    @property
    def load_elapsed_ms(self) -> float:
        """该条目冷启动耗时（毫秒），命中时为原始加载耗时。"""

        return self._entry.load_elapsed_ms

    @property
    def refs(self) -> int:
        """当前活跃引用计数（含本租约）。"""

        return self._entry.refs

    def acquire_transcribe_slot(
        self,
        *,
        timeout_sec: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
        poll_sec: float = 0.05,
    ) -> float:
        """获取该模型实例的并发转写许可，返回本次等待耗时（毫秒）。

        SPEC D 语义：`model_parallelism=1` 时取不到许可必须**排队等待**，绝不因等待
        过久而“放弃限流继续跑”（那会让同卡并发翻倍、显存 OOM），也不把任务判失败。

        - 不设总超时：取不到许可就一直排队（SPEC D 要求，绝不超时放行、也不判任务失败）。
        - `timeout_sec`：**仅为兼容旧调用保留**，不再是“放弃限流”的阈值，调用方不必传。
        - `cancel_check`：每 `poll_sec` 轮询一次（默认 50ms），返回 True 时抛
          `ExecutorCancelledError`，让 drain/失租的 worker 不会永远卡在排队里。
        - `poll_sec`：轮询步长（默认 50ms），避免长时间阻塞在单次 acquire 上。

        只能用 `release_transcribe_slot()` 或 `transcribe_guard()` 归还；超出等待阈值
        （`MODEL_SIGNAL_WARN_WAIT_MS`）只记一次告警，便于观察同卡排队。
        """

        started = time.perf_counter()
        poll = max(min(float(poll_sec), 1.0), 0.001)
        first_wait_logged = False
        while True:
            # 等待期间仍计入活跃引用，避免排队中的租约被 LRU 逐出。
            entry = self._entry
            with self._cache._lock:
                entry.inflight += 1
            try:
                acquired = entry.signal.acquire(timeout=poll)
            finally:
                with self._cache._lock:
                    entry.inflight -= 1
            if acquired:
                with self._cache._lock:
                    entry.inflight += 1
                with self._slot_lock:
                    self._slot_held = True
                break
            waited_ms = (time.perf_counter() - started) * 1000.0
            if not first_wait_logged and waited_ms >= MODEL_SIGNAL_WARN_WAIT_MS:
                first_wait_logged = True
                with self._cache._lock:
                    inflight = entry.inflight
                LOGGER.warning(
                    "model_cache_signal_wait key=%s waited_ms=%.1f inflight=%s "
                    "(model_parallelism=%s，排队等待中)",
                    entry.display_key,
                    waited_ms,
                    inflight,
                    self._cache._parallelism,
                )
            if cancel_check is not None:
                try:
                    cancelled = bool(cancel_check())
                except Exception as exc:  # noqa: BLE001
                    raise _cancelled_error(f"cancel_check_failed: {exc}") from exc
                if cancelled:
                    raise _cancelled_error("cancelled_while_waiting_model_slot")

        waited_ms = (time.perf_counter() - started) * 1000.0
        self._cache._record_signal_wait(waited_ms)
        return waited_ms

    def release_transcribe_slot(self) -> None:
        """归还并发转写许可；**只归还本租约确实获取到的许可**，且幂等。

        P1 回归点：早期实现无条件 `signal.release()`，导致“没拿到许可的排队者”
        归还许可后等于凭空增加容量（`semaphore._value` 上升），同卡并发失控。
        现在用实例级 `_slot_held` 记录持有状态：未持有则空操作，重复释放只告警一次。
        """

        with self._slot_lock:
            if not self._slot_held:
                LOGGER.warning(
                    "model_cache_slot_release_without_acquire key=%s（忽略）",
                    self._entry.display_key,
                )
                return
            self._slot_held = False
        try:
            self._entry.signal.release()
        except ValueError as exc:
            # 只有“超出 BoundedSemaphore 初始值”才会抛，说明计数已损坏。
            LOGGER.error(
                "model_cache_slot_over_release key=%s error=%s",
                self._entry.display_key,
                exc,
            )
        with self._cache._lock:
            if self._entry.inflight > 0:
                self._entry.inflight -= 1

    def release(self) -> None:
        """归还引用；可重复调用，只有第一次生效。"""

        with self._lock:
            if self._released:
                return
            self._released = True
        self._cache._release(self._entry, cached=self._cached)

    def __enter__(self) -> "ModelLease":
        """进入上下文，直接返回自身。"""

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """退出上下文时释放引用。"""

        self.release()


class _SingleFlight:
    """同 key 并发冷启动去重：只有一个线程真正加载，其余等待后复用结果。"""

    def __init__(self) -> None:
        """初始化内部互斥条件变量。"""

        self._cond = threading.Condition()
        self._loading: dict[Hashable, threading.Event] = {}
        self._failures: dict[Hashable, BaseException] = {}

    @contextmanager
    def guard(self, key: Hashable) -> Iterator[bool]:
        """返回 (是否由本线程负责加载)；负责加载的线程在退出时负责广播结果。"""

        with self._cond:
            while key in self._loading:
                event = self._loading[key]
                self._cond.release()
                try:
                    event.wait()
                finally:
                    self._cond.acquire()
            failure = self._failures.pop(key, None)
            if failure is not None:
                raise failure
            event = threading.Event()
            self._loading[key] = event
        owner = True
        try:
            yield owner
        except BaseException as exc:
            with self._cond:
                self._failures[key] = exc
                self._loading.pop(key, None)
                event.set()
            raise
        else:
            with self._cond:
                self._loading.pop(key, None)
                event.set()


class ModelCache:
    """线程安全的进程内模型缓存（LRU + 引用计数 + 每实例并发信号量）。"""

    def __init__(
        self,
        *,
        cache_size: int = DEFAULT_MODEL_CACHE_SIZE,
        parallelism: int = DEFAULT_MODEL_PARALLELISM,
        acquire_timeout_sec: float = DEFAULT_MODEL_PREREQ_WAIT_SEC,
        metrics: Any = _UNSET,
        whisper_model_loader: Callable[..., Any] | None = None,
        batched_pipeline_loader: Callable[..., Any] | None = None,
        whisperx_load_model: Callable[..., Any] | None = None,
        whisperx_vad_factory: Callable[..., Any] | None = None,
        whisperx_align_loader: Callable[..., Any] | None = None,
    ) -> None:
        """构造缓存；loader 全部可注入，默认按需惰性导入真实依赖。"""

        self._lock = threading.RLock()
        self._entries: "OrderedDict[Hashable, _CacheEntry]" = OrderedDict()
        self._singleflight = _SingleFlight()
        self._elapsed: dict[str, float] = {}
        # 模型实例 → (model, batched pipeline)，与模型同生命周期（弱引用语义按 id 校验）。
        self._pipelines: dict[int, tuple[Any, Any]] = {}
        self._cache_size = 0
        self._parallelism = 1
        self._acquire_timeout_sec = max(float(acquire_timeout_sec), 0.0)
        self._counters: dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "over_capacity": 0,
            "uncached_loads": 0,
            "signal_wait_ms_total": 0,
        }
        # SPEC G：可选注入 metrics（弱引用持有），命中/未命中时计数。
        self._metrics_ref: "weakref.ref[Any] | None" = None
        self._whisper_model_loader = whisper_model_loader
        self._batched_pipeline_loader = batched_pipeline_loader
        self._whisperx_load_model = whisperx_load_model
        self._whisperx_vad_factory = whisperx_vad_factory
        self._whisperx_align_loader = whisperx_align_loader
        self.configure(
            cache_size=cache_size,
            parallelism=parallelism,
            acquire_timeout_sec=acquire_timeout_sec,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # 配置与清理 API（供 bootstrap / runtime / admin 路由接入）
    # ------------------------------------------------------------------
    def configure(
        self,
        *,
        cache_size: int | None = None,
        parallelism: int | None = None,
        acquire_timeout_sec: float | None = None,
        metrics: Any = _UNSET,
    ) -> dict[str, Any]:
        """更新缓存容量/并发上限/metrics 绑定；只影响后续 acquire。

        `metrics` 语义（sentinel 区分“省略”与“显式 None”）：
        - 省略（默认 `_UNSET`）：保持当前绑定不变；
        - 传入 MetricsState 实例：绑定，命中/未命中调用 `record_model_cache(...)`；
        - 显式传 `None`：解绑（停止上报）。
        """

        if metrics is not _UNSET:
            self.set_metrics(metrics)
        with self._lock:
            if cache_size is not None:
                self._cache_size = max(int(cache_size), 0)
            if parallelism is not None:
                self._parallelism = max(int(parallelism), 1)
            if acquire_timeout_sec is not None:
                self._acquire_timeout_sec = max(float(acquire_timeout_sec), 0.0)
            if self._cache_size > 0:
                self._evict_until_within_limit_locked()
            return self._config_snapshot_locked()

    def set_metrics(self, metrics: Any | None) -> None:
        """绑定/解绑 metrics 实例（弱引用），供 bootstrap 或 /metrics 测试注入。"""

        if metrics is None:
            self._metrics_ref = None
            return
        try:
            self._metrics_ref = weakref.ref(metrics)
        except TypeError:
            # 不可弱引用的对象（如 SimpleNamespace 的绑定方法容器）退化为强引用。
            self._metrics_ref = lambda: metrics  # type: ignore[assignment]
        LOGGER.info(
            "model_cache metrics attached: type=%s", type(metrics).__name__
        )

    def _record_metrics(self, *, hit: bool) -> None:
        """把命中/未命中上报到 metrics；metrics 缺失或实现不兼容时静默降级。

        优先使用 `whisper_stt_service.service.metrics.record_model_cache(metrics, hit=...)`
        （真实 prometheus Registry 计数）；metrics 对象自带同名方法时也兼容使用。
        """

        ref = self._metrics_ref
        if ref is None:
            return
        metrics = ref()
        if metrics is None:
            return
        try:
            # 延迟导入，避免 service.metrics ↔ model_cache 的导入期耦合。
            from whisper_stt_service.service.metrics import (
                MetricsState,
                record_model_cache,
            )

            if isinstance(metrics, MetricsState):
                # 真实 MetricsState：走模块级 record_model_cache，计入 prometheus Registry。
                record_model_cache(metrics, hit=bool(hit))
                return
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("model_cache metrics record failed: error=%s", exc)
            return
        # 兼容实现了同名方法的 metrics 替身（测试或外部监控适配器）。
        recorder = getattr(metrics, "record_model_cache", None)
        if not callable(recorder):
            LOGGER.debug(
                "model_cache metrics ignored: unsupported type=%s",
                type(metrics).__name__,
            )
            return
        try:
            recorder(hit=bool(hit))
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("model_cache metrics method failed: error=%s", exc)

    def _config_snapshot_locked(self) -> dict[str, Any]:
        """返回配置快照（调用方需持有锁）。"""

        return {
            "cache_size": self._cache_size,
            "parallelism": self._parallelism,
            "acquire_timeout_sec": self._acquire_timeout_sec,
            "entries": len(self._entries),
        }

    @property
    def enabled(self) -> bool:
        """缓存是否启用（cache_size=0 表示禁用，仅用于调试）。"""

        with self._lock:
            return self._cache_size > 0

    def stats(self) -> dict[str, Any]:
        """返回命中率、条目数、活跃引用等只读统计，供 /admin/model-cache 与日志使用。"""

        with self._lock:
            hits = self._counters["hits"]
            misses = self._counters["misses"]
            total = hits + misses
            entries = [self._entries[k].describe() for k in self._entries]
            cached_elapsed = dict(self._elapsed)
            config = self._config_snapshot_locked()
        return {
            "config": asdict(
                _CacheConfigView(
                    cache_size=int(config["cache_size"]),
                    parallelism=int(config["parallelism"]),
                    acquire_timeout_sec=float(config["acquire_timeout_sec"]),
                    entries=int(config["entries"]),
                )
            ),
            "counters": {
                "hits": hits,
                "misses": misses,
                "hit_rate": (hits / total) if total else 0.0,
                "evictions": self._counters["evictions"],
                "over_capacity": self._counters["over_capacity"],
                "uncached_loads": self._counters["uncached_loads"],
                "signal_wait_ms_total": self._counters["signal_wait_ms_total"],
            },
            "entries": entries,
            "load_elapsed_ms": {
                key: round(value, 3) for key, value in cached_elapsed.items()
            },
        }

    def get_load_elapsed_ms(self, kind: str) -> float | None:
        """返回某个 stage/kind 最近一次冷启动耗时（毫秒），供 task 日志与快照使用。"""

        with self._lock:
            return self._elapsed.get(kind)

    def clear(self, *, force: bool = False) -> dict[str, Any]:
        """清空缓存并释放模型。

        `force=False`（默认）只清空没有活跃引用的条目，返回未释放的条目数；
        `force=True` 用于进程退出：丢弃全部条目，让在途 lease 的 release 变成空操作。
        """

        dropped = 0
        skipped = 0
        with self._lock:
            for key in list(self._entries.keys()):
                entry = self._entries[key]
                if not force and entry.refs > 0:
                    skipped += 1
                    continue
                self._detach_entry_locked(entry, reason="clear_forced" if force else "clear")
                dropped += 1
            self._elapsed.clear()
        LOGGER.info(
            "model_cache cleared: dropped=%s skipped_active=%s force=%s",
            dropped,
            skipped,
            force,
        )
        return {"dropped": dropped, "skipped_active": skipped, "forced": force}

    def cleanup(self, *, force: bool = False) -> dict[str, Any]:
        """缓存清理别名：供 shutdown/lifespan finally 与运维接口复用。"""

        return self.clear(force=force)

    # ------------------------------------------------------------------
    # acquire 系列 API
    # ------------------------------------------------------------------
    def get_whisper_model(
        self,
        model_path: str | Path,
        device: str,
        compute_type: str,
        *,
        vad_params: dict[str, Any] | None = None,
        language: str | None = None,
        use_batched_pipeline: bool = False,
    ) -> ModelLease:
        """获取即用即还的 faster-whisper 模型租约（SPEC 命名 `get_whisper_model`）。

        模型路径、解析后的设备、compute type、VAD 参数与语言共同决定 key；
        返回值 `lease.value` 为 `WhisperModel`，`lease.transcribe_slot` 为并发信号量。
        `use_batched_pipeline=True` 时在 key 中加入该维度，并把 pipeline 挂在条目上
        随模型同生命周期，通过 `get_batched_pipeline()` 取用。
        """

        resolved_device, resolved_compute = resolve_runtime_hint(device, compute_type)
        key = _hashable(
            (
                "whisper_rt",
                str(model_path),
                resolved_device,
                resolved_compute,
                bool(use_batched_pipeline),
                normalize_vad_params(vad_params or {}),
                language or "",
                self._parallelism,
            )
        )
        display = make_cache_key(
            "whisper_rt",
            model=str(model_path),
            device=resolved_device,
            compute=resolved_compute,
            batched=bool(use_batched_pipeline),
            vad=normalize_vad_params(vad_params or {}),
            language=language or "",
        )

        def _loader() -> tuple[Any, ...]:
            return self._load_whisper_runtime(
                device=resolved_device,
                compute=resolved_compute,
                model_path=str(model_path),
                use_batched_pipeline=bool(use_batched_pipeline),
            )

        return self._acquire(
            kind="stt",
            key=key,
            display_key=display,
            loader=_loader,
        )

    def get_batched_pipeline(self, model: Any) -> Any:
        """为已加载模型构造 batched pipeline（与模型同生命周期，由缓存条目持有）。

        SPEC D 命名约定：`get_batched_pipeline(model)`；同一模型实例重复调用返回
        同一 pipeline 对象，避免每次转写都重建。
        """

        with self._lock:
            existing = self._pipelines.get(id(model))
            if existing is not None and existing[0] is model:
                return existing[1]
        if self._batched_pipeline_loader is not None:
            pipeline = self._batched_pipeline_loader(model)
        else:
            from faster_whisper import BatchedInferencePipeline

            pipeline = BatchedInferencePipeline(model=model)
        with self._lock:
            self._pipelines[id(model)] = (model, pipeline)
        return pipeline

    def acquire_batched_pipeline(self, model: Any) -> Any:
        """兼容别名：等价于 `get_batched_pipeline`。"""

        return self.get_batched_pipeline(model)

    def get_whisperx_bundle(
        self,
        model: str | Path,
        *,
        device: str,
        compute_type: str,
        language: str,
        vad_config_path: str | Path,
        vad_backend: str,
        vad_onset: float,
        vad_offset: float,
        vad_model_fp: str | Path | None,
        align_model_path: str | Path | None,
        align_enabled: bool,
        asr_options: dict[str, Any] | None = None,
        vad_options: dict[str, Any] | None = None,
    ) -> ModelLease:
        """获取 WhisperX 三件套（ASR + VAD + align）租约（SPEC 命名）。

        `lease.resources` 依次为 `(asr, vad_model, align_model, align_meta)`；
        未启用 align 时后两项为 `None`。key 覆盖模型、设备、compute、语言、
        VAD config/backend/onset/offset、align 根目录与开关，保证参数不同不串用。
        """

        resolved_device, resolved_compute = resolve_runtime_hint(device, compute_type)
        key = _hashable(
            (
                "whisperx_bundle",
                str(model),
                resolved_device,
                resolved_compute,
                str(language),
                str(vad_config_path),
                str(vad_backend),
                float(vad_onset),
                float(vad_offset),
                str(vad_model_fp or ""),
                str(align_model_path or ""),
                bool(align_enabled),
                _hashable(asr_options or {}),
                _hashable(vad_options or {}),
                self._parallelism,
            )
        )
        display = make_cache_key(
            "whisperx_bundle",
            model=str(model),
            device=resolved_device,
            compute=resolved_compute,
            language=str(language),
            vad_config=str(vad_config_path),
            vad_backend=str(vad_backend),
            vad_onset=float(vad_onset),
            vad_offset=float(vad_offset),
            align_enabled=bool(align_enabled),
            align_model=str(align_model_path or ""),
        )

        def _loader() -> tuple[Any, ...]:
            return self._load_whisperx_bundle(
                model=str(model),
                device=resolved_device,
                compute=resolved_compute,
                language=str(language),
                vad_backend=str(vad_backend),
                vad_onset=float(vad_onset),
                vad_offset=float(vad_offset),
                vad_model_fp=str(vad_model_fp) if vad_model_fp else None,
                align_model_path=str(align_model_path) if align_model_path else None,
                align_enabled=bool(align_enabled),
                asr_options=dict(asr_options or {}),
                vad_options=dict(vad_options or {}),
            )

        return self._acquire(
            kind="stt_whisperx",
            key=key,
            display_key=display,
            loader=_loader,
        )

    def get_align_model(
        self,
        language: str,
        device: str,
        model_dir: str | Path | None = None,
    ) -> ModelLease:
        """按语言缓存 WhisperX align 模型，返回 `(align_model, align_meta)` 租约（SPEC 命名）。"""

        resolved_device, _ = resolve_runtime_hint(device, "auto")
        key = _hashable(("align", str(language), resolved_device, str(model_dir or "")))
        display = make_cache_key(
            "align", language=str(language), device=resolved_device, model_dir=str(model_dir or "")
        )

        def _loader() -> tuple[Any, ...]:
            return self._load_align(language=str(language), device=resolved_device, model_dir=model_dir)

        return self._acquire(kind="align", key=key, display_key=display, loader=_loader)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _acquire(
        self,
        *,
        kind: str,
        key: Hashable,
        display_key: str,
        loader: Callable[[], tuple[Any, ...]],
    ) -> ModelLease:
        """acquire 系列统一入口：命中返回缓存 lease，未命中走 single-flight 加载。"""

        hit = self._lookup(key)
        if hit is not None:
            return hit

        with self._singleflight.guard(key):
            # 等待期间可能已被其他线程加载完成，再查一次缓存避免重复加载。
            again = self._lookup(key)
            if again is not None:
                return again
            return self._load_and_register(
                kind=kind, key=key, display_key=display_key, loader=loader
            )

    def _lookup(self, key: Hashable, *, reserve: bool = True) -> ModelLease | None:
        """命中缓存条目时计入命中数并返回 lease；禁缓存或未命中返回 None。

        命中会刷新 LRU 顺序并递增引用计数，保证在途使用的条目不会被逐出。
        `reserve=False` 仅表示“探测是否存在”，不计命中数也不占用引用。
        """

        if not reserve:
            with self._lock:
                probe = self._entries.get(key)
            if probe is None:
                return None
            return ModelLease(probe, self, cached=True)
        with self._lock:
            if self._cache_size <= 0:
                return None
            entry = self._entries.get(key)
            if entry is None:
                return None
            entry.refs += 1
            entry.hits += 1
            self._counters["hits"] += 1
            self._entries.move_to_end(key)
        LOGGER.debug(
            "model_cache_hit key=%s elapsed_ms=%.1f refs=%s",
            entry.display_key,
            entry.load_elapsed_ms,
            entry.refs,
        )
        self._record_metrics(hit=True)
        return ModelLease(entry, self, cached=True)

    def _load_and_register(
        self,
        *,
        kind: str,
        key: Hashable,
        display_key: str,
        loader: Callable[[], tuple[Any, ...]],
    ) -> ModelLease:
        """执行加载（含计时与日志），按需入表，最后返回 lease。"""

        started = time.perf_counter()
        resources = tuple(loader())
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        with self._lock:
            cache_enabled = self._cache_size > 0
            entry = _CacheEntry(
                key=key,
                display_key=display_key,
                kind=kind,
                resources=resources,
                created_at=time.time(),
                load_elapsed_ms=elapsed_ms,
                signal=threading.BoundedSemaphore(self._parallelism),
                refs=1,
            )
            self._counters["misses"] += 1
            self._elapsed[kind] = elapsed_ms
            if not cache_enabled:
                self._counters["uncached_loads"] += 1
            else:
                self._entries[key] = entry
                self._entries.move_to_end(key)
                self._evict_until_within_limit_locked()
        LOGGER.info(
            "model_cache_miss key=%s kind=%s elapsed_ms=%.1f cached=%s resources=%s",
            display_key,
            kind,
            elapsed_ms,
            cache_enabled,
            ",".join(type(item).__name__ for item in resources),
        )
        # SPEC G：miss 计数与日志同步上报。
        self._record_metrics(hit=False)
        return ModelLease(entry, self, cached=False)

    def _release(self, entry: _CacheEntry, *, cached: bool) -> None:
        """归还引用；归还后按 LRU 把缓存收敛回 cache_size，并把在该窗口内被逐出的
        模型就地释放（P2：临时超限的条目在失活后必须真正回收）。

        - 禁缓存（cache_size=0）：归还后立即丢弃条目并释放模型；
        - 启用缓存：引用归零后仍保留在 LRU 表中（供后续命中），但若此时总条目数
          超过上限（此前因活跃引用而临时超限），则按 LRU 逐出空闲条目并释放。
          **绝不逐出仍有活跃引用的条目。**
        """

        evicted: list[_CacheEntry] = []
        with self._lock:
            if entry.refs > 0:
                entry.refs -= 1
            if self._cache_size <= 0:
                self._detach_entry_locked(entry, reason="uncached")
                return
            current = self._entries.get(entry.key)
            if current is entry:
                self._entries.move_to_end(entry.key)
            evicted = self._trim_to_limit_locked()
        # 释放模型资源放在锁外：close/empty_cache 可能耗时，避免阻塞其他 acquire。
        for victim in evicted:
            self._release_resources(victim)
            LOGGER.info(
                "model_cache_evict key=%s kind=%s reason=release_trim",
                victim.display_key,
                victim.kind,
            )

    def _trim_to_limit_locked(self) -> list[_CacheEntry]:
        """把缓存裁剪回 cache_size，返回被摘除的条目（调用方负责释放资源）。

        与 `_evict_until_within_limit_locked` 的区别：这里只摘表、不在这里释放，
        便于调用方在锁外做 close，避免长时间持锁。
        """

        removed: list[_CacheEntry] = []
        if self._cache_size <= 0:
            return removed
        while len(self._entries) > self._cache_size:
            victim_key: Hashable | None = None
            for candidate_key, candidate in self._entries.items():
                if candidate.refs <= 0:
                    victim_key = candidate_key
                    break
            if victim_key is None:
                self._counters["over_capacity"] += 1
                LOGGER.debug(
                    "model_cache_over_capacity entries=%s limit=%s (全部活跃，暂不逐出)",
                    len(self._entries),
                    self._cache_size,
                )
                break
            victim = self._entries.pop(victim_key)
            self._counters["evictions"] += 1
            removed.append(victim)
        return removed

    def _detach_entry_locked(self, entry: _CacheEntry, *, reason: str) -> None:
        """从表中移除条目并尽力释放模型资源（调用方需持有锁）。"""

        current = self._entries.get(entry.key)
        if current is entry:
            del self._entries[entry.key]
        self._release_resources(entry)
        # 条目被丢弃后，其 batched pipeline 一并失效，避免 id 复用命中陈旧对象。
        for model_id, (model, _pipeline) in list(self._pipelines.items()):
            if any(model is resource for resource in entry.resources):
                self._pipelines.pop(model_id, None)
        LOGGER.info(
            "model_cache_evict key=%s kind=%s reason=%s",
            entry.display_key,
            entry.kind,
            reason,
        )

    def _release_resources(self, entry: _CacheEntry) -> None:
        """尽力释放条目持有的模型（torch CUDA 缓存等），失败只记日志不抛出。"""

        for resource in entry.resources:
            if resource is None:
                continue
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning(
                        "model_cache close failed: key=%s error=%s dc=%s",
                        entry.display_key,
                        exc,
                        exc.__class__.__name__,
                    )
        if entry.kind in {"stt", "stt_whisperx"}:
            try:
                import torch  # type: ignore

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                # 未安装 torch 或 CUDA 不可用时无需释放显存缓存。
                pass

    def _record_signal_wait(self, waited_ms: float) -> None:
        """累计信号量等待时长（毫秒），供 stats/metrics 观察同卡排队情况。"""

        with self._lock:
            self._counters["signal_wait_ms_total"] += int(max(waited_ms, 0.0))

    def _evict_until_within_limit_locked(self) -> None:
        """按 LRU 顺序逐出，跳过仍有活跃引用的条目（调用方需持有锁）。"""

        if self._cache_size <= 0:
            return
        while len(self._entries) > self._cache_size:
            victim_key: Hashable | None = None
            for candidate_key, candidate in self._entries.items():
                if candidate.refs <= 0:
                    victim_key = candidate_key
                    break
            if victim_key is None:
                self._counters["over_capacity"] += 1
                LOGGER.warning(
                    "model_cache_over_capacity entries=%s limit=%s active=%s (不逐出活跃引用)",
                    len(self._entries),
                    self._cache_size,
                    sum(1 for item in self._entries.values() if item.refs > 0),
                )
                return
            victim = self._entries[victim_key]
            self._detach_entry_locked(victim, reason="lru")
            self._counters["evictions"] += 1

    # ------------------------------------------------------------------
    # 默认加载器（真实依赖惰性导入）
    # ------------------------------------------------------------------
    def _load_whisper_runtime(
        self,
        *,
        device: str,
        compute: str,
        model_path: str,
        use_batched_pipeline: bool,
    ) -> tuple[Any, ...]:
        """加载 faster-whisper 模型，并按需构造 batched pipeline。"""

        if self._whisper_model_loader is not None:
            model = self._whisper_model_loader(
                model_path, device=device, compute_type=compute
            )
        else:
            from faster_whisper import WhisperModel

            model = WhisperModel(
                model_path,
                device=device,
                compute_type=compute,
                local_files_only=True,
            )
        if use_batched_pipeline:
            pipeline = self.get_batched_pipeline(model)
            return (model, pipeline)
        return (model,)

    def _load_whisperx_bundle(
        self,
        *,
        model: str,
        device: str,
        compute: str,
        language: str,
        vad_backend: str,
        vad_onset: float,
        vad_offset: float,
        vad_model_fp: str | None,
        align_model_path: str | None,
        align_enabled: bool,
        asr_options: dict[str, Any],
        vad_options: dict[str, Any],
    ) -> tuple[Any, ...]:
        """加载 WhisperX ASR + VAD + align 三件套。"""

        vad_model = None
        if vad_backend == "pyannote":
            if self._whisperx_vad_factory is not None:
                vad_model = self._whisperx_vad_factory(
                    device=device,
                    model_fp=vad_model_fp,
                    vad_onset=vad_onset,
                    vad_offset=vad_offset,
                )
            else:
                # 统一以模块对象取子模块，避免测试注入的伪 whisperx（非 package）
                # 让 "whisperx.vads.pyannote" 的字符串导入失败。
                pyannote_mod = importlib.import_module("whisperx.vads.pyannote")
                vad_model = pyannote_mod.Pyannote(
                    device=device,
                    token=None,
                    model_fp=vad_model_fp,
                    vad_onset=vad_onset,
                    vad_offset=vad_offset,
                )

        if self._whisperx_load_model is not None:
            asr = self._whisperx_load_model(
                model,
                device=device,
                compute_type=compute,
                language=language,
                vad_model=vad_model,
                vad_method=vad_backend,
                asr_options=asr_options,
                vad_options=vad_options,
            )
        else:
            whisperx = importlib.import_module("whisperx")
            asr = whisperx.load_model(
                model,
                device=device,
                compute_type=compute,
                language=language,
                vad_model=vad_model,
                vad_method=vad_backend,
                asr_options=asr_options,
                vad_options=vad_options,
            )

        align_model = None
        align_meta = None
        if align_enabled and align_model_path:
            align_model, align_meta = self._load_align(
                language=language, device=device, model_dir=align_model_path
            )
        return (asr, vad_model, align_model, align_meta)

    def _load_align(
        self, *, language: str, device: str, model_dir: str | Path | None
    ) -> tuple[Any, ...]:
        """加载指定语言的 WhisperX align 模型。"""

        if self._whisperx_align_loader is not None:
            return tuple(
                self._whisperx_align_loader(
                    language_code=language, device=device, model_dir=model_dir
                )
            )
        import importlib

        whisperx = importlib.import_module("whisperx")
        return tuple(
            whisperx.load_align_model(
                language_code=language,
                device=device,
                model_dir=str(model_dir) if model_dir else None,
            )
        )


@dataclass
class _CacheConfigView:
    """stats() 中 config 字段的展示载体（避免直接暴露内部引用）。"""

    cache_size: int
    parallelism: int
    acquire_timeout_sec: float
    entries: int = 0
    enabled: bool = True

    def __post_init__(self) -> None:
        """按 cache_size 归一化 enabled，保持与运行期语义一致。"""

        self.enabled = self.cache_size > 0


_global_cache: ModelCache | None = None
_global_lock = threading.Lock()


def get_model_cache() -> ModelCache:
    """返回进程级全局缓存（默认 cache_size=2、parallelism=1）。"""

    global _global_cache
    with _global_lock:
        if _global_cache is None:
            _global_cache = ModelCache()
        return _global_cache


def configure_model_cache(
    *,
    cache_size: int | None = None,
    parallelism: int | None = None,
    acquire_timeout_sec: float | None = None,
    metrics: Any = _UNSET,
) -> dict[str, Any]:
    """配置全局缓存容量、并发上限与 metrics 绑定；由 bootstrap 在 settings 解析后调用。

    `metrics` 省略＝保持现有绑定；传 MetricsState 实例＝绑定；显式 `None`＝解绑。
    绑定后命中/未命中通过 `service.metrics.record_model_cache` 计入真实 Registry
    （SPEC G：whisper_stt_model_cache_hits_total / ..._misses_total）。
    """

    cache = get_model_cache()
    return cache.configure(
        cache_size=cache_size,
        parallelism=parallelism,
        acquire_timeout_sec=acquire_timeout_sec,
        metrics=metrics,
    )


def set_model_cache_metrics(metrics: Any | None) -> None:
    """仅绑定/解绑全局缓存的 metrics，不影响容量与并发参数。

    bootstrap 若分两步装配（先 configure 容量、再注入 metrics）可直接调用本函数；
    `None` 表示解绑。注意：本模块也导出同名函数 `set_model_cache_metrics` 供外部
    调用，`ModelCache.set_metrics` 为实例方法。
    """

    get_model_cache().set_metrics(metrics)


def get_model_cache_stats() -> dict[str, Any]:
    """返回全局缓存统计，供 /admin/model-cache 与 /metrics 复用。"""

    return get_model_cache().stats()


def clear_model_cache(*, force: bool = False) -> dict[str, Any]:
    """清空全局缓存；进程退出时用 `force=True` 释放全部模型。"""

    return get_model_cache().clear(force=force)


def cleanup_model_cache(*, force: bool = False) -> dict[str, Any]:
    """全局缓存清理别名（与 clear_model_cache 等价，语义更偏“释放”）。"""

    return get_model_cache().cleanup(force=force)


def get_whisper_model(
    model_path: str | Path,
    device: str,
    compute_type: str,
    *,
    vad_params: dict[str, Any] | None = None,
    language: str | None = None,
    use_batched_pipeline: bool = False,
) -> ModelLease:
    """兼容入口（SPEC 命名）：在全局缓存上获取 faster-whisper 模型租约。"""

    return get_model_cache().get_whisper_model(
        model_path,
        device,
        compute_type,
        vad_params=vad_params,
        language=language,
        use_batched_pipeline=use_batched_pipeline,
    )


def get_batched_pipeline(model: Any) -> Any:
    """兼容入口：按需为模型构造 batched pipeline（与模型同生命周期）。"""

    return get_model_cache().get_batched_pipeline(model)


def get_whisperx_bundle(**kwargs: Any) -> ModelLease:
    """兼容入口（SPEC 命名）：在全局缓存上获取 WhisperX 三件套租约。"""

    return get_model_cache().get_whisperx_bundle(**kwargs)


def get_align_model(
    language: str, device: str, model_dir: str | Path | None = None
) -> ModelLease:
    """兼容入口（SPEC 命名）：在全局缓存上按语言获取 align 模型租约。"""

    return get_model_cache().get_align_model(language, device, model_dir)


def acquire_whisper_model(*args: Any, **kwargs: Any) -> ModelLease:
    """语义化别名，等价于 `get_whisper_model`（acquire/with 上下文写法更直观）。"""

    return get_whisper_model(*args, **kwargs)


def acquire_whisperx_bundle(**kwargs: Any) -> ModelLease:
    """语义化别名，等价于 `get_whisperx_bundle`。"""

    return get_whisperx_bundle(**kwargs)


def acquire_align_model(
    language: str, device: str, model_dir: str | Path | None = None
) -> ModelLease:
    """语义化别名，等价于 `get_align_model`。"""

    return get_align_model(language, device, model_dir)


def reset_model_cache_for_tests() -> None:
    """单元测试专用：丢弃全局缓存实例，避免跨用例状态串扰。"""

    global _global_cache
    with _global_lock:
        if _global_cache is not None:
            _global_cache.clear(force=True)
        _global_cache = None


__all__ = [
    "DEFAULT_MODEL_CACHE_SIZE",
    "DEFAULT_MODEL_PARALLELISM",
    "MODEL_SIGNAL_WARN_WAIT_MS",
    "ModelCache",
    "ModelLease",
    "STT_VAD_PARAM_KEYS",
    "acquire_align_model",
    "acquire_whisper_model",
    "acquire_whisperx_bundle",
    "cleanup_model_cache",
    "clear_model_cache",
    "configure_model_cache",
    "get_align_model",
    "get_batched_pipeline",
    "get_model_cache",
    "get_model_cache_stats",
    "get_whisper_model",
    "get_whisperx_bundle",
    "make_cache_key",
    "normalize_vad_params",
    "reset_model_cache_for_tests",
    "resolve_runtime_hint",
    "set_model_cache_metrics",
]