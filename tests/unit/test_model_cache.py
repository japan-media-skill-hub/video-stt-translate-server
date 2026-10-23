"""service/model_cache 单元测试：key 覆盖、LRU、引用保护、信号量与日志。"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from whisper_stt_service.service import model_cache as mc
from whisper_stt_service.service.model_cache import (
    ModelCache,
    make_cache_key,
    normalize_vad_params,
    reset_model_cache_for_tests,
)


class FakeModel:
    """带 close 记录的假模型，用于验证逐出时资源释放。"""

    def __init__(self, name: str) -> None:
        """记录模型名与关闭状态。"""

        self.name = name
        self.closed = False

    def close(self) -> None:
        """标记为已关闭。"""

        self.closed = True


def _make_cache(**kwargs) -> tuple[ModelCache, list[FakeModel]]:
    """构造注入 fake loader 的缓存，返回 (cache, 已加载模型列表)。"""

    created: list[FakeModel] = []

    def loader(model_path, *, device, compute_type):
        """伪造模型加载，记录调用次数。"""

        model = FakeModel(f"{model_path}|{device}|{compute_type}")
        created.append(model)
        return model

    cache = ModelCache(whisper_model_loader=loader, **kwargs)
    return cache, created


VAD_JA = {
    "vad_filter": True,
    "vad_threshold": 0.45,
    "vad_min_speech_duration_ms": 200,
    "vad_max_speech_duration_s": 18.0,
    "vad_min_silence_duration_ms": 700,
    "vad_speech_pad_ms": 300,
}


def test_same_key_loads_model_once() -> None:
    """D：相同 key 只冷启动一次，第二次命中复用同一实例。"""

    cache, created = _make_cache(cache_size=2)
    with cache.get_whisper_model("/m", "cpu", "int8", vad_params=VAD_JA, language="ja") as first:
        assert first.cached is False
        model_first = first.value
    with cache.get_whisper_model("/m", "cpu", "int8", vad_params=VAD_JA, language="ja") as second:
        assert second.cached is True
        assert second.value is model_first
    assert len(created) == 1
    stats = cache.stats()
    assert stats["counters"]["misses"] == 1
    assert stats["counters"]["hits"] == 1


@pytest.mark.parametrize(
    "variant",
    [
        {"device": "cuda"},
        {"compute_type": "float16"},
        {"language": "en"},
        {"vad_params": {**VAD_JA, "vad_threshold": 0.5}},
        {"vad_params": {**VAD_JA, "vad_min_silence_duration_ms": 900}},
    ],
)
def test_cache_key_covers_device_compute_vad_language(variant) -> None:
    """D：设备/compute/VAD/语言任一不同都不得复用同一模型实例。"""

    cache, created = _make_cache(cache_size=8)
    kwargs = {"vad_params": VAD_JA, "language": "ja"}
    kwargs.update(variant)
    device = kwargs.pop("device", "cpu")
    compute = kwargs.pop("compute_type", "int8")

    with cache.get_whisper_model("/m", "cpu", "int8", vad_params=VAD_JA, language="ja") as base:
        base_model = base.value
    with cache.get_whisper_model("/m", device, compute, **kwargs) as other:
        assert other.value is not base_model
    assert len(created) == 2


def test_lru_evicts_least_recently_used() -> None:
    """D：超出 cache_size 时按 LRU 逐出，并关闭被逐出模型。"""

    cache, created = _make_cache(cache_size=2)
    with cache.get_whisper_model("/a", "cpu", "int8") as a:
        assert a.value in created
    with cache.get_whisper_model("/b", "cpu", "int8"):
        pass
    with cache.get_whisper_model("/c", "cpu", "int8"):
        pass

    assert [m.name.split("|")[0] for m in created] == ["/a", "/b", "/c"]
    assert created[0].closed is True  # /a 被逐出并释放
    assert created[1].closed is False
    assert cache.stats()["counters"]["evictions"] == 1


def test_lru_never_evicts_active_reference() -> None:
    """D：LRU 不得逐出仍有活跃引用的条目（临时超限并告警）。"""

    cache, created = _make_cache(cache_size=1)
    lease_a = cache.get_whisper_model("/a", "cpu", "int8")
    model_a = lease_a.value
    try:
        with cache.get_whisper_model("/b", "cpu", "int8") as lease_b:
            # size=1 且 /a 仍活跃：/b 入表后不逐出 /a；/b 释放后自身可被逐出。
            assert model_a.closed is False
            assert lease_b.cached is False
        assert cache.stats()["counters"]["over_capacity"] >= 1
        assert model_a.closed is False
    finally:
        lease_a.release()
    # /a 释放后，下一次 acquire 会按 LRU 先逐出 /a（最近未使用则逐出自身条目）。
    with cache.get_whisper_model("/c", "cpu", "int8"):
        pass
    assert len(cache.stats()["entries"]) <= 3


def test_release_after_over_capacity_trims_back_to_limit() -> None:
    """P2 回归：cache_size=1 时 A/B 同时活跃可临时超限；两次 release 后必须收敛回 1 并释放模型。"""

    cache, created = _make_cache(cache_size=1)
    lease_a = cache.get_whisper_model("/a", "cpu", "int8")
    lease_b = cache.get_whisper_model("/b", "cpu", "int8")
    # 两者都活跃：允许临时超限，且绝不逐出活跃引用。
    assert len(cache.stats()["entries"]) == 2
    assert cache.stats()["counters"]["over_capacity"] >= 1
    assert created[0].closed is False and created[1].closed is False

    lease_a.release()
    # 仍有一个活跃（b），不得逐出它；条目数仍未超限（2>1 但都/其一是活跃）。
    assert created[1].closed is False
    lease_b.release()

    # 全部失活后按 LRU 收敛：条目数回到 cache_size，且被逐出的模型被关闭。
    assert len(cache.stats()["entries"]) == 1
    assert created[0].closed is True
    assert created[1].closed is False


def test_release_never_evicts_active_entries() -> None:
    """P2：收敛时只淘汰空闲条目；仍被持有的条目必须保留。"""

    cache, created = _make_cache(cache_size=1)
    lease_a = cache.get_whisper_model("/a", "cpu", "int8")
    lease_b = cache.get_whisper_model("/b", "cpu", "int8")
    lease_c = cache.get_whisper_model("/c", "cpu", "int8")
    lease_b.release()  # b 空闲 → 可被逐出
    assert len(cache.stats()["entries"]) == 2
    assert created[1].closed is True  # b 被收敛逐出
    assert created[0].closed is False  # a 仍活跃
    assert created[2].closed is False  # c 仍活跃
    lease_a.release()
    lease_c.release()
    assert len(cache.stats()["entries"]) == 1


def test_cache_size_zero_releases_on_each_release() -> None:
    """cache_size=0：每次 release 都立即丢弃条目，不留空闲实例。"""

    cache, created = _make_cache(cache_size=0)
    lease_a = cache.get_whisper_model("/a", "cpu", "int8")
    lease_b = cache.get_whisper_model("/b", "cpu", "int8")
    assert created[0].closed is False
    lease_a.release()
    assert created[0].closed is True
    assert cache.stats()["entries"] == []
    lease_b.release()
    assert created[1].closed is True
    assert cache.stats()["entries"] == []


def test_cache_size_zero_disables_cache() -> None:
    """D：cache_size=0 禁用缓存，每次 acquire 都重新加载并在释放后丢弃。"""

    cache, created = _make_cache(cache_size=0)
    with cache.get_whisper_model("/m", "cpu", "int8") as first:
        model_first = first.value
    with cache.get_whisper_model("/m", "cpu", "int8") as second:
        assert second.value is not model_first
        assert second.cached is False
    assert len(created) == 2
    assert cache.enabled is False
    assert cache.stats()["entries"] == []
    assert cache.stats()["counters"]["uncached_loads"] == 2


def test_parallelism_semaphore_serializes_transcribe() -> None:
    """D：model_parallelism=1 时同实例并发转写串行，不失败。"""

    cache, _created = _make_cache(cache_size=1, parallelism=1)
    lease = cache.get_whisper_model("/m", "cpu", "int8")
    active = 0
    peak = 0
    peak_lock = threading.Lock()

    def worker() -> None:
        """模拟一次转写占用。"""

        nonlocal active, peak
        with lease.transcribe_guard():
            with peak_lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with peak_lock:
                active -= 1

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lease.release()
    assert peak == 1


def test_parallelism_two_allows_two_concurrent() -> None:
    """D：model_parallelism=2 时允许两个并发转写。"""

    cache, _created = _make_cache(cache_size=1, parallelism=2)
    lease = cache.get_whisper_model("/m", "cpu", "int8")
    active = 0
    peak = 0
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def worker() -> None:
        """两个线程同时进入临界区，观察峰值并发。"""

        nonlocal active, peak
        with lease.transcribe_guard():
            with lock:
                active += 1
                peak = max(peak, active)
            barrier.wait(timeout=5)
            with lock:
                active -= 1

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lease.release()
    assert peak == 2


def test_concurrent_cold_start_loads_once() -> None:
    """D：并发冷启动 same key 只真正加载一次（single-flight）。"""

    loads = []
    release_gate = threading.Event()

    def slow_loader(model_path, *, device, compute_type):
        """慢加载器，让并发线程在加载窗口内汇合。"""

        loads.append(1)
        release_gate.wait(timeout=5)
        return FakeModel("slow")

    cache = ModelCache(whisper_model_loader=slow_loader, cache_size=2)
    results: list[object] = []
    errors: list[BaseException] = []

    def acquire_one() -> None:
        """并发 acquire 并记录结果。"""

        try:
            with cache.get_whisper_model("/m", "cpu", "int8") as lease:
                results.append(lease.value)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=acquire_one) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.1)
    release_gate.set()
    for t in threads:
        t.join()

    assert errors == []
    assert len(loads) == 1
    assert len({id(item) for item in results}) == 1


def test_cold_and_hit_logging(caplog) -> None:
    """D：冷启动/命中分别写 model_cache_miss / model_cache_hit 日志。"""

    cache, _created = _make_cache(cache_size=2)
    with caplog.at_level(logging.INFO, logger=mc.LOGGER.name):
        with cache.get_whisper_model("/m", "cpu", "int8"):
            pass
        with caplog.at_level(logging.DEBUG, logger=mc.LOGGER.name):
            with cache.get_whisper_model("/m", "cpu", "int8"):
                pass
    text = caplog.text
    assert "model_cache_miss" in text
    assert "elapsed_ms=" in text
    assert "model_cache_hit" in text


def test_clear_skips_active_and_force_releases_all() -> None:
    """清理 API：默认跳过活跃引用，force=True 时全部丢弃。"""

    cache, created = _make_cache(cache_size=2)
    lease = cache.get_whisper_model("/m", "cpu", "int8")
    assert cache.clear()["skipped_active"] == 1
    assert created[0].closed is False
    assert cache.clear(force=True)["dropped"] == 1
    assert created[0].closed is True
    # force 之后旧 lease 的 release 不得报错。
    lease.release()


def test_configure_lowers_cache_size_evicts_idle_entries() -> None:
    """配置 API：调小 cache_size 会按 LRU 立即逐出空闲条目。"""

    cache, created = _make_cache(cache_size=3)
    with cache.get_whisper_model("/a", "cpu", "int8"):
        pass
    with cache.get_whisper_model("/b", "cpu", "int8"):
        pass
    config = cache.configure(cache_size=1)
    assert config["cache_size"] == 1
    assert len(cache.stats()["entries"]) == 1
    assert created[0].closed is True


def test_whisperx_bundle_key_covers_vad_and_align() -> None:
    """D：WhisperX 三件套 key 覆盖 VAD 与 align 参数，不同参数不复用。"""

    calls: list[dict] = []

    def load_model(model, **kwargs):
        """伪造 whisperx.load_model。"""

        calls.append({"model": model, **kwargs})
        return FakeModel("asr")

    def vad_factory(**kwargs):
        """伪造 pyannote VAD。"""

        return FakeModel("vad")

    def align_loader(*, language_code, device, model_dir):
        """伪造 align 模型加载。"""

        return (FakeModel("align"), {"language": language_code})

    cache = ModelCache(
        whisperx_load_model=load_model,
        whisperx_vad_factory=vad_factory,
        whisperx_align_loader=align_loader,
        cache_size=4,
    )
    base = dict(
        device="cpu",
        compute_type="int8",
        language="ja",
        vad_config_path="/vad/config.yaml",
        vad_backend="pyannote",
        vad_onset=0.35,
        vad_offset=0.2,
        vad_model_fp="/vad/pytorch_model.bin",
        align_model_path="/align/ja",
        align_enabled=True,
    )
    with cache.get_whisperx_bundle("/m", **base) as first:
        assert len(first.resources) == 4
        assert first.resources[2].name == "align"
    with cache.get_whisperx_bundle("/m", **base) as second:
        assert second.cached is True
    assert len(calls) == 1

    with cache.get_whisperx_bundle("/m", **{**base, "vad_onset": 0.5}) as third:
        assert third.cached is False
    with cache.get_whisperx_bundle("/m", **{**base, "language": "en"}) as fourth:
        assert fourth.cached is False
    assert len(calls) == 3


def test_align_model_cached_per_language() -> None:
    """D：align 模型按语言缓存，同语言复用、跨语言不串用。"""

    calls: list[str] = []

    def align_loader(*, language_code, device, model_dir):
        """伪造 align 加载并记录语言。"""

        calls.append(language_code)
        return (FakeModel(f"align-{language_code}"), {"language": language_code})

    cache = ModelCache(whisperx_align_loader=align_loader, cache_size=4)
    with cache.get_align_model("ja", "cpu", "/align") as ja:
        ja_value = ja.value
    with cache.get_align_model("ja", "cpu", "/align") as ja_again:
        assert ja_again.value is ja_value
    with cache.get_align_model("en", "cpu", "/align"):
        pass
    assert calls == ["ja", "en"]


def test_get_batched_pipeline_reuses_per_model() -> None:
    """D：batched pipeline 与模型同生命周期，同一模型实例只构造一次。"""

    built: list[object] = []

    def pipeline_loader(model):
        """伪造 pipeline 构造。"""

        built.append(model)
        return FakeModel("pipeline")

    cache = ModelCache(batched_pipeline_loader=pipeline_loader, cache_size=2)
    model = FakeModel("model")
    first = cache.get_batched_pipeline(model)
    second = cache.get_batched_pipeline(model)
    assert first is second
    assert len(built) == 1


def test_global_helpers_configure_stats_and_clear() -> None:
    """全局辅助函数：configure/stats/clear 可被 bootstrap 与运维链接入。"""

    reset_model_cache_for_tests()
    try:
        conf = mc.configure_model_cache(cache_size=1, parallelism=2)
        assert conf["cache_size"] == 1
        assert conf["parallelism"] == 2
        stats = mc.get_model_cache_stats()
        assert stats["config"]["cache_size"] == 1
        assert stats["config"]["parallelism"] == 2
        assert mc.clear_model_cache()["forced"] is False
        assert mc.get_model_cache() is mc.get_model_cache()
    finally:
        reset_model_cache_for_tests()


def test_slot_wait_is_unbounded_even_with_small_acquire_timeout() -> None:
    """P1 回归：acquire_timeout_sec 极小也不得“超时放行”，第二个 transcribe 必须排队。"""

    cache, _created = _make_cache(cache_size=1, parallelism=1, acquire_timeout_sec=0.01)
    lease_a = cache.get_whisper_model("/m", "cpu", "int8")
    # 用真 BoundedSemaphore 观测：a 持有时信号量值为 0。
    assert lease_a.transcribe_slot.acquire(blocking=False) is True
    lease_a.transcribe_slot.release()

    entered: list[float] = []
    lease_b = cache.get_whisper_model("/m", "cpu", "int8")

    def second_worker() -> None:
        """b 在 a 持有 slot 期间尝试获取，必须阻塞到 a 释放为止。"""

        lease_b.acquire_transcribe_slot()
        try:
            entered.append(time.monotonic())
        finally:
            lease_b.release_transcribe_slot()

    thread = threading.Thread(target=second_worker)
    lease_a.acquire_transcribe_slot()
    started = time.monotonic()
    thread.start()
    # 远超过 acquire_timeout_sec=0.01 后，b 仍不得进入。
    time.sleep(0.3)
    assert entered == [], "排队者不得在超时后放行"
    # 释放 a 的许可后，b 应在轮询周期内进入。
    lease_a.release_transcribe_slot()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(entered) == 1
    assert time.monotonic() - started >= 0.3


def test_release_without_acquire_never_increases_semaphore_capacity() -> None:
    """P1 回归：未获取到许可的租约调用 release 不得增加容量（无幽灵许可）。"""

    cache, _created = _make_cache(cache_size=1, parallelism=1)
    lease = cache.get_whisper_model("/m", "cpu", "int8")
    semaphore = lease.transcribe_slot

    # 未 acquire 就 release：必须是空操作。
    lease.release_transcribe_slot()
    assert semaphore.acquire(blocking=False) is True
    assert semaphore.acquire(blocking=False) is False, "未持有者释放凭空增加了容量"
    semaphore.release()
    lease.release()


def test_release_is_idempotent_and_only_returns_owned_slot() -> None:
    """P1 回归：重复 release 只归还一次；跨租约共享信号量也不会被多释放。"""

    cache, _created = _make_cache(cache_size=1, parallelism=1)
    lease_a = cache.get_whisper_model("/m", "cpu", "int8")
    lease_a.acquire_transcribe_slot()
    lease_a.release_transcribe_slot()
    lease_a.release_transcribe_slot()  # 重复释放：空操作
    # 容量仍为 1：只能拿到 1 个许可。
    assert lease_a.transcribe_slot.acquire(blocking=False) is True
    assert lease_a.transcribe_slot.acquire(blocking=False) is False
    lease_a.transcribe_slot.release()
    lease_a.release()


def test_waiting_slot_can_be_cancelled() -> None:
    """drain/失租：排队中的 acquire 通过 cancel_check 中止，而不是永远卡住。"""

    from whisper_stt_service.executor.common import ExecutorCancelledError

    cache, _created = _make_cache(cache_size=1, parallelism=1)
    lease_a = cache.get_whisper_model("/m", "cpu", "int8")
    lease_b = cache.get_whisper_model("/m", "cpu", "int8")
    lease_a.acquire_transcribe_slot()
    try:
        with pytest.raises(ExecutorCancelledError):
            lease_b.acquire_transcribe_slot(cancel_check=lambda: True, poll_sec=0.01)
    finally:
        lease_a.release_transcribe_slot()
        lease_a.release()
        lease_b.release()


def test_metrics_injection_counts_real_registry() -> None:
    """G：绑定 MetricsState 后，冷启动/命中计入真实 prometheus Registry 计数。"""

    from whisper_stt_service.service.metrics import build_metrics

    metrics = build_metrics()
    if getattr(metrics, "registry", None) is None:
        pytest.skip("prometheus_client 未安装（外部依赖缺失）")

    cache, _created = _make_cache(cache_size=2, metrics=metrics)
    with cache.get_whisper_model("/m", "cpu", "int8"):
        pass
    with cache.get_whisper_model("/m", "cpu", "int8"):
        pass

    assert metrics.model_cache_misses_total._value.get() == 1
    assert metrics.model_cache_hits_total._value.get() == 1
    text = metrics.render().decode("utf-8")
    assert "whisper_stt_model_cache_hits_total 1" in text
    assert "whisper_stt_model_cache_misses_total 1" in text


def test_configure_metrics_sentinel_semantics() -> None:
    """configure 的 metrics 参数：省略=不变、实例=绑定、显式 None=解绑。"""

    calls: list[bool] = []

    class RecordingMetrics:
        """记录 hit 调用的轻量 metrics 替身。"""

        def record_model_cache(self, *, hit: bool) -> None:
            """记录一次命中/未命中。"""

            calls.append(bool(hit))

    metrics = RecordingMetrics()
    cache, _created = _make_cache(cache_size=4)
    # 1) 省略 metrics：不改变（初始未绑定，因此不记录）。
    cache.configure(cache_size=4)
    with cache.get_whisper_model("/a", "cpu", "int8"):
        pass
    assert calls == []

    # 2) 传入实例：绑定后可记录。
    cache.configure(metrics=metrics)
    with cache.get_whisper_model("/a", "cpu", "int8"):
        pass
    assert calls == [True]

    # 3) 显式 None：解绑，不再记录。
    cache.configure(metrics=None)
    with cache.get_whisper_model("/a", "cpu", "int8"):
        pass
    assert calls == [True]

    # 4) set_model_cache_metrics 全局解绑不影响容量参数。
    mc.reset_model_cache_for_tests()
    try:
        mc.configure_model_cache(cache_size=3, metrics=metrics)
        assert mc.get_model_cache_stats()["config"]["cache_size"] == 3
        mc.set_model_cache_metrics(None)
        assert mc.get_model_cache_stats()["config"]["cache_size"] == 3
    finally:
        mc.reset_model_cache_for_tests()


def test_metrics_failure_does_not_break_cache() -> None:
    """metrics 实现异常时不得影响转写主流程（降级为 debug 日志）。"""

    class BrokenMetrics:
        """任何上报都会抛错的 metrics 替身。"""

        def record_model_cache(self, *, hit: bool) -> None:  # noqa: D401
            """故意抛错。"""

            raise RuntimeError("metrics backend down")

    cache, created = _make_cache(cache_size=2)

    # record_model_cache 走 service.metrics 模块函数；对 BrokenMetrics（无 registry 字段）
    # 会抛错，此时缓存必须继续工作并返回正常 lease。
    cache.configure(metrics=BrokenMetrics())
    with cache.get_whisper_model("/m", "cpu", "int8") as lease:
        assert lease.value in created


def test_make_cache_key_is_stable_and_readable() -> None:
    """key 拼接稳定：同样参数得到同样字符串，且包含关键维度。"""

    vad = normalize_vad_params({"vad_threshold": 0.45, "vad_filter": True})
    key_a = make_cache_key("whisper_rt", model="/m", device="cuda", compute="float16", vad=vad)
    key_b = make_cache_key(
        "whisper_rt", model="/m", device="cuda", compute="float16", vad=vad
    )
    assert key_a == key_b
    assert "model=/m" in key_a
    assert "device=cuda" in key_a
    assert "compute=float16" in key_a
    # VAD 以有序元组形式参与 key，便于排障时直接看出生效参数。
    assert "vad_threshold', '0.45'" in key_a
