"""执行器 wav 契约（SPEC E）、取消回调与 ASR 原子输出（C.7）单元测试。

全部测试使用 fake faster_whisper / whisperx / ffmpeg，不依赖真实模型、GPU 与媒体文件。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import threading

import pytest

from whisper_stt_service.executor.common import (
    ExecutorCancelledError,
    MissingAudioArtifactError,
)
from whisper_stt_service.executor.extract import run_extract
from whisper_stt_service.executor.stt import run_stt
from whisper_stt_service.executor.stt_whisperx import run_stt_whisperx
from whisper_stt_service.service.model_cache import reset_model_cache_for_tests


class _StopError(RuntimeError):
    """模拟模型在转写中途抛出的业务异常。"""


def _install_fake_faster_whisper(monkeypatch, segments) -> dict[str, list]:
    """安装 fake faster_whisper 模块，记录实例化与转写调用。"""

    calls: dict[str, list] = {"init": [], "transcribe": []}

    class FakeWhisperModel:
        """可迭代段生成器的假模型。"""

        def __init__(self, *args, **kwargs) -> None:
            """记录构造参数。"""

            calls["init"].append({"args": args, "kwargs": kwargs})

        def transcribe(self, audio, **kwargs):
            """记录转写参数并按需产出段（`segments` 可以是可调用对象以便抛错）。"""

            calls["transcribe"].append({"audio": audio, "kwargs": kwargs})
            payload = segments() if callable(segments) else segments
            return payload, None

    class FakeBatchedInferencePipeline:
        """假 batched pipeline。"""

        def __init__(self, model) -> None:
            """保存底层模型。"""

            self.model = model

        def transcribe(self, audio, **kwargs):
            """记录 batched 转写参数。"""

            calls["transcribe"].append({"audio": audio, "kwargs": kwargs, "batched": True})
            payload = segments() if callable(segments) else segments
            return payload, None

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(
            WhisperModel=FakeWhisperModel,
            BatchedInferencePipeline=FakeBatchedInferencePipeline,
        ),
    )
    return calls


def _segs(*items):
    """构造 faster-whisper 风格段对象。"""

    return iter([SimpleNamespace(text=t, start=s, end=e) for t, s, e in items])


def _quiet_progress(monkeypatch) -> None:
    """关闭进度事件与 ffprobe，避免测试依赖外部命令。"""

    for module in ("stt", "stt_whisperx", "extract"):
        monkeypatch.setattr(
            f"whisper_stt_service.executor.{module}._emit_progress",
            lambda *args, **kwargs: None,
        )
    monkeypatch.setattr(
        "whisper_stt_service.executor.stt._probe_duration",
        lambda _path: 10.0,
    )
    monkeypatch.setattr(
        "whisper_stt_service.executor.stt_whisperx._probe_duration",
        lambda _path: 10.0,
    )


def _audio(tmp_path: Path, name: str = "demo.16k.wav", content: bytes = b"RIFF-fake") -> Path:
    """创建满足 wav 契约的假音频文件。"""

    path = tmp_path / name
    path.write_bytes(content)
    return path


@pytest.fixture(autouse=True)
def _reset_cache():
    """每个用例前后重置全局模型缓存，避免跨用例命中。"""

    reset_model_cache_for_tests()
    yield
    reset_model_cache_for_tests()


# ---------------------------------------------------------------------------
# E：wav 契约
# ---------------------------------------------------------------------------
def test_run_stt_rejects_missing_wav(tmp_path: Path, monkeypatch) -> None:
    """E：wav 不存在时必须失败并抛 missing_audio_artifact，不得回退视频。"""

    _quiet_progress(monkeypatch)
    calls = _install_fake_faster_whisper(monkeypatch, _segs())
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")
    # ASR 输入是 extract 产出的 audio/<stem>.16k.wav，视频存在也不得被回退使用。
    missing_wav = tmp_path / "audio" / "demo.16k.wav"
    out = tmp_path / "asr" / "demo.ja.srt"

    with pytest.raises(MissingAudioArtifactError) as excinfo:
        run_stt(
            missing_wav,
            out,
            language="ja",
            timeout_sec=60,
            model="/m",
            device="cpu",
            require_audio_artifact=True,
        )

    assert excinfo.value.code == "missing_audio_artifact"
    assert str(excinfo.value).startswith("missing_audio_artifact:")
    assert str(missing_wav) in str(excinfo.value)
    assert calls["init"] == []
    assert not out.exists()


def test_run_stt_rejects_empty_wav(tmp_path: Path, monkeypatch) -> None:
    """E：0 字节 wav 同样按缺失处理。"""

    _quiet_progress(monkeypatch)
    _install_fake_faster_whisper(monkeypatch, _segs())
    empty = _audio(tmp_path, content=b"")
    out = tmp_path / "demo.ja.srt"

    with pytest.raises(MissingAudioArtifactError):
        run_stt(empty, out, language="ja", timeout_sec=60, model="/m", device="cpu", require_audio_artifact=True)
    assert not out.exists()


def test_run_stt_consumes_wav_input(tmp_path: Path, monkeypatch) -> None:
    """E：服务模式转写读取 wav（而不是视频），并写出正确 SRT。"""

    _quiet_progress(monkeypatch)
    calls = _install_fake_faster_whisper(monkeypatch, _segs((" こんにちは ", 0.0, 1.5)))
    wav = _audio(tmp_path)
    out = tmp_path / "asr" / "demo.ja.srt"

    result = run_stt(
        wav,
        out,
        language="ja",
        timeout_sec=60,
        model="/m",
        device="cpu",
        compute_type="int8",
        batch_size=1,
        require_audio_artifact=True,
    )

    assert calls["transcribe"][0]["audio"] == str(wav)
    assert calls["init"][0]["kwargs"]["compute_type"] == "int8"
    assert out.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,500\nこんにちは\n\n"
    assert result["model_cache_hit"] is False
    assert "model_cache_key" in result


def test_run_stt_second_task_hits_model_cache(tmp_path: Path, monkeypatch) -> None:
    """D：同一配置的两个 task 只冷启动一次模型（缓存命中）。"""

    _quiet_progress(monkeypatch)
    calls = _install_fake_faster_whisper(monkeypatch, _segs(("a", 0.0, 1.0)))
    wav = _audio(tmp_path)

    first = run_stt(wav, tmp_path / "a.ja.srt", language="ja", timeout_sec=60, model="/m", device="cpu")
    second = run_stt(wav, tmp_path / "b.ja.srt", language="ja", timeout_sec=60, model="/m", device="cpu")

    assert len(calls["init"]) == 1
    assert first["model_cache_hit"] is False
    assert second["model_cache_hit"] is True


def test_run_stt_uses_legacy_behaviour_without_require_flag(tmp_path: Path, monkeypatch) -> None:
    """兼容性：未传 require_audio_artifact 时保留旧行为（脚本模式可直接传视频）。"""

    _quiet_progress(monkeypatch)
    _install_fake_faster_whisper(monkeypatch, _segs(("a", 0.0, 1.0)))
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")
    out = tmp_path / "demo.ja.srt"
    run_stt(video, out, language="ja", timeout_sec=60, model="/m", device="cpu")
    assert out.exists()


def test_run_stt_accepts_io_path_aliases(tmp_path: Path, monkeypatch) -> None:
    """B.3/E：runtime 按 io 段传 input_path/output_path，执行器需接受且优先使用。"""

    _quiet_progress(monkeypatch)
    calls = _install_fake_faster_whisper(monkeypatch, _segs(("あ", 0.0, 1.0)))
    wav = _audio(tmp_path)
    out = tmp_path / "asr" / "demo.ja.srt"
    decoy = tmp_path / "decoy.mp4"
    decoy.write_bytes(b"video")

    run_stt(
        input_video=decoy,
        output_ja_srt=tmp_path / "decoy.ja.srt",
        language="ja",
        timeout_sec=60,
        input_path=wav,
        output_path=out,
        model="/m",
        device="cpu",
        require_audio_artifact=True,
    )

    assert calls["transcribe"][0]["audio"] == str(wav)
    assert out.exists()
    assert not (tmp_path / "decoy.ja.srt").exists()


def test_run_stt_requires_both_path_forms(tmp_path: Path, monkeypatch) -> None:
    """路径归一化：两种参数形态都缺失时明确报错，不静默使用视频。"""

    _quiet_progress(monkeypatch)
    _install_fake_faster_whisper(monkeypatch, _segs())
    with pytest.raises(ValueError, match="stt_io_paths_missing"):
        run_stt(language="ja", timeout_sec=60, model="/m", device="cpu")


class _LazySegmentModel:
    """模拟 faster-whisper 的惰性 generator：推理发生在迭代段时，而不是 transcribe 调用时。

    每次 `next` 之间用 Barrier 让两个线程汇合，从而**精确观测迭代期间的并发峰值**。
    """

    def __init__(self, barrier: threading.Barrier, state: dict) -> None:
        """保存同步屏障与共享计数状态。"""

        self._barrier = barrier
        self._state = state
        self.transcribe_called = 0
        self.iter_peak = 0
        self._lock = threading.Lock()

    def transcribe(self, audio, **kwargs):
        """返回惰性 generator（调用本身不做推理）。"""

        with self._lock:
            self.transcribe_called += 1

        def _gen():
            """真正的“推理”：迭代期间统计并发数。"""

            for index in range(3):
                with self._lock:
                    self._state["active"] += 1
                    self.iter_peak = max(self.iter_peak, self._state["active"])
                try:
                    # 两个线程若同时迭代，会在屏障处互相等待并超时失败。
                    self._barrier.wait(timeout=0.3)
                except threading.BrokenBarrierError:
                    # 串行执行时屏障只有一个人，超时即说明并发被限制住了。
                    pass
                finally:
                    with self._lock:
                        self._state["active"] -= 1
                yield SimpleNamespace(text=f"seg{index}", start=float(index), end=index + 1.0)

        return _gen(), None


def test_lazy_generator_iteration_is_serialized_by_model_parallelism(
    tmp_path: Path, monkeypatch
) -> None:
    """P1（Sol）：许可必须覆盖惰性 generator 的整个迭代区间，迭代峰值 <= 1。"""

    from whisper_stt_service.service.model_cache import configure_model_cache, reset_model_cache_for_tests

    reset_model_cache_for_tests()
    configure_model_cache(cache_size=2, parallelism=1)
    try:
        barrier = threading.Barrier(2)
        state = {"active": 0}
        models: list[_LazySegmentModel] = []

        class FakeWhisperModel:
            """每次 get_whisper_model 冷启动时创建，第二次命中缓存复用同一实例。"""

            def __init__(self, *args, **kwargs) -> None:
                """构造带屏障的惰性模型。"""

                self.impl = _LazySegmentModel(barrier, state)
                models.append(self.impl)

            def transcribe(self, audio, **kwargs):
                """委托给惰性实现。"""

                return self.impl.transcribe(audio, **kwargs)

        monkeypatch.setitem(
            sys.modules,
            "faster_whisper",
            SimpleNamespace(
                WhisperModel=FakeWhisperModel,
                BatchedInferencePipeline=lambda model: model,
            ),
        )
        _quiet_progress(monkeypatch)
        wav = _audio(tmp_path, "parallel.16k.wav")
        results: list[BaseException | None] = []

        def worker(tag: str) -> None:
            """并发执行一次 run_stt，记录异常。"""

            try:
                run_stt(
                    wav,
                    tmp_path / f"{tag}.ja.srt",
                    language="ja",
                    timeout_sec=30,
                    model="/m",
                    device="cpu",
                    compute_type="int8",
                    batch_size=1,
                    require_audio_artifact=True,
                    progress_every=1,
                )
                results.append(None)
            except BaseException as exc:  # noqa: BLE001
                results.append(exc)

        threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert results == [None, None], f"并发转写失败: {results}"
        assert len(models) == 1, "同 key 应只冷启动一次模型"
        assert models[0].transcribe_called == 2
        # 关键断言：惰性迭代期间的并发峰值必须 <= 1（model_parallelism=1）。
        assert models[0].iter_peak == 1
        for tag in ("t0", "t1"):
            assert (tmp_path / f"{tag}.ja.srt").is_file()
    finally:
        reset_model_cache_for_tests()


def test_lazy_generator_cancel_releases_slot(tmp_path: Path, monkeypatch) -> None:
    """取消导致迭代中途异常时，许可仍必须归还（同实例第二个任务能拿到）。"""

    reset_model_cache_for_tests = __import__(
        "whisper_stt_service.service.model_cache", fromlist=["x"]
    ).reset_model_cache_for_tests
    reset_model_cache_for_tests()

    class ExplodingModel:
        """迭代到第二段时抛错的惰性模型。"""

        def transcribe(self, audio, **kwargs):
            """返回会抛错的 generator。"""

            def _gen():
                """第一段正常，第二段抛错。"""

                yield SimpleNamespace(text="a", start=0.0, end=1.0)
                raise _StopError("boom during iteration")

            return _gen(), None

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(
            WhisperModel=lambda *a, **k: ExplodingModel(),
            BatchedInferencePipeline=lambda model: model,
        ),
    )
    _quiet_progress(monkeypatch)
    from whisper_stt_service.service.model_cache import get_model_cache

    wav = _audio(tmp_path, "boom.16k.wav")
    with pytest.raises(_StopError):
        run_stt(
            wav,
            tmp_path / "boom.ja.srt",
            language="ja",
            timeout_sec=30,
            model="/m",
            device="cpu",
            compute_type="int8",
            batch_size=1,
        )
    cache = get_model_cache()
    entries = cache.stats()["entries"]
    assert entries, "模型条目应仍在缓存中"
    assert entries[0]["refs"] == 0, "异常路径也必须归还引用/许可"
    reset_model_cache_for_tests()


# ---------------------------------------------------------------------------
# C.7：ASR 原子输出
# ---------------------------------------------------------------------------
def test_run_stt_failure_keeps_previous_srt_and_no_temp(tmp_path: Path, monkeypatch) -> None:
    """C.7：转写中途异常时不得留下半成品，也不破坏既有产物。"""

    _quiet_progress(monkeypatch)

    def boom():
        """先产出两段再抛错。"""

        yield SimpleNamespace(text="one", start=0.0, end=1.0)
        yield SimpleNamespace(text="two", start=1.0, end=2.0)
        raise _StopError("boom")

    _install_fake_faster_whisper(monkeypatch, boom)
    wav = _audio(tmp_path)
    out = tmp_path / "demo.ja.srt"
    out.write_text("old-content", encoding="utf-8")

    with pytest.raises(_StopError):
        run_stt(wav, out, language="ja", timeout_sec=60, model="/m", device="cpu")

    # 既有产物保持原样（不再 preclean 正式目标），也不得出现任何半成品替换。
    assert out.read_text(encoding="utf-8") == "old-content"
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".stt-")]
    assert leftovers == []


def test_run_stt_cancel_check_aborts_without_publishing(tmp_path: Path, monkeypatch) -> None:
    """A.1/C.7：取消回调返回 True 时抛 ExecutorCancelledError 且不发布产物。"""

    _quiet_progress(monkeypatch)
    _install_fake_faster_whisper(monkeypatch, _segs(*[(f"s{i}", i, i + 1) for i in range(10)]))
    wav = _audio(tmp_path)
    out = tmp_path / "demo.ja.srt"
    checks = {"count": 0}

    def cancel_after_first() -> bool:
        """第一次检查放行，第二次起要求取消。"""

        checks["count"] += 1
        return checks["count"] > 1

    with pytest.raises(ExecutorCancelledError):
        run_stt(
            wav,
            out,
            language="ja",
            timeout_sec=60,
            model="/m",
            device="cpu",
            progress_every=1,
            cancel_check=cancel_after_first,
        )

    assert checks["count"] >= 2
    assert not out.exists()
    assert [p for p in tmp_path.iterdir() if p.name.startswith(".stt-")] == []


def test_run_stt_cancel_check_failure_is_treated_as_cancel(tmp_path: Path, monkeypatch) -> None:
    """取消回调自身抛错（如租约查询失败）按“已取消”保守处理。"""

    _quiet_progress(monkeypatch)
    _install_fake_faster_whisper(monkeypatch, _segs(("a", 0.0, 1.0)))
    wav = _audio(tmp_path)

    def broken_check() -> bool:
        """模拟租约续期查询异常。"""

        raise RuntimeError("db down")

    with pytest.raises(ExecutorCancelledError):
        run_stt(wav, tmp_path / "x.ja.srt", language="ja", timeout_sec=60, model="/m", device="cpu", cancel_check=broken_check)


# ---------------------------------------------------------------------------
# WhisperX：wav 契约 + 缓存 + 原子输出
# ---------------------------------------------------------------------------
def _install_fake_whisperx(monkeypatch) -> dict[str, list]:
    """安装 fake whisperx 模块（load_audio / load_model / align）。"""

    calls: dict[str, list] = {"load_model": [], "load_audio": [], "align": []}

    class FakeAsr:
        """假 WhisperX ASR。"""

        def transcribe(self, audio, **kwargs):
            """返回两段结果。"""

            return {
                "segments": [
                    {"text": "ありがとう", "start": 0.0, "end": 1.0},
                    {"text": "ございます", "start": 1.0, "end": 2.0},
                ]
            }

    def load_model(model, **kwargs):
        """记录加载参数并返回 ASR。"""

        calls["load_model"].append({"model": model, **kwargs})
        return FakeAsr()

    def load_audio(path):
        """记录传入的音频路径。"""

        calls["load_audio"].append(path)
        return [0.0]

    def align(segments, model, meta, audio, device, **kwargs):
        """原样返回段列表。"""

        calls["align"].append({"segments": segments, "device": device})
        return {"segments": segments}

    # pyannote VAD 走 "whisperx.vads.pyannote" 子模块导入，这里一并注入伪子模块，
    # 让缓存加载器（service.model_cache）能完整跑通而不依赖真实 whisperx。
    fake_pyannote = SimpleNamespace(
        Pyannote=lambda **kwargs: SimpleNamespace(kind="pyannote", **kwargs)
    )
    monkeypatch.setitem(
        sys.modules,
        "whisperx",
        SimpleNamespace(
            load_model=load_model,
            load_audio=load_audio,
            align=align,
            load_align_model=lambda **kwargs: (object(), {"language": "ja"}),
            vads=SimpleNamespace(pyannote=fake_pyannote),
        ),
    )
    monkeypatch.setitem(sys.modules, "whisperx.vads", SimpleNamespace(pyannote=fake_pyannote))
    monkeypatch.setitem(sys.modules, "whisperx.vads.pyannote", fake_pyannote)
    return calls


def _whisperx_kwargs(tmp_path: Path, align_enabled: bool = False) -> dict:
    """构造 run_stt_whisperx 的必填关键字参数。"""

    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    vad_dir = tmp_path / "vad"
    vad_dir.mkdir(exist_ok=True)
    (vad_dir / "pytorch_model.bin").write_bytes(b"vad")
    (vad_dir / "config.yaml").write_text(
        "pipeline:\n  params:\n    segmentation: pytorch_model.bin\n",
        encoding="utf-8",
    )
    align_root = tmp_path / "align"
    (align_root / "ja").mkdir(parents=True, exist_ok=True)
    return {
        "model": str(model_dir),
        "device": "cpu",
        "compute_type": "int8",
        "batch_size": 8,
        "vad_config_path": str(vad_dir / "config.yaml"),
        "align_model_root": str(align_root),
        "align_enabled": align_enabled,
        "vad_backend": "pyannote",
        "vad_onset": 0.35,
        "vad_offset": 0.2,
        "local_files_only": True,
    }


def test_run_stt_whisperx_rejects_missing_wav(tmp_path: Path, monkeypatch) -> None:
    """E：WhisperX 同样拒绝缺失 wav，不回退视频。"""

    _quiet_progress(monkeypatch)
    calls = _install_fake_whisperx(monkeypatch)
    kwargs = _whisperx_kwargs(tmp_path)
    missing = tmp_path / "missing.16k.wav"

    with pytest.raises(MissingAudioArtifactError):
        run_stt_whisperx(missing, tmp_path / "o.ja.srt", language="ja", timeout_sec=60, require_audio_artifact=True, **kwargs)
    assert calls["load_model"] == []


def test_run_stt_whisperx_reads_wav_and_caches_bundle(tmp_path: Path, monkeypatch) -> None:
    """E/D：WhisperX 用 wav 做 load_audio，且第二个 task 命中缓存。"""

    _quiet_progress(monkeypatch)
    calls = _install_fake_whisperx(monkeypatch)
    kwargs = _whisperx_kwargs(tmp_path)
    wav = _audio(tmp_path)

    first = run_stt_whisperx(wav, tmp_path / "a.ja.srt", language="ja", timeout_sec=60, require_audio_artifact=True, **kwargs)
    second = run_stt_whisperx(wav, tmp_path / "b.ja.srt", language="ja", timeout_sec=60, require_audio_artifact=True, **kwargs)

    assert calls["load_audio"] == [str(wav), str(wav)]
    assert len(calls["load_model"]) == 1
    assert first["model_cache_hit"] is False
    assert second["model_cache_hit"] is True
    content = (tmp_path / "a.ja.srt").read_text(encoding="utf-8")
    assert content == (
        "1\n00:00:00,000 --> 00:00:01,000\nありがとう\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nございます\n\n"
    )


def test_run_stt_whisperx_holds_slot_around_transcribe_and_align(
    tmp_path: Path, monkeypatch
) -> None:
    """WhisperX 的 transcribe/align 是同步调用，许可必须覆盖二者且用完即还。"""

    from whisper_stt_service.service.model_cache import (
        get_model_cache,
        reset_model_cache_for_tests,
    )

    reset_model_cache_for_tests()
    _quiet_progress(monkeypatch)
    observed: dict[str, int] = {}
    calls = _install_fake_whisperx(monkeypatch)

    def record_stage(name: str) -> None:
        """记录调用时刻的信号量剩余许可数（0 表示正处于许可内）。

        加载阶段此时条目尚未入表（single-flight 中），记为 None 表示“许可尚未进入”。
        """

        cache = get_model_cache()
        entries = list(cache._entries.values())
        observed[name] = entries[0].signal._value if entries else -1

    class SlottedAsr:
        """在 transcribe 内部观测许可状态。"""

        def transcribe(self, audio, **kwargs):
            """记录许可状态并返回段列表。"""

            record_stage("transcribe")
            return {"segments": [{"text": "a", "start": 0.0, "end": 1.0}]}

    def recording_load_model(model, **kw):
        """记录一次 load_model 并返回“带许可观测”的 ASR。"""

        record_stage("load_model")
        return SlottedAsr()

    sys.modules["whisperx"].load_model = recording_load_model
    original_align = sys.modules["whisperx"].align

    def aligned(segments, model, meta, audio, device, **kwargs):
        """记录 align 同样处于许可内。"""

        record_stage("align")
        return original_align(segments, model, meta, audio, device, **kwargs)

    sys.modules["whisperx"].align = aligned
    kwargs = _whisperx_kwargs(tmp_path, align_enabled=True)
    wav = _audio(tmp_path, "wx.16k.wav")

    run_stt_whisperx(
        wav,
        tmp_path / "wx.ja.srt",
        language="ja",
        timeout_sec=60,
        require_audio_artifact=True,
        **kwargs,
    )

    # transcribe 与 align 都必须发生在持有许可期间（parallelism=1 → 剩余 0）；
    # load_model 属于缓存加载阶段（条目尚未入表，记为 -1），本就不在许可内。
    assert observed["load_model"] == -1
    assert observed["transcribe"] == 0
    assert observed["align"] == 0
    # 结束后许可已归还（未持有者再取一次能得到）。
    cache = get_model_cache()
    entry = next(iter(cache._entries.values()))
    assert entry.signal.acquire(blocking=False) is True
    entry.signal.release()
    cache.clear(force=True)


def test_run_stt_whisperx_cancel_skips_publish(tmp_path: Path, monkeypatch) -> None:
    """A.1/C.7：WhisperX 取消后不发布字幕。"""

    _quiet_progress(monkeypatch)
    _install_fake_whisperx(monkeypatch)
    kwargs = _whisperx_kwargs(tmp_path)
    wav = _audio(tmp_path)
    out = tmp_path / "o.ja.srt"

    with pytest.raises(ExecutorCancelledError):
        run_stt_whisperx(
            wav,
            out,
            language="ja",
            timeout_sec=60,
            require_audio_artifact=True,
            cancel_check=lambda: True,
            **kwargs,
        )
    assert not out.exists()


def test_run_stt_whisperx_align_failure_keeps_no_output(tmp_path: Path, monkeypatch) -> None:
    """WhisperX 对齐异常（非 sampling_rate）应向上抛出且不写半成品。"""

    _quiet_progress(monkeypatch)
    calls = _install_fake_whisperx(monkeypatch)

    def align_boom(*args, **kwargs):
        """模拟对齐实现异常。"""

        raise ValueError("unexpected align failure")

    sys.modules["whisperx"].align = align_boom
    kwargs = _whisperx_kwargs(tmp_path, align_enabled=True)
    wav = _audio(tmp_path)
    out = tmp_path / "o.ja.srt"

    with pytest.raises(ValueError):
        run_stt_whisperx(wav, out, language="ja", timeout_sec=60, require_audio_artifact=True, **kwargs)
    assert not out.exists()
    assert calls["load_model"]


# ---------------------------------------------------------------------------
# extract：临时 wav + 原子发布 + 取消
# ---------------------------------------------------------------------------
class _FakeProc:
    """假 ffmpeg 进程：按脚本产出 progress 行。"""

    def __init__(self, output: Path, lines: list[str], returncode: int = 0) -> None:
        """保存输出路径与进度行。"""

        self.output = output
        self.lines = lines
        self.returncode = returncode
        self.stdout = iter(lines)
        self.killed = False

    def wait(self, timeout=None):
        """返回码。"""

        return self.returncode

    def poll(self):
        """未完成时返回 None。"""

        return None

    def kill(self) -> None:
        """记录被 kill。"""

        self.killed = True


def _patch_ffmpeg(monkeypatch, tmp_path: Path, lines: list[str], returncode: int = 0, writer: bool = True) -> dict:
    """替换 subprocess.Popen 与 _probe_duration，模拟 ffmpeg 行为。"""

    state: dict = {"procs": []}

    def fake_popen(cmd, **kwargs):
        """按命令中的输出路径写入假 wav 并返回假进程。"""

        out_path = Path(cmd[-1])
        state["output"] = out_path
        if writer:
            out_path.write_bytes(b"RIFF" * 100)
        proc = _FakeProc(out_path, lines, returncode=returncode)
        state["procs"].append(proc)
        return proc

    monkeypatch.setattr("whisper_stt_service.executor.extract.subprocess.Popen", fake_popen)
    monkeypatch.setattr("whisper_stt_service.executor.extract._probe_duration", lambda _path: 60.0)
    monkeypatch.setattr(
        "whisper_stt_service.executor.extract._emit_progress",
        lambda *args, **kwargs: None,
    )
    return state


def test_extract_ffmpeg_command_declares_wav_container(tmp_path: Path, monkeypatch) -> None:
    """真实 bug 回归：临时文件扩展名不足以让 ffmpeg 推断容器，命令必须显式 -f wav。"""

    captured: dict = {"cmd": None}

    def fake_popen(cmd, **kwargs):
        """记录真实构造出的 ffmpeg 命令并返回假进程。"""

        captured["cmd"] = list(cmd)
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"RIFF" * 10)
        return _FakeProc(out_path, ["progress=end\n"])

    monkeypatch.setattr("whisper_stt_service.executor.extract.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "whisper_stt_service.executor.extract._probe_duration", lambda _path: 60.0
    )
    monkeypatch.setattr(
        "whisper_stt_service.executor.extract._emit_progress",
        lambda *args, **kwargs: None,
    )
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"v")
    out = tmp_path / "audio" / "demo.16k.wav"
    run_extract(video, out, 60)

    cmd = captured["cmd"]
    assert "-f" in cmd
    assert cmd[cmd.index("-f") + 1] == "wav"
    # 临时文件名本身也保持 .wav 结尾，双重保证容器推断。
    assert cmd[-1].endswith(".16k.wav")
    assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == "16000"
    assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "1"


def test_run_extract_writes_temp_then_replaces(tmp_path: Path, monkeypatch) -> None:
    """E：ffmpeg 先写临时文件，成功后才替换为 audio/<stem>.16k.wav。"""

    state = _patch_ffmpeg(monkeypatch, tmp_path, ["out_time_ms=5000000\n", "progress=end\n"])
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"v")
    out = tmp_path / "audio" / "demo.16k.wav"

    run_extract(video, out, 60)

    # 既有产物在成功时被新结果原子替换（失败/取消时不删除，见下一个用例）。
    assert out.is_file()
    assert state["output"] != out
    # 临时文件以 .wav 结尾（ffmpeg 依扩展名推断容器），且与正式产物不同名。
    assert str(state["output"]).endswith(".16k.wav")
    assert state["output"].name.startswith(".extract-")
    assert not state["output"].exists()


def test_run_extract_failure_leaves_no_temp(tmp_path: Path, monkeypatch) -> None:
    """E：ffmpeg 非 0 退出时删除临时 wav，不产生正式产物。"""

    state = _patch_ffmpeg(monkeypatch, tmp_path, ["progress=end\n"], returncode=1)
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"v")
    out = tmp_path / "audio" / "demo.16k.wav"

    with pytest.raises(RuntimeError, match="ffmpeg extraction failed"):
        run_extract(video, out, 60)
    assert not out.exists()
    assert not Path(state["output"]).exists()


def test_run_extract_cancel_kills_ffmpeg_and_cleans_temp(tmp_path: Path, monkeypatch) -> None:
    """A.1：取消回调触发时终止 ffmpeg 并删除临时 wav。"""

    state = _patch_ffmpeg(monkeypatch, tmp_path, ["out_time_ms=1000000\n"] * 5)
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"v")
    out = tmp_path / "audio" / "demo.16k.wav"
    checks = {"count": 0}

    def cancel_second() -> bool:
        """第二次检查开始取消。"""

        checks["count"] += 1
        return checks["count"] > 1

    with pytest.raises(ExecutorCancelledError):
        run_extract(video, out, 60, cancel_check=cancel_second)

    assert state["procs"][0].killed is True
    assert not out.exists()
    assert not Path(state["output"]).exists()
