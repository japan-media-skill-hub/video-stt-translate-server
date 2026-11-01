"""stt_qwen 阶段单元测试（SPEC §9.1，全程不加载真实权重）。

覆盖三块：

1. **接线**：白名单 / DAG 唯一 ASR / io 契约 / 配置默认值与非法参数；
2. **执行器行为**：用注入的假模型跑通"VAD → 定点切片 → 批处理 → 清洗 → 原子写"，
   并逐项验证 SPEC §6.2 的兜底与计数（空对齐、空文本、结果缺失、增量落盘）；
3. **fail-fast**：路径缺失 / CUDA 不可用 / wav 产物缺失都必须当场报错，不静默降级。
"""

from __future__ import annotations

from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from typing import Any
import wave

import numpy as np
import pytest

from whisper_stt_service.core.config import SttQwenSettings
from whisper_stt_service.core.dag import (
    ASR_STAGES,
    normalize_and_validate_dag,
    normalize_and_validate_job_config,
)
from whisper_stt_service.core.stages import STAGE_CONFIG_KEYS, SUPPORTED_STAGES
from whisper_stt_service.executor.common import MissingAudioArtifactError
from whisper_stt_service.executor.qwen_boundary import QwenBoundaryConfigError
from whisper_stt_service.executor.stt_qwen import (
    build_stt_qwen_effective_config,
    resolve_qwen_language,
    run_stt_qwen,
)
from whisper_stt_service.service.model_cache import ModelCache


# ---------------------------------------------------------------- 合成素材


def write_wav(path: Path, seconds: float, *, sample_rate: int = 16000) -> Path:
    """写一个 16k mono pcm_s16le wav（内容是与切片无关的低幅噪声）。"""

    samples = (np.sin(np.arange(int(seconds * sample_rate)) / 50.0) * 3000).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return path


def _word(text: str, start: float, end: float) -> SimpleNamespace:
    """构造一个字级对齐结果（字段名与 qwen_asr 的 time_stamps 元素一致）。"""

    return SimpleNamespace(text=text, start_time=start, end_time=end)


class _FakeVadPostprocessor:
    """假后处理：`process` 直接透传，`decision_to_segment` 返回预设语音段。"""

    def __init__(self, segments: list[tuple[float, float]]) -> None:
        self._segments = segments
        self.process_calls = 0

    def process(self, probs: list[float]) -> list[float]:
        """记录被调用过（阶段必须真的跑了 VAD 后处理）。"""

        self.process_calls += 1
        return list(probs)

    def decision_to_segment(self, decisions, dur):  # noqa: ANN001 - 与真实签名一致
        """返回预设语音段（忽略 decisions）。"""

        return [(float(a), float(b)) for a, b in self._segments]


class FakeVad:
    """假 FireRedVAD：只实现执行器用到的那三个成员。"""

    def __init__(self, probs: np.ndarray, segments: list[tuple[float, float]]) -> None:
        import torch

        self._probs = torch.as_tensor(probs, dtype=torch.float32)
        self.vad_postprocessor = _FakeVadPostprocessor(segments)
        self.detect_calls = 0

    def detect(self, audio, do_postprocess: bool = True):  # noqa: ANN001
        """只支持 `do_postprocess=False`（执行器就是按这个语义调用的）。"""

        assert do_postprocess is False, "执行器必须自己拿原始 probs 再做后处理"
        self.detect_calls += 1
        return None, self._probs


class FakeAsr:
    """假 Qwen3-ASR：可编程返回空文本 / 缺时间戳 / 少结果 / 抛异常。"""

    def __init__(
        self,
        *,
        mode: str = "ok",
        words_per_chunk: int = 2,
        long_stamp: bool = False,
        fail_on_call: int | None = None,
        custom_words: list[tuple[str, float, float]] | None = None,
    ) -> None:
        self.mode = mode
        self.words_per_chunk = words_per_chunk
        self.long_stamp = long_stamp
        self.fail_on_call = fail_on_call
        # 自定义字级时间戳（乱序/零长/重叠用例）：非空时忽略 words_per_chunk。
        self.custom_words = custom_words
        self.calls: list[list[Any]] = []
        self.languages: list[Any] = []

    def named_parameters(self):
        """参数落点查询接口（执行器用它校验模型真的在目标设备上）。"""

        return iter([("weight", SimpleNamespace(device="cpu"))])

    def transcribe(self, *, audio, language=None, return_time_stamps: bool = False):
        """按 mode 产出结果；记录每次调用收到的片数与语言。"""

        self.calls.append(list(audio))
        self.languages.append(language)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("fake_asr_failed")
        if self.mode == "missing":
            # 只回一半结果：模拟上游少返回。
            return [self._one(a) for a in list(audio)[: max(len(audio) // 2, 0)]]
        return [self._one(a) for a in audio]

    def _one(self, item) -> SimpleNamespace:  # noqa: ANN001
        """构造单片结果。"""

        samples, sample_rate = item
        length = float(len(samples)) / float(sample_rate)
        if self.mode == "empty":
            return SimpleNamespace(text="", time_stamps=None)
        if self.mode == "no_ts":
            return SimpleNamespace(text="ああああ", time_stamps=None)
        if self.mode == "none_ts":
            return SimpleNamespace(text="ああああ", time_stamps=[])
        if self.custom_words is not None:
            return SimpleNamespace(
                text="".join(text for text, _s, _e in self.custom_words),
                time_stamps=[_word(t, a, b) for t, a, b in self.custom_words],
            )
        stamps = []
        step = length / max(self.words_per_chunk, 1)
        for index in range(self.words_per_chunk):
            start = index * step
            end = min((index + 1) * step, length)
            # 句末标点 + 4 字：聚合函数会在这一字处立即断行，行数可精确预期。
            stamps.append(_word("テスト。", start, length * 100 if self.long_stamp else end))
        return SimpleNamespace(text="テスト。" * self.words_per_chunk, time_stamps=stamps)


def _install_fake_models(monkeypatch, vad: FakeVad, asr: FakeAsr) -> None:
    """把执行器拿到的模型缓存换成注入了假模型的本地缓存。"""

    cache = ModelCache(
        cache_size=2,
        parallelism=1,
        qwen_asr_bundle_loader=lambda **kwargs: (asr, vad),
    )
    monkeypatch.setattr(
        "whisper_stt_service.executor.stt_qwen.get_model_cache", lambda: cache
    )


def _prepare(tmp_path: Path, *, seconds: float = 100.0):
    """建好三个模型目录 + 输入 wav + 输出路径。"""

    for name in ("asr", "aligner", "vad"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    wav = write_wav(tmp_path / "demo.16k.wav", seconds)
    return wav, tmp_path / "asr" / "demo.ja.srt"


def _probs(seconds: float, threshold: float = 0.4) -> np.ndarray:
    """构造 100Hz 帧概率：全部高于阈值（切片只看语音段，占比只用于观测）。"""

    return np.full(int(seconds / 0.01), min(threshold + 0.5, 0.99), dtype=np.float32)


# ---------------------------------------------------------------- 1. 接线


def test_default_dag_is_unchanged() -> None:
    """§13.1：默认 DAG 必须仍是 extract→stt→translate（新增并列阶段不得改默认行为）。"""

    from whisper_stt_service.core.dag import build_default_dag

    default = build_default_dag()
    assert [item["stage"] for item in default["stages"]] == [
        "extract",
        "stt",
        "translate",
    ]


def test_missing_stt_qwen_section_falls_back_to_defaults(tmp_path: Path) -> None:
    """§9.1：config 里没有 [stt_qwen] 段时必须走默认档，不得抛异常。"""

    import configparser

    from whisper_stt_service.core.config import _load_stt_qwen_settings

    parser = configparser.ConfigParser()
    parser.read_string("[stt]\ndevice = cpu\n")
    settings = _load_stt_qwen_settings(parser)
    assert settings == SttQwenSettings()
    assert settings.model_dir.endswith("Qwen3-ASR-1.7B")


def test_stage_registry_and_asr_uniqueness() -> None:
    """stt_qwen 必须是受支持阶段、唯一 ASR 之一，且白名单只含可安全覆盖的键。"""

    assert "stt_qwen" in SUPPORTED_STAGES
    assert "stt_qwen" in ASR_STAGES
    allowed = set(STAGE_CONFIG_KEYS["stt_qwen"])
    # 切片参数可按 job 覆盖
    for key in (
        "nominal_seconds",
        "delta_back_seconds",
        "delta_forward_seconds",
        "fallback_forward_seconds",
        "min_chunk_seconds",
        "batch_size",
        "max_segment_sec",
    ):
        assert key in allowed
    # 部署事实（权重路径/设备/精度）绝不允许按 job 覆盖
    for key in ("model_dir", "aligner_dir", "vad_model_dir", "device", "dtype"):
        assert key not in allowed


def test_job_config_whitelist_rejects_path_keys() -> None:
    """job_config 里塞路径类键必须被拒（`job_config_key_not_supported`）。"""

    dag = normalize_and_validate_dag(
        {
            "version": 1,
            "stages": [
                {"stage": "extract", "depends_on": []},
                {"stage": "stt_qwen", "depends_on": ["extract"]},
                {"stage": "translate", "depends_on": ["stt_qwen"]},
            ],
        }
    )
    ok = normalize_and_validate_job_config(
        {"stt_qwen": {"nominal_seconds": 50, "hard_cut_on_miss": False}}, dag
    )
    assert ok["stt_qwen"] == {"nominal_seconds": 50, "hard_cut_on_miss": False}

    for bad in (
        {"model_dir": "/tmp/evil"},
        {"device": "cpu"},
        {"dtype": "float32"},
    ):
        with pytest.raises(ValueError) as excinfo:
            normalize_and_validate_job_config({"stt_qwen": bad}, dag)
        assert "job_config_key_not_supported" in str(excinfo.value)


def test_job_config_type_and_range_validation() -> None:
    """白名单内的键仍要过类型/范围校验（白名单与校验双处登记）。"""

    dag = normalize_and_validate_dag(
        {
            "version": 1,
            "stages": [
                {"stage": "extract", "depends_on": []},
                {"stage": "stt_qwen", "depends_on": ["extract"]},
                {"stage": "translate", "depends_on": ["stt_qwen"]},
            ],
        }
    )
    with pytest.raises(ValueError) as excinfo:
        normalize_and_validate_job_config({"stt_qwen": {"batch_size": 0}}, dag)
    assert "job_config_value_out_of_range" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        normalize_and_validate_job_config({"stt_qwen": {"hard_cut_on_miss": 1}}, dag)
    assert "job_config_value_type_invalid" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        normalize_and_validate_job_config({"stt_qwen": {"min_coverage": 1.5}}, dag)
    assert "job_config_value_out_of_range" in str(excinfo.value)


def test_stage_io_contract_uses_wav_not_video() -> None:
    """ASR io 契约：stt_qwen 必须吃 wav、产 ja.srt（漏登记会静默吃原视频）。"""

    from whisper_stt_service.repo.database import Database
    from whisper_stt_service.repo.job_repository import JobRepository

    repo = JobRepository(Database(Path(":memory:")))
    io_cfg = repo._build_stage_io(
        "stt_qwen",
        video_path="/v/movie.mp4",
        paths={
            "wav_path": "/a/movie.16k.wav",
            "ja_srt": "/a/movie.ja.srt",
            "zh_srt": "/a/movie.zh.srt",
            "progress_path": "/a/movie.progress.json",
        },
        copy_back_dir="/v",
    )
    assert io_cfg == {"input_path": "/a/movie.16k.wav", "output_path": "/a/movie.ja.srt"}


def test_default_config_is_landed_profile() -> None:
    """配置默认值必须是本轮落地档（30/±20/+40 → 上界 70s，batch 10）。"""

    settings = SttQwenSettings()
    assert settings.nominal_seconds == 30.0
    assert settings.delta_back_seconds == 20.0
    assert settings.delta_forward_seconds == 20.0
    assert settings.fallback_forward_seconds == 40.0
    assert settings.min_chunk_seconds == 10.0
    assert settings.batch_size == 10
    job_cfg = settings.as_job_config()
    # 路径/设备不进 job_config（避免被下游当成可覆盖项）
    for key in ("model_dir", "aligner_dir", "vad_model_dir"):
        assert key not in job_cfg


def test_language_resolution() -> None:
    """服务传 ISO 码，包内只认规范名：必须显式映射，映射不到即报错。"""

    assert resolve_qwen_language("ja") == "Japanese"
    assert resolve_qwen_language("JA") == "Japanese"
    assert resolve_qwen_language("Japanese") == "Japanese"
    assert resolve_qwen_language("zh") == "Chinese"
    with pytest.raises(QwenBoundaryConfigError):
        resolve_qwen_language("xx")
    with pytest.raises(QwenBoundaryConfigError):
        resolve_qwen_language("")


def test_effective_config_snapshot_shape() -> None:
    """快照必须携带两套参数的区分点（上界 = nominal + fallback_forward）。"""

    from whisper_stt_service.executor.qwen_boundary import BoundaryParams

    snapshot = build_stt_qwen_effective_config(
        language="Japanese",
        timeout_sec=5400,
        max_retries=2,
        model_dir="/m",
        aligner_dir="/a",
        vad_model_dir="/v",
        device="cuda:0",
        dtype="bfloat16",
        batch_size=10,
        params=BoundaryParams(),
        tokens_per_sec=16.0,
        max_segment_sec=30.0,
        vad_use_gpu=True,
        vad_speech_threshold=0.4,
        min_free_vram_gb=12.0,
        min_coverage=0.98,
        strict_output=False,
        align_missing_fallback="chunk_boundary",
    )
    assert snapshot["max_chunk_seconds"] == 70.0
    assert snapshot["absolute_max_chunk_seconds"] == 180.0
    assert snapshot["batch_size"] == 10


# ---------------------------------------------------------------- 2. 执行器行为


def test_run_produces_srt_and_counters(tmp_path: Path, monkeypatch) -> None:
    """主链路：定点切片 → 批处理 → 清洗 → 原子写，并产出全部计数。"""

    wav, out = _prepare(tmp_path)
    # 不屏蔽 _emit_progress：进度事件本身也是契约的一部分（开始/完成必须可见）。
    progress: Queue = Queue()
    # 每秒一个语音段结束点：切点应落在 30/60/90（命中正常窗，取最接近名义目标者）。
    vad = FakeVad(_probs(100.0), [(i, i + 1.0) for i in range(100)])
    asr = FakeAsr(words_per_chunk=2)
    _install_fake_models(monkeypatch, vad, asr)

    result = run_stt_qwen(
        input_path=wav,
        output_path=out,
        language="ja",
        model_dir=str(tmp_path / "asr"),
        aligner_dir=str(tmp_path / "aligner"),
        vad_model_dir=str(tmp_path / "vad"),
        device="cpu",
        nominal_seconds=30.0,
        delta_back_seconds=20.0,
        delta_forward_seconds=20.0,
        fallback_forward_seconds=40.0,
        min_chunk_seconds=10.0,
        batch_size=2,
        require_audio_artifact=True,
        progress_queue=progress,
        task_id="t1",
    )

    assert out.is_file()
    body = out.read_text(encoding="utf-8")
    # 4 片（30/30/30/10），每片 2 行 → 8 行
    assert result["chunks_total"] == 4
    assert result["batches_total"] == 2
    assert result["cut_policy_counts"] == {
        "vad_offset": 3,
        "vad_offset_extended": 0,
        "hard_cut": 0,
    }
    assert result["chunk_seconds_max"] <= 70.0
    assert result["chunk_seconds_min"] >= 10.0
    assert result["lines_total"] == 8
    assert result["align_missing_chunks"] == 0
    assert result["empty_chunks"] == 0
    # CPU 路径不产显存数字（None），但键必须存在，便于服务端快照统一消费。
    assert result["peak_vram_gb"] is None
    # 覆盖到片尾（末片结束 = 音频时长）
    assert result["coverage"] == pytest.approx(1.0, abs=1e-6)
    assert "テスト。" in body
    # 进度至少上报过开始与完成
    events: list[dict] = []
    while not progress.empty():
        events.append(progress.get_nowait())
    assert events[0]["message"] == "stt_qwen_started"
    assert events[-1]["message"] == "stt_qwen_done"
    assert events[-1]["percent"] == 100.0


def test_cleaning_bounds_long_segments(tmp_path: Path, monkeypatch) -> None:
    """对齐结果给出 >30s 的单行时，写盘前必须被清洗到 <= max_segment_sec。

    构造 61s 的长片（片长上界 70s 之内）+ 一个跨 45s 的字级结果：逃逸时间戳被
    片长夹住后仍然 >30s，只能由 `clean_segment_timeline`（与 stt/stt_whisperx
    同一函数）收紧——这条是清洗器在 stt_qwen 里的**唯一兜底职责**。
    """

    wav, out = _prepare(tmp_path, seconds=120.0)
    # 只有 61s 处有语音边界：正常窗（10~50）落空 → 扩窗（10~70）命中 61 → 首片 61s
    vad = FakeVad(_probs(120.0), [(60.0, 61.0)])
    asr = FakeAsr(custom_words=[("あいうえお", 0.0, 45.0)])
    _install_fake_models(monkeypatch, vad, asr)

    result = run_stt_qwen(
        input_path=wav,
        output_path=out,
        model_dir=str(tmp_path / "asr"),
        aligner_dir=str(tmp_path / "aligner"),
        vad_model_dir=str(tmp_path / "vad"),
        device="cpu",
        nominal_seconds=30.0,
        delta_back_seconds=20.0,
        delta_forward_seconds=20.0,
        fallback_forward_seconds=40.0,
        min_chunk_seconds=10.0,
        batch_size=10,
        max_segment_sec=30.0,
    )
    assert result["chunk_seconds_max"] > 30.0, result["chunk_seconds_max"]
    stats = result["timestamp_cleanup"]
    assert stats["adjusted"] > 0, stats
    assert stats["max_original_sec"] > 30.0, stats
    assert stats["max_output_sec"] <= 30.0 + 1e-6, stats
    for start_sec, end_sec, _text in _parse_srt(out):
        assert end_sec - start_sec <= 30.0 + 1e-6


def test_punctuation_only_lines_are_dropped_and_counted(tmp_path: Path, monkeypatch) -> None:
    """纯标点行必须丢弃并计数（2026-11-01 真片验收发现的噪声行）。

    实测现象：静音片被模型输出成「。」，对齐器又给它一整片时长，于是产物里出现
    "30 秒只有一个句号"的字幕（2049 行里 6 行，SAN 4 / CAWB 2）。这种行没有信息量，
    但不能静默丢——必须计进 `punct_only_lines` 并留 WARNING。
    """

    wav, out = _prepare(tmp_path, seconds=40.0)
    vad = FakeVad(_probs(40.0), [(i, i + 1.0) for i in range(40)])
    # 一个「。」占满整片 30s + 一个正常行
    asr = FakeAsr(custom_words=[("。", 0.0, 30.0), ("こんにちは。", 0.0, 10.0)])
    _install_fake_models(monkeypatch, vad, asr)

    result = run_stt_qwen(
        input_path=wav,
        output_path=out,
        model_dir=str(tmp_path / "asr"),
        aligner_dir=str(tmp_path / "aligner"),
        vad_model_dir=str(tmp_path / "vad"),
        device="cpu",
        batch_size=10,
    )
    # 每片都会产出 1 个噪声行 + 1 个正常行 → 2 片共 2 条噪声、2 条有效
    assert result["punct_only_lines"] == 2, result["punct_only_lines"]
    entries = _parse_srt(out)
    assert entries
    assert all(text.strip("。！？!?…") for _s, _e, text in entries), entries
    # 有效行一条不少（"こんにちは。" 两片各一行）
    assert sum(1 for _s, _e, t in entries if t == "こんにちは。") == 2


def test_punctuation_only_fallback_text_is_also_dropped(tmp_path: Path, monkeypatch) -> None:
    """无时间戳兜底路径同样要挡纯标点（否则回退成"整片时长的句号"）。"""

    wav, out = _prepare(tmp_path, seconds=40.0)
    vad = FakeVad(_probs(40.0), [(i, i + 1.0) for i in range(40)])

    class _PunctAsr:
        """只回纯标点、且不带时间戳的假模型（覆盖 align_missing 兜底分支）。"""

        def named_parameters(self):
            return iter([("weight", SimpleNamespace(device="cpu"))])

        def transcribe(self, *, audio, language=None, return_time_stamps=False):
            return [SimpleNamespace(text="。", time_stamps=None) for _ in audio]

    _install_fake_models(monkeypatch, vad, _PunctAsr())

    with pytest.raises(RuntimeError) as excinfo:
        run_stt_qwen(
            input_path=wav,
            output_path=out,
            model_dir=str(tmp_path / "asr"),
            aligner_dir=str(tmp_path / "aligner"),
            vad_model_dir=str(tmp_path / "vad"),
            device="cpu",
            batch_size=10,
        )
    # 全是噪声行 → 没有有效字幕行 → 明确失败，绝不输出满屏句号
    assert "stt_qwen_empty_srt" in str(excinfo.value)
    assert not out.exists()


def test_align_missing_falls_back_to_chunk_bounds(tmp_path: Path, monkeypatch) -> None:
    """`time_stamps` 缺失：退回片边界出单行并计数，绝不丢内容、绝不崩。"""

    wav, out = _prepare(tmp_path, seconds=40.0)
    vad = FakeVad(_probs(40.0), [(i, i + 1.0) for i in range(40)])
    asr = FakeAsr(mode="no_ts")
    _install_fake_models(monkeypatch, vad, asr)

    result = run_stt_qwen(
        input_path=wav,
        output_path=out,
        model_dir=str(tmp_path / "asr"),
        aligner_dir=str(tmp_path / "aligner"),
        vad_model_dir=str(tmp_path / "vad"),
        device="cpu",
        nominal_seconds=30.0,
        delta_back_seconds=20.0,
        delta_forward_seconds=20.0,
        fallback_forward_seconds=40.0,
        min_chunk_seconds=10.0,
        batch_size=10,
    )
    assert result["align_missing_chunks"] == result["chunks_total"] >= 1
    assert result["lines_total"] == result["chunks_total"]
    entries = _parse_srt(out)
    # 每一行的时长就是对应片的长度（没有被丢弃、也没有被编造）
    assert [round(e - s, 3) for s, e, _ in entries] == [30.0, 10.0]


def test_empty_stamps_list_also_falls_back(tmp_path: Path, monkeypatch) -> None:
    """`time_stamps=[]`（空列表）与 None 同义：同样兜底并计数。"""

    wav, out = _prepare(tmp_path, seconds=40.0)
    vad = FakeVad(_probs(40.0), [(i, i + 1.0) for i in range(40)])
    asr = FakeAsr(mode="none_ts")
    _install_fake_models(monkeypatch, vad, asr)

    result = run_stt_qwen(
        input_path=wav,
        output_path=out,
        model_dir=str(tmp_path / "asr"),
        aligner_dir=str(tmp_path / "aligner"),
        vad_model_dir=str(tmp_path / "vad"),
        device="cpu",
        batch_size=10,
    )
    assert result["align_missing_chunks"] >= 1


def test_empty_text_chunks_are_counted_not_written(tmp_path: Path, monkeypatch) -> None:
    """空文本片：不产出行，但必须计数（否则就是静默丢内容）。"""

    wav, out = _prepare(tmp_path, seconds=40.0)
    vad = FakeVad(_probs(40.0), [(i, i + 1.0) for i in range(40)])
    asr = FakeAsr(mode="empty")
    _install_fake_models(monkeypatch, vad, asr)

    with pytest.raises(RuntimeError) as excinfo:
        run_stt_qwen(
            input_path=wav,
            output_path=out,
            model_dir=str(tmp_path / "asr"),
            aligner_dir=str(tmp_path / "aligner"),
            vad_model_dir=str(tmp_path / "vad"),
            device="cpu",
            batch_size=10,
        )
    # 全部为空 → 没有任何字幕行 → 明确失败（stt_qwen_empty_srt），不写空文件
    assert "stt_qwen_empty_srt" in str(excinfo.value)
    assert not out.exists()


def test_short_result_is_counted_as_failed(tmp_path: Path, monkeypatch) -> None:
    """结果条数少于请求片数：计数告警，不编造内容。"""

    wav, out = _prepare(tmp_path, seconds=100.0)
    vad = FakeVad(_probs(100.0), [(i, i + 1.0) for i in range(100)])
    asr = FakeAsr(mode="missing")
    _install_fake_models(monkeypatch, vad, asr)

    result = run_stt_qwen(
        input_path=wav,
        output_path=out,
        model_dir=str(tmp_path / "asr"),
        aligner_dir=str(tmp_path / "aligner"),
        vad_model_dir=str(tmp_path / "vad"),
        device="cpu",
        batch_size=4,
    )
    assert result["failed_chunks"] > 0
    assert result["lines_total"] > 0


def test_incremental_flush_keeps_partial_results(tmp_path: Path, monkeypatch) -> None:
    """增量落盘：第 2 批抛异常时，第 1 批的结果必须已经写进 SRT。"""

    wav, out = _prepare(tmp_path, seconds=100.0)
    vad = FakeVad(_probs(100.0), [(i, i + 1.0) for i in range(100)])
    asr = FakeAsr(fail_on_call=2)
    _install_fake_models(monkeypatch, vad, asr)

    with pytest.raises(RuntimeError) as excinfo:
        run_stt_qwen(
            input_path=wav,
            output_path=out,
            model_dir=str(tmp_path / "asr"),
            aligner_dir=str(tmp_path / "aligner"),
            vad_model_dir=str(tmp_path / "vad"),
            device="cpu",
            batch_size=2,
        )
    assert "fake_asr_failed" in str(excinfo.value)
    # 第 1 批（2 片）的内容已经在盘上，算力没有白烧
    assert out.is_file()
    entries = _parse_srt(out)
    assert len(entries) == 4


def test_model_cache_is_reused_across_calls(tmp_path: Path, monkeypatch) -> None:
    """同参数第二次调用必须命中缓存（不重复冷启动 4.4G 权重）。"""

    wav, out = _prepare(tmp_path, seconds=40.0)
    vad = FakeVad(_probs(40.0), [(i, i + 1.0) for i in range(40)])
    asr = FakeAsr()
    _install_fake_models(monkeypatch, vad, asr)
    kwargs = dict(
        input_path=wav,
        output_path=out,
        model_dir=str(tmp_path / "asr"),
        aligner_dir=str(tmp_path / "aligner"),
        vad_model_dir=str(tmp_path / "vad"),
        device="cpu",
        batch_size=10,
    )
    first = run_stt_qwen(**kwargs)
    second = run_stt_qwen(**kwargs)
    assert first["model_cache_hit"] is False
    assert second["model_cache_hit"] is True
    assert second["model_cache_key"] == first["model_cache_key"]


def test_vram_gate_skipped_when_bundle_cached(tmp_path: Path, monkeypatch) -> None:
    """回归：模型已在缓存里时必须跳过显存门槛（2026-11-01 验收实测缺陷）。

    实测现象：第 1 部片成功后模型仍被 model_cache 持有，`torch.cuda.mem_get_info()`
    把缓存分配器持有的显存算成已用 → free≈0 → 第 2/3 部片连续 3 次
    `stt_qwen_insufficient_vram: free=0.00GiB` 失败、整个 job failed。
    判据：冷启动那次按配置值把关，命中缓存那次必须传 0（= 不设门槛）且留痕。
    """

    from whisper_stt_service.executor import stt_qwen as module

    wav, out = _prepare(tmp_path, seconds=40.0)
    vad = FakeVad(_probs(40.0), [(i, i + 1.0) for i in range(40)])
    asr = FakeAsr()
    _install_fake_models(monkeypatch, vad, asr)

    seen: list[float] = []
    real_resolve = module._resolve_device

    def _spy(device: str, *, require_cuda: bool, min_free_vram_gb: float) -> str:
        """记录每次门槛值，再走真实实现（cpu 路径下是直通）。"""

        seen.append(float(min_free_vram_gb))
        return real_resolve(
            device, require_cuda=require_cuda, min_free_vram_gb=min_free_vram_gb
        )

    monkeypatch.setattr(module, "_resolve_device", _spy)

    kwargs = dict(
        input_path=wav,
        output_path=out,
        model_dir=str(tmp_path / "asr"),
        aligner_dir=str(tmp_path / "aligner"),
        vad_model_dir=str(tmp_path / "vad"),
        device="cpu",
        batch_size=10,
        min_free_vram_gb=12.0,
    )
    first = run_stt_qwen(**kwargs)
    second = run_stt_qwen(**kwargs)

    assert seen == [12.0, 0.0], seen
    assert first["vram_gate_skipped"] is False
    assert second["vram_gate_skipped"] is True
    # 缓存命中判定必须与真实 key 一致：换 batch 就是另一个条目，不能算命中。
    from whisper_stt_service.service.model_cache import ModelCache

    bundle_args = dict(
        model_dir=str(tmp_path / "asr"),
        aligner_dir=str(tmp_path / "aligner"),
        vad_model_dir=str(tmp_path / "vad"),
        device="cpu",
        dtype="bfloat16",
        max_new_tokens=1120,
        batch_size=10,
        vad_use_gpu=True,
        vad_speech_threshold=0.4,
    )
    cache = ModelCache(cache_size=2, qwen_asr_bundle_loader=lambda **kw: (asr, vad))
    assert cache.has_qwen_asr_bundle(**bundle_args) is False
    cache.get_qwen_asr_bundle(**bundle_args).release()
    assert cache.has_qwen_asr_bundle(**bundle_args) is True
    assert cache.has_qwen_asr_bundle(**{**bundle_args, "batch_size": 13}) is False
    # 参数不完整时保守当作冷启动（照常走门槛），不得误判为命中
    assert cache.has_qwen_asr_bundle(model_dir=str(tmp_path / "asr")) is False


def test_cancellation_before_batches_aborts_cleanly(tmp_path: Path, monkeypatch) -> None:
    """协作取消：cancel_check 返回 True 时必须中止，不得发布产物。"""

    from whisper_stt_service.executor.common import ExecutorCancelledError

    wav, out = _prepare(tmp_path, seconds=40.0)
    vad = FakeVad(_probs(40.0), [(i, i + 1.0) for i in range(40)])
    asr = FakeAsr()
    _install_fake_models(monkeypatch, vad, asr)

    with pytest.raises(ExecutorCancelledError):
        run_stt_qwen(
            input_path=wav,
            output_path=out,
            model_dir=str(tmp_path / "asr"),
            aligner_dir=str(tmp_path / "aligner"),
            vad_model_dir=str(tmp_path / "vad"),
            device="cpu",
            batch_size=10,
            cancel_check=lambda: True,
        )
    assert not out.exists()


# ---------------------------------------------------------------- 3. fail-fast


def test_invalid_chunk_params_rejected(tmp_path: Path) -> None:
    """片长上界 > 180s：必须当场拒（否则会被包内二次切分，VAD 定点全废）。"""

    wav, out = _prepare(tmp_path, seconds=10.0)
    with pytest.raises(QwenBoundaryConfigError) as excinfo:
        run_stt_qwen(
            input_path=wav,
            output_path=out,
            model_dir=str(tmp_path / "asr"),
            aligner_dir=str(tmp_path / "aligner"),
            vad_model_dir=str(tmp_path / "vad"),
            device="cpu",
            nominal_seconds=180.0,
            fallback_forward_seconds=30.0,
        )
    assert excinfo.value.code == "stt_qwen_config_invalid"


def test_missing_model_dirs_fail_fast(tmp_path: Path) -> None:
    """本地权重目录缺失即失败（运行期禁止联网下载）。"""

    wav, out = _prepare(tmp_path, seconds=10.0)
    with pytest.raises(FileNotFoundError) as excinfo:
        run_stt_qwen(
            input_path=wav,
            output_path=out,
            model_dir=str(tmp_path / "nope"),
            aligner_dir=str(tmp_path / "aligner"),
            vad_model_dir=str(tmp_path / "vad"),
            device="cpu",
        )
    assert "stt_qwen_model_missing" in str(excinfo.value)


def test_missing_audio_artifact_raises(tmp_path: Path) -> None:
    """ASR 输入 wav 缺失：`missing_audio_artifact`，绝不回退原视频。"""

    for name in ("asr", "aligner", "vad"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    with pytest.raises(MissingAudioArtifactError):
        run_stt_qwen(
            input_path=tmp_path / "absent.16k.wav",
            output_path=tmp_path / "x.ja.srt",
            model_dir=str(tmp_path / "asr"),
            aligner_dir=str(tmp_path / "aligner"),
            vad_model_dir=str(tmp_path / "vad"),
            device="cpu",
            require_audio_artifact=True,
        )


def test_cuda_unavailable_fails_when_required(monkeypatch, tmp_path: Path) -> None:
    """cuda 不可用 + require_cuda=true：必须 `stt_qwen_cuda_unavailable`。"""

    import torch

    wav, out = _prepare(tmp_path, seconds=10.0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError) as excinfo:
        run_stt_qwen(
            input_path=wav,
            output_path=out,
            model_dir=str(tmp_path / "asr"),
            aligner_dir=str(tmp_path / "aligner"),
            vad_model_dir=str(tmp_path / "vad"),
            device="cuda:0",
            require_cuda=True,
        )
    assert "stt_qwen_cuda_unavailable" in str(excinfo.value)


def test_wrong_sample_rate_rejected(tmp_path: Path) -> None:
    """输入不是 16k：直接失败，不做静默重采样（extract 的契约必须被守住）。"""

    for name in ("asr", "aligner", "vad"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    wav = write_wav(tmp_path / "bad.wav", 2.0, sample_rate=8000)
    with pytest.raises(RuntimeError) as excinfo:
        run_stt_qwen(
            input_path=wav,
            output_path=tmp_path / "x.ja.srt",
            model_dir=str(tmp_path / "asr"),
            aligner_dir=str(tmp_path / "aligner"),
            vad_model_dir=str(tmp_path / "vad"),
            device="cpu",
        )
    assert "stt_qwen_expect_16k" in str(excinfo.value)


# ---------------------------------------------------------------- 辅助


def _parse_srt(path: Path) -> list[tuple[float, float, str]]:
    """解析 SRT 为 `(start, end, text)` 列表（只支持单行文本，测试用例够用）。"""

    blocks = [b for b in path.read_text(encoding="utf-8").split("\n\n") if b.strip()]
    out: list[tuple[float, float, str]] = []
    for block in blocks:
        lines = block.split("\n")
        start_text, end_text = lines[1].split(" --> ")
        out.append((_ts(start_text), _ts(end_text), lines[2]))
    return out


def _ts(value: str) -> float:
    """`HH:MM:SS,mmm` → 秒。"""

    hhmmss, millis = value.split(",")
    hours, minutes, seconds = hhmmss.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0
