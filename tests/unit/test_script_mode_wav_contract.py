"""脚本模式（whisper_stt/）wav 契约与默认并发测试。

覆盖 SPEC E（D1）"ASR 只吃 wav"：
- 预抽默认开启，ffmpeg 产出 16k mono wav 才进入模型；
- 关闭预抽时只接受已存在的 wav，绝不把视频交给 faster-whisper；
- wav 缺失 / 0 字节报 missing_audio_artifact；
- ja.srt 原子写入（中断不留半截）。
以及 D2：脚本模式翻译默认并发为 2（可通过参数覆盖）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "whisper_stt"

# 最小合法 wav 头（RIFF....WAVE），用于区分"真 wav"与"其实是视频"。
WAV_BYTES = b"RIFF\x24\x00\x00\x00WAVEfmt "


def _load_script_module(name: str):
    """按文件路径加载脚本模式模块（whisper_stt/ 不是可导入包）。"""

    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tv():
    """加载 transcribe_video 模块。"""

    return _load_script_module("transcribe_video")


@pytest.fixture(scope="module")
def tz():
    """加载 translate_srt_ja_to_zh 模块。"""

    return _load_script_module("translate_srt_ja_to_zh")


# --------------------------------------------------------------------- wav 契约


def test_ensure_wav_artifact_accepts_non_empty_wav(tmp_path: Path, tv) -> None:
    """存在的非空 wav 通过校验并原样返回。"""

    wav = tmp_path / "a.16k.wav"
    wav.write_bytes(WAV_BYTES)
    assert tv.ensure_wav_artifact(wav) == wav


def test_ensure_wav_artifact_rejects_missing(tmp_path: Path, tv) -> None:
    """wav 缺失时报 missing_audio_artifact。"""

    with pytest.raises(tv.MissingAudioArtifactError) as excinfo:
        tv.ensure_wav_artifact(tmp_path / "nope.wav")
    assert excinfo.value.code == "missing_audio_artifact"
    assert "missing_audio_artifact" in str(excinfo.value)


def test_ensure_wav_artifact_rejects_empty_file(tmp_path: Path, tv) -> None:
    """0 字节 wav 同样视为缺失。"""

    wav = tmp_path / "empty.16k.wav"
    wav.write_bytes(b"")
    with pytest.raises(tv.MissingAudioArtifactError):
        tv.ensure_wav_artifact(wav)


def test_ensure_wav_artifact_rejects_video_payload(tmp_path: Path, tv) -> None:
    """存在但内容是视频的输入不得通过（防止把 mp4 喂给模型）。"""

    fake = tmp_path / "actually-video.wav"
    fake.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32)
    with pytest.raises(tv.MissingAudioArtifactError):
        tv.ensure_wav_artifact(fake)


def test_probe_duration_returns_none_when_ffprobe_missing(
    tmp_path: Path, tv, monkeypatch
) -> None:
    """干净 CI 无 ffprobe 时探测降级为 None，不得抛 FileNotFoundError。"""

    def missing_ffprobe(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(tv.subprocess, "run", missing_ffprobe)

    assert tv._probe_duration(tmp_path / "sample.mp4") is None


def _install_fake_model(tv, monkeypatch, seen: dict) -> None:
    """把 WhisperModel 替换为记录输入路径的假模型。"""

    class FakeSegment:
        def __init__(self, start: float, end: float, text: str) -> None:
            self.start = start
            self.end = end
            self.text = text

    class FakeInfo:
        language = "ja"
        language_probability = 0.99

    class FakeModel:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def transcribe(self, audio_path, **_kwargs):
            seen["audio_path"] = str(audio_path)
            return iter([FakeSegment(0.0, 1.0, "こんにちは")]), FakeInfo()

    monkeypatch.setattr(tv, "WhisperModel", FakeModel)
    return FakeModel


def test_main_default_preextracts_wav_and_never_feeds_video(
    tmp_path: Path, tv, monkeypatch
) -> None:
    """默认预抽：ffmpeg 产出 wav，模型收到的是 wav 而不是原视频。"""

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"dummy-video")
    output = tmp_path / "sample.ja.srt"
    extracted: dict[str, object] = {}

    def fake_extract(input_path, output_wav, _duration):  # type: ignore[no-untyped-def]
        extracted["input"] = input_path
        extracted["wav"] = output_wav
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_wav.write_bytes(WAV_BYTES)

    monkeypatch.setattr(tv, "_extract_audio_with_progress", fake_extract)
    monkeypatch.setattr(tv, "_probe_duration", lambda _p: 1.0)
    monkeypatch.setattr(
        tv,
        "_parse_args",
        lambda: SimpleNamespace(
            input=video,
            output=output,
            model="fake-model",
            language="ja",
            device="cpu",
            compute_type="int8",
            beam_size=5,
            preextract_wav=True,
            temp_audio=None,
            keep_temp_audio=False,
            progress_every=25,
        ),
    )
    seen: dict = {}
    _install_fake_model(tv, monkeypatch, seen)

    assert tv.main() == 0

    assert extracted["input"] == video
    assert str(extracted["wav"]).endswith(".16k.wav")
    # 模型必须收到 wav，绝不能是原视频
    assert seen["audio_path"].endswith(".16k.wav")
    assert seen["audio_path"] != str(video)
    assert "こんにちは" in output.read_text(encoding="utf-8")


def test_main_default_preextract_survives_missing_ffprobe(
    tmp_path: Path, tv, monkeypatch
) -> None:
    """默认预抽在无 ffprobe 的 runner 上仍成功（进度降级，不阻断 wav 提取）。"""

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"dummy-video")
    output = tmp_path / "sample.ja.srt"

    def fake_extract(_input_path, output_wav, _duration):  # type: ignore[no-untyped-def]
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_wav.write_bytes(WAV_BYTES)

    def missing_ffprobe(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(tv, "_extract_audio_with_progress", fake_extract)
    monkeypatch.setattr(tv.subprocess, "run", missing_ffprobe)
    monkeypatch.setattr(
        tv,
        "_parse_args",
        lambda: SimpleNamespace(
            input=video,
            output=output,
            model="fake-model",
            language="ja",
            device="cpu",
            compute_type="int8",
            beam_size=5,
            preextract_wav=True,
            temp_audio=None,
            keep_temp_audio=False,
            progress_every=25,
        ),
    )
    seen: dict = {}
    _install_fake_model(tv, monkeypatch, seen)

    assert tv.main() == 0
    assert seen["audio_path"].endswith(".16k.wav")
    assert "こんにちは" in output.read_text(encoding="utf-8")


def test_main_without_preextract_rejects_video_input(
    tmp_path: Path, tv, monkeypatch
) -> None:
    """关闭预抽且未提供 wav 时：报 missing_audio_artifact，绝不把视频给模型。"""

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"dummy-video")
    output = tmp_path / "sample.ja.srt"

    def _must_not_probe(_path):  # type: ignore[no-untyped-def]
        raise AssertionError("视频输入必须在 ffprobe 前被 wav 契约拒绝")

    def _must_not_resolve(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("视频输入必须在 runtime/模型解析前被拒绝")

    # 即使 runner 有 ffprobe，也不允许先探测；契约必须先拒绝视频输入。
    monkeypatch.setattr(tv, "_probe_duration", _must_not_probe)
    monkeypatch.setattr(tv, "_resolve_runtime", _must_not_resolve)
    monkeypatch.setattr(
        tv,
        "_parse_args",
        lambda: SimpleNamespace(
            input=video,
            output=output,
            model="fake-model",
            language="ja",
            device="cpu",
            compute_type="int8",
            beam_size=5,
            preextract_wav=False,
            temp_audio=None,
            keep_temp_audio=False,
            progress_every=25,
        ),
    )

    seen: dict = {}
    _install_fake_model(tv, monkeypatch, seen)

    with pytest.raises(tv.MissingAudioArtifactError):
        tv.main()

    # 模型根本不应被调用
    assert "audio_path" not in seen
    assert not output.exists()


def test_main_without_preextract_accepts_existing_wav(
    tmp_path: Path, tv, monkeypatch
) -> None:
    """关闭预抽但显式给了 wav：直接用该 wav，不调用 ffmpeg。"""

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"dummy-video")
    wav = tmp_path / "pre.16k.wav"
    wav.write_bytes(WAV_BYTES)
    output = tmp_path / "sample.ja.srt"

    def _no_extract(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("不应再调用 ffmpeg 预抽")

    monkeypatch.setattr(tv, "_extract_audio_with_progress", _no_extract)
    monkeypatch.setattr(tv, "_probe_duration", lambda _p: 1.0)
    monkeypatch.setattr(
        tv,
        "_parse_args",
        lambda: SimpleNamespace(
            input=video,
            output=output,
            model="fake-model",
            language="ja",
            device="cpu",
            compute_type="int8",
            beam_size=5,
            preextract_wav=False,
            temp_audio=wav,
            keep_temp_audio=False,
            progress_every=25,
        ),
    )
    seen: dict = {}
    _install_fake_model(tv, monkeypatch, seen)

    assert tv.main() == 0
    assert seen["audio_path"] == str(wav)


def test_main_writes_srt_atomically(tmp_path: Path, tv, monkeypatch) -> None:
    """ja.srt 为原子写入：成功后无残留临时文件，旧文件被整体替换。"""

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"dummy-video")
    output = tmp_path / "sample.ja.srt"
    output.write_text("OLD CONTENT", encoding="utf-8")

    def fake_extract(_inp, wav, _dur):  # type: ignore[no-untyped-def]
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(WAV_BYTES)

    monkeypatch.setattr(tv, "_extract_audio_with_progress", fake_extract)
    monkeypatch.setattr(tv, "_probe_duration", lambda _p: 1.0)
    monkeypatch.setattr(
        tv,
        "_parse_args",
        lambda: SimpleNamespace(
            input=video,
            output=output,
            model="fake-model",
            language="ja",
            device="cpu",
            compute_type="int8",
            beam_size=5,
            preextract_wav=True,
            temp_audio=None,
            keep_temp_audio=True,
            progress_every=25,
        ),
    )
    _install_fake_model(tv, monkeypatch, {})

    assert tv.main() == 0

    content = output.read_text(encoding="utf-8")
    assert "OLD CONTENT" not in content
    assert "こんにちは" in content
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp-")]
    assert leftovers == []


def test_atomic_write_keeps_old_file_on_failure(tmp_path: Path, tv, monkeypatch) -> None:
    """原子写失败时保留旧产物并清理临时文件。"""

    target = tmp_path / "x.srt"
    target.write_text("OLD", encoding="utf-8")

    def boom(_src, _dst):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr(tv.os, "replace", boom)
    with pytest.raises(OSError):
        tv.atomic_write_text(target, "NEW")

    assert target.read_text(encoding="utf-8") == "OLD"
    assert [p.name for p in tmp_path.iterdir() if p.name != "x.srt"] == []


# ---------------------------------------------------------- WhisperX 脚本路径


@pytest.fixture(scope="module")
def wx():
    """加载 transcribe_video_whisperx 模块。"""

    return _load_script_module("transcribe_video_whisperx")


def _whisperx_args(tmp_path: Path, **overrides):
    """构造 WhisperX 脚本的最小参数集（避免重复样板）。"""

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"dummy-video")
    vad_cfg = tmp_path / "vad.yaml"
    vad_cfg.write_text("x", encoding="utf-8")
    align_root = tmp_path / "align"
    align_root.mkdir(exist_ok=True)
    values = dict(
        input=video,
        output=tmp_path / "sample.whisperx.ja.srt",
        model="fake-model",
        language="ja",
        device="cpu",
        compute_type="int8",
        batch_size=16,
        vad_config_path=vad_cfg,
        align_model_root=align_root,
        align_enabled=True,
        vad_backend="pyannote",
        vad_onset=0.5,
        vad_offset=0.363,
        local_files_only=True,
        timeout_sec=60,
        preextract_wav=True,
        temp_audio=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_whisperx_default_preextracts_wav_and_never_feeds_video(
    tmp_path: Path, wx, monkeypatch
) -> None:
    """WhisperX 默认预抽：交给 run_stt_whisperx 的是 wav 而非原视频。"""

    seen: dict = {}

    def fake_extract(_inp, wav, _dur):  # type: ignore[no-untyped-def]
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(WAV_BYTES)

    def fake_run(**kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return {"effective": True}

    args = _whisperx_args(tmp_path)
    monkeypatch.setattr(wx, "_extract_audio_with_progress", fake_extract)
    monkeypatch.setattr(wx, "_probe_duration", lambda _p: 1.0)
    monkeypatch.setattr(wx, "run_stt_whisperx", fake_run)
    monkeypatch.setattr(wx, "_parse_args", lambda: args)

    assert wx.main() == 0
    fed = str(seen["input_video"])
    assert fed.endswith(".16k.wav")
    assert fed != str(args.input)


def test_whisperx_without_preextract_rejects_video_input(
    tmp_path: Path, tv, wx, monkeypatch
) -> None:
    """WhisperX 关闭预抽且无 wav：报 missing_audio_artifact，不调用 ASR。"""

    called = {"n": 0}

    def fake_run(**_kwargs):  # type: ignore[no-untyped-def]
        called["n"] += 1
        return {}

    def _must_not_probe(_path):  # type: ignore[no-untyped-def]
        raise AssertionError("WhisperX 视频输入必须在 ffprobe 前被 wav 契约拒绝")

    args = _whisperx_args(tmp_path, preextract_wav=False)
    monkeypatch.setattr(wx, "_probe_duration", _must_not_probe)
    monkeypatch.setattr(wx, "run_stt_whisperx", fake_run)
    monkeypatch.setattr(wx, "_parse_args", lambda: args)

    # 复用 transcribe_video 的同一异常类型（两条脚本路径共用一份 wav 契约实现）。
    with pytest.raises(tv.MissingAudioArtifactError):
        wx.main()
    assert called["n"] == 0  # ASR 根本没被调用


def test_whisperx_without_preextract_accepts_existing_wav(
    tmp_path: Path, wx, monkeypatch
) -> None:
    """WhisperX 关闭预抽但给了 wav：直接用该 wav，不跑 ffmpeg。"""

    wav = tmp_path / "pre.16k.wav"
    wav.write_bytes(WAV_BYTES)
    seen: dict = {}

    def fake_run(**kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return {}

    def _no_extract(*_a, **_k):  # type: ignore[no-untyped-def]
        raise AssertionError("不应调用 ffmpeg 预抽")

    args = _whisperx_args(tmp_path, preextract_wav=False, temp_audio=wav)
    monkeypatch.setattr(wx, "_extract_audio_with_progress", _no_extract)
    monkeypatch.setattr(wx, "run_stt_whisperx", fake_run)
    monkeypatch.setattr(wx, "_parse_args", lambda: args)

    assert wx.main() == 0
    assert str(seen["input_video"]) == str(wav)


def test_whisperx_cleans_temporary_wav(tmp_path: Path, wx, monkeypatch) -> None:
    """预抽生成的临时 wav 在成功后删除，不污染输出目录。"""

    def fake_extract(_inp, wav, _dur):  # type: ignore[no-untyped-def]
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(WAV_BYTES)

    args = _whisperx_args(tmp_path)
    monkeypatch.setattr(wx, "_extract_audio_with_progress", fake_extract)
    monkeypatch.setattr(wx, "_probe_duration", lambda _p: 1.0)
    monkeypatch.setattr(wx, "run_stt_whisperx", lambda **_kw: {})
    monkeypatch.setattr(wx, "_parse_args", lambda: args)

    assert wx.main() == 0
    assert list(tmp_path.glob("*.tmp.16k.wav")) == []


# ------------------------------------------------------------ 主仓库共享 venv

SCRIPTS_DIR = REPO_ROOT / "scripts"


def _run_venv_resolver(origin_dir: Path, env: dict[str, str] | None = None) -> dict:
    """在子 shell 中 source scripts/_venv.sh 并返回解析结果。

    用真实 bash 执行，确保 set -euo pipefail、git 与数组等语义都被覆盖。
    """

    import os
    import subprocess

    probe = (
        'source "' + str(SCRIPTS_DIR / "_venv.sh") + '"\n'
        'echo "MAIN_REPO_ROOT=$MAIN_REPO_ROOT"\n'
        'echo "VENV_PYTHON=$VENV_PYTHON"\n'
    )
    merged_env = dict(os.environ)
    merged_env["SCRIPT_ORIGIN_DIR"] = str(origin_dir)
    merged_env.pop("WHISPER_STT_ROOT", None)
    merged_env.update(env or {})
    proc = subprocess.run(
        ["bash", "-c", probe],
        capture_output=True,
        text=True,
        env=merged_env,
        check=False,
    )
    result = {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            result[key] = value
    return result


def test_venv_helper_resolves_main_repo_from_worktree() -> None:
    """worktree 中 .venv 不存在，必须解析到主仓库共享 venv。"""

    result = _run_venv_resolver(SCRIPTS_DIR)
    assert result["returncode"] == 0, result["stderr"]
    assert result["VENV_PYTHON"].endswith("/.venv/bin/python")
    assert ".worktrees" not in result["VENV_PYTHON"]


def test_venv_helper_honours_whisper_stt_root(tmp_path: Path) -> None:
    """WHISPER_STT_ROOT 显式指定时优先采用。"""

    import sys

    fake_root = tmp_path / "main-repo"
    (fake_root / ".venv" / "bin").mkdir(parents=True)
    (fake_root / ".venv" / "bin" / "python").symlink_to(sys.executable)
    result = _run_venv_resolver(SCRIPTS_DIR, {"WHISPER_STT_ROOT": str(fake_root)})
    assert result["returncode"] == 0, result["stderr"]
    assert result["VENV_PYTHON"] == str(fake_root / ".venv" / "bin" / "python")


def test_scripts_never_hardcode_worktree_venv() -> None:
    """三个脚本都必须 source _venv.sh 且不在 worktree 内找 venv。"""

    for name in (
        "run_video_ja_srt.sh",
        "run_video_ja_srt_whisperx.sh",
        "run_video_ja_zh.sh",
    ):
        text = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        assert "_venv.sh" in text, f"{name} 未 source _venv.sh"
        assert "${REPO_ROOT}/.venv" not in text, f"{name} 仍在 worktree 内找 venv"
        assert "${VENV_PYTHON}" in text, f"{name} 未使用共享 venv"
        assert "SCRIPT_ORIGIN_DIR}/.." in text, f"{name} REPO_ROOT 推导错误"


def test_scripts_reference_existing_entrypoints() -> None:
    """脚本引用的 python 入口文件必须存在（防止 REPO_ROOT 推导错误）。"""

    expectations = {
        "run_video_ja_srt.sh": "whisper_stt/transcribe_video.py",
        "run_video_ja_srt_whisperx.sh": "whisper_stt/transcribe_video_whisperx.py",
        "run_video_ja_zh.sh": "whisper_stt/translate_srt_ja_to_zh.py",
    }
    for name, target in expectations.items():
        assert (REPO_ROOT / target).is_file(), f"{name} 指向不存在的 {target}"
        text = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        assert target in text, f"{name} 未引用 {target}"


# --------------------------------------------------------------------- 默认并发


def test_script_translate_parallel_defaults_to_two(tz) -> None:
    """D2：脚本模式翻译默认并发为 2。"""

    import configparser

    config = configparser.ConfigParser()
    value = tz._cfg_int(None, config, "translation", "parallel", 2)
    assert value == 2


def test_script_translate_parallel_is_overridable(tz) -> None:
    """--parallel 与 [translation] parallel 仍可覆盖默认值。"""

    import configparser

    config = configparser.ConfigParser()
    config.read_string("[translation]\nparallel = 5\n")
    # 配置覆盖默认
    assert tz._cfg_int(None, config, "translation", "parallel", 2) == 5
    # 命令行优先级最高
    assert tz._cfg_int(3, config, "translation", "parallel", 2) == 3


def test_script_translate_source_declares_parallel_default_two() -> None:
    """源码中默认值字面量为 2（默认并发 16 已废弃）。"""

    source = (SCRIPT_DIR / "translate_srt_ja_to_zh.py").read_text(encoding="utf-8")
    assert '"translation", "parallel", 2)' in source
    assert '"translation", "parallel", 16)' not in source
