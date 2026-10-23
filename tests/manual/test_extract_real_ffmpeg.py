"""extract 执行器的真实 ffmpeg 冒烟测试（manual）。

为什么单独放在 `tests/manual`：默认单测/CI 环境不保证有系统 ffmpeg/ffprobe，
而这里的价值正是**真实子进程**跑一遍 ffmpeg。曾经的真实 bug（临时文件名 `.part`
导致 ffmpeg 无法推断容器、任务报 `ffmpeg extraction failed`）只有真实执行才能发现，
所以保留为显式开关的冒烟测试；纯命令构造断言仍在
`tests/unit/test_executor_wav_contract.py` 里默认执行。

运行（只用主仓库共享 venv）：

```bash
WHISPER_STT_REAL_MEDIA=1 \
  /home/base/repo/video-stt-whisper-server/.venv/bin/python -m pytest \
  tests/manual/test_extract_real_ffmpeg.py -m manual -v
```

测试自己用 ffmpeg lavfi 生成一段 1s 正弦波 mp4，不依赖外部素材、模型或网络。
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from whisper_stt_service.executor import run_extract


pytestmark = pytest.mark.manual

# 环境开关：未显式开启时整个模块跳过（CI 与常规单测不会触碰真实 ffmpeg）。
REAL_MEDIA_ENABLED = os.getenv("WHISPER_STT_REAL_MEDIA", "").strip() in {
    "1",
    "true",
    "TRUE",
    "yes",
}

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
FFPROBE_AVAILABLE = shutil.which("ffprobe") is not None

_requires_real_ffmpeg = pytest.mark.skipif(
    not REAL_MEDIA_ENABLED,
    reason="set WHISPER_STT_REAL_MEDIA=1 to run real-ffmpeg manual smoke tests",
)
_requires_ffmpeg_binary = pytest.mark.skipif(
    not (FFMPEG_AVAILABLE and FFPROBE_AVAILABLE),
    reason="system ffmpeg/ffprobe not available (optional real-media smoke)",
)


def _make_sine_mp4(path: Path, seconds: float = 1.0) -> None:
    """用 ffmpeg 生成一段正弦波 mp4，作为真实媒体输入（无需外部素材）。"""

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-c:a",
            "aac",
            "-vn",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


@_requires_real_ffmpeg
@_requires_ffmpeg_binary
def test_run_extract_real_ffmpeg_produces_16k_mono_wav(tmp_path: Path) -> None:
    """真实 ffmpeg 冒烟：extract 产出真正 16k/mono/pcm_s16le 的 wav（回归容器推断 bug）。"""

    video = tmp_path / "sine.mp4"
    _make_sine_mp4(video)
    out_wav = tmp_path / "audio" / "sine.16k.wav"

    run_extract(video, out_wav, 60)

    assert out_wav.is_file() and out_wav.stat().st_size > 0
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-of",
            "default=noprint_wrappers=1",
            str(out_wav),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "codec_name=pcm_s16le" in probe
    assert "sample_rate=16000" in probe
    assert "channels=1" in probe
    # 成功路径不留中间文件。
    assert [p.name for p in out_wav.parent.iterdir() if p.name.startswith(".extract-")] == []


@_requires_real_ffmpeg
@_requires_ffmpeg_binary
def test_run_extract_real_ffmpeg_failure_keeps_previous_wav(tmp_path: Path) -> None:
    """真实 ffmpeg：失败时不删除既有正式 wav（不再 preclean 正式产物）。"""

    out_wav = tmp_path / "audio" / "keep.16k.wav"
    out_wav.parent.mkdir(parents=True)
    out_wav.write_bytes(b"OLD-WAV-CONTENT")
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not-a-media-file")

    with pytest.raises(RuntimeError, match="ffmpeg extraction failed"):
        run_extract(broken, out_wav, 30)

    assert out_wav.read_bytes() == b"OLD-WAV-CONTENT"
    assert [p.name for p in out_wav.parent.iterdir() if p.name.startswith(".extract-")] == []
