"""三阶段 worker 的真实素材单元测试。

已迁移 manual / wav 契约：原文件是 `tests/unit/test_workers_real_media_inputs.py`
（曾被 gitignore 忽略的单元测试），现迁移到 `tests/manual/` 并按 SPEC E 使用
extract 产出的 16k mono wav。原文保留如下，迁移细节见后。

从 `tests/unit/test_workers_real_media_inputs.py` 迁移而来：

- 原文件曾被 `.gitignore` 忽略，导致真实媒体用例在仓库中不可见；现已迁移到
  `tests/manual/` 并纳入版本跟踪。
- 默认跳过：需显式设置环境变量 `WHISPER_STT_REAL_MEDIA=1` 才会执行；
  同时 `pyproject.toml` 的 `addopts` 默认执行 `-m "not manual and not e2e"`。
- 适配新契约（SPEC E / D1）：ASR（stt / stt_whisperx）只消费 extract 产出的
  16k mono wav（`audio/<stem>.16k.wav`），不再直接读原视频。

运行方式（只使用主仓库共享 venv，不得创建独立 venv）：

```bash
cd /home/base/repo/video-stt-whisper-server/.worktrees/reliability-hardening
WHISPER_STT_REAL_MEDIA=1 \
  /home/base/repo/video-stt-whisper-server/.venv/bin/python -m pytest \
  tests/manual/test_real_media.py -m manual -v
```

可用环境变量覆盖素材位置：

- `WHISPER_STT_REAL_MEDIA`：设为 `1` 才执行（必填开关）。
- `WHISPER_STT_REAL_OUTPUT_DIR`：真实素材目录，默认 `<repo>/output`。
- `WHISPER_STT_REAL_VIDEO`：视频文件名，默认 `sample_30s.mp4`。
- `WHISPER_STT_MODEL`：本地 faster-whisper 模型目录，默认
  `/home/base/repo/video-stt-whisper-server/models/faster-whisper-small`。
- `WHISPER_STT_CONFIG`：服务配置（含 llm / translation 段），默认 `<repo>/config.ini`。
"""

from __future__ import annotations

import os
from pathlib import Path
from queue import Queue

import pytest

from whisper_stt_service.executor import run_extract, run_stt, run_translate


# 真实媒体用例全部标记为 manual：默认 addopts 与 skipif 双重拦截。
pytestmark = pytest.mark.manual

# 环境开关：未显式开启时整个模块跳过（CI 与常规单测不会触碰模型 / LLM / 真实视频）。
REAL_MEDIA_ENABLED = os.getenv("WHISPER_STT_REAL_MEDIA", "").strip() in {
    "1",
    "true",
    "TRUE",
    "yes",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(
    os.getenv("WHISPER_STT_REAL_OUTPUT_DIR", str(REPO_ROOT / "output"))
)
VIDEO_NAME = os.getenv("WHISPER_STT_REAL_VIDEO", "sample_30s.mp4")
MODEL_DIR = Path(
    os.getenv(
        "WHISPER_STT_MODEL",
        "/home/base/repo/video-stt-whisper-server/models/faster-whisper-small",
    )
)
SERVICE_CONFIG_PATH = Path(
    os.getenv("WHISPER_STT_CONFIG", str(REPO_ROOT / "config.ini"))
)


def _drain_messages(q: Queue[dict]) -> list[dict]:
    """清空队列并返回所有事件，便于断言进度是否上报。"""

    items: list[dict] = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


def _extract_wav(input_video: Path, out_wav: Path, task_id: str) -> None:
    """按新契约先用 extract 产出 16k mono wav，供后续 ASR 消费。"""

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run_extract(
        input_video,
        out_wav,
        timeout_sec=300,
        progress_queue=Queue(maxsize=256),
        task_id=task_id,
        worker_id="manual-worker-extract",
    )
    assert out_wav.is_file()
    assert out_wav.stat().st_size > 0


@pytest.mark.skipif(
    not REAL_MEDIA_ENABLED,
    reason="set WHISPER_STT_REAL_MEDIA=1 to run real-media manual tests",
)
def test_extract_worker_uses_real_output_video(tmp_path: Path) -> None:
    """extract worker 使用 output 目录真实视频并产出 wav。

    已迁移 manual / wav 契约：现产出 16k mono wav。
    """

    input_video = OUTPUT_DIR / VIDEO_NAME
    assert input_video.is_file(), f"missing input video: {input_video}"

    out_wav = tmp_path / "sample_30s.worker.16k.wav"
    progress_queue: Queue[dict] = Queue(maxsize=256)

    run_extract(
        input_video,
        out_wav,
        timeout_sec=300,
        progress_queue=progress_queue,
        task_id="extract-real-1",
        worker_id="test-worker-extract",
    )

    assert out_wav.is_file()
    assert out_wav.stat().st_size > 0
    events = _drain_messages(progress_queue)
    assert any(e.get("stage") == "extract" for e in events)
    assert any(e.get("message") == "extract_done" for e in events)


@pytest.mark.skipif(
    not REAL_MEDIA_ENABLED,
    reason="set WHISPER_STT_REAL_MEDIA=1 to run real-media manual tests",
)
def test_stt_worker_consumes_extracted_wav(tmp_path: Path) -> None:
    """stt worker 使用 output 目录真实视频并产出字幕。

    已迁移 manual / wav 契约：输入改为 extract 产出的 16k mono wav（不再直接读原视频）。
    """

    input_video = OUTPUT_DIR / VIDEO_NAME
    assert input_video.is_file(), f"missing input video: {input_video}"
    assert MODEL_DIR.is_dir(), f"missing model dir: {MODEL_DIR}"
    assert (MODEL_DIR / "model.bin").is_file(), (
        f"missing model weight: {MODEL_DIR / 'model.bin'}"
    )

    # 新契约：ASR 输入必须是 extract 产物，而不是原视频。
    input_wav = tmp_path / "audio" / "sample_30s.16k.wav"
    _extract_wav(input_video, input_wav, task_id="stt-prep-extract")

    out_srt = tmp_path / "asr" / "sample_30s.worker.ja.srt"
    out_srt.parent.mkdir(parents=True, exist_ok=True)
    progress_queue: Queue[dict] = Queue(maxsize=512)

    run_stt(
        input_wav,
        out_srt,
        language="ja",
        timeout_sec=1800,
        model=str(MODEL_DIR),
        device="cpu",
        compute_type="int8",
        progress_every=5,
        progress_queue=progress_queue,
        task_id="stt-real-1",
        worker_id="test-worker-stt",
    )

    assert out_srt.is_file()
    assert out_srt.read_text(encoding="utf-8").strip() != ""
    events = _drain_messages(progress_queue)
    assert any(e.get("stage") == "stt" for e in events)
    assert any(e.get("message") == "stt_done" for e in events)


@pytest.mark.skipif(
    not REAL_MEDIA_ENABLED,
    reason="set WHISPER_STT_REAL_MEDIA=1 to run real-media manual tests",
)
def test_translate_worker_uses_real_output_artifacts(tmp_path: Path) -> None:
    """translate worker 使用真实视频先转写再调用真实 LLM 翻译。

    已迁移 manual / wav 契约：先 extract 出 16k mono wav、STT 产 ja.srt，再调用真实 LLM。
    """

    input_video = OUTPUT_DIR / VIDEO_NAME
    assert input_video.is_file(), f"missing input video: {input_video}"
    assert MODEL_DIR.is_dir(), f"missing model dir: {MODEL_DIR}"
    assert (MODEL_DIR / "model.bin").is_file(), (
        f"missing model weight: {MODEL_DIR / 'model.bin'}"
    )
    assert SERVICE_CONFIG_PATH.is_file(), (
        f"missing service config: {SERVICE_CONFIG_PATH}"
    )

    # 仅保留视频素材时，先由 STT 产出真实 ja.srt，再进入翻译阶段。
    # 历史说明（manual / wav 契约迁移后）：先由 extract 产出 wav，再由 STT 产出真实
    # ja.srt，最后进入翻译阶段。
    input_wav = tmp_path / "audio" / "sample_30s.16k.wav"
    _extract_wav(input_video, input_wav, task_id="translate-prep-extract")

    input_ja_srt = tmp_path / "asr" / "sample_30s.from_stt.ja.srt"
    input_ja_srt.parent.mkdir(parents=True, exist_ok=True)
    run_stt(
        input_wav,
        input_ja_srt,
        language="ja",
        timeout_sec=1800,
        model=str(MODEL_DIR),
        device="cpu",
        compute_type="int8",
        progress_every=5,
        task_id="translate-prep-stt",
        worker_id="test-worker-translate",
    )
    assert input_ja_srt.is_file()

    out_zh_srt = tmp_path / "translate" / "sample_30s.worker.zh.srt"
    out_zh_srt.parent.mkdir(parents=True, exist_ok=True)
    progress_queue: Queue[dict] = Queue(maxsize=512)

    run_translate(
        input_ja_srt,
        out_zh_srt,
        config_path=SERVICE_CONFIG_PATH,
        timeout_sec=600,
        input_video_path=input_video,
        progress_queue=progress_queue,
        task_id="translate-real-1",
        worker_id="test-worker-translate",
    )

    assert out_zh_srt.is_file()
    content = out_zh_srt.read_text(encoding="utf-8")
    assert content.strip() != ""
    events = _drain_messages(progress_queue)
    assert any(e.get("stage") == "translate" for e in events)
    assert any(e.get("message") == "translate_done" for e in events)
