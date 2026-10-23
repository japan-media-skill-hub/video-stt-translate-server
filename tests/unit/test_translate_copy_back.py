"""translate 阶段 copy_back 回写行为测试。"""

from __future__ import annotations

from pathlib import Path

import logging

import pytest

from whisper_stt_service.executor.translate import run_translate


@pytest.fixture(autouse=True)
def _restore_logger_propagation():
    """恢复 `whisper_stt_service` 的 propagate，避免其他用例 dictConfig 残留影响 caplog。"""

    logger = logging.getLogger("whisper_stt_service")
    saved = logger.propagate
    logger.propagate = True
    yield
    logger.propagate = saved


def test_translate_copy_back_defaults_to_video_dir(monkeypatch, tmp_path: Path) -> None:
    """未指定 copy_back 时，应默认回写到输入视频目录。"""

    config_path = tmp_path / "config.ini"
    config_path.write_text(
        "\n".join(
            [
                "[llm]",
                "base_url = https://example.com/v1",
                "api_key = test-key",
                "model = fake-model",
                "",
                "[translation]",
                "chunk_minutes = 30",
                "retry = 1",
            ]
        ),
        encoding="utf-8",
    )

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    input_ja_srt = artifact_dir / "sample.ja.srt"
    input_ja_srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n\n", encoding="utf-8"
    )
    output_zh_srt = artifact_dir / "sample.zh.srt"

    video_dir = tmp_path / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    input_video = video_dir / "sample.mp4"
    input_video.write_bytes(b"dummy")

    def fake_call_translate_api(*_args, **kwargs):  # type: ignore[no-untyped-def]
        batch = kwargs["batch"]
        return {entry.index: f"zh-{entry.index}" for entry in batch}

    monkeypatch.setattr(
        "whisper_stt_service.executor.translate._call_translate_api",
        fake_call_translate_api,
    )

    run_translate(
        input_ja_srt,
        output_zh_srt,
        config_path=config_path,
        timeout_sec=30,
        input_video_path=input_video,
        retry=1,
    )

    copied_ja = video_dir / "sample.ja.srt"
    copied_zh = video_dir / "sample.zh.srt"
    assert copied_ja.is_file()
    assert copied_zh.is_file()
    assert "zh-1" in copied_zh.read_text(encoding="utf-8")


def test_translate_copy_back_failure_only_warns(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    """copy_back 失败时仅告警，不应导致 translate 阶段失败。"""

    config_path = tmp_path / "config.ini"
    config_path.write_text(
        "\n".join(
            [
                "[llm]",
                "base_url = https://example.com/v1",
                "api_key = test-key",
                "model = fake-model",
                "",
                "[translation]",
                "chunk_minutes = 30",
                "retry = 1",
                f"copy_back = {tmp_path / 'copy-back'}",
            ]
        ),
        encoding="utf-8",
    )

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    input_ja_srt = artifact_dir / "sample.ja.srt"
    input_ja_srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n\n", encoding="utf-8"
    )
    output_zh_srt = artifact_dir / "sample.zh.srt"

    input_video = tmp_path / "sample.mp4"
    input_video.write_bytes(b"dummy")

    def fake_call_translate_api(*_args, **kwargs):  # type: ignore[no-untyped-def]
        batch = kwargs["batch"]
        return {entry.index: f"zh-{entry.index}" for entry in batch}

    def fake_atomic_write(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise PermissionError("readonly")

    monkeypatch.setattr(
        "whisper_stt_service.executor.translate._call_translate_api",
        fake_call_translate_api,
    )
    # copy_back 改为原子写：注入写入失败以验证“仅告警、不导致阶段失败”。
    monkeypatch.setattr(
        "whisper_stt_service.executor.translate.atomic_write_text", fake_atomic_write
    )

    run_translate(
        input_ja_srt,
        output_zh_srt,
        config_path=config_path,
        timeout_sec=30,
        input_video_path=input_video,
        retry=1,
    )

    assert output_zh_srt.is_file()
    assert "copy_back failed" in caplog.text
