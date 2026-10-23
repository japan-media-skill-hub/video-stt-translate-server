"""translate 阶段可靠性加固测试（批次 4 C.1–C.7）。

覆盖：受限并发、三重切分与稳定 batch_id、检查点复用/失效、部分 id 降级保留原文、
strict_ids、4xx 不重试 / 5xx 重试 + 幂等键、原子写、取消检查、llm 回退 profile。
"""

from __future__ import annotations

import configparser
import json
import threading
import time
from pathlib import Path

import pytest
import requests

from whisper_stt_service.db import Database
from whisper_stt_service.executor.atomic import (
    StaleWriteRejected,
    atomic_write_text,
    atomic_write_text_fenced,
)
from whisper_stt_service.executor.common import (
    ExecutorCancelledError,
    MissingAudioArtifactError,
    SrtEntry,
    _dump_srt,
    build_translate_batches,
    resolve_translate_settings,
)
from whisper_stt_service.executor.translate import (
    TranslateCancelledError,
    _build_idempotency_key,
    _load_checkpoint,
    _load_llm_profiles,
    _pick_llm_profile,
    _save_checkpoint,
    _sub_batch_id,
    run_translate,
)
from whisper_stt_service.repository import JobRepository

@pytest.fixture(autouse=True)
def _restore_logger_propagation():
    """保证用例不受其他用例 dictConfig 残留的 propagate=False 影响。

    `configure_logging` 会把 `whisper_stt_service` logger 设为不向 root 传播，
    若前面的用例未恢复，`caplog` 将收不到本模块的日志。这里在用例前后复原。

    改用共享的整棵 logger 树快照/还原（tests/conftest.py），而不是只翻转
    `propagate` 一个字段：本模块只用 `caplog.set_level`，若残留了
    handlers/disabled 等其它状态，只还原 `propagate` 仍会漏日志。
    """

    from tests.conftest import _restore, _snapshot

    before = _snapshot()
    try:
        yield
    finally:
        _restore(before)


LLM_SECTION = [
    "[llm]",
    "base_url = https://primary.example/v1",
    "api_key = primary-key",
    "model = primary-model",
]


def _write_config(
    path: Path,
    *,
    translation_lines: list[str] | None = None,
    extra_sections: list[str] | None = None,
    fallback_enabled: bool | None = None,
) -> Path:
    """生成测试用 config.ini；可选追加 [translation] 与回退 LLM 段。"""

    lines = list(LLM_SECTION)
    if fallback_enabled is not None:
        lines.extend(
            [
                "",
                "[llm_fallback]",
                f"enabled = {'true' if fallback_enabled else 'false'}",
                "base_url = https://fallback.example/v1",
                "api_key = fallback-key",
                "model = fallback-model",
            ]
        )
    lines.extend(extra_sections or [])
    lines.extend(["", "[translation]"])
    lines.extend(
        translation_lines
        or [
            "chunk_minutes = 30",
            "retry = 3",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _make_srt(path: Path, count: int, *, text: str = "日本語テキスト") -> Path:
    """生成包含 count 条字幕的日文 SRT（每条相差 1 秒）。"""

    blocks = []
    for i in range(1, count + 1):
        start = f"00:00:{i:02d},000"
        end = f"00:00:{i:02d},900"
        blocks.append(f"{i}\n{start} --> {end}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


class FakeResponse:
    """最小化 requests.Response 替身：携带状态码与 JSON 内容。"""

    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        """保存状态码与响应体。"""

        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        """模拟 requests 的 HTTP 错误抛出行为。"""

        if self.status_code >= 400:
            error = requests.HTTPError(f"status={self.status_code}")
            error.response = self  # type: ignore[assignment]
            raise error

    def json(self) -> dict:
        """返回预置响应体。"""

        return self._payload


def _ok_payload(ids: list[int], prefix: str = "zh") -> dict:
    """构造模型正常返回的 choices 结构。"""

    items = [{"id": i, "text_zh": f"{prefix}-{i}"} for i in ids]
    return {"choices": [{"message": {"content": json.dumps(items, ensure_ascii=False)}}]}


# --------------------------------------------------------------------------- C.3 切分


def test_build_batches_three_way_split_with_stable_batch_ids() -> None:
    """时间窗 + 条目数 + 字符数三重上限同时生效，batch_id 稳定。"""

    entries = [
        SrtEntry(index=i, timestamp=f"00:00:{i:02d},000 --> 00:00:{i:02d},900", text="x" * 10)
        for i in range(1, 26)
    ]

    batches = build_translate_batches(
        entries, window_minutes=30, batch_size=10, max_chars=35
    )

    # 每批最多 3 条（3*10=30 字符，再加一条会超过 35）。
    assert [len(b.entries) for b in batches] == [3, 3, 3, 3, 3, 3, 3, 3, 1]
    assert [b.batch_id for b in batches] == [
        "0/0", "0/1", "0/2", "0/3", "0/4", "0/5", "0/6", "0/7", "0/8"
    ]
    # 顺序必须与原文一致，不得按批大小重排。
    assert [e.index for b in batches for e in b.entries] == list(range(1, 26))


def test_build_batches_batch_size_limit_wins_when_chars_unbounded() -> None:
    """字符上限足够时，条目数上限先生效。"""

    entries = [
        SrtEntry(index=i, timestamp="00:00:01,000 --> 00:00:02,000", text="a")
        for i in range(1, 8)
    ]
    batches = build_translate_batches(
        entries, window_minutes=30, batch_size=3, max_chars=12000
    )
    assert [b.batch_id for b in batches] == ["0/0", "0/1", "0/2"]
    assert [len(b.entries) for b in batches] == [3, 3, 1]


def test_build_batches_oversized_entry_becomes_own_batch_untruncated() -> None:
    """单条超长字幕单独成批且不截断原文。"""

    long_text = "あ" * 20000  # 超过默认 max_chars=12000
    entries = [
        SrtEntry(index=1, timestamp="00:00:01,000 --> 00:00:02,000", text="short"),
        SrtEntry(index=2, timestamp="00:00:03,000 --> 00:00:04,000", text=long_text),
        SrtEntry(index=3, timestamp="00:00:05,000 --> 00:00:06,000", text="short2"),
    ]
    batches = build_translate_batches(
        entries, window_minutes=30, batch_size=200, max_chars=12000
    )
    assert [b.ids for b in batches] == [[1], [2], [3]]
    assert batches[1].entries[0].text == long_text


def test_build_batches_window_partition_is_stable() -> None:
    """跨时间窗时批次号从新窗口重新编号，同一输入结果稳定。"""

    entries = [
        SrtEntry(index=1, timestamp="00:00:01,000 --> 00:00:02,000", text="a"),
        SrtEntry(index=2, timestamp="00:31:01,000 --> 00:31:02,000", text="b"),
    ]
    first = build_translate_batches(entries, window_minutes=30, batch_size=200)
    second = build_translate_batches(entries, window_minutes=30, batch_size=200)
    assert [b.batch_id for b in first] == ["0/0", "1/0"]
    assert [(b.batch_id, b.ids) for b in first] == [(b.batch_id, b.ids) for b in second]


# --------------------------------------------------------------------------- C.2 配置


def test_resolve_translate_settings_precedence(tmp_path: Path) -> None:
    """job_config 覆盖 > [translation] > 代码默认值。"""

    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=[
            "parallel = 5",
            "batch_size = 50",
            "max_chars = 9000",
            "disable_proxy = false",
            "strict_ids = true",
        ],
    )
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")

    resolved = resolve_translate_settings(cfg, {"parallel": 1, "batch_size": 7})

    assert resolved.parallel == 1  # job_config 覆盖
    assert resolved.batch_size == 7  # job_config 覆盖
    assert resolved.max_chars == 9000  # [translation] 生效
    assert resolved.disable_proxy is False  # [translation] 生效
    assert resolved.strict_ids is True  # [translation] 生效
    assert resolved.checkpoint_enabled is True  # 代码默认值
    assert resolved.chunk_minutes == 30  # 代码默认值


def test_resolve_translate_settings_legacy_kwargs_still_override(tmp_path: Path) -> None:
    """旧调用方的 chunk_minutes/retry/timeout_sec/copy_back 关键字继续生效。"""

    config_path = _write_config(
        tmp_path / "config.ini", translation_lines=["chunk_minutes = 30", "retry = 3"]
    )
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")

    resolved = resolve_translate_settings(
        cfg, None, chunk_minutes=5, retry=2, timeout_sec=99, copy_back="/tmp/x"
    )

    assert (resolved.chunk_minutes, resolved.retry) == (5, 2)
    assert resolved.timeout_sec == 99
    assert resolved.copy_back == "/tmp/x"


def test_resolve_translate_settings_parallel_defaults_to_two(tmp_path: Path) -> None:
    """缺省配置下并发上限为 2（D2）。"""

    config_path = _write_config(tmp_path / "config.ini", translation_lines=[])
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")

    assert resolve_translate_settings(cfg).parallel == 2


# --------------------------------------------------------------------------- C.1 并发


def test_translate_concurrency_never_exceeds_parallel(
    monkeypatch, tmp_path: Path
) -> None:
    """并发峰值不得超过 parallel（配置为 2），且确实并发而非串行。"""

    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=[
            "chunk_minutes = 1",
            "batch_size = 1",
            "parallel = 2",
            "retry = 1",
            "request_interval = 0",
            "copy_back = __video_dir__",
        ],
    )
    ja = _make_srt(tmp_path / "sample.ja.srt", 8)
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")

    lock = threading.Lock()
    state = {"active": 0, "peak": 0, "count": 0}

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            state["count"] += 1
        time.sleep(0.03)
        with lock:
            state["active"] -= 1
        return {e.index: f"zh-{e.index}" for e in kwargs["batch"]}

    monkeypatch.setattr(
        "whisper_stt_service.executor.translate._call_translate_api", fake_call
    )

    run_translate(
        ja,
        tmp_path / "sample.zh.srt",
        config_path=config_path,
        timeout_sec=30,
        input_video_path=video,
    )

    assert state["peak"] <= 2
    assert state["peak"] == 2  # 确认确实并发，而不是退化成串行
    assert state["count"] == 8


def test_translate_output_preserves_srt_order(monkeypatch, tmp_path: Path) -> None:
    """并发完成顺序打乱时，输出仍按原 SRT 顺序。"""

    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=[
            "chunk_minutes = 1",
            "batch_size = 1",
            "parallel = 2",
            "retry = 1",
            "request_interval = 0",
        ],
    )
    srt_blocks = []
    for i in range(1, 7):
        srt_blocks.append(f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},900\n原文{i}\n")
    ja = tmp_path / "sample.ja.srt"
    ja.write_text("\n".join(srt_blocks), encoding="utf-8")
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        batch_entries = kwargs["batch"]
        # 前面的批次更慢，故意制造完成顺序与序号相反。
        time.sleep(max(0.05 - batch_entries[0].index * 0.005, 0.0))
        return {e.index: f"zh-{e.index}" for e in batch_entries}

    monkeypatch.setattr(
        "whisper_stt_service.executor.translate._call_translate_api", fake_call
    )
    run_translate(
        ja,
        tmp_path / "sample.zh.srt",
        config_path=config_path,
        timeout_sec=30,
        input_video_path=video,
    )

    lines = (tmp_path / "sample.zh.srt").read_text(encoding="utf-8").split("\n\n")
    texts = [b.splitlines()[2] for b in lines if b.strip()]
    assert texts == [f"zh-{i}" for i in range(1, 7)]


# --------------------------------------------------------------------------- C.4 检查点


def test_checkpoint_written_and_reused_on_rerun(
    monkeypatch, tmp_path: Path
) -> None:
    """首批失败后重跑只补未完成批次（检查点复用）。"""

    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=[
            "chunk_minutes = 1",
            "batch_size = 1",
            "parallel = 1",
            "retry = 1",
            "request_interval = 0",
        ],
    )
    ja = _make_srt(tmp_path / "sample.ja.srt", 4)
    output = tmp_path / "sample.zh.srt"
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")
    checkpoint = output.with_suffix(output.suffix + ".progress.json")

    calls: list[list[int]] = []
    fail_from = {"value": 3}

    import whisper_stt_service.executor.translate as translate_module

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        ids = [e.index for e in kwargs["batch"]]
        calls.append(ids)
        if ids and ids[0] >= fail_from["value"]:
            raise requests.ConnectionError("boom")
        return {i: f"zh-{i}" for i in ids}

    original = translate_module._call_translate_api
    translate_module._call_translate_api = fake_call  # type: ignore[assignment]
    try:
        with pytest.raises(Exception):
            run_translate(
                ja, output, config_path=config_path, timeout_sec=30, input_video_path=video
            )

        assert checkpoint.is_file()
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert payload["version"] == 1
        assert payload["chunk_minutes"] == 1
        assert payload["max_chars"] == 12000
        done = [k for k, v in payload["batches"].items() if v.get("done")]
        assert done == ["0/0", "0/1"]
        assert not output.exists()  # 失败不得写出半成品

        calls.clear()
        fail_from["value"] = 99
        run_translate(
            ja, output, config_path=config_path, timeout_sec=30, input_video_path=video
        )
    finally:
        translate_module._call_translate_api = original  # type: ignore[assignment]

    # 重跑只请求未完成的两个批次。
    assert sorted(i for call in calls for i in call) == [3, 4]
    text = output.read_text(encoding="utf-8")
    assert all(f"zh-{i}" in text for i in range(1, 5))


def test_checkpoint_invalidated_when_source_changes(
    monkeypatch, tmp_path: Path
) -> None:
    """ja.srt 内容变化后不得复用旧检查点。"""

    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=["chunk_minutes = 1", "batch_size = 1", "parallel = 2",
                           "retry = 1", "request_interval = 0"],
    )
    ja = _make_srt(tmp_path / "sample.ja.srt", 3)
    output = tmp_path / "sample.zh.srt"
    checkpoint = output.with_suffix(output.suffix + ".progress.json")
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")

    import whisper_stt_service.executor.translate as translate_module

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        return {e.index: f"zh-{e.index}" for e in kwargs["batch"]}

    original = translate_module._call_translate_api
    translate_module._call_translate_api = fake_call  # type: ignore[assignment]
    try:
        run_translate(ja, output, config_path=config_path, timeout_sec=30, input_video_path=video)
        assert checkpoint.is_file()

        # 修改源文件后旧检查点必须失效
        ja.write_text(ja.read_text(encoding="utf-8").replace("日本語テキスト", "変更"), encoding="utf-8")
        cfg = configparser.ConfigParser()
        cfg.read(config_path, encoding="utf-8")
        settings = resolve_translate_settings(cfg)
        from whisper_stt_service.executor.common import _sha256_file

        restored = _load_checkpoint(
            checkpoint, source_sha256=_sha256_file(ja), settings=settings
        )
        assert restored == {}
    finally:
        translate_module._call_translate_api = original  # type: ignore[assignment]


def test_checkpoint_disabled_writes_no_file(monkeypatch, tmp_path: Path) -> None:
    """checkpoint_enabled=false 时不写检查点。"""

    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=["checkpoint_enabled = false", "retry = 1", "request_interval = 0"],
    )
    ja = _make_srt(tmp_path / "sample.ja.srt", 2)
    output = tmp_path / "sample.zh.srt"
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")

    import whisper_stt_service.executor.translate as translate_module

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        return {e.index: f"zh-{e.index}" for e in kwargs["batch"]}

    original = translate_module._call_translate_api
    translate_module._call_translate_api = fake_call  # type: ignore[assignment]
    try:
        run_translate(ja, output, config_path=config_path, timeout_sec=30, input_video_path=video)
    finally:
        translate_module._call_translate_api = original  # type: ignore[assignment]

    assert not output.with_suffix(output.suffix + ".progress.json").exists()


# --------------------------------------------------------------------------- C.5 部分 id


def test_partial_ids_fallback_keeps_source_text(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    """重试后仍缺失的 id 保留日文原文写入，且不丢行。"""

    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=["batch_size = 10", "retry = 2", "request_interval = 0"],
    )
    ja = _make_srt(tmp_path / "sample.ja.srt", 3)
    output = tmp_path / "sample.zh.srt"
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")

    import whisper_stt_service.executor.translate as translate_module

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        ids = [e.index for e in kwargs["batch"]]
        # 永远只返回 id=1 的译文，其余视为缺失
        return {i: "zh-1" for i in ids if i == 1}

    original = translate_module._call_translate_api
    translate_module._call_translate_api = fake_call  # type: ignore[assignment]
    caplog.set_level("WARNING")
    try:
        run_translate(ja, output, config_path=config_path, timeout_sec=30, input_video_path=video)
    finally:
        translate_module._call_translate_api = original  # type: ignore[assignment]

    content = output.read_text(encoding="utf-8")
    blocks = [b for b in content.split("\n\n") if b.strip()]
    assert len(blocks) == 3  # 三条字幕一条都不能少
    assert "zh-1" in blocks[0]
    assert "日本語テキスト" in blocks[1] and "日本語テキスト" in blocks[2]
    assert "translate_partial" in caplog.text


def test_strict_ids_fails_batch_on_missing(monkeypatch, tmp_path: Path) -> None:
    """strict_ids=true 时缺失即整批失败。"""

    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=[
            "batch_size = 10",
            "retry = 1",
            "strict_ids = true",
            "request_interval = 0",
        ],
    )
    ja = _make_srt(tmp_path / "sample.ja.srt", 3)
    output = tmp_path / "sample.zh.srt"
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")

    import whisper_stt_service.executor.translate as translate_module

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        return {1: "zh-1"}

    original = translate_module._call_translate_api
    translate_module._call_translate_api = fake_call  # type: ignore[assignment]
    try:
        with pytest.raises(Exception) as excinfo:
            run_translate(ja, output, config_path=config_path, timeout_sec=30, input_video_path=video)
    finally:
        translate_module._call_translate_api = original  # type: ignore[assignment]

    assert "strict_ids" in str(excinfo.value)
    assert not output.exists()


# --------------------------------------------------------------------------- C.6 重试与幂等


def test_non_retryable_4xx_fails_immediately(monkeypatch, tmp_path: Path) -> None:
    """4xx 直接失败，不重试。"""

    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=["retry = 5", "request_interval = 0"],
    )
    ja = _make_srt(tmp_path / "sample.ja.srt", 1)
    output = tmp_path / "sample.zh.srt"
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")

    import whisper_stt_service.executor.translate as translate_module

    calls = {"n": 0}

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        error = requests.HTTPError("401 Unauthorized")
        error.response = FakeResponse(401)  # type: ignore[assignment]
        raise error

    original = translate_module._call_translate_api
    translate_module._call_translate_api = fake_call  # type: ignore[assignment]
    try:
        with pytest.raises(Exception):
            run_translate(ja, output, config_path=config_path, timeout_sec=30, input_video_path=video)
    finally:
        translate_module._call_translate_api = original  # type: ignore[assignment]

    assert calls["n"] == 1  # 只尝试一次


def test_retryable_5xx_retries_and_succeeds(monkeypatch, tmp_path: Path) -> None:
    """5xx 会重试，直到成功。"""

    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=["retry = 4", "request_interval = 0"],
    )
    ja = _make_srt(tmp_path / "sample.ja.srt", 2)
    output = tmp_path / "sample.zh.srt"
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")

    import whisper_stt_service.executor.translate as translate_module

    calls = {"n": 0}

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] < 3:
            error = requests.HTTPError("503 Service Unavailable")
            error.response = FakeResponse(503)  # type: ignore[assignment]
            raise error
        return {e.index: f"zh-{e.index}" for e in kwargs["batch"]}

    original = translate_module._call_translate_api
    translate_module._call_translate_api = fake_call  # type: ignore[assignment]
    try:
        run_translate(ja, output, config_path=config_path, timeout_sec=30, input_video_path=video)
    finally:
        translate_module._call_translate_api = original  # type: ignore[assignment]

    assert calls["n"] == 3
    assert "zh-1" in output.read_text(encoding="utf-8")


def test_idempotency_key_is_stable_hash() -> None:
    """幂等键 = sha256(job_id|batch_id|source_sha256)，同输入稳定。"""

    first = _build_idempotency_key("job-1", "0/0", "abc")
    second = _build_idempotency_key("job-1", "0/0", "abc")
    other = _build_idempotency_key("job-1", "0/1", "abc")

    assert first == second
    assert first != other
    assert len(first) == 64
    int(first, 16)  # 合法十六进制


def test_idempotency_header_sent_on_request() -> None:
    """真实 _call_translate_api 会带上 Idempotency-Key 头。"""

    from whisper_stt_service.executor.translate import _call_translate_api

    seen: dict = {}

    def post(url, *, headers=None, json=None, timeout=None):  # type: ignore[no-untyped-def]
        seen["headers"] = headers
        return FakeResponse(200, _ok_payload([1]))

    _call_translate_api(
        post,
        base_url="https://x/v1",
        api_key="k",
        model="m",
        timeout_sec=5,
        batch=[SrtEntry(index=1, timestamp="00:00:01,000 --> 00:00:02,000", text="a")],
        idempotency_key="deadbeef",
    )

    assert seen["headers"]["Idempotency-Key"] == "deadbeef"


# --------------------------------------------------------------------------- C.7 原子写


def test_atomic_write_replaces_content(tmp_path: Path) -> None:
    """atomic_write_text 写入后可读回，且不残留临时文件。"""

    target = tmp_path / "out.txt"
    atomic_write_text(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello"
    assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]


def test_atomic_write_keeps_old_file_on_failure(monkeypatch, tmp_path: Path) -> None:
    """写入失败时不破坏已有产物，也不留临时文件。"""

    target = tmp_path / "out.txt"
    target.write_text("original", encoding="utf-8")

    import whisper_stt_service.executor.atomic as atomic_module

    def boom(src, dst):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr(atomic_module.os, "replace", boom)

    with pytest.raises(OSError):
        atomic_write_text(target, "new-content")

    assert target.read_text(encoding="utf-8") == "original"
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "out.txt"]
    assert leftovers == []


def test_dump_srt_is_atomic_and_fills_missing_with_source(tmp_path: Path) -> None:
    """_dump_srt 缺失译文时回填原文，不丢行。"""

    entries = [
        SrtEntry(index=1, timestamp="00:00:01,000 --> 00:00:02,000", text="原文1"),
        SrtEntry(index=2, timestamp="00:00:03,000 --> 00:00:04,000", text="原文2"),
    ]
    out = tmp_path / "x.zh.srt"
    _dump_srt(entries, {1: "译文1"}, out)
    content = out.read_text(encoding="utf-8")
    assert "译文1" in content and "原文2" in content


# --------------------------------------------------------------------------- 取消/llm profile


def test_cancel_check_aborts_without_writing_output(
    monkeypatch, tmp_path: Path
) -> None:
    """cancel_check 返回 True 后中止，不写 zh.srt。"""

    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=["batch_size = 1", "retry = 1", "parallel = 1",
                           "request_interval = 0"],
    )
    ja = _make_srt(tmp_path / "sample.ja.srt", 5)
    output = tmp_path / "sample.zh.srt"
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")

    import whisper_stt_service.executor.translate as translate_module

    state = {"calls": 0}

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        state["calls"] += 1
        return {e.index: f"zh-{e.index}" for e in kwargs["batch"]}

    original = translate_module._call_translate_api
    translate_module._call_translate_api = fake_call  # type: ignore[assignment]
    try:
        with pytest.raises(TranslateCancelledError):
            run_translate(
                ja,
                output,
                config_path=config_path,
                timeout_sec=30,
                input_video_path=video,
                cancel_check=lambda: state["calls"] >= 2,
            )
    finally:
        translate_module._call_translate_api = original  # type: ignore[assignment]

    assert not output.exists()


def test_translate_cancelled_error_is_executor_cancelled() -> None:
    """翻译取消异常继承公共 ExecutorCancelledError，供 runtime 统一判断。"""

    assert issubclass(TranslateCancelledError, ExecutorCancelledError)
    assert ExecutorCancelledError.code == "executor_cancelled"


def test_missing_audio_artifact_error_carries_code() -> None:
    """wav 缺失错误码为 missing_audio_artifact，并携带路径。"""

    error = MissingAudioArtifactError("/tmp/a.16k.wav")
    assert error.code == "missing_audio_artifact"
    assert "/tmp/a.16k.wav" in str(error)


def test_4xx_including_429_is_never_retried(monkeypatch, tmp_path: Path) -> None:
    """SPEC C.6：429 属于 4xx，同样不重试。"""

    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=["retry = 5", "request_interval = 0"],
    )
    ja = _make_srt(tmp_path / "sample.ja.srt", 1)
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")

    import whisper_stt_service.executor.translate as translate_module

    calls = {"n": 0}

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        error = requests.HTTPError("429 Too Many Requests")
        error.response = FakeResponse(429)  # type: ignore[assignment]
        raise error

    monkeypatch.setattr(translate_module, "_call_translate_api", fake_call)

    with pytest.raises(Exception):
        run_translate(
            ja,
            tmp_path / "sample.zh.srt",
            config_path=config_path,
            timeout_sec=30,
            input_video_path=video,
        )

    assert calls["n"] == 1


def test_checkpoint_without_translations_is_not_reused(tmp_path: Path) -> None:
    """只有 done=true 但无译文内容的检查点必须重算。"""

    config_path = _write_config(tmp_path / "config.ini")
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    settings = resolve_translate_settings(cfg)

    from whisper_stt_service.executor.common import _sha256_file

    ja = _make_srt(tmp_path / "sample.ja.srt", 2)
    checkpoint = tmp_path / "sample.zh.srt.progress.json"
    checkpoint.write_text(
        json.dumps(
            {
                "version": 1,
                "source_sha256": _sha256_file(ja),
                "chunk_minutes": settings.chunk_minutes,
                "max_chars": settings.max_chars,
                "batch_size": settings.batch_size,
                "batches": {
                    "0/0": {"done": True, "ids": [1, 2], "translations": {}},
                    "0/1": {"done": True, "ids": [1, 2], "translations": {"1": "zh-1"}},
                },
            }
        ),
        encoding="utf-8",
    )

    restored = _load_checkpoint(
        checkpoint, source_sha256=_sha256_file(ja), settings=settings
    )

    assert restored == {}


def test_checkpoint_invalidated_when_batching_config_changes(tmp_path: Path) -> None:
    """切分参数（batch_size/max_chars/chunk_minutes）变化后不得复用。"""

    config_path = _write_config(tmp_path / "config.ini")
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    settings = resolve_translate_settings(cfg)

    from whisper_stt_service.executor.common import _sha256_file

    ja = _make_srt(tmp_path / "sample.ja.srt", 1)
    digest = _sha256_file(ja)
    checkpoint = tmp_path / "sample.zh.srt.progress.json"
    checkpoint.write_text(
        json.dumps(
            {
                "version": 1,
                "source_sha256": digest,
                "chunk_minutes": settings.chunk_minutes,
                "max_chars": settings.max_chars + 1,  # 与当前配置不符
                "batch_size": settings.batch_size,
                "batches": {
                    "0/0": {"done": True, "ids": [1], "translations": {"1": "zh-1"}}
                },
            }
        ),
        encoding="utf-8",
    )

    assert _load_checkpoint(checkpoint, source_sha256=digest, settings=settings) == {}


def test_idempotency_key_stable_across_task_retries() -> None:
    """幂等键基于 job_id：同一 job 的不同 attempt 必须得到同一键。"""

    first_attempt = _build_idempotency_key("job-abc", "0/0", "src")
    second_attempt = _build_idempotency_key("job-abc", "0/0", "src")
    assert first_attempt == second_attempt


def test_copy_back_refuses_symlink_target(monkeypatch, tmp_path: Path, caplog) -> None:
    """回写目标为符号链接时跳过，避免写到白名单之外。"""

    config_path = _write_config(tmp_path / "config.ini", translation_lines=["retry = 1"])
    ja = _make_srt(tmp_path / "sample.ja.srt", 1)
    output = tmp_path / "sample.zh.srt"
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")

    copy_dir = tmp_path / "copy-back"
    copy_dir.mkdir()
    outside = tmp_path / "outside.zh.srt"
    outside.write_text("keep", encoding="utf-8")
    (copy_dir / "sample.zh.srt").symlink_to(outside)

    import whisper_stt_service.executor.translate as translate_module

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        return {e.index: f"zh-{e.index}" for e in kwargs["batch"]}

    monkeypatch.setattr(translate_module, "_call_translate_api", fake_call)
    caplog.set_level("WARNING")

    run_translate(
        ja,
        output,
        config_path=config_path,
        timeout_sec=30,
        input_video_path=video,
        copy_back=str(copy_dir),
    )

    # 外部文件不被覆盖，并记录告警
    assert outside.read_text(encoding="utf-8") == "keep"
    assert "symlink" in caplog.text


def test_load_llm_profiles_picks_fallback_and_dedupes(tmp_path: Path) -> None:
    """llm_fallback 启用时纳入候选，并去掉完全重复的配置。"""

    config_path = _write_config(tmp_path / "config.ini", fallback_enabled=True)
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + "\n[llm_backup]\nbase_url = https://primary.example/v1\n"
        + "api_key = primary-key\nmodel = primary-model\n",
        encoding="utf-8",
    )
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")

    profiles = _load_llm_profiles(cfg)

    assert [p["section"] for p in profiles] == ["llm", "llm_fallback"]


def test_load_llm_profiles_skips_disabled_fallback(tmp_path: Path) -> None:
    """enabled=false 的回退段不被纳入。"""

    config_path = _write_config(tmp_path / "config.ini", fallback_enabled=False)
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")

    assert [p["section"] for p in _load_llm_profiles(cfg)] == ["llm"]


def test_pick_llm_profile_uses_fallback_from_attempt_two() -> None:
    """首轮用主模型，第 2 次起使用回退模型。"""

    profiles = [
        {"section": "llm", "model": "primary"},
        {"section": "llm_fallback", "model": "fallback"},
    ]

    assert _pick_llm_profile(profiles, 1)["section"] == "llm"
    assert _pick_llm_profile(profiles, 2)["section"] == "llm_fallback"
    assert _pick_llm_profile(profiles, 3)["section"] == "llm_fallback"


def test_pick_llm_profile_single_profile_always_used() -> None:
    """只有一个候选时任何轮次都用它。"""

    profiles = [{"section": "llm", "model": "primary"}]
    assert _pick_llm_profile(profiles, 5)["model"] == "primary"


def test_fallback_profile_used_on_second_attempt(monkeypatch, tmp_path: Path) -> None:
    """端到端：首批 5xx 后第 2 次尝试切换到回退模型，并记录 attempt 日志。"""

    config_path = _write_config(
        tmp_path / "config.ini",
        fallback_enabled=True,
        translation_lines=["retry = 3", "request_interval = 0"],
    )
    ja = _make_srt(tmp_path / "sample.ja.srt", 1)
    output = tmp_path / "sample.zh.srt"
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")

    import whisper_stt_service.executor.translate as translate_module

    models: list[str] = []

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        models.append(kwargs["model"])
        if len(models) == 1:
            error = requests.HTTPError("500 boom")
            error.response = FakeResponse(500)  # type: ignore[assignment]
            raise error
        return {e.index: f"zh-{e.index}" for e in kwargs["batch"]}

    original = translate_module._call_translate_api
    translate_module._call_translate_api = fake_call  # type: ignore[assignment]
    try:
        run_translate(ja, output, config_path=config_path, timeout_sec=30, input_video_path=video)
    finally:
        translate_module._call_translate_api = original  # type: ignore[assignment]

    assert models == ["primary-model", "fallback-model"]
    assert "zh-1" in output.read_text(encoding="utf-8")

# ------------------------------------------------------- 产物级 fencing（P0）


def test_atomic_write_fenced_rejects_stale_writer(tmp_path: Path) -> None:
    """publish 回调返回 False（租约已失）时不写盘并抛 StaleWriteRejected。"""

    target = tmp_path / "x.zh.srt"
    target.write_text("OLD", encoding="utf-8")
    seen: dict = {}

    def publish(path, text):  # type: ignore[no-untyped-def]
        seen["path"] = path
        seen["text"] = text
        return False

    with pytest.raises(StaleWriteRejected):
        atomic_write_text_fenced(target, "NEW", publish)

    assert seen["text"] == "NEW"
    assert target.read_text(encoding="utf-8") == "OLD"  # 正式产物未被覆盖


def test_atomic_write_fenced_accepts_fresh_writer(tmp_path: Path) -> None:
    """publish 返回 True 表示回调已全权落盘，helper 不再重写。"""

    target = tmp_path / "x.zh.srt"

    def publish(path, text):  # type: ignore[no-untyped-def]
        atomic_write_text(path, text)
        return True

    atomic_write_text_fenced(target, "NEW", publish)
    assert target.read_text(encoding="utf-8") == "NEW"


def test_atomic_write_fenced_accepted_callback_owns_persistence(tmp_path: Path) -> None:
    """R3-3 单元回归：回调返回 True 后 helper 不得在锁外补写一次。

    回调只返回 True（模拟“已在租约事务内完成落盘”），helper 若仍无条件
    `atomic_write_text`，就会在事务提交后覆盖新 owner 的内容。这里用“回调不写、
    只有旧文件”证明 helper 没有第二次落盘。
    """

    target = tmp_path / "x.zh.srt"
    target.write_text("OLD", encoding="utf-8")

    atomic_write_text_fenced(target, "STALE", lambda _p, _t: True)

    assert target.read_text(encoding="utf-8") == "OLD"


def test_atomic_write_fenced_rejects_non_bool_publish_result(tmp_path: Path) -> None:
    """publish 返回值不是 bool 时立即报错，绝不绕过租约在锁外落盘。"""

    target = tmp_path / "x.zh.srt"
    target.write_text("OLD", encoding="utf-8")

    with pytest.raises(TypeError):
        atomic_write_text_fenced(target, "NEW", lambda _p, _t: None)

    assert target.read_text(encoding="utf-8") == "OLD"


def test_atomic_write_fenced_without_publish_is_plain_write(tmp_path: Path) -> None:
    """未提供 publish 时退化为普通原子写（向后兼容）。"""

    target = tmp_path / "x.zh.srt"
    atomic_write_text_fenced(target, "PLAIN", None)
    assert target.read_text(encoding="utf-8") == "PLAIN"


def test_checkpoint_save_uses_fenced_publish(tmp_path: Path) -> None:
    """_save_checkpoint 把 publish 透传到 fenced 写；拒绝时不生成检查点。"""

    path = tmp_path / "c.progress.json"
    rejected = {"n": 0}

    def reject(_p, _t):  # type: ignore[no-untyped-def]
        rejected["n"] += 1
        return False

    with pytest.raises(StaleWriteRejected):
        _save_checkpoint(path, {"version": 1}, publish=reject)
    assert rejected["n"] == 1
    assert not path.exists()


def test_run_translate_uses_explicit_checkpoint_path(
    monkeypatch, tmp_path: Path
) -> None:
    """checkpoint_path 生效：检查点写到指定路径而非 <output>.progress.json。"""

    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=["retry = 1", "request_interval = 0"],
    )
    ja = _make_srt(tmp_path / "sample.ja.srt", 2)
    staging_out = tmp_path / "staging" / "sample.zh.srt"
    canonical_ckpt = tmp_path / "canonical" / "sample.zh.srt.progress.json"
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        return {e.index: f"zh-{e.index}" for e in kwargs["batch"]}

    monkeypatch.setattr(
        "whisper_stt_service.executor.translate._call_translate_api", fake_call
    )
    run_translate(
        ja,
        staging_out,
        config_path=config_path,
        timeout_sec=30,
        input_video_path=video,
        checkpoint_path=canonical_ckpt,
        defer_copy_back=True,
    )

    assert canonical_ckpt.is_file()
    assert not staging_out.with_suffix(staging_out.suffix + ".progress.json").exists()


def test_run_translate_resumes_from_canonical_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    """第二次 attempt 用不同 staging 输出但共用 canonical 检查点 → 0 请求续跑。"""

    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=["retry = 1", "request_interval = 0"],
    )
    ja = _make_srt(tmp_path / "sample.ja.srt", 3)
    canonical_ckpt = tmp_path / "canonical" / "sample.zh.srt.progress.json"
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        return {e.index: f"zh-{e.index}" for e in kwargs["batch"]}

    monkeypatch.setattr(
        "whisper_stt_service.executor.translate._call_translate_api", fake_call
    )
    run_translate(
        ja,
        tmp_path / "staging" / "attempt1" / "sample.zh.srt",
        config_path=config_path,
        timeout_sec=30,
        input_video_path=video,
        checkpoint_path=canonical_ckpt,
        defer_copy_back=True,
    )

    calls = {"n": 0}

    def forbidden(post, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        raise AssertionError("不应重新请求上游")

    monkeypatch.setattr(
        "whisper_stt_service.executor.translate._call_translate_api", forbidden
    )
    attempt2 = tmp_path / "staging" / "attempt2" / "sample.zh.srt"
    run_translate(
        ja,
        attempt2,
        config_path=config_path,
        timeout_sec=30,
        input_video_path=video,
        checkpoint_path=canonical_ckpt,
        defer_copy_back=True,
    )

    assert calls["n"] == 0
    assert "zh-1" in attempt2.read_text(encoding="utf-8")


def test_run_translate_defer_copy_back_skips_writeback(
    monkeypatch, tmp_path: Path
) -> None:
    """defer_copy_back=True 时不得提前回写（由 runtime promotion 负责）。"""

    copy_dir = tmp_path / "copy-back"
    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=[
            "retry = 1",
            "request_interval = 0",
            f"copy_back = {copy_dir}",
        ],
    )
    ja = _make_srt(tmp_path / "sample.ja.srt", 1)
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video = video_dir / "sample.mp4"
    video.write_bytes(b"x")

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        return {e.index: f"zh-{e.index}" for e in kwargs["batch"]}

    monkeypatch.setattr(
        "whisper_stt_service.executor.translate._call_translate_api", fake_call
    )
    output = tmp_path / "staging" / "sample.zh.srt"
    run_translate(
        ja,
        output,
        config_path=config_path,
        timeout_sec=30,
        input_video_path=video,
        defer_copy_back=True,
    )

    assert output.is_file()
    assert not copy_dir.exists()  # 未回写配置目录
    assert list(video_dir.iterdir()) == [video]  # 未污染视频目录


def test_run_translate_default_copy_back_still_works(
    monkeypatch, tmp_path: Path
) -> None:
    """不传 defer_copy_back 时保持旧行为：正常回写。"""

    copy_dir = tmp_path / "copy-back"
    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=[
            "retry = 1",
            "request_interval = 0",
            f"copy_back = {copy_dir}",
        ],
    )
    ja = _make_srt(tmp_path / "sample.ja.srt", 1)
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        return {e.index: f"zh-{e.index}" for e in kwargs["batch"]}

    monkeypatch.setattr(
        "whisper_stt_service.executor.translate._call_translate_api", fake_call
    )
    run_translate(
        ja,
        tmp_path / "out" / "sample.zh.srt",
        config_path=config_path,
        timeout_sec=30,
        input_video_path=video,
    )

    assert (copy_dir / "sample.zh.srt").is_file()


def test_run_translate_fenced_checkpoint_rejection_aborts(
    monkeypatch, tmp_path: Path
) -> None:
    """checkpoint_publish 返回 False 时整次运行中止，不产出正式 zh.srt。"""

    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=["batch_size = 1", "retry = 1", "request_interval = 0"],
    )
    ja = _make_srt(tmp_path / "sample.ja.srt", 3)
    output = tmp_path / "sample.zh.srt"
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        return {e.index: f"zh-{e.index}" for e in kwargs["batch"]}

    def deny(_p, _t):  # type: ignore[no-untyped-def]
        return False

    monkeypatch.setattr(
        "whisper_stt_service.executor.translate._call_translate_api", fake_call
    )
    ckpt = tmp_path / "ckpt" / "c.json"
    with pytest.raises((ExecutorCancelledError, TranslateCancelledError)):
        run_translate(
            ja,
            output,
            config_path=config_path,
            timeout_sec=30,
            input_video_path=video,
            checkpoint_path=ckpt,
            checkpoint_publish=deny,
            defer_copy_back=True,
        )

    assert not output.exists()
    assert not ckpt.exists()


def test_run_translate_cancel_check_blocks_final_output(
    monkeypatch, tmp_path: Path
) -> None:
    """写正式产物前的取消检查：失租后不覆盖已有 zh.srt。"""

    config_path = _write_config(
        tmp_path / "config.ini", translation_lines=["retry = 1", "request_interval = 0"]
    )
    ja = _make_srt(tmp_path / "sample.ja.srt", 2)
    output = tmp_path / "sample.zh.srt"
    output.write_text("OFFICIAL-OLD", encoding="utf-8")
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")
    state = {"done": False}

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        state["done"] = True  # 翻译完成后立刻失租
        return {e.index: f"zh-{e.index}" for e in kwargs["batch"]}

    monkeypatch.setattr(
        "whisper_stt_service.executor.translate._call_translate_api", fake_call
    )
    with pytest.raises(TranslateCancelledError):
        run_translate(
            ja,
            output,
            config_path=config_path,
            timeout_sec=30,
            input_video_path=video,
            cancel_check=lambda: state["done"],
            defer_copy_back=True,
        )

    assert output.read_text(encoding="utf-8") == "OFFICIAL-OLD"

# -------------------------------- P1 回归：真实 _call_translate_api 路径（非 mock）


def _http_ok(ids: list[int]) -> FakeResponse:
    """构造只返回给定 id 的正常 HTTP 响应（真实 _call_translate_api 会解析它）。"""

    return FakeResponse(200, _ok_payload(ids))


def _requested_ids(kwargs: dict) -> list[int]:
    """从请求体反解本次实际请求的字幕 id（验证"只补缺失"）。"""

    payload = kwargs["json"]["messages"][1]["content"]
    return [item["id"] for item in json.loads(payload.split("\n\n", 1)[1])]


def test_real_api_partial_subset_keeps_translated_and_requests_only_missing(
    tmp_path: Path,
) -> None:
    """C5 真实路径：首次只返回 id1，第二轮只请求缺失的 id2，且保留 id1 译文。

    这里**不 mock** `_call_translate_api`，而是通过 post_func 提供 HTTP 响应，
    确保"接受子集 + 只补缺失"走的是生产代码路径。
    """

    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=["batch_size = 10", "retry = 3", "request_interval = 0"],
    )
    ja = tmp_path / "sample.ja.srt"
    ja.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n源一\n\n2\n00:00:03,000 --> 00:00:04,000\n源二\n\n",
        encoding="utf-8",
    )
    output = tmp_path / "sample.zh.srt"
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")

    requested: list[list[int]] = []
    lock = threading.Lock()

    def post(url, **kwargs):  # type: ignore[no-untyped-def]
        ids = _requested_ids(kwargs)
        with lock:
            requested.append(ids)
            first = len(requested) == 1
        # 首次只返回 id1；补缺失时返回本轮请求的全部 id
        return _http_ok([1]) if first else _http_ok(ids)

    run_translate(
        ja,
        output,
        config_path=config_path,
        timeout_sec=30,
        input_video_path=video,
        post_func=post,
        defer_copy_back=True,
    )

    assert requested[0] == [1, 2]
    assert requested[1] == [2]  # 只补缺失，不整批重发
    text = output.read_text(encoding="utf-8")
    assert "zh-1" in text  # 首轮已返回的译文被保留
    assert "zh-2" in text


def test_real_api_partial_exhausted_keeps_source_text(
    tmp_path: Path, caplog
) -> None:
    """C5 真实路径：每轮都只返回 id1，最终 id2 回退日文原文且不丢行。"""

    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=["batch_size = 10", "retry = 2", "request_interval = 0"],
    )
    ja = tmp_path / "sample.ja.srt"
    ja.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n源一\n\n2\n00:00:03,000 --> 00:00:04,000\n源二\n\n",
        encoding="utf-8",
    )
    output = tmp_path / "sample.zh.srt"
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")
    caplog.set_level("WARNING")

    def post(url, **kwargs):  # type: ignore[no-untyped-def]
        return _http_ok([1])  # 永远缺 id2

    run_translate(
        ja,
        output,
        config_path=config_path,
        timeout_sec=30,
        input_video_path=video,
        post_func=post,
        defer_copy_back=True,
    )

    blocks = [b for b in output.read_text(encoding="utf-8").split("\n\n") if b.strip()]
    assert len(blocks) == 2  # 两条都在，不丢行
    assert "zh-1" in blocks[0]
    assert "源二" in blocks[1]  # 缺失 id 回退日文原文
    assert "translate_partial" in caplog.text


def test_partial_subset_uses_distinct_idempotency_key() -> None:
    """补缺失批次 payload 已变，幂等键必须与原批不同（否则命中上游旧响应）。"""

    full = _build_idempotency_key("job", "0/0", "src")
    subset_id = _sub_batch_id("0/0", [2])
    subset = _build_idempotency_key("job", subset_id, "src")

    assert full != subset
    # 同一子集重复请求 key 稳定
    assert subset == _build_idempotency_key("job", _sub_batch_id("0/0", [2]), "src")
    # 完整批次沿用 SPEC 的 batch_id（不含 #missing 后缀）
    assert "#missing" not in "0/0"


def test_inflight_success_after_failure_still_commits_checkpoint(
    tmp_path: Path,
) -> None:
    """C1：批 A 失败后，在途批 B 成功**仍必须落检查点**，重试不得重发 B。

    确定性时序：B 先拿到译文但延迟返回，A 在此期间失败并标记 abort。
    """

    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=[
            "batch_size = 1",
            "parallel = 2",
            "retry = 1",
            "request_interval = 0",
        ],
    )
    ja = _make_srt(tmp_path / "sample.ja.srt", 2)
    output = tmp_path / "sample.zh.srt"
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")

    def post(url, **kwargs):  # type: ignore[no-untyped-def]
        idx = _requested_ids(kwargs)[0]
        if idx == 1:
            time.sleep(0.05)
            raise requests.ConnectionError("boom-A")
        time.sleep(0.20)  # B 晚于 A 的失败提交返回
        return _http_ok([2])

    with pytest.raises(Exception):
        run_translate(
            ja,
            output,
            config_path=config_path,
            timeout_sec=30,
            input_video_path=video,
            post_func=post,
            defer_copy_back=True,
        )

    checkpoint = output.with_suffix(output.suffix + ".progress.json")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    states = payload["batches"]
    # A 失败、B 成功且**已持久化**
    assert states["0/0"]["done"] is False
    assert states["0/1"]["done"] is True
    assert states["0/1"]["translations"] == {"2": "zh-2"}

    # 重跑：只应补 A，不再请求 B
    retried: list[int] = []

    def post_retry(url, **kwargs):  # type: ignore[no-untyped-def]
        ids = _requested_ids(kwargs)
        retried.extend(ids)
        return _http_ok(ids)

    run_translate(
        ja,
        output,
        config_path=config_path,
        timeout_sec=30,
        input_video_path=video,
        post_func=post_retry,
        defer_copy_back=True,
    )
    assert retried == [1]  # 只补未完成的 A


# -------------------------------- R3-3：真实 SQLite 下的锁外二写回归


def test_fenced_checkpoint_cannot_overwrite_new_owner_after_publish_commit(
    monkeypatch, tmp_path: Path
) -> None:
    """R3-3 真 SQLite 回归：w1 回调提交 -> 回收/新 owner 写 NEW -> helper 不得覆盖。

    时序在 publish 回调内部确定性构造：w1 在真实租约事务内写完检查点并提交，
    随后租约过期、被 reaper 回收、w2 领取 attempt=1 并写下 NEW OWNER 检查点，
    最后才返回到 w1 的 helper。修复后的 helper 在回调返回 True 后不得再写，
    因此 canonical 文件必须保持 NEW；旧实现会在锁外补写并把 NEW 覆盖成 w1 内容。
    """

    db = Database(tmp_path / "q.db")
    db.init_schema()
    repo = JobRepository(
        db, log_root=tmp_path / "logs", artifact_root=tmp_path / "artifacts"
    )
    repo.enqueue("/tmp/r3-3.mp4", "ja")
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is not None and claimed.attempt == 0

    config_path = _write_config(
        tmp_path / "config.ini",
        translation_lines=["batch_size = 1", "retry = 1", "request_interval = 0"],
    )
    ja = _make_srt(tmp_path / "sample.ja.srt", 2)
    output = tmp_path / "sample.zh.srt"
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x")
    canonical = tmp_path / "canonical" / "sample.zh.srt.progress.json"
    # 用非引擎结构的哨兵内容：任何一次 w1 补写都会让断言立刻变成失败。
    new_owner_payload = '{"owner":"NEW w2","version":1,"source_sha256":"new","batches":{}}'
    state = {"publishes": 0, "hijacked": False}

    def publish(path: Path, text: str) -> bool:
        """真实租约事务内落盘；首次提交后模拟回收并由新 owner 写 NEW。"""

        state["publishes"] += 1
        accepted = repo.with_task_lease(
            claimed.task_id,
            worker_id="w1",
            attempt=claimed.attempt,
            action=lambda: atomic_write_text(path, text),
        )
        if accepted and not state["hijacked"]:
            state["hijacked"] = True
            # w1 的事务已提交（租约锁已释放）：租约过期 -> reaper 回收 -> w2 接管。
            with repo.db.tx() as conn:
                conn.execute(
                    "UPDATE tasks SET lease_expires_at=? WHERE task_id=?",
                    ("2000-01-01T00:00:00+00:00", claimed.task_id),
                )
            assert repo.reclaim_expired_leases()["reclaimed"] == 1
            second = repo.claim_next("extract", "w2", 60)
            assert second is not None and second.attempt == 1
            # 新 owner 在自己的租约事务内写 canonical 检查点。
            assert (
                repo.with_task_lease(
                    claimed.task_id,
                    worker_id="w2",
                    attempt=second.attempt,
                    action=lambda: atomic_write_text(canonical, new_owner_payload),
                )
                is True
            )
        return accepted

    def fake_call(post, **kwargs):  # type: ignore[no-untyped-def]
        return {e.index: f"zh-{e.index}" for e in kwargs["batch"]}

    monkeypatch.setattr(
        "whisper_stt_service.executor.translate._call_translate_api", fake_call
    )

    with pytest.raises((TranslateCancelledError, ExecutorCancelledError)):
        run_translate(
            ja,
            output,
            config_path=config_path,
            timeout_sec=30,
            input_video_path=video,
            checkpoint_path=canonical,
            checkpoint_publish=publish,
            defer_copy_back=True,
        )

    assert state["hijacked"] is True
    # 关键断言走真实文件内容（而非 mock 调用次数）：NEW 不得被 w1 helper 覆盖。
    assert canonical.read_text(encoding="utf-8") == new_owner_payload
    assert not output.exists()
