"""CLI 翻译（whisper_stt/translate_srt_ja_to_zh.py）加固回归测试。

覆盖：CLI 默认委托统一引擎、llm_overrides 仅内存传递、旧结构 checkpoint 安全失效、
原子输出、batch_size 生效、4xx 不重试、部分 id 保留原文。
全部使用 fake HTTP（post_func / monkeypatch requests），不触网。
"""

from __future__ import annotations

import configparser
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from whisper_stt_service.executor.translate import TranslateNonRetryableError

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "whisper_stt"

WAV_UNUSED = None  # 占位，避免误用音频常量


def _load_cli():
    """按路径加载 CLI 模块（whisper_stt/ 不是可导入包）。"""

    name = "deck_translate_cli"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        name, SCRIPT_DIR / "translate_srt_ja_to_zh.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli():
    """加载 CLI 模块。"""

    return _load_cli()


class FakeResponse:
    """最小化 requests.Response 替身。"""

    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        """保存状态码与响应体。"""

        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        """模拟 requests 的 HTTP 错误抛出。"""

        if self.status_code >= 400:
            error = requests.HTTPError(f"status={self.status_code}")
            error.response = self  # type: ignore[assignment]
            raise error

    def json(self) -> dict:
        """返回预置响应体。"""

        return self._payload


def _ok_payload(ids: list[int]) -> dict:
    """构造只返回给定 id 的正常响应。"""

    items = [{"id": i, "text_zh": f"zh-{i}"} for i in ids]
    return {
        "choices": [{"message": {"content": json.dumps(items, ensure_ascii=False)}}]
    }


def _write_config(
    tmp_path: Path, *, overrides: dict[str, str] | None = None
) -> Path:
    """生成 CLI 用 config.ini（fake 凭据，仅本地）。

    用 ConfigParser.set 覆盖默认键：同一配置项只保留一个值，避免 append
    造成 DuplicateOptionError（fixture 基础配置已有 retry/chunk_minutes 等键）。
    """

    parser = configparser.ConfigParser()
    parser["llm"] = {
        "base_url": "https://fake.example/v1",
        "api_key": "fake-key",
        "model": "fake-model",
    }
    parser["translation"] = {
        "chunk_minutes": "1",
        "retry": "3",
        "request_interval": "0",
        "disable_proxy": "true",
    }
    for key, value in (overrides or {}).items():
        parser.set("translation", key, str(value))
    cfg = tmp_path / "config.ini"
    with cfg.open("w", encoding="utf-8") as handle:
        parser.write(handle)
    return cfg


def _write_srt(tmp_path: Path, count: int = 2) -> Path:
    """生成 ja.srt。"""

    blocks = [f"{i}\n00:00:0{i},000 --> 00:00:0{i},900\n源{i}\n" for i in range(1, count + 1)]
    path = tmp_path / "sample.ja.srt"
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def _args(cli, tmp_path: Path, cfg: Path, ja: Path, **overrides):
    """构造 CLI 参数命名空间。"""

    values = dict(
        config=cfg,
        input=ja,
        output=tmp_path / "sample.zh.srt",
        base_url=None,
        api_key=None,
        model=None,
        batch_size=None,
        chunk_minutes=None,
        timeout=None,
        retry=None,
        sleep=None,
        parallel=None,
        request_interval=None,
        disable_proxy=None,
        engine="unified",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _install_fake_http(cli, monkeypatch, handler) -> None:
    """把 CLI 模块内的 requests.post/Session.post 替换为 fake。"""

    class FakeSession:
        trust_env = True

        def post(self, url, **kwargs):  # type: ignore[no-untyped-def]
            return handler(url, **kwargs)

    monkeypatch.setattr(cli.requests, "post", handler)
    monkeypatch.setattr(cli.requests, "Session", FakeSession)


def test_cli_default_engine_is_unified(cli, tmp_path: Path, monkeypatch) -> None:
    """CLI 默认走统一引擎并产出 zh.srt（fake HTTP）。"""

    cfg = _write_config(tmp_path)
    ja = _write_srt(tmp_path, 2)
    args = _args(cli, tmp_path, cfg, ja)

    def handler(url, **kwargs):  # type: ignore[no-untyped-def]
        payload = kwargs["json"]["messages"][1]["content"]
        ids = [item["id"] for item in json.loads(payload.split("\n\n", 1)[1])]
        return FakeResponse(200, _ok_payload(ids))

    _install_fake_http(cli, monkeypatch, handler)

    assert cli._run_unified_engine(
        args, repo_root=tmp_path, config_path=cfg, config=cli._load_config(cfg)
    ) == 0

    text = args.output.read_text(encoding="utf-8")
    assert "zh-1" in text and "zh-2" in text
    # 原子输出：同目录不再残留任何 .tmp- 临时文件。
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp-")] == []
    # 检查点为统一引擎结构
    checkpoint = args.output.with_suffix(args.output.suffix + ".progress.json")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert {"version", "source_sha256", "batches"} <= set(payload.keys())


def test_cli_legacy_checkpoint_is_invalidated(cli, tmp_path: Path, monkeypatch) -> None:
    """旧结构 {id: text} 检查点必须被删除，不得复用旧视频译文。"""

    cfg = _write_config(tmp_path)
    ja = _write_srt(tmp_path, 2)
    args = _args(cli, tmp_path, cfg, ja)
    legacy = args.output.with_suffix(args.output.suffix + ".progress.json")
    legacy.write_text(json.dumps({"1": "旧视频的译文"}), encoding="utf-8")

    def handler(url, **kwargs):  # type: ignore[no-untyped-def]
        payload = kwargs["json"]["messages"][1]["content"]
        ids = [item["id"] for item in json.loads(payload.split("\n\n", 1)[1])]
        return FakeResponse(200, _ok_payload(ids))

    _install_fake_http(cli, monkeypatch, handler)
    cli._run_unified_engine(
        args, repo_root=tmp_path, config_path=cfg, config=cli._load_config(cfg)
    )

    text = args.output.read_text(encoding="utf-8")
    assert "旧视频的译文" not in text
    assert "zh-1" in text


def test_invalidate_legacy_checkpoint_keeps_engine_format(
    cli, tmp_path: Path
) -> None:
    """已是引擎结构的检查点不被删除（交由引擎校验 source）。"""

    engine_ckpt = tmp_path / "x.progress.json"
    engine_ckpt.write_text(
        json.dumps({"version": 1, "source_sha256": "abc", "batches": {}}),
        encoding="utf-8",
    )
    assert cli._invalidate_legacy_checkpoint(engine_ckpt) is False
    assert engine_ckpt.is_file()


def test_cli_4xx_is_not_retried(cli, tmp_path: Path, monkeypatch) -> None:
    """4xx（鉴权/参数）精确失败、不重试，只请求一次且带幂等键。"""

    # 通过 set 覆盖 retry（而非重复 append）：确认即便 retry=5 也不会重试 401。
    cfg = _write_config(tmp_path, overrides={"retry": "5"})
    ja = _write_srt(tmp_path, 1)
    args = _args(cli, tmp_path, cfg, ja)
    calls = {"n": 0}
    seen_headers: list[dict] = []

    def handler(url, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        seen_headers.append(dict(kwargs["headers"]))
        return FakeResponse(401, {})

    _install_fake_http(cli, monkeypatch, handler)
    with pytest.raises(TranslateNonRetryableError) as excinfo:
        cli._run_unified_engine(
            args, repo_root=tmp_path, config_path=cfg, config=cli._load_config(cfg)
        )

    # 必须是精确的 4xx 失败（而不是被 fixture 配置错误掩盖的任意 Exception）。
    cause = excinfo.value.__cause__
    assert isinstance(cause, requests.HTTPError)
    assert cause.response is not None and cause.response.status_code == 401
    assert calls["n"] == 1  # 4xx 不进入重试
    assert seen_headers and seen_headers[0].get("Idempotency-Key")


def test_cli_batch_size_limit_is_effective(cli, tmp_path: Path, monkeypatch) -> None:
    """--batch-size 通过 job_config 生效：每批条目数不超过该值。"""

    cfg = _write_config(tmp_path, overrides={"batch_size": "1"})
    ja = _write_srt(tmp_path, 3)
    args = _args(cli, tmp_path, cfg, ja)
    requested: list[list[int]] = []

    def handler(url, **kwargs):  # type: ignore[no-untyped-def]
        payload = kwargs["json"]["messages"][1]["content"]
        ids = [item["id"] for item in json.loads(payload.split("\n\n", 1)[1])]
        requested.append(ids)
        return FakeResponse(200, _ok_payload(ids))

    _install_fake_http(cli, monkeypatch, handler)
    cli._run_unified_engine(
        args, repo_root=tmp_path, config_path=cfg, config=cli._load_config(cfg)
    )

    assert all(len(ids) <= 1 for ids in requested)
    assert sorted(i for ids in requested for i in ids) == [1, 2, 3]


def test_cli_partial_ids_keep_source_text(
    cli, tmp_path: Path, monkeypatch, capsys
) -> None:
    """partial 走真实 main/HTTP：保留已得译文与日文原文，不丢行、不退回 legacy。"""

    cfg = _write_config(tmp_path, overrides={"batch_size": "10", "retry": "2"})
    ja = _write_srt(tmp_path, 2)
    output = tmp_path / "sample.zh.srt"
    calls = {"n": 0}

    def handler(url, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return FakeResponse(200, _ok_payload([1]))  # 永远缺 id2

    _install_fake_http(cli, monkeypatch, handler)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "translate_srt_ja_to_zh.py",
            "--config",
            str(cfg),
            "--input",
            str(ja),
            "--output",
            str(output),
        ],
    )

    assert cli.main() == 0
    # 默认 main 必须走统一引擎，不允许静默退回 legacy。
    assert "engine=unified" in capsys.readouterr().out
    assert calls["n"] == 2  # 首轮 + 只补缺失 id2 的一轮

    blocks = [b for b in output.read_text(encoding="utf-8").split("\n\n") if b.strip()]
    assert len(blocks) == 2
    assert "zh-1" in blocks[0]  # 已返回的译文保留
    assert "源2" in blocks[1]  # 缺失 id 保留日文原文，不丢行


def test_cli_llm_overrides_are_memory_only(cli, tmp_path: Path, monkeypatch) -> None:
    """--api-key 等覆盖只在内存生效，不产生任何临时配置文件。"""

    cfg = _write_config(tmp_path)
    ja = _write_srt(tmp_path, 1)
    args = _args(
        cli, tmp_path, cfg, ja, api_key="override-key", model="override-model"
    )
    seen: dict = {}

    def handler(url, **kwargs):  # type: ignore[no-untyped-def]
        seen["auth"] = kwargs["headers"]["Authorization"]
        seen["model"] = kwargs["json"]["model"]
        payload = kwargs["json"]["messages"][1]["content"]
        ids = [item["id"] for item in json.loads(payload.split("\n\n", 1)[1])]
        return FakeResponse(200, _ok_payload(ids))

    _install_fake_http(cli, monkeypatch, handler)
    cli._run_unified_engine(
        args, repo_root=tmp_path, config_path=cfg, config=cli._load_config(cfg)
    )

    assert seen["auth"] == "Bearer override-key"
    assert seen["model"] == "override-model"
    # 目录中不应出现任何 secret 临时文件
    leftovers = [
        p.name
        for p in tmp_path.iterdir()
        if "config" in p.name and p.name != "config.ini"
    ]
    assert leftovers == []
    # 覆盖用的凭据只能存在于内存：任何落盘文件都不得包含 override-key。
    for path in tmp_path.iterdir():
        if path.is_file():
            assert "override-key" not in path.read_text(
                encoding="utf-8", errors="ignore"
            )


def test_cli_legacy_engine_still_selectable(cli, tmp_path: Path, monkeypatch) -> None:
    """--engine legacy 仍可运行旧实现（兼容保留）。"""

    cfg = _write_config(tmp_path)
    ja = _write_srt(tmp_path, 1)
    args = _args(cli, tmp_path, cfg, ja, engine="legacy")

    def handler(url, **kwargs):  # type: ignore[no-untyped-def]
        payload = kwargs["json"]["messages"][1]["content"]
        ids = [item["id"] for item in json.loads(payload.split("\n\n", 1)[1])]
        return FakeResponse(200, _ok_payload(ids))

    _install_fake_http(cli, monkeypatch, handler)
    cli._run_legacy_engine(
        args, repo_root=tmp_path, config_path=cfg, config=cli._load_config(cfg)
    )

    assert "zh-1" in args.output.read_text(encoding="utf-8")
