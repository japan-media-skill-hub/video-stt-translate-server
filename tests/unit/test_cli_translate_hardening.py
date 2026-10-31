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

# ---------------- 对齐契约：脚本模式与服务模式同一份"以 id 为 key 的对象"契约
#
# (A) 真实故障：模型把一条日文拆成两条中文、又丢掉最后一条，但返回的 id 仍连续
# （84..107），旧数组契约只校验"id 集合相等"就放行，整段译文后移一位。
# 脚本模式（包括 --engine legacy 的排障路径）必须与服务端同一契约，否则用脚本
# 翻译出来的字幕照样错位。下面用 fake HTTP 走真实 main/legacy 路径断言三条：
# 新 dict 格式可用、少 key 只补缺失、多 key 不落盘。


def _dict_response(mapping: dict[str, str]) -> FakeResponse:
    """构造"模型返回 {id: 译文} 对象"的 200 响应（新契约）。"""

    content = json.dumps(mapping, ensure_ascii=False)
    return FakeResponse(200, {"choices": [{"message": {"content": content}}]})


def _requested_ids(kwargs: dict) -> list[int]:
    """从请求体反解本次实际请求的字幕 id（验证"只补缺失"）。"""

    payload = kwargs["json"]["messages"][1]["content"]
    return [item["id"] for item in json.loads(payload.split("\n\n", 1)[1])]


def _zh_blocks(path: Path) -> list[str]:
    """把 zh.srt 切成非空块，便于逐条断言"译文落在自己的 id 上"。"""

    return [b for b in path.read_text(encoding="utf-8").split("\n\n") if b.strip()]


def test_cli_main_accepts_id_keyed_dict_contract(
    cli, tmp_path: Path, monkeypatch, capsys
) -> None:
    """新 dict 格式经真实 main（unified 引擎）可用，且 prompt 已是新契约。"""

    cfg = _write_config(tmp_path)
    ja = _write_srt(tmp_path, 2)
    output = tmp_path / "sample.zh.srt"
    prompts: list[str] = []

    def handler(url, **kwargs):  # type: ignore[no-untyped-def]
        prompts.append(kwargs["json"]["messages"][1]["content"])
        return _dict_response({"1": "zh-1", "2": "zh-2"})

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
    assert "engine=unified" in capsys.readouterr().out
    blocks = _zh_blocks(output)
    assert len(blocks) == 2
    assert "zh-1" in blocks[0] and "zh-2" in blocks[1]
    # 脚本提示词必须与服务端同一契约（id 为 key、条数一致、禁止合并/拆分）。
    assert prompts and '"<id>"' in prompts[0]
    assert "no more, no fewer" in prompts[0]
    assert "never merge" in prompts[0] and "never split" in prompts[0]


def test_cli_main_dict_missing_key_refetches_only_missing(
    cli, tmp_path: Path, monkeypatch
) -> None:
    """少 key（拆行丢条）：只补缺失 id，且每条译文不后移一位。"""

    cfg = _write_config(tmp_path, overrides={"batch_size": "10", "retry": "3"})
    ja = _write_srt(tmp_path, 3)
    output = tmp_path / "sample.zh.srt"
    requested: list[list[int]] = []

    def handler(url, **kwargs):  # type: ignore[no-untyped-def]
        ids = _requested_ids(kwargs)
        requested.append(ids)
        mapping = {str(i): f"zh-{i}" for i in ids}
        if len(requested) == 1:
            mapping.pop("3")  # 拆行丢条：首轮少一个 key
        return _dict_response(mapping)

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
    assert requested == [[1, 2, 3], [3]]  # C.5：只补缺失，不整批重发
    blocks = _zh_blocks(output)
    assert len(blocks) == 3
    for item_id, block in zip((1, 2, 3), blocks):
        assert f"zh-{item_id}" in block  # 逐条绑定自己的 id，无错位


def test_cli_main_dict_extra_key_is_not_written(
    cli, tmp_path: Path, monkeypatch
) -> None:
    """多 key（拆行多出一条）：多出来的译文绝不落盘，正常条目仍写入。"""

    cfg = _write_config(tmp_path, overrides={"batch_size": "10", "retry": "2"})
    ja = _write_srt(tmp_path, 2)
    output = tmp_path / "sample.zh.srt"
    calls = {"n": 0}

    def handler(url, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return _dict_response({"1": "zh-1", "2": "zh-2", "99": "多出来的一条"})

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
    text = output.read_text(encoding="utf-8")
    assert "zh-1" in text and "zh-2" in text
    assert "多出来的一条" not in text  # 本批之外的 id 不得写入任何字幕行
    assert calls["n"] == 1  # 已得子集收下后 pending 为空，不再补请求


def test_legacy_engine_accepts_dict_contract(cli, tmp_path: Path, monkeypatch) -> None:
    """--engine legacy 的排障路径同样支持新 dict 契约（提示词同源）。"""

    cfg = _write_config(tmp_path)
    ja = _write_srt(tmp_path, 2)
    args = _args(cli, tmp_path, cfg, ja, engine="legacy")
    prompts: list[str] = []

    def handler(url, **kwargs):  # type: ignore[no-untyped-def]
        prompts.append(kwargs["json"]["messages"][1]["content"])
        return _dict_response({"1": "zh-1", "2": "zh-2"})

    _install_fake_http(cli, monkeypatch, handler)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args, **_kwargs: None)

    cli._run_legacy_engine(
        args, repo_root=tmp_path, config_path=cfg, config=cli._load_config(cfg)
    )

    blocks = _zh_blocks(args.output)
    assert len(blocks) == 2
    assert "zh-1" in blocks[0] and "zh-2" in blocks[1]
    assert prompts and '"<id>"' in prompts[0]


def test_legacy_engine_dict_missing_key_retries_only_missing(
    cli, tmp_path: Path, monkeypatch, capsys
) -> None:
    """legacy 引擎的 C.5：少 key 时只补缺失 id，已得译文与条数都不丢。"""

    cfg = _write_config(tmp_path, overrides={"retry": "3"})
    ja = _write_srt(tmp_path, 3)
    args = _args(cli, tmp_path, cfg, ja, engine="legacy")
    requested: list[list[int]] = []

    def handler(url, **kwargs):  # type: ignore[no-untyped-def]
        ids = _requested_ids(kwargs)
        requested.append(ids)
        mapping = {str(i): f"zh-{i}" for i in ids}
        if len(requested) == 1:
            mapping.pop("3")  # 首轮少一个 key
        return _dict_response(mapping)

    _install_fake_http(cli, monkeypatch, handler)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args, **_kwargs: None)

    cli._run_legacy_engine(
        args, repo_root=tmp_path, config_path=cfg, config=cli._load_config(cfg)
    )

    assert requested == [[1, 2, 3], [3]]
    blocks = _zh_blocks(args.output)
    assert len(blocks) == 3
    for item_id, block in zip((1, 2, 3), blocks):
        assert f"zh-{item_id}" in block
    assert "keep_source_text" not in capsys.readouterr().out  # 补齐了，无需保留原文


def test_legacy_engine_dict_extra_key_is_not_written(
    cli, tmp_path: Path, monkeypatch
) -> None:
    """legacy 引擎：多出来的 id 被上报，且绝不写入字幕。"""

    cfg = _write_config(tmp_path, overrides={"retry": "2"})
    ja = _write_srt(tmp_path, 2)
    args = _args(cli, tmp_path, cfg, ja, engine="legacy")
    calls = {"n": 0}

    def handler(url, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return _dict_response({"1": "zh-1", "2": "zh-2", "99": "多出来的一条"})

    _install_fake_http(cli, monkeypatch, handler)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args, **_kwargs: None)

    cli._run_legacy_engine(
        args, repo_root=tmp_path, config_path=cfg, config=cli._load_config(cfg)
    )

    text = args.output.read_text(encoding="utf-8")
    assert "zh-1" in text and "zh-2" in text
    assert "多出来的一条" not in text
    assert calls["n"] == 1


def test_legacy_engine_non_id_failure_still_fails_loudly(
    cli, tmp_path: Path, monkeypatch
) -> None:
    """legacy 引擎的失败语义未变：连接类错误耗尽重试即 RuntimeError，不静默降级。"""

    cfg = _write_config(tmp_path, overrides={"retry": "2"})
    ja = _write_srt(tmp_path, 2)
    args = _args(cli, tmp_path, cfg, ja, engine="legacy")

    def handler(url, **kwargs):  # type: ignore[no-untyped-def]
        raise requests.ConnectionError("boom")

    _install_fake_http(cli, monkeypatch, handler)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="failed"):
        cli._run_legacy_engine(
            args, repo_root=tmp_path, config_path=cfg, config=cli._load_config(cfg)
        )
    assert not args.output.exists()  # 整批失败不落半成品


def test_legacy_engine_partial_exhausted_keeps_source_text(
    cli, tmp_path: Path, monkeypatch, capsys
) -> None:
    """legacy 引擎重试耗尽仍缺 id：保留日文原文，不丢行、不静默丢批。"""

    cfg = _write_config(tmp_path, overrides={"retry": "2"})
    ja = _write_srt(tmp_path, 2)
    args = _args(cli, tmp_path, cfg, ja, engine="legacy")

    def handler(url, **kwargs):  # type: ignore[no-untyped-def]
        return _dict_response({"1": "zh-1"})  # 永远缺 id2

    _install_fake_http(cli, monkeypatch, handler)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args, **_kwargs: None)

    cli._run_legacy_engine(
        args, repo_root=tmp_path, config_path=cfg, config=cli._load_config(cfg)
    )

    blocks = _zh_blocks(args.output)
    assert len(blocks) == 2  # 条数不丢
    assert "zh-1" in blocks[0]
    assert "源2" in blocks[1]  # 缺失 id 保留日文原文
    assert "keep_source_text" in capsys.readouterr().out
