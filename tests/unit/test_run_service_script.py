"""启动脚本 scripts/run_service.sh 与 launcher 的只读测试（SPEC H）。

不启动真实服务：只验证 [security] bind_host/bind_port 的解析、CLI 覆盖
优先级、默认安全值，以及脚本本身能正确找到主项目 .venv。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from whisper_stt_service.service.launcher import (
    DEFAULT_BIND_HOST,
    DEFAULT_BIND_PORT,
    BindConfig,
    build_parser,
    read_bind_settings,
    resolve_bind_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_service.sh"


def _write_config(tmp_path: Path, body: str) -> Path:
    """写一个最小配置并返回路径。"""

    path = tmp_path / "config.ini"
    path.write_text(body, encoding="utf-8")
    return path


def test_defaults_when_section_missing(tmp_path: Path) -> None:
    """没有 [security] 段时回落安全默认 127.0.0.1:18000。"""

    path = _write_config(tmp_path, "[runtime]\ndb_path = x.db\n")
    assert read_bind_settings(path) == (DEFAULT_BIND_HOST, DEFAULT_BIND_PORT)
    assert (DEFAULT_BIND_HOST, DEFAULT_BIND_PORT) == ("127.0.0.1", 18000)


def test_reads_bind_settings_from_config(tmp_path: Path) -> None:
    """从 [security] 读取 bind_host/bind_port。"""

    path = _write_config(
        tmp_path, "[security]\nbind_host = 0.0.0.0\nbind_port = 19000\n"
    )
    assert read_bind_settings(path) == ("0.0.0.0", 19000)


@pytest.mark.parametrize("raw", ["not-a-port", "", "0", "70000", "-1"])
def test_invalid_port_falls_back_to_default(tmp_path: Path, raw: str) -> None:
    """非法端口回落默认值，避免启动即失败。"""

    path = _write_config(tmp_path, f"[security]\nbind_port = {raw}\n")
    assert read_bind_settings(path)[1] == DEFAULT_BIND_PORT


def test_cli_overrides_config(tmp_path: Path) -> None:
    """CLI 覆盖 > 配置文件 > 默认值。"""

    path = _write_config(tmp_path, "[security]\nbind_host = 10.0.0.1\nbind_port = 19000\n")
    from_file = resolve_bind_config(config_path=path)
    assert (from_file.host, from_file.port) == ("10.0.0.1", 19000)

    overridden = resolve_bind_config(config_path=path, host="127.0.0.1", port=1234)
    assert (overridden.host, overridden.port) == ("127.0.0.1", 1234)


def test_missing_config_file_is_tolerated(tmp_path: Path) -> None:
    """配置文件不存在时不抛异常（由 bootstrap 负责创建/校验）。"""

    resolved = resolve_bind_config(config_path=tmp_path / "absent.ini")
    assert resolved.port == DEFAULT_BIND_PORT


def test_bind_config_serializes_to_json() -> None:
    """BindConfig 可安全序列化用于日志/测试。"""

    payload = json.loads(
        json.dumps(BindConfig("127.0.0.1", 18000, Path("config.ini")).as_dict())
    )
    assert payload["host"] == "127.0.0.1"
    assert payload["port"] == 18000


def test_help_exits_without_starting(tmp_path: Path) -> None:
    """--help 为只读操作：退出码 0 且不启动服务。"""

    result = subprocess.run(
        [str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0
    assert "bind_host" in result.stdout


def test_print_config_does_not_start_server(tmp_path: Path) -> None:
    """--print-config 只输出 JSON 配置，不启动 uvicorn。"""

    config = _write_config(
        tmp_path, "[security]\nbind_host = 127.0.0.2\nbind_port = 18001\n"
    )
    env = dict(os.environ, WHISPER_STT_CONFIG=str(config))
    result = subprocess.run(
        [str(SCRIPT), "--print-config"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["host"] == "127.0.0.2"
    assert payload["port"] == 18001


def test_unknown_argument_fails_fast() -> None:
    """未知参数返回 2（argparse 语义），不会带着错配置启动。"""

    result = subprocess.run(
        [str(SCRIPT), "--definitely-not-an-option"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2


def test_parser_defaults_are_none() -> None:
    """CLI 未指定时不覆盖配置文件值。"""

    args = build_parser().parse_args([])
    assert args.host is None and args.port is None and args.reload is False


# ------------------------------------------ --config 与 bootstrap 同源（P1）


def _full_config(tmp_path: Path, *, host: str, port: int) -> Path:
    """基于 config.example.ini 生成一份带指定 bind 的完整配置。"""

    text = (REPO_ROOT / "config.example.ini").read_text(encoding="utf-8")
    security = f"[security]\nbind_host = {host}\nbind_port = {port}\n"
    if "[security]" in text:
        import re

        text = re.sub(r"\[security\][^\[]*", security, text, count=1)
    else:
        text = security + "\n" + text
    path = tmp_path / "selected.ini"
    path.write_text(text, encoding="utf-8")
    return path


def _run_bootstrap_probe(config: Path, env_config: Path) -> dict:
    """在子进程里模拟 launcher：pin 配置后再导入 bootstrap，返回读到的路径。

    用独立进程，避免 `whisper_stt_service.main` / bootstrap 的模块缓存污染。

    失败一律硬失败（fail-fast）：这是**关键启动验收**，任何导入/语法/名称错误
    都说明产品当前无法启动，绝不允许因产品自身损坏而 skip 掉验证。
    """

    code = (
        "import json;"
        "from whisper_stt_service.service.launcher import resolve_bind_config, pin_config_environment;"
        f"b = resolve_bind_config(config_path={str(config)!r});"
        "pin_config_environment(b);"
        "import whisper_stt_service.service.bootstrap as bs;"
        "settings, cfg = bs._resolve_settings();"
        "print(json.dumps({'bootstrap_reads': str(cfg), 'cli_selected': str(b.config_path), "
        "'host': b.host, 'port': b.port}))"
    )
    env = dict(os.environ, WHISPER_STT_CONFIG=str(env_config))
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        # 不做任何“产品坏就跳过”的降级：启动验收必须能看到真实失败。
        raise AssertionError(
            "bootstrap probe failed (product must be importable): "
            + result.stderr.strip()
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_config_flag_actually_drives_bootstrap(tmp_path: Path) -> None:
    """--config 选中的文件必须被 bootstrap 真正读取（不只是 print-config）。

    回归背景：launcher 曾只把 --config 用于解析 host/port，bootstrap 仍读
    WHISPER_STT_CONFIG（脚本预导出的旧默认），导致监听与鉴权/DB/roots 不同源。
    """

    selected = _full_config(tmp_path, host="127.0.0.9", port=18009)
    other = tmp_path / "other.ini"
    other.write_text("[security]\nbind_host = 10.9.9.9\nbind_port = 19999\n", encoding="utf-8")

    probe = _run_bootstrap_probe(selected, other)
    assert probe["bootstrap_reads"] == probe["cli_selected"]
    assert probe["bootstrap_reads"] == str(selected.resolve())
    assert probe["bootstrap_reads"] != str(other.resolve())
    assert probe["host"] == "127.0.0.9"
    assert probe["port"] == 18009


def test_pin_config_environment_sets_absolute_path(tmp_path: Path) -> None:
    """pin_config_environment 写回绝对路径，不受当前工作目录影响。"""

    from whisper_stt_service.service.launcher import BindConfig, pin_config_environment

    bind = BindConfig(host="127.0.0.1", port=18000, config_path=Path("some-config.ini"))
    previous = os.environ.get("WHISPER_STT_CONFIG")
    try:
        pinned = pin_config_environment(bind)
        assert pinned.is_absolute()
        assert os.environ["WHISPER_STT_CONFIG"] == str(pinned)
    finally:
        if previous is None:
            os.environ.pop("WHISPER_STT_CONFIG", None)
        else:
            os.environ["WHISPER_STT_CONFIG"] = previous


def test_cached_main_module_is_purged_before_import() -> None:
    """切换配置后必须剔除缓存的 main，避免复用旧配置构建的 app。"""

    from whisper_stt_service.service.launcher import _purge_cached_main

    sys.modules["whisper_stt_service.main"] = object()  # type: ignore[assignment]
    _purge_cached_main()
    assert "whisper_stt_service.main" not in sys.modules


def test_script_does_not_preexport_config_when_flag_given() -> None:
    """脚本在传了 --config 时不得预导出旧默认配置（否则覆盖用户选择）。"""

    source = SCRIPT.read_text(encoding="utf-8")
    assert "has_config_arg" in source