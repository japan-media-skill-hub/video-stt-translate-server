"""core/security.py 纯函数测试：路径白名单、copy_back 逃逸防护、token 占位与比较。

不依赖 FastAPI、数据库或真实服务；仅用 tmp_path 构造文件与符号链接。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whisper_stt_service.core.security import (
    SessionConfig,
    SecurityError,
    is_placeholder_token,
    mask_token,
    resolve_copy_back_dir,
    resolve_video_path,
    summarize_dag_error,
    token_is_secure_enough,
    tokens_match,
    validate_startup_token,
)


def _expect_code(code: str, func, *args, **kwargs) -> SecurityError:
    """断言抛出指定错误码，并确认异常文本只含错误码（不泄露路径/token）。"""

    with pytest.raises(SecurityError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.code == code
    assert str(excinfo.value) == code
    return excinfo.value


@pytest.fixture()
def video(tmp_path: Path) -> Path:
    """构造一个合法视频文件。"""

    path = tmp_path / "sample.mp4"
    path.write_bytes(b"video")
    return path


# ------------------------------------------------------------------ 视频路径


def test_resolve_video_path_accepts_allowed_extension(video: Path, tmp_path: Path) -> None:
    """合法绝对路径应返回 resolve 后的真实路径。"""

    resolved = resolve_video_path(video)
    assert resolved == video.resolve(strict=True)
    assert resolved.is_absolute()


@pytest.mark.parametrize(
    "suffix", [".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".m2ts"]
)
def test_resolve_video_path_extension_whitelist(tmp_path: Path, suffix: str) -> None:
    """SPEC 列出的 7 种扩展名都应被接受（大小写不敏感）。"""

    path = tmp_path / f"clip{suffix.upper()}"
    path.write_bytes(b"video")
    assert resolve_video_path(path) == path.resolve(strict=True)


@pytest.mark.parametrize("suffix", [".txt", ".srt", ".wav", ""])
def test_resolve_video_path_rejects_other_extensions(
    tmp_path: Path, suffix: str
) -> None:
    """非白名单扩展名应返回 video_path_extension_not_allowed。"""

    path = tmp_path / f"clip{suffix}"
    path.write_bytes(b"x")
    _expect_code("video_path_extension_not_allowed", resolve_video_path, path)


def test_resolve_video_path_rejects_empty_and_bad_type() -> None:
    """空字符串与非法类型应返回 video_path_invalid。"""

    _expect_code("video_path_invalid", resolve_video_path, "")
    _expect_code("video_path_invalid", resolve_video_path, "   ")
    _expect_code("video_path_invalid", resolve_video_path, 123)


def test_resolve_video_path_rejects_relative_path(monkeypatch, tmp_path: Path) -> None:
    """相对路径必须在 exists 之前被拒绝。"""

    _expect_code("video_path_not_absolute", resolve_video_path, "sample.mp4")


def test_resolve_video_path_rejects_missing(tmp_path: Path) -> None:
    """不存在的路径应返回 video_path_not_found。"""

    _expect_code(
        "video_path_not_found", resolve_video_path, tmp_path / "nope.mp4"
    )


def test_resolve_video_path_rejects_directory(tmp_path: Path) -> None:
    """目录应返回 video_path_not_file。"""

    directory = tmp_path / "dir.mp4"
    directory.mkdir()
    _expect_code("video_path_not_file", resolve_video_path, directory)


def test_resolve_video_path_rejects_broken_symlink(tmp_path: Path) -> None:
    """断链符号链接视为不存在（找不到目标文件）。"""

    link = tmp_path / "broken.mp4"
    link.symlink_to(tmp_path / "missing-target.mp4")
    _expect_code("video_path_not_found", resolve_video_path, link)


def test_resolve_video_path_inside_allowed_root(tmp_path: Path) -> None:
    """配置 allowed_video_roots 后，根目录内的视频应通过。"""

    root = tmp_path / "videos"
    root.mkdir()
    video = root / "clip.mp4"
    video.write_bytes(b"v")
    cfg = SessionConfig(allowed_video_roots=(root,))
    assert resolve_video_path(video, session=cfg) == video.resolve()


def test_resolve_video_path_outside_allowed_root(tmp_path: Path) -> None:
    """不在白名单根目录下的视频应被拒绝。"""

    allowed = tmp_path / "videos"
    allowed.mkdir()
    other = tmp_path / "elsewhere"
    other.mkdir()
    video = other / "clip.mp4"
    video.write_bytes(b"v")
    cfg = SessionConfig(allowed_video_roots=(allowed,))
    _expect_code(
        "video_path_outside_allowed_roots", resolve_video_path, video, session=cfg
    )


def test_resolve_video_path_blocks_symlink_escape(tmp_path: Path) -> None:
    """白名单外的文件通过白名单内符号链接引用时，必须被真实路径拦截。"""

    allowed = tmp_path / "videos"
    allowed.mkdir()
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    secret = secret_dir / "hidden.mp4"
    secret.write_bytes(b"v")
    link = allowed / "linked.mp4"
    link.symlink_to(secret)

    cfg = SessionConfig(allowed_video_roots=(allowed,))
    _expect_code(
        "video_path_outside_allowed_roots", resolve_video_path, link, session=cfg
    )


def test_resolve_video_path_accepts_str_and_list_roots(tmp_path: Path) -> None:
    """allowed roots 应同时接受 str（逗号分隔）与 list[Path]。"""

    root_a = tmp_path / "a"
    root_a.mkdir()
    root_b = tmp_path / "b"
    root_b.mkdir()
    video = root_b / "clip.mp4"
    video.write_bytes(b"v")

    # 逗号分隔的 str、list[Path]、tuple[Path] 都应被接受。
    for roots in (
        f"{root_a},{root_b}",
        [root_a, root_b],
        tuple([root_a, root_b]),
    ):
        cfg = SessionConfig(allowed_video_roots=roots)  # type: ignore[arg-type]
        assert resolve_video_path(video, session=cfg) == video.resolve()
    # 白名单单根写成 str 也可以。
    single = SessionConfig(allowed_video_roots=str(root_b))  # type: ignore[arg-type]
    assert resolve_video_path(video, session=single) == video.resolve()
    # 非法根类型必须给出固定错误码，而不是 TypeError。
    bad = SessionConfig(allowed_video_roots=42)  # type: ignore[arg-type]
    _expect_code("security_roots_invalid", resolve_video_path, video, session=bad)


# --------------------------------------------------------------- copy_back


def test_copy_back_defaults_to_video_dir(video: Path) -> None:
    """None / 空 / __video_dir__ 等占位都回写到视频所在目录。"""

    expected = video.resolve().parent
    for marker in (None, "", "  ", "__video_dir__", "video_dir", "input_video_dir"):
        assert resolve_copy_back_dir(marker, video_path=video) == expected


def test_copy_back_accepts_dir_inside_allowed_roots(
    video: Path, tmp_path: Path
) -> None:
    """配置了 allowed_copy_back_roots 时，根目录内的显式目标应被接受。"""

    target = tmp_path / "out"
    target.mkdir()
    cfg = SessionConfig(allowed_copy_back_roots=(tmp_path,))
    assert (
        resolve_copy_back_dir(target, video_path=video, session=cfg)
        == target.resolve()
    )


def test_copy_back_explicit_target_rejected_by_default(
    video: Path, tmp_path: Path
) -> None:
    """未配置 roots 时，默认只允许视频同目录，其他显式目录应被拒绝。"""

    target = tmp_path / "out"
    target.mkdir()
    _expect_code(
        "copy_back_outside_video_dir", resolve_copy_back_dir, target, video_path=video
    )


def test_copy_back_rejects_escape_via_symlink(tmp_path: Path) -> None:
    """视频同目录下的符号链接指向别处时，真实路径不在允许范围应被拒绝。"""

    video_dir = tmp_path / "media"
    video_dir.mkdir()
    video = video_dir / "clip.mp4"
    video.write_bytes(b"v")
    outside = tmp_path / "outside"
    outside.mkdir()
    escape = video_dir / "escape"
    escape.symlink_to(outside, target_is_directory=True)

    cfg = SessionConfig(allowed_copy_back_roots=(video_dir,))
    _expect_code(
        "copy_back_outside_allowed_roots",
        resolve_copy_back_dir,
        escape,
        video_path=video,
        session=cfg,
    )


def test_copy_back_requires_existing_parent(video: Path, tmp_path: Path) -> None:
    """目标父目录不存在时应报 copy_back_parent_missing（不隐式创建目录）。"""

    cfg = SessionConfig(allowed_copy_back_roots=(tmp_path,))
    target = tmp_path / "missing-parent" / "out"
    _expect_code(
        "copy_back_parent_missing",
        resolve_copy_back_dir,
        target,
        video_path=video,
        session=cfg,
    )
    assert not target.parent.exists()


def test_copy_back_rejects_existing_file_target(
    video: Path, tmp_path: Path
) -> None:
    """目标已存在但不是目录时应报 copy_back_target_exists_not_dir。"""

    target = tmp_path / "out.txt"
    target.write_text("x", encoding="utf-8")
    cfg = SessionConfig(allowed_copy_back_roots=(tmp_path,))
    _expect_code(
        "copy_back_target_exists_not_dir",
        resolve_copy_back_dir,
        target,
        video_path=video,
        session=cfg,
    )


def test_copy_back_rejects_relative_target_and_bad_types(video: Path) -> None:
    """相对目标与非法类型应返回固定错误码。"""

    _expect_code(
        "copy_back_target_not_absolute", resolve_copy_back_dir, "out", video_path=video
    )
    _expect_code("copy_back_target_invalid", resolve_copy_back_dir, 42, video_path=video)


def test_copy_back_requires_absolute_video_path(tmp_path: Path) -> None:
    """视频路径非绝对路径时无法定位同目录，应报 copy_back_video_path_invalid。"""

    _expect_code(
        "copy_back_video_path_invalid",
        resolve_copy_back_dir,
        None,
        video_path="clip.mp4",
    )


# ------------------------------------------------------------------- token


@pytest.mark.parametrize(
    "token",
    [
        None,
        "",
        "   ",
        "<CHANGE_ME_STRONG_TOKEN>",
        "change-me-token",
        "CHANGE_ME",
        "your-api-token",
        "test-token",
        "dummy-token",
        "abc<token>def",
    ],
)
def test_is_placeholder_token_true(token: object) -> None:
    """占位写法（含尖括号或示例值）都应被判为占位。"""

    assert is_placeholder_token(token) is True


@pytest.mark.parametrize(
    "token",
    ["S3cr3t-9f2b7c1d4e6a8b0c", "a" * 32, "your_super_long_secret_123"],
)
def test_is_placeholder_token_false(token: str) -> None:
    """真实 token 不应被误判为占位。"""

    assert is_placeholder_token(token) is False


def test_validate_startup_token_fail_closed() -> None:
    """require_token=true 且空 token 应拒绝启动。"""

    _expect_code("api_token_missing", validate_startup_token, "")
    _expect_code("api_token_missing", validate_startup_token, None)
    assert validate_startup_token("", require_token=False) is None


def test_validate_startup_token_rejects_placeholder() -> None:
    """占位 token 默认拒绝启动，显式允许时放行。"""

    _expect_code(
        "api_token_placeholder",
        validate_startup_token,
        "<CHANGE_ME_STRONG_TOKEN>",
    )
    assert (
        validate_startup_token(
            "<CHANGE_ME_STRONG_TOKEN>", allow_placeholder_token=True
        )
        is None
    )
    assert validate_startup_token("S3cr3t-9f2b7c1d4e6a8b0c") is None


def test_validate_startup_token_never_leaks_token_in_message() -> None:
    """异常文本不得包含 token 明文。"""

    token = "<SECRET_PLACEHOLDER_VALUE>"
    with pytest.raises(SecurityError) as excinfo:
        validate_startup_token(token)
    # str() 只给错误码；args 里也只应出现错误码与固定说明，绝无 token 明文。
    assert str(excinfo.value) == "api_token_placeholder"
    for part in excinfo.value.args:
        assert token not in str(part)
        assert "SECRET_PLACEHOLDER" not in str(part)


def test_token_is_secure_enough() -> None:
    """弱口令提示：占位或过短返回 False。"""

    assert token_is_secure_enough("short") is False
    assert token_is_secure_enough(None) is False
    assert token_is_secure_enough("a" * 16) is True


def test_tokens_match_constant_time_semantics() -> None:
    """比较结果正确，且空值一律不匹配。"""

    assert tokens_match("abc123", "abc123") is True
    assert tokens_match(" abc123 ", "abc123") is True
    assert tokens_match("abc123", "abc124") is False
    assert tokens_match("", "") is False
    assert tokens_match(None, "abc") is False
    assert tokens_match("abc", None) is False


def test_mask_token_never_exposes_content() -> None:
    """mask_token 只暴露长度，不包含任何原始字符。"""

    token = "S3cr3t-9f2b7c1d4e6a8b0c"
    masked = mask_token(token)
    assert masked == f"set(len={len(token)})"
    assert "S3cr3t" not in masked
    assert mask_token(None) == "unset"


def test_summarize_dag_error_returns_stable_detail() -> None:
    """错误摘要只返回稳定错误码。"""

    assert summarize_dag_error(SecurityError("video_path_not_found")) == {
        "detail": "video_path_not_found"
    }
    assert summarize_dag_error(ValueError("boom")) == {
        "detail": "security_validation_failed"
    }
