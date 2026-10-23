"""安全纯函数工具：路径白名单、copy_back 逃逸防护、API token 占位与常量时间比较。

本模块只依赖标准库，不读写配置文件、不访问数据库、不启动服务，供 API 层与运行时层调用。
设计约束（SPEC H、D4）：

- 所有失败都抛出 :class:`SecurityError`，其 `args[0]` 恒为稳定错误码（形如
  `video_path_not_found`），调用方直接映射为 HTTP 400/401；错误消息**不得**携带
  token 明文或解析后的绝对路径。
- 不引入新的必填配置项：`SessionConfig` 的所有字段都有安全默认值。
- 白名单参数同时接受 `tuple`/`list`/`str`（`str` 按逗号分隔多根），根目录允许是
  `str` 或 `pathlib.Path`。
"""

from __future__ import annotations

import hmac
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# 视频扩展名白名单（SPEC H：POST /jobs 路径校验）。
VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".m2ts"}
)

# 示例配置里的占位 token（SPEC H：allow_placeholder_token=false 时拒绝启动）。
PLACEHOLDER_TOKENS: frozenset[str] = frozenset(
    {
        "<change_me_strong_token>",
        "change-me",
        "change-me-token",
        "changeme",
        "your-api-token",
        "your_api_token",
        "test-token",
        "dummy-token",
    }
)

# copy_back 的“回写到视频所在目录”占位值（与 executor/translate.py 的历史约定一致）。
VIDEO_DIR_COPY_BACK_MARKERS: frozenset[str] = frozenset(
    {"", "__video_dir__", "video_dir", "input_video_dir"}
)

# token 最小长度阈值（低于该长度视为弱口令，仅用于启动告警/校验提示）。
MIN_TOKEN_LENGTH = 16


class SecurityError(ValueError):
    """安全校验失败。

    `args[0]` 为稳定错误码，供 API 直接映射状态码；`args[1]` 为不含敏感信息的说明，
    可安全写日志。
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code, detail)
        self.code = code
        self.detail = detail

    def __str__(self) -> str:
        """只返回稳定错误码，避免异常文本意外拼接路径或 token。"""

        return self.code


@dataclass(frozen=True)
class SessionConfig:
    """一次入队请求所用到的安全相关配置（缺省即安全默认值）。

    - `allowed_video_roots`：非空时视频路径 resolve 后必须位于其下；空 = 不限制。
    - `allowed_copy_back_roots`：非空时 copy_back 目标 resolve 后必须位于其下；
      空 = 仅允许视频所在目录（默认行为）。
    - `require_token`：true 且 token 为空 → 启动失败。
    - `allow_placeholder_token`：false 时占位 token 一律拒绝。
    """

    allowed_video_roots: tuple[Path, ...] = ()
    allowed_copy_back_roots: tuple[Path, ...] = ()
    allowed_video_extensions: tuple[str, ...] = tuple(sorted(VIDEO_EXTENSIONS))
    require_token: bool = True
    allow_placeholder_token: bool = False


def _coerce_roots(
    roots: Sequence[str | Path] | str | Path | None,
) -> tuple[Path, ...]:
    """把配置来的白名单根目录归一化为已解析的绝对 Path 元组。

    接受 `None`（空）、`str`（可用逗号分隔多根）、`Path`，或它们的 `list`/`tuple`。
    根目录不存在时仍保留其解析结果：越界判定本身不要求根目录存在，避免启动期
    因挂载顺序不同而误判。
    """

    if roots is None:
        return ()
    if isinstance(roots, (str, Path)):
        items: list[object] = [roots]
    elif isinstance(roots, (list, tuple, set, frozenset)):
        items = list(roots)
    else:
        raise SecurityError("security_roots_invalid", "roots must be str/list/tuple")

    normalized: list[Path] = []
    for item in items:
        if isinstance(item, str):
            candidates = [part.strip() for part in item.split(",")]
        elif isinstance(item, Path):
            candidates = [str(item)]
        else:
            raise SecurityError(
                "security_roots_invalid", "root entry must be str or pathlib.Path"
            )
        for candidate in candidates:
            if not candidate:
                continue
            normalized.append(Path(candidate).expanduser().resolve())
    return tuple(normalized)


def _coerce_extensions(extensions: Sequence[str] | None) -> frozenset[str]:
    """把扩展名白名单归一化为小写、带点前缀的集合。"""

    if extensions is None:
        return VIDEO_EXTENSIONS
    normalized: set[str] = set()
    for item in extensions:
        ext = str(item).strip().lower()
        if not ext:
            continue
        normalized.add(ext if ext.startswith(".") else f".{ext}")
    return frozenset(normalized) if normalized else VIDEO_EXTENSIONS


def _configured_roots(
    roots: Sequence[str | Path] | str | Path | None,
) -> tuple[Path, ...]:
    """返回已校验的白名单根元组；类型非法时抛固定错误码而非 TypeError。"""

    try:
        return _coerce_roots(roots)
    except SecurityError:
        raise
    except TypeError:
        raise SecurityError(
            "security_roots_invalid", "roots must be str/list/tuple"
        ) from None


def _is_within(path: Path, root: Path) -> bool:
    """判断 `path` 是否位于（或等于）`root` 之下，两者都必须已 resolve。"""

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_video_path(
    video_path: str | Path,
    *,
    session: SessionConfig | None = None,
) -> Path:
    """校验并解析待处理视频路径。

    校验顺序与错误码（稳定，可直接映射为 HTTP 400）：

    1. `video_path_invalid`：类型不对或为空字符串。
    2. `video_path_not_absolute`：不是绝对路径（先拒绝，避免相对路径污染进程 cwd）。
    3. `video_path_not_found`：路径不存在。
    4. `video_path_not_file`：存在但不是普通文件（含目录、设备、断链符号链接）。
    5. `video_path_extension_not_allowed`：扩展名不在白名单内。
    6. `video_path_outside_allowed_roots`：配置了 `allowed_video_roots` 且解析后的
       真实路径不在任一根目录之下（symlink 逃逸会在这里被拦下）。

    成功后返回 `resolve(strict=True)` 的真实路径；错误消息中不回显原始路径。
    """

    cfg = session or SessionConfig()
    if isinstance(video_path, Path):
        raw = str(video_path)
    elif isinstance(video_path, str):
        raw = video_path.strip()
    else:
        raise SecurityError("video_path_invalid", "video_path must be str or Path")
    if not raw:
        raise SecurityError("video_path_invalid", "video_path is empty")

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise SecurityError("video_path_not_absolute", "video_path must be absolute")
    if not candidate.exists():
        raise SecurityError("video_path_not_found", "video_path does not exist")
    if not candidate.is_file():
        raise SecurityError("video_path_not_file", "video_path is not a regular file")

    allowed_exts = _coerce_extensions(cfg.allowed_video_extensions)
    if candidate.suffix.lower() not in allowed_exts:
        raise SecurityError(
            "video_path_extension_not_allowed", "video extension not allowed"
        )

    try:
        # strict=True：路径在 exists 之后被删除或断链时同样失败。
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise SecurityError(
            "video_path_not_found", "video_path cannot be resolved"
        ) from None
    if not resolved.is_file():
        raise SecurityError("video_path_not_file", "video_path is not a regular file")

    roots = _configured_roots(cfg.allowed_video_roots)
    if roots and not any(_is_within(resolved, root) for root in roots):
        raise SecurityError(
            "video_path_outside_allowed_roots", "video_path is outside allowed roots"
        )
    return resolved


def normalize_video_path(
    video_path: str | Path,
    *,
    session: SessionConfig | None = None,
) -> Path:
    """规范化视频路径（**不要求文件已存在**），并执行与 API 同源的安全校验。

    供 repo 内部入队使用：内部入队不应绕过白名单，但历史设计允许“先入队、
    后落盘”（脚本/e2e/单测会用尚不存在的路径），因此这里只把“存在性”换成
    “允许不存在”，其余规则与 :func:`resolve_video_path` 完全一致：

    1. video_path_invalid：类型不对或为空字符串。
    2. video_path_not_absolute：不是绝对路径。
    3. video_path_extension_not_allowed：扩展名不在白名单内。
    4. video_path_outside_allowed_roots：canonical 路径不在 allowed_video_roots 下。

    返回 `resolve(strict=False)` 得到的 canonical 路径（去除 `..`、符号链接等
    等价写法），保证同文件的不同写法在库里折叠成同一个键，从而让 by-path 查询
    和同路径幂等判定一致。
    """

    cfg = session or SessionConfig()
    if isinstance(video_path, Path):
        raw = str(video_path)
    elif isinstance(video_path, str):
        raw = video_path.strip()
    else:
        raise SecurityError("video_path_invalid", "video_path must be str or Path")
    if not raw:
        raise SecurityError("video_path_invalid", "video_path is empty")

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise SecurityError("video_path_not_absolute", "video_path must be absolute")

    allowed_exts = _coerce_extensions(cfg.allowed_video_extensions)
    if candidate.suffix.lower() not in allowed_exts:
        raise SecurityError(
            "video_path_extension_not_allowed", "video extension not allowed"
        )

    try:
        # strict=False：路径暂不存在时也能得到规范化的绝对路径。
        canonical = candidate.resolve(strict=False)
    except OSError:
        raise SecurityError(
            "video_path_not_found", "video_path cannot be resolved"
        ) from None

    roots = _configured_roots(cfg.allowed_video_roots)
    if roots and not any(_is_within(canonical, root) for root in roots):
        raise SecurityError(
            "video_path_outside_allowed_roots", "video_path is outside allowed roots"
        )
    return canonical


def resolve_copy_back_dir(
    copy_back: str | Path | None,
    *,
    video_path: str | Path,
    session: SessionConfig | None = None,
    strict_parent: bool = True,
) -> Path:
    """解析并校验 copy_back 目标目录，防止符号链接逃逸。

    `copy_back` 为 `None`/`""`/`"__video_dir__"`/`"video_dir"`/`"input_video_dir"`
    时表示“回写到视频所在目录”（历史默认，与 executor 一致）。

    `strict_parent` 控制是否校验“目标父目录真实存在”（默认 True）：

    - API 层（对外信任边界）必须保持 True：任何不存在/非目录/断链都拒绝。
    - repo 层内部入队传 False：入队时视频可能尚未落盘、目标目录也可能稍后才创建，
      此时只做 **canonical 规范化 + 白名单** 校验，把存在性推迟到执行期；
      这样既不会让内部直入被绕开白名单，也不会因“目录还没建”误拒。

    校验顺序与错误码（稳定）：

    1. `copy_back_target_invalid`：类型不对。
    2. `copy_back_video_path_invalid`：`video_path` 不是绝对路径（无法定位同名目录）。
    3. `copy_back_target_empty`：显式目标为空白字符串。
    4. `copy_back_target_not_absolute`：显式目标是相对路径。
    5. `copy_back_target_exists_not_dir`：目标已存在但不是目录。
    6. `copy_back_parent_missing`：目标父目录不存在（**不隐式 mkdir**；仅 strict_parent=True）。
    7. `copy_back_target_resolve_failed`：resolve 失败（断链等）。
    8. `copy_back_outside_allowed_roots` / `copy_back_outside_video_dir`：超出允许范围。

    允许范围规则（与 strict_parent 无关，始终生效）：

    - 配置了 `allowed_copy_back_roots`：解析后目标（含其真实路径）必须位于任一根之下。
    - 否则：目标的真实路径必须等于视频所在目录（默认仅同目录）。

    返回 `resolve(strict=False)` 的目标目录；strict_parent=True 时因为父目录已
    `resolve(strict=True)`，返回值本身就是真实路径，符号链接逃逸不会带出去。
    """

    cfg = session or SessionConfig()
    if isinstance(video_path, Path):
        video_raw = str(video_path)
    elif isinstance(video_path, str):
        video_raw = video_path.strip()
    else:
        raise SecurityError(
            "copy_back_video_path_invalid", "video_path must be str or Path"
        )
    video_candidate = Path(video_raw).expanduser() if video_raw else None
    if video_candidate is None or not video_candidate.is_absolute():
        raise SecurityError(
            "copy_back_video_path_invalid", "video_path must be absolute"
        )
    video_dir = video_candidate.resolve().parent

    if copy_back is None:
        marker = ""
    elif isinstance(copy_back, Path):
        marker = str(copy_back).strip()
    elif isinstance(copy_back, str):
        marker = copy_back.strip()
    else:
        raise SecurityError(
            "copy_back_target_invalid", "copy_back must be str, Path or None"
        )

    if marker.lower() in VIDEO_DIR_COPY_BACK_MARKERS:
        resolved_target = video_dir
    else:
        if marker == "":
            raise SecurityError("copy_back_target_empty", "copy_back target is empty")
        target = Path(marker).expanduser()
        if not target.is_absolute():
            # 相对路径无法判定逃逸语义，直接拒绝（与 video_path 策略一致）。
            raise SecurityError(
                "copy_back_target_not_absolute", "copy_back target must be absolute"
            )
        if target.exists() and not target.is_dir():
            raise SecurityError(
                "copy_back_target_exists_not_dir", "copy_back target is not a directory"
            )
        parent = target.parent
        if strict_parent:
            if not parent.is_dir():
                # 父目录必须真实存在；不在这里 mkdir，避免替调用方创建任意目录。
                raise SecurityError(
                    "copy_back_parent_missing", "copy_back parent directory missing"
                )
            try:
                real_parent = parent.resolve(strict=True)
            except OSError:
                raise SecurityError(
                    "copy_back_target_resolve_failed",
                    "copy_back parent cannot be resolved",
                ) from None
            resolved_target = real_parent / target.name
            if target.is_dir():
                try:
                    resolved_target = target.resolve(strict=True)
                except OSError:
                    raise SecurityError(
                        "copy_back_target_resolve_failed",
                        "copy_back target cannot be resolved",
                    ) from None
        else:
            # repo 内部入队：目录可能稍后才创建，只做规范化不做存在性判定；
            # 白名单仍按 canonical 结果校验，内部调用无法借此绕过 allowed roots。
            try:
                resolved_target = target.resolve(strict=False)
            except OSError:
                raise SecurityError(
                    "copy_back_target_resolve_failed",
                    "copy_back target cannot be resolved",
                ) from None

    roots = _configured_roots(cfg.allowed_copy_back_roots)
    if roots:
        if not any(_is_within(resolved_target, root) for root in roots):
            raise SecurityError(
                "copy_back_outside_allowed_roots",
                "copy_back target is outside allowed roots",
            )
    elif resolved_target != video_dir:
        raise SecurityError(
            "copy_back_outside_video_dir",
            "copy_back target must be the video directory by default",
        )
    return resolved_target


def is_placeholder_token(token: str | None) -> bool:
    """判断 token 是否为占位值（含尖括号或命中示例值）。

    判定规则（任一命中即为占位）：

    - 空字符串或纯空白；
    - 含 `<` 或 `>`（`<CHANGE_ME_STRONG_TOKEN>` 这类示例）；
    - 小写去空白后命中 :data:`PLACEHOLDER_TOKENS`；
    - 命中 `change me`/`change_me`/`your ... token` 等模板文字。

    纯函数：不打印、不记录 token 内容。
    """

    if token is None:
        return True
    raw = str(token).strip()
    if raw == "":
        return True
    if "<" in raw or ">" in raw:
        return True
    lowered = raw.lower()
    if lowered in PLACEHOLDER_TOKENS:
        return True
    compact = lowered.replace("-", "").replace("_", "").replace(" ", "")
    if "changeme" in compact:
        return True
    if "yourtoken" in compact or "your_api" in lowered or "yourapi" in compact:
        return True
    return False


def validate_startup_token(
    token: str | None,
    *,
    require_token: bool = True,
    allow_placeholder_token: bool = False,
) -> None:
    """启动期 token 校验（SPEC H 的 fail-closed 策略）。

    错误码（稳定，bootstrap/app 可直接记录并拒绝启动）：

    - `api_token_missing`：`require_token=true` 且 token 为空。
    - `api_token_placeholder`：`allow_placeholder_token=false` 且 token 是占位值
      （空 token 在 `require_token=false` 时由调用方决定，不在此报错）。

    说明文本只含错误码语义，**不包含 token 内容**。
    """

    raw = "" if token is None else str(token).strip()
    if raw == "":
        if require_token:
            raise SecurityError("api_token_missing", "api_token is required")
        return
    if not allow_placeholder_token and is_placeholder_token(raw):
        raise SecurityError(
            "api_token_placeholder", "api_token looks like a placeholder"
        )


def token_is_secure_enough(token: str | None) -> bool:
    """弱口令提示：非占位且长度 >= :data:`MIN_TOKEN_LENGTH`（不抛异常，供启动告警）。"""

    raw = "" if token is None else str(token).strip()
    if raw == "" or is_placeholder_token(raw):
        return False
    return len(raw) >= MIN_TOKEN_LENGTH


def tokens_match(provided: str | None, expected: str | None) -> bool:
    """常量时间比较两个 token（SPEC H：`hmac.compare_digest`）。

    任一为空都视为不匹配（空 token 表示鉴权关闭，应由调用方先行短路）。长度不同时
    也走 `compare_digest`，避免时序差异泄露长度。
    """

    left = "" if provided is None else str(provided).strip()
    right = "" if expected is None else str(expected).strip()
    if left == "" or right == "":
        return False
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def mask_token(token: str | None) -> str:
    """返回可安全写日志的 token 描述（只暴露是否存在与长度，不含任何字符）。"""

    raw = "" if token is None else str(token).strip()
    if raw == "":
        return "unset"
    return f"set(len={len(raw)})"


def summarize_dag_error(exc: ValueError) -> dict[str, str]:
    """把安全校验异常转成可安全返回给客户端的 `{"detail": code}` 结构。"""

    if isinstance(exc, SecurityError):
        return {"detail": exc.code}
    return {"detail": "security_validation_failed"}
