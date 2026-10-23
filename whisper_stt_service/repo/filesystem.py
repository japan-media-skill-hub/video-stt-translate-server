"""数据库所在文件系统的探测：识别网络挂载，供 WAL 启用/回退决策使用。

为什么需要：SQLite 官方明确指出 WAL 依赖共享内存（`-shm`）与多进程一致性，**在
网络文件系统上不可靠**。只在 `PRAGMA journal_mode=WAL` 被拒绝时才回退太晚——
多数网络挂载会“成功”切到 WAL，然后在部分写入/锁竞争时损坏库。因此要在**尝试
WAL 之前**先做一次成本很低的静态识别（读取 `/proc/self/mounts`），识别到就跳过
WAL 并告警。

识别只做“尽力而为”的判断，绝不因为探测失败而阻断启动：
- `/proc/self/mounts` 不存在（非 Linux）或解析失败 -> 返回 unknown，由调用方
  继续按“尝试 WAL 并在失败时回退”的处理。
- 采用**最长前缀匹配**定位挂载点，保证嵌套挂载（如 /mnt 与 /mnt/fast）判定正确。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# 已知网络/分布式文件系统的内核类型（小写）。
NETWORK_FS_TYPES: frozenset[str] = frozenset(
    {
        "nfs",
        "nfs4",
        "cifs",
        "smb",
        "smb2",
        "smb3",
        "smbfs",
        "fuse.sshfs",
        "fuse.glusterfs",
        "fuse.ceph",
        "fuse.s3fs",
        "fuse.juicefs",
        "9p",
        "afs",
        "gfs",
        "gfs2",
        "ocfs2",
        "lustre",
        "cephfs",
        "gpfs",
        "davfs",
        "webdav",
        "ncpfs",
        "sshfs",
    }
)

# 已知不适合 WAL 的伪文件系统（内存盘/覆盖层）：不是“网络盘”，但同样需要告警。
EPHEMERAL_FS_TYPES: frozenset[str] = frozenset({"tmpfs", "ramfs", "overlay", "overlayfs"})

# 默认的挂载表位置（测试可注入自定义内容）。
PROC_MOUNTS_PATH = Path("/proc/self/mounts")


@dataclass(frozen=True)
class FilesystemInfo:
    """数据库文件所在文件系统的探测结果。

    - `fs_type`：内核挂载类型（小写），探测失败时为空字符串。
    - `mount_point`：命中的挂载点，探测失败时为空字符串。
    - `is_network`：是否判定为网络/分布式文件系统（WAL 不安全）。
    - `is_ephemeral`：是否为 tmpfs/overlay 等临时文件系统（WAL 可用但易丢）。
    - `source`：探测来源（`proc_mounts` 或 `unknown`），便于日志排障。
    """

    fs_type: str
    mount_point: str
    is_network: bool
    is_ephemeral: bool
    source: str

    @property
    def is_known(self) -> bool:
        """是否成功识别出挂载信息。"""

        return self.source == "proc_mounts"


def _unescape_mount_field(text: str) -> str:
    """还原 /proc/mounts 中对空格、制表符与反斜杠的八进制转义。"""

    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 3 < len(text) and text[i + 1 : i + 4].isdigit():
            out.append(chr(int(text[i + 1 : i + 4], 8)))
            i += 4
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def parse_mounts(text: str) -> list[tuple[str, str, str]]:
    """把 `/proc/self/mounts` 文本解析为 `[(source, mount_point, fs_type), ...]`。"""

    entries: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        source, mount_point, fs_type = parts[0], parts[1], parts[2]
        entries.append(
            (_unescape_mount_field(source), _unescape_mount_field(mount_point), fs_type.lower())
        )
    return entries


def detect_filesystem(
    path: str | Path,
    *,
    mounts: Iterable[tuple[str, str, str]] | None = None,
    mounts_path: Path | str | None = None,
) -> FilesystemInfo:
    """探测 `path` 所在文件系统的类型与是否为网络挂载。

    参数：
    - `path`：数据库文件或其父目录（按 `resolve(strict=False)` 后做最长前缀匹配）。
    - `mounts`：可注入的挂载表条目，便于单元测试完全脱离真实系统。
    - `mounts_path`：可注入的挂载表文件路径；缺省读 `/proc/self/mounts`。

    返回值见 :class:`FilesystemInfo`；无法识别时 `source="unknown"` 且各项布尔为 False，
    调用方应自行决定（默认继续尝试 WAL）。
    """

    entries: list[tuple[str, str, str]] | None = (
        list(mounts) if mounts is not None else None
    )
    if entries is None:
        path_to_read = Path(mounts_path) if mounts_path is not None else PROC_MOUNTS_PATH
        try:
            entries = parse_mounts(path_to_read.read_text(encoding="utf-8"))
        except OSError:
            return FilesystemInfo("", "", False, False, "unknown")

    try:
        target = Path(path).expanduser().resolve(strict=False)
    except OSError:
        target = Path(path).expanduser().absolute()

    best: tuple[str, str] | None = None
    best_len = -1
    for _source, mount_point, fs_type in entries:
        if not mount_point:
            continue
        try:
            mount_path = Path(mount_point)
        except (OSError, ValueError):
            continue
        try:
            target.relative_to(mount_path)
        except ValueError:
            continue
        # 最长前缀胜出，正确处理 /mnt 与 /mnt/fast 这样的嵌套挂载。
        mount_len = len(str(mount_path))
        if mount_len > best_len:
            best_len = mount_len
            best = (mount_point, fs_type)

    if best is None:
        return FilesystemInfo("", "", False, False, "unknown")
    mount_point, fs_type = best
    return FilesystemInfo(
        fs_type=fs_type,
        mount_point=mount_point,
        is_network=fs_type in NETWORK_FS_TYPES,
        is_ephemeral=fs_type in EPHEMERAL_FS_TYPES,
        source="proc_mounts",
    )
