"""产物原子写入工具。

所有执行器产物（字幕、检查点）都必须通过本模块落盘，避免进程被 kill 或
磁盘写满时留下半截文件：同目录临时文件 → flush + fsync → os.replace。
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Callable, Union

logger = logging.getLogger(__name__)


def _temp_path_for(path: Path) -> Path:
    """生成与目标同目录的临时文件名，保证 os.replace 处于同一文件系统。"""

    return path.with_name(f".tmp-{os.getpid()}-{uuid.uuid4().hex}.{path.name}")


def atomic_write_bytes(path: Union[str, Path], data: bytes) -> Path:
    """原子写入字节内容：同目录临时文件 + fsync + os.replace。

    任何异常都会删除临时文件并重新抛出，已有产物文件保持不变。
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _temp_path_for(target)
    try:
        # 先写临时文件并强制落盘，再用 os.replace 原子替换目标。
        with open(tmp_path, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        # 失败时清理临时文件，不影响已有产物，再把原始异常抛给调用方。
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "atomic_write cleanup failed: tmp=%s error=%s", tmp_path, exc
            )
        raise
    return target


def atomic_write_text(path: Union[str, Path], text: str, encoding: str = "utf-8") -> Path:
    """原子写入文本内容（默认 UTF-8），失败时不破坏已有文件。"""

    return atomic_write_bytes(path, text.encode(encoding))


class StaleWriteRejected(RuntimeError):
    """fenced 写入被拒：发布回调判定当前写入方已失去所有权（stale worker）。

    runtime 侧 `checkpoint_publish` 在事务内确认 `lease_owner`/`attempt` 失败时返回 False，
    此处即抛出该异常，调用方必须放弃本 attempt 的结果，不得写正式产物。
    """

    code = "fenced_stale_write"


def atomic_write_text_fenced(
    path: Union[str, Path],
    text: str,
    publish: Callable[[Path, str], bool] | None = None,
    *,
    encoding: str = "utf-8",
) -> Path:
    """带发布钩子的原子写：由 `publish` 在租约事务内确认所有权并全权落盘。

    `publish(path, text)` 的约定：
    - runtime 在 **BEGIN 事务** 中校验 `status='claimed' AND lease_owner=:worker AND attempt=:attempt`；
    - 校验通过时回调必须**自己完成落盘**（通常在同一事务内调用 `atomic_write_text`）
      并返回 True；返回 True 表示“发布已全权完成”，本函数**不会**再写第二次。
      这点是安全契约：事务提交后租约锁已释放，若 helper 再写一次，旧 worker 就能
      在新 owner 写入后覆盖 canonical 检查点（R3-3）；
    - 校验失败返回 False 表示本写入方已 stale，本函数抛 `StaleWriteRejected`，
      且不写任何字节；
    - 回调返回值必须是 bool；True/False 之外的返回值视为调用方编程错误并抛
      `TypeError`（宁可失败也不绕过租约在锁外落盘）；
    - 未提供 `publish` 时退化为普通 `atomic_write_text`（保持既有行为）。
    """

    if publish is None:
        return atomic_write_text(path, text, encoding=encoding)
    accepted = publish(Path(path), text)
    if accepted is False:
        raise StaleWriteRejected(f"stale write rejected: {path}")
    if accepted is not True:
        raise TypeError(
            "publish callback must return bool: "
            f"got {type(accepted).__name__}"
        )
    # 回调已全权落盘（含事务内提交）：此处绝不能再写，否则锁外二写会覆盖新 owner。
    return Path(path)
