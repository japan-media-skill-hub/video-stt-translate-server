"""真实跨文件系统 copyback 集成回归（R3-6 最终 sink）。

环境分工（由父指定）：
- artifact/staging（含 SQLite DB）放在当前 worktree 的 tmp/ 下唯一 tempdir；
- video 与 copyback 目标使用 pytest tmp_path（/tmp），从而形成真实 st_dev 不同的
  跨文件系统场景；
- 若运行环境里两者恰好同一 st_dev，只跳过这一个环境条件（不是产品错误）。

路径全部走真实 WorkerRuntime + 真实 run_translate（仅 HTTP 用 fake post_func）。
断言：ja/zh 均成功回写、无 EXDEV；copyback 临时文件与 rename 走目标目录自身的
同目录 fd（relative name + dir_fd），不是跨盘 os.replace。
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

import pytest

from whisper_stt_service.db import Database
from whisper_stt_service.repo.job_repository import JobRepository
from whisper_stt_service.service import runtime as runtime_module
from whisper_stt_service.service.runtime import CopyBackPlan

from tests.integration.test_runtime_lease_identity_and_copyback import (
    _force_prior_stages_succeeded,
    _log_file,
    _ok_post,
    _read_events,
    _runtime,
    _task_row,
    _translate_task,
    _video,
    _write_ja_srt,
    _write_translate_config,
)


def _worktree_root() -> Path:
    """返回当前 worktree 根目录（tests/integration/<file> 的上三级）。"""

    return Path(__file__).resolve().parents[2]


def test_cross_filesystem_copyback_uses_same_dir_dirfd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实跨 FS：artifact 在 worktree，copyback 目标在 /tmp，ja/zh 均成功且无 EXDEV。"""

    worktree = _worktree_root()
    base = worktree / "tmp"
    base.mkdir(parents=True, exist_ok=True)
    base_resolved = base.resolve()
    temp_handle = tempfile.TemporaryDirectory(
        prefix="crossfs-copyback-", dir=str(base_resolved)
    )
    unique = Path(temp_handle.name)
    unique_resolved = unique.resolve()
    # 准备期先确认“唯一路径”确实位于 worktree/tmp 下，避免清理时越界。
    assert unique_resolved.parent == base_resolved
    assert unique_resolved.name.startswith("crossfs-copyback-")
    assert not unique_resolved.is_symlink()

    try:
        artifact_root = unique_resolved / "artifacts"
        log_root = unique_resolved / "logs"
        artifact_root.mkdir(parents=True, exist_ok=True)
        db = Database(unique_resolved / "state.db")
        db.init_schema()
        repo = JobRepository(db, log_root=log_root, artifact_root=artifact_root)

        _write_translate_config(tmp_path)
        video = _video(tmp_path, "crossfs.mp4")
        video_dir = video.parent
        artifact_dev = os.stat(artifact_root).st_dev
        target_dev = os.stat(video_dir).st_dev
        if artifact_dev == target_dev:
            pytest.skip(
                "artifact 与 copyback 目标同一文件系统 (st_dev="
                f"{artifact_dev})，跨 FS 条件不成立"
            )
        assert artifact_dev != target_dev

        created = repo.enqueue(str(video), "ja")
        _force_prior_stages_succeeded(repo, created.job_id)
        task = _translate_task(repo, created.job_id)
        ja_path = Path(task["task_config"]["io"]["input_path"])
        _write_ja_srt(ja_path)
        # 输入 ja.srt 必须落在 worktree 的 artifact 根下（跨 FS 场景的一端）。
        assert str(ja_path.resolve()).startswith(str(artifact_root.resolve()))

        runtime = _runtime(repo, tmp_path)
        owner = runtime._worker_identity("translate", 1)
        claimed = repo.claim_next("translate", owner, 60)
        assert claimed is not None
        ctx = repo.get_task_execution_context(claimed.task_id)
        assert ctx is not None
        zh_final = Path(ctx.task_config["io"]["output_path"])
        assert str(zh_final.resolve()).startswith(str(artifact_root.resolve()))

        from whisper_stt_service.executor.translate import run_translate as real_run

        def _run_with_fake_http(*args, **kwargs):
            """真实 run_translate + fake HTTP（不付费、不触网）。"""

            kwargs["post_func"] = _ok_post
            return real_run(*args, **kwargs)

        commits: list[dict[str, Any]] = []
        original_commit = CopyBackPlan.commit

        def _record_commit(plan: CopyBackPlan) -> None:
            """记录 commit 时的目标目录/fd 真实路径与相对 rename 对。"""

            commits.append(
                {
                    "dir_path": plan.dir_path,
                    "fd_path": runtime_module._read_fd_path(plan.dir_fd),
                    "pairs": list(plan.pairs),
                }
            )
            original_commit(plan)

        monkeypatch.setattr(CopyBackPlan, "commit", _record_commit)
        original_run = runtime_module.run_translate
        runtime_module.run_translate = _run_with_fake_http
        try:
            runtime._execute_claimed(claimed, owner, "translate")
        finally:
            runtime_module.run_translate = original_run

        row = _task_row(repo, claimed.task_id)
        assert row["status"] == "succeeded", row["last_error"]
        assert zh_final.is_file()
        assert "zh-1" in zh_final.read_text(encoding="utf-8")

        ja_copy = video_dir / ja_path.name
        zh_copy = video_dir / zh_final.name
        assert ja_copy.read_text(encoding="utf-8") == ja_path.read_text(
            encoding="utf-8"
        )
        assert zh_copy.read_text(encoding="utf-8") == zh_final.read_text(
            encoding="utf-8"
        )

        # copyback 走目标目录自身的 fd：dir_path/fd 真实路径都必须是 video_dir，
        # rename 对使用相对文件名（同目录同 FS，不可能 EXDEV）。
        assert len(commits) == 1, commits
        commit = commits[0]
        assert commit["dir_path"] == video_dir
        assert commit["fd_path"] == video_dir
        assert len(commit["pairs"]) == 2
        for temp_name, target_name in commit["pairs"]:
            assert os.sep not in temp_name
            assert os.sep not in target_name
            assert target_name in {ja_path.name, zh_final.name}

        # 成功路径不允许出现 EXDEV / cross-device 错误。
        events = _read_events(_log_file(repo, claimed.task_id))
        errors = [str(e.get("error")) for e in events if e.get("error")]
        assert not any("EXDEV" in err or "cross-device" in err.lower() for err in errors)
        assert not any(e.get("event") == "task_failed" for e in events)

        # 两端目录均不得残留 copyback/staging 临时文件。
        for directory in (video_dir, zh_final.parent):
            leftovers = [
                p.name
                for p in directory.iterdir()
                if ".copyback-" in p.name or ".staging-" in p.name
            ]
            assert leftovers == [], (directory, leftovers)
    finally:
        # cleanup guard：清理前再次确认 resolved 父目录与名字前缀，只删自己创建的
        # 唯一 TemporaryDirectory；校验失败时不执行清理（宁可泄漏也不越界删除）。
        current = unique.resolve()
        assert current.parent == base_resolved
        assert current.name.startswith("crossfs-copyback-")
        assert not current.is_symlink()
        temp_handle.cleanup()
