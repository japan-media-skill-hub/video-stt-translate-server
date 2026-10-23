"""R3 可靠性 P1 回归：租约身份 ABA、停机保护新 owner、fenced 检查点、copyback 真 sink。

全部使用真实 SQLite + 真实文件 + 真实 WorkerRuntime 代码路径；fake 只替换
执行器/HTTP 客户端。覆盖 Sol 第三轮报告中的 R3-1/R3-2/R3-4、旧 P1 的
checkpoint 动态借身份、finally 无条件 pop 新 lease、translate copyback 缺 sink。

约束：不引入 ImportError skip、不空吞异常；等待统一用事件式轮询。
"""

from __future__ import annotations

from dataclasses import replace
import json as jsonlib
import os
from pathlib import Path
import threading
from typing import Any, Callable

from whisper_stt_service.db import Database
from whisper_stt_service.progress import ProgressStore
from whisper_stt_service.repo.job_repository import JobRepository
from whisper_stt_service.repository import ClaimedTask
from whisper_stt_service.service import runtime as runtime_module
from whisper_stt_service.service.runtime import (
    CopyBackPlan,
    FencedLease,
    WorkerRuntime,
)

from tests.integration.test_runtime_fake_executors import (
    _settings as _base_settings,
    wait_until,
)


# --------------------------------------------------------------------------- 公用


def _settings(
    tmp_path: Path,
    *,
    extract_workers: int = 0,
    stt_workers: int = 0,
    stt_whisperx_workers: int = 0,
    translate_workers: int = 0,
    recovery_counts_as_attempt: bool = True,
):
    """基于集成测试配置派生本文件所需 worker 数/恢复策略。"""

    base = _base_settings(tmp_path)
    return replace(
        base,
        workers=replace(
            base.workers,
            extract_workers=extract_workers,
            stt_workers=stt_workers,
            stt_whisperx_workers=stt_whisperx_workers,
            translate_workers=translate_workers,
            poll_interval_sec=1,
        ),
        runtime=replace(
            base.runtime,
            recovery_counts_as_attempt=recovery_counts_as_attempt,
        ),
    )


def _repo(tmp_path: Path, **kwargs: Any) -> JobRepository:
    """构造指向 tmp 的仓储（真实 SQLite 文件）。"""

    db = Database(tmp_path / "state.db")
    db.init_schema()
    return JobRepository(
        db,
        log_root=tmp_path / "logs",
        artifact_root=tmp_path / "artifacts",
        **kwargs,
    )


def _runtime(
    repo: JobRepository,
    tmp_path: Path,
    *,
    settings=None,
    exit_callback: Callable[[], None] | None = None,
) -> WorkerRuntime:
    """构造不启动线程的 WorkerRuntime（每次构造使用新的 boot nonce）。"""

    return WorkerRuntime(
        repo=repo,
        progress_store=ProgressStore(300),
        settings=settings or _settings(tmp_path),
        config_path=tmp_path / "config.ini",
        model_path=str(tmp_path / "model"),
        exit_callback=exit_callback or (lambda: None),
        heartbeat_interval_sec=0.05,
    )


def _video(tmp_path: Path, name: str = "sample.mp4") -> Path:
    """创建占位视频文件（fake/真实执行器都不读取其内容）。"""

    directory = tmp_path / "videos"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"fake-video")
    return path


def _lease(
    repo: JobRepository, claimed: ClaimedTask, worker_id: str
) -> FencedLease:
    """按领取结果构造租约对象（不启动心跳，避免测试后台线程）。"""

    return FencedLease(
        repo=repo,
        task_id=claimed.task_id,
        job_id=claimed.job_id,
        stage=claimed.stage,
        worker_id=worker_id,
        attempt=claimed.attempt,
        lease_timeout_sec=60,
        heartbeat_interval_sec=60.0,
    )


def _expire_and_reap(repo: JobRepository, task_id: str) -> None:
    """把租约设为已过期并运行 reaper（真实 SQLite 状态流转）。"""

    past = "2000-01-01T00:00:00+00:00"
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE tasks SET lease_expires_at=? WHERE task_id=?", (past, task_id)
        )
    repo.reclaim_expired_leases()


def _task_row(repo: JobRepository, task_id: str):
    """直接读 SQLite 的任务关键列（最终断言来源）。"""

    with repo.db.connect() as conn:
        return conn.execute(
            "SELECT status, attempt, lease_owner, last_error FROM tasks WHERE task_id=?",
            (task_id,),
        ).fetchone()


def _write_staged(staging, content: str) -> None:
    """模拟执行器只写本实例私有暂存文件。"""

    staging.staged_path.parent.mkdir(parents=True, exist_ok=True)
    staging.staged_path.write_text(content, encoding="utf-8")


def _log_file(repo: JobRepository, task_id: str) -> str:
    """读取任务日志路径（get_job_detail/list_tasks 不保证返回该列）。"""

    with repo.db.connect() as conn:
        row = conn.execute(
            "SELECT log_file FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
    assert row is not None
    return str(row["log_file"])


def _read_events(log_file: str) -> list[dict[str, Any]]:
    """读取 task.log jsonl 事件。"""

    path = Path(log_file)
    if not path.is_file():
        return []
    return [
        jsonlib.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _force_prior_stages_succeeded(repo: JobRepository, job_id: str) -> None:
    """把 extract/stt 直接置 succeeded，便于单独驱动 translate。"""

    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE tasks SET status='succeeded', finished_at=updated_at "
            "WHERE job_id=? AND stage IN ('extract','stt')",
            (job_id,),
        )


def _translate_task(repo: JobRepository, job_id: str) -> dict[str, Any]:
    """返回 translate 任务行（含 log_file）。"""

    detail = repo.get_job_detail(job_id)
    assert detail is not None
    for task in detail["tasks"]:
        if task["stage"] == "translate":
            return task
    raise AssertionError("translate task missing")


class _FakeResponse:
    """最小 requests.Response 替身。"""

    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        """保存状态码与 JSON 响应体。"""

        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        """2xx 不抛错。"""

        if self.status_code >= 400:
            raise RuntimeError(f"http_status={self.status_code}")

    def json(self) -> dict[str, Any]:
        """返回预置 JSON。"""

        return self._payload


def _ok_post(url: str, *, headers=None, json=None, timeout=None):
    """fake HTTP：按请求里的 id 返回 zh 译文（真实 run_translate 路径）。"""

    user_content = str(json["messages"][1]["content"])
    payload = jsonlib.loads(user_content.split("\n\n")[-1])
    ids = [int(item["id"]) for item in payload]
    body = jsonlib.dumps(
        [{"id": item_id, "text_zh": f"zh-{item_id}"} for item_id in ids],
        ensure_ascii=False,
    )
    return _FakeResponse(
        200, {"choices": [{"message": {"content": body}}]}
    )


def _write_translate_config(tmp_path: Path) -> Path:
    """写最小 LLM/translation 配置（真实 run_translate 读取）。"""

    path = tmp_path / "config.ini"
    path.write_text(
        "\n".join(
            [
                "[llm]",
                "base_url = https://fake.example/v1",
                "api_key = fake-key",
                "model = fake-model",
                "",
                "[translation]",
                "parallel = 1",
                "retry = 1",
                "chunk_minutes = 30",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_ja_srt(path: Path) -> None:
    """写一份 2 条目的日文 SRT。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "1\n00:00:01,000 --> 00:00:01,900\nこんにちは\n\n"
        "2\n00:00:02,000 --> 00:00:02,900\nさようなら\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- R3-4 ABA


def test_cross_runtime_retry_attempt_reset_old_owner_publish_rejected(
    tmp_path: Path,
) -> None:
    """跨 runtime 同名 worker + reaper + API retry(attempt=0)：旧 publish 必须被拒。

    真实 SQLite + 真实文件：旧 extract-w1-<nonceA>/attempt0 的租约在 reaper 后
    变成 failed/attempt1，retry_job 重置 attempt0，新 runtime 的
    extract-w1-<nonceB>/attempt0 领取；旧 publish 不得覆盖正式产物。
    """

    repo = _repo(tmp_path)
    created = repo.enqueue(str(_video(tmp_path, "aba.mp4")), "ja")
    old_runtime = _runtime(repo, tmp_path)
    old_owner = old_runtime._worker_identity("extract", 1)
    assert old_owner.startswith("extract-w1-")

    first = repo.claim_next("extract", old_owner, 60)
    assert first is not None and first.attempt == 0
    old_lease = _lease(repo, first, old_owner)
    old_ctx = repo.get_task_execution_context(first.task_id)
    assert old_ctx is not None
    old_staging = old_runtime._build_staging(old_ctx, "extract", old_lease)
    for owned in old_staging.iter_staged_paths():
        old_staging.note_owned(owned)
    _write_staged(old_staging, "STALE OLD RESULT")
    final_path = Path(old_ctx.task_config["io"]["output_path"])

    # 让 reaper 直接超预算 -> failed/attempt1（真实 0 重试预算），job 级联 failed。
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE tasks SET max_retries=0 WHERE task_id=?", (first.task_id,)
        )
    _expire_and_reap(repo, first.task_id)
    row = _task_row(repo, first.task_id)
    assert (row["status"], row["attempt"]) == ("failed", 1)
    assert repo.get_job_status(created.job_id) == "failed"

    # A.7：API retry 把 attempt 重置为 0，制造与旧执行相同的 attempt。
    retried = repo.retry_job(created.job_id)
    assert retried["retried_tasks"] >= 1
    assert _task_row(repo, first.task_id)["attempt"] == 0

    new_runtime = _runtime(repo, tmp_path)
    new_owner = new_runtime._worker_identity("extract", 1)
    assert new_owner.startswith("extract-w1-")
    assert new_owner != old_owner  # boot nonce 让 owner 身份唯一
    second = repo.claim_next("extract", new_owner, 60)
    assert second is not None and second.attempt == 0
    new_lease = _lease(repo, second, new_owner)
    new_ctx = repo.get_task_execution_context(second.task_id)
    assert new_ctx is not None

    # 旧执行实例此刻醒来发布：身份（owner）不匹配，必须拒绝且不触碰正式文件。
    published_stale, stale_status = old_runtime._publish_stage_outputs(
        old_ctx, old_lease, "extract", old_staging
    )
    assert stale_status == "fenced_stale_result"
    assert published_stale == []
    assert not final_path.exists()
    assert _task_row(repo, first.task_id)["lease_owner"] == new_owner

    new_staging = new_runtime._build_staging(new_ctx, "extract", new_lease)
    _write_staged(new_staging, "NEW OWNER RESULT")
    published, status = new_runtime._publish_stage_outputs(
        new_ctx, new_lease, "extract", new_staging
    )
    assert status == "succeeded" and published
    assert final_path.read_text(encoding="utf-8") == "NEW OWNER RESULT"
    row = _task_row(repo, first.task_id)
    assert (row["status"], row["attempt"]) == ("succeeded", 0)


def test_debug_recovery_without_attempt_increment_blocks_old_owner(
    tmp_path: Path,
) -> None:
    """debug recovery_counts_as_attempt=false：attempt 仍为 0，旧 owner publish 仍被拒。

    这是 Sol R3-4 指出的第二条 ABA 路径：恢复不递增 attempt，如果 owner 也可
    复用，旧执行实例的 (task, owner, attempt) 会与新执行完全相同。
    """

    repo = _repo(tmp_path)
    created = repo.enqueue(str(_video(tmp_path, "debug-recovery.mp4")), "ja")
    old_runtime = _runtime(repo, tmp_path)
    old_owner = old_runtime._worker_identity("extract", 1)
    first = repo.claim_next("extract", old_owner, 60)
    assert first is not None and first.attempt == 0
    old_lease = _lease(repo, first, old_owner)
    old_ctx = repo.get_task_execution_context(first.task_id)
    assert old_ctx is not None
    old_staging = old_runtime._build_staging(old_ctx, "extract", old_lease)
    _write_staged(old_staging, "STALE DEBUG RESULT")

    restarted = _runtime(
        repo,
        tmp_path,
        settings=_settings(tmp_path, recovery_counts_as_attempt=False),
    )
    restarted.start()
    try:
        assert restarted.startup_stats()["recovered"] == 1
        row = _task_row(repo, first.task_id)
        assert (row["status"], row["attempt"], row["lease_owner"]) == (
            "queued",
            0,
            None,
        )

        new_owner = restarted._worker_identity("extract", 1)
        assert new_owner != old_owner
        second = repo.claim_next("extract", new_owner, 60)
        assert second is not None and second.attempt == 0

        published_stale, stale_status = old_runtime._publish_stage_outputs(
            old_ctx, old_lease, "extract", old_staging
        )
        assert (published_stale, stale_status) == ([], "fenced_stale_result")
        final_path = Path(old_ctx.task_config["io"]["output_path"])
        assert not final_path.exists()

        new_lease = _lease(repo, second, new_owner)
        new_ctx = repo.get_task_execution_context(second.task_id)
        assert new_ctx is not None
        new_staging = restarted._build_staging(new_ctx, "extract", new_lease)
        _write_staged(new_staging, "NEW DEBUG RESULT")
        _, status = restarted._publish_stage_outputs(
            new_ctx, new_lease, "extract", new_staging
        )
        assert status == "succeeded"
        assert final_path.read_text(encoding="utf-8") == "NEW DEBUG RESULT"
        # 其余阶段没有 worker，job 聚合仍是 queued；关键是失败状态已被清理。
        assert repo.get_job_status(created.job_id) == "queued"
    finally:
        restarted.stop(timeout_sec=2)


# --------------------------------------------------------------------------- R3-1 停机


def test_stop_does_not_revoke_new_owner_after_takeover(tmp_path: Path) -> None:
    """旧 runtime stop() 不得按 task_id 撤掉已被新 owner/attempt 接管的 claimed。"""

    repo = _repo(tmp_path)
    repo.enqueue(str(_video(tmp_path, "stop-takeover.mp4")), "ja")
    old_runtime = _runtime(repo, tmp_path)
    old_owner = old_runtime._worker_identity("extract", 1)
    first = repo.claim_next("extract", old_owner, 60)
    assert first is not None
    old_lease = _lease(repo, first, old_owner)
    with old_runtime._state_lock:
        old_runtime._inflight_leases[first.task_id] = old_lease

    # 真实接管：旧租约过期 -> reaper 回收 attempt1 -> 新 runtime 领取。
    _expire_and_reap(repo, first.task_id)
    new_runtime = _runtime(repo, tmp_path)
    new_owner = new_runtime._worker_identity("extract", 1)
    second = repo.claim_next("extract", new_owner, 60)
    assert second is not None and second.attempt == 1

    old_runtime.stop(timeout_sec=1)

    row = _task_row(repo, first.task_id)
    assert (row["status"], row["lease_owner"], row["attempt"]) == (
        "claimed",
        new_owner,
        1,
    )
    assert old_lease.lost is True


def test_force_drain_timeout_does_not_revoke_new_owner(tmp_path: Path) -> None:
    """drain 超时强制回滚同样必须 tuple-fenced，不能撤新 owner。"""

    repo = _repo(tmp_path)
    repo.enqueue(str(_video(tmp_path, "drain-takeover.mp4")), "ja")
    old_runtime = _runtime(repo, tmp_path)
    old_owner = old_runtime._worker_identity("extract", 1)
    first = repo.claim_next("extract", old_owner, 60)
    assert first is not None
    old_lease = _lease(repo, first, old_owner)
    with old_runtime._state_lock:
        old_runtime._inflight_leases[first.task_id] = old_lease

    _expire_and_reap(repo, first.task_id)
    new_runtime = _runtime(repo, tmp_path)
    new_owner = new_runtime._worker_identity("extract", 1)
    second = repo.claim_next("extract", new_owner, 60)
    assert second is not None and second.attempt == 1

    old_runtime.request_shutdown("unit_test")
    old_runtime._force_drain_timeout()

    row = _task_row(repo, first.task_id)
    assert (row["status"], row["lease_owner"], row["attempt"]) == (
        "claimed",
        new_owner,
        1,
    )
    assert old_lease.lost is True


# --------------------------------------------------------------------------- 检查点身份


def _translate_setup(
    repo: JobRepository, tmp_path: Path
) -> tuple[str, ClaimedTask, Any, FencedLease]:
    """创建仅剩 translate 可执行的 job，并由指定 runtime 领取。

    返回 (job_id, claimed, ctx, lease)。
    """

    created = repo.enqueue(str(_video(tmp_path, "translate.mp4")), "ja")
    _force_prior_stages_succeeded(repo, created.job_id)
    task = _translate_task(repo, created.job_id)
    runtime = _runtime(repo, tmp_path)
    owner = runtime._worker_identity("translate", 1)
    claimed = repo.claim_next("translate", owner, 60)
    assert claimed is not None
    assert claimed.task_id == task["task_id"]
    ctx = repo.get_task_execution_context(claimed.task_id)
    assert ctx is not None
    lease = _lease(repo, claimed, owner)
    return created.job_id, claimed, ctx, lease


def test_checkpoint_publisher_freezes_lease_and_canonical_absolute_path(
    tmp_path: Path,
) -> None:
    """旧 callback 不能借新 lease；非 canonical 绝对目标被规范化到正式检查点。"""

    repo = _repo(tmp_path)
    job_id, claimed, ctx, old_lease = _translate_setup(repo, tmp_path)
    runtime = _runtime(repo, tmp_path)
    staging = runtime._build_staging(ctx, "translate", old_lease)
    canonical = staging.final_path.with_name(
        staging.final_path.name + ".progress.json"
    )
    old_publish = runtime._make_checkpoint_publisher(ctx, staging, old_lease)

    # 1) 绝对路径规范化：传入另一个目录的同名文件，必须仍写 canonical 绝对目标。
    elsewhere = tmp_path / "elsewhere" / canonical.name
    elsewhere.parent.mkdir(parents=True, exist_ok=True)
    assert old_publish(elsewhere, '{"owner": "OLD-CANONICAL"}') is True
    assert canonical.read_text(encoding="utf-8") == '{"owner": "OLD-CANONICAL"}'
    assert not elsewhere.exists()

    # 2) 接管：旧任务回 queued/attempt1 -> 新 runtime 领取并用新 lease 写检查点。
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE tasks SET status='queued', attempt=1, lease_owner=NULL, "
            "lease_expires_at=NULL WHERE task_id=?",
            (claimed.task_id,),
        )
    new_runtime = _runtime(repo, tmp_path)
    new_owner = new_runtime._worker_identity("translate", 1)
    new_claim = repo.claim_next("translate", new_owner, 60)
    assert new_claim is not None and new_claim.attempt == 1
    new_ctx = repo.get_task_execution_context(new_claim.task_id)
    assert new_ctx is not None
    new_lease = _lease(repo, new_claim, new_owner)
    new_staging = new_runtime._build_staging(new_ctx, "translate", new_lease)
    new_publish = new_runtime._make_checkpoint_publisher(
        new_ctx, new_staging, new_lease
    )
    assert new_publish(canonical, '{"owner": "NEW"}') is True
    assert canonical.read_text(encoding="utf-8") == '{"owner": "NEW"}'

    # 3) 旧 callback 只能用旧身份：fencing 拒绝，绝不覆盖新检查点。
    assert old_publish(canonical, '{"owner": "STALE"}') is False
    assert canonical.read_text(encoding="utf-8") == '{"owner": "NEW"}'
    assert _task_row(repo, claimed.task_id)["lease_owner"] == new_owner
    assert repo.get_job_status(job_id) == "running"


# --------------------------------------------------------------------------- finally 身份


def test_finally_keeps_new_inflight_lease_and_worker_state(tmp_path: Path) -> None:
    """旧执行实例 finally：不得 pop 新 lease，也不得清掉新执行实例的 worker 状态。"""

    repo = _repo(tmp_path)
    repo.enqueue(str(_video(tmp_path, "finally.mp4")), "ja")
    runtime = _runtime(repo, tmp_path)
    old_owner = runtime._worker_identity("extract", 1)
    claimed = repo.claim_next("extract", old_owner, 60)
    assert claimed is not None
    new_owner = "extract-w1-new-instance"
    holder: dict[str, FencedLease] = {}

    def _fake_extract(
        input_video,
        output_wav,
        timeout_sec,
        *,
        progress_queue=None,
        task_id=None,
        worker_id=None,
        cancel_check=None,
    ):
        """写本实例暂存，并在返回前模拟任务被新执行实例接管。"""

        Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
        Path(output_wav).write_text("stale-result", encoding="utf-8")
        with repo.db.tx() as conn:
            conn.execute(
                "UPDATE tasks SET status='queued', attempt=1, lease_owner=NULL, "
                "lease_expires_at=NULL WHERE task_id=?",
                (claimed.task_id,),
            )
        new_claim = repo.claim_next("extract", new_owner, 60)
        assert new_claim is not None
        new_lease = FencedLease(
            repo=repo,
            task_id=new_claim.task_id,
            job_id=new_claim.job_id,
            stage="extract",
            worker_id=new_owner,
            attempt=new_claim.attempt,
            lease_timeout_sec=60,
            heartbeat_interval_sec=60.0,
        )
        holder["new_lease"] = new_lease
        with runtime._state_lock:
            runtime._inflight_leases[new_claim.task_id] = new_lease
        runtime._set_worker_state(new_owner, "extract", new_claim.task_id)
        return {"effective_config": {}}

    original = runtime_module.run_extract
    runtime_module.run_extract = _fake_extract
    try:
        runtime._execute_claimed(claimed, old_owner, "extract")
    finally:
        runtime_module.run_extract = original

    new_lease = holder["new_lease"]
    assert runtime._inflight_leases[claimed.task_id] is new_lease
    workers = runtime.active_workers()
    assert workers[old_owner]["task_id"] is None
    assert workers[new_owner]["task_id"] == claimed.task_id
    row = _task_row(repo, claimed.task_id)
    assert (row["status"], row["lease_owner"], row["attempt"]) == (
        "claimed",
        new_owner,
        1,
    )
    with runtime._state_lock:
        runtime._inflight_leases.pop(claimed.task_id, None)


# --------------------------------------------------------------------------- R3-2 发布异常


def test_publish_oserror_keeps_worker_alive_and_retries(tmp_path: Path) -> None:
    """promotion 抛 OSError：任务按 fenced failure 重试，worker 线程不得退出。"""

    repo = _repo(tmp_path)
    runtime = _runtime(repo, tmp_path, settings=_settings(tmp_path, extract_workers=1))
    original_extract = runtime_module.run_extract

    def _fake_extract(
        input_video,
        output_wav,
        timeout_sec,
        *,
        progress_queue=None,
        task_id=None,
        worker_id=None,
        cancel_check=None,
    ):
        """只写 staging 的 fake extract。"""

        Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
        Path(output_wav).write_text("wav-content", encoding="utf-8")
        return {"effective_config": {}}

    original_publish = repo.publish_task_artifacts
    state = {"calls": 0}
    lock = threading.Lock()

    def _flaky_publish(*args, **kwargs):
        """第一次调用抛 OSError（模拟目标盘 I/O 错误），之后正常发布。"""

        with lock:
            state["calls"] += 1
            first = state["calls"] == 1
        if first:
            raise OSError("simulated destination I/O error")
        return original_publish(*args, **kwargs)

    runtime_module.run_extract = _fake_extract
    repo.publish_task_artifacts = _flaky_publish  # type: ignore[method-assign]
    runtime.start()
    try:
        created = repo.enqueue(str(_video(tmp_path, "publish-error.mp4")), "ja")
        detail = repo.get_job_detail(created.job_id)
        assert detail is not None
        extract_task = next(t for t in detail["tasks"] if t["stage"] == "extract")
        wait_until(
            lambda: _task_row(repo, extract_task["task_id"])["status"] == "succeeded",
            description="extract succeeded after publish retry",
        )
        row = _task_row(repo, extract_task["task_id"])
        assert row["attempt"] == 1  # 第一次发布失败消耗了一次 attempt
        assert state["calls"] >= 2
        worker_threads = [
            t
            for t in runtime._threads
            if t.name.startswith("worker-extract-w1-")
        ]
        assert worker_threads and all(t.is_alive() for t in worker_threads)
        events = _read_events(_log_file(repo, extract_task["task_id"]))
        failures = [e for e in events if e.get("event") == "task_failed"]
        assert failures, "publish failure must be logged as task_failed"
        assert str(failures[-1]["error"]).startswith("publish_failed:OSError")
    finally:
        runtime_module.run_extract = original_extract
        repo.publish_task_artifacts = original_publish  # type: ignore[method-assign]
        runtime.stop(timeout_sec=3)


# --------------------------------------------------------------------------- copyback 真 sink


def test_real_run_translate_copies_back_ja_and_zh_with_same_dir_temp(
    tmp_path: Path,
) -> None:
    """真 run_translate(fake HTTP) + WorkerRuntime：ja/zh 真实回写视频目录。

    同时断言 copyback 的 rename 对 (staged, final) 同目录：慢拷贝先落到目标
    目录内唯一临时文件，最终 rename 才在租约事务里执行（跨 FS 安全模式）。
    """

    repo = _repo(tmp_path)
    _write_translate_config(tmp_path)
    video = _video(tmp_path, "copyback.mp4")
    created = repo.enqueue(str(video), "ja")
    _force_prior_stages_succeeded(repo, created.job_id)
    task = _translate_task(repo, created.job_id)
    ja_path = Path(task["task_config"]["io"]["input_path"])
    _write_ja_srt(ja_path)

    runtime = _runtime(repo, tmp_path)
    owner = runtime._worker_identity("translate", 1)
    claimed = repo.claim_next("translate", owner, 60)
    assert claimed is not None
    ctx = repo.get_task_execution_context(claimed.task_id)
    assert ctx is not None

    from whisper_stt_service.executor.translate import run_translate as real_run

    def _run_with_fake_http(*args, **kwargs):
        """给真实 run_translate 注入 fake post_func。"""

        kwargs["post_func"] = _ok_post
        return real_run(*args, **kwargs)

    captured: list[dict[str, Any]] = []
    original_publish = repo.publish_task_artifacts

    def _capture_publish(task_id, **kwargs):
        """记录发布文件对与 prepublish 计划后委托真实实现。"""

        prepublish = kwargs.get("prepublish")
        captured.append(
            {
                "artifacts": [(str(a), str(b)) for a, b in kwargs["artifacts"]],
                "plan": getattr(prepublish, "__self__", None),
            }
        )
        return original_publish(task_id, **kwargs)

    original_run = runtime_module.run_translate
    runtime_module.run_translate = _run_with_fake_http
    repo.publish_task_artifacts = _capture_publish  # type: ignore[method-assign]
    try:
        runtime._execute_claimed(claimed, owner, "translate")
    finally:
        runtime_module.run_translate = original_run
        repo.publish_task_artifacts = original_publish  # type: ignore[method-assign]

    row = _task_row(repo, claimed.task_id)
    assert row["status"] == "succeeded"
    zh_final = Path(ctx.task_config["io"]["output_path"])
    assert zh_final.is_file()
    assert "zh-1" in zh_final.read_text(encoding="utf-8")

    video_dir = video.parent
    ja_copy = video_dir / ja_path.name
    zh_copy = video_dir / zh_final.name
    assert ja_copy.read_text(encoding="utf-8") == ja_path.read_text(encoding="utf-8")
    assert zh_copy.read_text(encoding="utf-8") == zh_final.read_text(encoding="utf-8")

    # copyback 通过 dir_fd 锚定的 prepublish 计划提交；主产物 rename 仍保持
    # (staged, final) 同目录（同文件系统），且发布后不残留任何临时文件。
    assert captured, "publish_task_artifacts must be called"
    plan = captured[0]["plan"]
    assert isinstance(plan, CopyBackPlan)
    assert plan.dir_path == video_dir
    assert len(plan.pairs) == 2
    for staged, final in captured[0]["artifacts"]:
        assert Path(staged).parent == Path(final).parent
    leftovers = [
        p.name
        for p in list(video_dir.iterdir()) + list(zh_final.parent.iterdir())
        if ".copyback-" in p.name or ".staging-" in p.name
    ]
    assert leftovers == []


def test_copyback_rejects_symlink_target_without_following(
    tmp_path: Path,
) -> None:
    """copy_back 目标为符号链接：拒绝写入，外部文件内容不变，任务按失败重试。"""

    repo = _repo(tmp_path)
    _write_translate_config(tmp_path)
    video = _video(tmp_path, "symlink.mp4")
    created = repo.enqueue(str(video), "ja")
    _force_prior_stages_succeeded(repo, created.job_id)
    task = _translate_task(repo, created.job_id)
    ja_path = Path(task["task_config"]["io"]["input_path"])
    _write_ja_srt(ja_path)
    outside = tmp_path / "outside-target.txt"
    outside.write_text("OUTSIDE", encoding="utf-8")
    zh_target = video.parent / Path(task["task_config"]["io"]["output_path"]).name
    zh_target.symlink_to(outside)

    runtime = _runtime(repo, tmp_path)
    owner = runtime._worker_identity("translate", 1)
    claimed = repo.claim_next("translate", owner, 60)
    assert claimed is not None
    ctx = repo.get_task_execution_context(claimed.task_id)
    assert ctx is not None

    def _fake_translate(*args, **kwargs):
        """只写 staging zh.srt 的 fake 执行器。"""

        output = Path(kwargs["output_zh_srt"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("zh-staged", encoding="utf-8")

    original_run = runtime_module.run_translate
    runtime_module.run_translate = _fake_translate
    try:
        runtime._execute_claimed(claimed, owner, "translate")
    finally:
        runtime_module.run_translate = original_run

    row = _task_row(repo, claimed.task_id)
    assert row["status"] == "queued"
    assert row["attempt"] == 1
    assert os.path.islink(zh_target)
    assert outside.read_text(encoding="utf-8") == "OUTSIDE"
    assert not (video.parent / ja_path.name).exists()
    events = _read_events(_log_file(repo, claimed.task_id))
    failures = [e for e in events if e.get("event") == "task_failed"]
    assert failures and "copy_back_target_symlink" in str(failures[-1]["error"])


def test_copyback_target_outside_roots_revalidated_at_execution(
    tmp_path: Path,
) -> None:
    """执行期重新校验 copy_back 目标：越界目标即使写进 DB 也被拒绝。"""

    repo = _repo(tmp_path)
    _write_translate_config(tmp_path)
    video = _video(tmp_path, "outside.mp4")
    created = repo.enqueue(str(video), "ja")
    _force_prior_stages_succeeded(repo, created.job_id)
    task = _translate_task(repo, created.job_id)
    ja_path = Path(task["task_config"]["io"]["input_path"])
    _write_ja_srt(ja_path)

    outside_dir = tmp_path / "elsewhere" / "copyback"
    outside_dir.parent.mkdir(parents=True, exist_ok=True)
    with repo.db.tx() as conn:
        raw = conn.execute(
            "SELECT task_config_json FROM tasks WHERE task_id=?",
            (task["task_id"],),
        ).fetchone()["task_config_json"]
        config = jsonlib.loads(raw)
        config["effective_config"]["copy_back"] = str(outside_dir)
        config["io"]["copy_back_dir"] = str(outside_dir)
        conn.execute(
            "UPDATE tasks SET task_config_json=? WHERE task_id=?",
            (jsonlib.dumps(config, ensure_ascii=False), task["task_id"]),
        )

    runtime = _runtime(repo, tmp_path)
    owner = runtime._worker_identity("translate", 1)
    claimed = repo.claim_next("translate", owner, 60)
    assert claimed is not None
    ctx = repo.get_task_execution_context(claimed.task_id)
    assert ctx is not None

    def _fake_translate(*args, **kwargs):
        """只写 staging zh.srt 的 fake 执行器。"""

        output = Path(kwargs["output_zh_srt"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("zh-staged", encoding="utf-8")

    original_run = runtime_module.run_translate
    runtime_module.run_translate = _fake_translate
    try:
        runtime._execute_claimed(claimed, owner, "translate")
    finally:
        runtime_module.run_translate = original_run

    row = _task_row(repo, claimed.task_id)
    assert row["status"] == "queued" and row["attempt"] == 1
    assert not outside_dir.exists()
    events = _read_events(_log_file(repo, claimed.task_id))
    failures = [e for e in events if e.get("event") == "task_failed"]
    assert failures and "copy_back_invalid:copy_back_outside_video_dir" in str(
        failures[-1]["error"]
    )


def test_copyback_rejects_directory_swapped_to_symlink_during_copy(
    tmp_path: Path,
) -> None:
    """慢拷贝完成后目录被替换为符号链接：事务内 dirfd 校验拒绝，不写 outside。

    回归 R3-6：准备期的一次性 roots/symlink 检查不能当作最终 sink 安全保证。
    plan 在慢拷贝前持有原目录 fd；publish 事务内 commit 发现 fd 真实路径已变，
    拒绝 rename 并回滚 DB；随后 close 仍能通过同一个 fd 清掉自己的临时文件。
    """

    repo = _repo(tmp_path)
    _write_translate_config(tmp_path)
    video = _video(tmp_path, "swap.mp4")
    created = repo.enqueue(str(video), "ja")
    _force_prior_stages_succeeded(repo, created.job_id)
    task = _translate_task(repo, created.job_id)
    ja_path = Path(task["task_config"]["io"]["input_path"])
    _write_ja_srt(ja_path)

    runtime = _runtime(repo, tmp_path)
    owner = runtime._worker_identity("translate", 1)
    claimed = repo.claim_next("translate", owner, 60)
    assert claimed is not None
    ctx = repo.get_task_execution_context(claimed.task_id)
    assert ctx is not None

    def _fake_translate(*args, **kwargs):
        """只写 staging zh.srt 的 fake 执行器。"""

        output = Path(kwargs["output_zh_srt"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("zh-staged", encoding="utf-8")

    video_dir = video.parent
    moved_dir = tmp_path / "videos-moved"
    outside_dir = tmp_path / "outside-copyback"
    outside_dir.mkdir()
    original_run = runtime_module.run_translate
    original_publish = repo.publish_task_artifacts

    def _swap_then_publish(task_id, **kwargs):
        """模拟慢拷贝后、发布事务前把目标目录替换成指向 outside 的符号链接。"""

        if not moved_dir.exists():
            video_dir.rename(moved_dir)
            video_dir.symlink_to(outside_dir, target_is_directory=True)
        return original_publish(task_id, **kwargs)

    runtime_module.run_translate = _fake_translate
    repo.publish_task_artifacts = _swap_then_publish  # type: ignore[method-assign]
    try:
        runtime._execute_claimed(claimed, owner, "translate")
    finally:
        runtime_module.run_translate = original_run
        repo.publish_task_artifacts = original_publish  # type: ignore[method-assign]

    row = _task_row(repo, claimed.task_id)
    assert row["status"] == "queued"
    assert row["attempt"] == 1
    assert os.path.islink(video_dir)
    # 绝不能往被替换后的 outside 目录写任何正式/临时文件。
    assert list(outside_dir.iterdir()) == []
    # 原目录（已 rename 到 videos-moved）里的 copyback 临时文件必须已被清理。
    assert [
        p.name for p in moved_dir.iterdir() if ".copyback-" in p.name
    ] == []
    events = _read_events(_log_file(repo, claimed.task_id))
    failures = [e for e in events if e.get("event") == "task_failed"]
    assert failures
    assert "copy_back_target_dir_changed" in str(failures[-1]["error"])


def test_stale_attempt_copyback_cannot_overwrite_new_owner(tmp_path: Path) -> None:
    """旧 attempt 预复制好的 copyback 临时文件，在接管后发布必须被拒绝。"""

    repo = _repo(tmp_path)
    video = _video(tmp_path, "stale-copyback.mp4")
    created = repo.enqueue(str(video), "ja")
    _force_prior_stages_succeeded(repo, created.job_id)
    task = _translate_task(repo, created.job_id)
    ja_path = Path(task["task_config"]["io"]["input_path"])
    _write_ja_srt(ja_path)

    old_runtime = _runtime(repo, tmp_path)
    old_owner = old_runtime._worker_identity("translate", 1)
    old_claim = repo.claim_next("translate", old_owner, 60)
    assert old_claim is not None
    old_ctx = repo.get_task_execution_context(old_claim.task_id)
    assert old_ctx is not None
    old_lease = _lease(repo, old_claim, old_owner)
    old_staging = old_runtime._build_staging(old_ctx, "translate", old_lease)
    for owned in old_staging.iter_staged_paths():
        old_staging.note_owned(owned)
    _write_staged(old_staging, "STALE ZH")
    old_task_cfg = old_ctx.task_config["effective_config"]
    old_plan = old_runtime._prepare_copy_back(
        old_ctx, old_lease, old_staging, old_task_cfg
    )
    assert len(old_plan.pairs) == 2

    # 接管：旧任务回 queued/attempt1，新 runtime 领取并成功发布新版本。
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE tasks SET status='queued', attempt=1, lease_owner=NULL, "
            "lease_expires_at=NULL WHERE task_id=?",
            (old_claim.task_id,),
        )
    new_runtime = _runtime(repo, tmp_path)
    new_owner = new_runtime._worker_identity("translate", 1)
    new_claim = repo.claim_next("translate", new_owner, 60)
    assert new_claim is not None and new_claim.attempt == 1
    new_ctx = repo.get_task_execution_context(new_claim.task_id)
    assert new_ctx is not None
    new_lease = _lease(repo, new_claim, new_owner)
    new_staging = new_runtime._build_staging(new_ctx, "translate", new_lease)
    for owned in new_staging.iter_staged_paths():
        new_staging.note_owned(owned)
    _write_staged(new_staging, "NEW ZH")
    new_task_cfg = new_ctx.task_config["effective_config"]
    new_plan = new_runtime._prepare_copy_back(
        new_ctx, new_lease, new_staging, new_task_cfg
    )
    try:
        published, status = new_runtime._publish_stage_outputs(
            new_ctx,
            new_lease,
            "translate",
            new_staging,
            copy_back_plan=new_plan,
        )
        assert status == "succeeded" and published

        video_dir = video.parent
        zh_target = video_dir / Path(new_ctx.task_config["io"]["output_path"]).name
        ja_target = video_dir / ja_path.name
        assert zh_target.read_text(encoding="utf-8") == "NEW ZH"

        # 旧 attempt 现在才发布：fencing 拒绝，copyback 目标保持新 owner 内容。
        stale_published, stale_status = old_runtime._publish_stage_outputs(
            old_ctx,
            old_lease,
            "translate",
            old_staging,
            copy_back_plan=old_plan,
        )
        assert (stale_published, stale_status) == ([], "fenced_stale_result")
        assert zh_target.read_text(encoding="utf-8") == "NEW ZH"
        assert ja_target.read_text(encoding="utf-8") == ja_path.read_text(
            encoding="utf-8"
        )
        # 旧实例自己的临时文件通过 dir_fd 清理，且不影响新 owner 的正式文件。
        old_plan.close()
        old_runtime._cleanup_staging(old_staging, keep=[])
        leftovers = [
            p.name
            for p in video_dir.iterdir()
            if ".copyback-" in p.name or ".staging-" in p.name
        ]
        assert leftovers == []
    finally:
        new_plan.close()
