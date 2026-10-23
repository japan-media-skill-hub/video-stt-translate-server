"""SPEC F/G 仓储层观测能力测试。

覆盖：
- claim_next 延迟有界采样与 P95/count/busy 指标（注入 clock，不依赖真实时间）；
- sqlite busy 只计数并原样重抛，绝不吞错；
- list_finished_tasks_since 统计窗口内终态（含 blocked/cancelled）；
- task 日志统一 SPEC 字段（含 request_id）；
- enqueue 的 request_id 经 task_config.metadata 固化；
- copy_back 全局 [translation] 默认值生效且仍受 roots 约束。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from whisper_stt_service.repo.database import Database
from whisper_stt_service.repo.job_repository import JobRepository


class FakeClock:
    """可手动推进的单调时钟，用于精确验证采样耗时。

    `step_sec` 非 0 时，每次**读取**都会前进该步长：claim_next 会在开始/结束
    各读一次，因此一次调用的耗时恰为 `step_sec`，不依赖真实时间。
    """

    def __init__(self, start: float = 0.0, step_sec: float = 0.0) -> None:
        self.value = float(start)
        self.step_sec = float(step_sec)

    def __call__(self) -> float:
        self.value += self.step_sec
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


def _repo(tmp_path: Path, *, clock: FakeClock | None = None, **kwargs) -> JobRepository:
    """构造独立仓储（产物/日志落在 tmp_path）。"""

    db = Database(tmp_path / "q.db")
    db.init_schema()
    clock = clock or FakeClock()
    repo = JobRepository(
        db,
        log_root=tmp_path / "logs",
        artifact_root=tmp_path / "artifacts",
        clock=clock,
        **kwargs,
    )
    repo._test_clock = clock  # type: ignore[attr-defined]
    return repo


# ------------------------------------------------------ claim 指标


def test_claim_metrics_empty_initially(tmp_path: Path) -> None:
    """未调用过 claim_next 时指标为零且百分位为 None。"""

    repo = _repo(tmp_path)
    metrics = repo.get_claim_metrics()
    assert metrics["count"] == 0
    assert metrics["p95_ms"] is None
    assert metrics["busy"] == 0


def test_claim_metrics_records_elapsed_time(tmp_path: Path) -> None:
    """注入 clock 后能精确得到每次 claim 的耗时。"""

    # 每次读时钟前进 0.25s：一次 claim 读两次 → 耗时恰为 250ms。
    repo = _repo(tmp_path, clock=FakeClock(step_sec=0.25))
    claimed = repo.claim_next("extract", "w1", 60)
    assert claimed is None  # 空队列也要计时

    metrics = repo.get_claim_metrics()
    assert metrics["count"] == 1
    assert metrics["p95_ms"] == pytest.approx(250.0, abs=1.0)
    assert metrics["samples"] == 1


def test_claim_metrics_p95_with_multiple_samples(tmp_path: Path) -> None:
    """多次调用后 P95 取窗口内偏高分位。"""

    repo = _repo(tmp_path, clock=FakeClock(step_sec=0.3))
    repo.enqueue("/tmp/a.mp4", "ja")
    for _ in range(3):
        repo.claim_next("extract", "w1", 60)
    metrics = repo.get_claim_metrics()
    assert metrics["count"] == 3
    assert metrics["p95_ms"] == pytest.approx(300.0, abs=1.0)


def test_claim_metrics_window_is_bounded(tmp_path: Path) -> None:
    """采样窗口有界：超出容量后只保留最近的样本，count 仍然累计。"""

    from whisper_stt_service.repo import job_repository as jr

    repo = _repo(tmp_path)
    for _ in range(jr._CLAIM_SAMPLE_WINDOW + 20):
        repo.claim_next("extract", "w1", 60)
    metrics = repo.get_claim_metrics()
    assert metrics["samples"] == jr._CLAIM_SAMPLE_WINDOW
    assert metrics["count"] == jr._CLAIM_SAMPLE_WINDOW + 20


def test_claim_busy_counts_and_reraises(tmp_path: Path, monkeypatch) -> None:
    """sqlite busy 只计数并原样重抛，异常不会被吞掉。"""

    repo = _repo(tmp_path)

    def _boom(conn, task_id):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(repo, "_fetch_task_row", _boom)
    monkeypatch.setattr(repo, "_claim_next_impl", lambda *a, **k: _boom(None, None))
    with pytest.raises(sqlite3.OperationalError):
        repo.claim_next("extract", "w1", 60)

    metrics = repo.get_claim_metrics()
    assert metrics["busy"] == 1
    assert metrics["errors"] == 0
    assert metrics["count"] == 1


def test_claim_non_busy_error_counts_separately(tmp_path: Path, monkeypatch) -> None:
    """非 busy 的 sqlite 错误计入 errors，同样重抛。"""

    repo = _repo(tmp_path)

    def _boom(*args, **kwargs):
        raise sqlite3.OperationalError("no such table: tasks")

    monkeypatch.setattr(repo, "_claim_next_impl", _boom)
    with pytest.raises(sqlite3.OperationalError):
        repo.claim_next("extract", "w1", 60)
    metrics = repo.get_claim_metrics()
    assert metrics["errors"] == 1
    assert metrics["busy"] == 0


def test_claim_metrics_thread_safe(tmp_path: Path) -> None:
    """并发调用后计数准确，不丢样本。"""

    import threading

    repo = _repo(tmp_path)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(10):
                repo.claim_next("extract", "w1", 60)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert repo.get_claim_metrics()["count"] == 40


# --------------------------------------------- list_finished_tasks_since


def test_list_finished_tasks_since_includes_terminal_states(tmp_path: Path) -> None:
    """窗口内终态任务应全部返回，包含 blocked 与 cancelled。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/done.mp4", "ja")
    detail = repo.get_job_detail(created.job_id)
    first = detail["tasks"][0]["task_id"]
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE tasks SET status='succeeded', finished_at=? WHERE task_id=?",
            ("2026-10-22T00:00:00+00:00", first),
        )
        conn.execute(
            "UPDATE tasks SET status='blocked', finished_at=? WHERE job_id=? AND stage='translate'",
            ("2026-10-22T00:00:01+00:00", created.job_id),
        )
        conn.execute(
            "UPDATE tasks SET status='cancelled', finished_at=? WHERE job_id=? AND stage='stt'",
            ("2026-10-22T00:00:02+00:00", created.job_id),
        )

    rows = repo.list_finished_tasks_since("2026-10-21T00:00:00+00:00")
    statuses = {row["status"] for row in rows}
    assert {"succeeded", "blocked", "cancelled"} <= statuses
    assert all(set(row) == {"stage", "status", "finished_at"} for row in rows)


def test_list_finished_tasks_since_filters_by_window(tmp_path: Path) -> None:
    """早于 since 的任务不返回。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/old.mp4", "ja")
    task_id = repo.get_job_detail(created.job_id)["tasks"][0]["task_id"]
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE tasks SET status='succeeded', finished_at=? WHERE task_id=?",
            ("2020-01-01T00:00:00+00:00", task_id),
        )
    assert repo.list_finished_tasks_since("2026-01-01T00:00:00+00:00") == []


def test_list_finished_tasks_since_used_by_queue_summary(tmp_path: Path) -> None:
    """/queue/summary 的吞吐读取器能直接消费该方法。"""

    from whisper_stt_service.api.routes.queue import _throughput_rows_from_repo
    from datetime import datetime, timezone

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/q.mp4", "ja")
    task_id = repo.get_job_detail(created.job_id)["tasks"][0]["task_id"]
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE tasks SET status='succeeded', finished_at=? WHERE task_id=?",
            ("2026-10-22T00:00:00+00:00", task_id),
        )
    rows = _throughput_rows_from_repo(
        repo, window_sec=300, now=datetime(2026, 10, 22, 0, 1, tzinfo=timezone.utc)
    )
    assert rows is not None and len(rows) == 1


# --------------------------------------------------------- task 日志字段

def _log_file(repo: JobRepository, task_id: str) -> str:
    """读取任务日志文件路径（get_job_detail 不返回 log_file，需直查库）。"""

    with repo.db.connect() as conn:
        row = conn.execute("SELECT log_file FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    return str(row["log_file"])


_SPEC_FIELDS = [
    "timestamp",
    "request_id",
    "job_id",
    "task_id",
    "stage",
    "worker_id",
    "event",
    "result",
    "attempt",
    "effective_config",
    "error",
]


def test_task_log_event_has_all_spec_fields(tmp_path: Path) -> None:
    """仓储侧写入的 task 日志必须包含 SPEC G 的全部字段（顺序固定）。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/log.mp4", "ja", request_id="req-abc")
    task = repo.get_job_detail(created.job_id)["tasks"][0]
    repo._append_task_log_event(
        job_id=created.job_id,
        task_id=task["task_id"],
        stage=task["stage"],
        event="task_test_event",
        result="queued",
    )
    lines = Path(_log_file(repo, task["task_id"])).read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[-1])
    assert list(payload)[: len(_SPEC_FIELDS)] == _SPEC_FIELDS
    assert payload["request_id"] == "req-abc"
    assert payload["attempt"] == 0
    assert isinstance(payload["effective_config"], dict)
    assert payload["error"] is None


def test_task_log_extra_cannot_override_standard_fields(tmp_path: Path) -> None:
    """extra 里的同名字段不得覆盖标准字段。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/log2.mp4", "ja")
    task = repo.get_job_detail(created.job_id)["tasks"][0]
    repo._append_task_log_event(
        job_id=created.job_id,
        task_id=task["task_id"],
        stage=task["stage"],
        event="e",
        result="queued",
        extra={"stage": "hacked", "job_id": "nope", "previous_status": "failed"},
    )
    payload = json.loads(
        Path(_log_file(repo, task["task_id"])).read_text(encoding="utf-8").splitlines()[-1]
    )
    assert payload["stage"] == task["stage"]
    assert payload["job_id"] == created.job_id
    assert payload["previous_status"] == "failed"


def test_enqueue_persists_request_id_in_task_metadata(tmp_path: Path) -> None:
    """request_id 写入每个 task 的 task_config.metadata，且无新增列。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/meta.mp4", "ja", request_id=" req-xyz ")
    detail = repo.get_job_detail(created.job_id)
    for task in detail["tasks"]:
        assert task["task_config"]["metadata"]["request_id"] == "req-xyz"


def test_enqueue_without_request_id_keeps_none(tmp_path: Path) -> None:
    """直接入队（脚本/测试）不传 request_id 时保持 None。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/noreq.mp4", "ja")
    detail = repo.get_job_detail(created.job_id)
    assert detail["tasks"][0]["task_config"]["metadata"]["request_id"] is None


# ------------------------------------------------------- copy_back 默认值


def test_global_translation_copy_back_is_applied(tmp_path: Path) -> None:
    """[translation] copy_back（经 stage_effective_defaults）必须生效。"""

    out_dir = tmp_path / "global-out"
    out_dir.mkdir()
    repo = _repo(
        tmp_path,
        stage_effective_defaults={"translate": {"copy_back": str(out_dir)}},
        allowed_copy_back_roots=(out_dir,),
    )
    created = repo.enqueue("/tmp/g.mp4", "ja")
    tasks = {t["stage"]: t for t in repo.get_job_detail(created.job_id)["tasks"]}
    assert tasks["translate"]["task_config"]["io"]["copy_back_dir"] == str(
        out_dir.resolve()
    )


def test_job_config_copy_back_overrides_global_default(tmp_path: Path) -> None:
    """job_config 优先于 [translation] 全局默认。"""

    global_dir = tmp_path / "global"
    job_dir = tmp_path / "job"
    global_dir.mkdir()
    job_dir.mkdir()
    repo = _repo(
        tmp_path,
        stage_effective_defaults={"translate": {"copy_back": str(global_dir)}},
        allowed_copy_back_roots=(global_dir, job_dir),
    )
    created = repo.enqueue(
        "/tmp/h.mp4", "ja", job_config={"translate": {"copy_back": str(job_dir)}}
    )
    tasks = {t["stage"]: t for t in repo.get_job_detail(created.job_id)["tasks"]}
    assert tasks["translate"]["task_config"]["io"]["copy_back_dir"] == str(
        job_dir.resolve()
    )


def test_global_copy_back_still_validated_against_roots(tmp_path: Path) -> None:
    """全局默认值同样要过白名单：越界时入队即拒绝，而不是静默接受。"""

    outside = tmp_path / "outside"
    allowed = tmp_path / "allowed"
    outside.mkdir()
    allowed.mkdir()
    repo = _repo(
        tmp_path,
        stage_effective_defaults={"translate": {"copy_back": str(outside)}},
        allowed_copy_back_roots=(allowed,),
    )
    with pytest.raises(ValueError, match="copy_back_outside_allowed_roots"):
        repo.enqueue("/tmp/i.mp4", "ja")


def test_default_copy_back_without_global_config_is_video_dir(tmp_path: Path) -> None:
    """没有任何配置时回退历史上的“视频所在目录”。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/j.mp4", "ja")
    tasks = {t["stage"]: t for t in repo.get_job_detail(created.job_id)["tasks"]}
    assert tasks["translate"]["task_config"]["io"]["copy_back_dir"] == "/tmp"
