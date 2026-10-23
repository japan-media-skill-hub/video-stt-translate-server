"""历史 DAG 重跑的 409 语义测试（SPEC B.2 / §6 兼容性）。

历史 job 的 dag_json 可能早于当前 DAG 语义校验（例如只有 translate、
或 ASR 未依赖 extract）。retry 必须拒绝并给出明确错误码，由路由映射 409，
而不是静默降级重跑。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whisper_stt_service.repo.database import Database
from whisper_stt_service.repo.job_repository import JobRepository


def _failed_job_with_dag(tmp_path: Path, dag_payload: object) -> tuple[JobRepository, str]:
    """构造一个 failed job，并把 dag_json 改写成指定的历史形态。"""

    db = Database(tmp_path / "q.db")
    db.init_schema()
    repo = JobRepository(db, log_root=tmp_path / "logs", artifact_root=tmp_path / "a")
    created = repo.enqueue("/tmp/retry.mp4", "ja")
    raw = dag_payload if isinstance(dag_payload, str) else json.dumps(dag_payload)
    with db.tx() as conn:
        conn.execute(
            "UPDATE jobs SET status='failed', dag_json=? WHERE job_id=?",
            (raw, created.job_id),
        )
    return repo, created.job_id


def _dag(*stages: tuple[str, list[str]], version: object = 1) -> dict:
    """构造 DAG 载荷。"""

    return {
        "version": version,
        "stages": [{"stage": name, "depends_on": deps} for name, deps in stages],
    }


@pytest.mark.parametrize(
    ("label", "payload", "code"),
    [
        (
            "asr_conflict",
            _dag(
                ("extract", []),
                ("stt", ["extract"]),
                ("stt_whisperx", ["extract"]),
                ("translate", ["stt"]),
            ),
            "dag_asr_stage_conflict",
        ),
        (
            "missing_asr",
            _dag(("extract", []), ("translate", ["extract"])),
            "dag_missing_asr_stage",
        ),
        (
            "missing_extract_dependency",
            _dag(("stt", []), ("translate", ["stt"])),
            "dag_missing_extract_dependency",
        ),
        (
            "translate_without_asr",
            _dag(("extract", []), ("stt", ["extract"]), ("translate", ["extract"])),
            "dag_translate_without_asr",
        ),
        (
            "version_not_supported",
            _dag(("extract", []), ("stt", ["extract"]), version=2),
            "dag_version_not_supported",
        ),
    ],
)
def test_retry_rejects_legacy_invalid_dag(
    tmp_path: Path, label: str, payload: dict, code: str
) -> None:
    """5 类非法历史 DAG 都应被拒绝，错误码带明确原因。"""

    repo, job_id = _failed_job_with_dag(tmp_path, payload)
    with pytest.raises(ValueError) as excinfo:
        repo.retry_job(job_id)
    assert "retry_job_dag_invalid" in str(excinfo.value)
    assert code in str(excinfo.value)


def test_retry_rejects_unparsable_dag(tmp_path: Path) -> None:
    """dag_json 无法解析时同样拒绝重跑。"""

    repo, job_id = _failed_job_with_dag(tmp_path, "{not-json")
    with pytest.raises(ValueError, match="retry_job_dag_invalid"):
        repo.retry_job(job_id)


def test_retry_keeps_job_failed_when_dag_invalid(tmp_path: Path) -> None:
    """拒绝重跑时不得改动 job/task 状态（事务已回滚）。"""

    repo, job_id = _failed_job_with_dag(
        tmp_path, _dag(("extract", []), ("translate", ["extract"]))
    )
    with pytest.raises(ValueError):
        repo.retry_job(job_id)
    assert repo.get_job_status(job_id) == "failed"


def test_retry_allows_valid_legacy_default_dag(tmp_path: Path) -> None:
    """合法（默认三阶段）DAG 仍可正常重跑。"""

    db = Database(tmp_path / "ok.db")
    db.init_schema()
    repo = JobRepository(db, log_root=tmp_path / "logs", artifact_root=tmp_path / "a")
    created = repo.enqueue("/tmp/retry-ok.mp4", "ja")
    with db.tx() as conn:
        conn.execute(
            "UPDATE jobs SET status='failed' WHERE job_id=?", (created.job_id,)
        )
        conn.execute(
            "UPDATE tasks SET status='failed' WHERE job_id=?", (created.job_id,)
        )
    result = repo.retry_job(created.job_id)
    assert result["retried_tasks"] == 3
    assert repo.get_job_status(created.job_id) == "queued"
