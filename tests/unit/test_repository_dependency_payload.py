"""依赖载荷编码与领取规则测试。"""

from pathlib import Path

from whisper_stt_service.db import Database
from whisper_stt_service.repository import JobRepository, decode_dependency_payload


def test_decode_dependency_payload_compatible_formats() -> None:
    """应兼容 NULL/单值/JSON 数组编码。"""

    assert decode_dependency_payload(None) == []
    assert decode_dependency_payload("") == []
    assert decode_dependency_payload("task-a") == ["task-a"]
    assert decode_dependency_payload('["a", "b"]') == ["a", "b"]


def test_claim_next_requires_all_dependencies_succeeded(tmp_path: Path) -> None:
    """多依赖节点在所有上游成功前不可领取。"""

    db = Database(tmp_path / "q.db")
    db.init_schema()
    repo = JobRepository(db)

    # 批次 2 的 DAG 语义要求 ASR 依赖 extract（wav 契约）；translate 仍保留
    # 多依赖（extract + stt），以覆盖多依赖解码路径。
    dag = {
        "version": 1,
        "stages": [
            {"stage": "extract", "depends_on": []},
            {"stage": "stt", "depends_on": ["extract"]},
            {"stage": "translate", "depends_on": ["extract", "stt"]},
        ],
    }
    created = repo.enqueue(video_path="/tmp/dep.mp4", language="ja", dag=dag)
    detail = repo.get_job_detail(created.job_id)
    assert detail is not None
    tasks = {t["stage"]: t for t in detail["tasks"]}

    # 批次 1：写回改成“必须仍处于 claimed”，因此先领取再标记成功。
    extract_claim = repo.claim_next("extract", worker_id="w1", lease_timeout_sec=60)
    assert extract_claim is not None
    assert (
        repo.mark_task_succeeded(
            tasks["extract"]["task_id"],
            worker_id="w1",
            attempt=extract_claim.attempt,
        )
        is True
    )
    claimed_before = repo.claim_next("translate", worker_id="w1", lease_timeout_sec=60)
    assert claimed_before is None

    stt_claim = repo.claim_next("stt", worker_id="w1", lease_timeout_sec=60)
    assert stt_claim is not None
    assert (
        repo.mark_task_succeeded(
            tasks["stt"]["task_id"], worker_id="w1", attempt=stt_claim.attempt
        )
        is True
    )
    claimed_after = repo.claim_next("translate", worker_id="w1", lease_timeout_sec=60)
    assert claimed_after is not None
