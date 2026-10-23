"""SPEC B.3 产物所有权（io 段）与 SPEC H 路径安全在仓储层的测试。

覆盖：
- enqueue 为每阶段写入 io，产物按 audio/asr/translate 分目录；
- canonical 规范化：同文件不同写法折叠为同一 job（幂等）；
- by-path 使用 canonical + 原始双键精确匹配且命中索引（不全表扫描）；
- TaskExecutionContext 暴露 input_path/output_path，历史 job 回退旧规则；
- copy_back 默认仅视频目录、roots 可放宽、repo 侧不要求目录已存在；
- 内部入队不能绕过白名单与扩展名限制。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whisper_stt_service.repo.database import Database
from whisper_stt_service.repo.job_repository import JobRepository


def _repo(tmp_path: Path, **kwargs) -> JobRepository:
    """构造独立数据库与仓储（产物/日志都落在 tmp_path 下）。"""

    db = Database(tmp_path / "q.db")
    db.init_schema()
    return JobRepository(
        db,
        log_root=tmp_path / "logs",
        artifact_root=tmp_path / "artifacts",
        **kwargs,
    )


def _tasks(repo: JobRepository, job_id: str) -> dict[str, dict]:
    """按 stage 索引 job 的任务明细。"""

    detail = repo.get_job_detail(job_id)
    assert detail is not None
    return {task["stage"]: task for task in detail["tasks"]}


# --------------------------------------------------------------- 产物路径


def test_enqueue_writes_io_for_each_stage(tmp_path: Path) -> None:
    """每阶段都要有 io，且 audio/asr/translate 分目录归属。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/demo.mp4", "ja")
    tasks = _tasks(repo, created.job_id)
    artifact_dir = tmp_path / "artifacts" / created.job_id

    extract_io = tasks["extract"]["task_config"]["io"]
    assert extract_io["input_path"] == "/tmp/demo.mp4"
    assert Path(extract_io["output_path"]) == artifact_dir / "audio" / "demo.16k.wav"

    stt_io = tasks["stt"]["task_config"]["io"]
    assert Path(stt_io["input_path"]) == artifact_dir / "audio" / "demo.16k.wav"
    assert Path(stt_io["output_path"]) == artifact_dir / "asr" / "demo.ja.srt"

    translate_io = tasks["translate"]["task_config"]["io"]
    assert Path(translate_io["input_path"]) == artifact_dir / "asr" / "demo.ja.srt"
    assert Path(translate_io["output_path"]) == artifact_dir / "translate" / "demo.zh.srt"
    assert Path(translate_io["progress_path"]) == (
        artifact_dir / "translate" / "demo.zh.srt.progress.json"
    )


def test_enqueue_output_columns_point_at_asr_and_translate(tmp_path: Path) -> None:
    """jobs.output_ja_path/output_zh_path 指向新布局（字段名保持不变）。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/demo.mp4", "ja")
    detail = repo.get_job_detail(created.job_id)
    artifact_dir = tmp_path / "artifacts" / created.job_id

    assert Path(detail["output_ja_path"]) == artifact_dir / "asr" / "demo.ja.srt"
    assert Path(detail["output_zh_path"]) == artifact_dir / "translate" / "demo.zh.srt"


def test_transcribe_only_dag_has_no_translate_io(tmp_path: Path) -> None:
    """只转写的 DAG 不生成 translate 阶段，也就不会写 translate io。"""

    repo = _repo(tmp_path)
    dag = {
        "version": 1,
        "stages": [
            {"stage": "extract", "depends_on": []},
            {"stage": "stt", "depends_on": ["extract"]},
        ],
    }
    created = repo.enqueue("/tmp/demo.mp4", "ja", dag=dag)
    tasks = _tasks(repo, created.job_id)
    assert set(tasks) == {"extract", "stt"}
    # 没有 translate 阶段，也就没有 translate 的 io 与产物目录。
    assert all(task["task_config"]["io"].get("progress_path") is None for task in tasks.values())


def test_stt_whisperx_uses_audio_input(tmp_path: Path) -> None:
    """stt_whisperx 同样消费 extract 产出的 wav（D1 契约）。"""

    repo = _repo(tmp_path)
    dag = {
        "version": 1,
        "stages": [
            {"stage": "extract", "depends_on": []},
            {"stage": "stt_whisperx", "depends_on": ["extract"]},
            {"stage": "translate", "depends_on": ["stt_whisperx"]},
        ],
    }
    created = repo.enqueue("/tmp/demo.mp4", "ja", dag=dag)
    tasks = _tasks(repo, created.job_id)
    artifact_dir = tmp_path / "artifacts" / created.job_id
    assert Path(tasks["stt_whisperx"]["task_config"]["io"]["input_path"]) == (
        artifact_dir / "audio" / "demo.16k.wav"
    )


# ----------------------------------------------------------- canonical 去重


def test_enqueue_normalizes_video_path_to_canonical(tmp_path: Path) -> None:
    """入库存 canonical 路径（消除 .. 等等价写法）。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/x/../demo.mp4", "ja")
    detail = repo.get_job_detail(created.job_id)
    assert detail["video_path"] == "/tmp/demo.mp4"


def test_enqueue_same_file_different_spelling_is_idempotent(tmp_path: Path) -> None:
    """同一文件的不同写法应折叠为同一个 job，而不是重复入队。"""

    repo = _repo(tmp_path)
    first = repo.enqueue("/tmp/demo.mp4", "ja")
    second = repo.enqueue("/tmp/x/../demo.mp4", "ja")
    assert second.job_id == first.job_id
    assert second.message == "idempotent_returned"


def test_by_path_matches_canonical_and_raw(tmp_path: Path) -> None:
    """by-path 对 canonical 与原始写法都应命中同一 job。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/demo.mp4", "ja")
    assert repo.get_job_latest_by_path("/tmp/demo.mp4")["job_id"] == created.job_id
    assert repo.get_job_latest_by_path("/tmp/x/../demo.mp4")["job_id"] == created.job_id


def test_by_path_and_enqueue_use_index_not_scan(tmp_path: Path) -> None:
    """历史兼容的双键精确查询必须命中 idx_jobs_video_created，不得全表扫描。"""

    repo = _repo(tmp_path)
    repo.enqueue("/tmp/demo.mp4", "ja")
    with repo.db.connect() as conn:
        for sql, params in (
            (
                "SELECT job_id FROM jobs WHERE video_path IN (?, ?) ORDER BY created_at DESC LIMIT 1",
                ("/tmp/demo.mp4", "/tmp/x/../demo.mp4"),
            ),
        ):
            plan = " | ".join(
                str(row["detail"]) for row in conn.execute("EXPLAIN QUERY PLAN " + sql, params)
            )
    assert "USING INDEX idx_jobs_video_created" in plan
    assert "SCAN jobs" not in plan


def test_legacy_raw_row_still_found(tmp_path: Path) -> None:
    """历史 job 存的是原始写法时，用原始写法仍能按精确等值查到。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/demo.mp4", "ja")
    # 模拟历史数据：直接把库里的路径改回非 canonical 写法。
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE jobs SET video_path=? WHERE job_id=?",
            ("/tmp/x/../demo.mp4", created.job_id),
        )
    found = repo.get_job_latest_by_path("/tmp/x/../demo.mp4")
    assert found is not None and found["job_id"] == created.job_id


# ------------------------------------------------------ 执行上下文 io 回填


def test_execution_context_exposes_io_paths(tmp_path: Path) -> None:
    """ctx.input_path/output_path 来自 io 段。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/demo.mp4", "ja")
    tasks = _tasks(repo, created.job_id)
    ctx = repo.get_task_execution_context(tasks["translate"]["task_id"])
    assert ctx is not None
    artifact_dir = tmp_path / "artifacts" / created.job_id
    assert Path(ctx.input_path) == artifact_dir / "asr" / "demo.ja.srt"
    assert Path(ctx.output_path) == artifact_dir / "translate" / "demo.zh.srt"


def test_execution_context_falls_back_for_legacy_jobs(tmp_path: Path) -> None:
    """历史 job 无 io 时回退：extract/stt 吃视频、translate 吃 ja 产 zh。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/demo.mp4", "ja")
    tasks = _tasks(repo, created.job_id)
    # 抹掉 task_config.io，模拟 B.3 之前入队的任务。
    with repo.db.tx() as conn:
        for task in tasks.values():
            cfg = dict(task["task_config"])
            cfg.pop("io", None)
            conn.execute(
                "UPDATE tasks SET task_config_json=? WHERE task_id=?",
                (__import__("json").dumps(cfg), task["task_id"]),
            )
    ctx = repo.get_task_execution_context(tasks["stt"]["task_id"])
    assert ctx.input_path == "/tmp/demo.mp4"
    assert Path(ctx.output_path) == Path(
        repo.get_job_detail(created.job_id)["output_ja_path"]
    )
    legacy_translate = repo.get_task_execution_context(tasks["translate"]["task_id"])
    assert Path(legacy_translate.output_path) == Path(
        repo.get_job_detail(created.job_id)["output_zh_path"]
    )


# ------------------------------------------------------------- copy_back


def test_copy_back_defaults_to_video_dir(tmp_path: Path) -> None:
    """默认回写到视频所在目录。"""

    repo = _repo(tmp_path)
    created = repo.enqueue("/tmp/demo.mp4", "ja")
    tasks = _tasks(repo, created.job_id)
    assert tasks["translate"]["task_config"]["io"]["copy_back_dir"] == "/tmp"


def test_copy_back_explicit_dir_rejected_by_default(tmp_path: Path) -> None:
    """未配置 roots 时，指向别处的 copy_back 在入队即拒绝。"""

    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="copy_back_outside_video_dir"):
        repo.enqueue(
            "/tmp/demo.mp4",
            "ja",
            job_config={"translate": {"copy_back": "/srv/subtitles"}},
        )


def test_copy_back_allowed_roots_permits_explicit_dir(tmp_path: Path) -> None:
    """配置 allowed_copy_back_roots 后可以放宽到指定根目录。"""

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    repo = _repo(tmp_path, allowed_copy_back_roots=(out_dir,))
    created = repo.enqueue(
        "/tmp/demo.mp4",
        "ja",
        job_config={"translate": {"copy_back": str(out_dir)}},
    )
    tasks = _tasks(repo, created.job_id)
    assert tasks["translate"]["task_config"]["io"]["copy_back_dir"] == str(
        out_dir.resolve()
    )


def test_repo_copy_back_does_not_require_existing_directory(tmp_path: Path) -> None:
    """repo 内部入队不要求目标目录已存在（strict_parent=False）。"""

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    target = allowed / "not-created-yet"
    repo = _repo(tmp_path, allowed_copy_back_roots=(allowed,))
    created = repo.enqueue(
        "/tmp/demo.mp4", "ja", job_config={"translate": {"copy_back": str(target)}}
    )
    assert created.accepted is True


# --------------------------------------------------------- 入队安全校验


def test_repo_enqueue_rejects_relative_path(tmp_path: Path) -> None:
    """绕过 API 直接入队也必须拒绝相对路径。"""

    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="video_path_not_absolute"):
        repo.enqueue("demo.mp4", "ja")


def test_repo_enqueue_rejects_bad_extension(tmp_path: Path) -> None:
    """扩展名白名单对内部入队同样生效。"""

    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="video_path_extension_not_allowed"):
        repo.enqueue("/tmp/demo.txt", "ja")


def test_repo_enqueue_cannot_bypass_video_roots(tmp_path: Path) -> None:
    """配置了 allowed_video_roots 时，内部入队无法写入白名单外的路径。"""

    allowed = tmp_path / "videos"
    allowed.mkdir()
    repo = _repo(tmp_path, allowed_video_roots=(allowed,))
    with pytest.raises(ValueError, match="video_path_outside_allowed_roots"):
        repo.enqueue("/tmp/demo.mp4", "ja")


def test_repo_enqueue_accepts_non_existent_file_inside_roots(tmp_path: Path) -> None:
    """白名单内尚不存在的文件仍可入队（保留“先入队后落盘”的用法）。"""

    allowed = tmp_path / "videos"
    allowed.mkdir()
    repo = _repo(tmp_path, allowed_video_roots=(allowed,))
    created = repo.enqueue(str(allowed / "later.mp4"), "ja")
    assert created.accepted is True
