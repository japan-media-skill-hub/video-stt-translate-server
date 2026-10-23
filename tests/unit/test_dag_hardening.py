"""SPEC B.2 DAG 语义校验与 C.2 translate job_config 白名单测试。

覆盖：
- 5 类非法 DAG 各自返回明确错误码；
- 3 种合法 DAG 形态（extract->stt->translate、extract->stt_whisperx->translate、
  extract->stt 仅转写）；
- translate 新增覆盖键（parallel/batch_size/timeout/disable_proxy/request_interval/
  retry/chunk_minutes/copy_back/max_chars/checkpoint_enabled/strict_ids）的白名单与
  类型/范围校验，并保持与既有 timeout_sec/max_retries 约定兼容。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whisper_stt_service.dag import (
    ASR_STAGES,
    SUPPORTED_DAG_VERSION,
    build_default_dag,
    normalize_and_validate_dag,
    normalize_and_validate_job_config,
)
from whisper_stt_service.db import Database
from whisper_stt_service.repository import JobRepository


def _dag(*stages: tuple[str, list[str]], version: object = 1) -> dict:
    """构造 DAG 载荷；version 允许传入任意值以便测试版本校验。"""

    return {
        "version": version,
        "stages": [{"stage": name, "depends_on": deps} for name, deps in stages],
    }


# ---------------------------------------------------------------- 合法形态


@pytest.mark.parametrize(
    "asr_stage",
    list(ASR_STAGES),
    ids=["stt", "stt_whisperx"],
)
def test_legal_dag_with_translate(asr_stage: str) -> None:
    """extract -> (stt|stt_whisperx) -> translate 均应通过校验。"""

    raw = _dag(
        ("extract", []),
        (asr_stage, ["extract"]),
        ("translate", [asr_stage]),
    )
    normalized = normalize_and_validate_dag(raw)
    assert normalized["version"] == SUPPORTED_DAG_VERSION
    assert [item["stage"] for item in normalized["stages"]] == [
        "extract",
        asr_stage,
        "translate",
    ]
    assert [item["depends_on"] for item in normalized["stages"]] == [
        [],
        ["extract"],
        [asr_stage],
    ]


def test_legal_dag_transcribe_only() -> None:
    """extract -> stt（不含 translate）是合法形态：只做转写。"""

    normalized = normalize_and_validate_dag(
        _dag(("extract", []), ("stt", ["extract"]))
    )
    assert [item["stage"] for item in normalized["stages"]] == ["extract", "stt"]


def test_legal_dag_with_transitive_extract_dependency() -> None:
    """extract 出现在 ASR 的传递依赖闭包中即满足 wav 契约。"""

    normalized = normalize_and_validate_dag(
        _dag(
            ("extract", []),
            # 仅为验证“依赖闭包”语义：ASR 直接依赖 extract，闭包自然包含 extract。
            ("stt", ["extract"]),
            ("translate", ["extract", "stt"]),
        )
    )
    assert normalized["version"] == 1


def test_default_dag_is_valid_and_version_defaults_to_one() -> None:
    """默认 DAG 合法；缺省 version 视为 1。"""

    default = build_default_dag()
    default.pop("version")
    normalized = normalize_and_validate_dag(default)
    assert normalized["version"] == 1
    assert normalize_and_validate_dag(None)["version"] == 1


# ---------------------------------------------------------------- 非法形态


def test_reject_asr_stage_conflict() -> None:
    """stt 与 stt_whisperx 并存会写同一 ASR 产物，必须拒绝。"""

    raw = _dag(
        ("extract", []),
        ("stt", ["extract"]),
        ("stt_whisperx", ["extract"]),
        ("translate", ["stt"]),
    )
    with pytest.raises(ValueError, match="^dag_asr_stage_conflict$"):
        normalize_and_validate_dag(raw)


def test_reject_missing_asr_stage() -> None:
    """只有 extract+translate（或只有 translate）时必须拒绝。"""

    with pytest.raises(ValueError, match="^dag_missing_asr_stage$"):
        normalize_and_validate_dag(_dag(("extract", []), ("translate", ["extract"])))
    with pytest.raises(ValueError, match="^dag_missing_asr_stage$"):
        normalize_and_validate_dag(_dag(("translate", [])))


def test_reject_missing_extract_dependency() -> None:
    """ASR 未依赖 extract 时必须拒绝（D1 的 wav 契约）。"""

    raw = _dag(("stt", []), ("translate", ["stt"]))
    with pytest.raises(ValueError, match="^dag_missing_extract_dependency$"):
        normalize_and_validate_dag(raw)


def test_reject_missing_extract_dependency_with_unrelated_extract() -> None:
    """extract 存在但不属于 ASR 依赖闭包时仍必须拒绝。"""

    raw = _dag(
        ("extract", []),
        ("stt", []),
        ("translate", ["extract", "stt"]),
    )
    with pytest.raises(ValueError, match="^dag_missing_extract_dependency$"):
        normalize_and_validate_dag(raw)


def test_reject_translate_without_asr_dependency() -> None:
    """translate 的依赖闭包不含唯一 ASR 阶段时必须拒绝。"""

    raw = _dag(
        ("extract", []),
        ("stt", ["extract"]),
        ("translate", ["extract"]),
    )
    with pytest.raises(ValueError, match="^dag_translate_without_asr$"):
        normalize_and_validate_dag(raw)


@pytest.mark.parametrize(
    "version",
    [2, 0, -1, "1", 1.0, True, False, [1], {"v": 1}, None],
    ids=["v2", "v0", "neg", "str", "float", "true", "false", "list", "dict", "null"],
)
def test_reject_unsupported_dag_version(version: object) -> None:
    """version 必须严格为整数 1；bool/浮点/字符串/其他整数/显式 null 一律拒绝。"""

    raw = _dag(("extract", []), ("stt", ["extract"]), version=version)
    with pytest.raises(ValueError, match="^dag_version_not_supported$"):
        normalize_and_validate_dag(raw)


def test_explicit_null_version_is_not_treated_as_absent() -> None:
    """显式 version=null 不等于缺省：缺省按 1 通过，null 必须拒绝。"""

    absent = _dag(("extract", []), ("stt", ["extract"]))
    absent.pop("version")
    assert normalize_and_validate_dag(absent)["version"] == 1

    explicit_null = {"version": None, "stages": absent["stages"]}
    with pytest.raises(ValueError, match="^dag_version_not_supported$"):
        normalize_and_validate_dag(explicit_null)


def test_structural_errors_take_precedence_over_semantics() -> None:
    """结构错误（环、缺失依赖、重复）保持既有错误码，不被语义校验覆盖。"""

    with pytest.raises(ValueError, match="^dag_cycle_detected$"):
        normalize_and_validate_dag(
            _dag(
                ("extract", ["translate"]),
                ("stt", ["extract"]),
                ("translate", ["stt"]),
            )
        )
    with pytest.raises(ValueError, match="^dag_dependency_missing$"):
        normalize_and_validate_dag(_dag(("stt", ["extract"])))
    with pytest.raises(ValueError, match="^dag_stage_duplicated$"):
        normalize_and_validate_dag(
            _dag(("extract", []), ("stt", ["extract"]), ("stt", ["extract"]))
        )
    with pytest.raises(ValueError, match="^dag_self_dependency$"):
        normalize_and_validate_dag(
            _dag(("extract", []), ("stt", ["extract", "stt"]))
        )


def test_enqueue_rejects_invalid_dag_before_persisting(tmp_path: Path) -> None:
    """入队即拒绝：非法 DAG 抛 ValueError 且不产生 job。"""

    db = Database(tmp_path / "dag.db")
    db.init_schema()
    repo = JobRepository(db, log_root=tmp_path / "logs")
    with pytest.raises(ValueError, match="^dag_missing_asr_stage$"):
        repo.enqueue(
            video_path="/tmp/dag-blocked.mp4",
            language="ja",
            dag=_dag(("extract", []), ("translate", ["extract"])),
        )
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(1) AS c FROM jobs").fetchone()["c"] == 0


# ------------------------------------------------- job_config：translate 白名单

_NEW_TRANSLATE_OVERRIDES: dict[str, object] = {
    "parallel": 3,
    "batch_size": 150,
    "timeout": 120,
    "disable_proxy": False,
    "request_interval": 0.5,
    "retry": 6,
    "chunk_minutes": 10,
    "copy_back": "/tmp/subtitles",
    "max_chars": 9000,
    "checkpoint_enabled": False,
    "progress_ttl_sec": 600,
    "strict_ids": True,
}


def test_translate_job_config_accepts_all_new_overrides() -> None:
    """新增 translate 覆盖键应全部被接受并原样归一化。"""

    normalized = normalize_and_validate_job_config(
        {"translate": dict(_NEW_TRANSLATE_OVERRIDES)}, build_default_dag()
    )
    assert normalized["translate"] == _NEW_TRANSLATE_OVERRIDES


def test_translate_job_config_keeps_legacy_timeout_convention() -> None:
    """既有 timeout_sec/max_retries 约定不变，且可与 timeout/retry 并存。"""

    normalized = normalize_and_validate_job_config(
        {"translate": {"timeout_sec": 600, "max_retries": 2, "timeout": 30, "retry": 1}},
        build_default_dag(),
    )
    assert normalized["translate"] == {
        "timeout_sec": 600,
        "max_retries": 2,
        "timeout": 30,
        "retry": 1,
    }
    with pytest.raises(ValueError, match="^job_config_value_out_of_range$"):
        normalize_and_validate_job_config(
            {"translate": {"timeout_sec": 0}}, build_default_dag()
        )
    with pytest.raises(ValueError, match="^job_config_value_out_of_range$"):
        normalize_and_validate_job_config(
            {"translate": {"max_retries": -1}}, build_default_dag()
        )


@pytest.mark.parametrize(
    "key",
    ["parallel", "batch_size", "timeout", "max_chars"],
)
def test_translate_int_overrides_reject_bool_and_str(key: str) -> None:
    """整数键不接受 bool / 字符串 / 浮点。"""

    dag = build_default_dag()
    for bad in (True, False, "2", 2.0, None):
        with pytest.raises(ValueError, match="^job_config_value_type_invalid$"):
            normalize_and_validate_job_config({"translate": {key: bad}}, dag)


@pytest.mark.parametrize(
    "key",
    ["parallel", "batch_size", "timeout", "max_chars"],
)
def test_translate_int_overrides_require_positive(key: str) -> None:
    """整数键必须为正整数。"""

    dag = build_default_dag()
    for bad in (0, -1):
        with pytest.raises(ValueError, match="^job_config_value_out_of_range$"):
            normalize_and_validate_job_config({"translate": {key: bad}}, dag)


@pytest.mark.parametrize(
    "key",
    ["disable_proxy", "checkpoint_enabled", "strict_ids"],
)
def test_translate_bool_overrides_reject_int_and_str(key: str) -> None:
    """布尔键只接受真正的 bool，禁止把 0/1 或 "false" 当作布尔。"""

    dag = build_default_dag()
    for bad in (0, 1, "true", "false", None):
        with pytest.raises(ValueError, match="^job_config_value_type_invalid$"):
            normalize_and_validate_job_config({"translate": {key: bad}}, dag)
    assert normalize_and_validate_job_config({"translate": {key: True}}, dag)[
        "translate"
    ][key] is True


def test_translate_request_interval_accepts_zero_and_numbers() -> None:
    """request_interval 接受 0（关闭节流）与非负数值。"""

    dag = build_default_dag()
    assert (
        normalize_and_validate_job_config({"translate": {"request_interval": 0}}, dag)[
            "translate"
        ]["request_interval"]
        == 0
    )
    assert normalize_and_validate_job_config(
        {"translate": {"request_interval": 2.5}}, dag
    )["translate"]["request_interval"] == 2.5
    with pytest.raises(ValueError, match="^job_config_value_out_of_range$"):
        normalize_and_validate_job_config(
            {"translate": {"request_interval": -0.1}}, dag
        )
    with pytest.raises(ValueError, match="^job_config_value_type_invalid$"):
        normalize_and_validate_job_config(
            {"translate": {"request_interval": True}}, dag
        )


def test_translate_retry_and_chunk_minutes_require_positive_int() -> None:
    """retry（批内重试次数）与 chunk_minutes（时间窗）沿用正整数约定。"""

    dag = build_default_dag()
    for key in ("retry", "chunk_minutes"):
        with pytest.raises(ValueError, match="^job_config_value_out_of_range$"):
            normalize_and_validate_job_config({"translate": {key: 0}}, dag)
        with pytest.raises(ValueError, match="^job_config_value_type_invalid$"):
            normalize_and_validate_job_config({"translate": {key: True}}, dag)


def test_translate_progress_ttl_sec_accepts_zero_and_positive_int() -> None:
    """progress_ttl_sec 允许 0（成功即清理），必须是非负整数且拒绝 bool。"""

    dag = build_default_dag()
    for good in (0, 1, 86400):
        normalized = normalize_and_validate_job_config(
            {"translate": {"progress_ttl_sec": good}}, dag
        )
        assert normalized["translate"]["progress_ttl_sec"] == good
    for bad in (-1, -3600):
        with pytest.raises(ValueError, match="^job_config_value_out_of_range$"):
            normalize_and_validate_job_config(
                {"translate": {"progress_ttl_sec": bad}}, dag
            )
    for bad in (True, False, "60", 60.0, None):
        with pytest.raises(ValueError, match="^job_config_value_type_invalid$"):
            normalize_and_validate_job_config(
                {"translate": {"progress_ttl_sec": bad}}, dag
            )


def test_translate_copy_back_requires_str() -> None:
    """copy_back 仍只接受字符串（含 __video_dir__ 占位）。"""

    dag = build_default_dag()
    normalized = normalize_and_validate_job_config(
        {"translate": {"copy_back": "__video_dir__"}}, dag
    )
    assert normalized["translate"]["copy_back"] == "__video_dir__"
    with pytest.raises(ValueError, match="^job_config_value_type_invalid$"):
        normalize_and_validate_job_config({"translate": {"copy_back": 1}}, dag)


def test_unknown_translate_key_still_rejected() -> None:
    """白名单之外的键仍然被拒绝。"""

    with pytest.raises(ValueError, match="^job_config_key_not_supported$"):
        normalize_and_validate_job_config(
            {"translate": {"max_chars_not_a_key": 1}}, build_default_dag()
        )


def test_translate_job_config_requires_translate_stage_in_dag() -> None:
    """DAG 不含 translate 时，translate 覆盖应被拒绝。"""

    dag = normalize_and_validate_dag(_dag(("extract", []), ("stt", ["extract"])))
    with pytest.raises(ValueError, match="^job_config_stage_not_in_dag$"):
        normalize_and_validate_job_config({"translate": {"parallel": 2}}, dag)
