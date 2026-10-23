"""DAG 规划、校验与配置归一化。"""

from __future__ import annotations

from collections import deque

from whisper_stt_service.core.stages import STAGE_CONFIG_KEYS, SUPPORTED_STAGES

# 本服务唯一支持的 DAG 版本；缺省视为 1（历史请求不带 version）。
SUPPORTED_DAG_VERSION = 1

# 互斥的 ASR 阶段：DAG 中必须恰好出现一个（SPEC B.2-2、D1 的 wav 契约）。
ASR_STAGES: tuple[str, ...] = ("stt", "stt_whisperx")

# ASR 的硬前置阶段：ASR 必须消费 extract 产出的 16k mono wav（SPEC B.2-3、E）。
REQUIRED_ASR_UPSTREAM_STAGE = "extract"


def build_default_dag() -> dict:
    """返回与历史行为兼容的默认三阶段 DAG。"""

    return {
        "version": 1,
        "stages": [
            {"stage": "extract", "depends_on": []},
            {"stage": "stt", "depends_on": ["extract"]},
            {"stage": "translate", "depends_on": ["stt"]},
        ],
    }


def normalize_and_validate_dag(raw: dict | None) -> dict:
    """归一化并校验 DAG。

    包含结构校验与 SPEC B.2 语义校验（版本、唯一 ASR、extract 依赖、translate 依赖 ASR）。
    """

    src = build_default_dag() if raw is None else raw
    _validate_dag_version(src)
    stages = src.get("stages") if isinstance(src, dict) else None
    if not isinstance(stages, list) or len(stages) == 0:
        raise ValueError("dag_empty")

    normalized_stages: list[dict[str, object]] = []
    stage_names: set[str] = set()
    for item in stages:
        if not isinstance(item, dict):
            raise ValueError("dag_stage_invalid")
        stage = str(item.get("stage", "")).strip()
        if stage == "":
            raise ValueError("dag_stage_empty")
        if stage not in SUPPORTED_STAGES:
            raise ValueError("dag_stage_not_supported")
        if stage in stage_names:
            raise ValueError("dag_stage_duplicated")
        depends_raw = item.get("depends_on", [])
        if depends_raw is None:
            depends_raw = []
        if not isinstance(depends_raw, list):
            raise ValueError("dag_depends_on_invalid")
        deps = [str(x).strip() for x in depends_raw if str(x).strip()]
        if stage in deps:
            raise ValueError("dag_self_dependency")
        normalized_stages.append({"stage": stage, "depends_on": deps})
        stage_names.add(stage)

    for item in normalized_stages:
        for dep in item["depends_on"]:
            if dep not in stage_names:
                raise ValueError("dag_dependency_missing")

    _ensure_no_cycle(normalized_stages)
    _validate_dag_semantics(normalized_stages)
    return {"version": SUPPORTED_DAG_VERSION, "stages": normalized_stages}


def _validate_dag_version(src: object) -> None:
    """校验 DAG 版本：只接受整数 1。

    “缺省”仅指请求里完全没有 `version` 键（历史客户端不带该字段），此时按 1 处理；
    显式传 `null`/None 不算缺省，必须拒绝——否则调用方可以用 null 绕过版本契约。
    其他类型（bool/float/str/list/dict）一律 `dag_version_not_supported`。
    """

    if not isinstance(src, dict):
        return
    if "version" not in src:
        return
    raw_version = src["version"]
    if isinstance(raw_version, bool) or not isinstance(raw_version, int):
        raise ValueError("dag_version_not_supported")
    if raw_version != SUPPORTED_DAG_VERSION:
        raise ValueError("dag_version_not_supported")


def _validate_dag_semantics(stages: list[dict[str, object]]) -> None:
    """校验 DAG 语义：恰好一个 ASR、ASR 依赖闭包含 extract、translate 依赖闭包含 ASR。"""

    dependencies = {
        str(item["stage"]): [str(dep) for dep in item["depends_on"]] for item in stages
    }
    present_asr = [name for name in ASR_STAGES if name in dependencies]
    if len(present_asr) > 1:
        raise ValueError("dag_asr_stage_conflict")
    if len(present_asr) == 0:
        raise ValueError("dag_missing_asr_stage")

    asr_stage = present_asr[0]
    asr_closure = _dependency_closure(asr_stage, dependencies)
    if REQUIRED_ASR_UPSTREAM_STAGE not in asr_closure:
        raise ValueError("dag_missing_extract_dependency")

    if "translate" in dependencies:
        translate_closure = _dependency_closure("translate", dependencies)
        if asr_stage not in translate_closure:
            raise ValueError("dag_translate_without_asr")


def _dependency_closure(stage: str, dependencies: dict[str, list[str]]) -> set[str]:
    """返回指定阶段依赖闭包（不含自身），`dependencies` 必须已通过结构校验。"""

    closure: set[str] = set()
    pending = list(dependencies.get(stage, []))
    while pending:
        cur = pending.pop()
        if cur in closure or cur == stage:
            continue
        closure.add(cur)
        pending.extend(dependencies.get(cur, []))
    return closure


def normalize_and_validate_job_config(job_config: dict | None, dag: dict) -> dict:
    """归一化并校验 job_config（按 stage 分组）。"""

    if job_config is None:
        return {}
    if not isinstance(job_config, dict):
        raise ValueError("job_config_invalid")

    dag_stages = {str(item["stage"]) for item in dag["stages"]}
    result: dict[str, dict] = {}
    for stage, raw_cfg in job_config.items():
        stage_name = str(stage)
        if stage_name not in dag_stages:
            raise ValueError("job_config_stage_not_in_dag")
        if not isinstance(raw_cfg, dict):
            raise ValueError("job_config_stage_invalid")
        allowed = set(STAGE_CONFIG_KEYS.get(stage_name, ()))
        normalized_cfg: dict[str, object] = {}
        for key, value in raw_cfg.items():
            key_name = str(key)
            if key_name not in allowed:
                raise ValueError("job_config_key_not_supported")
            _validate_stage_config_value(stage_name, key_name, value)
            normalized_cfg[key_name] = value
        result[stage_name] = normalized_cfg
    return result


def _validate_stage_config_value(stage: str, key: str, value: object) -> None:
    """校验 job_config 单个字段的类型和范围。"""

    int_min_rules: dict[tuple[str, str], int] = {
        ("extract", "timeout_sec"): 1,
        ("extract", "max_retries"): 0,
        ("stt", "timeout_sec"): 1,
        ("stt", "max_retries"): 0,
        ("stt", "batch_size"): 1,
        ("stt", "beam_size"): 1,
        ("stt", "best_of"): 1,
        ("stt", "vad_min_speech_duration_ms"): 50,
        ("stt", "vad_min_silence_duration_ms"): 50,
        ("stt", "vad_speech_pad_ms"): 0,
        ("stt_whisperx", "timeout_sec"): 1,
        ("stt_whisperx", "max_retries"): 0,
        ("stt_whisperx", "batch_size"): 1,
        ("translate", "timeout_sec"): 1,
        ("translate", "max_retries"): 0,
        ("translate", "chunk_minutes"): 1,
        ("translate", "retry"): 1,
        # SPEC C.1/C.3：并发上限、单批条目数与字符上限都必须为正整数。
        ("translate", "parallel"): 1,
        ("translate", "batch_size"): 1,
        ("translate", "max_chars"): 1,
        # 上游 HTTP 超时秒数，必须为正整数。
        ("translate", "timeout"): 1,
        # SPEC C.4：检查点保留秒数，允许 0（成功即清理）。
        ("translate", "progress_ttl_sec"): 0,
    }
    float_min_rules: dict[tuple[str, str], float] = {
        ("stt", "patience"): 0.1,
        ("stt", "vad_max_speech_duration_s"): 1.0,
        ("stt", "compression_ratio_threshold"): 0.1,
        ("stt", "hallucination_silence_threshold"): 0.0,
        # SPEC D2：批次派发间隔（秒），0 表示关闭节流。
        ("translate", "request_interval"): 0.0,
    }
    float_range_rules: dict[tuple[str, str], tuple[float, float]] = {
        ("stt", "vad_threshold"): (0.01, 0.99),
        ("stt", "no_speech_threshold"): (0.01, 0.99),
        ("stt_whisperx", "vad_onset"): (0.01, 0.99),
        ("stt_whisperx", "vad_offset"): (0.01, 0.99),
    }
    bool_rules = {
        ("stt", "condition_on_previous_text"),
        ("stt", "vad_filter"),
        ("stt_whisperx", "align_enabled"),
        ("stt_whisperx", "local_files_only"),
        # SPEC C.2/C.4/C.5：布尔开关不接受 int/str，避免 "false" 被当成 true。
        ("translate", "disable_proxy"),
        ("translate", "checkpoint_enabled"),
        ("translate", "strict_ids"),
    }
    str_rules = {
        ("stt", "device"),
        ("stt", "compute_type"),
        ("stt", "initial_prompt"),
        ("stt", "hotwords"),
        ("translate", "copy_back"),
        ("stt_whisperx", "model"),
        ("stt_whisperx", "device"),
        ("stt_whisperx", "compute_type"),
        ("stt_whisperx", "vad_config_path"),
        ("stt_whisperx", "align_model_root"),
        ("stt_whisperx", "vad_backend"),
    }

    key_pair = (stage, key)
    if key_pair in int_min_rules:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("job_config_value_type_invalid")
        if value < int_min_rules[key_pair]:
            raise ValueError("job_config_value_out_of_range")
        return

    if key_pair in float_min_rules:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("job_config_value_type_invalid")
        if float(value) < float_min_rules[key_pair]:
            raise ValueError("job_config_value_out_of_range")
        return

    if key_pair in float_range_rules:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("job_config_value_type_invalid")
        low, high = float_range_rules[key_pair]
        if float(value) < low or float(value) > high:
            raise ValueError("job_config_value_out_of_range")
        return

    if key_pair in bool_rules:
        if not isinstance(value, bool):
            raise ValueError("job_config_value_type_invalid")
        return

    if key_pair in str_rules:
        if not isinstance(value, str):
            raise ValueError("job_config_value_type_invalid")
        return


def _ensure_no_cycle(stages: list[dict[str, object]]) -> None:
    """使用 Kahn 算法检测环。"""

    graph: dict[str, list[str]] = {}
    indegree: dict[str, int] = {}
    for item in stages:
        stage = str(item["stage"])
        graph.setdefault(stage, [])
        indegree.setdefault(stage, 0)

    for item in stages:
        stage = str(item["stage"])
        deps = item["depends_on"]
        for dep in deps:
            graph.setdefault(dep, [])
            graph[dep].append(stage)
            indegree[stage] = indegree.get(stage, 0) + 1

    q = deque([n for n, d in indegree.items() if d == 0])
    visited = 0
    while q:
        cur = q.popleft()
        visited += 1
        for nxt in graph.get(cur, []):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)
    if visited != len(indegree):
        raise ValueError("dag_cycle_detected")
