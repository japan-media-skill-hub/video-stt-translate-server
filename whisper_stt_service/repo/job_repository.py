"""任务队列仓储层：入队、领取、状态流转与查询。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

from whisper_stt_service.core.dag import (
    build_default_dag,
    normalize_and_validate_dag,
    normalize_and_validate_job_config,
)
from whisper_stt_service.core.security import (
    SecurityError,
    SessionConfig,
    normalize_video_path,
    resolve_copy_back_dir,
)
from whisper_stt_service.repo.database import Database
from whisper_stt_service.core.stages import SUPPORTED_STAGES


LOGGER = logging.getLogger(__name__)

STAGES = SUPPORTED_STAGES
# 任务终态：到达后不再被调度；blocked 表示“上游失败，本任务不再执行”。
TERMINAL_TASK_STATUSES = frozenset({"succeeded", "failed", "blocked"})
_MAX_TASK_NAME_LEN = 64
_ID_RETRY_LIMIT = 32
_UUID_SUFFIX_LEN = 4
_NON_ALNUM_PATTERN = re.compile(r"[^0-9A-Za-z]+")

# SPEC F/G：claim_next 的有界延迟采样窗口（只保留最近 N 次，避免无界内存）。
_CLAIM_SAMPLE_WINDOW = 256
# SQLite 错误判定：busy/locked 表示瞬时争用（可重试），其余按普通错误计数。
_BUSY_ERROR_MARKERS = ("database is locked", "database is busy", "database table is locked")

# SPEC G：task 日志的标准字段。extra 中同名键不会覆盖这些字段。
_TASK_LOG_STANDARD_FIELDS = frozenset(
    {
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
    }
)


def _percentile(sorted_samples: Sequence[float], percent: float) -> float | None:
    """计算已排序样本的百分位（nearest-rank 法）；样本为空时返回 None。

    使用 nearest-rank 而不是插值：只有一个样本时 P95 就等于该样本，
    符合“单次调用即可观测”的直觉，也便于测试断言。
    """

    if not sorted_samples:
        return None
    count = len(sorted_samples)
    rank = max(1, min(count, int(round(percent / 100.0 * count + 0.5))))
    return float(sorted_samples[rank - 1])

def _derive_task_name(video_path: str) -> str:
    """从视频文件名提取可读 task_name（仅字母数字，保留末尾 64 位）。"""

    stem = Path(video_path).stem
    cleaned = _NON_ALNUM_PATTERN.sub("", stem)
    if len(cleaned) > _MAX_TASK_NAME_LEN:
        cleaned = cleaned[-_MAX_TASK_NAME_LEN:]
    return cleaned or "unnamed"


def _readable_timestamp() -> str:
    """生成 `YYYYMMDDHHMMSS` 格式时间戳。"""

    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _short_uuid_suffix() -> str:
    """生成 4 位十六进制短后缀。"""

    return uuid.uuid4().hex[:_UUID_SUFFIX_LEN]


def _build_readable_job_id(task_name: str, timestamp: str) -> str:
    """构造可读 job_id。"""

    return f"job-{task_name}-job-{timestamp}-{_short_uuid_suffix()}"


def _build_readable_task_id(task_name: str, stage: str, timestamp: str) -> str:
    """构造可读 task_id。"""

    return f"task-{task_name}-{stage}-{timestamp}-{_short_uuid_suffix()}"


def _is_id_collision_error(exc: sqlite3.IntegrityError) -> bool:
    """仅识别 job/task 主键冲突，用于触发可恢复重试。"""

    message = str(exc)
    return (
        "UNIQUE constraint failed: jobs.job_id" in message
        or "UNIQUE constraint failed: tasks.task_id" in message
    )


def _now() -> str:
    """返回当前 UTC 时间的 ISO8601 字符串。"""

    return datetime.now(timezone.utc).isoformat()


def _lease_expire(sec: int) -> str:
    """根据租约秒数计算过期时间戳。"""

    return (datetime.now(timezone.utc) + timedelta(seconds=sec)).isoformat()


def coerce_optional_int(value: object, default: int) -> int:
    """把可选的整数配置值归一化为 int。

    用于 fencing 参数：调用方可能省略（None）或用任意类型传入，
    非法值回退到默认值，避免跨类型比较导致 SQL 永远匹配不上。
    """

    if value is None:
        return int(default)
    if isinstance(value, bool):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


from whisper_stt_service.repo.dependency_payload import (
    decode_dependency_payload,
    encode_dependency_payload,
)
from whisper_stt_service.repo.models import (
    ClaimedTask,
    EnqueueResult,
    TaskExecutionContext,
)


class _PublishFencedError(RuntimeError):
    """发布产物时 SQL 层 fencing 失配（内部使用，最终转换为 False 返回）。

    前置校验通过后仍可能出现并发竞争（例如另一进程刚完成回收），此时必须
    回滚整个事务，避免留下“文件已发布但任务未置 succeeded”的不一致状态。
    """

    def __init__(self, task_id: str) -> None:
        super().__init__(f"fenced_stale_result:{task_id}")
        self.task_id = task_id


class JobRepository:
    """基于 SQLite 的 job/task 仓储实现。"""

    def __init__(
        self,
        db: Database,
        *,
        stage_max_retries: dict[str, int] | None = None,
        stage_timeouts: dict[str, int] | None = None,
        stage_effective_defaults: dict[str, dict] | None = None,
        log_root: Path | None = None,
        artifact_root: Path | None = None,
        allowed_video_roots: Sequence[str | Path] | str | Path | None = None,
        allowed_copy_back_roots: Sequence[str | Path] | str | Path | None = None,
        allowed_video_extensions: Sequence[str] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """注入数据库访问对象与可选阶段默认配置。

        路径安全（SPEC H）：repo 内部入队同样受白名单约束，配置为空时保持
        “不限视频目录 / copy_back 仅视频同目录”的安全默认。
        """

        self.db = db
        # 入队与执行前使用的安全策略；roots 允许 str（逗号分隔）/list/tuple/Path。
        self.security = SessionConfig(
            allowed_video_roots=allowed_video_roots,  # type: ignore[arg-type]
            allowed_copy_back_roots=allowed_copy_back_roots,  # type: ignore[arg-type]
            allowed_video_extensions=(
                tuple(allowed_video_extensions)
                if allowed_video_extensions
                else SessionConfig().allowed_video_extensions
            ),
        )
        self.stage_max_retries = stage_max_retries or {
            "extract": 2,
            "stt": 2,
            "stt_whisperx": 2,
            "translate": 2,
        }
        self.stage_timeouts = stage_timeouts or {
            "extract": 1200,
            "stt": 7200,
            "stt_whisperx": 7200,
            "translate": 7200,
        }
        self.stage_effective_defaults = stage_effective_defaults or {
            "extract": {},
            "stt": {
                "device": "auto",
                "compute_type": "auto",
                "batch_size": 8,
                "beam_size": 5,
                "best_of": 5,
                "patience": 1.0,
                "condition_on_previous_text": False,
                "vad_filter": True,
                "vad_threshold": 0.45,
                "vad_min_speech_duration_ms": 200,
                "vad_max_speech_duration_s": 18.0,
                "vad_min_silence_duration_ms": 700,
                "vad_speech_pad_ms": 300,
                "no_speech_threshold": 0.6,
                "compression_ratio_threshold": 2.2,
                "log_prob_threshold": -1.0,
                "hallucination_silence_threshold": 1.5,
                "initial_prompt": "",
                "hotwords": "",
            },
            "stt_whisperx": {
                "model": "models/faster-whisper-small",
                "device": "auto",
                "compute_type": "auto",
                "batch_size": 8,
                "vad_config_path": "models/whisperx/vad/pyannote/config.yaml",
                "align_model_root": "models/whisperx/align",
                "align_enabled": True,
                "vad_backend": "pyannote",
                "vad_onset": 0.35,
                "vad_offset": 0.2,
                "local_files_only": True,
            },
            "translate": {
                "chunk_minutes": 30,
                "retry": 4,
                "copy_back": "__video_dir__",
            },
        }
        self.log_root = log_root or Path("./tmp/logs")
        self.artifact_root = artifact_root or (self.log_root.parent / "artifacts")
        # 统一把各阶段重试预算归一化为非负整数；attempt 从 0 开始计数，
        # “attempt + 1 > max_retries” 即超预算。存量库里的负数预算按 0 处理，
        # 避免旧数据触发无限重试。
        self.stage_max_retries = {
            str(stage): max(int(value), 0)
            for stage, value in self.stage_max_retries.items()
        }
        # SPEC F/G：claim_next 延迟与 SQLite busy 观测。
        # 使用单调时钟（可注入以便测试），只保留有界样本，避免长跑进程内存增长。
        self._clock: Callable[[], float] = clock or time.monotonic
        self._metrics_lock = threading.Lock()
        self._claim_samples_ms: deque[float] = deque(maxlen=_CLAIM_SAMPLE_WINDOW)
        self._claim_count = 0
        self._claim_busy_count = 0
        self._claim_error_count = 0

    def _count_queue_ahead(self, conn, now: str) -> int:
        """统计当前排在新任务前方的 job 数量。"""

        row = conn.execute(
            """
            SELECT COUNT(1) AS c
            FROM jobs
            WHERE status IN ('queued', 'running') AND created_at < ?
            """,
            (now,),
        ).fetchone()
        return int(row["c"]) if row is not None else 0

    def _build_log_paths(
        self, job_id: str, stage: str, task_id: str
    ) -> tuple[str, str]:
        """按规范生成任务日志目录与日志文件路径。"""

        log_dir = self.log_root / job_id / stage / task_id
        return str(log_dir), str(log_dir / "task.log")

    def _build_output_paths(self, job_id: str, video_path: str) -> tuple[str, str]:
        """把阶段产物放到可写 artifact 目录，避免污染源视频目录。

        SPEC B.3：ASR 产物落在 `asr/`、翻译产物落在 `translate/`；jobs 表的
        output_ja_path/output_zh_path 继续指向这两个最终产物，保持 API 兼容。
        """

        paths = self._build_artifact_paths(job_id, video_path)
        return paths["ja_srt"], paths["zh_srt"]

    def _build_artifact_paths(self, job_id: str, video_path: str) -> dict[str, str]:
        """按 SPEC B.3 计算单 job 的全部产物路径（每阶段唯一写者）。

        artifacts/<job_id>/
          audio/<stem>.16k.wav                   # extract 输出
          asr/<stem>.ja.srt                      # stt / stt_whisperx 输出
          translate/<stem>.zh.srt                # translate 输出
          translate/<stem>.zh.srt.progress.json  # 翻译检查点
        """

        artifact_dir = self.artifact_root / job_id
        stem = Path(video_path).stem
        audio_dir = artifact_dir / "audio"
        asr_dir = artifact_dir / "asr"
        translate_dir = artifact_dir / "translate"
        zh_srt = translate_dir / f"{stem}.zh.srt"
        return {
            "audio_dir": str(audio_dir),
            "asr_dir": str(asr_dir),
            "translate_dir": str(translate_dir),
            "wav_path": str(audio_dir / f"{stem}.16k.wav"),
            "ja_srt": str(asr_dir / f"{stem}.ja.srt"),
            "zh_srt": str(zh_srt),
            "progress_path": f"{zh_srt}.progress.json",
        }

    def _resolve_copy_back_dir(
        self, normalized_job_config: dict, canonical_video_path: str
    ) -> str:
        """解析并校验 translate.copy_back 目标目录（SPEC H）。

        配置优先级与 executor/common.build_translate_effective_config 保持一致：
        job_config.translate.copy_back > [translation] copy_back（经
        stage_effective_defaults 注入）> "__video_dir__" 历史默认。

        此前只读 normalized_job_config，导致全局 [translation] copy_back
        配置被静默忽略；这里必须回退到 stage 默认值。

        repo 侧传 strict_parent=False：入队时视频可能尚未落盘、目标目录也可能
        稍后才创建，因此只做 canonical 规范化与白名单判定，存在性留给 API
        严格校验与执行期。
        """

        translate_cfg = normalized_job_config.get("translate") or {}
        configured = translate_cfg.get("copy_back")
        if configured is None or str(configured).strip() == "":
            # 回退到 stage 级默认（[translation] copy_back），再回退到历史默认。
            defaults = self.stage_effective_defaults.get("translate") or {}
            configured = defaults.get("copy_back")
        if configured is None or str(configured).strip() == "":
            configured = "__video_dir__"
        try:
            resolved = resolve_copy_back_dir(
                configured,
                video_path=canonical_video_path,
                session=self.security,
                strict_parent=False,
            )
        except SecurityError as exc:
            # 统一以 ValueError 冒泡：路由已把 ValueError 映射为 400。
            raise ValueError(exc.code) from exc
        return str(resolved)
    def _build_stage_io(
        self,
        stage: str,
        *,
        video_path: str,
        paths: dict[str, str],
        copy_back_dir: str,
    ) -> dict[str, str]:
        """为单个阶段生成 io 段（写入 task_config["io"]，不新增 DB 列）。

        extract 吃视频、产出 wav；ASR 吃 wav、产出 ja.srt；translate 吃 ja.srt、
        产出 zh.srt（附检查点路径与回写目录）。
        """

        if stage == "extract":
            return {"input_path": video_path, "output_path": paths["wav_path"]}
        if stage in {"stt", "stt_whisperx"}:
            return {"input_path": paths["wav_path"], "output_path": paths["ja_srt"]}
        if stage == "translate":
            return {
                "input_path": paths["ja_srt"],
                "output_path": paths["zh_srt"],
                "progress_path": paths["progress_path"],
                "copy_back_dir": copy_back_dir,
            }
        return {"input_path": video_path, "output_path": paths["ja_srt"]}

    def list_job_ids_by_status(
        self, *, job_ids: list[str], statuses: tuple[str, ...]
    ) -> set[str]:
        """返回给定 job_id 中状态命中的集合，用于批量清理判定。"""

        if not job_ids or not statuses:
            return set()
        with self.db.connect() as conn:
            id_placeholders = ",".join(["?"] * len(job_ids))
            status_placeholders = ",".join(["?"] * len(statuses))
            rows = conn.execute(
                f"SELECT job_id FROM jobs WHERE job_id IN ({id_placeholders}) AND status IN ({status_placeholders})",
                tuple(job_ids) + tuple(statuses),
            ).fetchall()
            return {str(row["job_id"]) for row in rows}

    def is_job_completed_for_cleanup(
        self, *, job_id: str, statuses: tuple[str, ...]
    ) -> bool:
        """按给定状态判定单个 job 是否可清理。"""

        if not job_id or not statuses:
            return False
        with self.db.connect() as conn:
            placeholders = ",".join(["?"] * len(statuses))
            row = conn.execute(
                f"SELECT 1 AS ok FROM jobs WHERE job_id=? AND status IN ({placeholders}) LIMIT 1",
                (job_id, *statuses),
            ).fetchone()
            return row is not None

    def _build_effective_config(
        self, stage: str, overrides: dict | None = None
    ) -> dict:
        """按 stage 生成 task 的最终生效配置。"""

        base = {
            "timeout_sec": int(self.stage_timeouts.get(stage, 3600)),
            "max_retries": max(
                int(self.stage_max_retries.get(stage, 2) or 0), 0
            ),
        }
        base.update(dict(self.stage_effective_defaults.get(stage, {})))
        if overrides:
            base.update(dict(overrides))
        return base

    def _loads_or_empty(self, raw: str | None) -> dict:
        """解析 JSON 文本，异常时返回空对象。"""

        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {}

    def enqueue(
        self,
        video_path: str,
        language: str,
        dag: dict | None = None,
        job_config: dict | None = None,
        request_id: str | None = None,
    ) -> EnqueueResult:
        """执行同路径幂等/拒绝判定，并按 DAG 创建 job/tasks。

        SPEC H：即使绕过 API 直接调用 repo，也必须经过绝对路径/扩展名/白名单
        校验（不强制文件已存在，保留“先入队后落盘”的历史用法）。
        SPEC B.3：为每个阶段写入 io 段，产物按 audio/asr/translate 分目录归属。
        """

        normalized_dag = normalize_and_validate_dag(dag)
        normalized_job_config = normalize_and_validate_job_config(
            job_config, normalized_dag
        )
        # 规范化成 canonical 绝对路径，让同一文件的不同写法折叠为同一个键。
        canonical_video_path = str(
            normalize_video_path(video_path, session=self.security)
        )
        copy_back_dir = self._resolve_copy_back_dir(
            normalized_job_config, canonical_video_path
        )
        # request_id 允许为空（脚本/测试直接入队）；仅在非空时写入元数据。
        resolved_request_id = (
            request_id.strip() if isinstance(request_id, str) and request_id.strip() else None
        )

        now = _now()
        with self.db.tx() as conn:
            # 历史 job 存的是入队时的原始写法；这里用 canonical 与原始两个精确值做
            # IN 查询，两者都命中 idx_jobs_video_created，避免全表扫描又保住向后兼容。
            row = conn.execute(
                "SELECT job_id FROM jobs WHERE video_path IN (?, ?) ORDER BY created_at DESC LIMIT 1",
                (canonical_video_path, video_path),
            ).fetchone()
            if row is not None:
                latest_job_id = row["job_id"]
                statuses = [
                    r["status"]
                    for r in conn.execute(
                        "SELECT status FROM tasks WHERE job_id=? ORDER BY stage",
                        (latest_job_id,),
                    ).fetchall()
                ]
                if statuses and all(s == "queued" for s in statuses):
                    return EnqueueResult(
                        latest_job_id,
                        False,
                        "idempotent_returned",
                        0,
                        "default" if dag is None else "custom",
                        [str(x["stage"]) for x in normalized_dag["stages"]],
                    )
                if any(s in {"claimed", "succeeded", "failed"} for s in statuses):
                    return EnqueueResult(
                        latest_job_id,
                        False,
                        "rejected_started",
                        0,
                        "default" if dag is None else "custom",
                        [str(x["stage"]) for x in normalized_dag["stages"]],
                    )

            queue_ahead = self._count_queue_ahead(conn, now)
            task_name = _derive_task_name(video_path)
            normalized_stages = normalized_dag["stages"]

            for _ in range(_ID_RETRY_LIMIT):
                readable_ts = _readable_timestamp()
                job_id = _build_readable_job_id(task_name, readable_ts)
                # B.3：路径在入队时一次算好，task_config.io 与 jobs 表字段共用同一份。
                paths = self._build_artifact_paths(job_id, canonical_video_path)
                ja, zh = paths["ja_srt"], paths["zh_srt"]
                stage_to_task_id = {
                    str(item["stage"]): _build_readable_task_id(
                        task_name,
                        str(item["stage"]),
                        readable_ts,
                    )
                    for item in normalized_stages
                }

                conn.execute("SAVEPOINT enqueue_id_retry")
                try:
                    conn.execute(
                        "INSERT INTO jobs(job_id,video_path,source_language,status,output_ja_path,output_zh_path,dag_json,job_config_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (
                            job_id,
                            canonical_video_path,
                            language,
                            "queued",
                            ja,
                            zh,
                            json.dumps(normalized_dag, ensure_ascii=False),
                            json.dumps(normalized_job_config, ensure_ascii=False)
                            if normalized_job_config
                            else None,
                            now,
                            now,
                        ),
                    )

                    for item in normalized_stages:
                        stage = str(item["stage"])
                        task_id = stage_to_task_id[stage]
                        stage_overrides = normalized_job_config.get(stage, {})
                        effective_config = self._build_effective_config(
                            stage, stage_overrides
                        )
                        task_config = {
                            "stage": stage,
                            "effective_config": effective_config,
                            # SPEC B.3：输入/输出路径随任务固化，不新增 DB 列。
                            "io": self._build_stage_io(
                                stage,
                                video_path=canonical_video_path,
                                paths=paths,
                                copy_back_dir=copy_back_dir,
                            ),
                            # SPEC G：把创建该 job 的 request_id 固化下来，便于
                            # 任务日志与 API 链路关联；无 DB 列变更。
                            "metadata": {"request_id": resolved_request_id},
                        }
                        log_dir, log_file = self._build_log_paths(
                            job_id, stage, task_id
                        )
                        conn.execute(
                            "INSERT INTO tasks(task_id,job_id,stage,status,depends_on_task_id,task_config_json,max_retries,timeout_sec,log_dir,log_file,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                task_id,
                                job_id,
                                stage,
                                "queued",
                                None,
                                json.dumps(task_config, ensure_ascii=False),
                                max(int(effective_config.get("max_retries", 2)), 0),
                                int(effective_config.get("timeout_sec", 3600)),
                                log_dir,
                                log_file,
                                now,
                                now,
                            ),
                        )

                    for item in normalized_stages:
                        stage = str(item["stage"])
                        dep_stage_names = [str(x) for x in item["depends_on"]]
                        dep_task_ids = [
                            stage_to_task_id[name] for name in dep_stage_names
                        ]
                        payload = encode_dependency_payload(dep_task_ids)
                        conn.execute(
                            "UPDATE tasks SET depends_on_task_id=?, updated_at=? WHERE task_id=?",
                            (payload, now, stage_to_task_id[stage]),
                        )
                except sqlite3.IntegrityError as exc:
                    conn.execute("ROLLBACK TO SAVEPOINT enqueue_id_retry")
                    conn.execute("RELEASE SAVEPOINT enqueue_id_retry")
                    if _is_id_collision_error(exc):
                        continue
                    raise
                else:
                    conn.execute("RELEASE SAVEPOINT enqueue_id_retry")
                    return EnqueueResult(
                        job_id,
                        True,
                        "created",
                        queue_ahead,
                        "default" if dag is None else "custom",
                        [str(x["stage"]) for x in normalized_stages],
                    )

            raise RuntimeError("enqueue_id_generation_exhausted")

    def force_mark_any_stage_started(self, job_id: str) -> None:
        """测试辅助：强制把 extract 置为 claimed，模拟任务已开始。"""

        with self.db.tx() as conn:
            conn.execute(
                "UPDATE tasks SET status='claimed', started_at=?, updated_at=? WHERE job_id=? AND stage='extract'",
                (_now(), _now(), job_id),
            )

    def _fetch_task_row(self, conn, task_id: str):
        """读取任务的状态/fencing 所需最小字段，不存在时返回 None。"""

        return conn.execute(
            "SELECT task_id, job_id, stage, status, attempt, max_retries, lease_owner, "
            "lease_expires_at, depends_on_task_id, last_error FROM tasks WHERE task_id=?",
            (task_id,),
        ).fetchone()

    def task_lease_matches(
        self, task_id: str, *, worker_id: str | None, attempt: int | None
    ) -> bool:
        """只读校验任务租约仍由给定 worker/attempt 持有。

        用于运行时在不写库的情况下做协作式取消检查；worker_id 或 attempt
        省略时跳过对应条件，便于旧调用方与临时脚本复用。
        """

        clauses = ["task_id=?", "status='claimed'"]
        params: list[object] = [task_id]
        if worker_id is not None:
            clauses.append("lease_owner=?")
            params.append(worker_id)
        if attempt is not None:
            clauses.append("attempt=?")
            params.append(coerce_optional_int(attempt, 0))
        with self.db.connect() as conn:
            row = conn.execute(
                f"SELECT 1 AS ok FROM tasks WHERE {' AND '.join(clauses)} LIMIT 1",
                tuple(params),
            ).fetchone()
            return row is not None

    def heartbeat_task(
        self, task_id: str, *, worker_id: str, attempt: int, lease_timeout_sec: int
    ) -> bool:
        """续租心跳：仅当任务仍由本 worker 的同一 attempt 持有时成功。

        续租 SQL 带 fencing（status/lease_owner/attempt）。返回 False 表示任务
        已被回收或被其他 worker 接管，调用方应立即中止执行并放弃写回结果。
        """

        now = _now()
        with self.db.tx() as conn:
            changed = conn.execute(
                """
                UPDATE tasks
                SET lease_expires_at=?, updated_at=?
                WHERE task_id=? AND status='claimed' AND lease_owner=? AND attempt=?
                """,
                (
                    _lease_expire(lease_timeout_sec),
                    now,
                    task_id,
                    worker_id,
                    coerce_optional_int(attempt, 0),
                ),
            ).rowcount
            return int(changed) > 0

    def reclaim_expired_leases(self, *, max_retries_default: int = 2) -> dict:
        """扫描并回收过期租约，返回本轮回收统计。

        单任务事务内：attempt+1 回到 queued 并记录 lease_expired；若超预算则
        置 failed（lease_expired_retry_exhausted）并执行失败级联。每次回收都会
        写任务日志事件 lease_reclaimed 并刷新所属 job 状态。
        """

        now = _now()
        stats: dict[str, object] = {
            "reclaimed": 0,
            "failed": 0,
            "reclaimed_task_ids": [],
            "failed_task_ids": [],
        }
        with self.db.tx() as conn:
            # 回收前二次校验：只处理“仍是 claimed 且真的过期”的任务行。
            rows = conn.execute(
                "SELECT task_id, job_id, stage, attempt, max_retries, lease_owner, log_file "
                "FROM tasks WHERE status='claimed' AND lease_expires_at IS NOT NULL "
                "AND lease_expires_at < ?",
                (now,),
            ).fetchall()

            for row in rows:
                task_id = str(row["task_id"])
                job_id = str(row["job_id"])
                old_owner = row["lease_owner"]
                attempt = int(row["attempt"]) + 1
                max_retries = row["max_retries"]
                if max_retries is None:
                    max_retries = max_retries_default
                exhausted = attempt > int(max_retries)

                if exhausted:
                    outcome = "failed"
                    changed = conn.execute(
                        """
                        UPDATE tasks
                        SET status='failed', attempt=?, lease_owner=NULL,
                            lease_expires_at=NULL, last_error=?, finished_at=?, updated_at=?
                        WHERE task_id=? AND status='claimed' AND lease_expires_at < ?
                        """,
                        (
                            attempt,
                            "lease_expired_retry_exhausted",
                            now,
                            now,
                            task_id,
                            now,
                        ),
                    ).rowcount
                else:
                    outcome = "reclaimed"
                    changed = conn.execute(
                        """
                        UPDATE tasks
                        SET status='queued', attempt=?, lease_owner=NULL,
                            lease_expires_at=NULL, last_error='lease_expired', updated_at=?
                        WHERE task_id=? AND status='claimed' AND lease_expires_at < ?
                        """,
                        (attempt, now, task_id, now),
                    ).rowcount

                if int(changed) == 0:
                    # 任务在本轮扫描期间已被心跳续租或被他人接管，跳过。
                    continue

                if exhausted:
                    self._cascade_blocked(conn, task_id)
                else:
                    # 上游失败后重新排队时，下游可能已被级联置 blocked；
                    # 本任务仍有重试预算，撤销其 blocked 以便继续执行。
                    self._unblock_dependents(conn, task_id)
                self._refresh_job_status(conn, job_id)

                stats[outcome] = int(stats[outcome]) + 1
                key = (
                    "reclaimed_task_ids" if outcome == "reclaimed" else "failed_task_ids"
                )
                stats[key].append(task_id)
                self._append_task_log_event(
                    job_id=job_id,
                    task_id=task_id,
                    stage=str(row["stage"]),
                    event="lease_reclaimed",
                    result=outcome,
                    log_file=row["log_file"],
                    extra={
                        "worker_id": old_owner,
                        "attempt": attempt,
                        "max_retries": int(max_retries),
                        "error": (
                            "lease_expired_retry_exhausted"
                            if exhausted
                            else "lease_expired"
                        ),
                    },
                )
                LOGGER.warning(
                    "lease reclaimed: task_id=%s job_id=%s old_owner=%s attempt=%s result=%s",
                    task_id,
                    job_id,
                    old_owner,
                    attempt,
                    outcome,
                )
        return stats

    def _cascade_blocked(self, conn, failed_task_id: str) -> list[str]:
        """把失败任务的全部直接/间接下游 queued 任务置为 blocked。

        返回被置为 blocked 的 task_id 列表。级联在同一事务内完成，保证
        “上游 failed → 下游 blocked → job failed”没有中间可见状态。
        """

        blocked: list[str] = []
        visited: set[str] = {failed_task_id}
        frontier: list[str] = [failed_task_id]
        while frontier:
            upstream_ids = frontier
            holders = ",".join(["?"] * len(upstream_ids))
            candidates = conn.execute(
                f"SELECT task_id, job_id, stage, depends_on_task_id "
                f"FROM tasks WHERE status='queued' AND depends_on_task_id IS NOT NULL "
                f"AND task_id NOT IN ({holders})",
                tuple(upstream_ids),
            ).fetchall()
            next_frontier: list[str] = []
            for row in candidates:
                task_id = str(row["task_id"])
                if task_id in visited:
                    continue
                try:
                    dep_ids = decode_dependency_payload(row["depends_on_task_id"])
                except ValueError:
                    # 依赖载荷非法由 claim_next 统一判定为失败，这里不越权处理。
                    continue
                if not any(dep in visited for dep in dep_ids):
                    continue
                changed = conn.execute(
                    """
                    UPDATE tasks
                    SET status='blocked', last_error=?, lease_owner=NULL,
                        lease_expires_at=NULL, updated_at=?
                    WHERE task_id=? AND status='queued'
                    """,
                    (f"upstream_failed:{failed_task_id}", _now(), task_id),
                ).rowcount
                if int(changed) == 0:
                    continue
                blocked.append(task_id)
                visited.add(task_id)
                next_frontier.append(task_id)
                LOGGER.warning(
                    "task blocked by upstream failure: task_id=%s job_id=%s upstream=%s",
                    task_id,
                    row["job_id"],
                    failed_task_id,
                )
            frontier = next_frontier
        return blocked

    def _unblock_dependents(self, conn, upstream_task_id: str) -> list[str]:
        """撤销“因该上游失败而被 blocked”的下游任务，回到 queued。

        用于上游被重新排队重跑时（如租约回收且仍有重试预算），避免下游
        永久停留在 blocked。返回被恢复的 task_id 列表。
        """

        suffix = f"upstream_failed:{upstream_task_id}"
        rows = conn.execute(
            "SELECT task_id FROM tasks WHERE status='blocked' AND last_error=?",
            (suffix,),
        ).fetchall()
        restored: list[str] = []
        for row in rows:
            changed = conn.execute(
                "UPDATE tasks SET status='queued', lease_owner=NULL, "
                "lease_expires_at=NULL, updated_at=? WHERE task_id=? AND status='blocked'",
                (_now(), row["task_id"]),
            ).rowcount
            if int(changed) == 0:
                continue
            restored.append(str(row["task_id"]))
        return restored

    def list_finished_tasks_since(self, since_iso: str) -> list[dict[str, str]]:
        """返回 `finished_at >= since_iso` 的终态任务，供 /queue/summary 统计吞吐。

        用于 SPEC G 的“最近 5 分钟完成数/阶段”。包含 blocked 与 cancelled：
        它们同样是终态，若不计入会让吞吐与队列深度对不上。

        查询走既有 `finished_at` 列与 `idx_tasks_stage_status_created`，
        不需要任何 schema 变更。返回按 stage/status/finished_at 三项的轻量字典。
        """

        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT stage, status, finished_at
                  FROM tasks
                 WHERE finished_at IS NOT NULL
                   AND finished_at >= ?
                   AND status IN ('succeeded', 'failed', 'blocked', 'cancelled')
                """,
                (since_iso,),
            ).fetchall()
        return [
            {
                "stage": str(row["stage"]),
                "status": str(row["status"]),
                "finished_at": str(row["finished_at"]),
            }
            for row in rows
        ]
    def get_claim_metrics(self) -> dict[str, Any]:
        """返回 claim_next 的观测指标（SPEC F/G）。

        字段：
        - `count`：claim_next 调用次数（不论是否领到任务）。
        - `p95_ms`：最近 `_CLAIM_SAMPLE_WINDOW` 次调用的 P95 耗时（毫秒）。
        - `p50_ms`：同窗口的中位数，便于判断是否是长尾问题。
        - `busy`：SQLite busy/locked 次数（瞬时争用）。
        - `errors`：其它 sqlite 错误次数（busy 之外）。
        - `samples`：当前窗口内的样本数。

        线程安全：读取时持锁并复制样本，不阻塞 claim_next 的写入。
        """

        with self._metrics_lock:
            samples = sorted(self._claim_samples_ms)
            count = self._claim_count
            busy = self._claim_busy_count
            errors = self._claim_error_count
        return {
            "count": count,
            "p50_ms": _percentile(samples, 50),
            "p95_ms": _percentile(samples, 95),
            "busy": busy,
            "errors": errors,
            "samples": len(samples),
        }

    def _record_claim_sample(self, elapsed_ms: float) -> None:
        """记录一次 claim_next 耗时（有界窗口）。"""

        with self._metrics_lock:
            self._claim_count += 1
            self._claim_samples_ms.append(float(elapsed_ms))

    def _record_claim_error(self, exc: BaseException) -> None:
        """登记一次 claim_next 中的 sqlite 错误；busy 与其它错误分开计数。

        只计数不吞错：调用方在 `except` 中先调用本方法，然后原样 `raise`。
        """

        message = str(exc).lower()
        is_busy = any(marker in message for marker in _BUSY_ERROR_MARKERS)
        with self._metrics_lock:
            if is_busy:
                self._claim_busy_count += 1
            else:
                self._claim_error_count += 1
        level = logging.WARNING if is_busy else logging.ERROR
        LOGGER.log(
            level,
            "claim_next sqlite error: busy=%s error=%s",
            is_busy,
            exc,
        )
    def claim_next(
        self, stage: str, worker_id: str, lease_timeout_sec: int
    ) -> ClaimedTask | None:
        """原子领取指定阶段的下一个可执行任务。

        附带耗时与 busy 观测（SPEC F/G）。

        观测只读：计时样本写入有界窗口，sqlite 错误先计数再原样重抛，
        绝不改变领取语义、也不吞掉异常。按 (stage, status) 走
        `idx_tasks_stage_status_created`，不需要 task_deps 重构（SPEC F）。
        """

        started_at = self._clock()
        try:
            return self._claim_next_impl(stage, worker_id, lease_timeout_sec)
        except sqlite3.Error as exc:
            self._record_claim_error(exc)
            raise
        finally:
            self._record_claim_sample((self._clock() - started_at) * 1000.0)

    def _claim_next_impl(
        self, stage: str, worker_id: str, lease_timeout_sec: int
    ) -> ClaimedTask | None:
        """claim_next 的实际实现（事务内选取并加租约）。"""

        with self.db.tx() as conn:
            rows = conn.execute(
                """
                SELECT task_id, job_id, stage, depends_on_task_id, attempt
                FROM tasks
                WHERE stage=? AND status='queued'
                ORDER BY created_at ASC, task_id ASC
                """,
                (stage,),
            ).fetchall()

            for row in rows:
                try:
                    dep_ids = decode_dependency_payload(row["depends_on_task_id"])
                except ValueError:
                    # A.4 级联：损坏的依赖载荷让本任务直接失败，必须在**同一事务内**
                    # 把下游 queued 任务一并置为 blocked，否则会留下永久 queued 的
                    # “幽灵排队”（job 看似 failed，下游却永远不可领取）。
                    now_invalid = _now()
                    conn.execute(
                        "UPDATE tasks SET status='failed', finished_at=?, last_error=?, updated_at=? WHERE task_id=?",
                        (
                            now_invalid,
                            "invalid_dependency_payload",
                            now_invalid,
                            row["task_id"],
                        ),
                    )
                    blocked_ids = self._cascade_blocked(conn, str(row["task_id"]))
                    if blocked_ids:
                        LOGGER.warning(
                            "downstream blocked by invalid dependency payload: task_id=%s job_id=%s blocked=%s",
                            row["task_id"],
                            row["job_id"],
                            blocked_ids,
                        )
                    self._refresh_job_status(conn, row["job_id"])
                    continue

                if dep_ids:
                    holders = ",".join(["?"] * len(dep_ids))
                    dep_rows = conn.execute(
                        f"SELECT task_id, status FROM tasks WHERE task_id IN ({holders})",
                        tuple(dep_ids),
                    ).fetchall()
                    dep_status = {r["task_id"]: r["status"] for r in dep_rows}
                    if any(dep_status.get(dep) != "succeeded" for dep in dep_ids):
                        continue

                now = _now()
                changed = conn.execute(
                    """
                    UPDATE tasks
                    SET status='claimed', lease_owner=?, lease_expires_at=?, claimed_at=?, started_at=?, updated_at=?
                    WHERE task_id=? AND status='queued'
                    """,
                    (
                        worker_id,
                        _lease_expire(lease_timeout_sec),
                        now,
                        now,
                        now,
                        row["task_id"],
                    ),
                ).rowcount
                if changed == 0:
                    continue
                self._refresh_job_status(conn, row["job_id"])
                return ClaimedTask(
                    task_id=row["task_id"],
                    job_id=row["job_id"],
                    stage=row["stage"],
                    attempt=int(row["attempt"]),
                )
            return None

    def get_task_execution_context(self, task_id: str) -> TaskExecutionContext | None:
        """查询 worker 执行某任务所需的上下文。"""

        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    t.task_id,
                    t.job_id,
                    t.stage,
                    t.timeout_sec,
                    t.attempt,
                    t.max_retries,
                    t.log_dir,
                    t.log_file,
                    t.task_config_json,
                    t.lease_owner,
                    t.lease_expires_at,
                    j.video_path,
                    j.source_language,
                    j.output_ja_path,
                    j.output_zh_path
                FROM tasks t
                JOIN jobs j ON j.job_id=t.job_id
                WHERE t.task_id=?
                """,
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            task_config = self._loads_or_empty(row["task_config_json"])
            effective = task_config.get("effective_config")
            if not isinstance(effective, dict):
                task_config = {
                    "stage": row["stage"],
                    "effective_config": self._build_effective_config(row["stage"]),
                }
                task_config["effective_config"]["timeout_sec"] = int(row["timeout_sec"])
                task_config["effective_config"]["max_retries"] = int(row["max_retries"])
            io_cfg = task_config.get("io")
            if not isinstance(io_cfg, dict):
                io_cfg = {}
            stage_name = str(row["stage"])
            # SPEC B.3：优先使用入队时固化的 io；历史 job 无 io 时按旧规则回退，
            # 保证已入队的存量任务仍可被重跑。
            input_path = str(io_cfg.get("input_path") or "").strip()
            output_path = str(io_cfg.get("output_path") or "").strip()
            if not input_path or not output_path:
                legacy_input, legacy_output = self._legacy_stage_paths(
                    stage_name, row
                )
                input_path = input_path or legacy_input
                output_path = output_path or legacy_output
            return TaskExecutionContext(
                task_id=row["task_id"],
                job_id=row["job_id"],
                stage=row["stage"],
                video_path=row["video_path"],
                source_language=row["source_language"],
                output_ja_path=row["output_ja_path"],
                output_zh_path=row["output_zh_path"],
                timeout_sec=int(row["timeout_sec"]),
                attempt=int(row["attempt"]),
                max_retries=int(row["max_retries"]),
                log_dir=row["log_dir"],
                log_file=row["log_file"],
                task_config=task_config,
                input_path=input_path,
                output_path=output_path,
            )

    def _legacy_stage_paths(self, stage: str, row) -> tuple[str, str]:
        """历史 job（无 task_config.io）的输入/输出回退规则。

        extract 吃视频、ASR 吃视频（旧实现直接读原视频）、translate 吃 ja 产 zh；
        这是 SPEC B.3 之前的布局，仅用于保证存量任务可继续执行。
        """

        video_path = str(row["video_path"])
        if stage == "translate":
            return video_path, str(row["output_zh_path"])
        return video_path, str(row["output_ja_path"])

    def _refresh_job_status(self, conn, job_id: str, *, now: str | None = None) -> str:
        """按该 job 全部任务聚合刷新状态。

        聚合规则（A.4）：任一任务 failed 或 blocked → job failed；全部
        succeeded → job succeeded；存在 claimed → running；否则 queued。

        cancelled 修正：cancelled 是任务终态。若全部任务都是 cancelled，
        job 也必须进入终态，否则会永久停留在 queued 形成“幽灵排队”。
        由于 SPEC §6 规定 jobs.status 不得新增 cancelled，这里归入 failed
        （配合 tasks 全为 cancelled 表达“用户取消”），并写入 finished_at。
        """

        # 允许调用方传入冻结的 now：publish 事务里 rename 可能耗时，用同一时间戳
        # 保证 task 与 job 的 finished_at/updated_at 一致，也避免二次取时。
        frozen_now = now or _now()
        rows = conn.execute(
            "SELECT status FROM tasks WHERE job_id=?",
            (job_id,),
        ).fetchall()
        statuses = [r["status"] for r in rows]
        job_status = "queued"
        finished_at: str | None = None
        if statuses and all(s == "succeeded" for s in statuses):
            job_status = "succeeded"
            finished_at = frozen_now
        elif any(s in {"failed", "blocked"} for s in statuses):
            job_status = "failed"
            finished_at = frozen_now
        elif any(s == "claimed" for s in statuses):
            job_status = "running"
        elif statuses and all(s == "cancelled" for s in statuses):
            # 全部任务已取消：job 收敛为 failed 终态（jobs.status 不含 cancelled）。
            job_status = "failed"
            finished_at = frozen_now

        conn.execute(
            "UPDATE jobs SET status=?, updated_at=?, finished_at=COALESCE(?, finished_at) WHERE job_id=?",
            (job_status, frozen_now, finished_at, job_id),
        )
        return job_status

    def mark_task_succeeded(
        self,
        task_id: str,
        *,
        worker_id: str | None = None,
        attempt: int | None = None,
    ) -> bool:
        """把任务标记为 succeeded 并写入完成时间。

        传入 worker_id/attempt 时启用写回 fencing：仅当任务仍处于 claimed
        且租约归该 worker/attempt 时才生效。返回 False 表示写回被拒绝
        （任务已被回收或由新 attempt 接管），此时不刷新 job 状态。
        """

        now = _now()
        with self.db.tx() as conn:
            row = self._fetch_task_row(conn, task_id)
            if row is None:
                return False
            fenced = worker_id is not None or attempt is not None
            clauses = ["task_id=?", "status='claimed'"]
            params: list[object] = [task_id]
            if worker_id is not None:
                clauses.append("lease_owner=?")
                params.append(worker_id)
            if attempt is not None:
                clauses.append("attempt=?")
                params.append(coerce_optional_int(attempt, 0))

            if fenced:
                changed = conn.execute(
                    f"""
                    UPDATE tasks
                    SET status='succeeded', finished_at=?, lease_owner=NULL,
                        lease_expires_at=NULL, updated_at=?
                    WHERE {' AND '.join(clauses)}
                    """,
                    (now, now, *params),
                ).rowcount
                if int(changed) == 0:
                    LOGGER.warning(
                        "fenced stale result rejected (succeeded): task_id=%s job_id=%s "
                        "worker_id=%s attempt=%s current_status=%s current_owner=%s current_attempt=%s",
                        task_id,
                        row["job_id"],
                        worker_id,
                        attempt,
                        row["status"],
                        row["lease_owner"],
                        row["attempt"],
                    )
                    return False
            else:
                # 兼容旧调用（测试/脚本）：仅接受仍处于 claimed 的任务，
                # 避免对已回收/已终态任务做隐式覆盖。
                changed = conn.execute(
                    """
                    UPDATE tasks
                    SET status='succeeded', finished_at=?, lease_owner=NULL,
                        lease_expires_at=NULL, updated_at=?
                    WHERE task_id=? AND status='claimed'
                    """,
                    (now, now, task_id),
                ).rowcount
                if int(changed) == 0:
                    return False
            self._refresh_job_status(conn, row["job_id"])
            return True

    def with_task_lease(
        self,
        task_id: str,
        *,
        worker_id: str,
        attempt: int,
        action: Callable[[], None],
    ) -> bool:
        """在“租约仍有效”的临界区内执行 `action`，但不改变任务状态。

        用于**检查点/进度类的 fenced 写**（SPEC A.1/A.3、C.4）：旧 attempt 在租约
        被回收后必须无法再覆盖新 attempt 写下的检查点，否则会丢失整批已完成翻译。

        语义：
        - 开启 `BEGIN IMMEDIATE` 取得写锁，并在**同一事务内**校验任务仍是 claimed、
          `lease_owner == worker_id`、`attempt` 匹配且租约未过期；
        - 校验通过后才调用 `action`，随后提交事务（action 通常是同一临界区内的
          原子文件写，例如 runtime 传入的 `atomic_write_text`）；
        - 校验失败返回 False 且**不调用 action**，因此旧 attempt 不会写任何字节；
        - `action` 抛异常时事务回滚并把异常向上抛出（不吞错）。

        重要约束：
        - 本方法不接受任意“可穿租约”的回调链：action 应当短小、只做与本次领取相关
          的写入，不要在其中再次开启事务或长耗时计算（会长时间持有写锁）；
        - repo 层**不 import executor.atomic**（会造成包循环），原子写工具由调用方
          （runtime，已加载该模块）注入：
          `checkpoint_publish=lambda path, text: repo.with_task_lease(
              task_id, worker_id=wid, attempt=att,
              action=lambda: atomic_write_text(path, text))`。

        返回 True 表示临界区已执行；False 表示租约已失效（stale）。
        """

        now = _now()
        with self.db.tx() as conn:
            row = self._fetch_task_row(conn, task_id)
            if row is None:
                return False
            if not self._lease_is_valid(row, worker_id, attempt, now):
                LOGGER.warning(
                    "fenced stale lease rejected (action skipped): task_id=%s job_id=%s "
                    "worker_id=%s attempt=%s current_status=%s current_owner=%s current_attempt=%s",
                    task_id,
                    row["job_id"],
                    worker_id,
                    attempt,
                    row["status"],
                    row["lease_owner"],
                    row["attempt"],
                )
                return False
            action()
            return True

    def publish_task_artifacts(
        self,
        task_id: str,
        *,
        worker_id: str,
        attempt: int,
        artifacts: Sequence[tuple[str | Path, str | Path]],
        request_id: str | None = None,
        prepublish: Callable[[], None] | None = None,
    ) -> bool:
        """在 fencing 保护下把产物从私有 staging 发布到正式路径并置 succeeded。

        SPEC A.3 / B.3 的产物级 fencing。顺序是**先校验后落盘**：

        1. 开启 `BEGIN IMMEDIATE`（db.tx 已保证）拿到写锁；
        2. 在同一事务内校验任务仍是 claimed、`lease_owner == worker_id`、
           `attempt == attempt`，且租约 `lease_expires_at` 未过期；
        3. 校验通过后才对每个 (staged, final) 执行 `os.replace`（同文件系统原子）；
        4. 置 succeeded 并刷新 job 状态，然后提交。

        关键不变量：**绝不先写正式路径再补 SQL fencing**。旧 worker 在步骤 2 就会被
        拒绝，返回 False 且不触碰任何文件，因此不会覆盖新 attempt 的产物。

        租约与时间语义（重要）：
        - 事务开始时冻结一次 `now`，用它校验租约未过期；
        - 步骤 4 的二次校验**只匹配 status/owner/attempt，不再比较过期时间**：
          rename 可能耗时并跨过租约到期点，若那里仍要求“当前未过期”，就会出现
          “文件已发布但 UPDATE 命中 0 行”的不一致状态。租约在事务开始时刻有效，
          且 `BEGIN IMMEDIATE` 持有写锁，reaper/其它 worker 无法在本事务提交前接管，
          因此该事务期间视为持有有效租约；
        - 这只保证 **DB 状态机**的原子性，**不**保证文件系统的原子性。

        崩溃恢复（跨多个 rename 无法整体回滚，**不存在文件系统回滚**）：
        - rename 严格按调用方给定顺序执行，调用方应把**最关键的产物放在最后**
          （单产物任务天然满足）；
        - 若进程在 rename 之后、提交之前崩溃，事务回滚后任务仍是 claimed，租约过期
          后被 reaper 回收并以新 attempt 重跑，重跑会用 staging 重新覆盖正式路径，
          因此正式的“半发布”状态由重跑收敛，而不是被回滚；
        - 若 os.replace 中途抛错，**已 rename 的文件保留在正式路径上，不会被撤销**，
          本方法抛异常且不置 succeeded，任务交由重试收敛；正式路径的最终内容以最后
          一次成功发布（新 attempt）为准，旧 worker 因 fencing 无法再覆盖。

        参数：
        - `artifacts`：`[(staged_path, final_path), ...]`，staged 必须是本 attempt
          私有的暂存文件（例如产物旁的 `.tmp-<pid>-<uuid>`），final 是正式产物路径。
        - `request_id`：可选，用于写入 task 日志的 request_id 字段。
        - `prepublish`：可选，在**同一 BEGIN IMMEDIATE 事务内、租约校验通过之后、
          任何 rename 之前**执行的短校验/提交回调（runtime 用它做 copy_back 的
          dirfd 锚定 rename，见 R3-6）。约束：只做短小操作，不得做慢 I/O、不得再开
          事务；抛出异常会回滚 DB 事务并向上传播（调用方按 fenced failure 处理）。
          注意：回调或后续 rename 中途失败时，文件系统 rename 不可整体回滚，
          正式路径由下一次成功发布收敛（与 artifacts 的崩溃语义一致）。

        返回：True 表示已发布并置 succeeded；False 表示 fencing 失败（stale）。
        staging 文件在失败时保持原样，由调用方/清理逻辑回收。
        """

        now = _now()
        try:
            with self.db.tx() as conn:
                outcome = self._publish_with_lease(
                    conn,
                    task_id,
                    worker_id=worker_id,
                    attempt=attempt,
                    artifacts=artifacts,
                    now=now,
                    prepublish=prepublish,
                )
                if outcome is None:
                    return False
                job_id, stage, published = outcome
        except _PublishFencedError:
            # 事务已回滚；已 rename 的文件由重跑覆盖，不得报告成功。
            return False

        self._append_task_log_event(
            job_id=job_id,
            task_id=task_id,
            stage=stage,
            event="artifacts_published",
            result="succeeded",
            extra={
                "worker_id": worker_id,
                "attempt": coerce_optional_int(attempt, 0),
                "request_id": request_id,
                "published": published,
            },
        )
        return True

    def _publish_with_lease(
        self,
        conn,
        task_id: str,
        *,
        worker_id: str,
        attempt: int,
        artifacts: Sequence[tuple[str | Path, str | Path]],
        now: str,
        prepublish: Callable[[], None] | None = None,
    ) -> tuple[str, str, list[str]] | None:
        """事务内完成“校验 → rename → 置 succeeded”，返回 (job_id, stage, published)。

        返回 None 表示 fencing 失败（调用方应返回 False 且不得写文件）。
        """

        row = self._fetch_task_row(conn, task_id)
        if row is None:
            return None
        job_id = str(row["job_id"])
        stage = str(row["stage"])
        if not self._lease_is_valid(row, worker_id, attempt, now):
            LOGGER.warning(
                "fenced stale result rejected (publish): task_id=%s job_id=%s "
                "worker_id=%s attempt=%s current_status=%s current_owner=%s current_attempt=%s",
                task_id,
                job_id,
                worker_id,
                attempt,
                row["status"],
                row["lease_owner"],
                row["attempt"],
            )
            return None

        # 租约已确认仍属于本 attempt：在锁内、rename 之前执行调用方的短校验/
        # 提交回调（runtime 的 copy_back dirfd 锚定 rename）。异常向上抛，由
        # db.tx 回滚；此时尚未执行本方法的 artifacts rename。
        if prepublish is not None:
            prepublish()

        # 先按调用方顺序完成全部 rename；任何一步失败都抛异常并回滚 DB。
        published: list[str] = []
        for staged, final in artifacts:
            staged_path = Path(staged)
            final_path = Path(final)
            if not staged_path.exists():
                raise FileNotFoundError(f"staged artifact missing: {staged_path}")
            final_path.parent.mkdir(parents=True, exist_ok=True)
            # os.replace 在同一文件系统内是原子的：读者只会看到旧文件或新文件。
            os.replace(staged_path, final_path)
            published.append(str(final_path))

        # 二次校验只用 status/owner/attempt，**不再比较 lease_expires_at**：
        # rename 可能耗时并跨过租约到期点，若这里仍要求“当前时间未过期”就会出现
        # “文件已发布但 UPDATE 命中 0 行 → 回滚并返回 False”的不一致。租约有效性
        # 已在事务开始时用冻结的 now 校验过一次，且 BEGIN IMMEDIATE 持有写锁，
        # reaper/其他 worker 无法在本事务提交前接管该任务，因此这段时间内的租约
        # 视为有效。注意：这只是 DB 层的保证，**文件系统的 rename 本身不可回滚**。
        changed = conn.execute(
            """
            UPDATE tasks
            SET status='succeeded', finished_at=?, lease_owner=NULL,
                lease_expires_at=NULL, last_error=NULL, updated_at=?
            WHERE task_id=? AND status='claimed' AND lease_owner=? AND attempt=?
            """,
            (now, now, task_id, worker_id, coerce_optional_int(attempt, 0)),
        ).rowcount
        if int(changed) == 0:
            # 理论上与前置校验一致；SQL 层再兜一次。抛异常让 db.tx 回滚，
            # 调用方看到 False（而不是半发布 + succeeded）。
            raise _PublishFencedError(task_id)
        # 传入冻结的 now：避免 rename 之后再次取时钟（若实现改成重取当前时间，
        # 会在租约已过期时产生“文件已发布但状态未更新”的不一致）。
        self._refresh_job_status(conn, job_id, now=now)
        return job_id, stage, published

    def _lease_is_valid(
        self, row, worker_id: str, attempt: int, now: str
    ) -> bool:
        """校验领取信息：仍 claimed、owner/attempt 匹配且租约未过期。

        时间比较使用与 claim/heartbeat 相同的 ISO8601 UTC 字符串前缀语义
        （统一 `+00:00` 后缀，字典序即时间序）。
        """

        if str(row["status"]) != "claimed":
            return False
        if str(row["lease_owner"] or "") != str(worker_id):
            return False
        if int(row["attempt"]) != coerce_optional_int(attempt, 0):
            return False
        expires_at = row["lease_expires_at"]
        if not expires_at:
            return False
        return str(expires_at) > now

    def mark_task_failed(
        self,
        task_id: str,
        error_message: str,
        *,
        worker_id: str | None = None,
        attempt: int | None = None,
    ) -> str:
        """失败后按重试策略更新任务状态，并返回新状态。

        传入 worker_id/attempt 时启用写回 fencing（同 mark_task_succeeded）。
        进入终态 failed 时在同一事务内级联把下游 queued 任务置为 blocked。
        """

        with self.db.tx() as conn:
            row = self._fetch_task_row(conn, task_id)
            if row is None:
                return "missing"
            fenced = worker_id is not None or attempt is not None
            clauses = ["task_id=?", "status='claimed'"]
            params: list[object] = [task_id]
            if worker_id is not None:
                clauses.append("lease_owner=?")
                params.append(worker_id)
            if attempt is not None:
                clauses.append("attempt=?")
                params.append(coerce_optional_int(attempt, 0))

            next_attempt = int(row["attempt"]) + 1
            trimmed_error = error_message[:2048]
            now = _now()
            if next_attempt <= int(row["max_retries"]):
                new_status = "queued"
                set_sql = (
                    "status='queued', attempt=?, last_error=?, lease_owner=NULL, "
                    "lease_expires_at=NULL, updated_at=?"
                )
                set_params: tuple[object, ...] = (next_attempt, trimmed_error, now)
            else:
                new_status = "failed"
                set_sql = (
                    "status='failed', attempt=?, last_error=?, finished_at=?, "
                    "lease_owner=NULL, lease_expires_at=NULL, updated_at=?"
                )
                set_params = (next_attempt, trimmed_error, now, now)

            if fenced:
                where_sql = " AND ".join(clauses)
                changed = conn.execute(
                    f"UPDATE tasks SET {set_sql} WHERE {where_sql}",
                    (*set_params, *params),
                ).rowcount
                if int(changed) == 0:
                    LOGGER.warning(
                        "fenced stale result rejected (failed): task_id=%s job_id=%s "
                        "worker_id=%s attempt=%s current_status=%s current_owner=%s current_attempt=%s",
                        task_id,
                        row["job_id"],
                        worker_id,
                        attempt,
                        row["status"],
                        row["lease_owner"],
                        row["attempt"],
                    )
                    return "fenced_stale_result"
            else:
                # 兼容旧调用：非 fenced 时也要求任务仍处于 claimed。
                changed = conn.execute(
                    f"UPDATE tasks SET {set_sql} WHERE task_id=? AND status='claimed'",
                    (*set_params, task_id),
                ).rowcount
                if int(changed) == 0:
                    return "fenced_stale_result"

            if new_status == "failed":
                self._cascade_blocked(conn, task_id)
            else:
                self._unblock_dependents(conn, task_id)
            self._refresh_job_status(conn, row["job_id"])
            return new_status

    def rollback_claimed_task(
        self,
        task_id: str,
        reason: str = "shutdown_rollback",
        *,
        worker_id: str | None = None,
        attempt: int | None = None,
    ) -> bool:
        """把单个 claimed 任务回退到 queued。

        传入 worker_id/attempt 时按 fencing 条件回退，避免误回退已被接管的
        任务；返回是否实际回退。
        """

        with self.db.tx() as conn:
            row = self._fetch_task_row(conn, task_id)
            if row is None:
                return False
            clauses = ["task_id=?", "status='claimed'"]
            params: list[object] = [task_id]
            if worker_id is not None:
                clauses.append("lease_owner=?")
                params.append(worker_id)
            if attempt is not None:
                clauses.append("attempt=?")
                params.append(coerce_optional_int(attempt, 0))
            changed = conn.execute(
                f"UPDATE tasks SET status='queued', lease_owner=NULL, lease_expires_at=NULL, last_error=?, updated_at=? WHERE {' AND '.join(clauses)}",
                (reason, _now(), *params),
            ).rowcount
            if int(changed) == 0:
                return False
            self._unblock_dependents(conn, task_id)
            self._refresh_job_status(conn, row["job_id"])
            return True

    def rollback_claimed_tasks(
        self,
        *,
        lease_owner: str | None = None,
        task_ids: list[str] | None = None,
        reason: str = "shutdown_rollback",
    ) -> int:
        """批量回退 claimed 任务（有界停机的强制回滚 / 本进程轮转）。

        按 lease_owner（本进程 worker）或显式 task_ids 限定范围，返回回退条数。
        回退后下游 blocked 会被撤销，job 状态重新聚合。
        """

        clauses = ["status='claimed'"]
        params: list[object] = []
        if lease_owner is not None:
            clauses.append("lease_owner=?")
            params.append(lease_owner)
        if task_ids:
            clauses.append(f"task_id IN ({','.join(['?'] * len(task_ids))})")
            params.extend(task_ids)
        if lease_owner is None and not task_ids:
            return 0
        with self.db.tx() as conn:
            rows = conn.execute(
                f"SELECT task_id, job_id FROM tasks WHERE {' AND '.join(clauses)}",
                tuple(params),
            ).fetchall()
            now = _now()
            rolled_back = 0
            job_ids: set[str] = set()
            for row in rows:
                changed = conn.execute(
                    """
                    UPDATE tasks
                    SET status='queued', lease_owner=NULL, lease_expires_at=NULL,
                        last_error=?, updated_at=?
                    WHERE task_id=? AND status='claimed'
                    """,
                    (reason, now, row["task_id"]),
                ).rowcount
                if int(changed) == 0:
                    continue
                rolled_back += 1
                job_ids.add(str(row["job_id"]))
                self._unblock_dependents(conn, str(row["task_id"]))
            for job_id in job_ids:
                self._refresh_job_status(conn, job_id)
            return rolled_back

    def rollback_claimed_leases(
        self,
        leases: Sequence[tuple[str, str, int]],
        reason: str = "shutdown_rollback",
    ) -> int:
        """按冻结的租约三元组回退 claimed 任务（R3-1，有界停机/强制回滚）。

        leases 的每项是执行实例领取时冻结的 (task_id, lease_owner, attempt)。
        回退 SQL 必须同时匹配这三项：

        - 只按 task_id 回退会撤掉“旧 runtime 本地 map 里残留、但数据库里已被
          新 worker/新 attempt 合法接管”的任务（Sol R3-1 实测：旧 runtime 的
          stop()/forceDrain 把新 owner 的 claimed 改回 queued）；
        - owner 含每 runtime boot nonce（R3-4），attempt 由 reaper/retry 推进，
          二者与 task_id 组成执行实例身份；任一不匹配即视为“不是本实例的
          租约”，保持数据库现状。

        整批在一个 BEGIN IMMEDIATE 事务内完成：逐条三元组 fenced UPDATE，
        成功的条目再撤销下游 blocked 并刷新 job 聚合状态。返回实际回退条数。
        """

        if not leases:
            return 0
        # 同一 task_id 只保留首个冻结身份：重复条目不应放大回退或破坏幂等。
        frozen: dict[str, tuple[str, int]] = {}
        for task_id, worker_id, attempt in leases:
            frozen.setdefault(str(task_id), (str(worker_id), int(attempt)))
        if not frozen:
            return 0

        now = _now()
        rolled_back = 0
        job_ids: set[str] = set()
        with self.db.tx() as conn:
            for task_id, (worker_id, attempt) in frozen.items():
                row = conn.execute(
                    "SELECT job_id FROM tasks WHERE task_id=?", (task_id,)
                ).fetchone()
                if row is None:
                    continue
                changed = conn.execute(
                    """
                    UPDATE tasks
                    SET status='queued', lease_owner=NULL, lease_expires_at=NULL,
                        last_error=?, updated_at=?
                    WHERE task_id=? AND status='claimed'
                      AND lease_owner=? AND attempt=?
                    """,
                    (reason, now, task_id, worker_id, attempt),
                ).rowcount
                if int(changed) == 0:
                    # 旧执行实例的租约已被回收/接管：绝不按 task_id 撤新 owner。
                    LOGGER.warning(
                        "fenced stale lease rollback skipped: task_id=%s worker_id=%s attempt=%s",
                        task_id,
                        worker_id,
                        attempt,
                    )
                    continue
                rolled_back += 1
                job_ids.add(str(row["job_id"]))
                self._unblock_dependents(conn, task_id)
            for job_id in job_ids:
                self._refresh_job_status(conn, job_id)
        return rolled_back

    def recover_claimed_to_queued(
        self, *, counts_as_attempt: bool = True, max_retries_default: int = 2
    ) -> dict:
        """服务启动时把全部 claimed 回退为 queued。

        按配置（recovery_counts_as_attempt）决定是否消耗重试预算。

        返回统计：recovered（回退条数）、failed（超预算转 failed 条数）。
        counts_as_attempt=False 时保持旧行为（回 queued 且 attempt 不变，仅调试用）。
        """

        now = _now()
        recovered = 0
        failed = 0
        with self.db.tx() as conn:
            rows = conn.execute(
                "SELECT task_id, job_id, stage, attempt, max_retries, lease_owner "
                "FROM tasks WHERE status='claimed'"
            ).fetchall()
            affected_jobs: set[str] = set()
            for row in rows:
                task_id = str(row["task_id"])
                job_id = str(row["job_id"])
                affected_jobs.add(job_id)
                if not counts_as_attempt:
                    changed = conn.execute(
                        """
                        UPDATE tasks
                        SET status='queued', lease_owner=NULL, lease_expires_at=NULL,
                            updated_at=?
                        WHERE task_id=? AND status='claimed'
                        """,
                        (now, task_id),
                    ).rowcount
                    if int(changed) > 0:
                        recovered += 1
                        self._unblock_dependents(conn, task_id)
                    continue

                attempt = int(row["attempt"]) + 1
                max_retries = row["max_retries"]
                if max_retries is None:
                    max_retries = max_retries_default
                if attempt > int(max_retries):
                    changed = conn.execute(
                        """
                        UPDATE tasks
                        SET status='failed', attempt=?, last_error=?, finished_at=?,
                            lease_owner=NULL, lease_expires_at=NULL, updated_at=?
                        WHERE task_id=? AND status='claimed'
                        """,
                        (
                            attempt,
                            "recovery_retry_exhausted",
                            now,
                            now,
                            task_id,
                        ),
                    ).rowcount
                    if int(changed) > 0:
                        failed += 1
                        self._cascade_blocked(conn, task_id)
                else:
                    changed = conn.execute(
                        """
                        UPDATE tasks
                        SET status='queued', attempt=?, last_error='recovered_after_restart',
                            lease_owner=NULL, lease_expires_at=NULL, updated_at=?
                        WHERE task_id=? AND status='claimed'
                        """,
                        (attempt, now, task_id),
                    ).rowcount
                    if int(changed) > 0:
                        recovered += 1
                        self._unblock_dependents(conn, task_id)

            # 对所有可能受影响的 job 重新聚合，避免状态残留为 running。
            job_ids = [
                r["job_id"] for r in conn.execute("SELECT job_id FROM jobs").fetchall()
            ]
            for job_id in job_ids:
                self._refresh_job_status(conn, job_id)
        LOGGER.info(
            "startup recovery: recovered=%s failed=%s counts_as_attempt=%s",
            recovered,
            failed,
            counts_as_attempt,
        )
        return {"recovered": recovered, "failed": failed}

    def get_job_detail(self, job_id: str) -> dict | None:
        """查询 job 元信息与任务明细。"""

        with self.db.connect() as conn:
            job = conn.execute(
                "SELECT * FROM jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if job is None:
                return None
            tasks = conn.execute(
                "SELECT task_id, stage, status, depends_on_task_id, task_config_json, attempt, max_retries, claimed_at, started_at, finished_at, last_error, lease_owner, lease_expires_at, updated_at FROM tasks WHERE job_id=? ORDER BY created_at ASC, task_id ASC",
                (job_id,),
            ).fetchall()
            normalized_tasks: list[dict] = []
            for task in tasks:
                item = dict(task)
                item["task_config"] = self._loads_or_empty(item.get("task_config_json"))
                if not isinstance(item["task_config"].get("effective_config"), dict):
                    item["task_config"] = {
                        "stage": item["stage"],
                        "effective_config": self._build_effective_config(item["stage"]),
                    }
                normalized_tasks.append(item)
            dag_payload = self._loads_or_empty(job["dag_json"])
            if not dag_payload:
                dag_payload = build_default_dag()
            return {
                "job_id": job["job_id"],
                "video_path": job["video_path"],
                "source_language": job["source_language"],
                "status": job["status"],
                "output_ja_path": job["output_ja_path"],
                "output_zh_path": job["output_zh_path"],
                "dag": dag_payload,
                "job_config": self._loads_or_empty(job["job_config_json"]),
                "created_at": job["created_at"],
                "updated_at": job["updated_at"],
                "finished_at": job["finished_at"],
                "tasks": normalized_tasks,
            }

    def get_job_latest_by_path(self, video_path: str) -> dict | None:
        """按视频路径返回最新一条 job。

        canonical 优先，保留原始写法兼容。

        与 enqueue 的幂等判定使用同一套键：canonical 路径和调用方原始写法都
        作为精确等值条件，两者共用 idx_jobs_video_created，不做全表扫描。
        """

        try:
            canonical = str(normalize_video_path(video_path, session=self.security))
        except SecurityError:
            # 查询不因非法路径报错：退回原始精确匹配，查不到即返回 None。
            canonical = video_path
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT job_id FROM jobs WHERE video_path IN (?, ?) ORDER BY created_at DESC LIMIT 1",
                (canonical, video_path),
            ).fetchone()
            if row is None:
                return None
            return self.get_job_detail(row["job_id"])

    def archive_job(self, job_id: str, reason: str) -> dict:
        """把 job 与其 tasks 迁移到归档表。"""

        archive_reason = reason.strip() if reason.strip() else "manual_archive"
        archived_at = _now()

        with self.db.tx() as conn:
            job = conn.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None:
                raise ValueError("job_not_found")

            statuses = [
                r["status"]
                for r in conn.execute(
                    "SELECT status FROM tasks WHERE job_id=?", (job_id,)
                ).fetchall()
            ]
            all_queued = bool(statuses) and all(s == "queued" for s in statuses)
            is_terminal = job["status"] in {"succeeded", "failed"}
            if not (all_queued or is_terminal):
                raise ValueError("archive_not_allowed")

            conn.execute(
                """
                INSERT INTO jobs_archive(
                    job_id, video_path, source_language, status,
                    output_ja_path, output_zh_path,
                    dag_json, job_config_json,
                    created_at, updated_at, finished_at,
                    archived_at, archive_reason
                )
                SELECT
                    job_id, video_path, source_language, status,
                    output_ja_path, output_zh_path,
                    dag_json, job_config_json,
                    created_at, updated_at, finished_at,
                    ?, ?
                FROM jobs
                WHERE job_id=?
                """,
                (archived_at, archive_reason, job_id),
            )
            conn.execute(
                """
                INSERT INTO tasks_archive(
                    task_id, job_id, stage, status,
                    depends_on_task_id, task_config_json,
                    attempt, max_retries, timeout_sec,
                    lease_owner, lease_expires_at,
                    claimed_at, started_at, finished_at,
                    last_error,
                    log_dir, log_file,
                    created_at, updated_at,
                    archived_at, archive_reason
                )
                SELECT
                    task_id, job_id, stage, status,
                    depends_on_task_id, task_config_json,
                    attempt, max_retries, timeout_sec,
                    lease_owner, lease_expires_at,
                    claimed_at, started_at, finished_at,
                    last_error,
                    log_dir, log_file,
                    created_at, updated_at,
                    ?, ?
                FROM tasks
                WHERE job_id=?
                """,
                (archived_at, archive_reason, job_id),
            )

            conn.execute("DELETE FROM tasks WHERE job_id=?", (job_id,))
            conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))

            return {
                "job_id": job_id,
                "archived": True,
                "archived_at": archived_at,
                "reason": archive_reason,
            }

    def list_jobs(
        self,
        *,
        page: int,
        page_size: int,
        status: str | None = None,
        video_path_like: str | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        language: str | None = None,
        has_failed_tasks: bool | None = None,
        sort_by: str = "created_at",
        order: str = "desc",
    ) -> dict:
        """分页查询 jobs，支持设计稿约定筛选字段。"""

        sort_key = sort_by if sort_by in {"created_at", "updated_at"} else "created_at"
        order_key = "ASC" if order.lower() == "asc" else "DESC"
        clauses: list[str] = []
        params: list[object] = []
        if status:
            clauses.append("j.status=?")
            params.append(status)
        if video_path_like:
            clauses.append("j.video_path LIKE ?")
            params.append(f"%{video_path_like}%")
        if created_from:
            clauses.append("j.created_at>=?")
            params.append(created_from)
        if created_to:
            clauses.append("j.created_at<=?")
            params.append(created_to)
        if language:
            clauses.append("j.source_language=?")
            params.append(language)
        if has_failed_tasks is True:
            clauses.append(
                "EXISTS (SELECT 1 FROM tasks tf WHERE tf.job_id=j.job_id AND tf.status='failed')"
            )
        if has_failed_tasks is False:
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM tasks tf WHERE tf.job_id=j.job_id AND tf.status='failed')"
            )

        where_sql = ""
        if clauses:
            where_sql = " WHERE " + " AND ".join(clauses)

        offset = max(page - 1, 0) * page_size
        with self.db.connect() as conn:
            total_row = conn.execute(
                f"SELECT COUNT(1) AS c FROM jobs j{where_sql}",
                tuple(params),
            ).fetchone()
            total = int(total_row["c"]) if total_row is not None else 0
            rows = conn.execute(
                f"SELECT j.* FROM jobs j{where_sql} ORDER BY j.{sort_key} {order_key} LIMIT ? OFFSET ?",
                tuple(params + [page_size, offset]),
            ).fetchall()
            return {
                "items": [dict(r) for r in rows],
                "total": total,
                "page": page,
                "page_size": page_size,
            }

    def list_tasks(
        self,
        *,
        page: int,
        page_size: int,
        stage: str | None = None,
        status: str | None = None,
        job_id: str | None = None,
        lease_owner: str | None = None,
        updated_from: str | None = None,
        updated_to: str | None = None,
    ) -> dict:
        """分页查询 tasks，支持运维筛选。"""

        clauses: list[str] = []
        params: list[object] = []
        if stage:
            clauses.append("stage=?")
            params.append(stage)
        if status:
            clauses.append("status=?")
            params.append(status)
        if job_id:
            clauses.append("job_id=?")
            params.append(job_id)
        if lease_owner:
            clauses.append("lease_owner=?")
            params.append(lease_owner)
        if updated_from:
            clauses.append("updated_at>=?")
            params.append(updated_from)
        if updated_to:
            clauses.append("updated_at<=?")
            params.append(updated_to)
        where_sql = ""
        if clauses:
            where_sql = " WHERE " + " AND ".join(clauses)

        offset = max(page - 1, 0) * page_size
        with self.db.connect() as conn:
            total_row = conn.execute(
                f"SELECT COUNT(1) AS c FROM tasks{where_sql}",
                tuple(params),
            ).fetchone()
            total = int(total_row["c"]) if total_row is not None else 0
            rows = conn.execute(
                f"SELECT * FROM tasks{where_sql} ORDER BY updated_at DESC, task_id ASC LIMIT ? OFFSET ?",
                tuple(params + [page_size, offset]),
            ).fetchall()
            return {
                "items": [dict(r) for r in rows],
                "total": total,
                "page": page,
                "page_size": page_size,
            }

    def queue_summary(self) -> dict:
        """返回各阶段状态计数摘要。"""

        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT stage, status, COUNT(1) AS c FROM tasks GROUP BY stage, status"
            ).fetchall()
            stage_map: dict[str, dict[str, int]] = {
                stage: {"queued": 0, "claimed": 0, "succeeded": 0, "failed": 0}
                for stage in STAGES
            }
            for row in rows:
                stage = row["stage"]
                status = row["status"]
                if stage not in stage_map:
                    stage_map[stage] = {}
                stage_map[stage][status] = int(row["c"])
            return {"stages": stage_map}

    def count_claimed_tasks(self) -> int:
        """返回全局处于 claimed 状态的任务数量。"""

        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(1) AS c FROM tasks WHERE status='claimed'"
            ).fetchone()
            return int(row["c"]) if row is not None else 0

    def count_jobs_with_blocked_tasks(self) -> int:
        """统计存在 blocked 任务的 job 数（停机状态/观测用的“幽灵排队”指标）。"""

        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT job_id) AS c FROM tasks WHERE status='blocked'"
            ).fetchone()
            return int(row["c"]) if row is not None else 0

    def count_claimed_by_owner_prefixes(self, prefixes: list[str]) -> int:
        """统计租约 owner 以给定前缀开头的 claimed 任务数。

        运行时用完整 worker_id 集合即可精确统计本进程在途任务；这里保留
        前缀版本供队列摘要之类的聚合观测使用。
        """

        if not prefixes:
            return 0
        clauses = " OR ".join(["lease_owner LIKE ?"] * len(prefixes))
        params = tuple(f"{prefix}%" for prefix in prefixes)
        with self.db.connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(1) AS c FROM tasks WHERE status='claimed' AND ({clauses})",
                params,
            ).fetchone()
            return int(row["c"]) if row is not None else 0

    def count_claimed_by_owners(self, owners: list[str]) -> int:
        """统计指定 lease_owner 集合下仍处于 claimed 的任务数。"""

        if not owners:
            return 0
        placeholders = ",".join(["?"] * len(owners))
        with self.db.connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(1) AS c FROM tasks WHERE status='claimed' AND lease_owner IN ({placeholders})",
                tuple(owners),
            ).fetchone()
            return int(row["c"]) if row is not None else 0

    def _append_task_log_event(
        self,
        *,
        job_id: str,
        task_id: str,
        stage: str,
        event: str,
        result: str,
        log_file: str | None = None,
        extra: dict | None = None,
    ) -> None:
        """向任务 jsonl 日志追加一条生命周期事件（仓储侧触发的事件）。

        SPEC G 统一字段（顺序固定）：
        timestamp, request_id, job_id, task_id, stage, worker_id,
        event, result, attempt, effective_config, error。
        其余信息全部放进 extra，避免字段名漂移。日志写入失败只记录告警，
        不影响数据库事务结果。
        """

        if not log_file:
            try:
                with self.db.connect() as conn:
                    row = conn.execute(
                        "SELECT log_file FROM tasks WHERE task_id=?", (task_id,)
                    ).fetchone()
            except sqlite3.Error as exc:
                LOGGER.warning(
                    "task log lookup failed: task_id=%s job_id=%s error=%s",
                    task_id,
                    job_id,
                    exc,
                )
                return
            log_file = str(row["log_file"]) if row is not None else None
        if not log_file:
            return
        payload: dict[str, object] = {
            "timestamp": _now(),
            "request_id": self._resolve_request_id(job_id, extra),
            "job_id": job_id,
            "task_id": task_id,
            "stage": stage,
            "worker_id": (extra or {}).get("worker_id"),
            "event": event,
            "result": result,
            "attempt": self._resolve_attempt(task_id, extra),
            "effective_config": self._resolve_effective_config(task_id),
            "error": self._resolve_error(event, result, extra),
        }
        if extra:
            # 统一字段已固定在上方，extra 只补充附加信息，避免覆盖标准字段。
            for key, value in extra.items():
                if key not in _TASK_LOG_STANDARD_FIELDS:
                    payload[key] = value
        try:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as exc:
            LOGGER.warning(
                "task log append failed: task_id=%s job_id=%s path=%s error=%s",
                task_id,
                job_id,
                log_file,
                exc,
            )

    def _resolve_request_id(self, job_id: str, extra: dict | None) -> str | None:
        """取本次事件涉及的 request_id：优先事件自带，其次 job metadata。

        request_id 在 enqueue 时写入 task_config["metadata"]，不新增 DB 列；
        因此从旧 job 读不到时返回 None，字段仍保留（值为 null）以固定 schema。
        """

        direct = (extra or {}).get("request_id")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT task_config_json FROM tasks WHERE job_id=? LIMIT 1", (job_id,)
            ).fetchone()
        if row is None:
            return None
        cfg = self._loads_or_empty(row["task_config_json"])
        metadata = cfg.get("metadata")
        if isinstance(metadata, dict):
            value = metadata.get("request_id")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _resolve_attempt(self, task_id: str, extra: dict | None) -> int:
        """取事件对应的 attempt：优先事件自带，其次当前任务行。"""

        direct = (extra or {}).get("attempt")
        if isinstance(direct, int) and not isinstance(direct, bool):
            return int(direct)
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT attempt FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        if row is None:
            return 0
        return int(row["attempt"] or 0)

    def _resolve_effective_config(self, task_id: str) -> dict:
        """取任务入队时固化的 effective_config（缺失时返回空 dict）。"""

        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT task_config_json FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        if row is None:
            return {}
        cfg = self._loads_or_empty(row["task_config_json"])
        effective = cfg.get("effective_config")
        return dict(effective) if isinstance(effective, dict) else {}

    def _resolve_error(self, event: str, result: str, extra: dict | None) -> str | None:
        """取错误描述：显式传入优先；失败类事件回退到 result 语义。"""

        direct = (extra or {}).get("error")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        if str(result) in {"failed", "blocked"}:
            return str(result)
        return None
    def retry_job(self, job_id: str) -> dict:
        """把 failed 的 job 拉回 queued，便于原地重跑（A.7）。

        仅允许 job.status == failed；failed/blocked 任务重置为 queued、
        attempt=0 并清空 last_error，succeeded 任务保持不变。返回受影响
        task 数。任何不满足前置条件的情况抛 ValueError，由路由映射状态码。
        """

        with self.db.tx() as conn:
            job = conn.execute(
                "SELECT job_id, status FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None:
                raise ValueError("job_not_found")
            if str(job["status"]) != "failed":
                raise ValueError("retry_not_allowed")
            _assert_replayable_dag(conn, job_id)

            rows = conn.execute(
                "SELECT task_id, stage, status, log_file FROM tasks WHERE job_id=?",
                (job_id,),
            ).fetchall()
            now = _now()
            affected = 0
            for row in rows:
                status = str(row["status"])
                if status not in {"failed", "blocked"}:
                    continue
                changed = conn.execute(
                    """
                    UPDATE tasks
                    SET status='queued', attempt=0, last_error=NULL, finished_at=NULL,
                        lease_owner=NULL, lease_expires_at=NULL, updated_at=?
                    WHERE task_id=? AND status IN ('failed', 'blocked')
                    """,
                    (now, row["task_id"]),
                ).rowcount
                if int(changed) == 0:
                    continue
                affected += 1
                self._append_task_log_event(
                    job_id=job_id,
                    task_id=str(row["task_id"]),
                    stage=str(row["stage"]),
                    event="task_retry_requested",
                    result="queued",
                    log_file=row["log_file"],
                    extra={"previous_status": status, "attempt": 0},
                )
            if affected == 0:
                raise ValueError("retry_not_allowed")

            conn.execute(
                "UPDATE jobs SET status='queued', finished_at=NULL, updated_at=? WHERE job_id=?",
                (now, job_id),
            )
            new_job_status = self._refresh_job_status(conn, job_id)
            return {
                "job_id": job_id,
                "retried_tasks": affected,
                "status": new_job_status,
            }

    def cancel_job(self, job_id: str) -> dict:
        """取消尚未开始执行的 job。

        仅当 job 为 queued 且所有任务都未开始时允许；把任务置为 cancelled
        并释放路径占用（job 随之进入终态，不再参与同路径幂等判定）。返回
        被取消的 task 数。运行中取消不在本轮范围，返回 409。整个判定与
        更新在同一 BEGIN IMMEDIATE 事务内完成，避免与并发的 claim_next 竞态。
        """

        with self.db.tx() as conn:
            job = conn.execute(
                "SELECT job_id, status FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None:
                raise ValueError("job_not_found")

            rows = conn.execute(
                "SELECT task_id, stage, status, log_file FROM tasks WHERE job_id=?",
                (job_id,),
            ).fetchall()
            statuses = [str(row["status"]) for row in rows]
            if not rows:
                raise ValueError("cancel_not_allowed")
            if any(
                status in {"claimed", "succeeded", "failed", "blocked"}
                for status in statuses
            ):
                raise ValueError("cancel_not_allowed")

            now = _now()
            cancelled = 0
            for row in rows:
                changed = conn.execute(
                    """
                    UPDATE tasks
                    SET status='cancelled', lease_owner=NULL, lease_expires_at=NULL,
                        finished_at=?, last_error='cancelled_by_user', updated_at=?
                    WHERE task_id=? AND status='queued'
                    """,
                    (now, now, row["task_id"]),
                ).rowcount
                if int(changed) == 0:
                    continue
                cancelled += 1
                self._append_task_log_event(
                    job_id=job_id,
                    task_id=str(row["task_id"]),
                    stage=str(row["stage"]),
                    event="task_cancelled",
                    result="cancelled",
                    log_file=row["log_file"],
                    extra={"previous_status": "queued"},
                )
            if cancelled == 0:
                raise ValueError("cancel_not_allowed")

            conn.execute(
                "UPDATE jobs SET finished_at=COALESCE(finished_at, ?), updated_at=? WHERE job_id=?",
                (now, now, job_id),
            )
            self._refresh_job_status(conn, job_id)
            status_row = conn.execute(
                "SELECT status FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            return {
                "job_id": job_id,
                "cancelled_tasks": cancelled,
                "status": str(status_row["status"]) if status_row else "queued",
            }

    def get_job_status(self, job_id: str) -> str | None:
        """返回 job 当前状态，不存在时返回 None。"""

        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT status FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            return str(row["status"]) if row is not None else None


def _assert_replayable_dag(conn, job_id: str) -> None:
    """校验 job 的 DAG 能被当前版本解析与归一化，否则拒绝重跑。

    历史 job 若 dag_json 缺失按默认三阶段 DAG 兼容；DAG 语义校验（批次 2）
    不在本函数职责内，这里只拒绝无法解析的形态，避免重跑放大历史错误。
    """

    row = conn.execute("SELECT dag_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if row is None:
        raise ValueError("job_not_found")
    raw = row["dag_json"]
    if not raw:
        return
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("retry_job_dag_invalid") from exc
    try:
        normalized = normalize_and_validate_dag(parsed)
    except ValueError as exc:
        raise ValueError(f"retry_job_dag_invalid:{exc}") from exc
    if not isinstance(normalized.get("stages"), list) or not normalized["stages"]:
        raise ValueError("retry_job_dag_invalid")
