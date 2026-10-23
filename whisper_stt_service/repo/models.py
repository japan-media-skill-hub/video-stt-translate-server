"""仓储层数据结构。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnqueueResult:
    """入队结果载体。"""

    job_id: str
    accepted: bool
    message: str
    queue_ahead: int
    plan_mode: str
    stages: list[str]


@dataclass
class ClaimedTask:
    """领取到的任务最小信息。"""

    task_id: str
    job_id: str
    stage: str
    # 领取时写入的 attempt 值，用于后续心跳/写回的 fencing 校验。
    attempt: int = 0


@dataclass
class TaskExecutionContext:
    """worker 执行任务时需要的上下文。"""

    task_id: str
    job_id: str
    stage: str
    video_path: str
    source_language: str
    output_ja_path: str
    output_zh_path: str
    timeout_sec: int
    attempt: int
    max_retries: int
    log_dir: str
    log_file: str
    task_config: dict
    # SPEC B.3：每阶段唯一写者。入队时写入 task_config["io"]，运行时按此执行；
    # 为空表示历史 job 未登记 io，调用方应回退到 video_path/output_*_path 规则。
    input_path: str = ""
    output_path: str = ""
