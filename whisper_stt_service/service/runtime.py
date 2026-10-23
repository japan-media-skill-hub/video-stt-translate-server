"""Worker 运行时：启动恢复、三阶段执行、进度消费与优雅停机。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import inspect
import hashlib
import json
import logging
import os
from pathlib import Path
from queue import Empty, Queue
import signal
import shutil
import stat
from threading import Event, Lock, Thread
from time import monotonic, sleep
from typing import Any, Callable
import uuid

# 执行器协作式取消的统一异常（由 executor/common 提供并重导出）。
from whisper_stt_service.executor.common import ExecutorCancelledError

from whisper_stt_service.core.config import Settings
from whisper_stt_service.executor import (
    build_stt_effective_config,
    build_stt_whisperx_effective_config,
    run_extract,
    run_stt,
    run_stt_whisperx,
    run_translate,
)
from whisper_stt_service.core.progress import ProgressStore
from whisper_stt_service.core.security import SecurityError, resolve_copy_back_dir
from whisper_stt_service.executor.atomic import atomic_write_text
from whisper_stt_service.repo.job_repository import JobRepository
from whisper_stt_service.service.metrics import (
    record_task_claimed,
    record_task_finished,
)


LOGGER = logging.getLogger(__name__)

# A.1：心跳间隔下限，避免按租约 1/3 推算时在极短租约下造成写放大。
_MIN_HEARTBEAT_INTERVAL_SEC = 10.0
# 带外心跳未生效时的兜底：距上次续租超过租约时长的该比例即视为疑似过期。
_LEASE_GUARD_RATIO = 0.8


def recover_claimed_to_queued(db, *, counts_as_attempt: bool = True) -> int:
    """兼容旧测试入口：启动时把 claimed 任务统一回退。

    返回回退条数；counts_as_attempt 与 A.5 配置语义一致。
    """

    repo = JobRepository(db)
    stats = repo.recover_claimed_to_queued(counts_as_attempt=counts_as_attempt)
    return int(stats.get("recovered", 0))


class LeaseLostError(RuntimeError):
    """租约已被回收或由新 attempt 接管：执行方必须尽快中止并放弃写回。"""


@dataclass
class WorkerState:
    """记录 worker 的当前活动状态，供队列摘要接口读取。"""

    worker_id: str
    stage: str
    task_id: str | None
    updated_at: str
class FencedLease:
    """单任务租约租期：承担心跳续租、fencing 写回与协作式取消检查。

    真实长执行（STT 数小时、translate 数千批）不会给心跳线程让路，因此
    心跳由独立守护线程按固定间隔续租，执行线程只做廉价的本地时间片检查：

    - ``check()`` 在调用安全点报告“租约是否已被回收”，执行器据此中止；
    - ``guard()`` 在写结果前做最后一次本地校验，关闭“写回窗口”内的竞态；
    - DB 层写回另行携带 worker_id/attempt 做 SQL fencing，双重保证。
    """

    def __init__(
        self,
        *,
        repo: JobRepository,
        task_id: str,
        job_id: str,
        stage: str,
        worker_id: str,
        attempt: int,
        lease_timeout_sec: int,
        heartbeat_interval_sec: float,
        on_heartbeat: Callable[["FencedLease", bool], None] | None = None,
    ) -> None:
        """记录租约标识并计算心跳间隔。"""

        self.repo = repo
        self.task_id = task_id
        self.job_id = job_id
        self.stage = stage
        self.worker_id = worker_id
        self.attempt = int(attempt)
        self.lease_timeout_sec = max(int(lease_timeout_sec), 1)
        self.heartbeat_interval_sec = max(float(heartbeat_interval_sec), 0.05)
        self.lost_reason: str | None = None
        self._on_heartbeat = on_heartbeat
        self._lost_event = Event()
        self._abort_event = Event()
        self._last_renew_monotonic = monotonic()
        # 运行时时间片校验的节流时间戳（由 WorkerRuntime 更新）。
        self._last_time_slice_check = 0.0
        self._thread: Thread | None = None
        self._thread_stop = Event()

    def start(self) -> None:
        """启动带外心跳线程（幂等）。"""

        if self._thread is not None:
            return
        self._thread = Thread(
            target=self._heartbeat_loop,
            name=f"lease-heartbeat-{self.worker_id}-{self.task_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """停止心跳线程并等待其退出。"""

        self._thread_stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        self._thread = None
    def _heartbeat_loop(self) -> None:
        """按固定间隔续租；续租被 fencing 拒绝即标记租约丢失。"""

        while not self._thread_stop.wait(self.heartbeat_interval_sec):
            try:
                ok = self.repo.heartbeat_task(
                    self.task_id,
                    worker_id=self.worker_id,
                    attempt=self.attempt,
                    lease_timeout_sec=self.lease_timeout_sec,
                )
            except Exception as exc:  # noqa: BLE001
                # 数据库瞬时错误不应被当成租约丢失，等下一轮再试；
                # 若租约真的过期，reaper 会回收，随后的写回会被 fencing 拒绝。
                LOGGER.warning(
                    "lease heartbeat failed: task_id=%s worker_id=%s attempt=%s error=%s",
                    self.task_id,
                    self.worker_id,
                    self.attempt,
                    exc,
                )
                continue
            if ok:
                self._last_renew_monotonic = monotonic()
                if self._on_heartbeat is not None:
                    self._on_heartbeat(self, True)
                continue
            self._mark_lost("lease_lost:fencing_rejected")
            if self._on_heartbeat is not None:
                self._on_heartbeat(self, False)
            return
    def _mark_lost(self, reason: str) -> None:
        """标记租约丢失（只记录首个原因，保持可观测性稳定）。"""

        if self.lost_reason is None:
            self.lost_reason = reason
            LOGGER.warning(
                "lease lost: task_id=%s job_id=%s worker_id=%s attempt=%s reason=%s",
                self.task_id,
                self.job_id,
                self.worker_id,
                self.attempt,
                reason,
            )
        self._lost_event.set()

    def check(self, site: str = "") -> None:
        """协作式取消检查：租约丢失或外部中止时抛出 LeaseLostError。

        供执行器安全点调用（extract 每行 progress、STT 每 N 段、translate
        每批）。只做本地标志与时间比较，不访问数据库，开销可忽略。
        """

        if self._abort_event.is_set():
            raise LeaseLostError(self.lost_reason or "cancelled_by_shutdown")
        if self._lost_event.is_set():
            raise LeaseLostError(self.lost_reason or "lease_lost")
        if self._thread is None and self.is_suspected_expired():
            # 无带外心跳（异常装配或测试替身）时，用租约时长的该比例兜底：
            # 超过阈值说明续租链路没有生效，主动中止而不是写回陈旧结果。
            suffix = ":" + site if site else ""
            self._mark_lost("lease_lost:no_heartbeat" + suffix)
            raise LeaseLostError(self.lost_reason or "lease_lost")

    def guard(self, site: str = "") -> None:
        """写回前的最后一道本地 fencing 校验（与 check 等价）。"""

        self.check(site)

    def request_abort(self, reason: str = "cancelled") -> None:
        """外部请求中止（drain 超时 / 运维取消）：后续安全点立即抛错。"""

        if self.lost_reason is None:
            self.lost_reason = reason
        self._abort_event.set()

    def is_suspected_expired(self) -> bool:
        """距上次续租是否已接近租约上限（兜底判定，非权威）。"""

        elapsed = monotonic() - self._last_renew_monotonic
        return elapsed > self.lease_timeout_sec * _LEASE_GUARD_RATIO

    @property
    def lost(self) -> bool:
        """租约是否已丢失或已被请求中止。"""

        return self._lost_event.is_set() or self._abort_event.is_set()

class WorkerRuntime:
    """管理 worker 线程池、进度消费与停机回滚。"""

    def __init__(
        self,
        *,
        repo: JobRepository,
        progress_store: ProgressStore,
        settings: Settings,
        config_path: Path,
        model_path: str,
        exit_callback: Callable[[], None] | None = None,
        heartbeat_interval_sec: float | None = None,
        lease_guard_enabled: bool = True,
        metrics: Any | None = None,
        checkpoint_ttl_sec: int | None = None,
    ) -> None:  # noqa: PLR0913 - 依赖较多但均为显式注入，便于测试替换
        """注入运行所需依赖。

        exit_callback 为 A.6 的注入式退出钩子（bootstrap 接 server.should_exit）；
        未注入时回退 SIGTERM。heartbeat_interval_sec 为 None 时按
        max(lease_timeout_sec / 3, 10s) 计算；测试可显式传入小值。
        metrics 为 service.metrics.MetricsState（bootstrap 注入），未注入时
        所有指标记录退化为空操作，不影响执行。checkpoint_ttl_sec 为磁盘上
        翻译检查点的保留时长（对应 [translation] progress_ttl_sec），与
        runtime.progress_ttl_sec（仅内存进度）是两套独立 TTL，不混同。
        """

        self.repo = repo
        self.progress_store = progress_store
        self.settings = settings
        self.config_path = config_path
        self.model_path = model_path
        self._exit_callback = exit_callback
        # 指标句柄：bootstrap 注入 build_metrics() 结果；测试可注入替身。
        self.metrics = metrics
        # 磁盘检查点 TTL：显式注入优先；未注入时按“不清理检查点”（保守保留）。
        self.checkpoint_ttl_sec = (
            None if checkpoint_ttl_sec is None else max(int(checkpoint_ttl_sec), 0)
        )

        lease_timeout = max(int(getattr(settings.timeouts, "lease_timeout_sec", 600)), 1)
        self.lease_timeout_sec = lease_timeout
        if heartbeat_interval_sec is not None:
            self.heartbeat_interval_sec = max(float(heartbeat_interval_sec), 0.05)
        else:
            self.heartbeat_interval_sec = max(
                lease_timeout / 3.0, _MIN_HEARTBEAT_INTERVAL_SEC
            )
        drain_timeout = getattr(settings.timeouts, "drain_timeout_sec", 300)
        self.drain_timeout_sec = max(int(drain_timeout), 1)
        self._lease_guard_enabled = lease_guard_enabled
        # 执行器安全点上的“是否被回收/接管”时间片检查节流间隔（秒）。
        self._time_slice_check_sec = 1.0
        # R3-4 租约身份：worker_id 只有 extract-w1 这类可复用文本时，API retry
        # 会把 attempt 重置为 0、debug 恢复（recovery_counts_as_attempt=false）
        # 也会保留 attempt 0，于是新执行的 (task_id, owner, attempt) 可能与旧
        # 执行完全相同（ABA），旧 publish 会被误认为合法。这里给每个 runtime
        # 启动周期生成一次性 nonce 并拼进 lease_owner（保留 stage-wN 可读前缀），
        # 让“执行实例身份”在数据库层唯一，而不只依赖暂存文件 nonce。
        self.boot_nonce = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"

        self._stop_event = Event()
        self._drain_event = Event()
        self._exit_triggered_event = Event()
        self._progress_queue: Queue[dict[str, Any]] = Queue(maxsize=4096)
        self._threads: list[Thread] = []
        self._state_lock = Lock()
        self._worker_states: dict[str, WorkerState] = {}
        self._shutdown_requested_at: str | None = None
        self._shutdown_reason: str = ""
        self._drain_deadline_monotonic: float | None = None
        self._drain_timed_out = False
        self._inflight_leases: dict[str, FencedLease] = {}
        self._startup_stats: dict[str, int] = {"recovered": 0, "failed": 0}
        self._reaper_stats: dict[str, int] = {"reclaimed": 0, "failed": 0}

    def start(self) -> int:
        """执行启动恢复并拉起所有后台线程。

        返回回退条数（兼容既有测试）；完整统计见 startup_stats()。
        """

        counts_as_attempt = bool(
            getattr(self.settings.runtime, "recovery_counts_as_attempt", True)
        )
        stats = self.repo.recover_claimed_to_queued(
            counts_as_attempt=counts_as_attempt
        )
        self._startup_stats = {
            "recovered": int(stats.get("recovered", 0)),
            "failed": int(stats.get("failed", 0)),
        }
        LOGGER.warning(
            "startup recovery: recovered=%s failed=%s counts_as_attempt=%s",
            self._startup_stats["recovered"],
            self._startup_stats["failed"],
            counts_as_attempt,
        )
        self._start_thread(self._progress_loop, name="progress-consumer")
        self._start_thread(self._cleanup_loop, name="progress-cleanup")
        self._start_thread(self._reaper_loop, name="lease-reaper")
        self._start_thread(self._shutdown_watch_loop, name="shutdown-watch")
        if self.settings.runtime.artifact_cleanup_enabled:
            self._start_thread(self._artifact_cleanup_loop, name="artifact-cleanup")

        self._spawn_stage_workers("extract", self.settings.workers.extract_workers)
        self._spawn_stage_workers("stt", self.settings.workers.stt_workers)
        self._spawn_stage_workers(
            "stt_whisperx", self.settings.workers.stt_whisperx_workers
        )
        self._spawn_stage_workers("translate", self.settings.workers.translate_workers)
        return self._startup_stats["recovered"]

    def startup_stats(self) -> dict[str, int]:
        """返回启动恢复统计：recovered（回退条数）、failed（超预算转 failed）。"""

        return dict(self._startup_stats)

    def stop(self, timeout_sec: int | None = None) -> None:
        """优雅停机：先停领取，再等待线程退出，最后回滚 claimed。

        A.6 追加：等待改为共享总 deadline（min(timeout_sec, drain_timeout_sec)），
        不再逐线程累加；超时后对残留租约 request_abort 并按冻结租约三元组回滚。

        总 deadline 为 min(timeout_sec 或 drain_timeout_sec)。此前逐线程
        join(timeout) 会累加，线程多时可能远超 drain 上限；这里改为共享
        deadline。超时后对仍在途的租约 request_abort（执行器在安全点中止且
        不发布结果），再按 (task_id, lease_owner, attempt) 精确回滚，旧 worker
        即使稍后返回，其写回也会被 fencing 拒绝。

        只回滚租约为本进程 worker 的任务，避免多实例部署时误伤其他进程；
        重启后的全量恢复由下次 start() 的 recover_claimed_to_queued 负责。

        R3-1：回退必须使用执行实例冻结的 (task_id, lease_owner, attempt) 三元组，
        不能只传 task_ids。任务被 reaper 回收并被新 worker 接管后，旧 runtime 的
        本地 map 仍可能残留旧租约；按 task_id 回滚会把新 owner 的 claimed 错误地
        改回 queued。三元组任一不匹配即跳过。
        """

        budget = self.drain_timeout_sec if timeout_sec is None else max(int(timeout_sec), 1)
        deadline = monotonic() + budget
        self._stop_event.set()
        for t in self._threads:
            remaining = deadline - monotonic()
            if remaining <= 0:
                LOGGER.warning(
                    "stop deadline reached, %s thread(s) still alive",
                    sum(1 for x in self._threads if x.is_alive()),
                )
                break
            t.join(timeout=remaining)

        # 超时残留的在途执行：先请求中止，避免其继续写产物或结果。
        with self._state_lock:
            leftovers = list(self._inflight_leases.values())
        for lease in leftovers:
            lease.request_abort("cancelled_by_shutdown")

        # 双保险：停机后把仍处于 claimed 的任务回退，确保下次可恢复。
        # 历史说明（R3-1）：此处原文来自旧实现，但旧代码按 task_id 批量回退会
        # 误撤已被新 owner/attempt 合法接管的 claimed。现在实际回滚范围仅限本
        # runtime 本地 _inflight_leases 冻结的 (task_id, lease_owner, attempt)
        # 三元组，repo 只更新仍匹配这三项的行；不匹配的新租约保持 claimed，
        # 因此不得把本注释误读为“全局撤销 claimed”。
        # 冻结三元组来自本地 _inflight_leases；repo 只回退仍匹配
        # owner/attempt 的行，已被新 worker 接管的任务保持 claimed 不变。
        rolled_back = self.repo.rollback_claimed_leases(
            self._frozen_lease_tuples(leftovers),
            reason="shutdown_rollback",
        )
        if rolled_back:
            LOGGER.warning("shutdown rollback: rolled_back=%s", rolled_back)
    def active_workers(self) -> dict[str, dict]:
        """返回当前 worker 活动视图。"""

        with self._state_lock:
            return {
                worker_id: {
                    "stage": state.stage,
                    "task_id": state.task_id,
                    "updated_at": state.updated_at,
                }
                for worker_id, state in self._worker_states.items()
            }

    def request_shutdown(self, reason: str = "manual_shutdown") -> dict[str, Any]:
        """触发 drain：不再领取新任务，并在可退出时优雅退出进程。"""

        with self._state_lock:
            if not self._drain_event.is_set():
                self._drain_event.set()
                self._shutdown_requested_at = datetime.now(timezone.utc).isoformat()
                self._drain_deadline_monotonic = (
                    monotonic() + self.drain_timeout_sec
                )
                self._shutdown_reason = (
                    reason.strip() if reason.strip() else "manual_shutdown"
                )
                LOGGER.warning(
                    "graceful shutdown requested: reason=%s drain_timeout_sec=%s",
                    self._shutdown_reason,
                    self.drain_timeout_sec,
                )
        return self.shutdown_status()

    def _lease_expired_in_db(self, lease: FencedLease) -> bool:
        """确认租约是否真的已被回收（时间片节流，避免安全点打库）。"""

        if not self._lease_guard_enabled:
            return False
        now = monotonic()
        if now - lease._last_time_slice_check < self._time_slice_check_sec:
            return False
        lease._last_time_slice_check = now
        try:
            return not self.repo.task_lease_matches(
                lease.task_id, worker_id=lease.worker_id, attempt=lease.attempt
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "lease time-slice check failed: task_id=%s worker_id=%s error=%s",
                lease.task_id,
                lease.worker_id,
                exc,
            )
            return False

    def _make_cancel_check(self, lease: FencedLease) -> Callable[[], bool]:
        """构造传给执行器的 cancel_check：返回 True 表示应尽快中止。

        每次调用只读本地事件；节流后额外做一次 DB 租约校验，用于感知
        被 reaper 回收 / 被新 worker 接管 / 他人 retry 导致的协作式取消。
        """

        def _cancel_check() -> bool:
            """执行器安全点回调：租约丢失或被外部中止时返回 True。"""

            if lease.lost:
                return True
            if self._drain_timed_out:
                lease.request_abort("cancelled_by_drain_timeout")
                return True
            if self._lease_expired_in_db(lease):
                lease._mark_lost("lease_lost:time_slice_check")
                return True
            return False

        return _cancel_check

    def _local_task_ids(self) -> list[str]:
        """返回本进程当前持有租约的 task_id 列表。"""

        with self._state_lock:
            return list(self._inflight_leases.keys())

    def inflight_task_ids(self) -> list[str]:
        """返回本进程在途（已领取未结算）任务 id，供停机状态接口读取。"""

        with self._state_lock:
            return sorted(self._inflight_leases.keys())
    def shutdown_status(self) -> dict[str, Any]:
        """返回 drain 停机状态快照。

        can_exit 只取决于“drain 已请求 且 本进程无在途任务 或 drain 已超时”；
        全局 claimed_count 仅作为告警指标（多实例场景），不再作为唯一判据。
        """

        with self._state_lock:
            inflight = sorted(self._inflight_leases.keys())
            requested_at = self._shutdown_requested_at
            shutdown_reason = self._shutdown_reason
            deadline_monotonic = self._drain_deadline_monotonic
            timed_out = self._drain_timed_out

        drain_requested = self._drain_event.is_set()
        remaining = (
            max(deadline_monotonic - monotonic(), 0.0)
            if deadline_monotonic is not None
            else None
        )
        deadline_at = (
            (datetime.now(timezone.utc) + timedelta(seconds=remaining)).isoformat()
            if remaining is not None
            else None
        )
        if drain_requested and not timed_out and remaining is not None:
            timed_out = remaining <= 0
        claimed_count = self.repo.count_claimed_tasks()
        blocked_count = self.repo.count_jobs_with_blocked_tasks()
        can_exit = bool(drain_requested and (not inflight or timed_out))
        return {
            "drain_requested": drain_requested,
            "shutdown_requested_at": requested_at,
            "shutdown_reason": shutdown_reason,
            "drain_deadline_at": deadline_at,
            "drain_remaining_sec": (
                round(remaining, 3) if remaining is not None else None
            ),
            "drain_timed_out": bool(timed_out),
            "claimed_count": claimed_count,
            "inflight": inflight,
            "inflight_count": len(inflight),
            "inflight_task_ids": inflight,
            "blocked": blocked_count,
            "can_exit": can_exit,
            "exit_triggered": self._exit_triggered_event.is_set(),
        }
    def _start_thread(self, target, *, name: str, args: tuple = ()) -> None:
        """统一创建守护线程并保存句柄。"""

        t = Thread(target=target, name=name, args=args, daemon=True)
        t.start()
        self._threads.append(t)

    def _worker_identity(self, stage: str, slot: int) -> str:
        """构造本 runtime 启动周期内唯一的 worker_id（保留可读 stage-wN 前缀）。

        R3-4：lease_owner 是写回 fencing 的一半身份。仅用 extract-w1 这类
        可复用文本会让“旧执行实例 + retry 后同 attempt 的新执行实例”产生完全
        相同的 (task_id, owner, attempt)，形成 ABA；因此把 boot_nonce 拼在
        可读前缀之后，数据库里的 owner 每次进程启动都不同。
        """

        return f"{stage}-w{int(slot)}-{self.boot_nonce}"

    def _spawn_stage_workers(self, stage: str, count: int) -> None:
        """按阶段配置数量启动 worker 线程。"""

        for idx in range(max(count, 0)):
            worker_id = self._worker_identity(stage, idx + 1)
            self._start_thread(
                self._worker_loop,
                name=f"worker-{worker_id}",
                args=(stage, worker_id),
            )

    def _all_worker_ids(self) -> list[str]:
        """按阶段配置枚举本进程全部 worker_id（用于停机回滚）。"""

        ids: list[str] = []
        for stage, count in (
            ("extract", self.settings.workers.extract_workers),
            ("stt", self.settings.workers.stt_workers),
            ("stt_whisperx", self.settings.workers.stt_whisperx_workers),
            ("translate", self.settings.workers.translate_workers),
        ):
            for idx in range(max(int(count), 0)):
                ids.append(self._worker_identity(stage, idx + 1))
        return ids

    def _set_worker_state(
        self, worker_id: str, stage: str, task_id: str | None
    ) -> None:
        """更新 worker 当前状态。"""

        now = datetime.now(timezone.utc).isoformat()
        with self._state_lock:
            self._worker_states[worker_id] = WorkerState(
                worker_id=worker_id,
                stage=stage,
                task_id=task_id,
                updated_at=now,
            )
    def _progress_loop(self) -> None:
        """消费 progress_queue 并更新 ProgressStore。"""

        while not self._stop_event.is_set() or not self._progress_queue.empty():
            try:
                event = self._progress_queue.get(timeout=0.5)
            except Empty:
                continue
            task_id = str(event.get("task_id", "")).strip()
            if not task_id:
                continue
            percent = float(event.get("percent", 0.0))
            message = str(event.get("message", ""))
            worker_id = str(event.get("worker_id", ""))
            self.progress_store.update(
                task_id,
                percent=percent,
                message=message,
                worker_id=worker_id,
            )
            if percent >= 100.0 or message.endswith("_done"):
                self.progress_store.mark_done(task_id)

    def _cleanup_loop(self) -> None:
        """定期清理内存中过期完成进度。"""

        interval = max(self.settings.workers.scheduler_interval_sec, 5)
        while not self._stop_event.wait(timeout=interval):
            self.progress_store.cleanup()

    def _reaper_loop(self) -> None:
        """A.2 过期回收循环：每 lease_timeout_sec/3（下限 1s）扫描一次。"""

        interval = max(self.lease_timeout_sec / 3.0, 1.0)
        while not self._stop_event.wait(timeout=interval):
            if self._drain_event.is_set():
                # drain 期间在途任务仍会续租；但不再主动回收，交由停机路径处理。
                continue
            try:
                stats = self.repo.reclaim_expired_leases()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("lease reaper tick failed: error=%s", exc)
                continue
            self._reaper_stats["reclaimed"] += int(stats.get("reclaimed", 0))
            self._reaper_stats["failed"] += int(stats.get("failed", 0))
            if stats.get("reclaimed") or stats.get("failed"):
                LOGGER.warning(
                    "lease reaper tick: reclaimed=%s failed=%s",
                    stats.get("reclaimed"),
                    stats.get("failed"),
                )

    def reaper_stats(self) -> dict[str, int]:
        """返回租约回收累计统计（reclaimed/failed）。"""

        return dict(self._reaper_stats)
    def _shutdown_watch_loop(self) -> None:
        """监听 drain 状态并在可退出时向当前进程发送 SIGTERM。

        历史说明：原实现直接向自身发送 SIGTERM。A.6 起改为注入回调
        （exit_callback / server.should_exit），仅在未注入时回退 SIGTERM；
        同时新增 drain 超时后的强制回滚与退出。
        """

        while not self._stop_event.wait(timeout=1.0):
            if not self._drain_event.is_set():
                continue
            status = self.shutdown_status()
            if status.get("drain_timed_out") and not self._drain_timed_out:
                # 超时：中止本地在途执行并强制回滚其 claimed（fencing 保护）。
                self._force_drain_timeout()
                status = self.shutdown_status()
            if status["can_exit"]:
                if self._drain_timed_out:
                    LOGGER.warning(
                        "graceful shutdown drain timeout reached: inflight=%s claimed=%s"
                        " drain_timeout_sec=%s",
                        status["inflight_count"],
                        status["claimed_count"],
                        self.drain_timeout_sec,
                    )
                else:
                    LOGGER.warning(
                        "graceful shutdown drain complete: claimed=%s inflight=%s",
                        status["claimed_count"],
                        status["inflight_count"],
                    )
                self._trigger_process_exit()
                return

    def _frozen_lease_tuples(
        self, leases: list[FencedLease]
    ) -> list[tuple[str, str, int]]:
        """把本地在途租约冻结为 (task_id, lease_owner, attempt) 三元组。

        R3-1：停机回滚只允许撤销“本执行实例仍持有”的领取；冻结三元组在回滚
        事务内逐条匹配，任何已被回收/接管的任务都不会被旧 runtime 改状态。
        """

        frozen: list[tuple[str, str, int]] = []
        seen: set[tuple[str, str, int]] = set()
        for lease in leases:
            item = (lease.task_id, lease.worker_id, int(lease.attempt))
            if item in seen:
                continue
            seen.add(item)
            frozen.append(item)
        return frozen

    def _force_drain_timeout(self) -> None:
        """drain 超时：中止本进程在途执行并强制回滚其 claimed（fencing 保护）。"""

        with self._state_lock:
            already = self._drain_timed_out
            self._drain_timed_out = True
            leases = list(self._inflight_leases.values())
        if already:
            return
        LOGGER.warning(
            "drain timeout, force rollback local claimed: inflight=%s drain_timeout_sec=%s",
            len(leases),
            self.drain_timeout_sec,
        )
        for lease in leases:
            lease.request_abort("cancelled_by_drain_timeout")
        # R3-1：tuple-fenced 回滚，绝不按 task_id 撤掉已被新 owner 接管的 claimed。
        rolled_back = self.repo.rollback_claimed_leases(
            self._frozen_lease_tuples(leases),
            reason="drain_timeout_rollback",
        )
        LOGGER.warning(
            "drain timeout rollback done: rolled_back=%s", rolled_back
        )

    def _trigger_process_exit(self) -> None:
        """仅触发一次优雅退出信号，避免重复发送。

        A.6 起优先调用注入的退出回调（bootstrap 接 server.should_exit），
        仅当未注入回调（或回调失败）时回退到向当前进程发送 SIGTERM。
        """

        if self._exit_triggered_event.is_set():
            return
        self._exit_triggered_event.set()
        if self._exit_callback is not None:
            try:
                self._exit_callback()
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "exit callback failed, fallback to SIGTERM: error=%s", exc
                )
                os.kill(os.getpid(), signal.SIGTERM)
            return
        os.kill(os.getpid(), signal.SIGTERM)
    def _worker_loop(self, stage: str, worker_id: str) -> None:
        """单个 worker 主循环：领取任务并执行阶段函数。

        领取成功后交给 _execute_claimed 建租约执行；drain 期间只等待不领取。
        """

        poll = max(self.settings.workers.poll_interval_sec, 1)
        lease_timeout = self.lease_timeout_sec
        while not self._stop_event.is_set():
            self._set_worker_state(worker_id, stage, None)
            if self._drain_event.is_set():
                sleep(poll)
                continue
            claimed = self.repo.claim_next(
                stage=stage, worker_id=worker_id, lease_timeout_sec=lease_timeout
            )
            if claimed is None:
                sleep(poll)
                continue
            self._execute_claimed(claimed, worker_id, stage)

    def _execute_claimed(self, claimed, worker_id: str, stage: str) -> None:
        """执行单个已领取任务：建租约、跑执行器、按 fencing 写回结果。

        负责租约生命周期（start/stop）与完成指标（claimed/finished/duration）。
        """

        task_id = claimed.task_id
        self._set_worker_state(worker_id, stage, task_id)
        # G：领取即计数，便于与完成数对比观察卡死/回收。
        record_task_claimed(self.metrics, stage)
        started = monotonic()
        ctx = self.repo.get_task_execution_context(task_id)
        if ctx is None:
            # 上下文缺失说明任务已被删除/归档；此时直接放弃，不做任何写回。
            LOGGER.warning(
                "task context not found, abandon claim: task_id=%s worker_id=%s",
                task_id,
                worker_id,
            )
            self.repo.rollback_claimed_task(
                task_id,
                reason="task_context_not_found",
                worker_id=worker_id,
                attempt=claimed.attempt,
            )
            return

        lease = FencedLease(
            repo=self.repo,
            task_id=task_id,
            job_id=ctx.job_id,
            stage=ctx.stage,
            worker_id=worker_id,
            attempt=claimed.attempt,
            lease_timeout_sec=self.lease_timeout_sec,
            heartbeat_interval_sec=self.heartbeat_interval_sec,
            on_heartbeat=self._on_lease_heartbeat,
        )
        with self._state_lock:
            self._inflight_leases[task_id] = lease
        lease.start()
        try:
            self._run_claimed_task(ctx, lease, worker_id, stage)
        finally:
            lease.stop()
            # R3-2：finally 只能用“对象身份”删除本执行实例登记的 inflight lease。
            # 若执行期间任务已被回收并交给新执行实例，map[task_id] 已是新 lease；
            # 无条件 pop 会把新 lease 从停机/取消视图中抹掉（新 lease 保留回归）。
            released = self._release_inflight_lease(lease)
            if not released:
                LOGGER.warning(
                    "skip releasing inflight lease owned by newer execution: "
                    "task_id=%s worker_id=%s attempt=%s",
                    task_id,
                    worker_id,
                    lease.attempt,
                )
            # worker 本地状态同理：只有仍指向本任务时才清空，避免覆盖新执行实例。
            self._clear_worker_state(worker_id, stage, task_id)
            # G：按最终状态记录完成数；被中止/未写回的执行记为 aborted，
            # 耗时直方图只统计真正的执行时长。
            final_status = self._final_task_status(task_id)
            if final_status != "claimed":
                record_task_finished(
                    self.metrics,
                    stage=stage,
                    result=final_status,
                    duration_sec=monotonic() - started,
                )

    def _release_inflight_lease(self, lease: FencedLease) -> bool:
        """仅移除仍指向本执行实例的 inflight 租约（R3-2 finally 竞态）。

        旧执行实例的 finally 可能晚于新执行实例登记租约：此时
        _inflight_leases[task_id] 已是新 lease，无条件 pop 会让停机/取消
        视图丢失新任务（drain 可能误判可退出）。这里用对象身份比较，只有同一
        实例才删除，返回是否实际删除。
        """

        with self._state_lock:
            if self._inflight_leases.get(lease.task_id) is lease:
                del self._inflight_leases[lease.task_id]
                return True
        return False

    def _clear_worker_state(
        self, worker_id: str, stage: str, task_id: str
    ) -> None:
        """仅当 worker 状态仍指向本执行实例时清空（避免覆盖新执行实例状态）。"""

        with self._state_lock:
            current = self._worker_states.get(worker_id)
            if current is not None and current.task_id != task_id:
                return
            self._worker_states[worker_id] = WorkerState(
                worker_id=worker_id,
                stage=stage,
                task_id=None,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )

    def _final_task_status(self, task_id: str) -> str:
        """读取任务当前状态，用于完成指标；读库失败返回 unknown。"""

        try:
            with self.repo.db.connect() as conn:
                row = conn.execute(
                    "SELECT status FROM tasks WHERE task_id=?", (task_id,)
                ).fetchone()
        except Exception as exc:  # noqa: BLE001 - 指标上报不能影响执行
            LOGGER.warning(
                "task status lookup for metrics failed: task_id=%s error=%s",
                task_id,
                exc,
            )
            return "unknown"
        return str(row["status"]) if row is not None else "unknown"


    def _on_lease_heartbeat(self, lease: FencedLease, ok: bool) -> None:
        """心跳回调：续租被拒绝时写入 lease_lost 事件，便于排障。"""

        if ok:
            return
        self._write_task_log(
            log_file=self._log_file_for(lease.task_id),
            job_id=lease.job_id,
            task_id=lease.task_id,
            stage=lease.stage,
            worker_id=lease.worker_id,
            event="lease_lost",
            result="aborting",
            attempt=lease.attempt,
            error=lease.lost_reason or "lease_lost",
        )

    def _log_file_for(self, task_id: str) -> str:
        """按 task_id 读取日志文件路径；缺失时返回空串（跳过写日志）。"""

        try:
            with self.repo.db.connect() as conn:
                row = conn.execute(
                    "SELECT log_file FROM tasks WHERE task_id=?", (task_id,)
                ).fetchone()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "task log file lookup failed: task_id=%s error=%s", task_id, exc
            )
            return ""
        return str(row["log_file"]) if row is not None else ""

    def _is_shutdown_rollback(self) -> bool:
        """判断是否处于“停机/超时导致的执行中止”场景。"""

        if self._stop_event.is_set():
            return True
        with self._state_lock:
            return bool(self._drain_timed_out)
    def _run_claimed_task(self, ctx, lease: FencedLease, worker_id: str, stage: str) -> None:
        """任务执行体：校验配置、调用执行器、按 fencing 写回成功或失败。

        执行器通过 cancel_check 感知失租/停机；失败与陈旧写回都只经 fencing
        路径落库，绝不无条件覆盖任务状态。

        产物级 fencing（A.3）：执行器只写本实例私有暂存，正式路径仅在 promotion
        时、经 repo 的 SQL 锁确认租约仍有效后 os.replace；不论成功、失败、取消、
        失租还是停机，finally 都会清掉本实例登记的暂存文件。
        """

        task_cfg = ctx.task_config.get("effective_config")
        if not isinstance(task_cfg, dict):
            self._mark_failed_with_log(ctx, lease, "invalid_task_config")
            return

        self._write_task_log(
            log_file=ctx.log_file,
            job_id=ctx.job_id,
            task_id=ctx.task_id,
            stage=ctx.stage,
            worker_id=worker_id,
            event="task_started",
            result="running",
            ctx=ctx,
            attempt=ctx.attempt,
            # G：快照必须来自该 task 的 effective_config（job 覆盖生效），
            # 而不是全局 settings。
            effective_config=self._effective_snapshot(
                ctx=ctx, stage=stage, task_cfg=task_cfg
            ),
            # 附加排障信息（io 路径/video_path/max_retries）只能走 extra，
            # 不允许覆盖 11 个固定字段。
            extra=self._build_task_started_extra(ctx=ctx, stage=stage, task_cfg=task_cfg),
        )

        cancel_check = self._make_cancel_check(lease)
        # A.3 产物级 fencing：执行器只写本次执行实例的私有暂存文件，正式路径
        # 仅在 promotion 时、经 repo 的 SQL 锁确认租约仍有效后 os.replace。
        staging = self._build_staging(ctx, stage, lease)
        staging.staged_path.parent.mkdir(parents=True, exist_ok=True)
        # 登记本实例允许清理的暂存路径（绝不盲扫同目录下其他 worker 的文件）。
        for owned in staging.iter_staged_paths():
            staging.note_owned(owned)
        effective_config: dict[str, Any] | None = None
        execution_error: Exception | None = None
        published: list[str] = []
        try:
            lease.check("before_execute")
            effective_config = self._invoke_stage_executor(
                ctx=ctx, stage=stage, task_cfg=task_cfg,
                staging=staging, cancel_check=cancel_check, worker_id=worker_id,
                lease=lease,
            )
            lease.guard("before_result")
        except LeaseLostError as exc:
            execution_error = exc
        except ExecutorCancelledError as exc:
            execution_error = exc
        except Exception as exc:  # noqa: BLE001
            execution_error = exc
        finally:
            pass

        # 执行结束：先处理失败/中止（不发布），再在 SQL 锁下 promotion。
        copy_back_plan: CopyBackPlan | None = None
        try:
            if isinstance(execution_error, LeaseLostError):
                self._abandon_lost_lease(ctx, lease, execution_error)
                return
            if isinstance(execution_error, ExecutorCancelledError):
                self._handle_cancelled(ctx, lease, execution_error)
                return
            if execution_error is not None:
                self._mark_failed_with_log(ctx, lease, str(execution_error))
                return

            if self._is_shutdown_rollback():
                # 停机/超时中止：执行器可能已返回，但不写结论、不发布，交给回滚路径。
                LOGGER.warning(
                    "shutdown rollback in progress, discard result: task_id=%s job_id=%s",
                    ctx.task_id,
                    ctx.job_id,
                )
                return

            # R3-6：translate 的 defer_copy_back=True 必须由 runtime 完成真实
            # 回写。慢拷贝（可能跨文件系统）在 DB 锁外进行且用 dir_fd 锚定目标
            # 目录；最终 rename 通过同一 fd 在租约事务内完成（短提交点），失败按
            # fenced failure/retry 收敛。
            if stage == "translate":
                copy_back_plan = self._prepare_copy_back(
                    ctx, lease, staging, task_cfg
                )

            # 产物 promotion + 状态写回：优先使用 repo 的 publish_task_artifacts
            # （同一 BEGIN IMMEDIATE 事务内先校验 claimed/owner/attempt，再执行
            # prepublish(copyback dirfd rename)，然后 os.replace 已暂存产物并置
            # succeeded）；旧版 repo 没有该接口时退化为 with_task_lease 校验 +
            # 本地 replace + 分两步写库。
            published, mark_status = self._publish_stage_outputs(
                ctx,
                lease,
                stage,
                staging,
                copy_back_plan=copy_back_plan,
            )
            if mark_status == "fenced_stale_result":
                self._record_fenced_stale(ctx, lease, "succeeded", None)
                return
            if mark_status != "succeeded":
                # 写回失败（例如 repo 侧状态已被他人改变）：不发布、不覆盖正式产物。
                self._record_fenced_stale(ctx, lease, "succeeded", mark_status)
                return
        except LeaseLostError as exc:
            # copy_back 前置检查发现失租：放弃回写，不按失败重试（任务已被接管）。
            self._abandon_lost_lease(ctx, lease, exc)
            return
        except Exception as exc:  # noqa: BLE001 - promotion/copyback I/O 必须收敛
            # R3-2：promotion/copyback 的 OSError 若逃出本方法，会杀死该阶段唯一
            # 的 worker 线程（reaper 之后也没有消费者）。这里统一转为 fenced
            # failure/retry 并记录完整日志，绝不把异常抛给 worker 主循环。
            self._handle_publish_failure(ctx, lease, stage, exc)
            return
        finally:
            # copy_back 计划：未成功提交时通过 dir_fd 删除自己的临时文件（即使
            # 目标目录已被移动/替换），然后关闭 fd；成功提交后 discard 为空操作。
            if copy_back_plan is not None:
                copy_back_plan.close()
            # 统一清理：成功时保留已发布路径，失败/取消/失租/停机全部丢弃暂存。
            self._cleanup_staging(staging, keep=published)

        self.progress_store.update(
            ctx.task_id,
            percent=100.0,
            message=f"{stage}_done",
            worker_id=worker_id,
        )
        self.progress_store.mark_done(ctx.task_id)
        self._write_task_log(
            log_file=ctx.log_file,
            job_id=ctx.job_id,
            task_id=ctx.task_id,
            stage=ctx.stage,
            worker_id=worker_id,
            event="task_finished",
            result="succeeded",
            ctx=ctx,
            attempt=ctx.attempt,
            effective_config=effective_config,
            extra={"artifacts_published": len(published)},
        )

    def _invoke_stage_executor(
        self,
        *,
        ctx,
        stage: str,
        task_cfg: dict,
        staging: ArtifactStaging,
        cancel_check: Callable[[], bool],
        worker_id: str,
        lease: FencedLease,
    ) -> dict[str, Any] | None:
        """按阶段调用执行器（输出一律写本次实例的暂存路径）。

        返回 STT/WhisperX 的生效配置快照；extract 返回执行器自报快照或 None。
        lease 同时下发给 translate 的 checkpoint 发布回调：fenced 写必须绑定
        本执行实例的 owner/attempt，不能运行时再按 task_id 查当前租约。
        """

        if stage == "extract":
            result = run_extract(
                Path(ctx.video_path),
                staging.staged_path,
                int(task_cfg.get("timeout_sec", ctx.timeout_sec)),
                progress_queue=self._progress_queue,
                task_id=ctx.task_id,
                worker_id=worker_id,
                cancel_check=cancel_check,
            )
            if isinstance(result, dict) and isinstance(
                result.get("effective_config"), dict
            ):
                return dict(result["effective_config"])
            return None
        if stage == "stt":
            # B.3：ASR 输出写暂存；输入始终是已正式化的 extract wav。
            return self._execute_stt(
                ctx, task_cfg, cancel_check, output_path=staging.staged_path
            )
        if stage == "stt_whisperx":
            return self._execute_stt_whisperx(
                ctx, task_cfg, cancel_check, output_path=staging.staged_path
            )
        if stage == "translate":
            self._execute_translate(
                ctx, task_cfg, cancel_check, staging=staging, lease=lease
            )
            return None
        raise RuntimeError(f"unknown stage: {stage}")


    def _build_staging(self, ctx, stage: str, lease: FencedLease) -> ArtifactStaging:
        """为本次 attempt 构造产物暂存与正式化计划。

        正式目标按阶段取 io 段（extract=audio wav、ASR=asr srt、translate=zh srt），
        旧 job 没有 io 段时沿用历史路径规则；暂存文件与正式目标同目录，保证
        `os.replace` 在同一文件系统内原子完成。
        """

        if stage == "extract":
            final_path = self._resolve_extract_output(ctx)
        elif stage in {"stt", "stt_whisperx"}:
            final_path = self._resolve_asr_output(ctx)
        elif stage == "translate":
            final_path = self._resolve_translate_output(ctx)
        else:
            raise RuntimeError(f"unknown stage: {stage}")
        # work_dir 只用于让暂存文件名全局唯一（worker 之间也不会撞名）。
        work_dir = Path(ctx.log_dir or ctx.job_id)
        return ArtifactStaging(
            final_path=final_path,
            work_dir=work_dir,
            attempt=lease.attempt,
            worker_id=lease.worker_id,
        )

    def _publish_stage_outputs(
        self,
        ctx,
        lease: FencedLease,
        stage: str,
        staging: ArtifactStaging,
        *,
        copy_back_plan: CopyBackPlan | None = None,
    ) -> tuple[list[str], str]:
        """在 SQL 锁下把暂存产物正式化并写回成功状态。

        返回 (已发布文件列表, mark 状态)。mark 状态为 `succeeded` 表示任务已被
        本次写回置为成功；`fenced_stale_result` 表示租约/attempt 已失效，未发布。

        copy_back_plan 是本执行实例在慢拷贝阶段构造的 dir_fd 锚定计划。
        plan.commit 作为 repo.publish_task_artifacts 的 prepublish 回调，在
        **同一租约事务内、主产物 rename 之前**执行短 rename；因此：
        - 旧 attempt 的 copy_back 无法覆盖新 attempt 的回写结果（R3-4）；
        - 慢拷贝期间目标目录被替换为符号链接时，commit 会因 fd 真实路径变化
          拒绝提交（绝不往 outside 写正式文件），整个事务回滚、任务按失败重试；
        - 若 repo 不支持 prepublish，宁可失败也不退化为锁外/未锚定回写。
        """

        artifacts = self._collect_staged_artifacts(staging)
        prepublish = copy_back_plan.commit if copy_back_plan is not None else None
        copy_targets = (
            [
                str(copy_back_plan.dir_path / target_name)
                for _temp_name, target_name in copy_back_plan.pairs
            ]
            if copy_back_plan is not None
            else []
        )
        publish_artifacts = getattr(self.repo, "publish_task_artifacts", None)
        if callable(publish_artifacts):
            # repo 侧：同一 BEGIN IMMEDIATE 内先 fencing 校验，再 prepublish
            # （copyback 的 dirfd 锚定短 rename），然后 os.replace 主产物并置
            # succeeded；返回 False 表示租约/attempt 已失效且未触碰文件。
            publish_kwargs: dict[str, Any] = {
                "worker_id": lease.worker_id,
                "attempt": lease.attempt,
                "artifacts": artifacts,
                "request_id": self._request_id(ctx),
            }
            if prepublish is not None:
                if not _supports_keyword(publish_artifacts, "prepublish"):
                    raise RuntimeError("copy_back_requires_prepublish_support")
                publish_kwargs["prepublish"] = prepublish
            published = publish_artifacts(ctx.task_id, **publish_kwargs)
            if published:
                return [str(a[1]) for a in artifacts] + copy_targets, "succeeded"
            return [], "fenced_stale_result"

        # 兼容旧 repo：先在同一写锁事务内校验租约，再落盘，最后写回状态。
        with_task_lease = getattr(self.repo, "with_task_lease", None)
        if prepublish is not None and not callable(with_task_lease):
            # 没有租约锁接口时，copy_back 无法与成功写回同事务；拒绝而不是
            # 假装成功后再无条件回写（R3-6）。
            raise RuntimeError("copy_back_requires_fenced_publish")

        def _locked_action() -> None:
            """锁内动作：先提交 copyback（如启用），再正式化主产物。"""

            if prepublish is not None:
                prepublish()
            self._os_replace_all(artifacts)

        published: list[str] = []
        if callable(with_task_lease):
            allowed = with_task_lease(
                ctx.task_id,
                worker_id=lease.worker_id,
                attempt=lease.attempt,
                action=_locked_action,
            )
            if not allowed:
                return [], "fenced_stale_result"
            published = [a[1] for a in artifacts] + copy_targets

        marked = self.repo.mark_task_succeeded(
            ctx.task_id, worker_id=lease.worker_id, attempt=lease.attempt
        )
        if not marked:
            return [], "fenced_stale_result"
        if not published and artifacts:
            # 无 SQL 锁接口的降级路径：写回成功后本地替换（仍有极小窗口，
            # 仅用于尚未提供 publish helper 的旧 repo）。copy_back 不会走到
            # 这里：上面已对 copy_back_plan + 无锁接口直接抛错。
            self._os_replace_all(artifacts)
            published = [a[1] for a in artifacts]
        return published, "succeeded"

    def _collect_staged_artifacts(
        self, staging: ArtifactStaging
    ) -> list[tuple[Path, Path]]:
        """收集本次 attempt 实际产生的暂存文件及其正式目标路径。"""

        pairs: list[tuple[Path, Path]] = []
        # 顺序即 rename 顺序：最关键的主产物（字幕/wav）放在最后，
        # 多文件 rename 中途失败时任务不置成功，由重试统一覆盖收敛。
        if staging.staged_checkpoint_path.exists():
            pairs.append(
                (
                    staging.staged_checkpoint_path,
                    staging.final_path.with_name(
                        staging.final_path.name + ".progress.json"
                    ),
                )
            )
        # 执行器可能随产物写一个 `<name>.published` 完成标记；它同样属于本次
        # attempt 的私有暂存内容，必须在同一 promotion 内一起正式化，否则正式
        # 目录会缺少发布凭证。
        published_marker = Path(f"{staging.staged_path}.published")
        if published_marker.exists():
            pairs.append(
                (published_marker, Path(f"{staging.final_path}.published"))
            )
        if staging.staged_path.exists():
            pairs.append((staging.staged_path, staging.final_path))
        return pairs

    def _os_replace_all(self, artifacts: list[tuple[Path, Path]]) -> bool:
        """逐个 os.replace 暂存文件到正式路径（调用方必须已持有租约校验锁）。"""

        for staged, final in artifacts:
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, final)
        return True

    def _cleanup_staging(self, staging: ArtifactStaging, *, keep: list[str]) -> None:
        """清理本实例的暂存残留；已发布（keep）或未登记的路径一律不动。

        统一在 `finally` 中调用，覆盖成功/失败/取消/失租/停机全部路径。
        只处理 `owned_paths()` 中由本实例登记的路径（暂存名含实例 nonce），
        因此不会误删其他 worker 正在写入的暂存。
        """

        keep_set = {str(Path(p)) for p in keep}
        for path in staging.owned_paths():
            if str(path) in keep_set or not path.exists():
                continue
            try:
                path.unlink()
            except OSError as exc:  # noqa: BLE001 - 清理失败不影响结果
                LOGGER.warning(
                    "staging cleanup failed: path=%s error=%s", path, exc
                )

    def _mark_failed_with_log(self, ctx, lease: FencedLease, error: str) -> None:
        """按 fencing 写回失败：被拒绝时只记 fenced_stale_result，不改任务状态。"""

        status = self.repo.mark_task_failed(
            ctx.task_id,
            error,
            worker_id=lease.worker_id,
            attempt=lease.attempt,
        )
        if status == "fenced_stale_result":
            self._record_fenced_stale(ctx, lease, "failed", error)
            return
        self.progress_store.update(
            ctx.task_id,
            percent=0.0,
            message=f"{ctx.stage}_failed",
            worker_id=lease.worker_id,
        )
        self._write_task_log(
            log_file=ctx.log_file,
            job_id=ctx.job_id,
            task_id=ctx.task_id,
            stage=ctx.stage,
            worker_id=lease.worker_id,
            event="task_failed",
            result=status,
            ctx=ctx,
            # 失败后 attempt 已在库内自增，日志记录新值便于与状态对齐。
            attempt=ctx.attempt + 1,
            effective_config=ctx.task_config.get("effective_config")
            if isinstance(ctx.task_config.get("effective_config"), dict)
            else None,
            error=error,
        )

    def _record_fenced_stale(
        self, ctx, lease: FencedLease, attempted_result: str, error: str | None
    ) -> None:
        """记录被 fencing 拒绝的陈旧写回（不改任务状态、不刷新 job）。"""

        LOGGER.warning(
            "fenced stale result rejected: task_id=%s job_id=%s worker_id=%s attempt=%s result=%s",
            ctx.task_id,
            ctx.job_id,
            lease.worker_id,
            lease.attempt,
            attempted_result,
        )
        self._write_task_log(
            log_file=ctx.log_file,
            job_id=ctx.job_id,
            task_id=ctx.task_id,
            stage=ctx.stage,
            worker_id=lease.worker_id,
            event="fenced_stale_result",
            result=attempted_result,
            ctx=ctx,
            attempt=lease.attempt,
            error=error or lease.lost_reason or "lease_reclaimed_by_other_worker",
        )
    def _abandon_lost_lease(self, ctx, lease: FencedLease, exc: Exception) -> None:
        """租约丢失：放弃写回，只记录事件；停机/超时场景交给回滚路径。"""

        reason = lease.lost_reason or str(exc) or "lease_lost"
        LOGGER.warning(
            "abort task due to lost lease: task_id=%s job_id=%s reason=%s",
            ctx.task_id,
            ctx.job_id,
            reason,
        )
        event = "shutdown_aborted" if self._is_shutdown_rollback() else "lease_lost"
        self._write_task_log(
            log_file=ctx.log_file,
            job_id=ctx.job_id,
            task_id=ctx.task_id,
            stage=ctx.stage,
            worker_id=lease.worker_id,
            event=event,
            result="aborted",
            ctx=ctx,
            attempt=lease.attempt,
            error=reason,
        )

    def _handle_cancelled(self, ctx, lease: FencedLease, exc: Exception) -> None:
        """执行器协作式取消：停机/超时放弃写回；运行中被取消则按失败重试。"""

        if self._is_shutdown_rollback():
            self._abandon_lost_lease(ctx, lease, exc)
            return
        self._mark_failed_with_log(ctx, lease, str(exc) or "cancelled")

    def _prepare_copy_back(
        self,
        ctx,
        lease: FencedLease,
        staging: ArtifactStaging,
        task_cfg: dict,
    ) -> CopyBackPlan:
        """执行期重验 copy_back 目标，并构造 fd 锚定的私有临时文件回写计划。

        SPEC C.2/H 与 R3-6：translate 使用 defer_copy_back=True，执行器不再回写；
        真正的 sink 由 runtime 在 promotion 前完成。安全与原子性要求：

        - **执行期**重新解析并校验目标目录（白名单 roots / 符号链接逃逸）；
        - 以 O_DIRECTORY|O_NOFOLLOW 打开目标目录 fd，并用该 fd 创建临时文件：
          慢拷贝（可能跨文件系统）期间即使路径被替换成符号链接，写入仍锚定在
          打开时的真实目录，不会逃逸；
        - 慢拷贝在 DB 事务外完成，逐文件 flush+fsync；最终 rename 由
          CopyBackPlan.commit 在同一租约事务内用 dir_fd 执行（同文件系统，
          且短小），绝不先 mark succeeded 再无条件回写；
        - 任何失败都以异常上抛，由调用方转为 fenced failure/retry；已复制的
          临时文件通过同一个 dir_fd 清理（目录被移动也能删掉自己的文件）。

        返回 CopyBackPlan（调用方必须在 finally 中 close，成功提交后 close 为空操作）。
        """

        lease.check("before_copy_back")
        # 优先使用入队时固化的 io.copy_back_dir（含 job_config 覆盖与 canonical
        # 规范化）；旧 job 没有 io 段时才回退 effective_config，再回退历史默认。
        io_copy_back = self._io_path(ctx, "copy_back_dir")
        configured = (
            str(io_copy_back)
            if io_copy_back is not None
            else str(task_cfg.get("copy_back") or "").strip()
        )
        if not configured:
            configured = "__video_dir__"
        session = getattr(self.repo, "security", None)
        try:
            target_dir = resolve_copy_back_dir(
                configured,
                video_path=ctx.video_path,
                session=session,
                strict_parent=True,
            )
        except SecurityError as exc:
            # 统一成稳定错误码，便于 task.log 与排障；不吞原始安全原因。
            raise RuntimeError(f"copy_back_invalid:{exc.code}") from exc

        flags = os.O_RDONLY
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            resolved_dir = target_dir.resolve(strict=True)
            dir_fd = os.open(target_dir, flags)
        except OSError as exc:
            raise RuntimeError(
                f"copy_back_target_unavailable:{target_dir}"
            ) from exc

        plan = CopyBackPlan(dir_fd=dir_fd, dir_path=target_dir, pairs=[])
        try:
            opened_path = _read_fd_path(dir_fd)
            if (
                target_dir.is_symlink()
                or resolved_dir != target_dir
                or (opened_path is not None and opened_path != target_dir)
            ):
                # 打开前/打开时目录已被替换为符号链接：拒绝回写。
                raise RuntimeError(f"copy_back_target_symlink:{target_dir}")

            ja_source = self._resolve_translate_input(ctx)
            sources: list[tuple[Path, str]] = [
                (ja_source, ja_source.name),
                (staging.staged_path, staging.final_path.name),
            ]
            for source, target_name in sources:
                if not source.is_file():
                    raise RuntimeError(f"copy_back_source_missing:{source}")
                try:
                    existing = os.stat(
                        target_name, dir_fd=dir_fd, follow_symlinks=False
                    )
                except FileNotFoundError:
                    existing = None
                if existing is not None and stat.S_ISLNK(existing.st_mode):
                    # 目标文件是符号链接：拒绝写入，避免借链接逃逸到白名单之外。
                    raise RuntimeError(
                        f"copy_back_target_symlink:{target_dir / target_name}"
                    )
                try:
                    if source.resolve() == (target_dir / target_name).resolve(
                        strict=False
                    ):
                        # 目标已是同一文件（例如 copy_back 指向 artifact 目录）。
                        continue
                except OSError:
                    # resolve 失败时继续走复制路径；最终 rename 仍在租约事务内。
                    pass
                temp_name = (
                    f".{target_name}.copyback-{staging.staging_token}"
                )
                raw_fd = -1
                try:
                    raw_fd = os.open(
                        temp_name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=dir_fd,
                    )
                    with os.fdopen(raw_fd, "wb") as dst_handle:
                        raw_fd = -1
                        with source.open("rb") as src_handle:
                            shutil.copyfileobj(
                                src_handle, dst_handle, length=1024 * 1024
                            )
                        dst_handle.flush()
                        os.fsync(dst_handle.fileno())
                except BaseException:
                    if raw_fd >= 0:
                        try:
                            os.close(raw_fd)
                        except OSError:
                            pass
                    try:
                        os.unlink(temp_name, dir_fd=dir_fd)
                    except FileNotFoundError:
                        pass
                    except OSError as cleanup_exc:
                        LOGGER.warning(
                            "copy_back temp cleanup failed: dir=%s name=%s error=%s",
                            target_dir,
                            temp_name,
                            cleanup_exc,
                        )
                    raise
                plan.pairs.append((temp_name, target_name))
            return plan
        except BaseException:
            # 准备阶段失败：通过 dir_fd 删除已复制的临时文件并关闭 fd，再上抛。
            plan.close()
            raise

    def _handle_publish_failure(
        self, ctx, lease: FencedLease, stage: str, exc: Exception
    ) -> None:
        """把 promotion/copyback 异常收敛为 fenced failure/retry，保证 worker 存活。

        R3-2：真实 worker loop 不会捕获 _run_claimed_task 的异常；若 OSError
        直接冒泡，该阶段唯一 worker 线程会退出，reaper 再 queued 也没有消费者。
        这里做两件事：

        1. 记录完整异常（含 task/job/worker/attempt）便于排障；
        2. 调用 _mark_failed_with_log 走既有 fencing 失败路径（仍在重试预算内
           则回 queued 等下一次领取，超预算则 failed + 级联 blocked）。

        写回本身失败（例如 DB 不可用）只记录日志、不再抛给 worker 主循环；
        任务仍是 claimed，由租约过期后的 reaper 收敛。
        """

        error = f"publish_failed:{type(exc).__name__}:{exc}"
        LOGGER.exception(
            "stage output promotion/copy_back failed: task_id=%s job_id=%s stage=%s "
            "worker_id=%s attempt=%s error=%s",
            ctx.task_id,
            ctx.job_id,
            stage,
            lease.worker_id,
            lease.attempt,
            exc,
        )
        try:
            self._mark_failed_with_log(ctx, lease, error)
        except Exception as mark_exc:  # noqa: BLE001 - 不能再让 worker 线程退出
            LOGGER.error(
                "mark failed after publish error also failed: task_id=%s worker_id=%s "
                "attempt=%s error=%r",
                ctx.task_id,
                lease.worker_id,
                lease.attempt,
                mark_exc,
            )

    def _resolve_extract_output(self, ctx) -> Path:
        """extract 输出 wav 路径（B.3 的 wav 契约）。

        优先使用 enqueue 写入的 `task_config["io"]["output_path"]`
        （`artifacts/<job>/audio/<stem>.16k.wav`）；旧 job 没有 io 段时回退
        到既有规则（与 ja.srt 同目录的 `<stem>.wav`）。服务模式绝不让 ASR
        直接读原视频。
        """

        io_path = self._io_path(ctx, "output_path")
        if io_path is not None:
            return io_path
        return Path(ctx.output_ja_path).with_name(
            f"{Path(ctx.video_path).stem}.wav"
        )

    def _io_path(self, ctx, key: str) -> Path | None:
        """读取 task_config.io 中的路径字段，缺失/为空时返回 None。"""

        io_cfg = ctx.task_config.get("io")
        if not isinstance(io_cfg, dict):
            return None
        raw = str(io_cfg.get(key) or "").strip()
        return Path(raw) if raw else None

    def _resolve_asr_input(self, ctx) -> Path:
        """ASR 输入路径（D1：ASR 只吃 extract 产出的 wav）。

        优先使用 io.input_path；旧 job 没有 io 段时按“旧 wav 规则”回退到
        `<ja.srt 同目录>/<stem>.wav`。即使该 wav 不存在，也不会回退原视频：
        执行器以 require_audio_artifact=True 判定为 missing_audio_artifact 失败。
        """

        io_path = self._io_path(ctx, "input_path")
        if io_path is not None:
            return io_path
        return Path(ctx.output_ja_path).with_name(
            f"{Path(ctx.video_path).stem}.wav"
        )

    def _resolve_asr_output(self, ctx) -> Path:
        """ASR 输出 ja.srt 路径：优先 io.output_path，旧 job 用 jobs 表字段。"""

        return self._io_path(ctx, "output_path") or Path(ctx.output_ja_path)

    def _resolve_translate_input(self, ctx) -> Path:
        """translate 输入 ja.srt：优先 io.input_path，旧 job 用 jobs 表字段。"""

        return self._io_path(ctx, "input_path") or Path(ctx.output_ja_path)

    def _resolve_translate_output(self, ctx) -> Path:
        """translate 输出 zh.srt：优先 io.output_path，旧 job 用 jobs 表字段。"""

        return self._io_path(ctx, "output_path") or Path(ctx.output_zh_path)
    def _execute_stt(
        self,
        ctx,
        task_cfg: dict,
        cancel_check: Callable[[], bool],
        *,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        """执行 STT 阶段并返回生效配置快照。

        output_path 为本次 attempt 的暂存输出路径；省略时直接用正式路径
        （仅供不经 staging 的旧调用）。输入始终读已正式化的 extract wav。
        """

        kwargs: dict[str, Any] = {
            "input_path": self._resolve_asr_input(ctx),
            "output_path": output_path or self._resolve_asr_output(ctx),
            "language": ctx.source_language,
            "timeout_sec": int(task_cfg.get("timeout_sec", ctx.timeout_sec)),
            "model": self.model_path,
            "device": str(task_cfg.get("device", self.settings.stt.device)),
            "compute_type": str(
                task_cfg.get("compute_type", self.settings.stt.compute_type)
            ),
            "batch_size": int(task_cfg.get("batch_size", self.settings.stt.batch_size)),
            "beam_size": int(task_cfg.get("beam_size", self.settings.stt.beam_size)),
            "best_of": int(task_cfg.get("best_of", self.settings.stt.best_of)),
            "patience": float(task_cfg.get("patience", self.settings.stt.patience)),
            "condition_on_previous_text": bool(
                task_cfg.get(
                    "condition_on_previous_text",
                    self.settings.stt.condition_on_previous_text,
                )
            ),
            "vad_filter": bool(task_cfg.get("vad_filter", self.settings.stt.vad_filter)),
            "vad_threshold": float(
                task_cfg.get("vad_threshold", self.settings.stt.vad_threshold)
            ),
            "vad_min_speech_duration_ms": int(
                task_cfg.get(
                    "vad_min_speech_duration_ms",
                    self.settings.stt.vad_min_speech_duration_ms,
                )
            ),
            "vad_max_speech_duration_s": float(
                task_cfg.get(
                    "vad_max_speech_duration_s",
                    self.settings.stt.vad_max_speech_duration_s,
                )
            ),
            "vad_min_silence_duration_ms": int(
                task_cfg.get(
                    "vad_min_silence_duration_ms",
                    self.settings.stt.vad_min_silence_duration_ms,
                )
            ),
            "vad_speech_pad_ms": int(
                task_cfg.get("vad_speech_pad_ms", self.settings.stt.vad_speech_pad_ms)
            ),
            "no_speech_threshold": float(
                task_cfg.get(
                    "no_speech_threshold",
                    self.settings.stt.no_speech_threshold,
                )
            ),
            "compression_ratio_threshold": float(
                task_cfg.get(
                    "compression_ratio_threshold",
                    self.settings.stt.compression_ratio_threshold,
                )
            ),
            "log_prob_threshold": float(
                task_cfg.get("log_prob_threshold", self.settings.stt.log_prob_threshold)
            ),
            "hallucination_silence_threshold": float(
                task_cfg.get(
                    "hallucination_silence_threshold",
                    self.settings.stt.hallucination_silence_threshold,
                )
            ),
            "initial_prompt": str(
                task_cfg.get("initial_prompt", self.settings.stt.initial_prompt)
            ),
            "hotwords": str(task_cfg.get("hotwords", self.settings.stt.hotwords)),
            "progress_queue": self._progress_queue,
            "task_id": ctx.task_id,
            "worker_id": self._current_worker_id(ctx.task_id),
            # D1：ASR 只吃 wav；缺失/空文件必须失败（missing_audio_artifact），
            # 不允许静默回退到原视频。
            "require_audio_artifact": True,
        }
        return _call_executor(run_stt, kwargs, cancel_check)
    def _execute_stt_whisperx(
        self,
        ctx,
        task_cfg: dict,
        cancel_check: Callable[[], bool],
        *,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        """执行 WhisperX STT 阶段并返回生效配置快照。

        output_path 语义同 `_execute_stt`：默认暂存路径，promotion 后才成为正式产物。
        """

        kwargs: dict[str, Any] = {
            "input_path": self._resolve_asr_input(ctx),
            "output_path": output_path or self._resolve_asr_output(ctx),
            "language": ctx.source_language,
            "timeout_sec": int(task_cfg.get("timeout_sec", ctx.timeout_sec)),
            "model": str(
                task_cfg.get("model", self.settings.stt_whisperx.model)
            ),
            "device": str(
                task_cfg.get("device", self.settings.stt_whisperx.device)
            ),
            "compute_type": str(
                task_cfg.get("compute_type", self.settings.stt_whisperx.compute_type)
            ),
            "batch_size": int(
                task_cfg.get("batch_size", self.settings.stt_whisperx.batch_size)
            ),
            "vad_config_path": str(
                task_cfg.get(
                    "vad_config_path",
                    str(self.settings.stt_whisperx.vad_config_path),
                )
            ),
            "align_model_root": str(
                task_cfg.get(
                    "align_model_root",
                    str(self.settings.stt_whisperx.align_model_root),
                )
            ),
            "align_enabled": bool(
                task_cfg.get(
                    "align_enabled",
                    self.settings.stt_whisperx.align_enabled,
                )
            ),
            "vad_backend": str(
                task_cfg.get("vad_backend", self.settings.stt_whisperx.vad_backend)
            ),
            "vad_onset": float(
                task_cfg.get("vad_onset", self.settings.stt_whisperx.vad_onset)
            ),
            "vad_offset": float(
                task_cfg.get("vad_offset", self.settings.stt_whisperx.vad_offset)
            ),
            "local_files_only": bool(
                task_cfg.get(
                    "local_files_only",
                    self.settings.stt_whisperx.local_files_only,
                )
            ),
            "progress_queue": self._progress_queue,
            "task_id": ctx.task_id,
            "worker_id": self._current_worker_id(ctx.task_id),
            # D1：同 run_stt，wav 缺失或为空即失败（missing_audio_artifact）。
            "require_audio_artifact": True,
        }
        return _call_executor(run_stt_whisperx, kwargs, cancel_check)
    def _execute_translate(
        self,
        ctx,
        task_cfg: dict,
        cancel_check: Callable[[], bool],
        *,
        staging: ArtifactStaging | None = None,
        lease: FencedLease | None = None,
    ) -> None:
        """执行 translate 阶段；job_config 覆盖与 cancel_check 一并下发。

        staging 非空时：字幕写暂存、检查点写暂存检查点，并传
        defer_copy_back=True 让执行器不在 promotion 前回写视频目录；
        checkpoint_publish 绑定 repo 的租约锁与本执行实例冻结的 lease，
        保证只有持锁 attempt 能落检查点（R3-3：旧 callback 不得借新身份）。
        """

        final_output = self._resolve_translate_output(ctx)
        output_path = staging.staged_path if staging is not None else final_output
        # 真实 run_translate 的形参名是 input_ja_srt/output_zh_srt（位置参数语义）；
        # 这里必须按真实签名传参，否则 service 模式 translate 会因 unexpected
        # keyword 在拿到任何产物前失败（fake 执行器用 **kwargs 掩盖过该问题）。
        kwargs: dict[str, Any] = {
            "input_ja_srt": self._resolve_translate_input(ctx),
            "output_zh_srt": output_path,
            "config_path": self.config_path,
            "timeout_sec": int(task_cfg.get("timeout_sec", ctx.timeout_sec)),
            "input_video_path": Path(ctx.video_path),
            "copy_back": str(
                task_cfg.get("copy_back", "")
            ).strip() or None,
            "chunk_minutes": int(task_cfg.get("chunk_minutes", 30)),
            "retry": int(task_cfg.get("retry", 4)),
            "progress_queue": self._progress_queue,
            "task_id": ctx.task_id,
            "job_id": ctx.job_id,
            "worker_id": self._current_worker_id(ctx.task_id),
            # job_config 承载 translate 段覆盖（max_chars/checkpoint/parallel 等）。
            "job_config": task_cfg,
        }
        if staging is not None:
            # 暂存阶段：字幕写暂存、copy_back 延后到 promotion 之后。
            # 检查点用 **canonical 正式路径**（C.4）：它与 stage 输出路径无关，
            # 因此重跑/换 stage 输出仍能复用已完成批次、0 请求续跑；写入本身
            # 必须经 repo 租约锁（checkpoint_publish），旧 attempt 写不进去。
            kwargs["checkpoint_path"] = staging.final_path.with_name(
                staging.final_path.name + ".progress.json"
            )
            kwargs["defer_copy_back"] = True
            kwargs["checkpoint_publish"] = self._make_checkpoint_publisher(
                ctx, staging, lease
            )
        _call_executor(run_translate, kwargs, cancel_check)

    def _make_checkpoint_publisher(
        self, ctx, staging: ArtifactStaging, lease: FencedLease | None
    ):
        """构造检查点发布回调：在 repo 租约锁内原子写正式检查点。

        repo 提供 `with_task_lease` 时，写盘与租约（claimed/owner/attempt）校验
        处于同一 BEGIN IMMEDIATE 事务内；否则退化为直接写暂存检查点路径（由
        promotion 阶段统一正式化），绝不绕过 promotion 直接写正式检查点。

        R3-3/R3-4：回调在构造时**冻结本执行实例**的 lease 身份（worker_id /
        attempt / staging nonce），不再从 _inflight_leases[task_id] 动态取值。
        动态取值会把旧回调升级成新租约的身份：旧 callback 在任务被接管后仍可
        用新 owner/attempt 通过 fencing，覆盖新执行实例的 canonical 检查点。
        冻结后旧回调只能用旧身份，SQL fencing 必然拒绝。

        目标路径按**绝对路径**规范化到 canonical 正式检查点（不能只比较
        basename：不同目录同名文件仍会绕过 canonical 路径的跨 attempt 复用语义）。
        """

        final_checkpoint = staging.final_path.with_name(
            staging.final_path.name + ".progress.json"
        )
        final_checkpoint_abs = Path(os.path.abspath(os.fspath(final_checkpoint)))
        if lease is not None and lease.task_id != ctx.task_id:
            raise RuntimeError("checkpoint publisher lease/task mismatch")
        frozen_worker_id = (
            lease.worker_id if lease is not None else staging.worker_id
        )
        frozen_attempt = lease.attempt if lease is not None else staging.attempt

        def _publish(path: Path, text: str) -> bool:
            """在租约锁内把检查点原子写入正式路径；租约已失效返回 False 且不落盘。"""

            # 规范化到 canonical 正式检查点（绝对路径比较），避免执行器传入
            # stage 相关路径导致跨 stage 复用时读不到已完成批次（C.4）。
            target_abs = Path(os.path.abspath(os.fspath(path)))
            canonical = (
                final_checkpoint_abs
                if target_abs != final_checkpoint_abs
                else target_abs
            )
            with_task_lease = getattr(self.repo, "with_task_lease", None)
            if not callable(with_task_lease):
                # 无租约锁接口（旧 repo）：只写暂存检查点，promotion 时再正式化。
                staged = staging.staged_checkpoint_path
                staged.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_text(staged, text)
                return True

            def _write() -> None:
                """锁内写盘动作（调用时已确认任务仍为本 attempt 持有）。"""

                canonical.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_text(canonical, text)

            try:
                return bool(
                    with_task_lease(
                        ctx.task_id,
                        worker_id=frozen_worker_id,
                        attempt=frozen_attempt,
                        action=_write,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - 检查点失败不影响主流程
                LOGGER.warning(
                    "checkpoint publish failed: task_id=%s worker_id=%s attempt=%s error=%s",
                    ctx.task_id,
                    frozen_worker_id,
                    frozen_attempt,
                    exc,
                )
                return False

        return _publish

    def _current_worker_id(self, task_id: str) -> str:
        """根据在途租约反查执行该任务的 worker_id（缺失时返回空串）。"""

        with self._state_lock:
            lease = self._inflight_leases.get(task_id)
        return lease.worker_id if lease is not None else ""
    def _artifact_cleanup_loop(self) -> None:
        """按小时调度清理已完成 job 的 artifact 目录。"""

        interval = max(self.settings.runtime.artifact_cleanup_interval_sec, 60)
        while not self._stop_event.wait(timeout=interval):
            self._run_artifact_cleanup_once()

    def _run_artifact_cleanup_once(self) -> None:
        """执行单轮 artifact 清理并输出摘要日志。"""

        artifact_root = Path(self.repo.artifact_root)
        cleanup_statuses = tuple(self.settings.runtime.artifact_cleanup_statuses)
        scanned = 0
        eligible = 0
        deleted = 0
        failed = 0
        retained_count = 0

        try:
            if not artifact_root.exists():
                LOGGER.info(
                    "artifact cleanup tick: root_missing root=%s statuses=%s",
                    artifact_root,
                    cleanup_statuses,
                )
                return
            if not artifact_root.is_dir():
                LOGGER.warning(
                    "artifact cleanup tick: root_not_dir root=%s statuses=%s",
                    artifact_root,
                    cleanup_statuses,
                )
                return

            root_resolved = artifact_root.resolve()
            candidate_dirs: dict[str, Path] = {}
            for item in artifact_root.iterdir():
                scanned += 1
                if item.is_symlink() or not item.is_dir():
                    continue
                try:
                    resolved = item.resolve()
                except Exception:
                    continue
                if resolved.parent != root_resolved:
                    continue
                candidate_dirs[item.name] = item

            if not candidate_dirs:
                LOGGER.info(
                    "artifact cleanup tick: scanned=%s eligible=%s deleted=%s failed=%s root=%s statuses=%s",
                    scanned,
                    eligible,
                    deleted,
                    failed,
                    artifact_root,
                    cleanup_statuses,
                )
                return

            cleanable_job_ids = self.repo.list_job_ids_by_status(
                job_ids=list(candidate_dirs.keys()),
                statuses=cleanup_statuses,
            )
            eligible = len(cleanable_job_ids)

            for job_id in cleanable_job_ids:
                target = candidate_dirs.get(job_id)
                if target is None:
                    continue
                # 二次校验，避免目录遍历/软链绕过导致越界删除。
                try:
                    target_resolved = target.resolve()
                except Exception:
                    failed += 1
                    continue
                if target.is_symlink() or target_resolved.parent != root_resolved:
                    failed += 1
                    continue
                if not self.repo.is_job_completed_for_cleanup(
                    job_id=job_id,
                    statuses=cleanup_statuses,
                ):
                    continue
                # C.4：成功 job 的整目录清理必须等检查点 TTL 到期，避免误删
                # 尚未过期、可供重试复用的 translation 检查点。
                retained = self._checkpoint_retention_deadline(job_id, target)
                if retained is not None:
                    retained_count += 1
                    LOGGER.info(
                        "artifact cleanup deferred: job_id=%s checkpoint_retained_until=%s",
                        job_id,
                        retained.isoformat(),
                    )
                    continue
                try:
                    shutil.rmtree(target)
                    deleted += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    LOGGER.warning(
                        "artifact cleanup remove failed: job_id=%s path=%s error=%s",
                        job_id,
                        target,
                        exc,
                    )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("artifact cleanup tick failed: error=%s", exc)
            return

        LOGGER.info(
            "artifact cleanup tick: scanned=%s eligible=%s deleted=%s failed=%s "
            "checkpoint_retained=%s root=%s statuses=%s",
            scanned,
            eligible,
            deleted,
            failed,
            retained_count,
            artifact_root,
            cleanup_statuses,
        )

    def _job_finished_at(self, job_id: str) -> datetime | None:
        """读取 job 的 finished_at（成功完成时刻）。

        优先用岗位表里的 finished_at；历史数据缺失时回退到该 job 任务里
        最晚的 finished_at；再缺失则返回 None（由调用方改用目录 mtime）。
        """

        try:
            with self.repo.db.connect() as conn:
                row = conn.execute(
                    "SELECT finished_at FROM jobs WHERE job_id=?", (job_id,)
                ).fetchone()
                if row is not None and row["finished_at"]:
                    parsed = _parse_iso_timestamp(row["finished_at"])
                    if parsed is not None:
                        return parsed
                task_row = conn.execute(
                    "SELECT MAX(finished_at) AS ts FROM tasks WHERE job_id=?",
                    (job_id,),
                ).fetchone()
        except Exception as exc:  # noqa: BLE001 - 清理判定失败时保守保留
            LOGGER.warning(
                "job finished_at lookup failed: job_id=%s error=%s", job_id, exc
            )
            return None
        if task_row is not None and task_row["ts"]:
            return _parse_iso_timestamp(str(task_row["ts"]))
        return None

    def _checkpoint_ttl_sec(self, job_id: str) -> int | None:
        """取该 job translate 任务的生效检查点 TTL（job 覆盖 > 配置注入）。

        读取 task_config.effective_config 中的 progress_ttl_sec；缺失时使用
        运行时注入的 checkpoint_ttl_sec（对应 [translation] progress_ttl_sec）。
        返回 0 表示“立即清理”，None 表示未配置（保守保留）。
        """

        try:
            with self.repo.db.connect() as conn:
                rows = conn.execute(
                    "SELECT task_config_json FROM tasks WHERE job_id=? AND stage='translate'",
                    (job_id,),
                ).fetchall()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "translate task config lookup failed: job_id=%s error=%s", job_id, exc
            )
            rows = []
        for row in rows:
            raw = row["task_config_json"]
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if not isinstance(parsed, dict):
                continue
            effective = parsed.get("effective_config")
            if not isinstance(effective, dict):
                continue
            value = effective.get("progress_ttl_sec")
            if value is None:
                continue
            try:
                return max(int(value), 0)
            except (TypeError, ValueError):
                continue
        return self.checkpoint_ttl_sec

    def _find_checkpoint_files(self, job_dir: Path) -> list[Path]:
        """在 job 目录内查找翻译检查点文件（`*.progress.json`）。"""

        try:
            return [p for p in job_dir.rglob("*.progress.json") if p.is_file()]
        except OSError as exc:
            LOGGER.warning(
                "checkpoint scan failed: job_dir=%s error=%s", job_dir, exc
            )
            return []

    def _checkpoint_retention_deadline(
        self, job_id: str, job_dir: Path
    ) -> datetime | None:
        """返回该 job 仍需保留检查点的截止时刻；无需保留时返回 None。

        判定依据：job 成功完成时刻（或检查点 mtime） + TTL。任一检查点未过期
        即延迟整个目录删除（SPEC 允许的最简实现）。TTL=0 表示立即可清。
        """

        checkpoints = self._find_checkpoint_files(job_dir)
        if not checkpoints:
            return None
        ttl_sec = self._checkpoint_ttl_sec(job_id)
        if ttl_sec is None:
            # 未配置检查点 TTL：保守保留，交由运维确认后再清理。
            return datetime.now(timezone.utc) + timedelta(days=36500)
        if ttl_sec <= 0:
            return None
        now = datetime.now(timezone.utc)
        latest: datetime | None = None
        for path in checkpoints:
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue
            finished = self._job_finished_at(job_id) or mtime
            deadline = max(finished, mtime) + timedelta(seconds=ttl_sec)
            if latest is None or deadline > latest:
                latest = deadline
        if latest is None:
            return None
        return latest if latest > now else None

    def _build_task_started_extra(
        self, *, ctx, stage: str, task_cfg: dict | None = None
    ) -> dict[str, Any]:
        """构造 task_started 事件扩展字段，STT 阶段补充生效参数快照。

        G：快照必须反映**该 task** 的生效配置（job_config 覆盖优先），因此
        优先使用 ctx.task_config["effective_config"]；缺失时回退到全局
        settings 推导（兼容历史 task）。stt_whisperx 同样补充快照；有 io 段时
        一并记录输入/输出路径便于排障。
        """

        effective = task_cfg if isinstance(task_cfg, dict) else {}
        payload: dict[str, Any] = {
            "attempt": ctx.attempt,
            "max_retries": ctx.max_retries,
            "video_path": ctx.video_path,
        }
        io_cfg = ctx.task_config.get("io")
        if isinstance(io_cfg, dict):
            payload["io"] = io_cfg
        if stage == "stt":
            merged_stt = {
                "model": self.model_path,
                "language": ctx.source_language,
                "timeout_sec": int(effective.get("timeout_sec", ctx.timeout_sec)),
                "max_retries": int(effective.get("max_retries", ctx.max_retries)),
                "device": str(effective.get("device", self.settings.stt.device)),
                "compute_type": str(
                    effective.get("compute_type", self.settings.stt.compute_type)
                ),
                "batch_size": int(
                    effective.get("batch_size", self.settings.stt.batch_size)
                ),
                "beam_size": int(effective.get("beam_size", self.settings.stt.beam_size)),
                "best_of": int(effective.get("best_of", self.settings.stt.best_of)),
                "patience": float(effective.get("patience", self.settings.stt.patience)),
                "condition_on_previous_text": bool(
                    effective.get(
                        "condition_on_previous_text",
                        self.settings.stt.condition_on_previous_text,
                    )
                ),
                "vad_filter": bool(
                    effective.get("vad_filter", self.settings.stt.vad_filter)
                ),
                "vad_threshold": float(
                    effective.get("vad_threshold", self.settings.stt.vad_threshold)
                ),
                "vad_min_speech_duration_ms": int(
                    effective.get(
                        "vad_min_speech_duration_ms",
                        self.settings.stt.vad_min_speech_duration_ms,
                    )
                ),
                "vad_max_speech_duration_s": float(
                    effective.get(
                        "vad_max_speech_duration_s",
                        self.settings.stt.vad_max_speech_duration_s,
                    )
                ),
                "vad_min_silence_duration_ms": int(
                    effective.get(
                        "vad_min_silence_duration_ms",
                        self.settings.stt.vad_min_silence_duration_ms,
                    )
                ),
                "vad_speech_pad_ms": int(
                    effective.get(
                        "vad_speech_pad_ms", self.settings.stt.vad_speech_pad_ms
                    )
                ),
                "no_speech_threshold": float(
                    effective.get(
                        "no_speech_threshold", self.settings.stt.no_speech_threshold
                    )
                ),
                "compression_ratio_threshold": float(
                    effective.get(
                        "compression_ratio_threshold",
                        self.settings.stt.compression_ratio_threshold,
                    )
                ),
                "log_prob_threshold": float(
                    effective.get(
                        "log_prob_threshold", self.settings.stt.log_prob_threshold
                    )
                ),
                "hallucination_silence_threshold": float(
                    effective.get(
                        "hallucination_silence_threshold",
                        self.settings.stt.hallucination_silence_threshold,
                    )
                ),
                "initial_prompt": str(
                    effective.get("initial_prompt", self.settings.stt.initial_prompt)
                ),
                "hotwords": str(effective.get("hotwords", self.settings.stt.hotwords)),
            }
            payload["effective_config"] = build_stt_effective_config(**merged_stt)
        if stage == "stt_whisperx":
            merged_whisperx = {
                "model": str(
                    effective.get("model", self.settings.stt_whisperx.model)
                ),
                "language": ctx.source_language,
                "timeout_sec": int(effective.get("timeout_sec", ctx.timeout_sec)),
                "max_retries": int(effective.get("max_retries", ctx.max_retries)),
                "device": str(
                    effective.get("device", self.settings.stt_whisperx.device)
                ),
                "compute_type": str(
                    effective.get(
                        "compute_type", self.settings.stt_whisperx.compute_type
                    )
                ),
                "batch_size": int(
                    effective.get("batch_size", self.settings.stt_whisperx.batch_size)
                ),
                "vad_config_path": str(
                    effective.get(
                        "vad_config_path",
                        str(self.settings.stt_whisperx.vad_config_path),
                    )
                ),
                "align_model_root": str(
                    effective.get(
                        "align_model_root",
                        str(self.settings.stt_whisperx.align_model_root),
                    )
                ),
                "align_enabled": bool(
                    effective.get(
                        "align_enabled", self.settings.stt_whisperx.align_enabled
                    )
                ),
                "vad_backend": str(
                    effective.get("vad_backend", self.settings.stt_whisperx.vad_backend)
                ),
                "vad_onset": float(
                    effective.get("vad_onset", self.settings.stt_whisperx.vad_onset)
                ),
                "vad_offset": float(
                    effective.get("vad_offset", self.settings.stt_whisperx.vad_offset)
                ),
                "local_files_only": bool(
                    effective.get(
                        "local_files_only", self.settings.stt_whisperx.local_files_only
                    )
                ),
            }
            payload["effective_config"] = build_stt_whisperx_effective_config(
                **merged_whisperx
            )
        return payload

    def _effective_snapshot(
        self, *, ctx, stage: str, task_cfg: dict | None
    ) -> dict | None:
        """返回 task_started 日志里的生效配置快照（无快照的阶段返回 None）。

        extract 阶段在 task 启动时尚不知道执行器产出的快照，因此为 None；
        STT/WhisperX 使用与 task_started extra 相同的推导结果，保证日志的
        `effective_config` 字段与该 task 实际生效配置一致。
        """

        if stage not in {"stt", "stt_whisperx"}:
            return None
        extra = self._build_task_started_extra(
            ctx=ctx, stage=stage, task_cfg=task_cfg
        )
        snapshot = extra.get("effective_config")
        return snapshot if isinstance(snapshot, dict) else None

    def _write_task_log(
        self,
        *,
        log_file: str,
        job_id: str,
        task_id: str,
        stage: str,
        worker_id: str,
        event: str,
        result: str,
        extra: dict[str, Any] | None = None,
        ctx: Any | None = None,
        attempt: int | None = None,
        effective_config: dict | None = None,
        error: str | None = None,
    ) -> None:
        """按 task.log jsonl 写业务事件，便于追踪 worker 与 job 行为。

        G 追加：字段统一为 SPEC 规定的 11 项
        `timestamp,request_id,job_id,task_id,stage,worker_id,event,result,attempt,
        effective_config,error`。这 11 项为固定字段，extra 只能追加其它附加
        信息，不能覆盖它们（被覆盖的键会被忽略）。request_id 取
        `task_config["metadata"]["request_id"]`，无值写 null。
        """

        base_fields = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": self._request_id(ctx),
            "job_id": job_id,
            "task_id": task_id,
            "stage": stage,
            "worker_id": worker_id,
            "event": event,
            "result": result,
            "attempt": attempt,
            "effective_config": effective_config,
            "error": error,
        }
        payload: dict[str, Any] = dict(base_fields)
        if extra:
            # 固定字段禁止被 extra 覆盖，避免不同调用点写出不一致的日志结构。
            payload.update(
                {k: v for k, v in extra.items() if k not in base_fields}
            )

        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _request_id(self, ctx: Any | None) -> str | None:
        """读取任务的 request_id（task_config.metadata.request_id），无值返回 None。"""

        if ctx is None:
            return None
        task_config = getattr(ctx, "task_config", None)
        if not isinstance(task_config, dict):
            return None
        metadata = task_config.get("metadata")
        if not isinstance(metadata, dict):
            return None
        value = str(metadata.get("request_id") or "").strip()
        return value or None


def _call_executor(
    func: Callable[..., Any],
    kwargs: dict[str, Any],
    cancel_check: Callable[[], bool] | None,
) -> Any:
    """调用执行器并按需注入 cancel_check（兼容尚未支持该参数的签名）。

    批次 1 的生命周期必须能在执行器未升级时继续工作，因此这里做一次签名
    探测：支持 cancel_check 则注入；不支持则由运行时的时间片做兜底，绝不
    因为参数不兼容而把可运行的任务判失败。
    """

    if cancel_check is not None and _supports_keyword(func, "cancel_check"):
        kwargs = dict(kwargs, cancel_check=cancel_check)
    return func(**kwargs)


class ArtifactStaging:
    """单 attempt 的产物暂存目录与正式化（promotion）编排（SPEC A.3 产物级 fencing）。

    执行器只写本 attempt 私有的暂存文件，正式产物路径仅在 `promote()` 时通过
    `os.replace` 落地，且落地前必须在 repo 的 BEGIN IMMEDIATE 事务内确认
    “任务仍由本 worker 的同一 attempt 持有”（`with_task_lease`）或调用
    `publish_task_artifacts`（publish 与 mark succeeded 同一事务）。

    这样即使旧 worker 在执行器内部已产生文件，它也只能写自己的私有暂存，
    无法覆盖新 attempt 的正式产物；校验失败时暂存被整体丢弃。
    """

    def __init__(
        self,
        *,
        final_path: Path,
        work_dir: Path,
        attempt: int,
        worker_id: str,
    ) -> None:
        """记录正式目标与私有暂存路径（与正式目标同文件系统，保证 os.replace 原子）。"""

        self.final_path = Path(final_path)
        self.work_dir = Path(work_dir)
        self.attempt = int(attempt)
        self.worker_id = worker_id
        # 暂存标记必须是“执行实例”唯一，而不能只依赖 attempt 递增：
        # `recovery_counts_as_attempt=false`（调试合法）重启时，同一 task 可能
        # 以相同 attempt 交给不同 worker，若共用暂存路径，旧进程仍能篡改新进程
        # 的暂存并被新租约合法发布。因此额外固定 worker 的安全哈希与实例 nonce：
        # - worker 哈希让不同 worker 天然隔离；
        # - nonce 在 __init__ 时生成一次（不能放在 property 里，否则每次调用都会
        #   生成新名字，导致同一实例读写不一致），使同 worker 的新老实例也隔离。
        digest = hashlib.sha256(str(worker_id).encode("utf-8")).hexdigest()[:8]
        self.staging_token = f"{self.attempt}-{digest}-{uuid.uuid4().hex[:8]}"
        # 本实例观测到的暂存文件集合：只清理这些，绝不盲扫目录里别人的文件。
        self._owned: set[Path] = set()

    @property
    def staged_path(self) -> Path:
        """本次执行实例的暂存输出路径（与正式目标同目录、保证 os.replace 原子）。"""

        return self.final_path.with_name(
            f".{self.final_path.name}.staging-{self.staging_token}"
        )

    @property
    def staged_checkpoint_path(self) -> Path:
        """translate 检查点的暂存路径（与正式检查点同目录，同样按实例唯一）。"""

        return self.final_path.with_name(
            f"{self.final_path.name}.progress.json.staging-{self.staging_token}"
        )

    def iter_staged_paths(self) -> list[Path]:
        """本次实例可能产生的全部暂存文件（产物 + 发布标记 + 检查点）。"""

        return [
            self.staged_path,
            Path(f"{self.staged_path}.published"),
            self.staged_checkpoint_path,
        ]

    def note_owned(self, path: Path) -> None:
        """登记一个本次实例会写入的暂存路径（仅这些路径允许被清理）。"""

        self._owned.add(Path(path))

    def owned_paths(self) -> list[Path]:
        """返回本实例登记过的暂存路径。"""

        return sorted(self._owned)


def _read_fd_path(fd: int) -> Path | None:
    """读取目录/文件 fd 指向的真实路径（Linux /proc 或 /dev/fd）。

    fd 是目录身份锚点：即使路径被 rename 或替换成符号链接，open 时拿到的
    仍是同一个目录对象，读取 fd 链接即可发现路径语义已变化。平台不提供
    fd 路径时返回 None（调用方仍需依赖 O_NOFOLLOW + dir_fd 相对 rename）。
    """

    for base in (f"/proc/self/fd/{fd}", f"/dev/fd/{fd}"):
        try:
            raw = os.readlink(base)
        except OSError:
            continue
        if raw.endswith(" (deleted)"):
            raw = raw[: -len(" (deleted)")]
        return Path(raw)
    return None


@dataclass
class CopyBackPlan:
    """copy_back 预复制计划：目录 fd 锚定 + 事务内短 rename（R3-6）。

    目标目录在慢拷贝前以 O_DIRECTORY|O_NOFOLLOW 打开并校验真实路径；临时文件
    通过该 fd 创建，最终 rename 通过同一 fd 的 src_dir_fd/dst_dir_fd 执行。
    这样慢拷贝期间目录被 rename/替换为符号链接时：

    - 慢拷贝不会跟随新路径写到 outside（写入锚定在打开时的目录对象上）；
    - commit 会发现 fd 真实路径已变化并拒绝，整个 DB 事务回滚；
    - close/discard 仍能通过同一 fd 删除本实例的临时文件，不留半成品。
    """

    dir_fd: int
    dir_path: Path
    pairs: list[tuple[str, str]]
    _closed: bool = False

    def commit(self) -> None:
        """在发布事务内校验目录身份并完成短 rename（同目录 fd，跨 FS 安全）。

        调用方必须保证已持有 repo 的租约写锁；本方法只做 O(1) 校验与 rename，
        不做慢 I/O。任何校验失败都抛异常，由 db.tx 回滚且不置 succeeded。
        """

        if self._closed:
            raise RuntimeError("copy_back_plan_closed")
        actual = _read_fd_path(self.dir_fd)
        if actual is not None and actual != self.dir_path:
            # 慢拷贝期间目录被移动/替换：拒绝提交，绝不把正式文件写到新路径。
            raise RuntimeError(
                f"copy_back_target_dir_changed:{self.dir_path}->{actual}"
            )
        for temp_name, target_name in self.pairs:
            try:
                target_stat = os.stat(
                    target_name, dir_fd=self.dir_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                target_stat = None
            if target_stat is not None and stat.S_ISLNK(target_stat.st_mode):
                raise RuntimeError(
                    f"copy_back_target_symlink:{self.dir_path / target_name}"
                )
            try:
                temp_stat = os.stat(
                    temp_name, dir_fd=self.dir_fd, follow_symlinks=False
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"copy_back_temp_missing:{self.dir_path / temp_name}"
                ) from exc
            if not stat.S_ISREG(temp_stat.st_mode):
                raise RuntimeError(
                    f"copy_back_temp_invalid:{self.dir_path / temp_name}"
                )
        for temp_name, target_name in self.pairs:
            # src/dst 都相对同一个已校验的目录 fd：路径替换无法改变落点。
            os.replace(
                temp_name,
                target_name,
                src_dir_fd=self.dir_fd,
                dst_dir_fd=self.dir_fd,
            )
        try:
            os.fsync(self.dir_fd)
        except OSError as exc:  # noqa: BLE001 - rename 已提交，无法撤销
            LOGGER.warning(
                "copy_back directory fsync failed: dir=%s error=%s",
                self.dir_path,
                exc,
            )

    def discard(self) -> None:
        """通过 dir_fd 删除尚未提交的临时文件（目录被移动也能清理自己的文件）。"""

        if self._closed:
            return
        for temp_name, _target_name in self.pairs:
            try:
                os.unlink(temp_name, dir_fd=self.dir_fd)
            except FileNotFoundError:
                continue
            except OSError as exc:  # noqa: BLE001 - 清理失败不影响主结论
                LOGGER.warning(
                    "copy_back temp cleanup failed: dir=%s name=%s error=%s",
                    self.dir_path,
                    temp_name,
                    exc,
                )

    def close(self) -> None:
        """丢弃未提交的临时文件并关闭目录 fd（幂等；成功提交后 discard 为空操作）。"""

        if self._closed:
            return
        try:
            self.discard()
        finally:
            self._closed = True
            try:
                os.close(self.dir_fd)
            except OSError as exc:  # noqa: BLE001
                LOGGER.warning(
                    "copy_back dir fd close failed: dir=%s error=%s",
                    self.dir_path,
                    exc,
                )


def _parse_iso_timestamp(raw: str) -> datetime | None:
    """把 ISO8601 字符串解析为 aware datetime；无时区按 UTC 处理，失败返回 None。"""

    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _supports_keyword(func: Callable[..., Any], name: str) -> bool:
    """判断可调用对象是否显式接受某个关键字参数。"""

    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
