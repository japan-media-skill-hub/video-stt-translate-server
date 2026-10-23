"""fake executor 集成测试：真实 WorkerRuntime + SQLite + TestClient，全程无模型/ffmpeg/网络。

覆盖 SPEC 批次 1/2 的关键链路（fake 执行器替换 GPU/ffmpeg/翻译上游）：

- 默认 DAG（extract→stt→translate）与 WhisperX DAG（extract→stt_whisperx→translate）成功闭环；
- 上游 stt 失败 → 下游 translate blocked → job failed；
- fake hang + 停止心跳 → 租约被 reaper 回收（attempt+1），旧 attempt 不得发布产物；
- 进程重启 recovery 计入 attempt；
- drain 有界：hang 任务在 drain 超时后被强制回滚并触发注入式退出回调；
- 旧 attempt 的写回被 fencing 拒绝。

等待全部基于事件/条件轮询（带超时），不使用固定 sleep 做断言。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import threading
from typing import Any, Callable

import logging

import pytest

from whisper_stt_service.api import create_app
from whisper_stt_service.core.config import (
    RetrySettings,
    RuntimeSettings,
    SecuritySettings,
    Settings,
    SttSettings,
    SttWhisperxSettings,
    TimeoutSettings,
    WorkerSettings,
)
from whisper_stt_service.core.progress import ProgressStore
from whisper_stt_service.executor.common import (
    ExecutorCancelledError,
    MissingAudioArtifactError,
)
from whisper_stt_service.repo.database import Database
from whisper_stt_service.repo.job_repository import JobRepository
from whisper_stt_service.service import runtime as runtime_module
from whisper_stt_service.service.runtime import WorkerRuntime


LOGGER = logging.getLogger(__name__)

LEASE_TIMEOUT_SEC = 2
DRAIN_TIMEOUT_SEC = 2
HEARTBEAT_INTERVAL_SEC = 0.05
# 成功链在秒级内完成（worker poll=1s，无真实模型）；失败等待只用于租约/停机超时场景。
WAIT_TIMEOUT_SEC = 15.0
POLL_INTERVAL_SEC = 0.02


def wait_until(
    predicate: Callable[[], Any],
    *,
    timeout: float = WAIT_TIMEOUT_SEC,
    description: str = "condition",
) -> Any:
    """事件式等待：轮询谓词直到为真（真值即返回），超时给出清晰断言信息。"""

    deadline = time.monotonic() + timeout
    last_exc: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except BaseException as exc:  # noqa: BLE001
            last_exc = exc
        time.sleep(POLL_INTERVAL_SEC)
    detail = f" last_error={last_exc!r}" if last_exc is not None else ""
    raise AssertionError(f"timed out waiting for {description} ({timeout}s){detail}")


import time  # noqa: E402  (放在函数定义之后以保持常量分区整洁)


def _settings(tmp_path: Path) -> Settings:
    """构造集成测试用配置：单 worker/阶段 + 秒级 lease/drain 超时。"""

    return Settings(
        workers=WorkerSettings(
            extract_workers=1,
            stt_workers=1,
            stt_whisperx_workers=1,
            translate_workers=1,
            scheduler_interval_sec=60,
            poll_interval_sec=1,
        ),
        timeouts=TimeoutSettings(
            extract_timeout_sec=60,
            stt_timeout_sec=60,
            stt_whisperx_timeout_sec=60,
            translate_timeout_sec=60,
            lease_timeout_sec=LEASE_TIMEOUT_SEC,
            drain_timeout_sec=DRAIN_TIMEOUT_SEC,
        ),
        retry=RetrySettings(
            extract_max_retries=2,
            stt_max_retries=2,
            stt_whisperx_max_retries=2,
            translate_max_retries=2,
        ),
        runtime=RuntimeSettings(
            db_path=tmp_path / "state.db",
            progress_ttl_sec=300,
            log_root=tmp_path / "logs",
            artifact_root=tmp_path / "artifacts",
            artifact_cleanup_enabled=False,
            artifact_cleanup_interval_sec=3600,
            artifact_cleanup_statuses=("succeeded",),
            model_path=tmp_path / "model",
        ),
        stt=SttSettings(
            device="cpu",
            compute_type="int8",
            batch_size=1,
            beam_size=1,
            best_of=1,
            patience=1.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_threshold=0.45,
            vad_min_speech_duration_ms=200,
            vad_max_speech_duration_s=18.0,
            vad_min_silence_duration_ms=700,
            vad_speech_pad_ms=300,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.2,
            log_prob_threshold=-1.0,
            hallucination_silence_threshold=1.5,
            initial_prompt="",
            hotwords="",
        ),
        stt_whisperx=SttWhisperxSettings(
            model=str(tmp_path / "model"),
            device="cpu",
            compute_type="int8",
            batch_size=1,
            vad_config_path=tmp_path / "vad" / "config.yaml",
            align_model_root=tmp_path / "align",
            align_enabled=False,
            vad_backend="pyannote",
            vad_onset=0.35,
            vad_offset=0.2,
            local_files_only=True,
        ),
        security=SecuritySettings(api_token=""),
    )


DEFAULT_DAG = {
    "version": 1,
    "stages": [
        {"stage": "extract", "depends_on": []},
        {"stage": "stt", "depends_on": ["extract"]},
        {"stage": "translate", "depends_on": ["stt"]},
    ],
}

WHISPERX_DAG = {
    "version": 1,
    "stages": [
        {"stage": "extract", "depends_on": []},
        {"stage": "stt_whisperx", "depends_on": ["extract"]},
        {"stage": "translate", "depends_on": ["stt_whisperx"]},
    ],
}


@dataclass
class FakeExecutors:
    """可编程 fake 执行器：成功 / 失败 / hang / 取消响应 + 产物发布标记。"""

    artifact_root: Path
    starts: dict[str, threading.Event] = field(default_factory=dict)
    gates: dict[str, threading.Event] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    fail_stages: set[str] = field(default_factory=set)
    hang_events: dict[str, threading.Event] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    # 需要“卡在执行器内部”的 key（stage:task_id）集合；未登记的执行器直接发布产物。
    blocked_keys: set[str] = field(default_factory=set)

    def _events(self, key: str) -> tuple[threading.Event, threading.Event]:
        """取出 (started, gate) 事件对，缺失时创建；gate 默认已置位（不阻塞）。"""

        with self.lock:
            started = self.starts.setdefault(key, threading.Event())
            gate = self.gates.setdefault(key, threading.Event())
            # 默认放行：只有显式登记为阻塞的 key 才需要测试后续 set()。
            if key not in self.blocked_keys:
                gate.set()
        return started, gate

    def block_key(self, key: str) -> threading.Event:
        """把某个执行器调用登记为“阻塞”，返回其 gate（测试稍后 set() 放行）。"""

        with self.lock:
            self.blocked_keys.add(key)
            gate = self.gates.setdefault(key, threading.Event())
            gate.clear()
        return gate

    def unblock_key(self, key: str) -> None:
        """解除阻塞登记，并放行可能正在等待的 gate。"""

        with self.lock:
            self.blocked_keys.discard(key)
            gate = self.gates.setdefault(key, threading.Event())
            gate.set()

    def _record(self, **payload: Any) -> None:
        """线程安全地记录一次执行器调用（含 wav 契约观测）。"""

        with self.lock:
            self.calls.append(payload)

    def calls_for(self, stage: str) -> list[dict[str, Any]]:
        """返回某阶段的调用记录列表。"""

        with self.lock:
            return [c for c in self.calls if c["stage"] == stage]

    def clear(self) -> None:
        """清空调用记录与失败设置（重启模拟时复用同一 fake）。"""

        with self.lock:
            self.calls.clear()
            self.fail_stages.clear()

    def release_all(self) -> None:
        """放行全部 gate 与 hang 事件，避免用例结束时线程悬挂。"""

        with self.lock:
            events = list(self.gates.values()) + list(self.hang_events.values())
        for event in events:
            event.set()

    # -- 通用流程 -----------------------------------------------------------
    def _gate_publish(
        self,
        key: str,
        stage: str,
        cancel_check: Callable[[], bool] | None,
        output_path: Path,
        content: str,
    ) -> None:
        """把产物写进 runtime 传入的 staging 路径；仅当 key 被登记为阻塞时才等待放行。

        A.3 产物级 fencing：执行器只写 attempt 私有暂存路径，正式路径由 runtime 在
        fencing 事务内 promotion。这里绝不触碰正式路径，否则旧 worker 会覆盖新产物。
        """

        started, gate = self._events(key)
        started.set()
        with self.lock:
            blocked = key in self.blocked_keys
        if blocked:
            while not gate.wait(timeout=POLL_INTERVAL_SEC):
                if cancel_check is not None and cancel_check():
                    raise ExecutorCancelledError("cancelled_by_upstream")
        if cancel_check is not None and cancel_check():
            raise ExecutorCancelledError("cancelled_by_upstream")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

    def _maybe_hang(
        self, stage: str, key: str, cancel_check: Callable[[], bool] | None
    ) -> None:
        """若该阶段被设为 hang：一直阻塞到测试放行或收到取消。"""

        event = self.hang_events.get(stage)
        if event is None:
            return
        started, _gate = self._events(key)
        started.set()
        while not event.wait(timeout=POLL_INTERVAL_SEC):
            if cancel_check is not None and cancel_check():
                raise ExecutorCancelledError("cancelled_by_upstream")

    @staticmethod
    def _check_wav(input_path: Path, require: Any) -> dict[str, Any]:
        """校验 wav 契约：require=True 时输入必须存在且非空（不得回退视频）。"""

        exists = input_path.is_file()
        size = input_path.stat().st_size if exists else 0
        if require and (not exists or size <= 0):
            raise MissingAudioArtifactError(input_path)
        return {"input_exists": exists, "input_size": size, "input_suffix": input_path.suffix}

    # -- fake 执行器（签名与真实执行器一致，便于插拔） -----------------------
    def run_extract(
        self,
        input_video: Path,
        output_wav: Path,
        timeout_sec: int,
        *,
        progress_queue: Any = None,
        task_id: str | None = None,
        worker_id: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """fake extract：把 16k wav 写入 runtime 传入的 staging 路径并返回契约快照。"""

        key = f"extract:{task_id}"
        self._record(
            stage="extract",
            task_id=task_id,
            worker_id=worker_id,
            input=str(input_video),
            output=str(output_wav),
            has_cancel_check=cancel_check is not None,
        )
        self._maybe_hang("extract", key, cancel_check)
        if "extract" in self.fail_stages:
            raise RuntimeError("fake_extract_failed")
        self._gate_publish(key, "extract", cancel_check, Path(output_wav), "wav-16k-mono")
        return {
            "effective_config": {
                "input_video": str(input_video),
                "output_wav": str(output_wav),
                "sample_rate": 16000,
                "channels": 1,
                "codec": "pcm_s16le",
            }
        }

    def _run_asr(self, stage: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """ASR 通用实现（stt / stt_whisperx）：校验 wav 契约后写 staging 字幕。"""

        task_id = kwargs.get("task_id")
        cancel_check = kwargs.get("cancel_check")
        input_path = Path(kwargs["input_path"])
        output_path = Path(kwargs["output_path"])
        key = f"{stage}:{task_id}"
        record = {
            "stage": stage,
            "task_id": task_id,
            "worker_id": kwargs.get("worker_id"),
            "input": str(input_path),
            "output": str(output_path),
            "require_audio_artifact": kwargs.get("require_audio_artifact"),
            "has_cancel_check": cancel_check is not None,
        }
        record.update(self._check_wav(input_path, kwargs.get("require_audio_artifact")))
        self._record(**record)
        self._maybe_hang(stage, key, cancel_check)
        if stage in self.fail_stages:
            raise RuntimeError(f"fake_{stage}_failed")
        # 字幕内容带上 worker，便于断言“正式产物来自哪个 attempt”。
        content = f"{stage}-by-{kwargs.get('worker_id')}"
        self._gate_publish(key, stage, cancel_check, output_path, content)
        return {"effective_config": {"model_cache_hit": False}}

    def run_stt(self, **kwargs: Any) -> dict[str, Any]:
        """fake stt 执行器。"""

        return self._run_asr("stt", kwargs)

    def run_stt_whisperx(self, **kwargs: Any) -> dict[str, Any]:
        """fake stt_whisperx 执行器。"""

        return self._run_asr("stt_whisperx", kwargs)

    def run_translate(self, **kwargs: Any) -> None:
        """fake translate：读取已发布的 ja.srt，写 staging zh.srt（空返回与真实一致）。"""

        task_id = kwargs.get("task_id")
        cancel_check = kwargs.get("cancel_check")
        # 真实 run_translate 使用 input_ja_srt/output_zh_srt 形参名。
        input_path = Path(kwargs["input_ja_srt"])
        output_path = Path(kwargs["output_zh_srt"])
        key = f"translate:{task_id}"
        self._record(
            stage="translate",
            task_id=task_id,
            worker_id=kwargs.get("worker_id"),
            input=str(input_path),
            output=str(output_path),
            input_exists=input_path.is_file(),
            has_cancel_check=cancel_check is not None,
        )
        self._maybe_hang("translate", key, cancel_check)
        if "translate" in self.fail_stages:
            raise RuntimeError("fake_translate_failed")
        self._gate_publish(key, "translate", cancel_check, output_path, "zh-translate")


class Stack:
    """SQLite + JobRepository + ProgressStore + WorkerRuntime + TestClient 的集成栈。"""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """装配依赖，并把 runtime 中的执行器引用替换为 fake。"""

        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.settings = _settings(tmp_path)
        self.db = Database(self.settings.runtime.db_path)
        self.db.init_schema()
        self.repo = JobRepository(
            self.db,
            stage_max_retries={
                "extract": 2,
                "stt": 2,
                "stt_whisperx": 2,
                "translate": 2,
            },
            stage_timeouts={
                "extract": 60,
                "stt": 60,
                "stt_whisperx": 60,
                "translate": 60,
            },
            log_root=self.settings.runtime.log_root,
            artifact_root=self.settings.runtime.artifact_root,
        )
        self.progress_store = ProgressStore(ttl_seconds=300)
        self.fake = FakeExecutors(artifact_root=self.settings.runtime.artifact_root)
        self.exit_calls = 0
        self.runtimes: list[WorkerRuntime] = []
        monkeypatch.setattr(runtime_module, "run_extract", self.fake.run_extract)
        monkeypatch.setattr(runtime_module, "run_stt", self.fake.run_stt)
        monkeypatch.setattr(runtime_module, "run_stt_whisperx", self.fake.run_stt_whisperx)
        monkeypatch.setattr(runtime_module, "run_translate", self.fake.run_translate)

    def _exit_callback(self) -> None:
        """注入式退出钩子：只计数，绝不向当前进程发信号（A.6）。"""

        self.exit_calls += 1

    def new_runtime(self) -> WorkerRuntime:
        """构造（不启动）一个 WorkerRuntime，便于模拟重启。"""

        runtime = WorkerRuntime(
            repo=self.repo,
            progress_store=self.progress_store,
            settings=self.settings,
            config_path=self.tmp_path / "config.ini",
            model_path=str(self.tmp_path / "model"),
            exit_callback=self._exit_callback,
            heartbeat_interval_sec=HEARTBEAT_INTERVAL_SEC,
        )
        self.runtimes.append(runtime)
        return runtime

    def app(self, runtime: WorkerRuntime):
        """构造绑定该 runtime 的 FastAPI 应用（TestClient 兼容处理见 _TestClientWithState）。"""

        return create_app(
            repo=self.repo,
            progress_store=self.progress_store,
            runtime=runtime,
            api_token="",
            metrics_enabled=False,
        )

    def submit(self, client, video_path: Path, *, dag: dict | None = None):
        """经 HTTP API 提交 job。"""

        payload: dict[str, Any] = {"video_path": str(video_path), "language": "ja"}
        if dag is not None:
            payload["dag"] = dag
        return client.post("/jobs", json=payload)

    def tasks(self, job_id: str) -> dict[str, dict]:
        """按 stage 返回任务行（直接读 SQLite 全列，作为最终断言来源）。

        用 `list_tasks` 而非 `get_job_detail`：后者只返回展示列（不含 log_file 等），
        集成测试需要核对任务日志与租约字段。
        """

        items = self.repo.list_tasks(page=1, page_size=200, job_id=job_id)["items"]
        assert items
        return {str(t["stage"]): t for t in items}

    def job_status(self, job_id: str) -> str | None:
        """读取 job 聚合状态。"""

        return self.repo.get_job_status(job_id)

    def stop_all(self) -> None:
        """停止全部 runtime。

        顺序很重要：先放行所有测试自建的 gate/hang 事件，再 stop，否则 worker 会一直
        阻塞在 fake 执行器里，join 只能等满超时。任何停止异常都记录日志并继续释放其余
        资源，最后统一抛出，避免静默吞掉（SPEC §7.3-3）。
        """

        self.fake.release_all()
        failures: list[BaseException] = []
        for runtime in self.runtimes:
            try:
                runtime.stop(timeout_sec=5)
            except BaseException as exc:  # noqa: BLE001
                LOGGER.exception("integration teardown: runtime.stop failed: %r", exc)
                failures.append(exc)
        if failures:
            raise RuntimeError(
                f"integration teardown: {len(failures)} runtime(s) failed to stop"
            ) from failures[0]


@pytest.fixture
def stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """提供集成栈与 TestClient 工厂，用例结束统一停机与放行。"""

    from fastapi.testclient import TestClient

    built = Stack(tmp_path, monkeypatch=monkeypatch)
    clients: list[Any] = []

    def make_client(runtime: WorkerRuntime):
        """创建并登记 TestClient（测试自行 with/关闭）。"""

        client = TestClient(built.app(runtime))
        clients.append(client)
        return client

    built.make_client = make_client  # type: ignore[attr-defined]
    try:
        yield built
    finally:
        failures: list[BaseException] = []
        for client in clients:
            try:
                client.close()
            except BaseException as exc:  # noqa: BLE001
                LOGGER.exception("integration teardown: client.close failed: %r", exc)
                failures.append(exc)
        try:
            built.stop_all()
        except BaseException as exc:  # noqa: BLE001
            failures.append(exc)
        if failures:
            raise RuntimeError(
                f"integration teardown: {len(failures)} cleanup step(s) failed"
            ) from failures[0]


def _video(tmp_path: Path, name: str) -> Path:
    """创建占位视频（fake 执行器不读取内容）。"""

    path = tmp_path / name
    path.write_bytes(b"fake-video")
    return path


def _read_events(log_file: str) -> list[dict[str, Any]]:
    """读取 task.log（jsonl）为事件列表。"""

    path = Path(log_file)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_default_dag_runs_three_stages(stack: Stack, tmp_path: Path) -> None:
    """起服务→提交默认 DAG→runtime 跑完 extract/stt/translate，产物与状态符合预期。"""

    runtime = stack.new_runtime()
    runtime.start()
    with stack.make_client(runtime) as client:
        response = stack.submit(client, _video(tmp_path, "sample.mp4"))
        assert response.status_code in {200, 202}, response.text
        job_id = response.json()["job_id"]

        wait_until(lambda: stack.job_status(job_id) == "succeeded", description="job succeeded")

        tasks = stack.tasks(job_id)
        assert {stage: row["status"] for stage, row in tasks.items()} == {
            "extract": "succeeded",
            "stt": "succeeded",
            "translate": "succeeded",
        }
        # 执行顺序：extract → stt → translate；ASR 的输入是 extract 的**正式**产物。
        # A.3 产物级 fencing：执行器只写 attempt 私有暂存（output_* 为 staging 路径），
        # promotion 后才成为正式路径；下游读到的必须是已发布的正式产物。
        assert [c["stage"] for c in stack.fake.calls] == ["extract", "stt", "translate"]
        extract_call = stack.fake.calls_for("extract")[0]
        stt_call = stack.fake.calls_for("stt")[0]
        assert ".staging-" in Path(extract_call["output"]).name
        assert stt_call["input"] != extract_call["output"]
        # ASR 读到的是 promotion 之后的正式 wav（内容等于 fake 写入 staging 的字节）。
        assert Path(stt_call["input"]).read_text(encoding="utf-8") == "wav-16k-mono"
        assert stt_call["input_suffix"] == ".wav"
        assert stt_call["require_audio_artifact"] is True
        assert stt_call["has_cancel_check"] is True

        detail = stack.repo.get_job_detail(job_id)
        assert detail is not None
        # 正式产物来自对应 attempt 的 staging（worker 名编码在内容里）。
        # R3-4：worker_id 保留 stt-w1 可读前缀，并追加每 runtime 启动 nonce。
        assert Path(detail["output_ja_path"]).read_text(encoding="utf-8").startswith(
            "stt-by-stt-w1-"
        )
        assert (
            Path(detail["output_zh_path"]).read_text(encoding="utf-8") == "zh-translate"
        )
        # 正式路径上不得残留任何 staging 文件。
        job_dir = stack.settings.runtime.artifact_root / job_id
        assert [p.name for p in job_dir.rglob("*") if ".staging-" in p.name] == []


def test_whisperx_dag_runs_and_consumes_wav(stack: Stack, tmp_path: Path) -> None:
    """显式 WhisperX DAG 闭环，stt_whisperx 消费 extract 的 wav。"""

    runtime = stack.new_runtime()
    runtime.start()
    with stack.make_client(runtime) as client:
        response = stack.submit(client, _video(tmp_path, "wx.mp4"), dag=WHISPERX_DAG)
        assert response.status_code in {200, 202}, response.text
        job_id = response.json()["job_id"]

        wait_until(
            lambda: stack.job_status(job_id) == "succeeded",
            description="whisperx job succeeded",
        )
        assert set(stack.tasks(job_id)) == {"extract", "stt_whisperx", "translate"}
        asr_call = stack.fake.calls_for("stt_whisperx")[0]
        # ASR 读正式 wav（promotion 结果），而非 extract 的私有 staging 路径。
        assert Path(asr_call["input"]).read_text(encoding="utf-8") == "wav-16k-mono"
        assert ".staging-" not in asr_call["input"]
        assert asr_call["input_suffix"] == ".wav"


def test_upstream_failure_blocks_downstream(stack: Stack, tmp_path: Path) -> None:
    """A.4：stt 失败 → translate blocked → job failed；blocked 任务从未执行。"""

    runtime = stack.new_runtime()
    # 提交前就注入失败：否则 stt 可能在“extract 成功后、注入之前”抢先跑完并成功。
    stack.fake.fail_stages.add("stt")
    runtime.start()
    with stack.make_client(runtime) as client:
        job_id = stack.submit(client, _video(tmp_path, "fail.mp4")).json()["job_id"]
        wait_until(
            lambda: stack.tasks(job_id)["extract"]["status"] == "succeeded",
            description="extract succeeded",
        )
        wait_until(
            lambda: stack.job_status(job_id) == "failed",
            description="job failed after stt retries exhausted",
        )
        tasks = stack.tasks(job_id)
        assert tasks["stt"]["status"] == "failed"
        assert tasks["translate"]["status"] == "blocked"
        assert str(tasks["translate"]["last_error"]).startswith("upstream_failed:")
        assert stack.fake.calls_for("translate") == []


def test_hung_task_reaped_attempt_incremented(stack: Stack, tmp_path: Path) -> None:
    """A.2：hang（不续租）→ 租约过期被 reaper 回收且 attempt+1，旧 attempt 不发布产物。"""

    runtime = stack.new_runtime()
    hang = threading.Event()
    stack.fake.hang_events["stt"] = hang
    original_start = runtime_module.FencedLease.start
    # 只在“第一次 stt 领取”时停掉心跳（模拟该 worker 卡死不再续租）；
    # 重试后的第二次领取必须正常续租，否则重试会被再次回收，无法收敛。
    disabled_once: set[str] = set()
    lock = threading.Lock()

    def start_without_heartbeat_once(lease) -> None:
        """仅首次 stt 租约不续租，等价于单个 worker 卡死。"""

        original_start(lease)
        with lock:
            disable = lease.stage == "stt" and lease.task_id not in disabled_once
            if disable:
                disabled_once.add(lease.task_id)
        if disable:
            lease.stop()

    stack.monkeypatch.setattr(
        runtime_module.FencedLease, "start", start_without_heartbeat_once
    )
    runtime.start()
    with stack.make_client(runtime) as client:
        job_id = stack.submit(client, _video(tmp_path, "hang.mp4")).json()["job_id"]
        wait_until(
            lambda: stack.tasks(job_id)["stt"]["status"] == "claimed",
            description="stt claimed",
        )
        first_attempt = int(stack.tasks(job_id)["stt"]["attempt"])
        wait_until(
            lambda: runtime.reaper_stats()["reclaimed"] >= 1,
            description="reaper reclaimed expired lease",
        )
        wait_until(
            lambda: int(stack.tasks(job_id)["stt"]["attempt"]) == first_attempt + 1,
            description="attempt incremented after reclaim",
        )
        tasks = stack.tasks(job_id)
        events = _read_events(tasks["stt"]["log_file"])
        assert any(e["event"] == "lease_reclaimed" for e in events)

        # 旧执行器此刻仍在 hang；放行后它会发现自己失租，必须放弃发布。
        stack.fake.hang_events.pop("stt", None)
        hang.set()
        wait_until(
            lambda: stack.job_status(job_id) in {"running", "succeeded"},
            description="retried stt progressing",
        )
        wait_until(
            lambda: stack.job_status(job_id) == "succeeded",
            description="job succeeded after retry",
        )
        tasks = stack.tasks(job_id)
        assert tasks["stt"]["status"] == "succeeded"
        assert int(tasks["stt"]["attempt"]) == first_attempt + 1

        # A.3 产物级 fencing：正式 ja.srt 必须来自**重试后的新 attempt**，
        # 旧 attempt（hang 中被取消）不得把自己的 staging 发布到正式路径。
        output_ja = Path(stack.repo.get_job_detail(job_id)["output_ja_path"])
        content = output_ja.read_text(encoding="utf-8")
        # R3-4：owner 文本为 stt-w1-<boot_nonce>，可读前缀必须保持稳定。
        assert content.startswith("stt-by-stt-w1-")
        stt_calls = stack.fake.calls_for("stt")
        assert len(stt_calls) >= 2
        # 每次 stt 调用的输出都必须是私有 staging，且没有残留文件。
        for call in stt_calls:
            assert ".staging-" in Path(call["output"]).name
            assert not Path(call["output"]).exists()
        job_dir = stack.settings.runtime.artifact_root / job_id
        assert [p.name for p in job_dir.rglob("*") if ".staging-" in p.name] == []
        # 发布事件必须由新 attempt 写入 task 日志。
        events = _read_events(tasks["stt"]["log_file"])
        published = [e for e in events if e.get("event") == "artifacts_published"]
        assert published, "missing artifacts_published event"
        assert int(published[-1]["attempt"]) == first_attempt + 1


def test_restart_recovery_counts_attempt(stack: Stack, tmp_path: Path) -> None:
    """A.5：进程重启把僵尸 claimed 回 queued 且 attempt+1，随后可继续跑完。"""

    runtime = stack.new_runtime()
    hang = threading.Event()
    stack.fake.hang_events["extract"] = hang
    original_start = runtime_module.FencedLease.start

    def start_without_heartbeat(lease) -> None:
        """启动租约但不续租（模拟崩溃前的执行状态）。"""

        original_start(lease)
        lease.stop()

    stack.monkeypatch.setattr(
        runtime_module.FencedLease, "start", start_without_heartbeat
    )
    runtime.start()
    with stack.make_client(runtime) as client:
        job_id = stack.submit(client, _video(tmp_path, "restart.mp4")).json()["job_id"]
        wait_until(
            lambda: stack.tasks(job_id)["extract"]["status"] == "claimed",
            description="extract claimed before crash",
        )
        claimed_attempt = int(stack.tasks(job_id)["extract"]["attempt"])

        # 模拟“进程消失”：不调用 stop（stop 会回滚本进程 claimed），
        # 直接以新的 runtime 做启动恢复。
        stack.fake.hang_events.pop("extract", None)
        hang.set()
        stack.fake.clear()

        restarted = stack.new_runtime()
        restarted.start()
        assert restarted.startup_stats()["recovered"] >= 1
        wait_until(
            lambda: int(stack.tasks(job_id)["extract"]["attempt"]) > claimed_attempt,
            description="attempt incremented by startup recovery",
        )
        wait_until(
            lambda: stack.job_status(job_id) == "succeeded",
            description="job succeeded after restart",
        )
        assert int(stack.tasks(job_id)["extract"]["attempt"]) >= claimed_attempt + 1


def test_drain_timeout_is_bounded_for_hung_task(stack: Stack, tmp_path: Path) -> None:
    """A.6：hang 任务下 drain 在 drain_timeout_sec 内完成强制回滚并触发退出回调。"""

    runtime = stack.new_runtime()
    stack.fake.hang_events["stt"] = threading.Event()
    runtime.start()
    with stack.make_client(runtime) as client:
        job_id = stack.submit(client, _video(tmp_path, "drain.mp4")).json()["job_id"]
        wait_until(
            lambda: stack.tasks(job_id)["stt"]["status"] == "claimed",
            description="stt claimed before drain",
        )
        started_status = runtime.request_shutdown(reason="integration_test")
        assert started_status["drain_requested"] is True

        wait_until(
            lambda: runtime.shutdown_status()["drain_timed_out"] is True,
            description="drain timeout reached",
        )
        wait_until(
            lambda: runtime.shutdown_status()["inflight"] == [],
            description="inflight cleared after drain timeout",
        )
        status = runtime.shutdown_status()
        assert status["can_exit"] is True
        wait_until(lambda: stack.exit_calls >= 1, description="exit callback triggered")
        # 强制回滚后该任务不得被标记 succeeded（结果被丢弃）。
        assert stack.tasks(job_id)["stt"]["status"] != "succeeded"


def test_stale_attempt_writeback_is_fenced(stack: Stack, tmp_path: Path) -> None:
    """A.3：旧 attempt 的 succeeded 写回被 fencing 拒绝，任务与 job 状态不变。"""

    runtime = stack.new_runtime()
    runtime.start()
    with stack.make_client(runtime) as client:
        job_id = stack.submit(client, _video(tmp_path, "fence.mp4")).json()["job_id"]
        wait_until(
            lambda: stack.job_status(job_id) == "succeeded", description="job succeeded"
        )
        extract_task = stack.tasks(job_id)["extract"]
        outcome = stack.repo.mark_task_succeeded(
            extract_task["task_id"],
            worker_id=str(extract_task["lease_owner"] or "extract-w1"),
            attempt=int(extract_task["attempt"]) - 1,
        )
        assert outcome is False
        assert stack.tasks(job_id)["extract"]["status"] == "succeeded"
        assert stack.job_status(job_id) == "succeeded"