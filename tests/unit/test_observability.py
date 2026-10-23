"""可观测性（SPEC G）单元测试：日志配置 / JSONL / request_id / 指标 / 健康检查。

全部使用 fake 仓储与 tmp_path，不依赖模型、ffmpeg、网络或真实硬件。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from whisper_stt_service.api.observability import (
    REQUEST_ID_HEADER,
    REQUEST_ID_MAX_LENGTH,
    check_db_query,
    check_db_writable,
    check_directory_writable,
    check_model_path,
    resolve_request_id,
    sanitize_request_id,
)
from whisper_stt_service.repo.database import Database
from whisper_stt_service.repo.job_repository import JobRepository
from whisper_stt_service.service import logging_config as lc
from whisper_stt_service.service import metrics as sm
from whisper_stt_service.service.metrics import (
    MetricsState,
    build_metrics,
    compute_throughput,
    record_model_cache,
    record_task_claimed,
    record_task_finished,
    record_translate_batch,
    record_translate_retry,
)


# --------------------------------------------------------------------- helpers


class _FakeDB:
    """最小 DB 替身：connect/tx 都返回支持 SELECT 1 的伪连接。"""

    def __init__(self, db_path: Path, *, broken: bool = False) -> None:
        """记录伪 DB 路径；broken=True 时模拟连接/事务失败。"""

        self.db_path = db_path
        self.broken = broken

    def connect(self):
        """返回伪连接；broken=True 时连接即抛 sqlite3.Error。"""

        if self.broken:
            raise sqlite3.OperationalError("unable to open database file")
        return _FakeConn()

    def tx(self):
        """返回伪事务上下文（目标场景不写数据）。"""

        from contextlib import contextmanager

        @contextmanager
        def _tx():
            """伪事务上下文：broken=True 时模拟写锁失败。"""

            if self.broken:
                raise sqlite3.OperationalError("database is locked")
            yield _FakeConn()

        return _tx()


class _FakeConn:
    """伪 sqlite 连接：只支持 execute('SELECT 1') 与上下文管理。"""

    def execute(self, sql: str, *args):
        """返回首个行为 1 的游标替身。"""

        assert "SELECT 1" in sql
        return _FakeCursor()

    def __enter__(self):
        """支持 with 语句，返回自身。"""

        return self

    def __exit__(self, *exc_info):
        """退出 with 语句，不吞异常。"""

        return False

    def commit(self) -> None:
        """伪提交。"""

    def rollback(self) -> None:
        """伪回滚。"""


class _FakeCursor:
    """游标替身。"""

    def fetchone(self):
        """返回一行 `(1,)`。"""

        return (1,)


class _FakeRepo:
    """只暴露 db 与 queue_summary 的仓储替身。"""

    def __init__(self, db_path: Path, *, broken: bool = False, summary: dict | None = None) -> None:
        """装配伪 DB 与固定队列摘要。"""

        self.db = _FakeDB(db_path, broken=broken)
        self._summary = summary or {"stages": {}}

    def queue_summary(self) -> dict:
        """返回固定摘要。"""

        return self._summary


def _flush_handlers(logger_names: tuple[str, ...]) -> None:
    """关闭并移除 logger 上的文件 handler，避免临时目录在 Windows 上被占用。"""

    for name in logger_names:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)


def _restore_logger_state(saved: dict) -> None:
    """把单个 logger 的 handlers/level/propagate/disabled 全部还原。

    dictConfig 会把目标 logger 的 `propagate` 置为 False，若不还原会让后续
    依赖 caplog（root handler 捕获）的用例全部收不到日志。
    """

    for name, state in saved.items():
        logger = logging.getLogger(name)
        logger.handlers = list(state["handlers"])
        logger.setLevel(state["level"])
        logger.propagate = state["propagate"]
        logger.disabled = state["disabled"]


@pytest.fixture()
def logging_isolation():
    """保存并恢复所有目标 logger 与 root 的完整状态，防止 dictConfig 污染其他用例。"""

    names = lc.APP_LOGGER_NAMES + lc.UVICORN_LOGGER_NAMES + ("root",)
    saved: dict[str, dict] = {}
    for name in names:
        logger = logging.getLogger() if name == "root" else logging.getLogger(name)
        saved[name] = {
            "handlers": list(logger.handlers),
            "level": logger.level,
            "propagate": logger.propagate,
            "disabled": logger.disabled,
        }
    try:
        yield
    finally:
        _flush_handlers(lc.APP_LOGGER_NAMES + lc.UVICORN_LOGGER_NAMES)
        _restore_logger_state(saved)


# --------------------------------------------------------------- logging_config


def test_resolve_log_level_env_and_fallback(monkeypatch) -> None:
    """WHISPER_STT_LOG_LEVEL 生效；非法值回退 INFO 而不是抛错。"""

    monkeypatch.setenv("WHISPER_STT_LOG_LEVEL", "debug")
    assert lc.resolve_log_level() == "DEBUG"
    monkeypatch.setenv("WHISPER_STT_LOG_LEVEL", "not-a-level")
    assert lc.resolve_log_level() == "INFO"
    monkeypatch.delenv("WHISPER_STT_LOG_LEVEL", raising=False)
    assert lc.resolve_log_level() == "INFO"
    assert lc.resolve_log_level("WARNING") == "WARNING"


def test_resolve_jsonl_precedence(monkeypatch) -> None:
    """explicit 参数 > log_format > 环境变量。"""

    monkeypatch.setenv("WHISPER_STT_LOG_JSONL", "1")
    assert lc.resolve_jsonl_enabled() is True
    assert lc.resolve_jsonl_enabled(log_format="text") is False
    assert lc.resolve_jsonl_enabled(jsonl=False, log_format="json") is False
    monkeypatch.delenv("WHISPER_STT_LOG_JSONL", raising=False)
    assert lc.resolve_jsonl_enabled() is False


def test_build_logging_config_has_rotating_file_and_uvicorn_loggers() -> None:
    """dictConfig 结构：轮转文件 handler + 应用/uvicorn logger 全部接入。"""

    config = lc.build_logging_config(
        log_dir="/tmp/logs", level="DEBUG", jsonl=True, max_bytes=1024, backup_count=2
    )
    file_handler = config["handlers"]["file"]
    assert file_handler["class"] == "logging.handlers.RotatingFileHandler"
    assert file_handler["maxBytes"] == 1024
    assert file_handler["backupCount"] == 2
    assert file_handler["formatter"] == "json"
    assert config["handlers"]["console"]["formatter"] == "text"
    for name in ("whisper_stt_service", "uvicorn", "uvicorn.error", "uvicorn.access"):
        assert config["loggers"][name]["level"] == "DEBUG"
        assert set(config["loggers"][name]["handlers"]) == {"console", "file"}
    # 不关闭既有 logger，避免破坏宿主进程（pytest）的日志。
    assert config["disable_existing_loggers"] is False


def test_configure_logging_writes_jsonl_with_request_id(
    tmp_path: Path, logging_isolation
) -> None:
    """落盘的 service.log 为 JSONL，且带 request_id 与 extra 字段。"""

    lc.configure_logging(log_dir=tmp_path, level="INFO", jsonl=True)
    token = lc.bind_request_id("req-abc")
    try:
        logging.getLogger("whisper_stt_service.api.test").info(
            "hello world", extra={"job_id": "job-1", "stage": "stt"}
        )
    finally:
        lc.reset_request_id(token)
    _flush_handlers(lc.APP_LOGGER_NAMES)

    lines = (tmp_path / lc.LOG_FILE_NAME).read_text(encoding="utf-8").strip().splitlines()
    payloads = [json.loads(line) for line in lines]
    target = next(item for item in payloads if item.get("message") == "hello world")
    assert target["request_id"] == "req-abc"
    assert target["job_id"] == "job-1"
    assert target["stage"] == "stt"
    assert target["level"] == "INFO"
    # SPEC G 字段顺序：timestamp 在最前。
    assert list(target)[0] == "timestamp"


def test_configure_logging_rotation_creates_backup(
    tmp_path: Path, logging_isolation
) -> None:
    """超过 max_bytes 时轮转出 service.log.1，证明轮转配置真实生效。"""

    lc.configure_logging(
        log_dir=tmp_path, level="INFO", jsonl=True, max_bytes=200, backup_count=2
    )
    logger = logging.getLogger("whisper_stt_service.api.rotate")
    for index in range(30):
        logger.info("rotation line %d %s", index, "x" * 40)
    _flush_handlers(lc.APP_LOGGER_NAMES)

    assert (tmp_path / lc.LOG_FILE_NAME).exists()
    assert (tmp_path / f"{lc.LOG_FILE_NAME}.1").exists()


def test_configure_logging_unwritable_dir_falls_back_to_console(
    tmp_path: Path, logging_isolation, monkeypatch
) -> None:
    """日志目录不可创建时降级为仅控制台，不阻断启动。"""

    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    config = lc.configure_logging(log_dir=blocker / "logs", level="INFO")
    assert "file" not in config["handlers"]
    assert "console" in config["handlers"]


def test_jsonl_formatter_handles_unserializable_extra() -> None:
    """extra 里塞不可序列化对象也不能让日志写入失败。"""

    formatter = lc.JsonLineFormatter()
    record = logging.LogRecord(
        "whisper_stt_service.x", logging.INFO, __file__, 1, "msg", None, None
    )
    record.custom = object()
    payload = json.loads(formatter.format(record))
    assert payload["message"] == "msg"
    assert isinstance(payload["custom"], str)


def test_text_formatter_sanitizes_uvicorn_access_quotes() -> None:
    """uvicorn.access 的 \"GET /healthz\" 200 被清洗成无引号文本。"""

    formatter = lc.TextFormatter(lc.DEFAULT_TEXT_FORMAT)
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '"%s" %s',
        ("GET /healthz", 200),
        None,
    )
    rendered = formatter.format(record)
    assert "GET /healthz 200" in rendered


def test_request_id_filter_falls_back_to_dash_outside_request() -> None:
    """请求上下文外 request_id 记为 '-'，不会抛出/留空。"""

    filter_ = lc.RequestIdFilter()
    record = logging.LogRecord(
        "whisper_stt_service.x", logging.INFO, __file__, 1, "m", None, None
    )
    assert filter_.filter(record) is True
    assert record.request_id == "-"


# ----------------------------------------------------------------- request_id


def test_sanitize_request_id_accepts_safe_values() -> None:
    """合法 request_id 原样保留，非法字符/超长返回空串（由服务生成新 id）。"""

    assert sanitize_request_id("abc-123_X.Y:Z@1") == "abc-123_X.Y:Z@1"
    assert sanitize_request_id("  spaced  ") == "spaced"
    assert sanitize_request_id("bad id") == ""
    assert sanitize_request_id("bad\nnewline") == ""
    assert sanitize_request_id("x" * (REQUEST_ID_MAX_LENGTH + 1)) == ""


def test_resolve_request_id_prefers_incoming_header() -> None:
    """有合法上游 id 时沿用，否则生成 32 位十六进制 uuid。"""

    assert resolve_request_id("trace-1") == "trace-1"
    generated = resolve_request_id("bad id")
    assert len(generated) == 32
    assert all(ch in "0123456789abcdef" for ch in generated)
    assert resolve_request_id(None) != generated


# -------------------------------------------------------------------- metrics


def _counter_value(metrics: MetricsState, name: str, labels: dict[str, str] | None = None) -> float:
    """从 registry 读取 counter 当前值，便于断言埋点。"""

    body = metrics.render().decode("utf-8")
    label_part = ""
    if labels:
        rendered = ",".join(f'{key}="{value}"' for key, value in sorted(labels.items()))
        label_part = "{" + rendered + "}"
    prefix = f"{name}{label_part} "
    for line in body.splitlines():
        if line.startswith(prefix):
            return float(line[len(prefix) :])
    return 0.0


def test_metrics_registry_is_isolated_per_build() -> None:
    """反复构建（等价多次 create_app）不会触发重复注册异常。"""

    states = [build_metrics(stages=("extract",)) for _ in range(5)]
    assert len({id(state.registry) for state in states}) == 5
    record_task_claimed(states[0], "extract")
    assert _counter_value(states[0], "whisper_stt_tasks_claimed_total", {"stage": "extract"}) == 1
    assert _counter_value(states[1], "whisper_stt_tasks_claimed_total", {"stage": "extract"}) == 0


def test_record_helpers_write_expected_series() -> None:
    """领取/完成/翻译批次/重试/模型缓存埋点写到约定指标名。"""

    state = build_metrics(stages=("stt",))
    record_task_claimed(state, "stt")
    record_task_finished(state, stage="stt", result="succeeded", duration_sec=3.5)
    record_task_finished(state, stage="stt", result="failed")
    record_translate_batch(state, "partial")
    record_translate_retry(state)
    record_translate_retry(state, 2)
    record_model_cache(state, hit=True)
    record_model_cache(state, hit=False)

    body = state.render().decode("utf-8")
    assert (
        _counter_value(
            state, "whisper_stt_tasks_finished_total", {"stage": "stt", "result": "failed"}
        )
        == 1
    )
    assert _counter_value(state, "whisper_stt_translate_batches_total", {"result": "partial"}) == 1
    assert _counter_value(state, "whisper_stt_translate_retries_total") == 3
    assert _counter_value(state, "whisper_stt_model_cache_hits_total") == 1
    assert _counter_value(state, "whisper_stt_model_cache_misses_total") == 1
    assert "whisper_stt_task_duration_seconds_count" in body
    assert "whisper_stt_task_duration_seconds_sum" in body


def test_record_helpers_tolerate_none_metrics() -> None:
    """metrics 为 None（脚本模式）时调用不抛异常。"""

    record_task_claimed(None, "extract")
    record_task_finished(None, stage="extract", result="succeeded", duration_sec=1)
    record_translate_batch(None, "succeeded")
    record_translate_retry(None)
    record_model_cache(None, hit=True)


def test_queue_depth_collector_reads_repo_on_scrape() -> None:
    """queue_depth 在采集时读 DB；缺失的 stage/status 补 0。"""

    class _Repo:
        def __init__(self) -> None:
            """记录采集调用次数。"""

            self.calls = 0

        def queue_summary(self) -> dict:
            """返回固定摘要并累计调用次数（断言"采集时读 DB"）。"""

            self.calls += 1
            return {"stages": {"extract": {"queued": 3}}}

    repo = _Repo()
    state = build_metrics(
        repo_factory=lambda: repo,
        stages=("extract", "translate"),
        statuses=("queued", "claimed"),
    )
    first = state.render().decode("utf-8")
    assert 'whisper_stt_queue_depth{stage="extract",status="queued"} 3.0' in first
    assert 'whisper_stt_queue_depth{stage="translate",status="queued"} 0.0' in first
    state.render()
    assert repo.calls == 2


def test_queue_depth_collector_survives_repo_failure() -> None:
    """仓储抛错时只输出 0 并记日志，/metrics 不中断。"""

    class _BrokenRepo:
        def queue_summary(self) -> dict:
            """模拟 DB 加锁失败。"""

            raise sqlite3.OperationalError("database is locked")

    state = build_metrics(repo_factory=lambda: _BrokenRepo(), stages=("stt",))
    body = state.render().decode("utf-8")
    assert 'whisper_stt_queue_depth{stage="stt",status="queued"} 0.0' in body


def test_build_metrics_reports_missing_dependency(monkeypatch) -> None:
    """prometheus_client 缺失时降级为 noop，不抛 ImportError。"""

    monkeypatch.setattr(sm, "PROMETHEUS_AVAILABLE", False)
    state = build_metrics(stages=("extract",))
    assert isinstance(state.tasks_claimed_total, sm._NoopMetric)
    assert state.render() == b""
    record_task_claimed(state, "extract")


# ------------------------------------------------------------------ throughput


def test_compute_throughput_counts_only_recent_window() -> None:
    """只统计窗口内完成数，并按阶段给出 succeeded/failed/total。"""

    now = datetime(2026, 10, 22, 1, 0, tzinfo=timezone.utc)
    rows = [
        {"stage": "extract", "status": "succeeded", "finished_at": (now - timedelta(minutes=1)).isoformat()},
        {"stage": "extract", "status": "failed", "finished_at": (now - timedelta(minutes=4)).isoformat()},
        # 窗口外（> 5min）：不计入
        {"stage": "extract", "status": "succeeded", "finished_at": (now - timedelta(minutes=6)).isoformat()},
        {"stage": "stt", "status": "succeeded", "finished_at": (now - timedelta(seconds=30)).isoformat()},
        # 无 finished_at：跳过
        {"stage": "stt", "status": "succeeded", "finished_at": None},
    ]
    result = compute_throughput(rows, stages=("extract", "stt", "translate"), now=now)
    assert result["extract"] == {
        "succeeded": 1,
        "failed": 1,
        "total": 2,
        "window_sec": 300,
        "per_minute": pytest.approx(0.4),
    }
    assert result["stt"]["total"] == 1
    # 未出现完成的阶段也返回 0，前端无需判空。
    assert result["translate"]["total"] == 0


# ------------------------------------------------------------------- health API


def _real_repo(tmp_path: Path) -> JobRepository:
    """创建真实 SQLite 仓储（tmp_path），用于 health/ready 端到端断言。"""

    db = Database(tmp_path / "svc.db")
    db.init_schema()
    return JobRepository(db, log_root=tmp_path / "logs", artifact_root=tmp_path / "artifacts")


def _settings_stub(artifact_root: Path, model_path: Path, *, token: str = "strong-token-123456"):
    """构造带 security/runtime 的 settings 替身。"""

    from types import SimpleNamespace

    return SimpleNamespace(
        security=SimpleNamespace(
            api_token=token,
            require_token=True,
            allow_placeholder_token=False,
            docs_enabled=False,
        ),
        # 与 RuntimeSettings 同形的最小片段：readyz 只读取这两个字段。
        runtime=SimpleNamespace(artifact_root=artifact_root, model_path=model_path),
        metrics_enabled=True,
    )


def test_check_db_query_real_sqlite(tmp_path: Path) -> None:
    """healthz 的 SELECT 1 检查在真实 SQLite 上通过；坏库返回明确错误。"""

    repo = _real_repo(tmp_path)
    assert check_db_query(repo) == (True, None)
    broken = _FakeRepo(tmp_path / "missing" / "x.db", broken=True)
    ok, error = check_db_query(broken)
    assert ok is False and error and "sqlite_error" in error
    assert check_db_query(None) == (False, "repo_not_configured")


def test_check_db_writable_uses_real_transaction(tmp_path: Path) -> None:
    """可写检查会真实开启/提交事务并清理探测文件。"""

    repo = _real_repo(tmp_path)
    assert check_db_writable(repo) == (True, None)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".readyz-probe")]
    assert leftovers == []
    broken = _FakeRepo(tmp_path / "x.db", broken=True)
    ok, error = check_db_writable(broken)
    assert ok is False and error and "sqlite_not_writable" in error


def test_check_directory_writable_and_model_path(tmp_path: Path) -> None:
    """artifact_root 可写 / 模型路径存在性检查给出稳定错误码。"""

    assert check_directory_writable(tmp_path / "artifacts", "artifact_root") == (True, None)
    assert (tmp_path / "artifacts").is_dir()

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    # 空目录 / 只有隐藏文件视为未就位，避免"mkdir 后假就绪"。
    assert check_model_path(model_dir) == (False, "model_path_empty")
    (model_dir / ".gitkeep").write_text("", encoding="utf-8")
    assert check_model_path(model_dir) == (False, "model_path_empty")
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    assert check_model_path(model_dir) == (True, None)
    ok, error = check_model_path(tmp_path / "nope")
    assert ok is False and error == "model_path_missing"
    assert check_model_path(None) == (False, "model_path_not_configured")


def test_healthz_readyz_and_metrics_endpoints(tmp_path: Path) -> None:
    """healthz/readyz 200、/metrics 200 且含关键指标；全部需要 token。"""

    from fastapi.testclient import TestClient

    from whisper_stt_service.api.app import create_app

    repo = _real_repo(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    # readyz 要求模型产物真实存在（非空目录 + 已知扩展名）。
    (model_dir / "model.bin").write_bytes(b"fake-weights")
    settings = _settings_stub(tmp_path / "artifacts", model_dir)
    app = create_app(repo=repo, settings=settings)
    client = TestClient(app)
    headers = {"X-API-Token": "strong-token-123456"}

    health = client.get("/healthz", headers=headers)
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["checks"]["db_query"]["ok"] is True
    assert REQUEST_ID_HEADER in health.headers

    ready = client.get("/readyz", headers=headers)
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert set(ready.json()["checks"]) == {
        "config",
        "db_writable",
        "model_path",
        "artifact_root",
    }

    metrics_resp = client.get("/metrics", headers=headers)
    assert metrics_resp.status_code == 200
    body = metrics_resp.text
    for name in (
        "whisper_stt_tasks_claimed_total",
        "whisper_stt_tasks_finished_total",
        "whisper_stt_task_duration_seconds",
        "whisper_stt_queue_depth",
        "whisper_stt_translate_batches_total",
        "whisper_stt_translate_retries_total",
        "whisper_stt_model_cache_hits_total",
        "whisper_stt_model_cache_misses_total",
    ):
        assert name in body

    # 健康检查与指标同样受全局鉴权保护，不能匿名访问。
    for path in ("/healthz", "/readyz", "/metrics"):
        assert client.get(path).status_code == 401


def test_readyz_reports_503_with_details_when_not_ready(tmp_path: Path) -> None:
    """模型路径缺失时 readyz 返回 503 且逐项给出 ok/error 明细。"""

    from fastapi.testclient import TestClient

    from whisper_stt_service.api.app import create_app

    repo = _real_repo(tmp_path)
    settings = _settings_stub(tmp_path / "artifacts", tmp_path / "missing-model")
    client = TestClient(create_app(repo=repo, settings=settings))
    resp = client.get("/readyz", headers={"X-API-Token": "strong-token-123456"})
    assert resp.status_code == 503
    payload = resp.json()
    assert payload["status"] == "not_ready"
    assert payload["checks"]["model_path"]["ok"] is False
    assert payload["checks"]["model_path"]["error"] == "model_path_missing"
    assert payload["checks"]["db_writable"]["ok"] is True


def test_healthz_503_when_db_broken(tmp_path: Path) -> None:
    """DB 不可查询时 healthz 返回 503，而不是 500/200。"""

    from fastapi.testclient import TestClient

    from whisper_stt_service.api.app import create_app

    app = create_app(repo=_FakeRepo(tmp_path / "x.db", broken=True))
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 503
    assert resp.json()["status"] == "error"


def test_metrics_disabled_returns_404(tmp_path: Path) -> None:
    """metrics_enabled=false 时不注册 /metrics（404），健康检查仍然可用。"""

    from fastapi.testclient import TestClient

    from whisper_stt_service.api.app import create_app

    repo = _real_repo(tmp_path)
    client = TestClient(create_app(repo=repo, metrics_enabled=False))
    assert client.get("/metrics").status_code == 404
    assert client.get("/healthz").status_code == 200
    # 鉴权关闭时 disabled metrics 同样是 404（端点不存在）。
    assert client.get("/metrics").status_code == 404
    # disabled 时也不应保留指标状态，避免误报。
    assert client.app.state.metrics is None


def test_repeated_create_app_does_not_duplicate_metrics(tmp_path: Path) -> None:
    """反复 create_app（每个自带 registry）不会因重复注册而报错。"""

    from fastapi.testclient import TestClient

    from whisper_stt_service.api.app import create_app

    repo = _real_repo(tmp_path)
    for _ in range(3):
        client = TestClient(create_app(repo=repo))
        assert client.get("/metrics").status_code == 200


# --------------------------------------------------------------- queue summary


def test_queue_summary_throughput_reports_recent_completions(tmp_path: Path) -> None:
    """throughput 不再为空 dict：按阶段给出最近 5 分钟完成数。"""

    from fastapi.testclient import TestClient

    from whisper_stt_service.api.app import create_app

    repo = _real_repo(tmp_path)
    job = repo.enqueue(str(tmp_path / "a.mp4"), "ja")
    # 直接把 extract 任务标成 1 分钟前完成，模拟窗口内的完成记录。
    recent = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    with repo.db.tx() as conn:
        conn.execute(
            "UPDATE tasks SET status='succeeded', finished_at=? WHERE job_id=? AND stage='extract'",
            (recent, job.job_id),
        )

    client = TestClient(create_app(repo=repo))
    resp = client.get("/queue/summary")
    assert resp.status_code == 200
    throughput = resp.json()["throughput"]
    assert throughput["available"] is True
    assert throughput["window_sec"] == sm.THROUGHPUT_WINDOW_SEC
    assert throughput["stages"]["extract"]["succeeded"] >= 1
    assert throughput["stages"]["extract"]["total"] >= 1
    # 未发生的阶段也有 0 值，结构稳定。
    assert throughput["stages"]["translate"]["total"] == 0


def test_queue_summary_throughput_degrades_when_query_fails(tmp_path: Path) -> None:
    """读库失败时 throughput 标记 available=false，接口仍 200。"""

    from fastapi.testclient import TestClient

    from whisper_stt_service.api.app import create_app

    class _BrokenSummaryRepo(_FakeRepo):
        def __init__(self) -> None:
            """构造读库必然失败的仓储替身。"""

            super().__init__(tmp_path / "missing-dir" / "x.db", broken=True)

        def queue_summary(self) -> dict:
            """返回最小摘要（吞吐查询本身会失败）。"""

            return {"stages": {"extract": {"queued": 0}}}

    client = TestClient(create_app(repo=_BrokenSummaryRepo()))
    resp = client.get("/queue/summary")
    assert resp.status_code == 200
    # 读库抛 sqlite3.Error => 明确降级标记，而不是让 /queue/summary 500。
    assert resp.json()["throughput"]["available"] is False
    assert resp.json()["throughput"]["window_sec"] == sm.THROUGHPUT_WINDOW_SEC


def test_queue_summary_uses_repository_reader_when_available(tmp_path: Path) -> None:
    """仓储提供 list_finished_tasks_since 时优先走该方法（后续仓储接入点）。"""

    from fastapi.testclient import TestClient

    from whisper_stt_service.api.app import create_app

    class _ReaderRepo(_FakeRepo):
        def __init__(self) -> None:
            """记录最近一次查询用的时间窗起点。"""

            super().__init__(tmp_path / "x.db")
            self.since_seen: str | None = None

        def queue_summary(self) -> dict:
            """返回最小队列摘要。"""

            return {"stages": {"stt": {"queued": 1}}}

        def list_finished_tasks_since(self, since_iso: str) -> list[dict]:
            """模拟仓储新增的"窗口内已终态任务"读取接口。"""

            self.since_seen = since_iso
            return [
                {
                    "stage": "stt",
                    "status": "succeeded",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
            ]

    repo = _ReaderRepo()
    client = TestClient(create_app(repo=repo))
    resp = client.get("/queue/summary")
    assert resp.status_code == 200
    assert repo.since_seen
    assert resp.json()["throughput"]["stages"]["stt"]["succeeded"] == 1


# --------------------------------------------------------- translate 埋点接线


def test_translate_runner_records_batch_and_retry_metrics(tmp_path: Path) -> None:
    """批次成功/失败与重试都会写指标（不发起任何真实网络请求）。"""

    import requests

    from whisper_stt_service.executor import common as executor_common
    from whisper_stt_service.executor.translate import (
        SrtEntry,
        TranslateSettings,
        _TranslateRunner,
        _error_class,
    )

    state = build_metrics(stages=("translate",))
    entry = SrtEntry(index=1, timestamp="00:00:00,000 --> 00:00:01,000", text="こんにちは")
    settings = TranslateSettings(
        chunk_minutes=20,
        batch_size=10,
        max_chars=1000,
        parallel=2,
        retry=2,
        timeout_sec=60,
        request_interval=0.0,
        disable_proxy=True,
        copy_back="__video_dir__",
        checkpoint_enabled=False,
        strict_ids=False,
    )
    batches = executor_common.build_translate_batches(
        [entry], window_minutes=settings.chunk_minutes, batch_size=settings.batch_size
    )

    calls = {"count": 0}

    def _flaky_post(*args, **kwargs):
        """模拟连接失败，用于验证重试指标（不发起真实网络请求）。"""

        calls["count"] += 1
        raise requests.ConnectionError("boom")

    runner = _TranslateRunner(
        settings=settings,
        llm_profiles=[{"section": "llm", "base_url": "http://127.0.0.1:9", "api_key": "k", "model": "m"}],
        batches=batches,
        entries=[entry],
        progress_artifact=tmp_path / "out.progress.json",
        output_zh_srt=tmp_path / "out.zh.srt",
        source_sha256="deadbeef",
        job_id="job-x",
        deadline_at=float("inf"),
        request_interval=0.0,
        parallel=2,
        progress_queue=None,
        task_id="task-x",
        worker_id="worker-x",
        cancel_check=None,
        post_func=_flaky_post,
        metrics=state,
    )
    with pytest.raises(Exception):
        runner.execute()
    assert calls["count"] == settings.retry
    # 每次失败（含最后一次）都会进入 retry 分支，因此重试指标 == 尝试次数。
    assert _counter_value(state, "whisper_stt_translate_retries_total") == settings.retry
    assert _counter_value(state, "whisper_stt_translate_batches_total", {"result": "failed"}) == 1
    assert _error_class(requests.ConnectionError("x"))