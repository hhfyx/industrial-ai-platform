"""W13 周三：告警引擎测试——规则、静默期、恢复、通道幂等。

演练三条（任务书 15:40–16:40）的注入形态：
① 人为 sleep → P95 触发；② 注入错误 → 错误率触发；
③ 同类不重复（静默期）+ 解除后收到恢复。
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from app.core.alerts import (
    RULES,
    AlertEngine,
    reset_alert_engine,
)


@pytest.fixture(autouse=True)
def _clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir(parents=True)
    monkeypatch.setenv("TRACE_DIR", str(trace_dir))
    monkeypatch.setenv("ALERT_NOTIFY_ENDPOINT", "http://127.0.0.1:59999/notify")  # 不可达
    reset_alert_engine()
    return trace_dir


def _seed_db(trace_dir: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(trace_dir / "traces.db")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS trace_summary ("
        "trace_id TEXT PRIMARY KEY, root_name TEXT, started_at_ms INTEGER,"
        " duration_ms INTEGER, status TEXT, span_count INTEGER,"
        " generation_count INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER,"
        " cost REAL, request_path TEXT, created_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS metrics_hourly ("
        "hour TEXT, request_path TEXT, kind TEXT, requests INTEGER, errors INTEGER,"
        " degraded INTEGER DEFAULT 0, p50_ms INTEGER, p95_ms INTEGER, p99_ms INTEGER,"
        " prompt_tokens INTEGER, completion_tokens INTEGER, cost REAL,"
        " retry_cost REAL DEFAULT 0, cache_hits INTEGER DEFAULT 0,"
        " cache_saved REAL DEFAULT 0, PRIMARY KEY (hour, request_path, kind))"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS usage_ledger ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, model TEXT, endpoint TEXT,"
        " trace_id TEXT, prompt_tokens INTEGER, completion_tokens INTEGER,"
        " cached_tokens INTEGER, retry INTEGER, request_kind TEXT, cost REAL)"
    )
    return conn


def _seed_search_p95(conn: sqlite3.Connection, p95_ms: int) -> None:
    hour = time.strftime("%Y-%m-%dT%H")
    conn.execute(
        "INSERT OR REPLACE INTO metrics_hourly (hour, request_path, kind, requests,"
        " errors, p50_ms, p95_ms, p99_ms, prompt_tokens, completion_tokens, cost)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (hour, "/api/v2/search", "trace", 10, 0, 5, p95_ms, p95_ms + 5, 0, 0, 0.0),
    )
    conn.commit()


def _seed_errors(conn: sqlite3.Connection, n_error: int, n_ok: int) -> None:
    now_ms = int(time.time() * 1000)
    for i in range(n_error):
        conn.execute(
            "INSERT OR IGNORE INTO trace_summary VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"err-{i}",
                "http.request",
                now_ms + i,
                10,
                "error",
                1,
                0,
                0,
                0,
                0.0,
                "/api/v1/extract",
                time.strftime("%Y-%m-%dT%H:%M:%S"),
            ),
        )
    for i in range(n_ok):
        conn.execute(
            "INSERT OR IGNORE INTO trace_summary VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"ok-{i}",
                "http.request",
                now_ms + 1000 + i,
                10,
                "ok",
                1,
                0,
                0,
                0,
                0.0,
                "/api/v1/extract",
                time.strftime("%Y-%m-%dT%H:%M:%S"),
            ),
        )
    conn.commit()


# ============================================================ 规则清单


def test_rules_have_rationale() -> None:
    """每条规则的阈值必须写明出处（行业默认值禁令的落地形式）。"""
    for rule in RULES:
        assert len(rule.rationale) > 40, f"{rule.name} 的 rationale 太短——没写依据"
        assert (
            "W1" in rule.rationale
            or "alert-thresholds" in rule.rationale
            or "docs" in rule.rationale
            or "W1" in rule.rationale
        )


def test_retrieval_p95_threshold_matches_doc() -> None:
    """阈值 20ms 必须与 alert-thresholds.md §一 的推导一致。"""
    rule = next(r for r in RULES if r.name == "retrieval_p95_high")
    assert rule.threshold == 20.0 and rule.compare == ">"


# ============================================================ 触发与恢复


def _engine(trace_dir: Path, **kw) -> AlertEngine:
    return AlertEngine(trace_dir, **kw)


def test_p95_breach_fires_then_recovers(_clean: Path) -> None:
    """演练①③合体：慢请求 → 触发；解除 → 恢复通知。"""
    conn = _seed_db(_clean)
    _seed_search_p95(conn, p95_ms=50)  # > 20ms 阈值（模拟人为 sleep 注入）
    conn.close()

    eng = _engine(_clean)
    events = eng.evaluate()
    fired = [e for e in events if e["action"] == "fired" and e["rule"] == "retrieval_p95_high"]
    assert fired, f"应触发 P95 告警，events={events}"
    assert fired[0]["sent"]["delivered"] is False  # 端点不可达，账本仍记录

    # 解除
    conn = _seed_db(_clean)
    _seed_search_p95(conn, p95_ms=8)  # 回到基线内
    conn.close()
    events = eng.evaluate()
    rec = [e for e in events if e["action"] == "recovered" and e["rule"] == "retrieval_p95_high"]
    assert rec, "解除后应收到恢复通知"


def test_error_rate_breach(_clean: Path) -> None:
    """演练②：注入错误 → 错误率告警。"""
    conn = _seed_db(_clean)
    _seed_errors(conn, n_error=3, n_ok=1)  # 75% 错误率
    conn.close()
    eng = _engine(_clean)
    events = eng.evaluate()
    assert any(e["action"] == "fired" and e["rule"] == "extract_error_rate" for e in events)


def test_error_rate_429_not_counted(_clean: Path) -> None:
    """429 不进错误率（W12 判读约定）：错误全被 429 解释时不告警。

    模拟：trace 里有 4 条 error，但限流计数解释了 4 条 → 净错误 0。
    """
    conn = _seed_db(_clean)
    _seed_errors(conn, n_error=4, n_ok=1)
    hour = time.strftime("%Y-%m-%dT%H")
    # 限流计数行（request_path 含 429 惯例名）——把 4 条 error 全解释掉
    conn.execute(
        "INSERT OR REPLACE INTO metrics_hourly (hour, request_path, kind, requests,"
        " errors, p50_ms, p95_ms, p99_ms, prompt_tokens, completion_tokens, cost)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (hour, "rate_limit_429", "trace", 4, 4, 0, 0, 0, 0, 0, 0.0),
    )
    conn.commit()
    conn.close()
    eng = _engine(_clean)
    events = eng.evaluate()
    assert not any(e["action"] == "fired" and e["rule"] == "extract_error_rate" for e in events), (
        f"429 解释掉的错误不该告警，events={events}"
    )


def test_silence_window_no_spam(_clean: Path) -> None:
    """静默期：同类告警在窗口内只发一次（不刷屏）。"""
    conn = _seed_db(_clean)
    _seed_search_p95(conn, p95_ms=50)
    conn.close()

    clock = {"t": 1000.0}
    eng = _engine(_clean, silence_seconds=600)
    eng.now = lambda: clock["t"]

    e1 = eng.evaluate()
    assert any(e["action"] == "fired" for e in e1)
    clock["t"] += 60  # 1 分钟后仍超阈值 → 静默
    e2 = eng.evaluate()
    assert any(e["action"] == "silenced" for e in e2)
    clock["t"] += 700  # 静默期过后仍超阈值 → 再次投递（跨静默期复发）
    e3 = eng.evaluate()
    assert any(e["action"] == "fired" for e in e3)


def test_no_data_no_conclusion(_clean: Path) -> None:
    """空库不评估不下结论（None 跳过）——空数据告警是最恼人的误报。"""
    eng = _engine(_clean)
    assert eng.evaluate() == []


def test_idempotency_key_per_hour(_clean: Path) -> None:
    """幂等键含小时：同一小时同规则只发一次，跨小时复发是新事件。"""
    conn = _seed_db(_clean)
    _seed_search_p95(conn, p95_ms=50)
    conn.close()
    eng = _engine(_clean)
    e1 = eng.evaluate()
    k1 = next(e for e in e1 if e["action"] == "fired")["sent"]["key"]
    assert "alert-retrieval_p95_high-" in k1
    assert k1.endswith(time.strftime("%Y%m%d%H"))
