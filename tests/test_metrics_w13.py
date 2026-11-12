"""W13 周二：指标层与成本核算的测试。

覆盖：成本公式、直录账本、缓存省钱口径、小时聚合、对账（含三类
漏记的 signature）、W11/W12 口径对齐的不变量。
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from app.core.metrics import (
    UsageRecord,
    aggregate_hourly,
    compute_cost,
    get_usage_ledger,
    reconcile,
    reset_usage_ledger,
)


@pytest.fixture(autouse=True)
def _clean_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir(parents=True)
    monkeypatch.setenv("TRACE_DIR", str(trace_dir))
    reset_usage_ledger()
    return trace_dir


# ============================================================ 成本公式


def test_cost_formula_matches_price_table() -> None:
    """输入 2 / 输出 8 元每百万（docs/cost.md 锚点）。

    复算 cost.md 单次成本表的"标准合同 · 一次成功"行：
    227 in + 89 out → 227e-6*2 + 89e-6*8 = 0.001166 元
    """
    c = compute_cost(227, 89)
    assert abs(c - 0.001166) < 1e-6


def test_cost_output_heavier_than_input() -> None:
    """同样 1000 token，输出比输入贵 4 倍（W2 D1 的口径不变量）。"""
    assert compute_cost(0, 1000) == pytest.approx(4 * compute_cost(1000, 0))


# ============================================================ 直录账本


def test_ledger_records_and_totals(_clean_ledger: Path) -> None:
    ledger = get_usage_ledger()
    ledger.record(
        UsageRecord(
            ts=time.time(),
            model="m",
            endpoint="llm.structured",
            trace_id="t1",
            prompt_tokens=100,
            completion_tokens=20,
        )
    )
    ledger.record(
        UsageRecord(
            ts=time.time(),
            model="m",
            endpoint="llm.structured",
            trace_id="t1",
            prompt_tokens=100,
            completion_tokens=20,
            retry=True,
        )
    )
    assert ledger.flush_to(_clean_ledger) == 2
    totals = ledger.totals(_clean_ledger)
    assert totals["calls"] == 2
    assert totals["retry_calls"] == 1
    assert totals["prompt_tokens"] == 200
    assert totals["completion_tokens"] == 40
    # 重试成本 = 单笔重试的成本
    assert totals["retry_cost"] == pytest.approx(compute_cost(100, 20))


def test_ledger_is_insert_only(_clean_ledger: Path) -> None:
    """账本只追加：flush 后再 flush 不覆盖旧行（对账锚点不可变性）。"""
    ledger = get_usage_ledger()
    ledger.record(
        UsageRecord(
            ts=time.time(),
            model="m",
            endpoint="e",
            trace_id="t",
            prompt_tokens=1,
            completion_tokens=1,
        )
    )
    ledger.flush_to(_clean_ledger)
    ledger.record(
        UsageRecord(
            ts=time.time(),
            model="m",
            endpoint="e",
            trace_id="t",
            prompt_tokens=1,
            completion_tokens=1,
        )
    )
    ledger.flush_to(_clean_ledger)
    conn = sqlite3.connect(_clean_ledger / "traces.db")
    n = conn.execute("SELECT COUNT(*) FROM usage_ledger").fetchone()[0]
    conn.close()
    assert n == 2


# ============================================================ 小时聚合


def _seed_trace(
    conn: sqlite3.Connection,
    *,
    path: str,
    dur_ms: int,
    error: bool,
    pin: int,
    pout: int,
    cost: float,
    gens: int = 0,
) -> None:
    status = "error" if error else "ok"
    conn.execute(
        "INSERT OR IGNORE INTO trace_summary VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            f"tid-{path}-{dur_ms}-{int(error)}-{time.time_ns()}",
            "http.request",
            int(time.time() * 1000),
            dur_ms,
            status,
            1,
            gens,
            pin,
            pout,
            cost,
            path,
            time.strftime("%Y-%m-%dT%H:%M:%S"),
        ),
    )


def test_aggregate_hourly_metrics(_clean_ledger: Path) -> None:
    conn = sqlite3.connect(_clean_ledger / "traces.db")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS trace_summary ("
        "trace_id TEXT PRIMARY KEY, root_name TEXT, started_at_ms INTEGER,"
        " duration_ms INTEGER, status TEXT, span_count INTEGER,"
        " generation_count INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER,"
        " cost REAL, request_path TEXT, created_at TEXT)"
    )
    for dur, err in [(10, False), (100, False), (500, True), (300, False)]:
        _seed_trace(conn, path="/api/v2/extract", dur_ms=dur, error=err, pin=10, pout=5, cost=0.001)
    conn.commit()
    conn.close()

    # 直录侧
    ledger = get_usage_ledger()
    ledger.record(
        UsageRecord(
            ts=time.time(),
            model="m",
            endpoint="llm.structured",
            trace_id="t",
            prompt_tokens=500,
            completion_tokens=100,
        )
    )
    ledger.record(
        UsageRecord(
            ts=time.time(),
            model="m",
            endpoint="llm.structured",
            trace_id="t",
            prompt_tokens=500,
            completion_tokens=100,
            retry=True,
        )
    )
    ledger.flush_to(_clean_ledger)

    aggregate_hourly(_clean_ledger)

    conn = sqlite3.connect(_clean_ledger / "traces.db")
    rows = conn.execute(
        "SELECT kind, requests, errors, p50_ms, p95_ms, cost FROM metrics_hourly"
        " WHERE kind='trace' AND request_path='/api/v2/extract'"
    ).fetchall()
    assert rows, "应有 trace 聚合行"
    _, requests, errors, p50, p95, _cost = rows[0]
    assert (requests, errors) == (4, 1)  # 流量 + 错误率分开计数
    assert p50 <= p95  # 分位单调

    llm = conn.execute(
        "SELECT requests, prompt_tokens, cost, retry_cost FROM metrics_hourly WHERE kind='llm'"
    ).fetchone()
    assert llm[0] == 2
    assert llm[1] == 1000
    assert llm[3] > 0  # 重试成本单独成列
    conn.close()


def test_aggregate_is_idempotent(_clean_ledger: Path) -> None:
    """聚合幂等：重跑不翻倍（聚合坏了可从明细重建）。"""
    conn = sqlite3.connect(_clean_ledger / "traces.db")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS trace_summary ("
        "trace_id TEXT PRIMARY KEY, root_name TEXT, started_at_ms INTEGER,"
        " duration_ms INTEGER, status TEXT, span_count INTEGER,"
        " generation_count INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER,"
        " cost REAL, request_path TEXT, created_at TEXT)"
    )
    _seed_trace(conn, path="/x", dur_ms=5, error=False, pin=1, pout=1, cost=0.0)
    conn.commit()
    conn.close()
    n1 = aggregate_hourly(_clean_ledger)
    n2 = aggregate_hourly(_clean_ledger)
    assert n1 == n2


# ============================================================ 对账


def _make_trace_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS trace_summary ("
        "trace_id TEXT PRIMARY KEY, root_name TEXT, started_at_ms INTEGER,"
        " duration_ms INTEGER, status TEXT, span_count INTEGER,"
        " generation_count INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER,"
        " cost REAL, request_path TEXT, created_at TEXT)"
    )


def test_reconcile_within_tolerance(_clean_ledger: Path) -> None:
    """聚合侧与直录侧一致 → within_5pct=true（对账通过）。"""
    conn = sqlite3.connect(_clean_ledger / "traces.db")
    _make_trace_table(conn)
    _seed_trace(
        conn,
        path="/x",
        dur_ms=5,
        error=False,
        pin=500,
        pout=100,
        cost=compute_cost(500, 100),
        gens=1,
    )
    conn.commit()
    conn.close()
    ledger = get_usage_ledger()
    ledger.record(
        UsageRecord(
            ts=time.time(),
            model="m",
            endpoint="e",
            trace_id="t",
            prompt_tokens=500,
            completion_tokens=100,
        )
    )
    ledger.flush_to(_clean_ledger)

    r = reconcile(_clean_ledger)
    assert r["within_5pct"] is True
    assert r["token_diff"] == 0.0
    assert not r["signals"]


def test_reconcile_catches_missing_generation(_clean_ledger: Path) -> None:
    """漏记 judge/guard 的 signature：直录有调用、trace 全部 0 generation。"""
    conn = sqlite3.connect(_clean_ledger / "traces.db")
    _make_trace_table(conn)
    _seed_trace(conn, path="/x", dur_ms=5, error=False, pin=0, pout=0, cost=0.0, gens=0)
    conn.commit()
    conn.close()
    ledger = get_usage_ledger()
    ledger.record(
        UsageRecord(
            ts=time.time(),
            model="m",
            endpoint="llm.plan_decompose",
            trace_id="t",
            prompt_tokens=800,
            completion_tokens=50,
        )
    )
    ledger.flush_to(_clean_ledger)

    r = reconcile(_clean_ledger)
    assert r["within_5pct"] is False
    assert any("judge/guard" in s for s in r["signals"])


def test_reconcile_catches_retry_not_in_aggregate(_clean_ledger: Path) -> None:
    """漏记重试的 signature：直录含重试、但 trace 侧 generation 总数小于直录调用数。"""
    conn = sqlite3.connect(_clean_ledger / "traces.db")
    _make_trace_table(conn)
    _seed_trace(
        conn, path="/x", dur_ms=5, error=False, pin=100, pout=20, cost=compute_cost(100, 20), gens=0
    )  # trace 没看到 generation
    conn.commit()
    conn.close()
    ledger = get_usage_ledger()
    ledger.record(
        UsageRecord(
            ts=time.time(),
            model="m",
            endpoint="e",
            trace_id="t",
            prompt_tokens=100,
            completion_tokens=20,
        )
    )
    ledger.record(
        UsageRecord(
            ts=time.time(),
            model="m",
            endpoint="e",
            trace_id="t",
            prompt_tokens=100,
            completion_tokens=20,
            retry=True,
        )
    )
    ledger.flush_to(_clean_ledger)

    r = reconcile(_clean_ledger)
    # token 差 120/240 = 50% → 超差
    assert r["within_5pct"] is False
    assert any("重试" in s for s in r["signals"])


def test_reconcile_catches_stream_usage_missing(_clean_ledger: Path) -> None:
    """流式 usage 未开的 signature：直录有调用但 token 全 0。"""
    conn = sqlite3.connect(_clean_ledger / "traces.db")
    _make_trace_table(conn)
    _seed_trace(conn, path="/x", dur_ms=5, error=False, pin=0, pout=0, cost=0.0)
    conn.commit()
    conn.close()
    ledger = get_usage_ledger()
    # 模拟网关不回 usage：显式 0 直录（钱烧了没记数）
    ledger.record(
        UsageRecord(
            ts=time.time(),
            model="m",
            endpoint="llm.stream",
            trace_id="t",
            prompt_tokens=0,
            completion_tokens=0,
        )
    )
    ledger.flush_to(_clean_ledger)

    r = reconcile(_clean_ledger)
    assert any("usage" in s for s in r["signals"])


# ============================================================ HTTP 视图


@pytest.mark.anyio
async def test_metrics_cost_endpoint_no_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    from app.main import app

    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "empty"))
    reset_usage_ledger()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        resp = await ac.get("/metrics/cost")
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_data"
