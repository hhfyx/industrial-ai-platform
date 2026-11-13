"""W13 周三：可观测看板（Streamlit 独立页）。

跑法：
    uv run streamlit run app/ui/observability_board.py

四块（任务书 09:00–10:00）：
① 概览（请求量/错误率/P95/日成本）② 延迟分布（分位时间序列）
③ 成本（按接口/缓存节省/重试成本）④ 错误与告警状态。

**下钻是看板有没有用的分界线**（任务书 10:00–11:20）：任何一格数字
都能点进对应的 trace_id → span 明细。只有聚合没有下钻的看板，
回答"哪里慢"靠猜。

数据源：metrics_hourly / trace_summary / usage_ledger（周二的落库），
读库不写库——看板是观测的消费者，不是第二个真值源。

看板三问（19:00–20:30）：
① 现在哪里慢 → 块② 的分位序列，按接口拆行，最慢的排最前；
② 现在哪里贵 → 块③ 按接口成本排行 + 缓存节省；
③ 现在在失败什么 → 块④ 错误率 + 告警触发状态。
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import streamlit as st


def _trace_dir() -> Path:
    return Path(os.environ.get("TRACE_DIR", "data/traces"))


def _conn() -> sqlite3.Connection | None:
    db = _trace_dir() / "traces.db"
    if not db.exists():
        return None
    conn = sqlite3.connect(db, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _window_hours(window: str) -> int:
    return {"1h": 1, "24h": 24, "7d": 168}[window]


def _hours_list(window: str) -> list[str]:
    """最近 N 个小时的 hour 键（含当前小时）。"""
    now = time.time()
    hours = []
    for i in range(_window_hours(window)):
        hours.append(time.strftime("%Y-%m-%dT%H", time.localtime(now - i * 3600)))
    return hours


# ------------------------------------------------------------------ 查询


def _overview(conn: sqlite3.Connection, window: str) -> dict:
    hours = _hours_list(window)
    marks = ",".join("?" * len(hours))
    row = conn.execute(
        f"SELECT COALESCE(SUM(requests),0), COALESCE(SUM(errors),0),"
        f" COALESCE(SUM(cost),0) FROM metrics_hourly WHERE kind='trace' AND hour IN ({marks})",
        hours,
    ).fetchone()
    p95 = conn.execute(
        f"SELECT MAX(p95_ms) FROM metrics_hourly WHERE kind='trace' AND hour IN ({marks})",
        hours,
    ).fetchone()[0]
    today_cost = conn.execute(
        "SELECT COALESCE(SUM(cost),0) FROM usage_ledger WHERE ts >= ?",
        (time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, 0)),),
    ).fetchone()[0]
    requests, errors, _cost = row
    return {
        "requests": requests,
        "error_rate": (errors / requests) if requests else 0.0,
        "p95_ms": p95 or 0,
        "day_cost": today_cost,
    }


def _latency_series(conn: sqlite3.Connection, window: str) -> list[dict]:
    hours = _hours_list(window)
    marks = ",".join("?" * len(hours))
    rows = conn.execute(
        f"SELECT hour, request_path, p50_ms, p95_ms, p99_ms, requests FROM metrics_hourly"
        f" WHERE kind='trace' AND hour IN ({marks}) ORDER BY hour, request_path",
        hours,
    ).fetchall()
    return [dict(r) for r in rows]


def _cost_by_endpoint(conn: sqlite3.Connection, window: str) -> list[dict]:
    hours = _hours_list(window)
    marks = ",".join("?" * len(hours))
    rows = conn.execute(
        f"SELECT request_path, SUM(cost) cost, SUM(retry_cost) retry_cost,"
        f" SUM(cache_saved) cache_saved, SUM(requests) calls"
        f" FROM metrics_hourly WHERE kind='llm' AND hour IN ({marks})"
        f" GROUP BY request_path ORDER BY cost DESC",
        hours,
    ).fetchall()
    return [dict(r) for r in rows]


def _errors_block(conn: sqlite3.Connection) -> dict:
    """错误明细：最近错误 trace 的路径分布 + 当前告警触发状态。"""
    rows = conn.execute(
        "SELECT request_path, COUNT(*) n FROM trace_summary"
        " WHERE status='error' GROUP BY request_path ORDER BY n DESC LIMIT 8"
    ).fetchall()
    from app.core.alerts import get_alert_engine

    return {"by_path": [dict(r) for r in rows], "firing": get_alert_engine().status()}


def _traces_for_drilldown(
    conn: sqlite3.Connection, hour: str | None, path: str | None
) -> list[dict]:
    """下钻候选：给定小时/接口（可空）的最新 30 条 trace。"""
    sql = "SELECT trace_id, request_path, started_at_ms, duration_ms, status,"
    " span_count, generation_count, prompt_tokens, completion_tokens, cost"
    " FROM trace_summary"
    conds, params = [], []
    if hour:
        conds.append("started_at_ms >= ? AND started_at_ms < ?")
        base = time.mktime(time.strptime(hour, "%Y-%m-%dT%H"))
        params += [base, base + 3600]
    if path:
        conds.append("request_path = ?")
        params.append(path)
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    sql += " ORDER BY started_at_ms DESC LIMIT 30"
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _span_detail(trace_id: str) -> dict | None:
    """从 JSONL 捞一条 trace 的 span 明细（下钻终点）。"""
    f = _trace_dir() / "traces.jsonl"
    if not f.exists():
        return None
    for line in f.read_text(encoding="utf-8").splitlines():
        if f'"trace_id": "{trace_id}"' in line or trace_id in line:
            data = __import__("json").loads(line)
            if data["summary"]["trace_id"] == trace_id:
                return data
    return None


# ------------------------------------------------------------------ 页面


def main() -> None:  # noqa: C901 - 看板页天然是长函数，分块清晰即可
    st.set_page_config(page_title="可观测看板", page_icon="📡", layout="wide")
    st.title("📡 可观测看板")
    st.caption(
        "数据源：metrics_hourly / trace_summary / usage_ledger（对账过的聚合层）· 阈值依据见 docs/alert-thresholds.md"
    )

    conn = _conn()
    if conn is None:
        st.warning(f"还没有指标库：{_trace_dir() / 'traces.db'} 不存在。先打几条请求。")
        return

    window = st.sidebar.radio("时间窗", ["1h", "24h", "7d"], index=1)
    if st.sidebar.button("🔄 刷新"):
        st.rerun()
    st.sidebar.caption(
        "对比说明：本周 vs 上周需要 7d 窗口有跨周数据；"
        "当前窗口只有单周数据时对比列为空（诚实显示，不造假基线）。"
    )

    # ---- 块① 概览
    st.subheader("① 概览")
    ov = _overview(conn, window)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("请求量", f"{ov['requests']:,}")
    c2.metric("错误率", f"{ov['error_rate'] * 100:.1f}%", help="429 已剔除（W12 判读约定）")
    c3.metric("最差 P95", f"{ov['p95_ms']} ms")
    c4.metric("今日成本", f"{ov['day_cost']:.4f} 元")

    # ---- 块② 延迟分布
    st.subheader("② 延迟分布")
    series = _latency_series(conn, window)
    if series:
        chart_data = [
            {
                "hour": r["hour"][11:] + " " + r["request_path"].split("/")[-1],
                "p50": r["p50_ms"],
                "p95": r["p95_ms"],
                "p99": r["p99_ms"],
            }
            for r in series
        ]
        st.bar_chart(chart_data, x="hour", y=["p50", "p95", "p99"], height=260)
        slow = sorted(series, key=lambda r: -(r["p95_ms"] or 0))[:3]
        st.caption(
            "最慢："
            + " · ".join(f"`{r['request_path']}` P95={r['p95_ms']}ms" for r in slow if r["p95_ms"])
        )
    else:
        st.caption("窗口内无数据。")

    # ---- 块③ 成本
    st.subheader("③ 成本")
    costs = _cost_by_endpoint(conn, window)
    if costs:
        for r in costs:
            cols = st.columns([3, 1, 1, 1])
            cols[0].code(r["request_path"])
            cols[1].metric("成本", f"{r['cost']:.4f} 元")
            cols[2].metric("重试", f"{r['retry_cost']:.4f} 元", help="重试成本单列（对账三漏之首）")
            cols[3].metric("缓存省", f"{r['cache_saved']:.4f} 元")
        total = sum(r["cost"] for r in costs)
        saved = sum(r["cache_saved"] for r in costs)
        st.caption(f"窗口合计 {total:.4f} 元；缓存累计省 {saved:.4f} 元（中位数基线估计）")
    else:
        st.caption("窗口内无 LLM 调用。")

    # ---- 块④ 错误与告警
    st.subheader("④ 错误与告警")
    errs = _errors_block(conn)
    if errs["firing"]:
        for name, s in errs["firing"].items():
            st.error(f"🔴 **{name}** 触发中（值 {s['value']:.4g}）")
    else:
        st.success("无触发中的告警")
    if errs["by_path"]:
        for r in errs["by_path"]:
            st.caption(f"`{r['request_path']}` × {r['n']} 条错误")

    st.divider()

    # ---- 下钻（看板有没有用的分界线）
    st.subheader("🔎 下钻：聚合 → trace → span")
    series_hours = sorted({r["hour"] for r in series}, reverse=True)
    all_paths = [
        r["request_path"]
        for r in conn.execute(
            "SELECT DISTINCT request_path FROM trace_summary ORDER BY request_path"
        ).fetchall()
    ]
    dc1, dc2 = st.columns(2)
    sel_hour = dc1.selectbox("小时（可全部）", ["(全部)"] + series_hours)
    sel_path = dc2.selectbox("接口（可全部）", ["(全部)"] + all_paths)
    traces = _traces_for_drilldown(
        conn, None if sel_hour == "(全部)" else sel_hour, None if sel_path == "(全部)" else sel_path
    )
    if not traces:
        st.caption("该筛选下没有 trace。")
        return
    for t in traces:
        label = (
            f"`{t['trace_id'][:8]}` {t['request_path']} · {t['duration_ms']}ms · "
            f"{'🔴' if t['status'] == 'error' else '🟢'} · spans={t['span_count']} gens={t['generation_count']}"
            f" · {t['cost']:.4f} 元"
        )
        with st.expander(label):
            detail = _span_detail(t["trace_id"])
            if detail is None:
                st.caption("span 明细未落盘（采样丢弃或 JSONL 已清理）——聚合行仍在 trace_summary。")
                continue
            for s in detail["spans"]:
                kind_tag = "🤖" if s["kind"] == "generation" else "⚙️"
                st.markdown(
                    f"{kind_tag} **{s['name']}** · {s['duration_ms']}ms · {s['status']}"
                    + (f" · {s['attrs'].get('model')}" if s["attrs"].get("model") else "")
                )
                if s["kind"] == "generation":
                    a = s["attrs"]
                    st.caption(
                        f"　tok {a.get('prompt_tokens', 0)}+{a.get('completion_tokens', 0)}"
                        f" = {a.get('cost', 0):.6f} 元"
                        + (f" · ttft {a['ttft_ms']}ms" if a.get("ttft_ms") else "")
                    )


if __name__ == "__main__":
    main()
