"""W13 周二：指标层与成本核算（三层指标 + 对账 + 落库）。

## 三层指标清单（定版，见 docs/metrics-layer.md 的映射表）

| 层 | 指标 | 数据来源（不新造，全部复用现有埋点） |
| --- | --- | --- |
| 流量 | QPS / 并发 / 队列深度 / 请求总数 | trace_summary 按小时聚合 + GenerationQueue 深度（W12） |
| 质量 | P50/P95/P99 / TTFT / 错误率 / **429 与 5xx 分开** / 降级率 | trace_summary.duration_ms 分位 + http span status_code + registry.degraded |
| 成本 | token/请求 / 日成本 / 缓存命中率 / 每任务成本 / **重试成本单列** | generation span 的 usage + cost（周一已埋） |

**口径纪律（与 W12 相同）**：429 是限流背压不是故障，错误率只算 5xx +
应用错误；延迟与错误率成对报；质量层的一切分位数注明"端到端"还是
"服务端 elapsed"。

## 成本核算（对齐 W2 D1 与 docs/cost.md 的单价锚点）

单价：输入 2 元 / 输出 8 元每百万 token（`scripts/cost_model.json` 的
示例锚点，**不是任何供应商报价**；换供应商改 `COST_PRICE_IN/OUT` 环境变量）。

三层成本都在这一处算，杜绝"两个模块各算一遍口径漂移"：

1. **generation span**（调用级）：周一的 `record_usage` 已算 `attrs.cost`；
2. **请求级**：trace flush 时累加全部 generation（tracing._flush 已做）；
3. **聚合级**（本模块）：按小时/日/接口/任务类型滚入 SQLite。

**缓存命中按"省下的钱"计**：命中 = 这次没调模型 = 省一次等价调用。
省多少按"同接口未命中成本的中位数"估（运行时滚动维护），**绝不拍脑袋
写死一个单价**——估计口径随流量形态自动更新。

**重试成本单独列**：generation span 带 `retry=True`（HTTP 重试 /
校验回灌的第二轮起）时计入 `retry_cost` 列——对账时它是最常见的
低估源（W2 C3 的教训），必须在聚合里一眼可见。

## 对账（本周关键，必答问题的落点）

对账是所有成本数字的唯一真值锚点。**不对账，你会漏掉三类低估**：
重试（每次都计费）、judge/guard 调用（decompose 的 plan_tokens）、
流式 usage 未开（tok 数记 0 但钱照烧）。

本网关（8326）**没有账单汇总端点**（探测过 /v1/usage 等全部 404），
所以对账锚点取"逐笔 usage 直录"：LLM 调用层每次拿到的
`response.usage` 原样落 `usage_ledger` 表（含 cached_tokens），
对账 = `trace_summary` 聚合侧 vs `usage_ledger` 直录侧逐 token 比对。
误差 >5% 触发排查（表见 docs/metrics-layer.md §四：三类漏记各有
特征 signature）。

## 落库（与周一的 trace 库同一文件、两张独立表）

- `usage_ledger`：直录逐笔（对账真值锚点，只 INSERT 不改）；
- `metrics_hourly`：小时聚合（面板查询用）；
- 周一的 `trace_summary`：trace 级聚合——**三层分离**，聚合坏了
  可以从 ledger 重建，原始事实永远在。
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

# ------------------------------------------------------------------ 价格

DEFAULT_PRICE_IN_PER_1M: Final[float] = 2.0
DEFAULT_PRICE_OUT_PER_1M: Final[float] = 8.0
"""docs/cost.md 的示例锚点（W2 D1）。**不是任何供应商的报价**。"""


def price_in_per_1m() -> float:
    try:
        return float(os.environ.get("COST_PRICE_IN", DEFAULT_PRICE_IN_PER_1M))
    except ValueError:
        return DEFAULT_PRICE_IN_PER_1M


def price_out_per_1m() -> float:
    try:
        return float(os.environ.get("COST_PRICE_OUT", DEFAULT_PRICE_OUT_PER_1M))
    except ValueError:
        return DEFAULT_PRICE_OUT_PER_1M


def compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """单次调用成本（元）。唯一的成本公式，别处不许再写一遍。"""
    return round(
        prompt_tokens / 1_000_000 * price_in_per_1m()
        + completion_tokens / 1_000_000 * price_out_per_1m(),
        8,
    )


# ------------------------------------------------------------------ 直录账本


@dataclass(slots=True)
class UsageRecord:
    """一次 LLM 调用的直录（对账真值锚点）。

    `retry` 标记"这次调用是不是某次逻辑请求的第二轮起"——HTTP 重试与
    校验回灌都算。重试单独计数是对账的第一道防线：它们和正常调用
    形状完全一样，不标记就会混进正常成本里被忘掉。
    """

    ts: float
    model: str
    endpoint: str  # llm.structured / llm.stream / llm.qa_answer / agent.model_step
    trace_id: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    retry: bool = False
    request_kind: str = ""  # extract / summarize / chat / agent_task / qa ...


class UsageLedger:
    """逐笔直录 + 查询。**只 INSERT**：对账锚点的语义是"发生过的事实"，
    聚合层算错可以重算，账本改过一次就再也不是锚点了。"""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path  # None = 跟随 trace 目录（_db_path 解析时定）
        self._lock = threading.Lock()
        self._mem: list[UsageRecord] = []
        self._saved = 0

    def record(self, rec: UsageRecord) -> None:
        with self._lock:
            self._mem.append(rec)

    def flush_to(self, trace_dir: Path) -> int:
        """把内存中的直录写进 SQLite（与 trace 库同一文件）。返回写入条数。"""
        with self._lock:
            pending, self._mem = self._mem, []
        if not pending:
            return 0
        trace_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(trace_dir / "traces.db")
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL, model TEXT, endpoint TEXT, trace_id TEXT,
                    prompt_tokens INTEGER, completion_tokens INTEGER,
                    cached_tokens INTEGER, retry INTEGER, request_kind TEXT,
                    cost REAL
                )
                """
            )
            rows = [
                (
                    r.ts,
                    r.model,
                    r.endpoint,
                    r.trace_id,
                    r.prompt_tokens,
                    r.completion_tokens,
                    r.cached_tokens,
                    int(r.retry),
                    r.request_kind,
                    compute_cost(r.prompt_tokens, r.completion_tokens),
                )
                for r in pending
            ]
            conn.executemany(
                "INSERT INTO usage_ledger (ts, model, endpoint, trace_id, prompt_tokens,"
                " completion_tokens, cached_tokens, retry, request_kind, cost)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()
        with self._lock:
            self._saved += len(rows)
        return len(rows)

    def totals(self, trace_dir: Path, *, retries_only: bool = False) -> dict[str, Any]:
        """直录侧汇总（对账的右侧数字）。"""
        conn = sqlite3.connect(trace_dir / "traces.db")
        try:
            where = "WHERE retry=1" if retries_only else ""
            row = conn.execute(
                f"SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0),"
                f" COALESCE(SUM(completion_tokens),0), COALESCE(SUM(cost),0),"
                f" COALESCE(SUM(cached_tokens),0) FROM usage_ledger {where}"
            ).fetchone()
            n, pin, pout, cost, cached = row
            retry_row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(cost),0) FROM usage_ledger WHERE retry=1"
            ).fetchone()
            return {
                "calls": n,
                "prompt_tokens": pin,
                "completion_tokens": pout,
                "cached_tokens": cached,
                "cost": round(cost, 6),
                "retry_calls": retry_row[0],
                "retry_cost": round(retry_row[1], 6),
            }
        finally:
            conn.close()


_ledger: UsageLedger | None = None
_ledger_lock = threading.Lock()


def get_usage_ledger() -> UsageLedger:
    global _ledger
    with _ledger_lock:
        if _ledger is None:
            _ledger = UsageLedger()
        return _ledger


def reset_usage_ledger() -> None:
    """测试隔离用。"""
    global _ledger
    with _ledger_lock:
        _ledger = None


# ------------------------------------------------------------------ 直录挂点


def observe_call_usage(
    *,
    endpoint: str,
    trace_id: str | None = None,
    request_kind: str = "",
    retry: bool = False,
    response: Any = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """LLM 调用层的统一直录口。三个入口各加一行，漏一格 = 对账缺口。

    usage 优先从响应读（网关的真值）；响应缺 usage 时用显式参数兜底
    并把数字标 0——**缺 usage 本身是对账要抓的信号**，静默补一个
    估计值等于把缺口藏起来。
    """
    cached = 0
    if response is not None:
        u = getattr(response, "usage", None)
        if u is not None:
            prompt_tokens = int(getattr(u, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(u, "completion_tokens", 0) or 0)
            cached = int(
                getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            )
    from app.core.tracing import get_trace_id

    get_usage_ledger().record(
        UsageRecord(
            ts=time.time(),
            model="",
            endpoint=endpoint,
            trace_id=trace_id or get_trace_id(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached,
            retry=retry,
            request_kind=request_kind,
        )
    )


# ------------------------------------------------------------------ 缓存省钱口径


class _CostBaseline:
    """滚动维护"同接口未命中成本中位数"，缓存省钱按它估。

    为什么不用单价 ×(0)：命中的请求省的不只是 token 钱，还有一次
    调用的方差；为什么不用全局均值：extract 3 分钱、Agent 5 分钱，
    混在一起算会把命中率的价值估歪。按 endpoint 分桶滚动，样本
    不足（<5）时不下结论（返 0），诚实优于精确的错。
    """

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self._by_endpoint: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def observe(self, endpoint: str, cost: float) -> None:
        if cost <= 0:
            return
        with self._lock:
            bucket = self._by_endpoint.setdefault(endpoint, [])
            bucket.append(cost)
            if len(bucket) > self.window:
                del bucket[: len(bucket) - self.window]

    def saved_of(self, endpoint: str) -> float:
        with self._lock:
            bucket = sorted(self._by_endpoint.get(endpoint, []))
        if len(bucket) < 5:
            return 0.0
        return bucket[len(bucket) // 2]


_baseline = _CostBaseline()


def observe_call_cost(endpoint: str, cost: float) -> None:
    """真实调用后登记成本（_request / stream_chat 的直录点顺带调）。"""
    _baseline.observe(endpoint, cost)


def cache_saved_estimate(endpoint: str) -> float:
    """一次缓存命中"省了多少钱"的运行时估计（同接口未命中成本中位数）。"""
    return _baseline.saved_of(endpoint)


# ------------------------------------------------------------------ 小时聚合


def aggregate_hourly(trace_dir: Path) -> int:
    """从 trace_summary + usage_ledger 滚出 metrics_hourly。返回小时行数。

    幂等（INSERT OR REPLACE，主键 = hour+接口+类型），重跑不翻倍——
    聚合层坏了随时从明细重建，这是"原始表与聚合表分离"的意义。
    """
    conn = sqlite3.connect(trace_dir / "traces.db")
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics_hourly (
                hour TEXT,               -- YYYY-MM-DDTHH
                request_path TEXT,       -- 接口（root_name 的 path / root 名）
                kind TEXT,               -- 'trace' 行 / 'llm' 行
                requests INTEGER, errors INTEGER, degraded INTEGER DEFAULT 0,
                p50_ms INTEGER, p95_ms INTEGER, p99_ms INTEGER,
                prompt_tokens INTEGER, completion_tokens INTEGER,
                cost REAL, retry_cost REAL DEFAULT 0,
                cache_hits INTEGER DEFAULT 0, cache_saved REAL DEFAULT 0,
                PRIMARY KEY (hour, request_path, kind)
            )
            """
        )
        # ---- trace 侧：流量 + 质量（分位数在 SQL 里排好后 Python 取位）----
        rows = conn.execute(
            "SELECT started_at_ms, request_path, status, duration_ms,"
            " prompt_tokens, completion_tokens, cost FROM trace_summary"
        ).fetchall()
        bucket: dict[tuple[str, str], dict[str, Any]] = {}
        for started_ms, path, status, dur, pin, pout, cost in rows:
            hour = time.strftime("%Y-%m-%dT%H", time.localtime(started_ms / 1000))
            key = (hour, path)
            b = bucket.setdefault(
                key,
                {
                    "requests": 0,
                    "errors": 0,
                    "durs": [],
                    "pin": 0,
                    "pout": 0,
                    "cost": 0.0,
                },
            )
            b["requests"] += 1
            if status == "error":
                b["errors"] += 1
            b["durs"].append(dur)
            b["pin"] += pin
            b["pout"] += pout
            b["cost"] += cost

        def _pct(sorted_list: list[int], p: float) -> int:
            if not sorted_list:
                return 0
            idx = min(int(len(sorted_list) * p), len(sorted_list) - 1)
            return sorted_list[idx]

        for (hour, path), b in bucket.items():
            durs = sorted(b["durs"])
            conn.execute(
                "INSERT OR REPLACE INTO metrics_hourly (hour, request_path, kind,"
                " requests, errors, p50_ms, p95_ms, p99_ms,"
                " prompt_tokens, completion_tokens, cost) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    hour,
                    path,
                    "trace",
                    b["requests"],
                    b["errors"],
                    _pct(durs, 0.50),
                    _pct(durs, 0.95),
                    _pct(durs, 0.99),
                    b["pin"],
                    b["pout"],
                    round(b["cost"], 6),
                ),
            )

        # ---- llm 侧：成本 + 重试单列 + 缓存省钱 ----
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, model TEXT, endpoint TEXT, trace_id TEXT,
                prompt_tokens INTEGER, completion_tokens INTEGER,
                cached_tokens INTEGER, retry INTEGER, request_kind TEXT,
                cost REAL
            )
            """
        )
        llm_rows = conn.execute(
            "SELECT ts, endpoint, prompt_tokens, completion_tokens, cost, retry,"
            " cached_tokens FROM usage_ledger"
        ).fetchall()
        lb: dict[tuple[str, str], dict[str, Any]] = {}
        for ts, endpoint, pin, pout, cost, retry, cached in llm_rows:
            hour = time.strftime("%Y-%m-%dT%H", time.localtime(ts))
            key = (hour, endpoint)
            b = lb.setdefault(
                key, {"calls": 0, "pin": 0, "pout": 0, "cost": 0.0, "retry_cost": 0.0, "cached": 0}
            )
            b["calls"] += 1
            b["pin"] += pin
            b["pout"] += pout
            b["cost"] += cost
            if retry:
                b["retry_cost"] += cost
            b["cached"] += cached
        for (hour, endpoint), b in lb.items():
            saved = cache_saved_estimate(endpoint) * b["cached"]
            conn.execute(
                "INSERT OR REPLACE INTO metrics_hourly (hour, request_path, kind,"
                " requests, prompt_tokens, completion_tokens, cost, retry_cost,"
                " cache_saved) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    hour,
                    endpoint,
                    "llm",
                    b["calls"],
                    b["pin"],
                    b["pout"],
                    round(b["cost"], 6),
                    round(b["retry_cost"], 6),
                    round(saved, 6),
                ),
            )
        conn.commit()
        return conn.execute("SELECT COUNT(*) FROM metrics_hourly").fetchone()[0]
    finally:
        conn.close()


def hourly_report(trace_dir: Path, *, hours: int = 24) -> list[dict[str, Any]]:
    """面板查询：最近 N 小时的聚合行。"""
    conn = sqlite3.connect(trace_dir / "traces.db")
    try:
        rows = conn.execute(
            "SELECT hour, request_path, kind, requests, errors, p50_ms, p95_ms, p99_ms,"
            " prompt_tokens, completion_tokens, cost, retry_cost, cache_saved"
            " FROM metrics_hourly ORDER BY hour DESC, request_path LIMIT ?",
            (hours * 20,),
        ).fetchall()
        keys = (
            "hour",
            "request_path",
            "kind",
            "requests",
            "errors",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "prompt_tokens",
            "completion_tokens",
            "cost",
            "retry_cost",
            "cache_saved",
        )
        return [dict(zip(keys, r, strict=True)) for r in rows]
    finally:
        conn.close()


# ------------------------------------------------------------------ 对账


def reconcile(trace_dir: Path) -> dict[str, Any]:
    """对账：trace_summary 聚合侧 vs usage_ledger 直录侧，逐 token 比对。

    误差 >5% 必须查。三类常见漏记的 signature（docs/metrics-layer.md §四）：
    - **漏记重试**：ledger.retry_calls > 0 但聚合侧 retry_cost = 0；
    - **漏记 guard/judge**：聚合侧 generation_count 为 0 但 ledger 有
      `llm.plan` 类直录（decompose 的 plan_tokens 没进 generation span）；
    - **流式 usage 未开**：ledger 侧某 endpoint 的 completion_tokens 全 0
      但 calls > 0（钱烧了没记数）。

    比对前先幂等刷新一次小时聚合——聚合侧的重试成本在 metrics_hourly
    （llm 行），不在 trace_summary（那张表是 trace 级，无重试维度）。
    """
    aggregate_hourly(trace_dir)
    ledger = get_usage_ledger().totals(trace_dir)
    conn = sqlite3.connect(trace_dir / "traces.db")
    try:
        agg = conn.execute(
            "SELECT COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0),"
            " COALESCE(SUM(cost),0) FROM trace_summary"
        ).fetchone()
        agg_retry_cost = conn.execute(
            "SELECT COALESCE(SUM(retry_cost),0) FROM metrics_hourly WHERE kind='llm'"
        ).fetchone()[0]
        gens = conn.execute(
            "SELECT COUNT(*) FROM trace_summary WHERE generation_count > 0"
        ).fetchone()[0]
        total_gen_spans = conn.execute(
            "SELECT COALESCE(SUM(generation_count),0) FROM trace_summary"
        ).fetchone()[0]
    finally:
        conn.close()

    agg_tokens = agg[0] + agg[1]
    ledger_tokens = ledger["prompt_tokens"] + ledger["completion_tokens"]

    def _diff(a: float, b: float) -> float:
        if max(a, b) == 0:
            return 0.0
        return round(abs(a - b) / max(a, b), 4)

    token_diff = _diff(agg_tokens, ledger_tokens)
    cost_diff = _diff(agg[2], ledger["cost"])

    checks = {
        "token_diff": token_diff,
        "cost_diff": cost_diff,
        "within_5pct": token_diff <= 0.05 and cost_diff <= 0.05,
        "signals": [],
    }
    # 三类漏记的自动 signature。
    #
    # 重试信号的判定不是"两侧 retry_cost 相等"（hourly 的 llm 行本来就
    # 从 ledger 滚出来，恒等），而是**交叉验证两个独立维度**：
    # 直录 calls 总数 vs trace 侧 generation 总数——重试与 judge 是
    # "多出来的调用"，若 trace 侧 generation 总数 < 直录总数，说明
    # 有调用没进观测面（观测的 span 被采样丢弃或漏埋）。
    if ledger["retry_calls"] > 0 and total_gen_spans < ledger["calls"]:
        checks["signals"].append(
            f"疑似漏记重试：直录含 {ledger['retry_calls']} 笔重试"
            f"（{ledger['retry_cost']} 元），"
            f"但 trace 侧 generation 总数 {total_gen_spans} < 直录调用数 {ledger['calls']}"
        )
    if ledger["calls"] > 0 and ledger_tokens == 0:
        checks["signals"].append("疑似流式 usage 未开：直录有调用但 token 全 0")
    if gens == 0 and ledger["calls"] > 0:
        checks["signals"].append("疑似漏记 judge/guard：直录有调用但无一条 trace 带 generation")
    checks["ledger"] = ledger
    checks["aggregate"] = {
        "prompt_tokens": agg[0],
        "completion_tokens": agg[1],
        "cost": round(agg[2], 6),
        "retry_cost": round(agg_retry_cost, 6),
        "traces_with_generation": gens,
    }
    return checks


__all__ = [
    "UsageLedger",
    "UsageRecord",
    "aggregate_hourly",
    "cache_saved_estimate",
    "compute_cost",
    "get_usage_ledger",
    "hourly_report",
    "observe_call_cost",
    "observe_call_usage",
    "price_in_per_1m",
    "price_out_per_1m",
    "reconcile",
    "reset_usage_ledger",
]
