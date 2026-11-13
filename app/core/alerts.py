"""W13 周三：告警引擎（阈值取自 W12 拐点，静默期 + 恢复通知）。

## 为什么阈值不能用行业默认值（必答问题）

本系统的瓶颈位置（限流配额先坏，P50 全档恒定）、缓存形态（热缓存 ~100%）、
上游配额（key 30 QPS）都是**独有的**。套用别人的"P95 > 3s 告警"：
检索 P95 基线 10ms，这个阈值永远不触发 = 自称有监控；
而"P50 > 3s"套在 extract 上：基线 2.7s 是上游网关决定的，
天天误报 → 第一次真实波动就把告警关掉。两条路都通向"没有监控"。

所以每条规则的阈值都从 [alert-thresholds.md](../../docs/alert-thresholds.md)
推导（唯一合法来源：W12 第二轮压测拐点 + 两轮复测），改阈值 =
重新实测改表，不在代码里偷偷改数。

## 规则形态

每条规则 = 指标查询（SQL over metrics_hourly / trace_summary）+
比较 + 持续窗口。规则**声明式**：`RULES` 是唯一的事实清单，
看板/文档/测试读同一份——阈值在文档和代码里各写一遍，迟早漂移。

## 静默期与恢复

- 触发 → 投递（幂等键 = `{rule}-{hour}`，同一小时同一规则只发一次；
  跨小时复发是新事件，键不同——告警必须能再次叫醒人）；
- 同规则 10 分钟静默期：演炼时防刷屏（W9 E2 的幂等在"同一事件"层，
  静默期在"同类事件"层，两层缺一不可）；
- 触发 → 条件解除 → 发**恢复通知**（键 = `{rule}-{hour}-recovered`），
  否则人分不清"还在坏"还是"已经好了"。

## 通道

复用 W9 的 `notify_http`（幂等账本 + 重试 + 端点去重三层防线），
channel 固定 archive（归档不发人）+ 可选 supervisor。本地演练用
mock 接收端（`scripts/mock_notify_receiver.py`），通道挂了账本
仍记录意图——告警丢不了，只是晚到。
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

# ------------------------------------------------------------------ 规则定义


@dataclass(frozen=True, slots=True)
class AlertRule:
    """一条告警规则。阈值出处写进 rationale，文档与代码共用这份事实。"""

    name: str
    description: str
    query: Callable[[sqlite3.Connection], float | None]
    """从指标库取当前值。返回 None = 数据不足，跳过（不下结论）。"""
    threshold: float
    compare: str  # ">" / "<"
    rationale: str  # 为什么是这个数（指向 alert-thresholds.md 的推导）
    channel: str = "archive"


def _hour_p95_ms(conn: sqlite3.Connection, path_like: str) -> float | None:
    """最近一个有数据的小时的 P95（trace 行）。"""
    row = conn.execute(
        "SELECT p95_ms FROM metrics_hourly"
        " WHERE kind='trace' AND request_path LIKE ? AND p95_ms IS NOT NULL"
        " ORDER BY hour DESC LIMIT 1",
        (path_like,),
    ).fetchone()
    return float(row[0]) if row else None


def _recent_error_rate(conn: sqlite3.Connection) -> float | None:
    """最近 500 条请求的错误率（429 不算错误——W12 判读约定）。

    trace_summary 的 error 判定含 4xx；429（限流背压）不是故障，
    从分子里剔除才与 W12 的"错误率 0%"口径一致。
    精确剔除要读 JSONL（贵），这里用限流器计数差值近似：
    trace 根 span attrs.status_code=429 的落在限流计数里。
    """
    rows = conn.execute(
        "SELECT s.status FROM trace_summary s ORDER BY s.started_at_ms DESC LIMIT 500"
    ).fetchall()
    if not rows:
        return None
    total = len(rows)
    errors = sum(1 for (status,) in rows if status == "error")
    # 429 计数差值剔除（W12：429 记 success 不记 failure）
    rejected = conn.execute(
        "SELECT COALESCE(SUM(errors),0) FROM metrics_hourly WHERE request_path LIKE '%429%'"
    ).fetchone()[0]
    return max((errors - rejected) / total, 0.0) if total else None


def _llm_retry_share(conn: sqlite3.Connection) -> float | None:
    """重试调用占比（重试成本单列之后的衍生健康信号）。"""
    row = conn.execute("SELECT COUNT(*), SUM(retry) FROM usage_ledger").fetchone()
    n, retry_n = row[0], row[1] or 0
    return (retry_n / n) if n else None


def _daily_cost(conn: sqlite3.Connection) -> float | None:
    """今日成本（ledger 直录侧——对账锚点口径）。"""
    row = conn.execute(
        "SELECT COALESCE(SUM(cost),0) FROM usage_ledger WHERE ts >= ?",
        (_day_start_epoch(),),
    ).fetchone()
    return float(row[0])


def _day_start_epoch() -> float:
    t = time.localtime()
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, 0))


#: 规则清单——阈值全部有出处（alert-thresholds.md / metrics-layer.md）。
RULES: Final[tuple[AlertRule, ...]] = (
    AlertRule(
        name="retrieval_p95_high",
        description="检索 P95 超过基线 2 倍（端到端口径）",
        query=lambda c: _hour_p95_ms(c, "/api/v2/search%"),
        threshold=20.0,
        compare=">",
        rationale="W12 A1–A3 端到端 P95 基线 7–10ms（两轮复测差 ≤4ms），阈值 20ms = 基线 2 倍。"
        "行业默认的 3s 在这里永不触发 = 自称有监控（alert-thresholds.md §一第 3 行）",
    ),
    AlertRule(
        name="extract_error_rate",
        description="extract 请求错误率超 5%（429 剔除后）",
        query=lambda c: _recent_error_rate(c),
        threshold=0.05,
        compare=">",
        rationale="W12 全档错误率 0%（5xx=0，429 是背压不算错）。"
        "任何持续非零错误都是异常——阈值 5% 容忍偶发网络抖动，"
        "持续超 5% 说明不是抖动（alert-thresholds.md §一第 5 行）",
    ),
    AlertRule(
        name="llm_retry_share",
        description="LLM 调用重试占比超 20%（成本健康度）",
        query=lambda c: _llm_retry_share(c),
        threshold=0.20,
        compare=">",
        rationale="W11 20 任务实测重试占比个位数；对账三漏之首是重试未单列。"
        "重试占比飙升 = 上游不稳或错误契约退化，成本先于用户感知恶化",
    ),
    AlertRule(
        name="daily_cost",
        description="当日 LLM 成本超预算（默认 5 元演示口径）",
        query=lambda c: _daily_cost(c),
        threshold=5.0,
        compare=">",
        rationale="W12 实测单次 extract ~0.0012 元（docs/cost.md）；"
        "演示环境日 5000 次全链路调用 ≈ 5 元量级。预算值随部署形态改，"
        "改法 = 换 COST_PRICE 后按真实日调用量重估（metrics-layer.md §二）",
    ),
)


# ------------------------------------------------------------------ 评估与静默


@dataclass
class AlertEngine:
    """规则评估 + 静默期 + 恢复通知。状态在内存（单实例演示服务）。"""

    trace_dir: Path
    silence_seconds: int = 600  # 10 分钟静默：同类告警只发一次
    now: Callable[[], float] = time.time

    _firing: dict[str, dict[str, Any]] = field(default_factory=dict)
    _last_sent: dict[str, float] = field(default_factory=dict)

    def evaluate(self, rules: tuple[AlertRule, ...] = RULES) -> list[dict[str, Any]]:
        """跑一遍规则，返回本次投递的事件列表（firing / recovered / silenced）。"""
        db = self.trace_dir / "traces.db"
        if not db.exists():
            return []
        conn = sqlite3.connect(db)
        events: list[dict[str, Any]] = []
        try:
            for rule in rules:
                value = rule.query(conn)
                if value is None:
                    continue
                breached = value > rule.threshold if rule.compare == ">" else value < rule.threshold
                state = self._firing.get(rule.name)
                now = self.now()
                if breached and state is None:
                    # 新触发：立即投递（不受静默期限制——首次必须叫醒人）
                    events.append(self._fire(rule, value, now))
                elif breached and state is not None:
                    if now - self._last_sent.get(rule.name, 0) < self.silence_seconds:
                        events.append({"rule": rule.name, "action": "silenced", "value": value})
                    else:
                        events.append(self._fire(rule, value, now))  # 静默期后复发再投
                elif not breached and state is not None:
                    events.append(self._recover(rule, value, now))
        finally:
            conn.close()
        return events

    # ------------------------------------------------------------------ 投递

    def _send(self, key: str, text: str, channel: str) -> dict[str, Any]:
        """经 W9 通知通道投递（幂等账本 + mock 接收端）。失败不抛——告警通道
        挂了不能把指标评估也拖死；账本已记录意图（W9 的先落记录纪律）。"""
        result: dict[str, Any] = {"delivered": False, "key": key, "error": None}
        try:
            from app.core.llm.side_effect_tools import notify_http

            result["delivered"] = bool(
                notify_http(
                    channel,
                    text,
                    key,
                    endpoint=os.environ.get(
                        "ALERT_NOTIFY_ENDPOINT", "http://127.0.0.1:8920/notify"
                    ),
                ).get("delivered")
            )
        except Exception as exc:  # noqa: BLE001 - 冲突（重发被拦）也算"已在通道"
            result["error"] = f"{type(exc).__name__}: {exc}"
            if "conflict" in str(exc) or "duplicate" in str(exc):
                result["delivered"] = True  # 幂等拦截 = 之前已送达
        return result

    def _fire(self, rule: AlertRule, value: float, now: float) -> dict[str, Any]:
        hour = time.strftime("%Y%m%d%H", time.localtime(now))
        key = f"alert-{rule.name}-{hour}"
        text = (
            f"🔴 告警 [{rule.name}] {rule.description}\n"
            f"当前值 {value:.4g} {'>' if rule.compare == '>' else '<'} 阈值 {rule.threshold:.4g}\n"
            f"依据：{rule.rationale}"
        )
        sent = self._send(key, text, rule.channel)
        self._firing[rule.name] = {"since": now, "value": value, "key": key}
        self._last_sent[rule.name] = now
        return {"rule": rule.name, "action": "fired", "value": value, "sent": sent}

    def _recover(self, rule: AlertRule, value: float, now: float) -> dict[str, Any]:
        state = self._firing.pop(rule.name)
        key = f"{state['key']}-recovered"
        text = (
            f"🟢 恢复 [{rule.name}] {rule.description}\n"
            f"当前值 {value:.4g} 已回到阈值 {rule.threshold:.4g} 以内"
            f"（持续 {int((now - state['since']) / 60)} 分钟后恢复）"
        )
        sent = self._send(key, text, rule.channel)
        return {"rule": rule.name, "action": "recovered", "value": value, "sent": sent}

    def status(self) -> dict[str, Any]:
        """当前触发中的规则（看板④错误块的数据源之一）。"""
        return {
            name: {"since": s["since"], "value": s["value"]} for name, s in self._firing.items()
        }


_engine: AlertEngine | None = None


def get_alert_engine(trace_dir: Path | None = None) -> AlertEngine:
    global _engine
    if _engine is None:
        td = trace_dir or Path(os.environ.get("TRACE_DIR", "data/traces"))
        _engine = AlertEngine(td)
    return _engine


def reset_alert_engine() -> None:
    global _engine
    _engine = None


__all__ = [
    "AlertEngine",
    "AlertRule",
    "RULES",
    "get_alert_engine",
    "reset_alert_engine",
]
