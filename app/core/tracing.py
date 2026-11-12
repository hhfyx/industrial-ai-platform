"""W13 周一：三级 trace 模型与全链路观测（自建，不上 Langfuse）。

## 三级模型（必答问题的落点）

    trace        一次请求（trace_id = request_id，见下）
     └ span      一段操作（系统做了什么：rag.embed / agent.tool / http.request）
          └ generation   一次 LLM 调用（模型做了什么：token / ttft / 成本）

**为什么三级而不是两级**：span 描述"系统做了什么"，generation 描述
"模型做了什么"。两者的成本结构（毫秒 vs 秒、0 元 vs 真金白银）、
延迟形态（恒定 vs 长尾）、失败语义（超时重试 vs 降级拒答）完全不同；
合成一级，就无法回答"这次请求贵在检索还是贵在生成"——那正是
W12 分段表的教训在观测层的复刻：分段表告诉你哪段最贵，generation
告诉你贵的那些段里模型占了多少。

## trace_id 为什么等于 request_id

W8 已经把 request_id 贯穿到日志、错误体、响应头三处。trace 再造一个
独立 id 体系，等于同一请求两套编号——报障时对不上的那天，你会先怀疑
日志丢了。**trace_id 复用 request_id**，span 结构用 span_id/parent_id
表达。响应头新增 `X-Trace-Id`（与 `X-Request-ID` 同值），显式回写是
为了让"观测头"不依赖"巧合相等"。

## 为什么自建而不上 Langfuse 自托管

| 方案 | 需要的资源 | 本项目现实 |
| --- | --- | --- |
| Langfuse 自托管 | Postgres + ClickHouse + 两个容器，内存 2GB+ | 本机 11.6GB 已经跑着双 compose + 网关 + bge；再塞一套分析库，观测成了主要负载 |
| Langfuse Cloud | 外网依赖 + 数据出境 | 工业语料不出去；断网即失去观测 |
| **自建（本模块）** | 一个后台写线程 + 一个 SQLite 文件 | 单实例演示服务的观测体量（W12 全档压测 × 全采样 ≈ 每天 MB 级） |

Cloud 保留为**可选对照**：本模块的 span 形状刻意接近 OTel 语义
（trace/span/parent、status、attrs），将来要导出时写一个
JSONL → OTLP 的转换器即可，不用改埋点。

## 今天点名要防的坑：埋点不能同步写盘

压测时同步日志 IO 会把 P95 拉高 30%，然后你会去"优化检索"——
观测成了瓶颈，而你看不出来。所以：

- 请求路径上只做两件事：往内存 list 塞 dict（微秒级）+ `put_nowait`
  进内存队列（微秒级）；
- JSONL / SQLite 落盘在**单条后台守护线程**里做，队列满或线程死了
  就丢弃并计数（`trace_stats().dropped`）——观测可以丢，业务不能等；
- span 内绝不 await、绝不做 IO。

## 采样

- 错误 trace：**100% 保留**（任何 span status=error，或请求 5xx）；
- 正常 trace：`TRACE_SAMPLE_RATE`（默认 1.0=全量，本阶段要测磁盘增长；
  生产建议 0.1–0.2）；
- 判定在**请求结束时**做（此时才有"是否错误"的完整信息），
  期间 spans 待在内存里——单请求几十个 span，量级无害。

## 体积控制

大文本（prompt、observation、正文）不进 span：`clip_text()` 只落
sha256 前 16 位 + 长度 + 前 200 字。完整原文本来就在业务日志/任务表里，
trace 里存它们只会让 JSONL 膨胀到没法 grep。

## 边界（如实）

- `react_loop.py`（W9 评测 harness）未埋点：它不被服务路径调用，
  埋了也只会让评测数字漂移。服务路径的 Agent 走 `agent_graph.py`，
  两边的轨迹字段同源（W11 定版），需要时照 `agent_graph.model_step`
  的两行样式补即可。
- bm25 / rerank / fusion / compress 四层**在线链路默认关闭**（W6 定版），
  它们的 span 只在对应层启用时出现（评测/显式开启）。六个 span 都已
  埋上，可见性跟随链路——给"没跑的层"造假 span 才是观测的谎言。
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import queue
import random
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from app.core.logging import get_request_id

TRACE_ID_HEADER: Final[str] = "X-Trace-Id"

# 当前请求的 trace_id（= request_id）。trace_scope 负责写。
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")


def get_trace_id() -> str:
    return trace_id_var.get()


# ---------------------------------------------------------------- 文本瘦身


def clip_text(value: Any, *, keep: int = 200) -> Any:
    """大字符串只落 hash + 长度 + 前 200 字。

    trace 是"发生了什么"的索引，不是原文仓库。prompt / 观测 / 正文
    这类动辄几千字的东西进 span，JSONL 会在一次压测后大到没法打开。
    原文的权威副本在业务数据里（usage_log / 任务表 / chunk 库），
    span 里的 hash 足够把两边对上。
    """
    if not isinstance(value, str):
        return value
    if len(value) <= keep:
        return value
    return {
        "sha256_16": hashlib.sha256(value.encode("utf-8")).hexdigest()[:16],
        "len": len(value),
        "head": value[:keep],
    }


# ---------------------------------------------------------------- trace 上下文


@dataclass
class _TraceCollector:
    """单条 trace 的内存收集器。请求结束时按采样决定是否落盘。"""

    trace_id: str
    spans: list[dict[str, Any]] = field(default_factory=list)
    stack: list[dict[str, Any]] = field(default_factory=list)


_trace_ctx: contextvars.ContextVar[_TraceCollector | None] = contextvars.ContextVar(
    "trace_collector", default=None
)


def _span_id() -> str:
    return uuid.uuid4().hex[:16]


@contextlib.contextmanager
def trace_scope(
    trace_id: str | None = None,
    *,
    root_name: str,
    root_attrs: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """开一条新 trace（新 collector），块结束时按采样 flush。

    - `trace_id` 不传则取当前上下文的 request_id（中间件已 set）；
      都没有就现场生成——后台任务/CLI 也能独立成 trace。
    - 每次调用都换**新 collector**：后台任务（create_task 拷贝了请求的
      collector 引用）必须用自己的 collector，否则请求侧 flush 之后
      后台再往里写的 span 会丢。
    """
    tid = trace_id or trace_id_var.get() or get_request_id() or uuid.uuid4().hex
    token = trace_id_var.set(tid)
    collector = _TraceCollector(trace_id=tid)
    token_ctx = _trace_ctx.set(collector)
    try:
        with span(root_name, **(root_attrs or {})) as root:
            yield root
    finally:
        _flush(collector)
        _trace_ctx.reset(token_ctx)
        trace_id_var.reset(token)


@contextlib.contextmanager
def span(name: str, kind: str = "span", **attrs: Any) -> Iterator[dict[str, Any]]:
    """一段操作的 span。同步/异步通用（块内无 await、无 IO）。

    `kind` 是记录级字段（span / generation），不是 attrs 的一员；
    其余关键字全部进 attrs。

    yield 出来的 dict 就是 span 记录本体——调用方在执行中途补充 attrs
    （如 agent 步号、HTTP 状态码）直接写 `rec["attrs"][k] = v`。
    异常会把 span 标成 error 再原样上抛（错误也是数据）。
    """
    collector = _trace_ctx.get()
    if collector is None:
        # 不在任何 trace 里（测试直调 / 无 trace 的后台路径）：零开销 no-op。
        yield {"attrs": attrs}
        return

    parent = collector.stack[-1] if collector.stack else None
    rec: dict[str, Any] = {
        "trace_id": collector.trace_id,
        "span_id": _span_id(),
        "parent_id": parent["span_id"] if parent else None,
        "name": name,
        "kind": kind,
        "start_ms": int(time.time() * 1000),
        "duration_ms": 0,
        "status": "ok",
        "attrs": attrs,
    }
    collector.spans.append(rec)
    collector.stack.append(rec)
    t0 = time.perf_counter()
    try:
        yield rec
    except BaseException as exc:  # noqa: BLE001 - GeneratorExit/取消也要留痕
        rec["status"] = "cancelled" if isinstance(exc, GeneratorExit) else "error"
        rec["attrs"]["error"] = f"{type(exc).__name__}"
        raise
    finally:
        rec["duration_ms"] = int((time.perf_counter() - t0) * 1000)
        collector.stack.pop()


@contextlib.contextmanager
def generation_span(name: str, *, model: str, **attrs: Any) -> Iterator[dict[str, Any]]:
    """一次 LLM 调用的 generation span（三级模型的第三级）。

    与普通 span 的差别：kind=generation，约定 attrs 里有
    `model / prompt_tokens / completion_tokens / cost / ttft_ms`。
    非流式调用没有真正的首 token 时刻，`ttft_ms` 记总时长并在
    attrs 里带 `streaming=False` 如实标注——流式的真 ttft 由
    `streaming.py` 在首个内容块到达时写。
    """
    collector = _trace_ctx.get()
    if collector is None:
        yield {"attrs": attrs}
        return
    with span(name, kind="generation", model=model, **attrs) as rec:
        yield rec


def record_usage(rec: dict[str, Any], *, prompt_tokens: int, completion_tokens: int) -> None:
    """把 usage 写进 generation span 并算成本。

    成本公式**只有一个出处**：`metrics.compute_cost`（docs/cost.md 的
    2/8 元每百万锚点）。周一版本曾自带一套 `TRACE_COST_*` 单价（默认 0），
    导致同一笔调用两个模块算出两个价——对账第一天就把它抓出来了
    （trace 侧 cost 全 0、ledger 侧 0.0547，diff=100%）。环境变量
    `COST_PRICE_IN/OUT` 由 metrics 统一读，这里不再重复定义。
    """
    attrs = rec.get("attrs")
    if attrs is None:
        return
    attrs["prompt_tokens"] = prompt_tokens
    attrs["completion_tokens"] = completion_tokens
    try:
        from app.core.metrics import compute_cost

        attrs["cost"] = compute_cost(prompt_tokens, completion_tokens)
    except Exception:  # noqa: BLE001 - 观测失败不影响主路径
        attrs["cost"] = 0.0


def set_ttft(rec: dict[str, Any], ttft_ms: int) -> None:
    attrs = rec.get("attrs")
    if attrs is not None:
        attrs["ttft_ms"] = ttft_ms


# ---------------------------------------------------------------- flush 与采样


def _sample_rate() -> float:
    try:
        return max(0.0, min(1.0, float(os.environ.get("TRACE_SAMPLE_RATE", "1.0"))))
    except ValueError:
        return 1.0


def _flush(collector: _TraceCollector) -> None:
    if not collector.spans:
        return
    has_error = any(s["status"] in ("error", "cancelled") for s in collector.spans)
    if not has_error and random.random() > _sample_rate():
        return  # 正常请求按采样率丢弃；错误/取消的 100% 保留

    root = collector.spans[0]
    gens = [s for s in collector.spans if s["kind"] == "generation"]
    summary = {
        "trace_id": collector.trace_id,
        "root_name": root["name"],
        "started_at_ms": root["start_ms"],
        "duration_ms": sum(s["duration_ms"] for s in collector.spans if s["parent_id"] is None)
        or max(s["duration_ms"] for s in collector.spans),
        "status": "error" if any(s["status"] == "error" for s in collector.spans) else "ok",
        "span_count": len(collector.spans),
        "generation_count": len(gens),
        "prompt_tokens": sum(s["attrs"].get("prompt_tokens", 0) for s in gens),
        "completion_tokens": sum(s["attrs"].get("completion_tokens", 0) for s in gens),
        "cost": round(sum(s["attrs"].get("cost", 0.0) for s in gens), 6),
        "request_path": root["attrs"].get("path", root["name"]),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _sink().submit({"summary": summary, "spans": collector.spans})


# ---------------------------------------------------------------- 后台落盘线程


class _TraceSink:
    """单守护线程写 JSONL + SQLite。观测可以丢，业务不能等。"""

    def __init__(self) -> None:
        self._q: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self.flushed = 0
        self.dropped = 0
        self.write_errors = 0
        self._lock = threading.Lock()

    def submit(self, item: dict[str, Any]) -> None:
        with self._lock:
            self._ensure()
            self._q.put_nowait(item)  # SimpleQueue 无界：内存队列，永不想 put

    def _ensure(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="trace-sink", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            try:
                self._write(item)
                with self._lock:
                    self.flushed += 1
            except Exception:  # noqa: BLE001 - 写失败不能让观测线程死掉
                with self._lock:
                    self.write_errors += 1

    def _write(self, item: dict[str, Any]) -> None:
        trace_dir = Path(os.environ.get("TRACE_DIR", "data/traces"))
        trace_dir.mkdir(parents=True, exist_ok=True)

        # 顺带 flush 直录账本（W13 周二）：对账锚点与 trace 同盘同节奏，
        # 不额外开线程——观测的复杂度预算有限。
        try:
            from app.core.metrics import get_usage_ledger

            get_usage_ledger().flush_to(trace_dir)
        except Exception:  # noqa: BLE001 - 账本 flush 失败不拖垮 trace 落盘
            pass

        # JSONL：全量 span 明细（一行一条，可 grep）
        line = json.dumps(
            {"summary": item["summary"], "spans": item["spans"]},
            ensure_ascii=False,
            default=str,
        )
        with (trace_dir / "traces.jsonl").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        # SQLite：trace 级聚合（面板/查询走这里，明细回 JSONL 捞）
        db = trace_dir / "traces.db"
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trace_summary (
                    trace_id TEXT PRIMARY KEY,
                    root_name TEXT, started_at_ms INTEGER, duration_ms INTEGER,
                    status TEXT, span_count INTEGER, generation_count INTEGER,
                    prompt_tokens INTEGER, completion_tokens INTEGER,
                    cost REAL, request_path TEXT, created_at TEXT
                )
                """
            )
            s = item["summary"]
            conn.execute(
                "INSERT OR REPLACE INTO trace_summary VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    s["trace_id"],
                    s["root_name"],
                    s["started_at_ms"],
                    s["duration_ms"],
                    s["status"],
                    s["span_count"],
                    s["generation_count"],
                    s["prompt_tokens"],
                    s["completion_tokens"],
                    s["cost"],
                    s["request_path"],
                    s["created_at"],
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "flushed": self.flushed,
                "dropped": self.dropped,
                "write_errors": self.write_errors,
                "queue_depth": self._q.qsize(),
                "sample_rate": _sample_rate(),
            }


_sink_instance: _TraceSink | None = None
_sink_lock = threading.Lock()


def _sink() -> _TraceSink:
    global _sink_instance
    with _sink_lock:
        if _sink_instance is None:
            _sink_instance = _TraceSink()
        return _sink_instance


def trace_stats() -> dict[str, int]:
    """观测自身健康度（/metrics 暴露）：丢了多少、积压多少。"""
    return _sink().stats()


__all__ = [
    "TRACE_ID_HEADER",
    "clip_text",
    "generation_span",
    "get_trace_id",
    "record_usage",
    "set_ttft",
    "span",
    "trace_scope",
    "trace_stats",
]
