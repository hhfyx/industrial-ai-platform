"""W10 周日 09:00–09:50 · 图执行进度查询（W12 服务化的接口）。

把 get_state(config) 包成一层稳定的查询面：调用方（W12 的任务状态接口）
不需要懂 LangGraph 的 StateSnapshot 结构，拿到的是四个业务字段：

- current_node：当前停在哪个节点（挂起/中断时是「下一步要跑的节点」）
- steps_done：已完成步数
- waiting_human：是否正等人工批复（review 节点挂起中）
- partial_result：已有部分结果（ stopped 且非 completed 时的 final_answer）

CLI（对周日 HITL 演示库直接可用）：

    .\\.venv\\Scripts\\python.exe -X utf8 scripts\\query_graph_state.py --db learn/llm/w10_hitl.sqlite --graph hitl --thread hitl::task-approve

比 W8 任务状态更细：W8 只有 queued/running/done；这里能回答
「卡在哪个节点、第几步、要不要人来批、半成品是什么」。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]


def query_progress(app: Any, config: dict) -> dict:
    """从一个已编译图的 checkpointer 里读执行进度（纯查询，不推进执行）。"""
    snap = app.get_state(config)
    waiting_human = False
    for task in getattr(snap, "tasks", ()) or ():
        if getattr(task, "interrupts", None):
            waiting_human = True
    values = snap.values or {}
    stopped = values.get("stopped_reason", "completed")
    partial = None
    if stopped not in ("completed",) and values.get("final_answer"):
        partial = values["final_answer"]
    if waiting_human:
        partial = values.get("draft")  # 挂起等批复时，半成品就是待批的草稿
    return {
        "current_node": list(snap.next) if snap.next else [],
        "steps_done": values.get("step_no", 0) if "step_no" in values else _count_steps(values),
        "waiting_human": waiting_human,
        "partial_result": partial,
        "values_keys": sorted(values),
    }


def _count_steps(values: dict) -> int:
    """HITL 演示图没有 step_no 计数，用轨迹长度兜底。"""
    outcomes = values.get("outcomes") or []
    return len(outcomes) if outcomes else (1 if values.get("draft") else 0)


# ============================================================ CLI


class _HitlState(TypedDict):
    topic: str
    draft: str
    decision: str
    final_answer: str


def _build_hitl_graph(saver):
    """重建周五的 HITL 演示图（与 learn/llm/w10_hitl.py 同构，供查询用）。"""
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    def generate(state):
        return {"draft": f"《{state['topic']}》草稿"}

    def review(state):
        decision = interrupt({"question": "批复？", "draft": state["draft"]})
        return {"decision": decision}

    def act(state):
        return {"final_answer": f"已执行:{state['decision']}"}

    def regen(state):
        return {"final_answer": f"回退:{state['decision']}"}

    g = StateGraph(_HitlState)
    g.add_node("generate", generate)
    g.add_node("review", review)
    g.add_node("act", act)
    g.add_node("regen", regen)
    g.add_edge(START, "generate")
    g.add_edge("generate", "review")
    g.add_conditional_edges(
        "review",
        lambda s: "act" if s.get("decision") == "approve" else "regen",
        {"act": "act", "regen": "regen"},
    )
    g.add_edge("act", END)
    g.add_edge("regen", END)
    return g.compile(checkpointer=saver)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="checkpointer sqlite 路径")
    parser.add_argument("--graph", default="hitl", choices=["hitl"])
    parser.add_argument("--thread", required=True, help="thread_id，如 hitl::task-approve")
    args = parser.parse_args()

    with SqliteSaver.from_conn_string(args.db) as saver:
        app = _build_hitl_graph(saver)
        result = query_progress(app, {"configurable": {"thread_id": args.thread}})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
