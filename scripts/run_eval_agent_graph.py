"""跑批 harness（W10 周六）：同一 10 任务跑 LangGraph 图版 → qa_runs_w10。

与 scripts/run_eval_agent.py 的差异只有循环载体：
W9 = run_react_loop（手写 while）；W10 = agent_graph.build_agent_graph（声明图）。
判据（eval_tasks.TASKS）、轨迹格式、基线表、归因表全部同源复用——
对比公平性由冻结声明（docs/weekly/w10.md）保证。

运行：

    .\\.venv\\Scripts\\python.exe -X utf8 scripts\\run_eval_agent_graph.py --outdir qa_runs_w10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.llm.agent_graph import build_agent_graph, result_from_state  # noqa: E402
from app.core.llm.client import get_llm_client  # noqa: E402
from app.core.llm.eval_tasks import TASKS  # noqa: E402
from app.core.llm.react_loop import REACT_SYSTEM_PROMPT  # noqa: E402

TITLE = "=" * 72

#: W9 harness 同款：判据函数吃 ReactResult；图版从 state 组装同形对象。
from app.core.llm.react_loop import ReactResult, StepRecord  # noqa: E402


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        with suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")


def state_to_react_result(state: dict[str, Any]) -> ReactResult:
    """把图终态转成 ReactResult，让 task.verify(result) 原样工作。"""
    stopped = state.get("stopped_reason", "completed")
    steps = [
        StepRecord(
            step=m["step"],
            mode=m["mode"],
            thought=m.get("thought", ""),
            actions=m.get("actions", []),
            model_ms=m.get("model_ms", 0),
            tokens=m.get("tokens", {}),
            observation=m.get("observation", ""),
        )
        for m in (merged_steps(state))
    ]
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in state.get("usage_log", []):
        for k in usage_total:
            usage_total[k] += u.get(k, 0)
    return ReactResult(
        final_answer=state.get("final_answer", ""),
        messages=list(state.get("messages", [])),
        steps=steps,
        stopped_reason=stopped,
        usage_total=usage_total,
    )


def merged_steps(state: dict[str, Any]) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for rec in state.get("steps", []):
        m = merged.setdefault(
            rec["step"],
            {
                "step": rec["step"],
                "mode": rec["mode"],
                "thought": rec.get("thought", ""),
                "model_ms": rec.get("model_ms", 0),
                "tokens": rec.get("tokens", {}),
                "observation": rec.get("observation", ""),
                "actions": [],
            },
        )
        m["actions"].extend(a for a in rec.get("actions", []) if a not in m["actions"])
    return [merged[k] for k in sorted(merged)]


def write_baseline(out_dir: Path, rows: list[dict[str, Any]], model: str) -> None:
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    completed = sum(1 for r in rows if r["completed"])
    avg_steps = sum(r["steps"] for r in rows) / total if total else 0
    avg_tokens = sum(r["tokens"] for r in rows) / total if total else 0
    avg_latency = sum(r["latency_ms"] for r in rows) / total if total else 0
    tool_calls = [c for r in rows for c in r["tool_results"]]
    tool_error_rate = (sum(1 for c in tool_calls if not c) / len(tool_calls)) if tool_calls else 0.0

    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    lines = [
        f"# Agent 基线（W10 周六 · LangGraph 图版 · {model}）",
        "",
        f"- 任务数: {total}（判据与 W9 同源，跑后不改）",
        f"- **判据成功率: {passed}/{total} ({passed / total * 100:.0f}%)**",
        f"- 循环 completed 率: {completed}/{total}",
        f"- 平均步数: {avg_steps:.1f}",
        f"- 平均 token: {avg_tokens:.0f}",
        f"- 平均端到端延迟: {avg_latency / 1000:.1f}s",
        f"- 工具错误率（错误调用/全部调用）: {tool_error_rate:.0%}（{len(tool_calls)} 次调用）",
        "",
        "## 按类别",
        "",
        "| 类别 | 通过 | 明细 |",
        "| --- | --- | --- |",
    ]
    for cat, cat_rows in by_cat.items():
        cat_passed = sum(1 for r in cat_rows if r["passed"])
        detail = ", ".join(f"{r['task_id']}={'√' if r['passed'] else '×'}" for r in cat_rows)
        lines.append(f"| {cat} | {cat_passed}/{len(cat_rows)} | {detail} |")

    lines += [
        "",
        "## 逐任务",
        "",
        "| task | 判据 | stopped | steps | tokens | 延迟s |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['task_id']} | {'√' if r['passed'] else '×'} {r['verify_note'][:40]} "
            f"| {r['stopped']} | {r['steps']} | {r['tokens']} | {r['latency_ms'] / 1000:.1f} |"
        )
    (out_dir / "baseline.md").write_text("\n".join(lines), encoding="utf-8")


async def main() -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="trae/qwen-3.5")
    parser.add_argument("--outdir", default="qa_runs_w10")
    parser.add_argument("--max-steps", type=int, default=8)  # 与 W9 harness 相同
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="qa_eval_w10_"))
    os.chdir(workdir)
    runs_dir = Path(args.outdir)
    runs_dir.mkdir(exist_ok=True)
    (runs_dir / "trajectories").mkdir(exist_ok=True)

    settings = get_settings()
    client = get_llm_client(settings)
    app = build_agent_graph(client)

    rows: list[dict[str, Any]] = []
    print(
        f"{TITLE}\nAgent 评测（LangGraph 图版）：{len(TASKS)} 任务 · model={args.model} · max_steps={args.max_steps}\n{TITLE}"
    )

    try:
        for task in TASKS:
            print(f"\n--- {task.task_id} [{task.category}] ---")
            print(f"  指令: {task.instruction[:60]}")
            started = time.perf_counter()
            cfg = {"configurable": {"thread_id": f"w10::{task.task_id}"}}
            try:
                final_state = await app.ainvoke(
                    {
                        "instruction": task.instruction,
                        "model": args.model,
                        "max_steps": args.max_steps,
                        "messages": [
                            {"role": "system", "content": REACT_SYSTEM_PROMPT},
                            {"role": "user", "content": task.instruction},
                        ],
                    },
                    config=cfg,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  运行异常: {type(exc).__name__}: {str(exc)[:100]}")
                rows.append(
                    {
                        "task_id": task.task_id,
                        "category": task.category,
                        "passed": False,
                        "verify_note": f"运行异常 {type(exc).__name__}",
                        "completed": False,
                        "stopped": "exception",
                        "steps": 0,
                        "tokens": 0,
                        "latency_ms": int((time.perf_counter() - started) * 1000),
                        "tool_results": [],
                        "expected_tools": task.expected_tools,
                        "trace": {"steps": [], "final_answer": ""},
                    }
                )
                continue
            latency_ms = int((time.perf_counter() - started) * 1000)

            trace = result_from_state(
                task.task_id, task.category, args.model, task.instruction, final_state
            )
            (runs_dir / "trajectories" / f"{task.task_id}.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            react_result = state_to_react_result(final_state)
            passed, note = task.verify(react_result)
            tool_results = [
                bool(a.get("ok"))
                for s in merged_steps(final_state)
                for a in s.get("actions", [])
                if a.get("executed")
            ]
            mark = "√" if passed else "×"
            print(
                f"  [{mark}] {note} | stopped={final_state.get('stopped_reason', 'completed')} "
                f"steps={final_state.get('step_no', 0)}"
            )
            if not passed:
                print(f"  答案片段: {final_state.get('final_answer', '')[:120]}")
            rows.append(
                {
                    "task_id": task.task_id,
                    "category": task.category,
                    "passed": passed,
                    "verify_note": note,
                    "completed": final_state.get("stopped_reason", "completed") == "completed",
                    "stopped": final_state.get("stopped_reason", "completed"),
                    "steps": final_state.get("step_no", 0),
                    "tokens": trace["usage_total"]["total_tokens"],
                    "latency_ms": latency_ms,
                    "tool_results": tool_results,
                    "expected_tools": task.expected_tools,
                    "trace": trace,
                }
            )
    finally:
        await client.close()

    write_baseline(runs_dir, rows, args.model)
    print(f"\n{TITLE}\n产物：{runs_dir / 'baseline.md'} · trajectories/{len(rows)} 份")
    passed = sum(1 for r in rows if r["passed"])
    print(f"判据成功率: {passed}/{len(rows)}\n{TITLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
