"""跑批 harness（W9 周六）：跑 10 任务 → 轨迹落盘 → 基线数字 → 失败归因表。

验收③④⑤⑥⑦的载体。产物（qa_runs/ 目录）：

    trajectories/{task_id}.json   一个任务一份轨迹（可回放）
    baseline.md                   五项基线数字（成功率/步数/token/延迟/错误率）
    attribution.md                失败归因表（四类）

运行（模型路由可用 --model 覆盖；不传用 trae/qwen-3.5，避开 glm-pool 的
链式循环干扰 —— 周二实测结论）：

    .\\.venv\\Scripts\\python.exe -X utf8 scripts\\run_eval_agent.py
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
from app.core.llm.agent_registry import AGENT_TOOLS_SCHEMA, execute_agent_tool  # noqa: E402
from app.core.llm.client import get_llm_client  # noqa: E402
from app.core.llm.eval_tasks import TASKS, Task, result_to_trajectory  # noqa: E402
from app.core.llm.react_loop import ReactResult, run_react_loop  # noqa: E402

TITLE = "=" * 72

#: 观察摘要进轨迹的长度（截断即可读）。
_OBS_IN_TRACE: int = 160


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        with suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")


def _summary_observation(result: ReactResult) -> None:
    """把每步回灌的 observation 摘要写回 StepRecord（轨迹可读性）。"""
    for msg, step in _pair_messages_with_steps(result):
        step.observation = msg[:_OBS_IN_TRACE]


def _pair_messages_with_steps(result: ReactResult) -> list[tuple[str, Any]]:
    """从 messages 里抽出 tool 回复，与 steps 里的 action 顺序配对。

    messages 是「assistant(tool_calls) → tool → …」交错结构；steps 是按
    步聚合的。配对规则：tool 消息按出现顺序填进有 executed action 的步。
    """
    pairs: list[tuple[str, Any]] = []
    tool_msgs = [m["content"] for m in result.messages if m.get("role") == "tool"]
    idx = 0
    for step in result.steps:
        executed = [a for a in step.actions if a.get("executed")]
        for _ in executed:
            if idx < len(tool_msgs):
                pairs.append((tool_msgs[idx], step))
                idx += 1
    return pairs


async def run_one(
    client: Any,
    model: str,
    extra: dict[str, Any],
    task: Task,
    *,
    max_steps: int = 8,
) -> ReactResult:
    """跑一个任务。max_steps=8：复杂任务 5–8 步的判据区间。"""
    return await run_react_loop(
        client,
        model=model,
        user_message=task.instruction,
        tools=AGENT_TOOLS_SCHEMA,
        executor=execute_agent_tool,
        max_steps=max_steps,
        extra_request=extra,
    )


def write_baseline(
    out_dir: Path,
    rows: list[dict[str, Any]],
    model: str,
) -> None:
    """五项基线数字落表（验收⑤）。成功率按判据，不是按 completed。"""
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
        f"# Agent 基线（W9 周六 · {model}）",
        "",
        f"- 任务数: {total}（判据先写，跑后不改）",
        f"- **判据成功率: {passed}/{total} ({passed / total:.0%})**",
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


def write_attribution(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    """失败归因表（验收⑦）：四类——描述不清 / 规划错 / 参数错 / observation 问题。

    自动归因规则（从轨迹可判的）：
    - 参数错：有 invalid_arg / invalid_json 的失败调用
    - observation 问题：成功调用的 observation 被截断（truncated）
    - 描述不清：调了不该调的工具、或重复调用被警告
    - 规划错：其余（含 completed 但判据不过的）
    """
    lines = [
        "# 失败归因表（W9 周六首轮）",
        "",
        "| task | 判据 | 归因 | 证据 |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        if r["passed"]:
            continue
        category, evidence = _attribute(r)
        lines.append(f"| {r['task_id']} | × | {category} | {evidence} |")
    (out_dir / "attribution.md").write_text("\n".join(lines), encoding="utf-8")


def _attribute(r: dict[str, Any]) -> tuple[str, str]:
    bad_args = [
        a
        for s in r["trace"]["steps"]
        for a in s["actions"]
        if a.get("kind") in ("invalid_arg", "invalid_json")
    ]
    if bad_args:
        return "参数错", f"{len(bad_args)} 次 invalid 参数（{bad_args[0].get('tool')}）"
    truncated = [
        a
        for s in r["trace"]["steps"]
        for a in s["actions"]
        if a.get("executed") and a.get("ok") and s.get("observation", "").find("截断") >= 0
    ]
    if truncated:
        return "observation 截断丢信息", f"{len(truncated)} 次成功调用返回被截断"
    repeat = [
        a for s in r["trace"]["steps"] for a in s["actions"] if a.get("kind") == "repeat_call"
    ]
    if repeat:
        return "工具描述不清（引发重复/乱调）", f"{len(repeat)} 次 repeat_call"
    wrong_tool = [
        a.get("tool")
        for s in r["trace"]["steps"]
        for a in s["actions"]
        if a.get("executed") and a.get("tool") not in r.get("expected_tools", [])
    ]
    if wrong_tool:
        return "工具描述不清（选错工具）", f"调用了 {sorted(set(wrong_tool))}"
    return "规划错", f"stopped={r['stopped']}，答案片段: {r['trace']['final_answer'][:80]}"


async def main() -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="trae/qwen-3.5")
    parser.add_argument("--outdir", default="qa_runs")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    # 工作目录切到隔离沙盘：报告/账本/演示库都落在 qa_runs 的父级临时目录
    workdir = Path(tempfile.mkdtemp(prefix="qa_eval_"))
    os.chdir(workdir)
    runs_dir = Path(args.outdir)
    runs_dir.mkdir(exist_ok=True)
    (runs_dir / "trajectories").mkdir(exist_ok=True)

    settings = get_settings()
    extra = {"reasoning_effort": "none"}
    client = get_llm_client(settings)

    rows: list[dict[str, Any]] = []
    print(
        f"{TITLE}\nAgent 评测：{len(TASKS)} 任务 · model={args.model} · max_steps={args.max_steps}\n{TITLE}"
    )

    try:
        for task in TASKS:
            print(f"\n--- {task.task_id} [{task.category}] ---")
            print(f"  指令: {task.instruction[:60]}")
            started = time.perf_counter()
            try:
                result = await run_one(client, args.model, extra, task, max_steps=args.max_steps)
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

            _summary_observation(result)
            trace = result_to_trajectory(task, result, args.model)
            (runs_dir / "trajectories" / f"{task.task_id}.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            passed, note = task.verify(result)
            tool_results = [
                bool(a.get("ok")) for s in result.steps for a in s.actions if a.get("executed")
            ]
            mark = "√" if passed else "×"
            print(f"  [{mark}] {note} | stopped={result.stopped_reason} steps={len(result.steps)}")
            if not passed:
                print(f"  答案片段: {result.final_answer[:120]}")
            rows.append(
                {
                    "task_id": task.task_id,
                    "category": task.category,
                    "passed": passed,
                    "verify_note": note,
                    "completed": result.completed,
                    "stopped": result.stopped_reason,
                    "steps": len(result.steps),
                    "tokens": result.usage_total.get("total_tokens", 0),
                    "latency_ms": latency_ms,
                    "tool_results": tool_results,
                    "expected_tools": task.expected_tools,
                    "trace": trace,
                }
            )
    finally:
        await client.close()

    write_baseline(runs_dir, rows, args.model)
    write_attribution(runs_dir, rows)
    print(
        f"\n{TITLE}\n产物：{runs_dir / 'baseline.md'} · {runs_dir / 'attribution.md'} · trajectories/{len(rows)} 份"
    )
    passed = sum(1 for r in rows if r["passed"])
    print(f"判据成功率: {passed}/{len(rows)}\n{TITLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
