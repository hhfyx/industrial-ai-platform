"""跑 20 任务集（W11 周五起：小样试跑 / 回归子集 / 全量）。

## 三种用法

    # 小样试跑：只跑 5 个新任务（15:30–17:00 的验收）
    python -X utf8 scripts/run_eval_20.py --only N1,N3,N7,N9,N12

    # 回归子集：10 个 W10 任务，对照基线（19:00–21:00 的验收）
    python -X utf8 scripts/run_eval_20.py --regression

    # 全量 20 个
    python -X utf8 scripts/run_eval_20.py

## 两个必须显式声明的开关（都比"默认值"更重要）

**`--confirm-constraint`**：是否给 prompt 拼上"有副作用动作先确认"的增量约束。
N7/N8 这类任务的成败取决于它 —— 所以跑"无机制对照组"时**必须关掉**，
否则你不知道是机制有效还是 prompt 有效。默认**开**（新任务组的正确配置），
但跑对照时要显式 `--no-confirm-constraint`。

**`--model`**：默认取 `.env`，但**曲线与基线对比必须显式传
`trae/qwen-3.5`** —— 周四踩过一次：`.env` 换成 buddy-pool 后
6 个任务全 repeat_loop，差点被当成"预算问题"。

## 判据来源

- 回归子集：`AgentTask.checks` 里那个 `legacy` 项 → 转调 W10 原判据（零改动）
- 新增 10 个：声明式 checks → `check_one()`
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.llm.agent_graph import (  # noqa: E402
    build_agent_graph,
    result_from_state,
    state_to_react_result,
)
from app.core.llm.budget import BudgetConfig  # noqa: E402
from app.core.llm.react_loop import build_system_prompt  # noqa: E402
from app.core.llm.tasks_20 import NEW_TASKS, REGRESSION_IDS, AgentTask, all_tasks_20  # noqa: E402

TITLE = "=" * 84


def _select(*, only: list[str] | None, regression: bool, new_only: bool) -> list[AgentTask]:
    tasks = all_tasks_20()
    if only:
        wanted = [t.strip() for t in only if t.strip()]
        picked: list[AgentTask] = []
        unmatched: list[str] = []
        for token in wanted:
            # 支持完整 id，也支持"前缀 + 下划线"（`N7_` → `N7_confirm_before_notify`）。
            # 为什么要下划线：不加的话 `N1` 会同时命中 N10/N11/N12，
            # 而手打完整 id 容易错一个字符就白跑一轮真模型。
            exact = [t for t in tasks if t.task_id == token]
            if exact:
                picked.extend(exact)
                continue
            stem = token if token.endswith("_") else f"{token}_"
            by_prefix = [t for t in tasks if t.task_id.startswith(stem)]
            if len(by_prefix) == 1:
                picked.extend(by_prefix)
            elif len(by_prefix) > 1:
                candidates = [t.task_id for t in by_prefix]
                unmatched.append(f"{token}（前缀匹配到多个：{candidates}）")
            else:
                unmatched.append(token)
        if unmatched:
            raise SystemExit(f"无法唯一定位任务：{unmatched}")
        # 去重保序
        seen: set[str] = set()
        unique: list[AgentTask] = []
        for t in picked:
            if t.task_id not in seen:
                seen.add(t.task_id)
                unique.append(t)
        return unique
    if regression:
        ids = set(REGRESSION_IDS)
        return [t for t in tasks if t.task_id in ids]
    if new_only:
        ids = {t.task_id for t in NEW_TASKS}
        return [t for t in tasks if t.task_id in ids]
    return tasks


async def run_one(
    task: AgentTask,
    *,
    client: Any,
    model: str,
    confirmation: bool,
) -> dict[str, Any]:
    """跑一个任务。预算按任务自带（compute 类已放宽）。

    ## 确认约束**只对 `confirm` 类生效**（1/01 实测踩出来的）

    第一版把约束对所有任务都拼上，结果回归子集从 9/10 掉到 7/10：
    T7（出报告）和 T8（出报告+通知）都被模型"先请求确认"卡住了 ——
    模型的判断没错（它确实该问），但那两个任务的判据要求"真的生成"。

    这暴露一个设计错误：**确认约束不是全局安全，是任务级语义**。
    把它当全局提示词，等于让每个任务都变成"需确认类"，
    而 W10 基线的那 10 个任务本来就是"信任这个 Agent 会自己做完"。

    所以约束跟着任务走：只有 `category == "confirm"` 的任务拼它。
    对照组的做法也相应明确 —— 关掉它跑 confirm 类，看模型会不会直接动手。
    """
    app = build_agent_graph(client)
    budget = BudgetConfig(
        max_tokens=task.budget_tokens,
        max_steps=10,
        max_wall_time_s=task.budget_wall_time_s,
    )
    use_confirmation = confirmation and task.category == "confirm"
    system_prompt = build_system_prompt(confirmation=use_confirmation)
    started = time.perf_counter()
    state = await app.ainvoke(
        {
            "instruction": task.instruction,
            "model": model,
            "max_steps": budget.max_steps,
            "budget": budget.to_dict(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task.instruction},
            ],
        },
        config={"configurable": {"thread_id": f"w11-20::{task.task_id}"}},
    )
    wall_ms = int((time.perf_counter() - started) * 1000)

    trace = result_from_state(task.task_id, task.category, model, task.instruction, state)
    result = state_to_react_result(state)
    passed, why = task.verify(result)
    return {
        "task_id": task.task_id,
        "category": task.category,
        "passed": passed,
        "why": why,
        "stopped_reason": trace["stopped_reason"],
        "steps": len(trace["steps"]),
        "tokens": trace["usage_total"].get("total_tokens", 0),
        "latency_ms": wall_ms,
        "confirmation_applied": use_confirmation,
        "expected_without_mechanisms": task.expected_without_mechanisms,
        "trace": trace,
    }


def render_summary(rows: list[dict[str, Any]], *, label: str) -> str:
    lines = [TITLE, f"{label} · {len(rows)} 任务", TITLE]
    passed = sum(1 for r in rows if r["passed"])
    lines.append(f"通过：{passed}/{len(rows)}")
    lines.append("")
    lines.append(f"{'task':34s}{'类':10s}{'判据':6s}{'停止原因':20s}{'步':>4s}{'token':>8s}")
    lines.append("-" * 84)
    for r in rows:
        mark = "√" if r["passed"] else "×"
        lines.append(
            f"{r['task_id']:34s}{r['category']:10s}{mark:6s}"
            f"{r['stopped_reason']:20s}{r['steps']:>4d}{r['tokens']:>8d}"
        )
        if not r["passed"]:
            lines.append(f"    判据：{r['why']}")
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> int:
    tasks = _select(
        only=args.only.split(",") if args.only else None,
        regression=args.regression,
        new_only=args.new_only,
    )
    if not tasks:
        print("没有选中任何任务", file=sys.stderr)
        return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "trajectories").mkdir(parents=True, exist_ok=True)

    if args.live:  # pragma: no cover - 真网关
        from app.core.config import get_settings
        from app.core.llm.client import get_llm_client

        settings = get_settings()
        model = args.model or settings.llm.model
        client = get_llm_client(settings)
    else:
        from scripts._fake_agent_client import make_fake_client

        model = args.model or "fake"
        client = make_fake_client()

    confirmation = not args.no_confirm_constraint
    print(
        f"{TITLE}\n20 任务集 · {'真模型' if args.live else '离线假 client'} · "
        f"model={model}\n确认约束={'开' if confirmation else '关'}（仅需确认类任务生效）· "
        f"选中 {len(tasks)} 个\n{TITLE}"
    )

    rows: list[dict[str, Any]] = []
    for task in tasks:
        row = await run_one(task, client=client, model=model, confirmation=confirmation)
        rows.append(row)
        (outdir / "trajectories" / f"{task.task_id}.json").write_text(
            json.dumps(row["trace"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        mark = "√" if row["passed"] else "×"
        print(
            f"  [{mark}] {task.task_id:34s} {row['stopped_reason']:18s} "
            f"steps={row['steps']} tokens={row['tokens']}"
        )
        if not row["passed"]:
            print(f"        {row['why']}")

    print("\n" + render_summary(rows, label="汇总"))

    passed = sum(1 for r in rows if r["passed"])
    payload = {
        "mode": "live" if args.live else "offline",
        "model": model,
        "confirmation_constraint": confirmation,
        "count": len(rows),
        "passed": passed,
        "rows": [{k: v for k, v in r.items() if k != "trace"} for r in rows],
    }
    out = outdir / "summary.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘：{out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跑 20 任务集（默认离线）")
    parser.add_argument("--live", action="store_true", help="打真模型")
    parser.add_argument("--model", default="", help="模型路由；基线对比请显式传 trae/qwen-3.5")
    parser.add_argument("--only", default="", help="逗号分隔的任务 id，如 N1,N7")
    parser.add_argument("--regression", action="store_true", help="只跑 W10 的 10 个回归子集")
    parser.add_argument("--new-only", action="store_true", help="只跑新增的 10 个")
    parser.add_argument(
        "--no-confirm-constraint",
        action="store_true",
        help="关掉「有副作用动作先确认」的增量约束（跑无机制对照组时必须关）",
    )
    parser.add_argument("--outdir", default="qa_runs_w11_20")
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
