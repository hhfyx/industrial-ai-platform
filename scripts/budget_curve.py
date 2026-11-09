"""预算-成功率曲线（W11 周四 21:15–21:40，本周最有信息量的一张图）。

## 必答问题

**预算砍半会掉多少成功率？** 这张脚本就是那个数的来源。

不测这个数，"成本下降"就是拿用户可用性换的，而且不知道换了多少。
今天顺序必须是：**先测曲线，再定档**。

## 三个坑（都是任务原文点出来或实测遇到的）

1. **别先定档再找理由。** 脚本先跑三档，再让人看数据选档；
   `DEFAULT_BUDGET` 的注释里写着"改它必须改曲线的数字"。
2. **别把预算触顶的失败归因成"任务太难"。**
   汇总表把 `budget_*` / `no_progress` 单列一栏，与"判据失败"分开。
3. **token 上限要卡在"单任务总量"而不是"单次调用"。**
   模型单次 `max_tokens=800` 是响应长度上限（冻结项，不能动），
   预算是**跨步累加**的总量。两者混为一谈，预算就永远不触发。

## 用法

    python -X utf8 scripts/budget_curve.py --live            # 真模型，三档 × 6 任务
    python -X utf8 scripts/budget_curve.py --live --runs 2   # 每档跑 2 轮
    python -X utf8 scripts/budget_curve.py                   # 离线（验证脚本链路）

离线模式用假 client 造"每步固定花 T token"，只为验证：
预算真的会在超限前停、优雅退出载荷齐全、曲线能算出来。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.llm.agent_graph import (  # noqa: E402
    build_agent_graph,
    result_from_state,
    state_to_react_result,
)
from app.core.llm.budget import BUDGET_REASONS, BudgetConfig, BudgetUsed  # noqa: E402
from app.core.llm.eval_tasks import TASKS, Task  # noqa: E402
from app.core.llm.react_loop import REACT_SYSTEM_PROMPT  # noqa: E402

TITLE = "=" * 84

#: 三档 token 预算。5k 明显不够（W10 基线均 7455），10k 刚好过线，15k 宽松。
#: 选这三档是因为它们把"掉点"的拐点夹在中间 —— 曲线要看的是拐点，不是三个点。
TOKEN_TIERS: tuple[int, ...] = (5_000, 10_000, 15_000)

#: 6 个代表任务：覆盖 simple/complex/impossible 三类，且包含 token 高与低的两端。
#: 为什么不是全 10 个：曲线求的是趋势，6 个 × 3 档 = 18 次运行已经够贵；
#: 而且 T4（1 步）与 T7（19626 token）已经覆盖了最短与最长两端。
REPRESENTATIVE_IDS: tuple[str, ...] = (
    "T1_simple_query",
    "T3_simple_metric",
    "T5_simple_batch_list",
    "T6_complex_verdict",
    "T7_complex_report",
    "T10_impossible_metric",
)


# ============================================================ 离线假 client


class _TokenEatingClient:
    """离线假 client：每步固定花 `tokens_per_step`，前几步调工具，最后由答案收尾。

    目的不是模拟真实模型，而是验证**预算机制本身**：
    给定确定的每步开销，预算 N 应该在 ceil(N / tokens_per_step) 步附近触发。

    两个必须模拟的细节（第一版漏了，导致三档都是 1 步就 completed）：

    1. **前几步必须调工具** —— 否则模型第 1 步就给自然语言，图直接 finalize，
       预算根本没机会触发。真实任务的 2-5 步全是工具步。
    2. **最后一步给答案** —— 让"预算够时能正常完成"这条路也可测。
       否则三档全部触顶，曲线没有上端。
    """

    def __init__(
        self, *, tokens_per_step: int = 3000, tool_steps: int = 2, answer_after: int = 6
    ) -> None:
        self.tokens_per_step = tokens_per_step
        self.tool_steps = tool_steps
        self.answer_after = answer_after
        self.calls = 0

    async def _create(self, **kwargs: Any) -> Any:
        self.calls += 1
        usage = SimpleNamespace(
            prompt_tokens=self.tokens_per_step - 100,
            completion_tokens=100,
            total_tokens=self.tokens_per_step,
        )
        if self.calls < self.answer_after:
            # 所有中间步都**带工具调用**：图里"没有 tool_calls"就等于模型给出
            # 最终答案（这是 W9 的语义），中间步一旦不带工具，任务第 3 步就
            # 被当成完成了 —— 曲线因此看不出任何预算影响（第一版就踩了这个坑）。
            # 所以中间步一律调用工具，用不同 limit 避开重复熔断。
            msg = SimpleNamespace(
                content=f"第 {self.calls} 步：继续取数",
                tool_calls=[
                    SimpleNamespace(
                        id=f"c{self.calls}",
                        function=SimpleNamespace(
                            name="query_metrics_sql",
                            arguments=f'{{"table": "inspections", "limit": {self.calls + 1}}}',
                        ),
                    )
                ],
            )
        else:
            msg = SimpleNamespace(
                content=(
                    "结论：B20241207-01 抗压强度 42.5 MPa（规范限值 40.0，合格），"
                    "含水量 8.2%；B20241129-03 抗压强度 36.1 MPa（不合格）。"
                    "B20241207-01 与 B20241129-03 两个批次均已核对。"
                    "放射性指标：该批次没有此指标数据，无法给出数值。"
                ),
                tool_calls=None,
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)

    @property
    def chat(self) -> Any:
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))


# ============================================================ 跑一格


async def run_cell(
    task: Task,
    *,
    budget: BudgetConfig,
    client: Any,
    model: str,
) -> dict[str, Any]:
    """跑一个任务 × 一档预算，返回这一格的原始数据。"""
    app = build_agent_graph(client)
    started = time.perf_counter()
    state = await app.ainvoke(
        {
            "instruction": task.instruction,
            "model": model,
            "max_steps": budget.max_steps,  # 向后兼容：老 harness 用这个字段
            "budget": budget.to_dict(),
            "messages": [
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": task.instruction},
            ],
        },
        config={"configurable": {"thread_id": f"budget::{task.task_id}::{budget.max_tokens}"}},
    )
    wall_ms = int((time.perf_counter() - started) * 1000)

    trace = result_from_state(task.task_id, task.category, model, task.instruction, state)
    result = state_to_react_result(state)
    passed, why = task.verify(result)
    used = BudgetUsed.from_dict(state.get("budget_used"))
    stopped = trace["stopped_reason"]

    return {
        "task_id": task.task_id,
        "passed": passed,
        "why": why,
        "stopped_reason": stopped,
        # **预算触顶单独标出来** —— 这一栏是今天最重要的设计：
        # 它就是防"把预算失败归因成任务难"的那道闸。
        "budget_hit": stopped in BUDGET_REASONS,
        "no_progress_hit": stopped == "no_progress",
        "tokens": used.tokens,
        "steps": used.steps,
        "wall_time_s": round(wall_ms / 1000.0, 2),
        "budget_exit": trace.get("budget_exit"),
    }


# ============================================================ 汇总


def summarize(cells: list[dict[str, Any]], tier: int) -> dict[str, Any]:
    """某一档的汇总。成功率、平均 token、以及**失败归因的拆分**。"""
    total = len(cells)
    passed = sum(1 for c in cells if c["passed"])
    tokens = [c["tokens"] for c in cells]
    budget_hit = [c for c in cells if c["budget_hit"]]
    noprog = [c for c in cells if c["no_progress_hit"]]
    judge_fail = [
        c for c in cells if not c["passed"] and not c["budget_hit"] and not c["no_progress_hit"]
    ]
    return {
        "tier": tier,
        "runs": total,
        "passed": passed,
        "success_rate": passed / total if total else 0.0,
        "avg_tokens": round(sum(tokens) / len(tokens), 1) if tokens else 0.0,
        "max_tokens_used": max(tokens) if tokens else 0,
        "avg_steps": round(sum(c["steps"] for c in cells) / total, 2) if total else 0.0,
        "avg_wall_time_s": round(sum(c["wall_time_s"] for c in cells) / total, 2) if total else 0.0,
        # 失败的三类拆分：判据不过 / 预算触顶 / 无进展早停
        "failed_judge": len(judge_fail),
        "failed_budget": len(budget_hit),
        "failed_no_progress": len(noprog),
    }


def render_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        TITLE,
        "预算-成功率曲线（W11 周四）",
        TITLE,
        f"{'档位':>8s}{'成功率':>14s}{'平均token':>12s}{'峰值token':>12s}"
        f"{'平均步数':>10s}{'判据失败':>10s}{'预算触顶':>10s}{'无进展':>8s}",
        "-" * 84,
    ]
    for r in rows:
        rate = f"{r['success_rate']:.0%} ({r['passed']}/{r['runs']})"
        lines.append(
            f"{r['tier']:>8d}{rate:>14s}{r['avg_tokens']:>12.0f}"
            f"{r['max_tokens_used']:>12d}{r['avg_steps']:>10.2f}"
            f"{r['failed_judge']:>10d}{r['failed_budget']:>10d}{r['failed_no_progress']:>8d}"
        )
    lines.append("")
    if len(rows) >= 2:
        lo, hi = rows[0], rows[-1]
        drop_pp = (lo["success_rate"] - hi["success_rate"]) * 100
        save_tok = hi["avg_tokens"] - lo["avg_tokens"]
        lines.append(
            f"关键对比：{lo['tier']} vs {hi['tier']} —— "
            f"成功率 {lo['success_rate']:.0%} vs {hi['success_rate']:.0%}"
            f"（{'掉' if drop_pp > 0 else '升'} {abs(drop_pp):.0f} 个百分点），"
            f"平均 token {lo['avg_tokens']:.0f} vs {hi['avg_tokens']:.0f}"
            f"（省 {save_tok:.0f}）"
        )
        lines.append(
            "读表提示：看「预算触顶」这一列。如果低档的失败全都落在这一列，"
            "那不是任务难，是预算不够 —— 归因时不许混为一谈。"
        )
    return "\n".join(lines)


# ============================================================ CLI


async def main_async(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model = args.model

    if args.live:  # pragma: no cover - 真网关
        from app.core.config import get_settings
        from app.core.llm.client import get_llm_client

        settings = get_settings()
        model = model or settings.llm.model
        client_factory = lambda: get_llm_client(settings)  # noqa: E731
    else:
        client_factory = None

    tasks = [t for t in TASKS if t.task_id in set(args.tasks or REPRESENTATIVE_IDS)]
    if not tasks:
        print(f"没有匹配的任务：{args.tasks}", file=sys.stderr)
        return 2

    print(
        f"{TITLE}\n预算-成功率曲线 · {'真模型' if args.live else '离线假 client'} · "
        f"{len(tasks)} 任务 × {len(TOKEN_TIERS)} 档 × {args.runs} 轮\n{TITLE}"
    )

    all_rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for tier in TOKEN_TIERS:
        budget = BudgetConfig(max_tokens=tier)
        cells: list[dict[str, Any]] = []
        for run_idx in range(args.runs):
            for task in tasks:
                client = (
                    client_factory()  # type: ignore[misc]
                    if args.live
                    else _TokenEatingClient(tokens_per_step=args.offline_tokens)
                )
                cell = await run_cell(task, budget=budget, client=client, model=model)
                cell["run"] = run_idx
                cells.append(cell)
                mark = "√" if cell["passed"] else "×"
                flag = " [预算触顶]" if cell["budget_hit"] else ""
                print(
                    f"  [{mark}] {task.task_id:28s} tier={tier:>6d} "
                    f"tokens={cell['tokens']:>6d} steps={cell['steps']:>2d} "
                    f"stopped={cell['stopped_reason']}{flag}"
                )
        row = summarize(cells, tier)
        all_rows.append(row)
        detail[str(tier)] = cells
        print(
            f"  → tier {tier}: {row['passed']}/{row['runs']} "
            f"({row['success_rate']:.0%})，均 {row['avg_tokens']:.0f} token\n"
        )

    print("\n" + render_table(all_rows))

    payload = {
        "mode": "live" if args.live else "offline",
        "model": model,
        "runs_per_tier": args.runs,
        "tiers": list(TOKEN_TIERS),
        "summary": all_rows,
        "cells": detail,
    }
    out = outdir / "budget_curve.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘：{out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="预算-成功率曲线（默认离线）")
    parser.add_argument("--live", action="store_true", help="打真模型")
    parser.add_argument("--model", default="", help="真模型路由；留空取 settings")
    parser.add_argument("--runs", type=int, default=1, help="每档跑几轮（默认 1）")
    parser.add_argument("--tasks", nargs="*", default=None, help="指定任务 id 列表")
    parser.add_argument("--outdir", default="qa_runs_w11_budget")
    parser.add_argument(
        "--offline-tokens",
        type=int,
        default=3000,
        help="离线模式每步假 token（默认 3000，让 5k 档第 2 步就触顶）",
    )
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
