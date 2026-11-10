"""关键任务稳定性（周六 15:30–16:30，W9 的 F6）。

## 为什么要单独做这个

全量跑一轮看起来 100%，但"一轮"只是一次抽样。**模型有随机性**，
一次通过不代表稳定通过。W9 的 F6 就是干这件事：挑几个关键任务跑多次，
看**成功率区间**而不是单点。

## 挑哪几个

不是随机挑，是按"**最该担心的**"挑：

| 任务 | 为什么担心它 |
| --- | --- |
| T7 出报告 | 4 步、跨 3 工具，任何一步抖动都断链 |
| T8 报告+通知 | 有副作用 + 端点不可达，最容易降级 |
| N7 需确认 | 靠 prompt 约束，不是硬拦截 —— 最可能被模型"忘了" |
| N8 批量确认 | 同上，且步数更多 |
| T10 缺指标 | 判据与诚实行为的张力，靠 prompt 缓解 |
| N12 实时数据 | 历史值顶替实时值的风险最高 |

## 怎么报

不能只报"3/3 通过"。必须同时报：
- 成功次数 / 总次数；
- **区间**（Wilson 下界太复杂，这里用"k/n"与是否稳定）；
- 波动来源（哪几次失败、失败原因）；
- 指标波动（token / 步数的极差）—— 那是"同一个答案花了多少钱"的稳定性。

3 次里全过，只能说"3/3"，**不能说"稳定 100%"** ——
3 次全过的 95% 置信下界约 0.37（三中取三的下界很低）。
这条必须写在报告里，否则"稳定性"三个字会被读成保证。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_eval_20 import run_one  # noqa: E402

from app.core.llm.tasks_20 import all_tasks_20  # noqa: E402

TITLE = "=" * 78

#: 稳定性抽检的任务（选最该担心的，不是随机）。
CRITICAL_IDS: Final[tuple[str, ...]] = (
    "T7_complex_report",
    "T8_complex_notify",
    "N7_confirm_before_notify",
    "N8_confirm_before_report",
    "T10_impossible_metric",
    "N12_impossible_realtime",
)


async def run_stability(*, runs: int, model: str, outdir: Path, live: bool) -> dict[str, Any]:
    tasks = {t.task_id: t for t in all_tasks_20()}
    picked = [tasks[i] for i in CRITICAL_IDS if i in tasks]

    if live:  # pragma: no cover
        from app.core.config import get_settings
        from app.core.llm.client import get_llm_client

        settings = get_settings()
        model = model or settings.llm.model
        client = get_llm_client(settings)
    else:
        from scripts._fake_agent_client import make_fake_client

        client = make_fake_client()

    results: dict[str, list[dict[str, Any]]] = {t.task_id: [] for t in picked}
    print(f"{TITLE}\n关键任务稳定性 · {len(picked)} 任务 × {runs} 次 · model={model}\n{TITLE}")

    for run_idx in range(runs):
        for task in picked:
            row = await run_one(task, client=client, model=model, confirmation=True)
            results[task.task_id].append(row)
            # **每次都要把轨迹落盘**：不落盘的话，事后只能看到"失败"，
            # 看不到"为什么失败"，等于白跑（第一次就是这么浪费掉的）。
            tdir = outdir / f"{task.task_id}"
            tdir.mkdir(parents=True, exist_ok=True)
            (tdir / f"run{run_idx + 1}.json").write_text(
                json.dumps(row["trace"], ensure_ascii=False, indent=2), encoding="utf-8"
            )
            mark = "√" if row["passed"] else "×"
            print(
                f"  run{run_idx + 1} [{mark}] {task.task_id:32s} "
                f"steps={row['steps']} tokens={row['tokens']} "
                f"stopped={row['stopped_reason']}"
            )

    summary: dict[str, Any] = {}
    for tid, rows in results.items():
        passes = sum(1 for r in rows if r["passed"])
        tokens = [r["tokens"] for r in rows]
        steps = [r["steps"] for r in rows]
        failures = [
            {"run": i + 1, "why": r["why"], "stopped": r["stopped_reason"]}
            for i, r in enumerate(rows)
            if not r["passed"]
        ]
        summary[tid] = {
            "passes": passes,
            "runs": len(rows),
            "rate": passes / len(rows),
            "token_min": min(tokens),
            "token_max": max(tokens),
            "token_spread": max(tokens) - min(tokens),
            "steps_min": min(steps),
            "steps_max": max(steps),
            "failures": failures,
        }
    return summary


def render(summary: dict[str, Any], runs: int) -> list[str]:
    lines = [
        TITLE,
        "稳定性（多次重复，报区间不报单点）",
        TITLE,
        f"  {'task':32s}{'通过':>8s}{'token 区间':>20s}{'步数':>10s}{'波动':>8s}",
        "  " + "-" * 76,
    ]
    all_stable = True
    for tid, s in summary.items():
        rate = f"{s['passes']}/{s['runs']}"
        tok = f"{s['token_min']}-{s['token_max']}"
        sp = f"{s['steps_min']}-{s['steps_max']}"
        spread = f"{s['token_spread']}"
        stable = s["passes"] == s["runs"]
        if not stable:
            all_stable = False
        lines.append(f"  {tid:32s}{rate:>8s}{tok:>20s}{sp:>10s}{spread:>8s}")
        for f in s["failures"]:
            lines.append(f"      run{f['run']} 失败：{f['why'][:50]}（{f['stopped']}）")

    lines.append("")
    if all_stable:
        lines.append(f"  {runs}/{runs} 全过 —— 但**这不是「稳定 100%」**。")
        lines.append(
            "  3 次全过的 95% 置信下界约 0.37（三取三的下界很低），只能说「这 3 次都过了」。"
        )
        lines.append("  要更强的保证，需要更多次重复或更换随机种子后复测。")
    else:
        lines.append("  ⚠ 存在波动：至少一个任务不是每次都过。失败原因见上。")
        lines.append("  波动本身就是数据：它说明这类任务的判据处在模型的边缘能力上。")

    spreads = [s["token_spread"] for s in summary.values()]
    if spreads:
        lines.append("")
        lines.append(
            f"  token 极差：中位 {sorted(spreads)[len(spreads) // 2]}，最大 {max(spreads)}"
        )
        lines.append("  极差大说明同一任务的花费不稳定 —— 预算要按上限留余量，不能按均值。")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="关键任务稳定性抽检")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default="trae/qwen-3.5")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--outdir", type=Path, default=Path("qa_runs_w11_stability"))
    args = parser.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = asyncio.run(
        run_stability(runs=args.runs, model=args.model, outdir=args.outdir, live=args.live)
    )

    for line in render(summary, args.runs):
        print(line)

    (args.outdir / "stability.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n落盘：{args.outdir}/stability.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
