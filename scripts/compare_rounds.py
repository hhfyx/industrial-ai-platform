"""两轮对比：六项指标并排 + 配对胜负 + 边际收益（周六 14:30–15:30）。

## 为什么必须"配对"

整体成功率从 90% 涨到 100%，但这个数字本身不能证明改进有效：
模型有随机性，整体涨 2 个点可能全是运气。

配对（paired）只看**同一个任务**在两轮里的变化：
- 改进前 × → 改进后 √ ：**胜**（+1）
- 改进前 √ → 改进后 × ：**负**（-1）
- 两轮都 √ / 都 × ：平（0）

只有"胜 > 0 且负 = 0"才能说"改进有效且没引入回归"。
如果出现"胜 1 负 1"（净 0），那是**换了一个问题**，不是修好了 ——
这种必须单独说明，不能笼统写成"改进后持平"。

## 边际收益怎么算

按改动逐一归因（一次只改一处是纪律，所以能归）：

    某处改动的收益 = 该处针对的任务里，由 × 变 √ 的个数
    该处改动的成本 = 全 20 个任务的 token 增量（那条约束对所有任务生效）
    边际收益 = 收益点数 / token 增量

**成本要按全部任务算，不能只按被修复的任务算** ——
系统提示词的增量是所有任务都付的，只按受益任务算是自欺。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TITLE = "=" * 78

#: 本轮的两处改动（11:00–12:00 的改动清单）。
#: 每次只改一处，所以能这样逐条登记 —— 一次改三处就没法归因了。
IMPROVEMENTS: Final[list[dict[str, Any]]] = [
    {
        "id": "I1",
        "layer": "工具描述",
        "where": "GENERATE_REPORT_FILE_DESCRIPTION",
        "what": "明确要求「拿到 path 后必须在最终答案里原样写出路径」",
        "targets": ["T7_complex_report", "T8_complex_notify"],
        "why": "T7 报告已生成但答案只说「报告已生成」没给路径，判据（含路径）失败",
    },
    {
        "id": "I2",
        "layer": "prompt",
        "where": "MISSING_METRIC_CONSTRAINT（新增增量约束）",
        "what": "问某指标而库里没有时，只列指标名称、不列其他指标的数值",
        "targets": ["T9_impossible_batch", "T10_impossible_metric"],
        "why": "T10 诚实说明缺失，但引用了其他指标的数值作对比，被冻结判据误判为编造",
    },
]


def load(outdir: Path) -> dict[str, dict[str, Any]]:
    data = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
    return {r["task_id"]: r for r in data["rows"]}


def six(m: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(m.values())
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    calls = sum(r.get("tool_calls", 0) for r in rows)
    errors = sum(r.get("tool_errors", 0) for r in rows)
    return {
        "passed": passed,
        "success_rate": passed / total,
        "avg_steps": round(sum(r["steps"] for r in rows) / total, 2),
        "avg_tokens": round(sum(r["tokens"] for r in rows) / total, 1),
        "total_tokens": sum(r["tokens"] for r in rows),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / total),
        "tool_errors": errors,
        "tool_calls": calls,
        "error_rate": errors / calls if calls else 0.0,
        "degraded": sum(1 for r in rows if r.get("degraded")),
        "degrade_rate": sum(1 for r in rows if r.get("degraded")) / total,
    }


def render_metrics(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    def row(label: str, x: str, y: str, delta: str) -> str:
        return f"  {label:16s}{x:>14s}{y:>14s}{delta:>14s}"

    def pct(v: float) -> str:
        return f"{v:.1%}"

    lines = [
        TITLE,
        "六项指标并排",
        TITLE,
        row("指标", "第一轮（改前）", "第二轮（改后）", "变化"),
        "  " + "-" * 56,
        row(
            "成功率",
            f"{a['passed']}/20 ({pct(a['success_rate'])})",
            f"{b['passed']}/20 ({pct(b['success_rate'])})",
            f"{(b['success_rate'] - a['success_rate']) * 100:+.0f} pp",
        ),
        row(
            "平均步数",
            f"{a['avg_steps']}",
            f"{b['avg_steps']}",
            f"{b['avg_steps'] - a['avg_steps']:+.2f}",
        ),
        row(
            "平均 token",
            f"{a['avg_tokens']:.0f}",
            f"{b['avg_tokens']:.0f}",
            f"{b['avg_tokens'] - a['avg_tokens']:+.0f}",
        ),
        row(
            "平均延迟",
            f"{a['avg_latency_ms']:.0f}ms",
            f"{b['avg_latency_ms']:.0f}ms",
            f"{b['avg_latency_ms'] - a['avg_latency_ms']:+.0f}ms",
        ),
        row(
            "工具错误率",
            f"{pct(a['error_rate'])} ({a['tool_errors']}/{a['tool_calls']})",
            f"{pct(b['error_rate'])} ({b['tool_errors']}/{b['tool_calls']})",
            f"{(b['error_rate'] - a['error_rate']) * 100:+.1f} pp",
        ),
        row(
            "降级率",
            pct(a["degrade_rate"]),
            pct(b["degrade_rate"]),
            f"{(b['degrade_rate'] - a['degrade_rate']) * 100:+.0f} pp",
        ),
    ]
    return lines


def paired(ma: dict[str, dict[str, Any]], mb: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """配对胜负：只看同一任务在两轮里的变化。"""
    ids = sorted(set(ma) & set(mb))
    wins, losses, both_pass, both_fail = [], [], [], []
    for tid in ids:
        pa, pb = ma[tid]["passed"], mb[tid]["passed"]
        if not pa and pb:
            wins.append(tid)
        elif pa and not pb:
            losses.append(tid)
        elif pa and pb:
            both_pass.append(tid)
        else:
            both_fail.append(tid)
    return {
        "ids": ids,
        "wins": wins,
        "losses": losses,
        "both_pass": both_pass,
        "both_fail": both_fail,
        "net": len(wins) - len(losses),
    }


def render_paired(p: dict[str, Any]) -> list[str]:
    lines = [
        TITLE,
        "配对胜负表（同任务、改前 vs 改后）",
        TITLE,
        f"  胜（× → √）：{len(p['wins'])}  {p['wins']}",
        f"  负（√ → ×）：{len(p['losses'])}  {p['losses']}",
        f"  两轮都 √   ：{len(p['both_pass'])}",
        f"  两轮都 ×   ：{len(p['both_fail'])}  {p['both_fail']}",
        f"  净胜       ：{p['net']:+d}",
        "",
    ]
    if not p["losses"]:
        lines.append("  ✅ 没有回归：改进没有把任何原本通过的任务弄坏。")
    else:
        lines.append(f"  ⚠ 有回归：{p['losses']} —— 必须单独归因，不能笼统写「改进后持平」。")
    return lines


def marginal(
    improvements: list[dict[str, Any]],
    ma: dict[str, dict[str, Any]],
    mb: dict[str, dict[str, Any]],
    sa: dict[str, Any],
    sb: dict[str, Any],
) -> list[dict[str, Any]]:
    """逐条改动的边际收益。成本按**全部任务**的 token 增量算。"""
    detail: list[dict[str, Any]] = []
    for imp in improvements:
        fixed = [
            tid
            for tid in imp["targets"]
            if tid in ma and tid in mb and not ma[tid]["passed"] and mb[tid]["passed"]
        ]
        broke = [
            tid
            for tid in imp["targets"]
            if tid in ma and tid in mb and ma[tid]["passed"] and not mb[tid]["passed"]
        ]
        detail.append(
            {
                "id": imp["id"],
                "layer": imp["layer"],
                "where": imp["where"],
                "what": imp["what"],
                "why": imp["why"],
                "targets": imp["targets"],
                "fixed": fixed,
                "broke": broke,
                "gain": len(fixed) - len(broke),
            }
        )
    return detail


def render_marginal(
    detail: list[dict[str, Any]], sa: dict[str, Any], sb: dict[str, Any]
) -> list[str]:
    lines = [
        TITLE,
        "边际收益（按改动逐条）",
        TITLE,
    ]
    for d in detail:
        lines.append(f"\n  [{d['id']}] {d['layer']}：{d['where']}")
        lines.append(f"      改法：{d['what']}")
        lines.append(f"      起因：{d['why']}")
        lines.append(f"      修好：{d['fixed'] or '（无）'}")
        if d["broke"]:
            lines.append(f"      ⚠ 弄坏：{d['broke']}")
        lines.append(f"      净收益：{d['gain']:+d} 个任务")

    delta = sb["total_tokens"] - sa["total_tokens"]
    per_task = delta / 20
    lines.append("")
    lines.append("  成本（按全部 20 个任务分摊，因为约束对所有任务生效）：")
    lines.append(f"    总 token 变化：{delta:+d}（均摊每个任务 {per_task:+.0f}）")
    lines.append(
        f"    成功率的净收益：+{(sb['success_rate'] - sa['success_rate']) * 100:.0f} 个百分点"
    )
    if delta > 0:
        lines.append(
            f"    每 +1 个百分点成功率，代价约 {delta / max(1, (sb['success_rate'] - sa['success_rate']) * 100):.0f} token"
        )
    lines.append("")
    lines.append("  读法提示：token 涨了不等于亏。要看「净胜」与「每次净胜的 token 代价」，")
    lines.append("  再和预算上限（10k）对照 —— 只要没顶到上限，涨的是余量不是风险。")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="两轮对比")
    parser.add_argument("--before", type=Path, default=Path("qa_runs_w11_r1"))
    parser.add_argument("--after", type=Path, default=Path("qa_runs_w11_r2"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    ma, mb = load(args.before), load(args.after)
    sa, sb = six(ma), six(mb)
    p = paired(ma, mb)
    detail = marginal(IMPROVEMENTS, ma, mb, sa, sb)

    if args.json:
        print(
            json.dumps(
                {"before": sa, "after": sb, "paired": p, "improvements": detail},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    for line in render_metrics(sa, sb):
        print(line)
    print()
    for line in render_paired(p):
        print(line)
    print()
    for line in render_marginal(detail, sa, sb):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
