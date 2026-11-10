"""汇总所有轮次，给出**区间**而不是挑最好的一轮（周六收尾）。

## 为什么必须做这个

跑了 5 轮全量，结果分别是：18/20、20/20、17/20、17/20、19/20。

如果只报"最好的一轮 20/20"，那是**挑樱桃**（cherry-picking），
是评测里最脏的做法之一。正确做法是报**区间与波动来源**。

## 输出

| 表 | 内容 |
| --- | --- |
| 轮次总表 | 每轮的六项指标 + 通过数 |
| 每任务通过率 | 20 个任务各自跨 5 轮过了几次 |
| 不稳定任务 | 不是全过也不是全不过的（那些才是真问题）|
| 配置差异 | 哪一轮改了什么（否则轮次之间不可比）|
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final

TITLE = "=" * 90

#: 各轮改了什么。**没有这张表，跨轮比较就是耍流氓**。
ROUND_NOTES: Final[dict[str, str]] = {
    "r1": "基线（工具描述未改 / prompt 未加缺失指标约束 / 预算有漏洞）",
    "r2": "改进 I1（路径）+ I2（缺失指标约束）之后",
    "r3": "修预算漏洞（model_step 后也判预算）—— 引入过度预判",
    "r4": "预判只在调工具时启用 —— 仍误伤 T6/T7/T8",
    "r5": "去掉预判，只看实际用量（当前版本）",
}


def load(outdir: Path) -> list[dict[str, Any]] | None:
    p = outdir / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["rows"]


def six(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    calls = sum(r.get("tool_calls", 0) for r in rows)
    errors = sum(r.get("tool_errors", 0) for r in rows)
    return {
        "passed": passed,
        "total": total,
        "rate": passed / total if total else 0.0,
        "avg_steps": round(sum(r["steps"] for r in rows) / total, 2) if total else 0,
        "avg_tokens": round(sum(r["tokens"] for r in rows) / total, 0) if total else 0,
        "avg_latency": round(sum(r["latency_ms"] for r in rows) / total, 0) if total else 0,
        "error_rate": errors / calls if calls else 0.0,
        "degraded": sum(1 for r in rows if r.get("degraded")),
        "degrade_rate": sum(1 for r in rows if r.get("degraded")) / total if total else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="汇总所有轮次")
    parser.add_argument(
        "--rounds",
        nargs="*",
        default=["r1", "r2", "r3", "r4", "r5"],
        help="轮次目录名（qa_runs_w11_<name>）",
    )
    parser.add_argument("--base", default="qa_runs_w11_")
    args = parser.parse_args(argv)

    rounds: dict[str, list[dict[str, Any]]] = {}
    for name in args.rounds:
        rows = load(Path(f"{args.base}{name}"))
        if rows:
            rounds[name] = rows

    print(TITLE)
    print("各轮总表（**不要只报最好的一轮**）")
    print(TITLE)
    print(
        f"  {'轮次':6s}{'通过':>10s}{'均步':>7s}{'均token':>10s}"
        f"{'均延迟':>9s}{'错误率':>9s}{'降级率':>9s}"
    )
    print("  " + "-" * 66)
    for name, rows in rounds.items():
        m = six(rows)
        passed_str = f"{m['passed']}/{m['total']}"
        print(
            f"  {name:6s}{passed_str:>10s}"
            f"{m['avg_steps']:>7.2f}{m['avg_tokens']:>10.0f}"
            f"{m['avg_latency']:>8.0f}ms{m['error_rate']:>8.1%}{m['degrade_rate']:>8.1%}"
        )

    print()
    print("  各轮改了什么（跨轮比较的前提）：")
    for name in rounds:
        print(f"    {name}: {ROUND_NOTES.get(name, '（未记录）')}")

    # ---- 每任务通过率 ----
    print()
    print(TITLE)
    print("每任务跨轮通过率")
    print(TITLE)
    task_ids = sorted({r["task_id"] for rows in rounds.values() for r in rows})
    per_task: dict[str, list[bool]] = {tid: [] for tid in task_ids}
    for _name, rows in rounds.items():
        by_id = {r["task_id"]: r["passed"] for r in rows}
        for tid in task_ids:
            if tid in by_id:
                per_task[tid].append(by_id[tid])

    stable_pass, stable_fail, flaky = [], [], []
    for tid in task_ids:
        marks = per_task[tid]
        n_pass = sum(marks)
        n = len(marks)
        bar = "".join("√" if m else "×" for m in marks)
        flag = ""
        if n_pass == n:
            stable_pass.append(tid)
            flag = "稳定通过"
        elif n_pass == 0:
            stable_fail.append(tid)
            flag = "稳定失败"
        else:
            flaky.append((tid, n_pass, n))
            flag = "**波动**"
        print(f"  {tid:34s} {bar}  {n_pass}/{n}  {flag}")

    print()
    print(TITLE)
    print("结论")
    print(TITLE)
    rates = [six(r)["rate"] for r in rounds.values()]
    print(f"  成功率区间：{min(rates):.0%} ~ {max(rates):.0%}（{len(rounds)} 轮）")
    print(f"  稳定通过：{len(stable_pass)} 个")
    print(f"  稳定失败：{len(stable_fail)} 个  {stable_fail}")
    print(f"  **波动**：{len(flaky)} 个")
    for tid, k, n in flaky:
        print(f"      {tid}: {k}/{n}")
    print()
    print("  判读纪律：")
    print("    · 波动任务的通过率只能报区间，不能报单点；")
    print("    · 稳定失败的任务才是「机制没解决」的证据；")
    print("    · 稳定通过的任务不证明机制有效，只证明它没把好事弄坏。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
