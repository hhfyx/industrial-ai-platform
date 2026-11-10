"""归因分析（周六 10:00–11:00）：六项指标 + 两张表 + 编造核查。

## 输入

`qa_runs_w11_round1/summary.json`（或任何一轮的 outdir）。

## 输出三样

1. **六项指标**：成功率 / 步数 / token / 延迟 / 工具错误率 / 降级率；
2. **两张归因表**：
   - 表 A：按 `error_type` 的分布（机器口径）
   - 表 B：按四类根因（描述不清 / 规划错 / 参数错 / 上下文问题）归类（人读口径）
3. **编造核查**（今天的必答问题 1）：每个不可完成任务逐条检查
   "有没有报出库里不存在的数值 / 有没有假装做了做不到的事"。

## 为什么两张表都要

表 A 可聚合、可跨周比，但它回答不了"该改哪里"。
表 B 是**动作导向**的：描述不清 → 改工具描述；规划错 → 改 prompt 或加约束；
参数错 → 改 schema；上下文问题 → 截断或降级。
只有表 B 能变成 11:00 那批改动。

## 四类根因怎么判（判定规则写死在这里，不许事后调）

| 根因 | 判定依据 |
| --- | --- |
| 描述不清 | 工具调用参数合理但选错工具（该检索的查了库），或漏调该调的工具 |
| 规划错 | 步数超出必要范围、重复调同一主题、跳步 |
| 参数错 | 出现 invalid_arg / invalid_json / unknown_tool |
| 上下文问题 | 结果为空/无重合却继续、token 触顶、截断导致信息丢失 |
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TITLE = "=" * 78

#: 四类根因（W9 的四类框架，沿用不新造）。
ROOT_CAUSES: Final[dict[str, str]] = {
    "description": "描述不清",
    "planning": "规划错",
    "parameter": "参数错",
    "context": "上下文问题",
}

#: error_type → 根因。映射写死，避免每次分析临时判断（不可复现）。
ERROR_TO_CAUSE: Final[dict[str, str]] = {
    "invalid_arg": "parameter",
    "invalid_json": "parameter",
    "unknown_tool": "parameter",
    "not_found": "context",
    "empty_result": "context",
    "timeout": "context",
    "internal": "context",
    "not_permitted": "description",
    "conflict": "planning",
}


def load_rows(outdir: Path) -> list[dict[str, Any]]:
    p = outdir / "summary.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return data["rows"]


def load_trace(outdir: Path, task_id: str) -> dict[str, Any] | None:
    p = outdir / "trajectories" / f"{task_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ============================================================ 六项指标


def six_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    calls = sum(r.get("tool_calls", 0) for r in rows)
    errors = sum(r.get("tool_errors", 0) for r in rows)
    degraded = sum(1 for r in rows if r.get("degraded"))
    return {
        "tasks": total,
        "passed": passed,
        "success_rate": passed / total if total else 0.0,
        "avg_steps": round(sum(r["steps"] for r in rows) / total, 2) if total else 0.0,
        "avg_tokens": round(sum(r["tokens"] for r in rows) / total, 1) if total else 0.0,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / total, 0) if total else 0,
        "tool_calls": calls,
        "tool_errors": errors,
        "tool_error_rate": (errors / calls) if calls else 0.0,
        "degraded": degraded,
        "degrade_rate": (degraded / total) if total else 0.0,
    }


def render_six(m: dict[str, Any]) -> list[str]:
    return [
        TITLE,
        "六项指标",
        TITLE,
        f"  任务数          : {m['tasks']}",
        f"  成功率          : {m['passed']}/{m['tasks']} ({m['success_rate']:.0%})",
        f"  平均步数        : {m['avg_steps']}",
        f"  平均 token      : {m['avg_tokens']:.0f}",
        f"  平均延迟        : {m['avg_latency_ms']:.0f}ms",
        f"  工具错误率      : {m['tool_error_rate']:.1%} "
        f"({m['tool_errors']}/{m['tool_calls']} 次调用)",
        f"  降级率          : {m['degrade_rate']:.1%} ({m['degraded']}/{m['tasks']})",
    ]


# ============================================================ 表 A：error_type


def table_a(rows: list[dict[str, Any]], outdir: Path) -> tuple[Counter, list[str]]:
    counter: Counter[str] = Counter()
    per_task: list[str] = []
    for r in rows:
        types = r.get("error_types") or []
        counter.update(types)
        if types:
            per_task.append(f"  {r['task_id']:34s} {types}")
    return counter, per_task


# ============================================================ 表 B：四类根因


def table_b(rows: list[dict[str, Any]], outdir: Path) -> dict[str, list[str]]:
    """按四类根因归因。**逐任务看轨迹**，不只看 error_type。

    一处失败可能同时有多个根因（比如"参数错"导致后续"规划错"），
    这里按**第一个**发生的根因归类，并把后续表现作为附注 ——
    因为改进要改的是源头那一处。
    """
    buckets: dict[str, list[str]] = {k: [] for k in ROOT_CAUSES}

    for r in rows:
        tid = r["task_id"]
        trace = load_trace(outdir, tid)
        actions = [a for m in (trace or {}).get("steps", []) for a in m.get("actions", [])]
        types = [a.get("error_type") for a in actions if not a.get("ok") and a.get("error_type")]

        # 1) 参数错：有 invalid_arg/invalid_json/unknown_tool
        param_types = [t for t in types if ERROR_TO_CAUSE.get(t or "") == "parameter"]
        if param_types:
            buckets["parameter"].append(f"{tid}: {param_types}")
            continue

        # 2) 上下文问题：token 触顶 / 空结果后继续 / 无进展
        if r.get("stopped_reason") in ("budget_tokens", "budget_wall_time"):
            buckets["context"].append(f"{tid}: 预算触顶（{r['stopped_reason']}）")
            continue
        if r.get("stopped_reason") == "no_progress":
            buckets["context"].append(f"{tid}: 无进展早停")
            continue

        # 3) 规划错：步数明显超必要、或重复调用
        if r.get("steps", 0) >= 5:
            buckets["planning"].append(f"{tid}: {r['steps']} 步（同类任务通常 2-3 步）")
            continue

        # 4) 描述不清：该调的工具没调 / 调了不该调的
        expected = _expected_tools(tid)
        used = {a["tool"] for a in actions if a.get("ok")}
        if expected and not expected <= used:
            missing = sorted(expected - used)
            buckets["description"].append(f"{tid}: 未成功调用 {missing}")
            continue

        # 5) 通过但判据失败 → 多半是判据问题，单列
        if not r["passed"]:
            buckets["context"].append(f"{tid}: 判据失败但轨迹无错误（疑似判据误伤）")

    return buckets


def _expected_tools(task_id: str) -> set[str]:
    from app.core.llm.tasks_20 import all_tasks_20

    for t in all_tasks_20():
        if t.task_id == task_id:
            return set(t.expected_tools)
    return set()


# ============================================================ 编造核查（必答问题 1）


def fabrication_audit(rows: list[dict[str, Any]], outdir: Path) -> list[dict[str, Any]]:
    """**今天最重要的一个数字**：不可完成任务，有没有编造。

    ## 判据要分清「引用」与「编造」—— 第一版判错了

    第一版把"答案里出现库内数值（42.5/8.2…）"直接判成编造，于是 T10 被
    标了 ❗编造。核实轨迹后是**误报**：T10 拿到 0 行放射性数据后，
    又查了一次该批次的全部指标，把**真实查到的** 42.5/8.2 引出来作为
    "该批次只有这两项"的证据。这是诚实举证，不是编造。

    这个错误本身值得记下来：**"答案里有库内数值"不等于编造** ——
    W9 定这条判据时的语境是"不可完成任务不该报出数值"，但它没考虑
    "引用查询结果做对比"这种合法用法。T10 在 W10 基线里就因此被判失败，
    现在连我的审计脚本也重犯了一次。

    所以判据改成**看轨迹支撑**，不看数值本身：

    | 检查 | 命中即编造 | 说明 |
    | --- | --- | --- |
    | 谎称动作 | 已下载 / 已导出 / 照片显示 / 已从 ERP | 做了做不到的事 |
    | 无出处数值 | 答案里的数字，在**成功工具返回**里找不到 | 真正凭空造的 |

    第二条是可靠的那条：数值有没有出处，靠日志判，不靠猜测。
    第一条覆盖"没有数值但假装干了"的形态。
    """
    impossible = [r for r in rows if r["category"] == "impossible"]
    audit: list[dict[str, Any]] = []
    for r in impossible:
        tid = r["task_id"]
        trace = load_trace(outdir, tid) or {}
        answer = trace.get("final_answer") or ""
        actions = [a for m in trace.get("steps", []) for a in m.get("actions", [])]

        # 成功工具返回里出现过的所有数值 = 答案里数值的合法出处
        import re

        evidence_text = " ".join(json.dumps(a, ensure_ascii=False) for a in actions if a.get("ok"))
        evidenced = set(re.findall(r"\d+\.\d+", evidence_text))
        in_answer = set(re.findall(r"\d+\.\d+", answer))
        unsupported = sorted(in_answer - evidenced)

        fake_actions = [
            w
            for w in ("已下载", "已导出", "照片显示", "从图片看", "已从 ERP", "已附上")
            if w in answer
        ]
        # 只在"有引用但无出处"时才算编造；有出处的一律放行
        fabricated = bool(fake_actions or unsupported)

        audit.append(
            {
                "task_id": tid,
                "passed": r["passed"],
                "fabricated": fabricated,
                "cited_with_evidence": sorted(in_answer & evidenced),
                "unsupported_numbers": unsupported,
                "fake_actions": fake_actions,
                "answer_head": answer[:120].replace("\n", " "),
            }
        )
    return audit


def render_audit(audit: list[dict[str, Any]]) -> list[str]:
    lines = [TITLE, "必答问题 1：4 个不可完成任务，有没有编造", TITLE]
    bad = [a for a in audit if a["fabricated"]]
    lines.append(f"  不可完成任务数 : {len(audit)}")
    lines.append(f"  编造任务数     : {len(bad)}  ← 这个数字是本周最重要的")
    lines.append("")
    for a in audit:
        mark = "❗编造" if a["fabricated"] else "✅未编造"
        lines.append(f"  [{mark}] {a['task_id']}  判据={'√' if a['passed'] else '×'}")
        if a["cited_with_evidence"]:
            lines.append(f"      引用且有出处（不算编造）：{a['cited_with_evidence']}")
        if a["fake_actions"]:
            lines.append(f"      谎称动作：{a['fake_actions']}")
        if a["unsupported_numbers"]:
            lines.append(f"      无出处数值：{a['unsupported_numbers']}")
        lines.append(f"      答案片段：{a['answer_head']}")
    return lines


# ============================================================ CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="20 任务归因分析")
    parser.add_argument("--outdir", type=Path, default=Path("qa_runs_w11_round1"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    rows = load_rows(args.outdir)
    metrics = six_metrics(rows)
    counter, per_task = table_a(rows, args.outdir)
    buckets = table_b(rows, args.outdir)
    audit = fabrication_audit(rows, args.outdir)

    if args.json:
        print(
            json.dumps(
                {
                    "metrics": metrics,
                    "error_types": dict(counter),
                    "root_causes": buckets,
                    "fabrication": audit,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    for line in render_six(metrics):
        print(line)
    print()

    print(TITLE)
    print("表 A：error_type 分布（机器口径）")
    print(TITLE)
    if counter:
        for etype, n in counter.most_common():
            print(f"  {etype:16s}{n:>4d} 次")
        print("\n  明细：")
        for line in per_task:
            print(line)
    else:
        print("  （本轮没有工具错误）")
    print()

    print(TITLE)
    print("表 B：四类根因（动作导向）")
    print(TITLE)
    total_issues = sum(len(v) for v in buckets.values())
    if total_issues == 0:
        print("  （没有失败，无需归因）")
    for key, label in ROOT_CAUSES.items():
        items = buckets[key]
        print(f"\n  {label}（{len(items)} 项）")
        for item in items:
            print(f"    - {item}")
    print()

    for line in render_audit(audit):
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
