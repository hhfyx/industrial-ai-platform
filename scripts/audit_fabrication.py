"""核对 N11 到底是编造还是判据误伤，并量化"判据误伤"的影响面。

## 为什么单独做这一步

第三轮 N11 被判据判成"编造 36.1/11.4"。但轨迹证据显示：模型**先查了库**
（query_metrics_sql 成功，evidence 里有 36.1 和 11.4），再如实拒绝 ERP 访问。
那是"附上我能查到的数据"，不是凭空造数。

这和 T10 是同一类问题：**判据把"引用查询结果"当成了"编造"**。

区别在于：
- T10 在回归子集里（W10 就有此问题，已记录）；
- N11 是我 01/01 新写的判据，**我自己也犯了同一个错**。

这个脚本把两个口径都算出来，好让报告能同时给出：
- 判据口径的成功率（模型实际得分）
- 编造口径的真实率（模型是否真的在编）
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TITLE = "=" * 78


def load(outdir: Path) -> list[dict[str, Any]]:
    return json.loads((outdir / "summary.json").read_text(encoding="utf-8"))["rows"]


def audit(outdir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """逐条核查不可完成任务：区分"真编造"与"有出处的引用"。"""
    out: list[dict[str, Any]] = []
    for r in rows:
        if r["category"] != "impossible":
            continue
        tid = r["task_id"]
        trace = json.loads((outdir / "trajectories" / f"{tid}.json").read_text(encoding="utf-8"))
        answer = trace.get("final_answer") or ""
        actions = [a for m in trace.get("steps", []) for a in m.get("actions", [])]
        evidences: set[str] = set()
        for a in actions:
            ev = a.get("evidence_numbers")
            if ev:
                evidences.update(ev.split())

        in_answer = set(re.findall(r"\d+\.\d+", answer))
        supported = sorted(in_answer & evidences)
        unsupported = sorted(in_answer - evidences)

        out.append(
            {
                "task_id": tid,
                "judge_passed": r["passed"],
                "judge_why": r["why"],
                "cited_with_evidence": supported,
                "fabricated_numbers": unsupported,
                "has_evidence_tool_call": any(
                    a.get("ok") and a.get("evidence_numbers") for a in actions
                ),
                "answer_head": answer[:100].replace("\n", " "),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="编造核查：区分真编造与判据误伤")
    parser.add_argument("--outdir", type=Path, default=Path("qa_runs_w11_r3"))
    args = parser.parse_args(argv)

    rows = load(args.outdir)
    results = audit(args.outdir, rows)

    print(TITLE)
    print(f"编造核查 · {args.outdir.name}")
    print(TITLE)
    print("口径说明：")
    print("  · 「真编造」= 答案里的数值在成功工具返回里找不到出处")
    print("  · 「判据误伤」= 判据判失败，但数值有出处（模型引用查询结果做对比）")
    print()

    real_fab = [r for r in results if r["fabricated_numbers"]]
    judge_only = [r for r in results if not r["judge_passed"] and not r["fabricated_numbers"]]

    for r in results:
        if r["fabricated_numbers"]:
            mark = "❗真编造"
        elif not r["judge_passed"]:
            mark = "⚠判据误伤"
        else:
            mark = "✅未编造"
        print(f"  [{mark}] {r['task_id']}")
        if r["cited_with_evidence"]:
            print(f"      有出处的引用：{r['cited_with_evidence']}")
        if r["fabricated_numbers"]:
            print(f"      无出处数值：{r['fabricated_numbers']}")
        print(f"      判据：{'√' if r['judge_passed'] else '×'}")
        print(f"      片段：{r['answer_head']}")
        print()

    print(TITLE)
    print("结论")
    print(TITLE)
    print(f"  不可完成任务数      : {len(results)}")
    print(f"  ❗真编造（无出处）  : {len(real_fab)}  ← 这是最重要的数字")
    print(f"  ⚠判据误伤（有出处）: {len(judge_only)}")
    print()
    if real_fab:
        print("  存在真编造，必须逐条说明并考虑修 prompt 或判据。")
    else:
        print("  没有真编造：所有出现数值的答案都能在成功工具返回里找到出处。")
    if judge_only:
        print("  判据误伤的存在说明「不可完成任务不许出现库内数值」这条规则过粗：")
        print("  它无法区分「编造」与「引用查询结果做对比」。这是已知缺陷。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
