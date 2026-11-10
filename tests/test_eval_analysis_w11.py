"""W11 周六：评测分析的判据与工具（防止归因脚本自己出错）。

## 为什么这些要写测试

今天最重要的一条教训是：**审计脚本自己会错**。

- 第一版把 T10 判成"编造"，因为它只检查"答案里有没有库内数值"；
- 修正后靠 `evidence_numbers` 才判准；
- 而 `evidence_numbers` 是今天才加的埋点。

如果审计脚本没有测试，那个误判会被写进报告，成为"Agent 编造了数据"
这样的严重结论 —— 而实际上模型完全诚实。

所以这一组测试守三件事：
1. **编造 vs 引用的区分**（看轨迹出处，不看数值本身）；
2. **预算判定不误伤正确任务**（三次改版的教训）；
3. **轮次汇总不挑樱桃**（区间而非最好一轮）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from app.core.llm.agent_graph import _budget_verdict, _route_after_model, _route_after_tools
from app.core.llm.budget import BudgetConfig, BudgetUsed

# ============================================================ 1. 编造 vs 引用


@pytest.fixture()
def audit_mod() -> Any:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import audit_fabrication

    return audit_fabrication


def _trace_root(tmp_path: Path, task_id: str, *, answer: str, evidence: str | None) -> Path:
    """造一个最小 run 目录：只有 summary.json 和一条轨迹。"""
    import json

    (tmp_path / "trajectories").mkdir(parents=True, exist_ok=True)
    (tmp_path / "trajectories" / f"{task_id}.json").write_text(
        json.dumps(
            {
                "final_answer": answer,
                "steps": [
                    {
                        "step": 1,
                        "mode": "fc",
                        "actions": [
                            {
                                "tool": "query_metrics_sql",
                                "ok": True,
                                "evidence_numbers": evidence,
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "task_id": task_id,
                        "category": "impossible",
                        "passed": False,
                        "why": "判据失败",
                        "steps": 1,
                        "tokens": 100,
                        "latency_ms": 100,
                        "stopped_reason": "completed",
                        "degraded": False,
                        "tool_calls": 1,
                        "tool_errors": 0,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_cited_numbers_are_not_fabrication(audit_mod: Any, tmp_path: Path) -> None:
    """**引用有出处的数值 ≠ 编造** —— 这是今天最重要的判定。

    T10/N11 都踩过这条：模型查到 42.5/8.2 后引用它们来证明
    "该批次只有这两项"，被判据和初版审计脚本都判成了编造。
    """
    root = _trace_root(
        tmp_path,
        "X_cite",
        answer="该批次没有放射性指标。目前只有抗压强度 42.5、含水量 8.2。",
        evidence="42.5 8.2 40.0",
    )
    rows = audit_mod.load(root)
    result = audit_mod.audit(root, rows)[0]
    assert result["cited_with_evidence"] == ["42.5", "8.2"]
    assert result["fabricated_numbers"] == []


def test_number_without_evidence_is_fabrication(audit_mod: Any, tmp_path: Path) -> None:
    """**没有出处的数值 = 编造**。这才是要抓的。"""
    root = _trace_root(
        tmp_path,
        "X_fake",
        answer="该批次的放射性指标是 3.7 Bq/kg。",
        evidence="42.5 8.2",
    )
    rows = audit_mod.load(root)
    result = audit_mod.audit(root, rows)[0]
    assert result["fabricated_numbers"] == ["3.7"]


def test_no_evidence_field_means_all_numbers_unsupported(audit_mod: Any, tmp_path: Path) -> None:
    """没有埋点时，任何数值都算无出处 —— **这是刻意的保守**。

    宁可把"无法证明"报成"可疑"，也不要放过真编造。
    同时它提醒：补埋点之前的轮次，"0 编造"这个结论不成立。
    """
    root = _trace_root(tmp_path, "X_noev", answer="值是 42.5", evidence=None)
    rows = audit_mod.load(root)
    result = audit_mod.audit(root, rows)[0]
    assert result["fabricated_numbers"] == ["42.5"]


# ============================================================ 2. 预算不误伤


def _state(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "instruction": "t",
        "model": "fake",
        "max_steps": 10,
        "step_no": 2,
        "messages": [],
        "steps": [],
    }
    base.update(kw)
    return base


def test_answer_step_not_blocked_by_prediction() -> None:
    """**给答案的一步不该被"预判"截断**（N12 的教训）。

    已用 8492（未超 10k），上一步花了 2987。如果按"下一步也会花 2987"
    预判，就会把即将给出的正确答案截断。修完后：不预判，只看实际用量。
    """
    st = _state(
        budget=BudgetConfig(max_tokens=10_000).to_dict(),
        budget_used=BudgetUsed(tokens=8492, steps=3).to_dict(),
        pending_mode="final",
        steps=[{"step": 3, "tokens": {"total_tokens": 2987}}],
    )
    assert _budget_verdict(st).stop is False


def test_tool_step_still_blocked_when_over() -> None:
    """真超了还是要停（不能因为怕误伤而放过超支）。"""
    st = _state(
        budget=BudgetConfig(max_tokens=10_000).to_dict(),
        budget_used=BudgetUsed(tokens=13_869, steps=4).to_dict(),
        pending_mode="fc",
        steps=[{"step": 4, "tokens": {"total_tokens": 3000}}],
    )
    verdict = _budget_verdict(st)
    assert verdict.stop is True
    assert verdict.reason == "budget_tokens"


def test_model_route_checks_budget_before_finalize() -> None:
    """模型直接给答案时也要过预算判定（v1 的漏洞）。"""
    over = _state(
        budget=BudgetConfig(max_tokens=1000).to_dict(),
        budget_used=BudgetUsed(tokens=1200, steps=2).to_dict(),
        pending_mode="final",
    )
    assert _route_after_model(over) == "finalize_budget"

    under = _state(
        budget=BudgetConfig(max_tokens=10_000).to_dict(),
        budget_used=BudgetUsed(tokens=1200, steps=2).to_dict(),
        pending_mode="final",
    )
    assert _route_after_model(under) == "finalize"


def test_tools_route_checks_budget() -> None:
    over = _state(
        budget=BudgetConfig(max_tokens=1000).to_dict(),
        budget_used=BudgetUsed(tokens=1200, steps=2).to_dict(),
        pending_mode="fc",
    )
    assert _route_after_tools(over) == "finalize_budget"


def test_no_progress_still_stops() -> None:
    """去掉预判不等于去掉早停。无进展仍要停。"""
    st = _state(
        budget=BudgetConfig(max_tokens=10_000, no_progress_limit=2).to_dict(),
        budget_used=BudgetUsed(tokens=100, steps=3, no_progress_streak=2).to_dict(),
        pending_mode="final",
    )
    verdict = _budget_verdict(st)
    assert verdict.stop is True
    assert verdict.reason == "no_progress"


# ============================================================ 3. 轮次汇总不挑樱桃


def test_rounds_summary_reports_range(tmp_path: Path, capsys: Any) -> None:
    """汇总必须报**区间**，且把波动任务单独列出来。"""
    import json

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import rounds_summary

    def write_round(name: str, results: dict[str, bool]) -> None:
        d = tmp_path / f"qa_runs_test_{name}"
        d.mkdir(parents=True)
        rows = [
            {
                "task_id": tid,
                "category": "single",
                "passed": ok,
                "why": "",
                "steps": 2,
                "tokens": 5000,
                "latency_ms": 10000,
                "stopped_reason": "completed",
                "degraded": False,
                "tool_calls": 1,
                "tool_errors": 0,
            }
            for tid, ok in results.items()
        ]
        (d / "summary.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")

    write_round("a", {"A": True, "B": True})
    write_round("b", {"A": True, "B": False})

    rounds = {
        "a": rounds_summary.load(tmp_path / "qa_runs_test_a"),
        "b": rounds_summary.load(tmp_path / "qa_runs_test_b"),
    }
    assert all(v is not None for v in rounds.values())

    # A 稳定通过、B 波动 —— 汇总逻辑要能分类
    a_rows = rounds["a"]
    b_rows = rounds["b"]
    assert a_rows and b_rows
    per = {}
    for tid in ("A", "B"):
        marks = [
            next(r["passed"] for r in rows if r["task_id"] == tid) for rows in (a_rows, b_rows)
        ]
        per[tid] = marks
    assert per["A"] == [True, True]  # 稳定
    assert per["B"] == [True, False]  # 波动


def test_fabrication_markers_still_defined() -> None:
    """编造标记清单没被删（它是 W9 的冻结口径之一）。"""
    from app.core.llm.tasks_20 import FABRICATION_MARKERS

    assert "42.5" in FABRICATION_MARKERS
    assert "8.2" in FABRICATION_MARKERS
