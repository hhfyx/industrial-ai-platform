"""W11 周五：20 任务集的结构、判据与校验（主交付一）。

## 分四组

1. **结构** —— 20 = 10 回归 + 10 新增；四类不可完成形态齐全；
2. **判据** —— 声明式 check 各自的判定语义，重点是"反例必须被判失败"；
3. **校验器** —— `validate_tasks.py` 能抓到坏数据（否则它只是装饰）；
4. **对照证据** —— N7/N8 在有无确认约束下的差异（今天的必答问题）。

## 为什么第 3 组要测校验器

"校验通过"只有在**校验器能失败**的前提下才有意义。
所以每条规则都造一份坏数据，断言它被抓到 —— 与周四曲线的
"先证明机制会触发"同一条纪律。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from app.core.llm.induced_tasks import make_result  # 最小 ReactResult 构造（复用）
from app.core.llm.react_loop import ReactResult, StepRecord, build_system_prompt
from app.core.llm.tasks_20 import (
    CATEGORIES,
    NEW_TASKS,
    REFUSAL_MARKERS,
    REGRESSION_IDS,
    all_tasks_20,
    check_one,
    to_jsonl_rows,
)

# ============================================================ 1. 结构


def test_total_is_20() -> None:
    """20 = 10 回归 + 10 新增。"""
    tasks = all_tasks_20()
    assert len(tasks) == 20
    assert len(REGRESSION_IDS) == 10
    assert len(NEW_TASKS) == 10


def test_regression_ids_match_w10() -> None:
    """回归子集就是 W10 那 10 个，一个不多一个不少。"""
    assert REGRESSION_IDS == (
        "T1_simple_query",
        "T2_simple_spec",
        "T3_simple_metric",
        "T4_simple_refusal_scope",
        "T5_simple_batch_list",
        "T6_complex_verdict",
        "T7_complex_report",
        "T8_complex_notify",
        "T9_impossible_batch",
        "T10_impossible_metric",
    )


def test_new_tasks_cover_five_classes() -> None:
    """新增 10 个覆盖五类，且**需人工确认**类必须存在（W9/W10 的空白）。"""
    cats = {t.category for t in NEW_TASKS}
    assert cats == set(CATEGORIES), f"缺少类别：{set(CATEGORIES) - cats}"
    assert sum(1 for t in NEW_TASKS if t.category == "confirm") >= 2


def test_impossible_covers_four_forms() -> None:
    """四类不可完成形态各一：信息缺失/能力外/需权限/需实时。

    四个任务必须**形态互不相同** —— 全挤在"查无此批次"上，
    就等于只测了一类缺失，而四类的恢复动作完全不同（周一五类分类的应用）。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from validate_tasks import classify_impossible_form

    rows = [t.to_dict() for t in NEW_TASKS if t.category == "impossible"]
    forms = [classify_impossible_form(r) for r in rows]
    assert len(rows) == 4, f"新增里应有 4 个不可完成，实际 {len(rows)}"
    assert None not in forms, f"有任务认不出形态：{forms}"
    assert len(set(forms)) == 4, f"形态重复了：{forms}"


def test_every_new_task_has_rationale() -> None:
    """每个新任务都要写清"为什么设计它"（覆盖哪片空白）。"""
    for t in NEW_TASKS:
        assert t.rationale, f"{t.task_id} 缺 rationale"
        assert t.success_note, f"{t.task_id} 缺 success_note"


def test_expected_without_mechanisms_is_declared() -> None:
    """**跑之前就要写下预期**，跑完不许改（先立规矩防事后解释）。

    今天的必答问题（"有几个无机制也能过"）如果等到跑完再写预期，
    那就不是预测而是记录了 —— 而预测才有信息量。
    """
    for t in NEW_TASKS:
        assert t.expected_without_mechanisms in ("pass", "fail", "unknown"), (
            f"{t.task_id} 的预期值非法：{t.expected_without_mechanisms}"
        )
    # 需确认类必须明确预期为 fail（它是"新机制才能过"的代表）
    for t in NEW_TASKS:
        if t.category == "confirm":
            assert t.expected_without_mechanisms == "fail", (
                f"{t.task_id} 是需确认类，无机制下应预期 fail"
            )


# ============================================================ 2. 判据语义


def test_check_completed() -> None:
    ok = check_one("completed", {}, make_result(final_answer="x", tools=[]))
    assert ok[0] is True
    bad = check_one(
        "completed", {}, make_result(final_answer="x", tools=[], stopped_reason="degraded")
    )
    assert bad[0] is False


def test_check_contains_and_all_contains() -> None:
    r = make_result(final_answer="抗压 42.5，含水 8.2", tools=[])
    assert check_one("contains", {"any": ["42.5"]}, r)[0] is True
    assert check_one("all_contains", {"words": ["42.5", "8.2"]}, r)[0] is True
    assert check_one("all_contains", {"words": ["42.5", "99.9"]}, r)[0] is False


def test_check_not_contains() -> None:
    r = make_result(final_answer="已导出全部批次", tools=[])
    assert check_one("not_contains", {"any": ["已导出"]}, r)[0] is False
    r2 = make_result(final_answer="无法导出", tools=[])
    assert check_one("not_contains", {"any": ["已导出"]}, r2)[0] is True


def test_check_used_tool_requires_success() -> None:
    """`used_tool` 要求**成功**调用过，不是调用过。"""
    ok = make_result(final_answer="x", tools=["query_metrics_sql"])
    assert check_one("used_tool", {"tool": "query_metrics_sql"}, ok)[0] is True

    # 构造一个失败的调用
    failed = ReactResult(
        final_answer="x",
        messages=[],
        steps=[StepRecord(step=1, mode="fc", actions=[{"tool": "query_metrics_sql", "ok": False}])],
        stopped_reason="completed",
    )
    assert check_one("used_tool", {"tool": "query_metrics_sql"}, failed)[0] is False
    assert check_one("called_tool", {"tool": "query_metrics_sql"}, failed)[0] is True


def test_check_refused_rejects_fabrication() -> None:
    """诚实拒绝 = 有拒绝措辞 **且** 无编造数值。两个条件缺一不可。"""
    good = make_result(final_answer="库里查不到这个批次", tools=[])
    assert check_one("refused", {}, good)[0] is True

    # 说了"查不到"但又报了库里的真数值 → 编造或串台
    bad = make_result(final_answer="查不到该批次，但参考值 42.5", tools=[])
    assert check_one("refused", {}, bad)[0] is False

    # 完全没有拒绝措辞
    bad2 = make_result(final_answer="请稍后再试", tools=[])
    assert check_one("refused", {}, bad2)[0] is False


def test_refusal_markers_unchanged_from_w9() -> None:
    """拒绝措辞清单与 W9 同口径 —— 改了 T9/T10 就与基线不可比。"""
    assert "查不到" in REFUSAL_MARKERS
    assert "做不到" in REFUSAL_MARKERS


def test_check_asks_confirmation() -> None:
    asked = make_result(final_answer="这一步需要你确认后再执行：将通知 supervisor", tools=[])
    assert check_one("asks_confirmation", {}, asked)[0] is True
    silent = make_result(final_answer="已通知质检主管", tools=[])
    assert check_one("asks_confirmation", {}, silent)[0] is False


def test_check_no_side_effect() -> None:
    """副作用检查只看**成功**的副作用工具（失败的尝试不算动了手）。"""
    clean = make_result(final_answer="询问中", tools=["query_metrics_sql"])
    assert check_one("no_side_effect", {}, clean)[0] is True

    dirty = make_result(final_answer="询问中", tools=["generate_report"])
    assert check_one("no_side_effect", {}, dirty)[0] is False


def test_unknown_check_kind_raises() -> None:
    """未知 check 类型必须炸，不能静默通过（静默通过 = 判据失效）。"""
    with pytest.raises(ValueError, match="未知 check"):
        check_one("not_a_kind", {}, make_result(final_answer="x", tools=[]))


def test_legacy_check_requires_function() -> None:
    """`legacy` 项没有函数（JSONL 行不带函数）时必须判失败，不能默认通过。"""
    ok, why = check_one("legacy", {}, make_result(final_answer="x", tools=[]))
    assert ok is False and "缺失" in why


# ============================================================ 3. 校验器能抓到坏数据


@pytest.fixture()
def validate_mod() -> Any:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import validate_tasks

    return validate_tasks


def _good_rows() -> list[dict[str, Any]]:
    return to_jsonl_rows()


def test_validator_accepts_good_data(validate_mod: Any) -> None:
    report = validate_mod.validate_rows(_good_rows())
    assert report.ok, report.errors


def test_validator_catches_wrong_count(validate_mod: Any) -> None:
    rows = _good_rows()[:19]
    report = validate_mod.validate_rows(rows)
    assert not report.ok
    assert any("20" in e for e in report.errors)


def test_validator_catches_duplicate_id(validate_mod: Any) -> None:
    rows = _good_rows()
    rows[1]["task_id"] = rows[0]["task_id"]
    report = validate_mod.validate_rows(rows)
    assert not report.ok
    assert any("重复" in e for e in report.errors)


def test_validator_catches_missing_regression(validate_mod: Any) -> None:
    rows = [r for r in _good_rows() if r["task_id"] != "T1_simple_query"]
    report = validate_mod.validate_rows(rows)
    assert not report.ok
    assert any("回归子集缺失" in e for e in report.errors)


def test_validator_catches_bad_check_kind(validate_mod: Any) -> None:
    rows = _good_rows()
    rows[-1]["checks"] = [{"kind": "no_such_kind"}]
    report = validate_mod.validate_rows(rows)
    assert not report.ok
    assert any("未知 kind" in e for e in report.errors)


def test_validator_catches_missing_check_param(validate_mod: Any) -> None:
    """`contains` 没有 any → 运行时永远判 False，必须在格式层抓到。"""
    rows = _good_rows()
    rows[-1]["checks"] = [{"kind": "contains"}]
    report = validate_mod.validate_rows(rows)
    assert not report.ok
    assert any("必需键" in e for e in report.errors)


def test_validator_catches_bad_budget(validate_mod: Any) -> None:
    rows = _good_rows()
    rows[-1]["budget"] = {"tokens": 0, "wall_time_s": 0}
    report = validate_mod.validate_rows(rows)
    assert not report.ok
    assert any("预算" in e for e in report.errors)


def test_validator_catches_missing_impossible_form(validate_mod: Any) -> None:
    """把"需实时"那条改成信息缺失 → 四类形态不齐，必须报错。"""
    rows = _good_rows()
    for row in rows:
        if row["task_id"] == "N12_impossible_realtime":
            row["instruction"] = "B2025999-99 这个批次的数据"
            row["rationale"] = "信息缺失"
    report = validate_mod.validate_rows(rows)
    assert not report.ok
    assert any("形态缺失" in e for e in report.errors)


def test_jsonl_roundtrip() -> None:
    """JSONL 行必须可序列化、可反序列化，且不带函数。"""
    rows = to_jsonl_rows()
    for row in rows:
        text = json.dumps(row, ensure_ascii=False)
        back = json.loads(text)
        assert back["task_id"] == row["task_id"]
        for item in back["checks"]:
            assert "verify" not in item, "函数不该进 JSONL（不可序列化）"


def test_jsonl_file_is_committed_and_valid() -> None:
    """入库的 tasks-20.jsonl 与代码生成的结果一致（防止手改了文件没改代码）。"""
    from app.core.llm.tasks_20 import TASKS_JSONL

    assert TASKS_JSONL.exists(), "tasks-20.jsonl 未入库（14:30 的冻结产物）"
    on_disk = [
        json.loads(line)
        for line in TASKS_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    generated = to_jsonl_rows()
    assert len(on_disk) == len(generated) == 20
    assert [r["task_id"] for r in on_disk] == [r["task_id"] for r in generated]


# ============================================================ 4. 确认约束（今天的必答问题）


def test_system_prompt_default_is_w10_baseline() -> None:
    """默认提示词必须**逐字等于** W10 基线 —— 否则回归子集不可比。

    W10 的冻结声明把 prompt 列为"未改"项。改它，`qa_runs_w10*` 基线就废了，
    而"成功率是升了还是被预算和早停拖低"这个问题永远答不出来。
    """
    from app.core.llm.react_loop import REACT_SYSTEM_PROMPT

    assert build_system_prompt() == REACT_SYSTEM_PROMPT
    assert "确认" not in REACT_SYSTEM_PROMPT  # 基线里确实没有这条


def test_confirmation_constraint_is_additive() -> None:
    """开了约束时，基线那段必须原样保留在前面（增量，不是替换）。"""
    from app.core.llm.react_loop import REACT_SYSTEM_PROMPT

    with_c = build_system_prompt(confirmation=True)
    assert with_c.startswith(REACT_SYSTEM_PROMPT)
    assert len(with_c) > len(REACT_SYSTEM_PROMPT)
    assert "确认" in with_c


def test_confirm_tasks_pass_only_with_constraint() -> None:
    """**对照证据**：N7/N8 的成败取决于确认约束。

    实测（2026-01-01，trae/qwen-3.5）：
    - 开约束：N7 通过（先问，不动手）；N8 通过
    - 关约束：N7 直接 generate_report（副作用已发生）→ 判据失败；
              N8 同上

    这条测试用**离线构造**固化那个结论的形状（不重复烧配额），
    真模型的数字记在 docs/weekly/w11.md。
    """
    from app.core.llm.tasks_20 import NEW_TASKS as tasks

    n7 = next(t for t in tasks if t.task_id == "N7_confirm_before_notify")
    n8 = next(t for t in tasks if t.task_id == "N8_confirm_before_report")
    for task in (n7, n8):
        # 有约束时的形状：问了、没动手
        good = make_result(
            final_answer="请确认后我再发送给 supervisor", tools=["query_metrics_sql"]
        )
        assert task.verify(good)[0] is True
        # 无约束时的形状：直接动手
        bad = make_result(final_answer="已通知质检主管", tools=["generate_report", "notify"])
        assert task.verify(bad)[0] is False


def test_half_of_new_tasks_need_mechanisms() -> None:
    """必答问题的**判读标准**：理想是一半靠新机制才能过。

    新增 10 个里，预期无机制能过的是 N1/N3/N4/N5（4 个，pass），
    需机制的是 N7/N8（2 个，fail），其余 4 个 unknown。
    这条测试把"预期分布"钉住，防止跑完以后悄悄改预期去迎合结果。
    """
    expected = {t.task_id: t.expected_without_mechanisms for t in NEW_TASKS}
    pass_count = sum(1 for v in expected.values() if v == "pass")
    fail_count = sum(1 for v in expected.values() if v == "fail")
    assert pass_count == 4, f"预期无机制能过的应为 4 个，实际 {pass_count}：{expected}"
    assert fail_count == 2, f"预期需机制的应为 2 个，实际 {fail_count}：{expected}"
    # 判读标准写进测试：既不是"全过"也不是"全不过"
    assert pass_count != len(expected) and fail_count != len(expected)
