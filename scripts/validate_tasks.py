"""校验 tasks-20.jsonl 格式（W11 周五 14:30–15:30 的验收项）。

## 为什么需要它

判据是**数据**了（JSONL），数据的坏法比代码隐蔽得多：字段名拼错、
checks 写成字符串、task_id 重复、预算缺项 —— 这些都不会报错，
只会在周六全量跑时变成"某个任务永远过不了"，然后你怀疑是模型的问题。

今天纪律说得最准："不然你会写出一套没人验证过的判据，
周六全量跑时才发现格式不匹配。" 所以 15:30 之前必须先跑这个。

## 校验什么

| 项 | 规则 | 不通过的含义 |
| --- | --- | --- |
| 数量 | 恰好 20 行 | 结构没定版 |
| 唯一性 | task_id 不重复 | 覆盖会悄悄丢任务 |
| 分类 | category ∈ 五类；新增的 10 个每类 ≥1 | 结构表对不上 |
| 回归子集 | 10 个 W10 任务全在，且 verify_kind=legacy | 判据被偷改 / 丢任务 |
| checks | 非空列表，每项有合法 kind | 判据写坏 |
| check 参数 | contains 类必须有 any/words；used_tool 必须有 tool | 判据永远不过 |
| 预算 | tokens > 0 且 ≤ 50000；wall_time_s > 0 | 预算漏填 |
| 不可完成 | 至少 4 条，且形态互不相同 | 四类缺失没覆盖 |

## 用法

    python -X utf8 scripts/validate_tasks.py
    python -X utf8 scripts/validate_tasks.py --json   # 机器可读
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

from app.core.llm.tasks_20 import (  # noqa: E402
    CATEGORIES,
    REGRESSION_IDS,
    TASKS_JSONL,
    write_jsonl,
)

#: 合法的 check 类型。新增类型必须先在这里登记 —— 否则校验通过、
#: 运行时 check_one 抛 ValueError，那就是"格式对但跑不了"。
VALID_CHECK_KINDS: Final[frozenset[str]] = frozenset(
    {
        "legacy",
        "completed",
        "contains",
        "all_contains",
        "not_contains",
        "used_tool",
        "called_tool",
        "not_used_tool",
        "refused",
        "asks_confirmation",
        "no_side_effect",
        "degraded",
    }
)

#: 需要额外参数的 check 类型 → 必需键。
CHECK_REQUIRED_KEYS: Final[dict[str, str]] = {
    "contains": "any",
    "all_contains": "words",
    "not_contains": "any",
    "used_tool": "tool",
    "called_tool": "tool",
    "not_used_tool": "tool",
}

#: 不可完成的四种形态。四条任务必须各占一种 —— 全挤在"信息缺失"上
#: 就等于只测了一类。
IMPOSSIBLE_FORMS: Final[tuple[str, ...]] = (
    "missing_data",  # 库里没有
    "capability",  # 能力外（图像）
    "permission",  # 需外部系统权限
    "realtime",  # 需实时数据
)


class Report:
    """校验结果收集器。错误与警告分开：错误 = 必须改，警告 = 该看一眼。"""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def ok(self) -> bool:
        return not self.errors


def classify_impossible_form(task: dict[str, Any]) -> str | None:
    """从任务内容里认出它属于哪种"不可完成"形态。

    用文本特征而不是加一个字段：加字段需要改 schema 并重跑生成，
    而形态本来就能从指令里读出来。判错的代价是"漏一种形态"，
    校验会报出来，不会静默。
    """
    text = str(task.get("instruction", ""))
    rationale = str(task.get("rationale", ""))
    blob = text + rationale
    if any(k in blob for k in ("照片", "图像", "图片", "识别")):
        return "capability"
    if any(k in blob for k in ("ERP", "下载", "外部系统", "登录", "权限")):
        return "permission"
    if any(k in blob for k in ("实时", "现在生产线", "当前温度")):
        return "realtime"
    if any(k in blob for k in ("查无", "不存在", "没有这个批次", "B2025")):
        return "missing_data"
    return None


def validate_rows(rows: list[dict[str, Any]]) -> Report:
    report = Report()

    # ---- 1. 数量与唯一性 ----
    if len(rows) != 20:
        report.error(f"任务数应为 20，实际 {len(rows)}")
    ids = [r.get("task_id") for r in rows]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    if dupes:
        report.error(f"task_id 重复：{dupes}")

    # ---- 2. 回归子集完整且判据未被改动 ----
    legacy_rows = [r for r in rows if r.get("verify_kind") == "legacy"]
    legacy_ids = {r.get("task_id") for r in legacy_rows}
    missing_legacy = [i for i in REGRESSION_IDS if i not in legacy_ids]
    if missing_legacy:
        report.error(f"回归子集缺失：{missing_legacy}（W10 的 10 个必须全在）")
    if len(legacy_rows) != len(REGRESSION_IDS):
        report.error(f"verify_kind=legacy 的行应为 {len(REGRESSION_IDS)}，实际 {len(legacy_rows)}")

    # ---- 3. 分类 ----
    for row in rows:
        cat = row.get("category")
        if cat not in CATEGORIES:
            report.error(f"{row.get('task_id')} 的 category 非法：{cat!r}")
    new_rows = [r for r in rows if r.get("verify_kind") != "legacy"]
    if len(new_rows) != 10:
        report.error(f"新增任务应为 10 个，实际 {len(new_rows)}")
    new_cats = Counter(r.get("category") for r in new_rows)
    for cat in CATEGORIES:
        if new_cats.get(cat, 0) < 1:
            report.error(f"新增任务里缺少「{CATEGORIES[cat]}」类")

    # ---- 4. checks ----
    for row in rows:
        tid = row.get("task_id")
        checks = row.get("checks")
        if not isinstance(checks, list) or not checks:
            report.error(f"{tid} 的 checks 必须是非空列表")
            continue
        for idx, item in enumerate(checks):
            if not isinstance(item, dict):
                report.error(f"{tid}.checks[{idx}] 必须是对象")
                continue
            kind = item.get("kind")
            if kind not in VALID_CHECK_KINDS:
                report.error(f"{tid}.checks[{idx}] 未知 kind：{kind!r}")
                continue
            need = CHECK_REQUIRED_KEYS.get(kind)
            if need and not item.get(need):
                report.error(f"{tid}.checks[{idx}]（{kind}）缺少必需键 {need!r}")
            if kind == "legacy" and row.get("verify_kind") == "legacy":
                # JSONL 行不带函数（不可序列化），这是**预期**的；
                # 运行时按 task_id 回查。所以这里只确认有标记，不要求有 verify。
                continue

    # ---- 5. 预算 ----
    for row in rows:
        tid = row.get("task_id")
        budget = row.get("budget") or {}
        tokens = budget.get("tokens")
        wall = budget.get("wall_time_s")
        if not isinstance(tokens, int) or tokens <= 0:
            report.error(f"{tid} 的预算 tokens 非法：{tokens!r}")
        elif tokens > 50_000:
            report.error(f"{tid} 的 tokens={tokens} 超出上限 50000")
        elif tokens < 1_000:
            report.warn(f"{tid} 的 tokens={tokens} 偏低，可能触顶（周四定档 10k）")
        if not isinstance(wall, (int, float)) or wall <= 0:
            report.error(f"{tid} 的预算 wall_time_s 非法：{wall!r}")

    # ---- 6. 不可完成：四种形态 ----
    # 四种形态只在**新增 10 个**里要求齐全。回归子集 T9/T10 是 W10 老口径
    # （信息缺失/指标缺失），不参与这个统计——把它们的形态混进来，
    # 会让"四类缺失各一"的验收变成"六条里大概有四种"，那就不叫各一了。
    impossible_rows = [r for r in rows if r.get("category") == "impossible"]
    new_impossible = [r for r in impossible_rows if r.get("verify_kind") != "legacy"]
    forms = Counter()
    for row in new_impossible:
        form = classify_impossible_form(row)
        if form is None:
            report.error(f"{row.get('task_id')} 认不出属于哪种不可完成形态（四类各一）")
        else:
            forms[form] += 1
    if len(new_impossible) < 4:
        report.error(f"新增里不可完成任务应 ≥4 个，实际 {len(new_impossible)}")
    for form in IMPOSSIBLE_FORMS:
        if forms.get(form, 0) == 0:
            report.error(f"新增里不可完成形态缺失：{form}（四种都要有）")

    # ---- 7. 期望路径 ----
    for row in rows:
        tid = row.get("task_id")
        tools = row.get("expected_tools")
        if not isinstance(tools, list):
            report.error(f"{tid} 的 expected_tools 必须是列表")
            continue
        # expected_tools 为空**不一定是错**：T4（越界拒绝）与 N10-N12（能力外/
        # 需权限/需实时）的正确路径本来就是"不调工具"。所以只在
        # "既有工具又没写"时报——通过 instruction 里是否提到可调工具来判。
        if not tools:
            text = str(row.get("instruction", ""))
            mentions_tool_domain = any(
                k in text for k in ("批次", "指标", "规范", "限值", "报告", "数据")
            )
            if mentions_tool_domain and row.get("category") != "impossible":
                report.warn(f"{tid}（{row.get('category')}）指令涉及可查内容但 expected_tools 为空")

    return report


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"第 {line_no} 行不是合法 JSON：{exc}") from exc
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验 tasks-20.jsonl 格式")
    parser.add_argument("--path", type=Path, default=TASKS_JSONL)
    parser.add_argument("--json", action="store_true", help="机器可读输出")
    parser.add_argument("--write", action="store_true", help="先按代码重新生成 JSONL 再校验")
    args = parser.parse_args(argv)

    if args.write:
        target = write_jsonl(args.path)
        print(f"已重新生成：{target}")

    if not args.path.exists():
        print(f"文件不存在：{args.path}", file=sys.stderr)
        return 2

    rows = load_rows(args.path)
    report = validate_rows(rows)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "rows": len(rows),
                    "errors": report.errors,
                    "warnings": report.warnings,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("=" * 70)
        print(f"tasks-20 校验 · {args.path.name} · {len(rows)} 行")
        print("=" * 70)
        if report.errors:
            print(f"\n❌ 错误 {len(report.errors)} 条（必须改）：")
            for msg in report.errors:
                print(f"  - {msg}")
        if report.warnings:
            print(f"\n⚠ 警告 {len(report.warnings)} 条（该看一眼）：")
            for msg in report.warnings:
                print(f"  - {msg}")
        if report.ok:
            print("\n✅ 格式校验通过")
            cats = Counter(r.get("category") for r in rows)
            print("\n分类分布：")
            for cat, label in CATEGORIES.items():
                print(f"  {label:10s} {cats.get(cat, 0)}")
            new_cats = Counter(r.get("category") for r in rows if r.get("verify_kind") != "legacy")
            print("\n新增 10 个的分布：")
            for cat, label in CATEGORIES.items():
                print(f"  {label:10s} {new_cats.get(cat, 0)}")

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
