"""校验 40 条输入池：格式、判定标准齐全、采集纪律。

采集纪律的三条硬检查：
  1. **每条都有 source**（可追溯到开发语料的具体文件），不是凭空编的
  2. **每条都有判定标准**（qa: gold_points / classify: gold_label /
     extract: gold / summarize: must_include）
  3. **不与已做过的实验输入重叠**（数据泄漏检查）

第 3 条是本脚本的重点：W7 的 RAG 评测集不能复用这批问题，
否则调 prompt 时"见过"会让 W7 评测虚高。这里先锁住"我调 prompt 用过什么"。

运行：python -X utf8 scripts/validate_prompt_sets.py
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

SETS_DIR = Path(__file__).resolve().parents[1] / "eval" / "prompt-sets"
SCRIPTS_DIR = Path(__file__).resolve().parent

# 每类必须带的判定字段
REQUIRED_JUDGE_KEYS = {
    "qa": ("gold_points",),
    "classify": ("gold_label",),
    "extract": ("gold",),
    "summarize": ("must_include",),
}


def load_set(name: str) -> list[dict]:
    path = SETS_DIR / f"{name}.jsonl"
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{name}.jsonl 第 {lineno} 行不是合法 JSON：{exc}") from exc
    return rows


def check_required(name: str, rows: list[dict]) -> list[str]:
    issues = []
    keys = REQUIRED_JUDGE_KEYS[name]
    for r in rows:
        for k in keys:
            if k not in r or not r[k]:
                issues.append(f"{r.get('id', '?')} 缺少判定标准 {k}")
        if not r.get("source"):
            issues.append(f"{r.get('id', '?')} 缺少 source（采集纪律要求可追溯）")
        if not r.get("id"):
            issues.append(f"{name} 有行缺少 id")
    return issues


def collect_used_texts() -> set[str]:
    """收集我调 prompt 时实际发出去过的文本，用于泄漏检查。"""
    used: set[str] = set()

    for f in SCRIPTS_DIR.glob("*_result.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        _walk(d, used)

    # 周二对抗输入与周一评测集是明确用过的
    for f in ("system_prompt_ab.json", "prompt_patterns_dataset.json"):
        p = SCRIPTS_DIR / f
        if p.exists():
            with contextlib.suppress(Exception):
                _walk(json.loads(p.read_text(encoding="utf-8")), used)
    return used


def _walk(o: object, out: set[str]) -> None:
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("text", "question", "input") and isinstance(v, str):
                out.add(v.strip())
            _walk(v, out)
    elif isinstance(o, list):
        for v in o:
            _walk(v, out)


def check_leakage(name: str, rows: list[dict], used: set[str]) -> list[str]:
    """输入池里的 text/question 不能出现在已用过的集合里。"""
    issues = []
    key = "question" if name == "qa" else "text"
    for r in rows:
        text = str(r.get(key, "")).strip()
        if not text:
            continue
        if text in used:
            issues.append(f"{r['id']} 与已用过的实验输入完全重叠（数据泄漏）")
        # 也检查前缀重叠（同一段话被截断复用）
        for u in used:
            if len(text) > 20 and len(u) > 20 and (text[:20] in u or u[:20] in text) and text != u:
                issues.append(f"{r['id']} 与已用输入高度相似：{u[:30]}...")
                break
    return issues


def main() -> None:
    names = ("qa", "classify", "extract", "summarize")
    total = 0
    all_issues: list[str] = []

    used = collect_used_texts()
    print(f"已用过的实验输入：{len(used)} 条（泄漏检查基线）\n")

    for name in names:
        rows = load_set(name)
        total += len(rows)
        print(f"[{name}] {len(rows)} 条")
        if len(rows) != 10:
            all_issues.append(f"{name} 应为 10 条，实际 {len(rows)}")

        req = check_required(name, rows)
        leak = check_leakage(name, rows, used)
        all_issues.extend(req + leak)
        print(f"   判定标准齐全: {'✅' if not req else '❌ ' + str(len(req)) + ' 处'}")
        print(f"   无数据泄漏  : {'✅' if not leak else '❌ ' + str(len(leak)) + ' 处'}")

    # 类别分布（分类任务必须真实不均衡，不能手工凑平衡）
    cls = load_set("classify")
    from collections import Counter

    dist = Counter(r["gold_label"] for r in cls)
    print("\n[classify] 类别分布（真实不均衡，未手工凑平衡）：")
    for label, n in dist.most_common():
        print(f"   {label:<10} {n}")
    if len(dist) == len(cls):
        all_issues.append("classify 每个类别各 1 条 = 手工凑平衡，违反采集纪律")

    print(f"\n合计 {total} 条")
    if all_issues:
        print(f"\n❌ 发现 {len(all_issues)} 个问题：")
        for i in all_issues:
            print(f"   - {i}")
        sys.exit(1)

    print("\n✅ 全部通过：40 条齐全、判定标准完整、无数据泄漏、类别分布真实")


if __name__ == "__main__":
    main()
