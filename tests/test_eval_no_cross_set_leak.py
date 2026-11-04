"""**跨集去重**：防止评测集之间"考同一道题"。

## 这个缺口是怎么发现的（2026-11-04 核实）

现有的 `check_no_leak()`（`scripts/run_retrieval_eval.py`）只做**单条自洽**检查：
"这一条的问题有没有照抄它自己的 quote"。它**不跨集比较**，
所以下面这种情况完全查不出来：

    retrieval-gold  rg-09  页码为什么必须在切块的那一刻就记下来，事后再补不行吗？
    questions-50    m-05   跨页段落的页码信息为什么必须在解析阶段就记下来？

题面相似度 **0.510**，锚点都指向 `design-decisions.md` 的同一句
「没有页码，引用只能说到"某份文档里说的"」。

**两条题考的是同一个事实。**

## 为什么这很严重

`retrieval-gold`（20 条）用于调参，`questions-50`（50 条）用于报成绩。
如果两者有重叠，那么：

- 在 20 条上"调出来"的参数，会在 50 条上显得同样好 —— **但那是调参的效果，不是泛化**
- 报告的 Recall / 准确率**虚高**，且虚高的量无法估计

这不是"数据有点脏"，是**评测结论的效力问题**。

## 判据

三层，从严到宽：

1. **题面归一化后完全相同** -> 一定重复
2. **题面相似度 ≥ 0.85** -> 几乎肯定是同一题改写
3. **题面相似度 ≥ 0.5 且锚点指向同一 source + 同一句话** -> 同一事实换问法

第 3 条是 rg-09/m-05 那种情况：题面不完全一样，但**答案在原文里是同一句**。
只比题面会漏掉它。

## 已知的重叠（登记在此，不算测试失败）

下面的重叠**已被人工确认是同一个事实**，属于历史遗留。
新增重叠会让测试红；修历史遗留需要重新出题，不在本轮范围。
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).parents[1] / "eval"

#: 已知并接受的重叠 `(set_a, id_a, set_b, id_b, 说明)`。
#:
#: **不要再往这里加**。加一条等于承认"多考了一次同一道题"——
#: 应该重新出题，而不是登记。
KNOWN_OVERLAPS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "retrieval-gold",
        "rg-09",
        "questions-50",
        "m-05",
        "同一事实：页码必须在切分那一刻记录（design-decisions.md 同一句）",
    ),
)

#: 参与跨集比对的题集（都是"问题"形态的文件）。
QUESTION_SETS: tuple[str, ...] = (
    "questions-50.jsonl",
    "retrieval-gold.jsonl",
    "table-gold.jsonl",
    "refusal-gold.jsonl",
)


def _load(name: str) -> list[dict]:
    path = EVAL_DIR / name
    if not path.exists():
        pytest.skip(f"缺 {name}")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _norm(text: str) -> str:
    """归一化：只留汉字与字母数字，去掉标点空白。

    中文换问法时标点会全变（问号、逗号位置），不归一化会让相似度虚低。
    """
    return "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _anchors_of(item: dict) -> list[str]:
    """把一条题的所有锚点原文抽出来（统一成字符串列表）。

    两个题集字段名不同：`retrieval-gold` 用 `anchor`（单数，字符串），
    `questions-50` 用 `anchors`（复数，对象列表）。这里抹平。
    """
    out: list[str] = []
    single = item.get("anchor")
    if isinstance(single, str) and single.strip():
        out.append(single)
    for anchor in item.get("anchors") or []:
        if isinstance(anchor, dict):
            quote = anchor.get("quote") or ""
            if quote.strip():
                out.append(quote)
    return out


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _anchor_overlap(a: dict, b: dict) -> float:
    """两条题的锚点原文之间的最高相似度。"""
    best = 0.0
    for x in _anchors_of(a):
        for y in _anchors_of(b):
            # 锚点可能一段长一段短，用"短串被长串包含"来判断更实在
            nx, ny = _norm(x), _norm(y)
            if not nx or not ny:
                continue
            if nx in ny or ny in nx:
                return 1.0
            best = max(best, _similarity(x, y))
    return best


def test_question_sets_have_no_total_duplicates() -> None:
    """**完全相同的题面**在任何两个集之间都不允许（归一化后比较）。"""
    sets = {name: _load(name) for name in QUESTION_SETS}
    seen: dict[str, tuple[str, str]] = {}
    duplicates: list[str] = []

    for name, items in sets.items():
        for item in items:
            key = _norm(item.get("question", ""))
            if not key:
                continue
            if key in seen:
                prev_set, prev_id = seen[key]
                if prev_set != name:
                    duplicates.append(f"{prev_set}/{prev_id} == {name}/{item.get('id')}")
            else:
                seen[key] = (name, item.get("id", "?"))

    assert not duplicates, "存在跨集完全相同的题面（同一道题考了两次）：\n  " + "\n  ".join(
        duplicates
    )


def test_no_high_similarity_cross_set_pairs() -> None:
    """**题面相似度 ≥ 0.85** 的跨集题对必须登记在 `KNOWN_OVERLAPS`。

    0.85 这个阈值：改几个字的同义改写通常在 0.85~0.95；
    碰巧都问"为什么"但内容不同的，实测都在 0.6 以下。
    """
    sets = {name: _load(name) for name in QUESTION_SETS}
    known = {(a, b) for _, a, _, b, _ in KNOWN_OVERLAPS} | {
        (b, a) for _, a, _, b, _ in KNOWN_OVERLAPS
    }

    offenders: list[str] = []
    names = list(sets)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            for a in sets[name_a]:
                for b in sets[name_b]:
                    sim = _similarity(a.get("question", ""), b.get("question", ""))
                    if sim >= 0.85 and (a.get("id"), b.get("id")) not in known:
                        offenders.append(
                            f"{name_a}/{a.get('id')} ~ {name_b}/{b.get('id')} 相似度 {sim:.2f}"
                        )

    assert not offenders, (
        "跨集高相似题对（≥0.85）未登记：\n  "
        + "\n  ".join(offenders)
        + "\n\n要么换个问题问，要么（下策）登记进 KNOWN_OVERLAPS。"
    )


def test_no_same_fact_with_rewording() -> None:
    """**同一事实换问法**也要抓：题面中等相似 + 锚点高度重合。

    这是 rg-09 / m-05 那类：
    - 题面相似度 0.51（不算高，光看题面会放过）
    - 但锚点都指向 `design-decisions.md` 的同一句

    判据：题面 ≥0.5 **且** 锚点重合度 ≥0.6 -> 视为同一事实。
    """
    sets = {name: _load(name) for name in QUESTION_SETS}
    known = {(a, b) for _, a, _, b, _ in KNOWN_OVERLAPS} | {
        (b, a) for _, a, _, b, _ in KNOWN_OVERLAPS
    }

    offenders: list[str] = []
    names = list(sets)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            for a in sets[name_a]:
                for b in sets[name_b]:
                    if (a.get("id"), b.get("id")) in known:
                        continue
                    sim = _similarity(a.get("question", ""), b.get("question", ""))
                    if sim < 0.5:
                        continue
                    overlap = _anchor_overlap(a, b)
                    if overlap >= 0.6:
                        offenders.append(
                            f"{name_a}/{a.get('id')} ~ {name_b}/{b.get('id')} "
                            f"题面 {sim:.2f} / 锚点 {overlap:.2f}"
                        )

    assert not offenders, "跨集同一事实（题面中等相似 + 锚点高度重合）未登记：\n  " + "\n  ".join(
        offenders
    )


def test_known_overlaps_are_documented() -> None:
    """`KNOWN_OVERLAPS` 里每一条都要有**可读的说明**。

    没有说明的登记项，三个月后没人知道它为什么存在 ——
    然后下一个人会把它当成"忘记删的临时 hack"删掉。
    """
    for set_a, id_a, set_b, id_b, reason in KNOWN_OVERLAPS:
        assert reason and len(reason) >= 10, (
            f"{set_a}/{id_a} ~ {set_b}/{id_b} 的说明太短，至少写清'为什么是同一件事'"
        )


def test_question_set_sizes_are_what_they_are() -> None:
    """**把真实规模钉在测试里**，防止有人继续引用"200 条"。

    2026-11-04 核实：全仓所有题目集加起来是 **80 条**：

        questions-50  50
        retrieval-gold 20
        table-gold     5
        refusal-gold   5

    没有任何地方存着 200 条。如果哪天有人加了题，这条会红 ——
    那时请同步更新这个数字（**以及所有引用它总数的文档**）。
    """
    expected = {
        "questions-50.jsonl": 50,
        "retrieval-gold.jsonl": 20,
        "table-gold.jsonl": 5,
        "refusal-gold.jsonl": 5,
    }
    actual = {name: len(_load(name)) for name in expected}
    assert actual == expected, (
        f"题集规模变了：{actual}（期望 {expected}）。\n"
        "改了题量请同步更新这里，以及文档里引用总数的说法。"
    )
    assert sum(actual.values()) == 80, (
        f"题目总数是 {sum(actual.values())}，不是 200 —— 引用时请写实数"
    )
