"""两段式工程化的判定逻辑测试。**不加载模型、不跑检索。**

## 为什么这些逻辑值得单测

`scripts/rerank_pipeline.py` 里有两个纯函数，它们的语义直接决定结论对错：

| 函数 | 语义 | 算错的后果 |
| --- | --- | --- |
| `_gold_rank` | 多锚点 + `limit` 命名次 | **天花板算错 → 该修哪一层的判断全错** |
| `_recall_at_n` | 分母 = 可答题总数 | 分母用成"命中数"会让召回率虚高 |

**Recall@N 是"精排的天花板"** —— 它算错的话，
"问题在召回层还是精排层"这个本周最重要的判定就会指向错误的方向，
而修错层的代价是整周做无用功。

## 为什么不测 `_run_sweep`

它要真的加载 2.27 GB 模型跑前向。那部分由
`scripts/rerank_pipeline.py` 与 `scripts/rerank_eval.py` 实测，
不进单测 —— 否则每次 `pytest` 都要几十秒。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import rerank_pipeline as rp  # noqa: E402

# --------------------------------------------------------------------------- _gold_rank


def test_gold_rank_returns_one_based_position() -> None:
    """名次是 **1-based**，与检索输出、报告表格一致。"""
    assert rp._gold_rank(["a", "b", "c"], ["a"]) == 1
    assert rp._gold_rank(["a", "b", "c"], ["c"]) == 3


def test_gold_rank_returns_none_when_absent() -> None:
    assert rp._gold_rank(["a", "b"], ["z"]) is None


def test_gold_rank_requires_all_anchors_for_multi_hop() -> None:
    """**多跳题要求全部锚点都命中** —— 与 `run_retrieval_eval` 同一口径。

    口径不一致会让"天花板"与"Recall@5"两个数不可比，
    而它们正是判定"该修哪层"的两个输入。
    """
    ids = ["a", "b", "c", "d"]
    assert rp._gold_rank(ids, ["a", "b"]) == 2, "两个锚点都在，取较靠后的名次"
    assert rp._gold_rank(ids, ["a", "z"]) is None, "只命中一个不算命中"
    assert rp._gold_rank(ids, ["z", "y"]) is None


def test_gold_rank_takes_the_worst_anchor() -> None:
    """取**较靠后**的锚点名次（要看到两处，起决定作用的是晚的那处）。"""
    ids = ["a", "b", "c", "d", "e"]
    assert rp._gold_rank(ids, ["a", "e"]) == 5
    assert rp._gold_rank(ids, ["e", "a"]) == 5, "锚点顺序不该影响结果"


def test_gold_rank_limit_is_inclusive() -> None:
    """`limit` 是**闭区间**：第 N 名算"进了 N 条候选"，第 N+1 名不算。

    这条边界直接对应 Recall@N 的定义。差一位的 off-by-one 会让
    天花板虚高或虚低整整一档。
    """
    ids = ["a", "b", "c", "d", "e"]
    assert rp._gold_rank(ids, ["c"], limit=3) == 3, "第 3 名在 limit=3 内"
    assert rp._gold_rank(ids, ["d"], limit=3) is None, "第 4 名超出 limit=3"
    assert rp._gold_rank(ids, ["e"], limit=5) == 5


def test_gold_rank_limit_applies_to_the_worst_anchor() -> None:
    """多跳题的 `limit` 判在**较晚那个锚点**上。

    两个锚点在第 2 和第 7 名，`limit=5` 时应判未命中 ——
    因为要同时看到两处，而第 7 处不在前 5 条里。
    """
    ids = [f"c{i}" for i in range(1, 11)]
    assert rp._gold_rank(ids, ["c2", "c7"], limit=5) is None
    assert rp._gold_rank(ids, ["c2", "c7"], limit=10) == 7


def test_gold_rank_empty_gold_returns_none() -> None:
    """没有 gold（拒答题）返回 `None`，不抛。"""
    assert rp._gold_rank(["a", "b"], []) is None


def test_gold_rank_empty_candidates_returns_none() -> None:
    """候选为空时也返回 `None` —— 别让它变成 0 或抛异常。"""
    assert rp._gold_rank([], ["a"]) is None


# --------------------------------------------------------------------------- _recall_at_n


def test_recall_at_n_uses_answerable_as_denominator() -> None:
    """**分母是全部可答题**，不是命中数。

    分母写成"命中数"会让召回率永远是 100% —— 那种错误很隐蔽，
    因为数字看起来"很好"。
    """
    questions = [
        rp.rre.GoldItem(id="q1", type="fact", question="q1", anchors=[], gold_chunk_ids=["c1"]),
        rp.rre.GoldItem(id="q2", type="fact", question="q2", anchors=[], gold_chunk_ids=["c2"]),
        rp.rre.GoldItem(id="q3", type="fact", question="q3", anchors=[], gold_chunk_ids=["c3"]),
    ]
    # q1 的 gold 在第 1 名；q2 在第 3 名；q3 根本没有。
    # **键是 question**（不是 id）—— `_recall_at_n` 用 `item.question` 调 retrieve。
    layout = {
        "q1": ["c1", "x", "y"],
        "q2": ["x", "y", "c2"],
        "q3": ["x", "y", "z"],
    }

    def retrieve(query: str, n: int) -> list[str]:
        return layout[query][:n]

    table = rp._recall_at_n(questions, retrieve, (1, 3))
    assert table[1]["total"] == 3
    assert table[1]["hits"] == 1  # 只有 q1
    assert table[1]["rate"] == pytest.approx(0.3333)
    assert table[3]["hits"] == 2  # q1 + q2
    assert table[3]["rate"] == pytest.approx(0.6667)


def test_recall_at_n_is_monotonic() -> None:
    """**Recall@N 必须随 N 单调不减** —— 多给候选不可能让命中变少。

    这条是基本的健全性检查：如果它不单调，说明 `limit` 传错了或
    名次算错了，而那会让"天花板"这个数失去意义。
    """
    questions = [
        rp.rre.GoldItem(
            id=f"q{i}", type="fact", question=f"q{i}", anchors=[], gold_chunk_ids=[f"g{i}"]
        )
        for i in range(5)
    ]
    # 每条题的 gold 分别在第 1/2/4/8/16 名（键是 question）
    layout = {
        "q0": ["g0"] + [f"x{i}" for i in range(20)],
        "q1": ["x", "g1"] + [f"x{i}" for i in range(20)],
        "q2": ["x", "x", "x", "g2"] + [f"x{i}" for i in range(20)],
        "q3": ["x"] * 7 + ["g3"] + [f"x{i}" for i in range(20)],
        "q4": ["x"] * 15 + ["g4"] + [f"x{i}" for i in range(20)],
    }

    def retrieve(query: str, n: int) -> list[str]:
        return layout[query][:n]

    table = rp._recall_at_n(questions, retrieve, (1, 2, 4, 8, 16))
    rates = [table[n]["rate"] for n in (1, 2, 4, 8, 16)]
    assert rates == sorted(rates), f"Recall@N 必须单调不减，实际：{rates}"
    assert rates[-1] == pytest.approx(1.0), "给到 16 名时 5 条全该命中"


def test_recall_at_n_handles_empty_questions() -> None:
    """空题集不除零。"""
    table = rp._recall_at_n([], lambda q, n: [], (5,))
    assert table[5]["rate"] == 0.0
    assert table[5]["total"] == 0


# --------------------------------------------------------------------------- 判定阈值


def test_candidate_windows_and_top_k_match_the_task_spec() -> None:
    """参数是任务书点名的：N ∈ {20,30,50}、K 固定 5。

    这些不只是常量 —— 它们决定产物里表格的列，改了就与文档对不上。
    """
    assert rp.CANDIDATES == (20, 30, 50)
    assert rp.TOP_K == 5
    # Recall@N 表必须包含 5（当前线上水平）与 50（天花板）
    assert 5 in rp.RECALL_N
    assert 50 in rp.RECALL_N


def test_two_routes_are_defined() -> None:
    """两条路线必须都在 —— 对比"纯向量→精排"与"混合→精排"是今天的任务。"""
    names = {route.name for route in rp.ROUTES}
    assert names == {"pure_vec", "hybrid_rrf"}
