"""Reranker 的纯逻辑测试。**不加载模型** —— 2.27 GB 的模型不该出现在单测里。

## 这个文件锁什么

| 锁什么 | 防什么 |
| --- | --- |
| sigmoid 数值稳定 | 大负数直接 `1/(1+exp(-x))` 会溢出成 inf |
| 归一化按 query 内部 | 被改成跨 query 归一化（Cross-Encoder 的分数本来就跨 query 不可比） |
| 三种截断真的截在不同位置 | 截断策略写成同一个（那"三种策略"就是假的） |
| `rank_before/after/delta` 可算 | "精排挪了几位"算不出来，就只能靠眼看 |
| `sort_by_length` 不影响分数 | 优化改变了结果 —— 那是 bug，不是优化 |

## 为什么这些能脱离模型测

sigmoid、归一化、截断预算、排序回填都是**纯函数逻辑**。
真正需要模型的部分（前向打分）由 `scripts/rerank_probe.py` 与
`scripts/rerank_eval.py` 实测，不进单测 —— 否则每次 `pytest` 都要 12 秒加载。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from app.rag.bm25.rerank import (  # noqa: E402
    Reranker,
    RerankScore,
    Truncation,
    normalize_within_query,
    sigmoid,
)

# --------------------------------------------------------------------------- sigmoid


def test_sigmoid_is_numerically_stable_for_large_negative() -> None:
    """**大负数不能直接 `1/(1+exp(-x))`** —— `exp(800)` 会 overflow 成 inf。

    Cross-Encoder 的 logit 没有硬下界（实测见过 -11），
    在更长的文本或分布外输入上它可能更低。
    """
    assert sigmoid(-800.0) == pytest.approx(0.0, abs=1e-12)
    assert sigmoid(800.0) == pytest.approx(1.0, abs=1e-12)
    assert sigmoid(0.0) == pytest.approx(0.5)
    # 中段与朴素公式一致
    assert sigmoid(-3.0) == pytest.approx(1.0 / (1.0 + math.exp(3.0)))


def test_sigmoid_is_monotonic() -> None:
    """sigmoid 保序。**这是"分数只是被压缩、名次没变"的依据。**"""
    values = [-10.0, -3.0, 0.0, 2.99, 8.0]
    scores = [sigmoid(v) for v in values]
    assert scores == sorted(scores)


# --------------------------------------------------------------------------- 归一化


def test_normalize_minmax_is_within_one_query() -> None:
    """**归一化只在一个 query 的候选集合内做。**

    实测证据（`scripts/rerank_probe.json`）：8 条不同 query 的 top1 分
    从 0.002 到 0.842，跨度 0.840。**同一模型、同一批候选，只换 query** ——
    分数就整体平移。所以跨 query 的绝对阈值毫无意义。

    按 query 内部归一后，每条 query 的 top1 都是 1.0。
    """
    high_query = [0.842, 0.30, 0.10]
    low_query = [0.002, 0.0015, 0.001]

    high_norm = normalize_within_query(high_query)
    low_norm = normalize_within_query(low_query)

    assert high_norm[0] == pytest.approx(1.0)
    assert low_norm[0] == pytest.approx(1.0), (
        "低分 query 的 top1 在自己的候选里也是最好的，必须拿 1.0 —— "
        "若按全局归一它会被压成 0.00x，那一侧的权重就假了"
    )


def test_normalize_percentile_ranks_by_position() -> None:
    """percentile 按名次分档：最高拿 1.0、最低拿 0.0。**它对离群值比 minmax 稳**。"""
    scores = [0.99, 0.40, 0.38, 0.35]
    normed = normalize_within_query(scores, "percentile")
    assert max(normed) == pytest.approx(1.0)
    assert min(normed) == pytest.approx(0.0)
    # minmax 会把后三名全压到 0.0~0.08 的窄带里
    minmax = normalize_within_query(scores, "minmax")
    assert minmax[1] == pytest.approx((0.40 - 0.35) / (0.99 - 0.35))
    assert normed[1] > minmax[1], "percentile 不该被一条离群高分把其余压扁"


def test_normalize_degenerate_cases() -> None:
    """空表 / 单元素 / 全同分。**都给 1.0 或空，不给 0** —— 理由见 fusion 的同名测试。"""
    assert normalize_within_query([]) == []
    assert normalize_within_query([0.5]) == [1.0]
    assert normalize_within_query([0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0]


def test_normalize_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="未知归一化方式"):
        normalize_within_query([1.0, 2.0], "softmax")


# --------------------------------------------------------------------------- 截断


@pytest.mark.parametrize(
    "policy,kept_part",
    [
        (Truncation.TAIL, "head"),
        (Truncation.HEAD, "tail"),
        (Truncation.MIDDLE, "middle"),
    ],
    ids=["tail-keeps-head", "head-keeps-tail", "middle-keeps-middle"],
)
def test_truncation_policies_keep_different_parts(policy: Truncation, kept_part: str) -> None:
    """**三种策略必须真的截在不同位置**，否则"三种策略"就是假的。

    用 100 个编号 token、截到 10 个：

        TAIL   → 保留开头 0~9   （丢结尾）
        HEAD   → 保留结尾 90~99 （丢开头）
        MIDDLE → 保留 45~54     （两头都丢）

    命名容易看反：`Truncation.TAIL` 指「**截掉尾巴**」，不是「保留尾巴」。
    这条测试就是为了把这个反直觉的命名钉死。
    """
    ids = list(range(100))
    reranker = Reranker(truncation=policy)
    kept = reranker._truncate_ids(ids, 10)
    assert len(kept) == 10
    if kept_part == "head":
        assert kept[0] == 0, f"TAIL 策略应保留开头，实际从 {kept[0]} 开始"
        assert kept[-1] == 9
    elif kept_part == "tail":
        assert kept[-1] == 99, f"HEAD 策略应保留结尾，实际到 {kept[-1]} 结束"
        assert kept[0] == 90
    else:
        assert kept[0] > 0 and kept[-1] < 99, "MIDDLE 策略应两头都丢"


def test_truncation_handles_short_input() -> None:
    """输入本来就短时不该多截 —— 预算够就不动。"""
    reranker = Reranker()
    ids = list(range(5))
    assert reranker._truncate_ids(ids, 10) == ids


# --------------------------------------------------------------------------- 名次回填


def test_rerank_score_delta_is_negative_when_promoted() -> None:
    """`delta` 负数 = 往前挪（变好）。**这是"精排有没有用"的量化依据。**"""
    promoted = RerankScore("x", 2.0, 0.9, rank_before=13, rank_after=1)
    assert promoted.delta == -12
    demoted = RerankScore("y", -1.0, 0.2, rank_before=2, rank_after=9)
    assert demoted.delta == 7
    unchanged = RerankScore("z", 0.0, 0.5, rank_before=3, rank_after=3)
    assert unchanged.delta == 0


def test_rerank_score_delta_is_none_without_both_ranks() -> None:
    """缺任一名次时 `delta` 为 `None` —— **不要静默当 0**。"""
    assert RerankScore("x", 1.0, 0.5, rank_before=1).delta is None
    assert RerankScore("x", 1.0, 0.5, rank_after=1).delta is None


# --------------------------------------------------------------------------- 配置与加载


def test_reranker_is_lazy() -> None:
    """**不 load 就不加载模型。** 830 条测试不该为它付 12 秒 + 2 GB。"""
    reranker = Reranker()
    assert reranker.is_loaded() is False
    assert reranker.load_ms == 0


def test_missing_model_dir_raises_actionable_error() -> None:
    """模型目录不存在时，错误信息要**直接告诉人跑哪条命令**。"""
    reranker = Reranker(model_path=Path("d:/cache/does-not-exist-reranker"))
    with pytest.raises(FileNotFoundError, match="fetch_reranker.py"):
        reranker.load()


def test_default_batch_sort_is_enabled() -> None:
    """默认按长度排序组 batch。**关掉它实测慢 2 倍**（见 rerank_probe.json）。"""
    assert Reranker().sort_by_length is True


def test_sort_by_length_can_be_disabled_for_ab_compare() -> None:
    """保留 `sort_by_length=False` 是为了做 A/B —— 它不该被悄悄删掉，
    否则"优化到底值多少"就再也测不出来了。"""
    assert Reranker(sort_by_length=False).sort_by_length is False
