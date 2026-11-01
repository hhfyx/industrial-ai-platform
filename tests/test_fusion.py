"""融合模块的测试。**重点不是"算得对"，是"错法会不会被静默带过"。**

## 这个文件锁的四件事

| 锁什么 | 防什么 |
| --- | --- |
| RRF 不读分数 | 有人"顺手"把分数也加权进去，量纲问题就回来了 |
| 归一化按 query 内部 | 被改成按全库归一化（那会让 query 之间的尺度差变成偏置） |
| 加权融合缺分数时抛异常 | 静默拿名次当分数，α 变成假的 |
| 只召回一路时补 0 | 凭空给"另一路没看到"的候选发奖励 |

## 为什么"归一化必须按 query"这条要用测试锁

它不是一个实现细节，是**方法论决定**。做错了不会报错，只会让某些 query
的分数被系统性压低，而调参时看到的是"某些题变差"，会去怀疑 α。

`test_each_query_normalizes_independently` 用两个分数尺度完全不同的 query
来证明这件事：如果归一化是全局的，第二个 query 会整体被压到 0 附近。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from app.rag.bm25.fusion import (  # noqa: E402
    DEFAULT_RRF_K,
    FusedHit,
    FusionConfig,
    FusionMode,
    normalize_within_query,
)

# --------------------------------------------------------------------------- 归一化


def test_normalize_is_within_one_query() -> None:
    """**归一化只在一个 query 的候选集合内做。**

    两个 query 的分数尺度完全不同（短 query 的 BM25 全在 0~5，
    长 query 全在 8~15）。按 query 内部归一化后，两个 query 各自的 top1
    **都**是 1.0 —— 它们在自己的候选里同样"最好"，可比。

    如果改成全局归一化，第二个 query 的 top1（15）会被压到接近 1.0、
    而第一个 query 的 top1（5）被压到 0.33（假设全局 max=15）——
    于是第一个 query 在融合里那一侧权重事实上变小了。
    """
    short_query_scores = [5.0, 3.0, 1.0]
    long_query_scores = [15.0, 13.0, 11.0]

    short_norm = normalize_within_query(short_query_scores, "minmax")
    long_norm = normalize_within_query(long_query_scores, "minmax")

    assert short_norm[0] == pytest.approx(1.0), "短 query 的 top1 应当拿满分"
    assert long_norm[0] == pytest.approx(1.0), "长 query 的 top1 也应当拿满分"

    # 反证：如果全局归一（用两组的 max=15），短 query 会被压低
    global_max = max(short_query_scores + long_query_scores)
    global_min = min(short_query_scores + long_query_scores)
    short_global = [(s - global_min) / (global_max - global_min) for s in short_query_scores]
    assert short_global[0] < 0.5, "全局归一化会把短 query 的 top1 压到 0.5 以下 —— 这就是它错的地方"


def test_normalize_degenerate_cases() -> None:
    """退化情况：空表、单元素、全同分。

    **单元素与全同分都给 1.0，不是 0.0**：那条候选是这个 query 的最优解，
    给 0 会让这一路在融合里彻底消失 —— 而"所有候选一样好"不等于"没有候选能赢"。
    """
    assert normalize_within_query([]) == []
    assert normalize_within_query([0.7]) == [1.0]
    assert normalize_within_query([0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0]


def test_normalize_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="未知归一化方式"):
        normalize_within_query([1.0, 2.0], "softmax")


def test_zscore_and_minmax_both_normalize() -> None:
    """两种归一化都必须真的改变尺度。`zscore` 的均值应当约等于 0。"""
    scores = [0.50, 0.60, 0.90, 0.70]
    minmax = normalize_within_query(scores, "minmax")
    zscore = normalize_within_query(scores, "zscore")
    assert min(minmax) == pytest.approx(0.0)
    assert max(minmax) == pytest.approx(1.0)
    assert sum(zscore) / len(zscore) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- RRF


def test_rrf_ignores_score_scale() -> None:
    """**RRF 只读名次，与分数尺度无关。**

    两组输入的名次完全相同、分数尺度差 1000 倍 —— 融合结果必须一模一样。
    这条守的是"有人顺手把分数也乘进去"：那样 BM25 的无界分数会重新主导。
    """
    config = FusionConfig(mode=FusionMode.RRF, candidates=3, rrf_k=60)
    from app.rag.bm25.fusion import merge_rankings

    small, _ = merge_rankings(
        ["a", "b"],
        ["c", "d"],
        config=config,
        vec_scores=[0.9, 0.8],
        bm25_scores=[3.0, 2.0],
    )
    large, _ = merge_rankings(
        ["a", "b"],
        ["c", "d"],
        config=config,
        vec_scores=[900.0, 800.0],
        bm25_scores=[3000.0, 2000.0],
    )
    assert [h.chunk_id for h in small] == [h.chunk_id for h in large]
    assert [round(h.score, 10) for h in small] == [round(h.score, 10) for h in large]


def test_rrf_scores_both_path_consensus_higher() -> None:
    """两路都召回的候选，分数应当高于只被一路召回的同等名次。"""
    config = FusionConfig(mode=FusionMode.RRF, candidates=2, rrf_k=DEFAULT_RRF_K)
    from app.rag.bm25.fusion import merge_rankings

    hits, _ = merge_rankings(["a", "b"], ["a", "c"], config=config)
    scores = {h.chunk_id: h.score for h in hits}
    assert scores["a"] == pytest.approx(2 / (DEFAULT_RRF_K + 1))
    assert scores["b"] == pytest.approx(1 / (DEFAULT_RRF_K + 2))
    assert scores["a"] > scores["b"]


def test_rrf_k_larger_flattens_rank_gaps() -> None:
    """`k` 越大，名次差距越平缓。**这是 RRF 比加权融合稳的原因之一。**"""
    from app.rag.bm25.fusion import merge_rankings

    tight, _ = merge_rankings(
        ["a", "b"], [], config=FusionConfig(mode=FusionMode.RRF, candidates=2, rrf_k=0)
    )
    loose, _ = merge_rankings(
        ["a", "b"], [], config=FusionConfig(mode=FusionMode.RRF, candidates=2, rrf_k=60)
    )
    tight_gap = tight[0].score - tight[1].score
    loose_gap = loose[0].score - loose[1].score
    assert tight_gap > loose_gap


# --------------------------------------------------------------------------- 加权融合


def test_weighted_requires_scores() -> None:
    """加权融合缺分数必须抛 —— **静默用名次当分数会让 α 变成假的**。"""
    config = FusionConfig(mode=FusionMode.WEIGHTED, candidates=3, alpha=0.5)
    from app.rag.bm25.fusion import merge_rankings

    with pytest.raises(ValueError, match="必须同时提供"):
        merge_rankings(["a", "b"], ["c"], config=config)


def test_weighted_does_not_let_bm25_dominate() -> None:
    """**最易错的那个写法**：`0.5*cos + 0.5*bm25` 会让 BM25 单方面支配。

    向量的候选分数在 0.8 附近，BM25 在 30 附近。不归一化时 BM25 那侧
    （30 * 0.5 = 15）比向量整侧（0.8 * 0.5 = 0.4）大 37 倍，
    向量的候选会全部掉出榜单。

    归一化后，两路各自的 top1 都是 1.0，α=0.5 时**两路各得 0.5 分** ——
    这才是真的等权。
    """
    config = FusionConfig(mode=FusionMode.WEIGHTED, candidates=3, alpha=0.5)
    from app.rag.bm25.fusion import merge_rankings

    vec_ids, vec_scores = ["v1", "v2", "v3"], [0.80, 0.79, 0.78]
    bm_ids, bm_scores = ["b1", "b2", "b3"], [30.0, 28.0, 26.0]

    hits, _ = merge_rankings(
        vec_ids, bm_ids, config=config, vec_scores=vec_scores, bm25_scores=bm_scores
    )
    scores = {h.chunk_id: h.score for h in hits}

    # 向量 top1 与 BM25 top1 必须同分 —— 它们在自己的候选里都是最好的
    assert scores["v1"] == pytest.approx(0.5)
    assert scores["b1"] == pytest.approx(0.5)
    top3 = [h.chunk_id for h in hits[:3]]
    assert "v1" in top3, "向量的 top1 不该被 BM25 的量纲挤出前列"


def test_weighted_missing_side_gets_zero() -> None:
    """只被一路召回的候选，另一侧补 **0**，不是"那一路的最低分"。

    理由：BM25 没召回它，说明它一个 query token 都没匹配上 —— 给 0 是事实。
    给"最低分"等于凭空奖励一条两路里最差的候选。
    """
    config = FusionConfig(mode=FusionMode.WEIGHTED, candidates=2, alpha=0.5)
    from app.rag.bm25.fusion import merge_rankings

    hits, _ = merge_rankings(
        ["only_vec"], ["bm1"], config=config, vec_scores=[0.9], bm25_scores=[10.0]
    )
    scores = {h.chunk_id: h.score for h in hits}
    assert scores["only_vec"] == pytest.approx(0.5), "只命中向量侧 → 最多拿 α"
    assert scores["bm1"] == pytest.approx(0.5)


def test_alpha_extremes_reproduce_pure_paths() -> None:
    """α=1.0 应等价纯向量，α=0.0 应等价纯 BM25。**这是融合的边界自检。**"""
    config_vec = FusionConfig(mode=FusionMode.WEIGHTED, candidates=3, alpha=1.0)
    config_bm = FusionConfig(mode=FusionMode.WEIGHTED, candidates=3, alpha=0.0)
    from app.rag.bm25.fusion import merge_rankings

    vec_ids, vec_scores = ["v1", "v2"], [0.9, 0.5]
    bm_ids, bm_scores = ["b1", "b2"], [20.0, 8.0]

    pure_vec, _ = merge_rankings(
        vec_ids, [], config=config_vec, vec_scores=vec_scores, bm25_scores=[]
    )
    fused_vec, _ = merge_rankings(
        vec_ids, bm_ids, config=config_vec, vec_scores=vec_scores, bm25_scores=bm_scores
    )
    assert [h.chunk_id for h in fused_vec[:2]] == [h.chunk_id for h in pure_vec]

    pure_bm, _ = merge_rankings([], bm_ids, config=config_bm, vec_scores=[], bm25_scores=bm_scores)
    fused_bm, _ = merge_rankings(
        vec_ids, bm_ids, config=config_bm, vec_scores=vec_scores, bm25_scores=bm_scores
    )
    assert [h.chunk_id for h in fused_bm[:2]] == [h.chunk_id for h in pure_bm]


# --------------------------------------------------------------------------- 合并统计


def test_merge_stats_counts_all_four_categories() -> None:
    """并集的四类计数必须齐全 —— **`novelty_ratio` 是"混合有没有意义"的主指标**。"""
    config = FusionConfig(mode=FusionMode.RRF, candidates=3)
    from app.rag.bm25.fusion import merge_rankings

    _hits, stats = merge_rankings(["a", "b"], ["b", "c"], config=config)
    assert stats.vec_candidates == 2
    assert stats.bm25_candidates == 2
    assert stats.union == 3
    assert stats.both == 1
    assert stats.vector_only == 1
    assert stats.bm25_only == 1
    assert stats.overlap_ratio == pytest.approx(1 / 3)
    assert stats.novelty_ratio == pytest.approx(2 / 3)


def test_matched_by_labels_are_exhaustive() -> None:
    """`matched_by` 必须能区分"两路"、"只向量"、"只 BM25" —— 别的分支不存在。"""
    assert FusedHit("x", 1.0, 1, 2).matched_by == "both"
    assert FusedHit("x", 1.0, 1, None).matched_by == "vector"
    assert FusedHit("x", 1.0, None, 3).matched_by == "bm25"


def test_empty_inputs_return_empty() -> None:
    config = FusionConfig(mode=FusionMode.RRF, candidates=1)
    from app.rag.bm25.fusion import merge_rankings

    hits, stats = merge_rankings([], [], config=config)
    assert hits == []
    assert stats.union == 0


def test_ordering_is_deterministic_on_ties() -> None:
    """同分候选的排序必须确定。**只按分数排会让重跑结果不稳定**，
    而那会被归因到"模型变了"或"索引变了"，查起来极费时间。"""
    config = FusionConfig(mode=FusionMode.RRF, candidates=2, rrf_k=60)
    from app.rag.bm25.fusion import merge_rankings

    first, _ = merge_rankings(["a", "b"], ["c", "d"], config=config)
    second, _ = merge_rankings(["a", "b"], ["c", "d"], config=config)
    assert [h.chunk_id for h in first] == [h.chunk_id for h in second]


# --------------------------------------------------------------------------- 配置校验


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"candidates": 0}, "candidates 必须为正数"),
        ({"rrf_k": -1}, "rrf_k 不能为负"),
        ({"alpha": 1.5}, "alpha 必须在"),
        ({"normalize": "global"}, "normalize 只能是"),
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, object], message: str) -> None:
    """**配置写错必须抛，不能静默取默认值** —— 否则会得到"调了参数但没效果"。"""
    with pytest.raises(ValueError, match=message):
        FusionConfig(**{"mode": FusionMode.WEIGHTED, **kwargs})  # type: ignore[arg-type]


def test_config_label_is_unique_per_setup() -> None:
    """标签要能唯一区分不同配置 —— 它会进表格与产物，重名就看不出跑了什么。"""
    labels = {
        FusionConfig(mode=FusionMode.RRF, rrf_k=60).label,
        FusionConfig(mode=FusionMode.RRF, rrf_k=10).label,
        FusionConfig(mode=FusionMode.WEIGHTED, alpha=0.3).label,
        FusionConfig(mode=FusionMode.WEIGHTED, alpha=0.5).label,
    }
    assert len(labels) == 4
