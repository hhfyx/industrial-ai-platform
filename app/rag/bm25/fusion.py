"""混合检索：**两路候选的合并与融合**。

## 这个模块存在的理由

向量检索和 BM25 的**分数不可比**：

| 路 | 量纲 | 典型范围 |
| --- | --- | --- |
| 向量 | 余弦相似度 | [-1, 1]，本项目实测 0.5~0.8 |
| BM25 | 无界正数 | [0, 30+]，实测均值 8 上下 |

所以下面这一行看起来最自然、实际最错的写法：

    score = 0.5 * cos_sim + 0.5 * bm25_score        ← 让 BM25 单方面支配

余弦那 0.5 分被 BM25 的 8 分彻底淹没，`α` 变成假的 —— 无论怎么扫 0.3/0.5/0.7，
结果都由 BM25 单方面决定，而人会以为"α 调了没差别是因为两路贡献相近"。

## 三条路线

| 路线 | 需要归一化吗 | 复杂度 | 什么时候用 |
| --- | --- | --- | --- |
| **RRF** | **不需要** | 低 | 首选。只用名次，免疫量纲 |
| 加权融合 | **必须，且按 query 内部** | 中 | 要精细控制两路权重时 |
| 加权融合（全库归一） | 必须但**做错了** | 中 | **不要用**，见下 |

### RRF 为什么不需要归一化

    rrf(d) = Σ_{路 ∈ {vec, bm25}} 1 / (k + rank_路(d))      k 常取 60

它只读名次。第 1 名在两路都拿 `1/61`，第 20 名拿 `1/80` —— 与分数是 0.9 还是 30 无关。
所以 BM25 的 30 分和余弦的 0.9 分处在同一个量纲上：**都是"第几名"**。

### 加权融合的归一化为什么必须**按 query 内部**

这是本周的必答问题，也是最容易做错的地方。

**反例（全库统一 min-max）**：把全库所有 query 的 BM25 分数一起归一化，
用同一个 `min`/`max`。后果是 —— 一个 query 的两路分数分布尺度与另一个 query 完全不同：

- 短 query（「TTFT」）的 BM25 分数可能全在 0~5，最高分 5 就是它的"好"
- 长 query（「多进程部署时本地模型怎么避免每个进程各加载一份」）可能全在 8~15，
  它的 8 分在全局尺度上低于短 query 的 5 分

全局归一化后，**长 query 的分数被系统性压低**，于是在融合里它那一侧的 `α` 权重
事实上变小了。而调参时看到的是"α=0.5 时某些题变差"，会去怀疑 α，
真正的原因是归一化把 query 之间的尺度差压成了偏置。

**正确做法**：归一化**只在一个 query 自己的候选集合内**做。
每次查询重新算 `min`/`max`，两路各自归一，再按 α 合。这样"这个 query 里
向量第 1 名"与"这个 query 里 BM25 第 1 名"都是 1.0，可比。

本模块把这条**写进类型里**：`normalize_within_query()` 只接受**单个 query 的分数列表**，
拿不到「全库」那个上下文。接口层面就做不出全库归一化。

## 为什么不直接把两路塞进 `VectorStore`

`SearchHit.score` 的契约是**余弦相似度**（[-1, 1]，越大越像）。
BM25 分数无界，塞进去会让"分数"在库里有两个含义 —— W4 因为
"Chroma distance vs 相似度"已经踩过一次。所以融合层单独存在，
两路各自保留自己的命中类型（`SearchHit` / `BM25Hit`），融合在边界上发生。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_RRF_K: int = 60
"""RRF 的平滑常数。

取 60 是 Cormack 等人 2009 年那篇论文的推荐值，也是 Elasticsearch
`reciprocal_rank_fusion` 的默认。它的作用是**压平名次差异的陡度**：

- `k=0` 时第 1 名拿 1.0、第 2 名 0.5、第 10 名 0.1 —— 前几名垄断
- `k=60` 时第 1 名 1/61、第 2 名 1/62、第 10 名 1/70 —— 差距温和

本项目候选窗口 20~30 条，`k=60` 让窗口内所有名次都落在 [1/90, 1/61] 区间，
即"谁更靠前"有意义但不会一家独大。**这是 RRF 比加权融合稳的第一个原因**：
它对名次的小抖动不敏感。
"""

DEFAULT_CANDIDATES: int = 20
"""每路先取多少条候选再合并。

任务书写的是 20~30。取 20 的理由：**

- 两路的第 20 名在 RRF 里只拿 `1/80`，权重已经很低，再往后加对结果影响 < 1%
- 但每增加 10 条候选，两路各多 10 次 embed/BM25 打分 —— BM25 便宜，embed 不便宜
- **窗口太宽会掩盖"召回不进去"这个真问题**：top30 里都没有 gold，那是切分或表述问题，
  扩到 50 只是把问题推后

这个数可调，并且它会进产物元数据（`candidates` 字段）。
"""


class FusionMode(StrEnum):
    """融合方式。**做成枚举而不是字符串**，让"拼错一个词就静默降级"不可能发生。"""

    RRF = "rrf"
    WEIGHTED = "weighted"


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """融合参数。**frozen**：融合参数进评测记录后不该被人改。"""

    mode: FusionMode = FusionMode.RRF
    candidates: int = DEFAULT_CANDIDATES
    rrf_k: int = DEFAULT_RRF_K
    alpha: float = 0.5
    """加权融合里**向量侧**的权重。`1 - alpha` 给 BM25。

    只在 `mode=WEIGHTED` 时生效。默认 0.5 是"两路等权"，
    但注意它不等于"两路贡献相等" —— 归一化后两路各自最高分都是 1.0，
    所以 α=0.5 时两路各能贡献 0.5 分，这是真的等权。
    """

    normalize: str = "minmax"
    """归一化方式：`minmax` 或 `zscore`。

    两者都是**按 query 内部**做，区别只在"分布形状不同时谁更稳"：

    - `minmax`：压到 [0, 1]，对极端值敏感（一个离群高分会把其余全压扁）
    - `zscore`：均值 0 方差 1，保留相对距离，但两路的"零"不对齐
      （向量均值≈0.6、BM25 均值≈8，z-score 后两者都是 0，此时 0 分代表不同含义）

    默认 `minmax`：它的语义最直白（"这个 query 里最好的那条 = 1.0"），
    调 α 时时能解释 α 到底在加权什么。`zscore` 保留着做对照。
    """

    def __post_init__(self) -> None:
        if self.candidates <= 0:
            raise ValueError(f"candidates 必须为正数，收到 {self.candidates}")
        if self.rrf_k < 0:
            raise ValueError(f"rrf_k 不能为负，收到 {self.rrf_k}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha 必须在 [0, 1]，收到 {self.alpha}")
        if self.normalize not in ("minmax", "zscore"):
            raise ValueError(f"normalize 只能是 minmax / zscore，收到 {self.normalize!r}")

    def as_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "mode": self.mode.value,
            "candidates": self.candidates,
        }
        if self.mode is FusionMode.RRF:
            data["rrf_k"] = self.rrf_k
        else:
            data["alpha"] = self.alpha
            data["normalize"] = self.normalize
        return data

    @property
    def label(self) -> str:
        """人读的短标签，进表格与产物文件名。**必须唯一可辨**。"""
        if self.mode is FusionMode.RRF:
            return f"rrf(k={self.rrf_k})"
        return f"weighted(α={self.alpha:g},{self.normalize})"


def normalize_within_query(scores: Sequence[float], method: str = "minmax") -> list[float]:
    """把一个 query 的**一路**分数归一化到可比区间。

    ## 参数是"一个 query 的分数列表"，不是"全库"

    这个签名是刻意的：它拿不到全库上下文，所以**在接口层面就做不出全库归一化**。
    本周最容易错的事就是顺手对全库做 min-max —— 那会让不同 query 的
    分数尺度差变成系统性偏置（见模块 docstring 的反例）。

    ## 退化情况怎么处理

    | 情况 | 处理 | 理由 |
    | --- | --- | --- |
    | 空列表 | 返回空 | —— |
    | 只有一个元素 | 返回 `[1.0]` | **不是 0.0**：它是这个 query 的最优候选，该拿满分 |
    | 全部相同分数 | 全部 `1.0` | 同上。全 0 会让这一路在融合里彻底消失，而"所有候选一样好"不等于"没有候选能赢" |
    """
    if not scores:
        return []
    if method == "minmax":
        low, high = min(scores), max(scores)
        if high - low < 1e-12:
            # 全部同分：**给 1.0 而不是 0.0**。见 docstring 的表格。
            return [1.0] * len(scores)
        return [(s - low) / (high - low) for s in scores]
    if method == "zscore":
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = variance**0.5
        if std < 1e-12:
            return [1.0] * len(scores)
        return [(s - mean) / std for s in scores]
    raise ValueError(f"未知归一化方式：{method!r}（可用：minmax / zscore）")


@dataclass(frozen=True, slots=True)
class FusedHit:
    """融合后的一条候选。**分数只在同一份融合配置内可比**。

    它刻意不复用 `SearchHit`（余弦契约）也不复用 `BM25Hit`（无界正数）——
    三种分数语义混在一个类型里，调用方迟早会拿一个去和另一个比。
    """

    chunk_id: str
    score: float
    """融合后的分数。RRF 下是 [0, 2/(k+1)] 量级；加权下是 [0, 1]。"""

    vec_rank: int | None
    bm25_rank: int | None
    """两路各自的名次。`None` 表示这一路没进候选。

    **必须留着**：没有它，"融合后这条是 BM25 顶上来的还是向量顶上来的"
    就查不到了 —— 而这正是判断"混合有没有意义"的唯一线索。
    """

    @property
    def matched_by(self) -> str:
        """哪一路（或两路）召回了它。用于统计交集/独有比例。"""
        if self.vec_rank is not None and self.bm25_rank is not None:
            return "both"
        if self.vec_rank is not None:
            return "vector"
        return "bm25"


@dataclass
class MergeStats:
    """**两路候选的合并比例** —— 任务点名要的那个数字。

    它直接回答"混合有没有意义"：如果两路命中的是同一批块（`both` 占比接近 100%），
    融合就不会改变任何结果，加一层复杂度是白加。反之如果 `vector_only` 与
    `bm25_only` 都很大，说明两路各自看到了对方看不到的东西，融合才有价值。
    """

    vec_candidates: int = 0
    bm25_candidates: int = 0
    union: int = 0
    both: int = 0
    vector_only: int = 0
    bm25_only: int = 0

    @property
    def overlap_ratio(self) -> float:
        """交集占并集的比例（Jaccard）。1.0 = 两路完全一样。"""
        return self.both / self.union if self.union else 0.0

    @property
    def novelty_ratio(self) -> float:
        """独有候选占并集的比例。**这是"混合有意义"的主指标**。"""
        return (self.vector_only + self.bm25_only) / self.union if self.union else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "vec_candidates": self.vec_candidates,
            "bm25_candidates": self.bm25_candidates,
            "union": self.union,
            "both": self.both,
            "vector_only": self.vector_only,
            "bm25_only": self.bm25_only,
            "overlap_ratio": round(self.overlap_ratio, 4),
            "novelty_ratio": round(self.novelty_ratio, 4),
        }


def merge_rankings(
    vec_ranked: Sequence[str],
    bm25_ranked: Sequence[str],
    *,
    config: FusionConfig,
    vec_scores: Sequence[float] | None = None,
    bm25_scores: Sequence[float] | None = None,
) -> tuple[list[FusedHit], MergeStats]:
    """合并两路候选并打分。

    ## 输入是**名次列表**（chunk_id 按分数降序排列），不是分数列表

    这样 RRF 与加权融合共用同一个入口 —— 两种模式的区别只在"怎么从名次/分数
    得到一个数"，而"有哪些候选"这件事两边一致。

    ## 加权融合必须同时给分数

    `vec_scores` / `bm25_scores` 与 `*_ranked` 一一对应（同序）。
    RRF 模式下忽略分数（**这是它的设计目的**）；加权模式下缺分数直接抛 ——
    静默用名次当分数会让 α 变成假的。
    """
    if not vec_ranked and not bm25_ranked:
        return [], MergeStats()

    vec_rank = {cid: i for i, cid in enumerate(vec_ranked, start=1)}
    bm25_rank = {cid: i for i, cid in enumerate(bm25_ranked, start=1)}
    union_ids = list(dict.fromkeys([*vec_ranked, *bm25_ranked]))

    stats = MergeStats(
        vec_candidates=len(vec_ranked),
        bm25_candidates=len(bm25_ranked),
        union=len(union_ids),
        both=sum(1 for cid in union_ids if cid in vec_rank and cid in bm25_rank),
        vector_only=sum(1 for cid in union_ids if cid in vec_rank and cid not in bm25_rank),
        bm25_only=sum(1 for cid in union_ids if cid in bm25_rank and cid not in vec_rank),
    )

    if config.mode is FusionMode.RRF:
        scores = _rrf_scores(union_ids, vec_rank, bm25_rank, k=config.rrf_k)
    else:
        if vec_scores is None or bm25_scores is None:
            raise ValueError(
                "加权融合必须同时提供 vec_scores 与 bm25_scores ——"
                "拿名次当分数会让 α 变成假的（见 FusionConfig.alpha 的说明）"
            )
        # **两路各自按 query 内部归一化**。这里传进去的就是"这个 query 的候选分数"，
        # 拿不到全库上下文 —— 见 normalize_within_query 的签名说明。
        vec_norm = normalize_within_query(vec_scores, config.normalize)
        bm25_norm = normalize_within_query(bm25_scores, config.normalize)
        fused = _weighted_scores(
            union_ids,
            vec_rank,
            bm25_rank,
            vec_norm=dict(zip(vec_ranked, vec_norm, strict=True)),
            bm25_norm=dict(zip(bm25_ranked, bm25_norm, strict=True)),
            alpha=config.alpha,
        )
        scores = fused

    # **遍历 `scores` 而不是 `union_ids`**：加权模式下权重为 0 的那一路的独有候选
    # 已被 `_weighted_scores` 剔掉（见那里的边界说明），它们不该出现在结果里。
    hits = [
        FusedHit(
            chunk_id=cid,
            score=score,
            vec_rank=vec_rank.get(cid),
            bm25_rank=bm25_rank.get(cid),
        )
        for cid, score in scores.items()
    ]
    # **排序键要确定性**：分数相同时按 (vec_rank, bm25_rank, chunk_id) 排。
    # 只按分数排会在等分候选上得到依赖 dict 顺序的结果 —— 那在不同机器上不稳定，
    # 而"重跑一次结果变了"会被归因到模型或索引，查起来极费时间。
    hits.sort(
        key=lambda h: (
            -h.score,
            h.vec_rank if h.vec_rank is not None else 10**9,
            h.bm25_rank if h.bm25_rank is not None else 10**9,
            h.chunk_id,
        )
    )
    return hits, stats


def _rrf_scores(
    union_ids: Sequence[str],
    vec_rank: dict[str, int],
    bm25_rank: dict[str, int],
    *,
    k: int,
) -> dict[str, float]:
    """`Σ 1/(k + rank)`。**只读名次，完全不碰分数。**"""
    scores: dict[str, float] = {}
    for cid in union_ids:
        total = 0.0
        if (rank := vec_rank.get(cid)) is not None:
            total += 1.0 / (k + rank)
        if (rank := bm25_rank.get(cid)) is not None:
            total += 1.0 / (k + rank)
        scores[cid] = total
    return scores


def _weighted_scores(
    union_ids: Sequence[str],
    vec_rank: dict[str, int],
    bm25_rank: dict[str, int],
    *,
    vec_norm: dict[str, float],
    bm25_norm: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    """`α·vec_norm + (1-α)·bm25_norm`。

    ## 只有一路召回的候选怎么算另一半

    这是加权融合里第二个容易错的点。候选 `X` 只在向量候选里、不在 BM25 候选里 ——
    BM25 那半该给 0，还是给"这个 query 里 BM25 的最低分"？

    **给 0**，理由是：

    - BM25 没召回它，说明它没能匹配上任何 query token。给 0 是**事实**
    - 给"最低分"等于凭空奖励一条"两路里最差"的候选，而那会让
      "BM25 完全没看到它" 变成 "BM25 认为它最差" —— 两者不是一回事

    代价是：只被一路召回的候选天然吃亏（最多拿 `α` 而不是 `α + (1-α)` 的一部分）。
    这是加权融合相对于 RRF 的**结构性劣势**：

        RRF 里只被一路召回的第 1 名拿 1/61 ≈ 0.0164
        两路都召回的第 5 名拿 2/65 ≈ 0.0308      ← 两路共识赢

    加权融合下同理（共识候选拿两份加权分）。所以**融合本来就偏好共识**，
    这不是 bug，是设计。真正的问题是：如果这个偏好太强，独有召回就白费了 ——
    而那正是"混合有没有意义"这个统计（`novelty_ratio`）要盯的东西。

    ## 权重为 0 的那一路，它的独有候选要**整个剔掉**（边界修正）

    α=0.0 时向量的权重是 0，但它的候选归一化后仍可能是 1.0 ——
    乘上 0 得 0，会与 BM25 侧归一化得 0 的候选**同分**，
    再被名次兜底一排序，纯 BM25 的排名就被向量候选插队了。

    这不只是数值巧合：**权重 0 意味着"这条路不参与决策"**，
    那它的独有候选就根本不该出现在结果里。剔掉之后 α=0 严格等价纯 BM25、
    α=1.0 严格等价纯向量 —— 边界自检（`test_alpha_extremes_reproduce_pure_paths`）
    就是靠这条成立的。
    """
    scores: dict[str, float] = {}
    vec_weight = alpha
    bm25_weight = 1.0 - alpha
    for cid in union_ids:
        in_vec = cid in vec_norm
        in_bm25 = cid in bm25_norm
        # 只被权重为 0 的那一路召回 → 它的分数必然是 0，且不含任何该路的信号
        if in_vec and not in_bm25 and vec_weight <= 0.0:
            continue
        if in_bm25 and not in_vec and bm25_weight <= 0.0:
            continue
        scores[cid] = vec_weight * vec_norm.get(cid, 0.0) + bm25_weight * bm25_norm.get(cid, 0.0)
    return scores


__all__ = [
    "DEFAULT_CANDIDATES",
    "DEFAULT_RRF_K",
    "FusedHit",
    "FusionConfig",
    "FusionMode",
    "MergeStats",
    "merge_rankings",
    "normalize_within_query",
]
