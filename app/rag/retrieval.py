"""检索段（Retrieval）：query 向量化 → 相似度 → TopK。

**这一段是"每次都跑、必须快"的**。它和索引段的区别写在 `indexing.py` 顶部；
这里只强调一件事：**检索段不产生新信息，它只负责"从已有的块里挑"**。
任何"检索时顺手做点加工"的想法（重写 query、二次摘要、在线聚类）都属于
"加了一层看不见的延迟"，要加就得单独一轮、单独一轮的指标。

## 三步各自的输入输出

    ① query 向量化  str                → list[float]
    ② 相似度        vector + store     → list[SearchHit]（带分数）
    ③ TopK          list[SearchHit]    → list[Chunk]（可再按预算裁）

**① 必须用和索引完全相同的 embedder**，这不是约定而是硬约束：
查询向量和文档向量不在同一个空间时，余弦相似度算出来的是噪声，
而且**不会报错**。`Retriever` 因此在构造时要求显式传入 embedder，
不给"忘了传就默认一个"的机会 —— 默认值在这是有害的。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.rag.embedding import Embedder
from app.rag.schema import Chunk
from app.rag.stores import SearchHit, VectorStore

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
"""默认取 5 条。**这是全项目唯一一处 top_k 默认值，别处都必须引用它。**

## 为什么从 4 改成 5（不是"调了个参数"）

原来这里写 4，而 `QAEngine` / CLI `--top-k` / UI 滑块都是 5 —— 于是
"默认取几条"这个问题有四个答案，且**没有一个地方写得出来它是哪个**。
W5 要跑 `top_k ∈ {3, 5, 8}` 的矩阵，如果基线本身是 4 还是 5 都说不清，
那张矩阵的第一个问题就是"你拿什么当对照"。

收敛到 5 的理由是**对齐已经上线的三处**，不是我认为 5 比 4 好：
4 与 5 都还没被实验证明过（README 已知问题 #5、design-decisions §4 都写着这点）。

## 为什么不是 10

按每块 500 字算，5 块 ≈ 2500 字 ≈ 1690 token，加系统提示词与输出仍在预算内。
取 10 条会把这个数字翻到 3500 token，而在召回率没验证过的前提下
（见 `HashingEmbedder` 的说明），多给的几条里大部分是噪声 ——
**噪声进上下文会实实在在拉低答案质量**，这不是"多给点总没坏处"。

> 具体数字见 `docs/design-decisions.md` §4：500 字 + 50 重叠、5 × 500 = 2500 字
> ≈ 1690 token，与这里的口径一致。
"""


def default_top_k() -> int:
    """默认 top_k 的**取值出口**。

    存在的意义不是多一层函数，而是让"谁在用默认值"这件事可被搜索到：
    调用方写 `default_top_k()` 时，W5 改参数只改一处；
    写死 `top_k=5` 的地方则在评审时一眼能看见。`tests/test_rag_skeleton.py`
    有一条断言守住"CLI / UI / QAEngine / Retriever 四处一致"。
    """
    return DEFAULT_TOP_K


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """检索段的完整产物。"""

    query: str
    hits: tuple[SearchHit, ...]
    top_k: int
    elapsed_ms: int = 0

    @property
    def chunks(self) -> list[Chunk]:
        """只要块（丢掉分数）。分数跨库不可比，见 `stores/base.SearchHit`。"""
        return [hit.chunk for hit in self.hits]

    @property
    def is_empty(self) -> bool:
        return not self.hits

    def scores(self) -> list[float]:
        return [hit.score for hit in self.hits]


class Retriever:
    """检索器：query → TopK 命中块。"""

    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k 必须为正数")
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        k: int | None = None,
    ) -> RetrievalResult:
        """检索。

        ## 参数名契约：`top_k` 是规范名，`k` 是它的别名

        W4 的开工清单里写的是 `retrieve(query, k=5)`，而实现里叫 `top_k` ——
        两者都留，是因为两边都已经有真实调用方：项目内部一路用 `top_k`
        （`QAEngine`、CLI、UI 都是 `top_k=`），而外部/文档口径写 `k`。
        只留一个，另一方就要改调用点；而这类"改个参数名"的改动最容易
        在某个没被测试覆盖的分支上变成 `TypeError`。

        **两者同时传且不相等时直接抛** —— 不静默取其一。同时传两个不同的
        条数说明调用方自己也没想清要几条，猜一个会得到"检索 5 条、上下文按 3 条拼"
        这种事后无法解释的结果。

        **空 query 直接返回空结果，不调 embedder**：空字符串的向量是零向量，
        零向量与任何向量的余弦都是 0，于是会返回一堆"分数全是 0"的块 ——
        看起来检索成功，实际是随便挑的。让它在源头就返回空，问题才能被看见。
        """
        started = time.perf_counter()
        if k is not None and top_k is not None and k != top_k:
            raise ValueError(f"top_k({top_k}) 与 k({k}) 同时给出且不相等，无法判断取几条")
        requested = top_k if top_k is not None else k
        k_value = self.top_k if requested is None else requested
        if k_value <= 0:
            raise ValueError("top_k 必须为正数")

        text = (query or "").strip()
        if not text:
            return RetrievalResult(query=query, hits=(), top_k=k_value, elapsed_ms=_ms(started))

        # ① query 向量化：与索引共用 embedder，见模块 docstring。
        #
        # **调 embed_query 而不是 embed**：bge 这类模型要求 query 侧加检索指令前缀
        # （"为这个句子生成表示以用于检索相关文章："），passage 侧不加。
        # 忘加不会有任何报错，只会让命中率低一点 —— 而"低一点"最容易被
        # 归因到切分/top_k/模型不行。所以把前缀这件事放进接口，调用方碰不到。
        query_vector = self.embedder.embed_query([text])[0]

        # ②③ 相似度 + TopK 由 store 一次完成（store 才知道怎么做相似度）
        hits = self.store.search(query_vector, top_k=k_value)

        result = RetrievalResult(
            query=query,
            hits=tuple(hits),
            top_k=k_value,
            elapsed_ms=_ms(started),
        )
        logger.debug(
            "检索完成：%r → %d 条（%d ms，最高分 %.4f）",
            text[:40],
            len(hits),
            result.elapsed_ms,
            hits[0].score if hits else 0.0,
        )
        return result


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


__all__ = ["DEFAULT_TOP_K", "RetrievalResult", "Retriever", "default_top_k"]
