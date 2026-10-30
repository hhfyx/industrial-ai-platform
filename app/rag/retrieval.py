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

DEFAULT_TOP_K = 4
"""默认取 4 条。

**为什么不是 10**：按每块 500 字算，4 块 ≈ 2000 字 ≈ 1400 token，
留出了系统提示词、问题和输出的空间。取 10 条会把这个数字翻到 3500 token，
而在召回率没验证过的前提下（见 `HashingEmbedder` 的说明），
多给的 6 条里大部分是噪声 —— **噪声进上下文会实实在在拉低答案质量**，
这不是"多给点总没坏处"。
"""


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

    def retrieve(self, query: str, *, top_k: int | None = None) -> RetrievalResult:
        """检索。

        **空 query 直接返回空结果，不调 embedder**：空字符串的向量是零向量，
        零向量与任何向量的余弦都是 0，于是会返回一堆"分数全是 0"的块 ——
        看起来检索成功，实际是随便挑的。让它在源头就返回空，问题才能被看见。
        """
        started = time.perf_counter()
        k = self.top_k if top_k is None else top_k
        if k <= 0:
            raise ValueError("top_k 必须为正数")

        text = (query or "").strip()
        if not text:
            return RetrievalResult(query=query, hits=(), top_k=k, elapsed_ms=_ms(started))

        # ① query 向量化：与索引共用 embedder，见模块 docstring。
        #
        # **调 embed_query 而不是 embed**：bge 这类模型要求 query 侧加检索指令前缀
        # （"为这个句子生成表示以用于检索相关文章："），passage 侧不加。
        # 忘加不会有任何报错，只会让命中率低一点 —— 而"低一点"最容易被
        # 归因到切分/top_k/模型不行。所以把前缀这件事放进接口，调用方碰不到。
        query_vector = self.embedder.embed_query([text])[0]

        # ②③ 相似度 + TopK 由 store 一次完成（store 才知道怎么做相似度）
        hits = self.store.search(query_vector, top_k=k)

        result = RetrievalResult(
            query=query,
            hits=tuple(hits),
            top_k=k,
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


__all__ = ["DEFAULT_TOP_K", "RetrievalResult", "Retriever"]
