"""内存向量库：零依赖、可测、可复现的默认实现。

**存在的理由不是"简单"，是"把正确性和外部服务解耦"。**
一套 RAG 链路里，"检索命中与否"至少要能被断言。如果 store 是远程服务，
那条断言就同时测到了网络、鉴权、服务版本和我们的代码 —— 失败了无法归因。

所以测试与本地跑默认走这里；生产换 `chroma`。两者同一接口，
"换了之后行为变了吗"可以用同一套测试回答。

## 相似度为什么用余弦

对 L2 归一化过的向量，余弦相似度 = 内积。这里仍然显式算余弦（除以两个模长），
因为**不能假设所有 embedder 都归一化** —— `HashingEmbedder` 归了，模型导入的
第三方 embedder 不一定。少了除法，未归一化的实现会让"长文本向量模长大"
变成一种虚假的相关性（长块天然更容易被命中）。

## 性能与边界

暴力全扫 O(n·d)。这个实现的上限大约在几千条 —— 对本项目当前的样本量
（50 页 PDF ≈ 十几块）绰绰有余。**不在这里加 HNSW**：为一个还没出现的规模问题
引入一个自己实现的近似索引，等于用一个不确定的正确性换一个不存在的性能收益。
到达几千条时，正确的动作是换 `chroma`，而不是优化这个类。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from app.rag.schema import Chunk
from app.rag.stores.base import SearchHit


class InMemoryVectorStore:
    """进程内向量库。重启即丢，适合测试与本地联调。"""

    name = "memory"

    def __init__(self) -> None:
        # dict 而不是 list：`upsert` 的语义就是按 id 覆盖，
        # 用 dict 让"覆盖"成为语言层面的事实，而不是一段手写查重逻辑。
        # **顺序**靠插入序（Python 3.7+ 保证），检索时同分的情况下顺序稳定，
        # 测试才不会因为"同分块随机排序"而间歇失败。
        self._rows: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> int:
        if len(chunks) != len(vectors):
            # 数量不匹配时**必须炸**，不能截断到较短的：
            # 静默截断会让"漏了一条"表现成"索引成功，但检索就是差一点"。
            raise ValueError(
                f"chunks({len(chunks)}) 与 vectors({len(vectors)}) 数量不一致，"
                "向量与块的按位对应关系已失效"
            )
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._rows[chunk.id] = (chunk, list(vector))
        return len(chunks)

    def search(self, vector: list[float], top_k: int = 5) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k 必须为正数")
        if not self._rows:
            return []

        scored: list[SearchHit] = []
        for chunk, stored in self._rows.values():
            scored.append(SearchHit(chunk=chunk, score=_cosine(vector, stored)))

        # `-score` 排序：同分时按插入序稳定输出，测试可复现。
        scored.sort(key=lambda hit: -hit.score)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._rows)

    def clear(self) -> None:
        self._rows.clear()

    # ------------------------------------------------------------------ 仅供测试

    def ids(self) -> list[str]:
        """导出当前所有 id（按插入序）。测试用它断言幂等入库 ——
        "重复 upsert 后条数不变、id 不变"这一条内容派生的核心性质，
        没有这个方法就只能靠 `count()` 间接推。"""
        return list(self._rows)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度。任一为零向量时返回 0.0（而不是抛异常）。

    零向量出现在"整页扫描件"这类空文本块上。它永远不该被命中，
    但也不该让一次检索整个失败 —— 返回 0.0 就是"排到最后"。
    """
    if len(a) != len(b):
        # 维度不一致仍然要炸：那是索引与查询用了不同 embedder 的**结构性错误**，
        # 报 0.0 会把"配错了"伪装成"没命中"。
        raise ValueError(f"向量维度不一致：查询 {len(a)} vs 库内 {len(b)}")
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


__all__ = ["InMemoryVectorStore"]
