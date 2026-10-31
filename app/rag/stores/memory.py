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

from collections.abc import Sequence

from app.rag.embedding.vector import cosine
from app.rag.schema import Chunk, MetadataValue
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
            # **按 `store_id` 而不是 `id` 存**：与 ChromaStore 保持同一个覆盖语义
            # （位置派生 → 同位置的新内容覆盖旧内容）。两个实现用不同主键，
            # 会让"内存里测通过的幂等性"在 Chroma 上不成立 —— 那种不一致最难查。
            # `id`（内容派生）仍在 chunk 上，可用于去重与调试。
            self._rows[chunk.store_id] = (chunk, list(vector))
        return len(chunks)

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        *,
        where: dict[str, MetadataValue] | None = None,
    ) -> list[SearchHit]:
        """按向量检索，可选精确匹配过滤。"""
        if top_k <= 0:
            raise ValueError("top_k 必须为正数")
        if not self._rows:
            return []

        scored: list[SearchHit] = []
        for chunk, stored in self._rows.values():
            if where and not _matches(chunk, where):
                continue
            scored.append(SearchHit(chunk=chunk, score=_cosine(vector, stored)))

        # `-score` 排序：同分时按插入序稳定输出，测试可复现。
        scored.sort(key=lambda hit: -hit.score)
        return scored[:top_k]

    def search_in_pages(
        self,
        vector: list[float],
        *,
        page_from: int,
        page_to: int,
        top_k: int = 5,
        where: dict[str, MetadataValue] | None = None,
    ) -> list[SearchHit]:
        """按页码区间检索（含两端）。与 `ChromaStore` 的语义保持一致。"""
        if page_from > page_to:
            raise ValueError(f"page_from({page_from}) 不能大于 page_to({page_to})")
        if top_k <= 0:
            raise ValueError("top_k 必须为正数")
        if not self._rows:
            return []

        scored: list[SearchHit] = []
        for chunk, stored in self._rows.values():
            if chunk.page is None or not (page_from <= chunk.page <= page_to):
                continue
            if where and not _matches(chunk, where):
                continue
            scored.append(SearchHit(chunk=chunk, score=_cosine(vector, stored)))
        scored.sort(key=lambda hit: -hit.score)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._rows)

    def clear(self) -> None:
        self._rows.clear()

    def delete_by_source(self, source: str) -> int:
        """按 source 删除。**与 Chroma 的语义必须一致** —— 见 `base.VectorStore`。"""
        doomed = [key for key, (chunk, _) in self._rows.items() if chunk.source == source]
        for key in doomed:
            del self._rows[key]
        return len(doomed)

    def count_by_source(self, source: str) -> int:
        """该 source 已有多少块。追加时用来算 `index_offset`。"""
        return sum(1 for chunk, _ in self._rows.values() if chunk.source == source)

    # ------------------------------------------------------------------ 仅供测试

    def ids(self) -> list[str]:
        """导出当前所有存储主键（`store_id`，按插入序）。

        测试用它断言"重复 upsert 后条数不变、主键不变"——
        没有这个方法就只能靠 `count()` 间接推。
        """
        return list(self._rows)


def _matches(chunk: Chunk, where: dict[str, MetadataValue]) -> bool:
    """精确匹配过滤（与 Chroma 的 `where={"k": v}` 语义对齐）。

    查的字段范围与 Chroma 那边一致：`source` / `page` / `chunk_index` / `doc_id`
    都在 `to_payload()` 里，这里也按同样的字段解析 —— 两个实现的过滤结果必须相同，
    否则"内存里测通过、Chroma 上过滤不到"会变成一类无法复现的 bug。
    """
    for key, expected in where.items():
        if key == "source":
            actual: MetadataValue = chunk.source
        elif key == "page":
            actual = -1 if chunk.page is None else chunk.page
        elif key == "chunk_index":
            actual = chunk.index
        elif key == "doc_id":
            actual = chunk.doc_id
        elif key == "id":
            actual = chunk.id
        else:
            actual = chunk.metadata.get(key)  # type: ignore[assignment]
        if actual != expected:
            return False
    return True


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度。**实现已统一到 `embedding.vector.cosine`。**

    这里刻意不留自己的实现：两处各写一份的代价是"改了一处忘了另一处"，
    而相似度是排序的唯一依据 —— 两个 store 用不同口径算分，
    换成别的后端时"为什么结果变了"会查很久。

    维度不一致仍然要炸：那是索引与查询用了不同 embedder 的**结构性错误**，
    报 0.0 会把"配错了"伪装成"没命中"。
    """
    if len(a) != len(b):
        raise ValueError(f"向量维度不一致：查询 {len(a)} vs 库内 {len(b)}")
    if not a or not b:
        return 0.0
    return cosine(a, b)


__all__ = ["InMemoryVectorStore"]
