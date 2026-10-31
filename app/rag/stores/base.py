"""向量库接口与返回结构。

**接口定成"几个方法"是分析出来的，不是拍脑袋**：

| 方法 | 谁需要 | 什么时候需要 |
| --- | --- | --- |
| `upsert` | 索引段 | 每次入库 |
| `search` | 检索段 | 每次提问 |
| `count` | 索引报告 | 每次入库后 |
| `clear` | 运维 | 重建索引前 |
| `delete_by_source` | 运维 / 增量更新 | **W4 周四新增**：重建某份文档前先删干净 |
| `search(where=...)` | 检索段 | **W4 周四新增**：按 source / 页码过滤 |

`delete_by_source` 是这周四才加的 —— 它原来被刻意不加（"没有调用方就不加，
接口宁缺勿滥"）。现在有了真实的调用方：**位置派生 id 的代价**就是
"改切分参数会静默覆盖"，所以重建前必须能按文档删干净（见 `schema.Chunk.store_id`）。
**接口是按需求长的，不是先画好再往里塞。**

**`SearchHit` 为什么带 `chunk` 而不是只带 `id + score`**：
检索链路到生成链路之间需要正文与出处（页码/来源）。如果只回 `id`，
调用方还得回查一次才能拼上下文 —— 多一次往返，且那次往返的结果可能已经变了
（索引在跑增量）。让命中结果**自带完整快照**，生成层就不依赖第二次查询。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.rag.schema import Chunk, MetadataValue


@dataclass(frozen=True, slots=True)
class SearchHit:
    """一条检索命中：块 + 相似度分数。

    **`score` 的语义被接口钉死为"余弦相似度"**（越大越像，范围 [-1, 1]）。

    这条在 W4 周四改了，原来是"由具体 store 决定"。改的理由很硬：
    Chroma 的 `query()` 返回的是 **distance**（cosine 距离），
    而 `distance = 1 - similarity` —— **越小越像，与相似度方向相反**。
    两种口径混在同一个类型里，调用方写阈值判断时必然有一半概率搞反。

    所以现在规定：**store 负责把自家口径换算成相似度再返回**。
    `ChromaStore` 里做 `1 - distance`，`InMemoryVectorStore` 本来就返回余弦。
    上层只认一种口径，不需要知道底下是谁。
    """

    chunk: Chunk
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """向量库的唯一接口。"""

    name: str
    """实现名，进日志与索引元数据。"""

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> int:
        """写入或覆盖。返回实际写入条数。

        **必须是 upsert 而不是 insert**：重复入库同一份文档是常态（重跑、增量、补索引）。
        如果这里语义是 insert，"重跑一次"就会让库里的条数翻倍，
        而检索结果看起来还挺正常 —— 这种坏法最难发现。

        **覆盖是按 `Chunk.store_id`（位置派生）而不是 `Chunk.id`（内容派生）**：
        理由与代价见 `schema.Chunk.store_id` 的 docstring。简言之：
        同一个位置的新内容应该**覆盖**旧内容，而不是并存两条。
        """
        ...

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        *,
        where: dict[str, MetadataValue] | None = None,
    ) -> list[SearchHit]:
        """按向量检索，返回**按相似度降序**的前 `top_k` 条。库为空时返回空列表。

        `where` 是**精确匹配**的元数据过滤（`{"source": "合同.pdf"}`）。
        需要范围查询（按页码区间）时用 `page_range` 参数。

        **`score` 必须是余弦相似度**（越大越像）—— 见 `SearchHit` 的说明。
        """
        ...

    def search_in_pages(
        self,
        vector: list[float],
        *,
        page_from: int,
        page_to: int,
        top_k: int = 5,
        where: dict[str, MetadataValue] | None = None,
    ) -> list[SearchHit]:
        """按页码区间检索（含两端）。

        **为什么不塞进 `search` 的 kwargs**：页码范围是 `$and` + 两个比较，
        与精确匹配的写法完全不同（Chroma 的 `where` 里每个操作符只能出现一次，
        实测 `{"page": {"$gte": 1, "$lte": 1}}` 会直接 `ValueError`）。
        两种语义混在一个参数里，"传了 but 没生效"会变成沉默故障。
        """
        ...

    def count(self) -> int:
        """当前条数。用于索引报告的"入库了多少块"。"""
        ...

    def clear(self) -> None:
        """清空。重建索引前必须能清干净 —— 否则旧向量会和新向量混在一起。"""
        ...

    def delete_by_source(self, source: str) -> int:
        """按 `source` 删除该文档的全部块。返回删除条数。

        **重建单份文档的标准流程**：先 `delete_by_source` 再 `upsert`。
        不删的话，位置派生 id 会把"旧的第 3 块"静默覆盖成"新的第 3 块"，
        而**多出来的块**（旧文档 10 块、新文档 8 块）会留在库里 ——
        那两条孤儿会一直被检索到，且内容来自旧版本。

        这是 `upsert` 幂等性的**盲区**：幂等只保证"同样的输入不重复"，
        不保证"输入变少时自动收缩"。
        """
        ...

    def count_by_source(self, source: str) -> int:
        """该 source 已有多少块。

        **为什么需要它**：`Chunk.store_id` 是 `{doc_id}_{page}_{chunk_index}`，
        而 `index` 在每次 `index_documents()` 里都**从 0 开始**。
        对同一 source 连续追加时，每批的第一块都是 `_0`，于是后一批覆盖前一批 ——
        实测连续追加 3 条不同文本，库里只剩 1 条，而每批的 `IndexReport.ok` 都是 True。

        有了这个计数，追加方就能算出 `index_offset`，让新块的 id 接在已有块后面。
        """
        ...
