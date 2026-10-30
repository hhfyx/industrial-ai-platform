"""向量库接口与返回结构。

**接口定成"四个方法"是分析出来的，不是拍脑袋**：索引要 `upsert` / `count`，
检索要 `search`，运维要 `clear`。再多一个 `delete_by_source` 就够用了 ——
但那是"支持增量删除"时才需要，现在没有调用方，不加。

> 这条规则值得单独说：**接口宁缺勿滥**。多出来的方法一旦存在，
> 就会有人用；用起来之后就成了"不能删的兼容责任"。
> 而现在多写一个 `delete_by_source` 的成本是零，将来删它的成本不是。

**`SearchHit` 为什么带 `chunk` 而不是只带 `id + score`**：
检索链路到生成链路之间需要正文与出处（页码/来源）。如果只回 `id`，
调用方还得回查一次才能拼上下文 —— 多一次往返，且那次往返的结果可能已经变了
（索引在跑增量）。让命中结果**自带完整快照**，生成层就不依赖第二次查询。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.rag.schema import Chunk


@dataclass(frozen=True, slots=True)
class SearchHit:
    """一条检索命中：块 + 相似度分数。

    `score` 的语义**由具体 store 决定**（余弦相似度 / 内积 / L2 距离）。
    上层只做**排序**，不做跨库的阈值比较 —— 因为不同 store 的分数不可比。
    这条限制写在类型旁边而不是文档里：想看分数做阈值判断的人一定会先读到这里。
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

        **必须是 upsert 而不是 insert**：`Chunk.id` 是内容派生的
        （见 `schema.py`），重复入库同一份文档是常态（重跑、增量、补索引）。
        如果这里语义是 insert，"重跑一次"就会让库里的条数翻倍，
        而检索结果看起来还挺正常 —— 这种坏法最难发现。
        """
        ...

    def search(self, vector: list[float], top_k: int = 5) -> list[SearchHit]:
        """按向量检索，返回按分数降序的前 `top_k` 条。库为空时返回空列表。"""
        ...

    def count(self) -> int:
        """当前条数。用于索引报告的"入库了多少块"。"""
        ...

    def clear(self) -> None:
        """清空。重建索引前必须能清干净 —— 否则旧向量会和新向量混在一起。"""
        ...
