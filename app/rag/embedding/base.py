"""Embedder 接口。

**接口只有两个方法，是刻意的**：

    dimension        向量维度（建集合时要用，事后改等于重建索引）
    embed(texts)     批量向量化

没有 `embed_one`：单条调用在接口层出现，调用方就会一条一条地调，
本地模型的批处理收益直接归零。想 embed 一条就传一个元素的列表 ——
让"批量"成为唯一的路径。

没有 `async`：理由见 `embedding/__init__.py` 的 docstring。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """向量化器的唯一接口。"""

    name: str
    """实现名。会写进索引元数据 —— 换模型后要能一眼看出"这批向量是谁产的"。"""

    dimension: int
    """向量维度。**入库前必须核对**：维度不匹配的库在检索时会静默返回错结果。"""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """批量向量化。返回顺序与入参严格一致，长度必须相同。"""
        ...
