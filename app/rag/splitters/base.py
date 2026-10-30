"""切分器的接口与参数对象。

**为什么先定接口再写实现**：切分是最容易长出十几个变体的地方
（固定长度 / 递归 / 语义 / 按标题 / 按表格行…）。如果实现先写、接口后抽，
所有调用点就会长成"直接 new 那个具体类"。有了 Protocol，
上层 `indexing` 只认 `Splitter`，换策略是改一行装配。

**`ChunkSize` 为什么是独立对象而不是两个散参数**：
`chunk_chars` 与 `overlap_chars` 有硬约束（重叠必须小于块长，否则切分不收敛），
这条约束挂在 `__post_init__` 上，比散在两个参数里靠调用方自觉可靠得多。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.rag.schema import Chunk, Document


@dataclass(frozen=True, slots=True)
class ChunkSize:
    """块大小参数（**字符**为单位，不是 token）。"""

    chunk_chars: int = 500
    """单块目标字符数。500 字 ≈ 一段完整论述（中文 0.7 tok/字 ≈ 350 token）。"""

    overlap_chars: int = 50
    """相邻块的重叠字符数。必须 `< chunk_chars`，否则切分永远前进不了。"""

    def __post_init__(self) -> None:
        if self.chunk_chars <= 0:
            raise ValueError("chunk_chars 必须为正数")
        if self.overlap_chars < 0:
            raise ValueError("overlap_chars 不能为负")
        # **这条校验不是形式主义**：overlap >= chunk_chars 时步进为 0，
        # 循环要么死锁、要么被 `while` 的边界条件静默吞成一块 ——
        # 后者更糟，因为看起来"跑通了"，实际全文挤进一个块。
        if self.overlap_chars >= self.chunk_chars:
            raise ValueError(
                f"overlap_chars({self.overlap_chars}) 必须小于 chunk_chars({self.chunk_chars})"
            )


@runtime_checkable
class Splitter(Protocol):
    """切分器的唯一接口。

    `runtime_checkable` 让测试能 `isinstance(splitter, Splitter)` 直接验证装配结果，
    不用 `hasattr` 手搓 —— 和 `app/core/deps.py` 里 `Extractor` 的处理保持一致。
    """

    name: str
    """实现名，进日志与索引元数据（排查"这批向量是哪版策略切的"要用）。"""

    def split(self, documents: Sequence[Document]) -> list[Chunk]:
        """把一批 `Document` 切成 `Chunk` 列表。空输入返回空列表，不抛异常。"""
        ...
