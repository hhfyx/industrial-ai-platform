"""生成：把命中的块拼成上下文，调用模型，产出**带引用**的答案。

**这一层最容易做错的地方不是调模型，是"引用"**：

朴素做法是把 TopK 的正文拼起来喂给模型，答案是出来了，但用户问"这句是哪来的"，
你答不上来 —— 因为拼上下文的那一刻，`chunk` 和 `prompt 里的第几段` 的对应关系
只存在于一个局部列表里，模型返回的文本里没有任何东西指向它。

所以这里的做法是：**上下文里每一块都带编号，并要求模型在答案里用 `[n]` 标注**。
编号 → `Chunk` 的映射由 `Citation` 承载，随答案一起返回。这样"溯源"是一条
可核对的链路，而不是一句承诺：

    Chunk(source, page) → 编号 [n] → 模型输出里的 [n] → Citation → 前端显示 "合同.pdf 第 7 页"

**为什么不用"让模型自己说出文件名和页码"**：它会把页码说错，而且错得很自信。
编号是唯一由**我们**（而非模型）掌握的信息，让模型只负责"引用哪一条"，
不让它负责"那一条在哪个文件"。这是本项目一贯的分工（见 `llm_extractor` 的原文回指校验）。
"""

from __future__ import annotations

from app.rag.generation.base import (
    DEFAULT_CONTEXT_CHARS,
    Answer,
    Citation,
    build_context,
    generate_answer,
)

__all__ = [
    "DEFAULT_CONTEXT_CHARS",
    "Answer",
    "Citation",
    "build_context",
    "generate_answer",
]
