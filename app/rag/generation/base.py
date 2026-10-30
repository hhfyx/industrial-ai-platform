"""生成层：上下文拼装 + 一次模型调用 + 引用回填。

**为什么把 `build_context` 单独抽出来**：它是**纯函数**，不碰模型、不碰网络。
这意味着上下文格式（编号怎么写、块之间怎么分、超长怎么截）可以被单独断言，
而不是"跑一次模型看输出像不像"。RAG 的很多质量问题其实出在拼上下文这一步，
不是模型不行；能单独测它，才能分辨是哪一个。

**为什么超长要在这里截断，而不是信任 top_k**：top_k 是"返回几条"，
不是"多少字符"。500 字一块 × top_k=5 = 2500 字，接近上下文窗口安全线的三分之一；
换成 1000 字一块的切分参数，同样的 top_k 就超了。**截断按字符预算做，
不按条数做** —— 条数是个会随切分参数漂移的量，字符预算才是稳定的约束。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

from app.rag.schema import Chunk

logger = logging.getLogger(__name__)

# 拼进 prompt 的上下文字符上限。取 6000 的理由：
# 中文 0.7 tok/字 ≈ 4200 token，给系统提示词与输出留出足够空间后仍安全。
# 这个数字必须**小于**任何目标模型的窗口减去输出预算 —— 宁可少放两块，
# 不要在"看起来刚好"的时候被网关静默截断（那会让你以为模型读到了）。
DEFAULT_CONTEXT_CHARS: Final[int] = 6000

SYSTEM_PROMPT: Final[str] = (
    "你是一个严谨的文档问答助手。规则：\n"
    "1. 只依据下面提供的资料回答，不要补充资料里没有的信息。\n"
    "2. 每一条结论后面用 [编号] 标注依据，例如 [1]、[2][3]。\n"
    "3. 资料里找不到答案时，直接说「资料中没有提到」，不要猜测。\n"
    "4. 保留原文中的数字、日期、金额、责任主体等具体信息。"
)

ANSWER_PROMPT: Final[str] = (
    "资料：\n{context}\n\n问题：{question}\n\n请依据资料回答，并在每条结论后标注 [编号]。"
)


@dataclass(frozen=True, slots=True)
class Citation:
    """一条可核对的引用：编号 ↔ 原文位置。"""

    index: int
    """从 1 开始的编号，与 prompt 里的 `[n]` 一一对应。"""

    source: str
    """来源文件名。"""

    page: int | None
    """页码；非分页格式为 None。**这个字段就是"page 从第一步就要在结构里"的回报**。"""

    chunk_id: str
    """对应块的稳定 id。用于"同一条引用被重复命中"的去重，也用于回查原文。"""

    @property
    def label(self) -> str:
        """人类可读的引用标签，给前端直接显示。"""
        return f"{self.source} 第 {self.page} 页" if self.page is not None else self.source


@dataclass
class Answer:
    """一次问答的完整产物：答案 + 引用 + 过程统计。"""

    text: str
    citations: list[Citation] = field(default_factory=list)
    context_chars: int = 0
    """实际拼进 prompt 的上下文字符数。用于回答"这次为什么没命中第 4 条"这类问题。"""

    used_chunks: int = 0
    """真正被拼进上下文的块数（可能少于检索返回的块数，因为按预算截断了）。"""

    def citation_block(self) -> str:
        """渲染成可直接显示的引用列表。没有引用时返回空串。"""
        if not self.citations:
            return ""
        return "\n".join(f"[{c.index}] {c.label}" for c in self.citations)


def build_context(
    chunks: Sequence[Chunk],
    *,
    max_chars: int = DEFAULT_CONTEXT_CHARS,
) -> tuple[str, list[Citation]]:
    """把块拼成带编号的上下文，返回 `(上下文文本, 引用列表)`。

    编号从 1 开始按传入顺序分配。**顺序即检索分数序** —— 检索层已经排好，
    这里不再重排：重排会让"第 1 条最相关"这个直觉失效，而用户读引用时正是靠这个直觉。

    超预算时**整块丢弃而不是截断块**：截断一半的块会让模型读到残句，
    然后照着残句给出一个看起来有依据的错答案 —— 比少给一块糟得多。
    """
    if max_chars <= 0:
        raise ValueError("max_chars 必须为正数")

    parts: list[str] = []
    citations: list[Citation] = []
    used = 0
    for chunk in chunks:
        # 编号用 len(citations)+1 而不是 enumerate：被预算挡掉的块不占编号，
        # 否则答案里会出现 [3] 但引用列表里没有 3 —— 前端只能显示一个断链接。
        index = len(citations) + 1
        block = f"[{index}] 来源：{chunk.location}\n{chunk.text}"
        if used + len(block) > max_chars:
            if not citations:
                # 一条都没放进去就超预算：说明单块就比预算大。
                # 这时**必须放一条**，否则模型收到空资料 —— 那必然导致编造。
                parts.append(block[:max_chars])
                citations.append(_to_citation(index, chunk))
                used = max_chars
            break
        parts.append(block)
        citations.append(_to_citation(index, chunk))
        used += len(block)

    return "\n\n".join(parts), citations


def _to_citation(index: int, chunk: Chunk) -> Citation:
    return Citation(
        index=index,
        source=chunk.source,
        page=chunk.page,
        chunk_id=chunk.id,
    )


async def generate_answer(
    question: str,
    chunks: Sequence[Chunk],
    *,
    client: object,
    model: str,
    max_context_chars: int = DEFAULT_CONTEXT_CHARS,
    max_tokens: int = 800,
    extra_request: dict[str, object] | None = None,
) -> Answer:
    """拼上下文 → 调模型 → 回填引用。

    **没有命中时直接返回固定话术，不调模型**：让模型对着空上下文回答，
    它唯一能做的就是编。省这一次调用不是目的，避免一次必然的幻觉才是。

    `client` 标成 `object` 与 `summarize._call_model` 一致：这里只需要
    `.chat.completions.create`，鸭子类型是对的，也便于测试注入假 client。
    """
    context, citations = build_context(chunks, max_chars=max_context_chars)
    if not citations:
        return Answer(
            text="资料中没有提到相关内容。",
            citations=[],
            context_chars=0,
            used_chunks=0,
        )

    prompt = ANSWER_PROMPT.format(context=context, question=question)
    response = await client.chat.completions.create(  # type: ignore[attr-defined]
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        **(extra_request or {}),
    )
    text = (response.choices[0].message.content or "").strip()

    return Answer(
        text=text,
        citations=citations,
        context_chars=len(context),
        used_chunks=len(citations),
    )


__all__ = [
    "ANSWER_PROMPT",
    "DEFAULT_CONTEXT_CHARS",
    "SYSTEM_PROMPT",
    "Answer",
    "Citation",
    "build_context",
    "generate_answer",
]
