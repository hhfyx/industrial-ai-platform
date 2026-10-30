"""map-reduce 文档摘要：超长文档分而治之。

**为什么超长文档不能"直接塞给模型"：**
- 上下文窗口有上限（一份 50 页报告通常超过很多模型的窗口）
- 即使塞得进去，**注意力和成本都随长度恶化**：越长的 prompt，中间部分越容易被忽略
  （"lost in the middle"），而成本是线性的

所以用经典的两阶段：

    map    每段各自摘要（段足够小时，模型注意力集中）
    reduce 把各段摘要汇总成总摘要（此时输入已经是摘要，不再是原文）

**为什么 `reduce` 也要分批（`_hierarchical_reduce`）：**
50 页 → 按 3000 token 分段 → 约 15 段。如果 15 段摘要一次性喂给模型，
又回到了"输入太长"的老问题。所以 reduce 阶段同样按预算分批，
每批先局部汇总，最后再汇总一次 —— 这就是"段数 > 阈值时先聚类再汇总"。

**这里的"聚类"是刻意简化的**：不真的做向量聚类（那需要 embedding 模型，
是 W4 的事）。这里按**顺序分批**，因为文档的语义通常沿章节渐进 ——
相邻段落谈同一主题的概率远高于随机两段。够用，且不引入额外依赖。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Final

from app.rag import Document

logger = logging.getLogger(__name__)

# 单段目标 token 数。取 3000 是因为：留出空间给系统提示词和模型输出后，
# 仍在多数模型的有效注意力范围内；太小则段数爆炸（每次 map 都是一次计费调用）。
DEFAULT_CHUNK_TOKENS: Final[int] = 3000

# 每字符 token 数（中文偏高的保守估计，见 learn/llm/03_cost_and_reliability.md：
# 中文约 0.68 tok/字）。这里取 0.7 是**保守上界** —— 宁可切得碎一点，
# 也不要在"看起来刚好"的时候超出模型窗口。
CHARS_PER_TOKEN: Final[float] = 0.7

# 段数超过这个值就启用"先分批再汇总"（hierarchical reduce）
HIERARCHICAL_THRESHOLD: Final[int] = 10

# 每批喂给 reduce 的最大段数
REDUCE_BATCH_SIZE: Final[int] = 5

SYSTEM_PROMPT: Final[str] = (
    "你是一个严谨的文档摘要器。规则：\n"
    "1. 只依据给定内容总结，不要补充任何原文没有的信息。\n"
    "2. 保留关键数字、日期、金额、责任主体等具体信息，不要只说'有相关条款'。\n"
    "3. 用简洁的中文，不要用 Markdown 标题层级。"
)

CHUNK_PROMPT: Final[str] = (
    "下面是文档的第 {index}/{total} 部分（{location}）。"
    "请用 2-4 句话提炼这部分的核心内容：讲了什么、涉及哪些具体条款或数据。\n\n{text}"
)

REDUCE_PROMPT: Final[str] = (
    "下面是一份文档各部分的摘要。请把它们汇总成一段连贯的总摘要，"
    "覆盖全文要点，不要逐条罗列，也不要编造原文没有的内容。\n\n{summaries}"
)


@dataclass(frozen=True, slots=True)
class SummarySegment:
    """一个**待摘要**的分段。

    保留 `pages`（覆盖的页码范围）：摘要里要能回答"这部分对应原文哪里"，
    否则用户没法核对。

    **为什么从 `Chunk` 改名为 `SummarySegment`**：W4 引入 RAG 后，
    `Chunk` 这个名字被 [`app.rag.schema.Chunk`](schema.py) 占用了 ——
    那是要落库、要有稳定主键和可过滤元数据的"索引单位"。
    两者形状相似（一段文字 + 出处），但**生命周期完全不同**：

    | | `SummarySegment` | `schema.Chunk` |
    | --- | --- | --- |
    | 生命周期 | 单次请求内，随摘要结束而消失 | 落库，跨请求、跨进程存活 |
    | 有无 id | 无 | **必须有**（幂等入库靠它） |
    | 分页表达 | `pages` 元组（只用于展示"第几页"） | `page` + `pages`（`page` 给引用，`pages` 给覆盖范围） |
    | 谁消费 | map-reduce 摘要 | embedding / 向量库 / 生成层 |

    名字不分开的代价是具体的：想给"检索块"加个 `score` 字段时，
    会发现自己在改摘要链路的类型 —— 而摘要根本不需要 `score`。
    """

    index: int
    text: str
    pages: tuple[int, ...]

    @property
    def location(self) -> str:
        if not self.pages:
            return "未知位置"
        if len(self.pages) == 1:
            return f"第 {self.pages[0]} 页"
        return f"第 {self.pages[0]}–{self.pages[-1]} 页"


@dataclass
class MapReduceResult:
    """摘要结果 + 过程统计（用于 CLI 进度与延迟分析）。"""

    summary: str
    chunk_summaries: list[str] = field(default_factory=list)
    chunk_count: int = 0
    reduce_rounds: int = 0
    hierarchical: bool = False
    api_calls: int = 0


def segment_documents(
    documents: Sequence[Document],
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
) -> list[SummarySegment]:
    """把文档按 token 预算切成段。

    **关键取舍：尽量不切断段落。**

    朴素做法是"按字符数硬切"，但那会把一句话劈成两半 ——
    两半各自摘要都会得到不完整的语义，这是摘要质量最大的杀手。
    这里的策略是**贪心装箱**：按 Document 逐个累加，超过预算就开新段。
    只有单个 Document 本身就超预算时才硬切（那种情况通常是整篇塞在一页里）。

    `page` 信息跟着段走，用于展示"这部分对应原文第几页"。

    **`chunk_tokens` 参数名保留不改**（即使函数已叫 `segment_documents`）：
    它是 API 层 `/summarize/stream` 已经暴露给前端的字段名（`app/api/v1/summarize.py`），
    改它等于改 HTTP 契约。**内部命名可以迁移，对外契约不该跟着动** ——
    这条边界一旦模糊，"顺手重命名"就会变成一次前端回归。
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens 必须为正数")

    # 每段至少要有这么多字符，否则"一句话一段"没有意义（每段都要一次计费调用）。
    # **但不能硬编成 200**：那会让原本很小的预算（如 chunk_tokens=30）被静默抬高，
    # 调用方以为自己在用小预算，实际拿到的是 200 字/段 ——
    # 这种"参数不生效"比报错更难查。所以改成 1/CHARS_PER_TOKEN 的自然下限，
    # 同时仍挡掉"1 token 一段"这种无意义配置。
    min_chars = max(int(10 / CHARS_PER_TOKEN), 10)
    budget_chars = max(int(chunk_tokens / CHARS_PER_TOKEN), min_chars)
    chunks: list[SummarySegment] = []
    buffer: list[str] = []
    pages: list[int] = []
    used = 0
    index = 0

    def flush() -> None:
        nonlocal buffer, pages, used, index
        if not buffer:
            return
        chunks.append(
            SummarySegment(index=index, text="\n\n".join(buffer), pages=tuple(sorted(set(pages))))
        )
        index += 1
        buffer, pages, used = [], [], 0

    for doc in documents:
        text = doc.text.strip()
        if not text:
            continue

        # 单个 Document 就超预算：硬切（无法避免，但要尽量在句子边界切）
        if len(text) > budget_chars:
            flush()
            for piece in _split_long_text(text, budget_chars):
                chunks.append(
                    SummarySegment(
                        index=index,
                        text=piece,
                        pages=(doc.page,) if doc.page is not None else (),
                    )
                )
                index += 1
            continue

        # 再加就超预算 → 先收束当前段
        if used + len(text) > budget_chars:
            flush()

        buffer.append(text)
        if doc.page is not None:
            pages.append(doc.page)
        used += len(text)

    flush()
    return chunks


def _split_long_text(text: str, budget_chars: int) -> list[str]:
    """把超长文本切开，**优先在句子边界切**。

    找 `budget_chars` 附近最近的句末标点（前后各看 20%），
    找不到才硬切。这个"往回找一点"的动作很便宜，但能显著减少"半句话"。
    """
    pieces: list[str] = []
    remaining = text
    while len(remaining) > budget_chars:
        window = remaining[:budget_chars]
        # 从后往前找句末标点，最多回退 20% 的预算
        floor = int(budget_chars * 0.8)
        cut = -1
        for i in range(len(window) - 1, floor, -1):
            if window[i] in "。！？.!?\n":
                cut = i + 1
                break
        if cut <= 0:
            cut = budget_chars
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    if remaining.strip():
        pieces.append(remaining.strip())
    return pieces


async def _call_model(
    client: object,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 400,
    extra: dict[str, object] | None = None,
) -> str:
    """调用模型拿一段文本（非流式）。

    **为什么用 `object` 标注 client**：这里只需要 `.chat.completions.create`，
    标成具体类型会让测试必须造一个真的 AsyncOpenAI。鸭子类型在这里是对的。
    """
    response = await client.chat.completions.create(  # type: ignore[attr-defined]
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        **(extra or {}),
    )
    return (response.choices[0].message.content or "").strip()


async def summarize_document(
    documents: Sequence[Document],
    *,
    client: object,
    model: str,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    extra_request: dict[str, object] | None = None,
    on_progress: callable | None = None,  # type: ignore[valid-type]
) -> MapReduceResult:
    """map-reduce 摘要主入口。

    `on_progress(done, total, label)` 在每段完成时回调，用于 CLI 进度显示 ——
    **没有进度显示的超长文档摘要体验是灾难**：用户不知道卡住了还是在跑，
    50 页文档可能要几十秒。
    """
    chunks = segment_documents(documents, chunk_tokens=chunk_tokens)
    if not chunks:
        return MapReduceResult(summary="（文档没有可摘要的文本）")

    api_calls = 0
    chunk_summaries: list[str] = []

    # ---- map：逐段摘要 ----
    for i, segment in enumerate(chunks, start=1):
        prompt = CHUNK_PROMPT.format(
            index=i, total=len(chunks), location=segment.location, text=segment.text
        )
        try:
            piece = await _call_model(client, model, prompt, extra=extra_request)
            api_calls += 1
        except Exception as exc:  # noqa: BLE001 - 单段失败不该毁掉整篇
            # 记下失败而不是整体失败：50 页里坏 1 段，剩下 49 段的摘要仍然有价值
            logger.warning("第 %d/%d 段摘要失败：%s: %s", i, len(chunks), type(exc).__name__, exc)
            piece = f"（第 {i} 部分摘要失败）"
        if not piece:
            piece = f"（第 {i} 部分摘要为空）"
        chunk_summaries.append(piece)
        if on_progress is not None:
            on_progress(i, len(chunks), segment.location)

    # ---- reduce：汇总 ----
    hierarchical = len(chunk_summaries) > HIERARCHICAL_THRESHOLD
    reduce_rounds = 0

    if not hierarchical:
        # 段数不多：一次汇总
        joined = _format_summaries(chunk_summaries)
        summary = await _call_model(
            client,
            model,
            REDUCE_PROMPT.format(summaries=joined),
            max_tokens=600,
            extra=extra_request,
        )
        api_calls += 1
        reduce_rounds = 1
    else:
        # 段数 > 阈值：先分批各汇总一次，再把中间摘要汇总一次
        levels = list(chunk_summaries)
        while len(levels) > 1:
            batches = [
                levels[i : i + REDUCE_BATCH_SIZE] for i in range(0, len(levels), REDUCE_BATCH_SIZE)
            ]
            next_level: list[str] = []
            for batch in batches:
                if len(batch) == 1 and len(levels) <= REDUCE_BATCH_SIZE:
                    next_level.extend(batch)
                    continue
                joined = _format_summaries(batch)
                merged = await _call_model(
                    client,
                    model,
                    REDUCE_PROMPT.format(summaries=joined),
                    max_tokens=600,
                    extra=extra_request,
                )
                api_calls += 1
                reduce_rounds += 1
                next_level.append(merged)
            if len(next_level) == len(levels):
                # 没有收敛（全是单元素批次），直接合并文本兜底
                break
            levels = next_level

        summary = "\n\n".join(levels)
        if len(levels) > 1:
            summary = await _call_model(
                client,
                model,
                REDUCE_PROMPT.format(summaries=_format_summaries(levels)),
                max_tokens=600,
                extra=extra_request,
            )
            api_calls += 1
            reduce_rounds += 1

    return MapReduceResult(
        summary=summary,
        chunk_summaries=chunk_summaries,
        chunk_count=len(chunks),
        reduce_rounds=reduce_rounds,
        hierarchical=hierarchical,
        api_calls=api_calls,
    )


def _format_summaries(summaries: Sequence[str]) -> str:
    """给各段摘要编号 —— 编号让模型知道"这是第几部分"，汇总时更容易保序。"""
    return "\n".join(f"[{i}] {s}" for i, s in enumerate(summaries, start=1))


async def summarize_streaming(
    documents: Sequence[Document],
    *,
    client: object,
    model: str,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    extra_request: dict[str, object] | None = None,
) -> AsyncIterator[tuple[str, dict[str, object]]]:
    """流式版：先产出进度事件，最后产出摘要 token。

    产出的是 `(type, payload)` 二元组：
    - `("progress", {done, total, label})`  —— 分段进度
    - `("delta", {text})`                   —— 摘要正文增量
    - `("done", {stats})`                   —— 结束

    **为什么把进度和正文放在同一个流里**：用户上传 50 页文档后，
    最长的等待是 map 阶段（十几段摘要），那期间一个字节都没有。
    如果不把进度也放进流里，前端就只能干等 —— 那是极差的体验。
    """
    from app.core.llm.streaming import stream_chat

    chunks = segment_documents(documents, chunk_tokens=chunk_tokens)
    if not chunks:
        yield ("done", {"summary": "（文档没有可摘要的文本）", "chunks": 0})
        return

    total = len(chunks)
    chunk_summaries: list[str] = []
    queue: asyncio.Queue[tuple[int, str, str]] = asyncio.Queue()
    api_calls = 0

    async def run_map(index: int, segment: SummarySegment) -> None:
        prompt = CHUNK_PROMPT.format(
            index=index + 1, total=total, location=segment.location, text=segment.text
        )
        try:
            piece = await _call_model(client, model, prompt, extra=extra_request)
        except Exception as exc:  # noqa: BLE001
            logger.warning("第 %d 段失败：%s", index + 1, type(exc).__name__)
            piece = f"（第 {index + 1} 部分摘要失败）"
        await queue.put((index, piece, segment.location))

    # 并发跑 map，但**限制并发数**：同时打多个 LLM 请求容易触发 429，
    # 而且收益有限（每个请求本身就是长任务）。4 路是经验值。
    semaphore = asyncio.Semaphore(4)

    async def guarded(index: int, segment: SummarySegment) -> None:
        async with semaphore:
            await run_map(index, segment)

    tasks = [asyncio.create_task(guarded(i, c)) for i, c in enumerate(chunks)]
    results: dict[int, str] = {}

    for done in range(1, total + 1):
        index, piece, location = await queue.get()
        results[index] = piece
        yield ("progress", {"done": done, "total": total, "label": location})

    await asyncio.gather(*tasks)
    chunk_summaries = [results.get(i, "") for i in range(total)]
    api_calls += total

    # reduce 阶段：流式输出最终摘要 —— **这才是用户真正要看的逐字效果**
    joined = _format_summaries(chunk_summaries)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": REDUCE_PROMPT.format(summaries=joined)},
    ]
    parts: list[str] = []
    async for delta in stream_chat(
        client,  # type: ignore[arg-type]
        model=model,
        messages=messages,
        max_tokens=600,
        extra_request=extra_request,
    ):
        parts.append(delta)
        yield ("delta", {"text": delta})

    yield (
        "done",
        {
            "summary": "".join(parts),
            "chunks": total,
            "chunk_summaries": chunk_summaries,
            "api_calls": api_calls + 1,
        },
    )


__all__ = [
    "DEFAULT_CHUNK_TOKENS",
    "HIERARCHICAL_THRESHOLD",
    "SummarySegment",
    "MapReduceResult",
    "segment_documents",
    "summarize_document",
    "summarize_streaming",
]
