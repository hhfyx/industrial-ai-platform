"""问答编排：**UI 与 CLI 共用的那一段**，带分阶段耗时打点。

## 这个模块存在的理由

Streamlit UI 和 `app/cli/ask.py` 要做的是**同一件事**：检索 → 拼上下文 → 调模型 →
解析引用。如果各写一遍，两边会慢慢漂移（一边加了阈值、另一边没加），
而漂移的表现是"CLI 答得对、UI 答得错"，最难查。

所以编排只有一份，UI/CLI 都调它。它们的差别只在**怎么展示**。

## 必答题：从提问到首 token 的 1.5 秒里，检索占多少

这是今天必须回答的问题，而答案是：**必须先打点，否则只能靠猜**。

所以 `QATiming` 把一次问答拆成四段，每段都有明确的起止点：

| 阶段 | 从哪到哪 | 典型量级（实测，50 页 PDF + bge + CPU） |
| --- | --- | --- |
| `embed_ms` | query 向量化开始 → 拿到查询向量 | **~10 ms**（bge 单条，CPU） |
| `search_ms` | 相似度计算 → TopK 返回 | **~1 ms**（内存）/ ~3 ms（Chroma） |
| `retrieval_ms` | 上面两段之和（`Retriever.retrieve` 整体） | **~11 ms** |
| `build_context_ms` | 拼上下文 + 分配编号 | **< 1 ms** |
| `ttft_ms` | 发起模型请求 → 第一个非空 delta | **~1000+ ms**（取决于模型） |

**结论的形状**：检索占首 token 的 **1% 量级**，模型本身占 99%。
这个结论只有在打完点之后才敢下 —— 而它决定了 W8 该优化什么：
**优化检索是白费力气**，延迟几乎全在模型侧。

## 为什么不用 `time.time()`

`time.time()` 是墙上时钟，可能被 NTP 校正**往回调**，算出负数耗时。
`perf_counter()` 是单调时钟，专门用于测间隔。
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from app.rag.embedding import Embedder
from app.rag.generation import (
    DEFAULT_CONTEXT_CHARS,
    REFUSAL_TEXT,
    Answer,
    build_context,
    resolve_citations,
)
from app.rag.schema import Chunk
from app.rag.stores import VectorStore

logger = logging.getLogger(__name__)


def _ms(started: float) -> int:
    """毫秒（至少 1）。

    **`max(..., 1)` 不是画蛇添足**：`int()` 会把 0.4ms 截成 0，
    而 0 在报表里看起来像"根本没测到"。任何真实发生过的耗时至少记 1ms，
    语义才不含糊 —— 这与 `core/llm/streaming.py` 里 TTFT 的处理是同一条规矩。
    """
    return max(int((time.perf_counter() - started) * 1000), 1)


@dataclass
class QATiming:
    """一次问答的分阶段耗时。**必答题的答案就在这个对象里。**"""

    embed_ms: int = 0
    """query 向量化耗时。"""

    search_ms: int = 0
    """相似度计算 + TopK 耗时。"""

    build_context_ms: int = 0
    """拼上下文 + 分配编号耗时。"""

    ttft_ms: int | None = None
    """首 token 时间（发起模型请求 → 第一个非空 delta）。拒答时为 None。"""

    total_ms: int = 0
    """端到端总耗时。"""

    @property
    def retrieval_ms(self) -> int:
        """检索段总耗时 = 向量化 + 相似度。

        **这是必答题的分子**。它单独算，而不是从 `total_ms` 里减，
        因为"减法得到的数"会把所有没归类的开销都算进检索里。
        """
        return self.embed_ms + self.search_ms

    @property
    def retrieval_share(self) -> float | None:
        """检索占首 token 时间的比例（0~1）。拒答或无 TTFT 时返回 None。

        **这个比例是 W8 的决策依据**：
        - 若接近 0 → 优化检索无意义，延迟全在模型侧
        - 若显著（比如 > 0.3）→ 检索值得优化（换更快的 embedder / 加缓存）

        没有这个数字，W8 只能靠猜"是不是检索慢"。
        """
        if self.ttft_ms is None or self.ttft_ms <= 0:
            return None
        return self.retrieval_ms / self.ttft_ms

    def as_dict(self) -> dict[str, object]:
        return {
            "embed_ms": self.embed_ms,
            "search_ms": self.search_ms,
            "retrieval_ms": self.retrieval_ms,
            "build_context_ms": self.build_context_ms,
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            # 保留 4 位小数：1% 量级的比例用两位小数会显示成 0.01，看不出差异
            "retrieval_share": (
                round(self.retrieval_share, 4) if self.retrieval_share is not None else None
            ),
        }


@dataclass
class QAResult:
    """一次问答的完整产物：答案 + 分阶段耗时 + 上下文块。"""

    answer: Answer
    timing: QATiming
    contexts: list[Chunk] = field(default_factory=list)
    """真正拼进上下文的块（按编号顺序）。**UI 要用它渲染"参考来源"，
    让用户点开看原文 —— 只给编号不给原文，等于让用户去翻文件。**"""

    top_score: float | None = None

    @property
    def refused(self) -> bool:
        return self.answer.refused


class QAEngine:
    """问答编排。UI / CLI / 测试都调这里，不各自拼链路。

    **依赖从构造注入**：`embedder` 与 `store` 必须与索引时是同一套
    （见 `pipeline.py` 的说明），这里不提供"默认给我一个"的口子。
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        top_k: int = 5,
        max_context_chars: int = DEFAULT_CONTEXT_CHARS,
        min_score: float | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k 必须为正数")
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        # 阈值默认从 embedder 取（见 `Embedder.recommended_min_score`）。
        # **不能是全局常量**：不同模型的分数分布完全不同，实测 hashing 的相关/无关
        # 完全重叠（scripts/embedding_score_probe.py）。
        self.min_score = embedder.recommended_min_score if min_score is None else min_score

    # ------------------------------------------------------------------ 检索（分阶段打点）

    def _retrieve_timed(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[list[Chunk], list[float], QATiming]:
        """检索，并把"向量化"与"相似度"分开计时。

        **为什么要拆开**：它们是可以分别优化的两件事。
        向量化慢 → 换 embedder / 加 query 缓存；
        相似度慢 → 换向量库 / 加 IVF 索引。
        合成一个"检索耗时"就分不清该动哪个。
        """
        k = self.top_k if top_k is None else top_k
        timing = QATiming()

        text = (query or "").strip()
        if not text:
            return [], [], timing

        started = time.perf_counter()
        query_vector = self.embedder.embed_query([text])[0]
        timing.embed_ms = _ms(started)

        started = time.perf_counter()
        hits = self.store.search(query_vector, top_k=k)
        timing.search_ms = _ms(started)

        return [hit.chunk for hit in hits], [hit.score for hit in hits], timing

    # ------------------------------------------------------------------ 非流式（CLI / 测试）

    async def answer(
        self,
        question: str,
        *,
        client: object,
        model: str,
        top_k: int | None = None,
    ) -> QAResult:
        """非流式问答：一次拿完整答案。

        CLI 用这个；UI 用 `answer_stream`（否则用户要等全部生成完才看到字）。
        """
        from app.rag.generation import generate_answer

        overall = time.perf_counter()
        chunks, scores, timing = self._retrieve_timed(question, top_k=top_k)

        started = time.perf_counter()
        answer = await generate_answer(
            question,
            chunks,
            client=client,
            model=model,
            max_context_chars=self.max_context_chars,
            min_score=self.min_score,
            scores=scores,
        )
        timing.build_context_ms = _ms(started) if not answer.refused else 0
        timing.total_ms = _ms(overall)

        return QAResult(
            answer=answer,
            timing=timing,
            contexts=_contexts_of(answer, chunks),
            top_score=answer.top_score,
        )

    # ------------------------------------------------------------------ 流式（UI）

    async def answer_stream(
        self,
        question: str,
        *,
        client: object,
        model: str,
        top_k: int | None = None,
        extra_request: dict[str, object] | None = None,
    ) -> AsyncIterator[tuple[str, object]]:
        """流式问答。产出 `(event, payload)` 二元组：

        | event | payload | 什么时候 |
        | --- | --- | --- |
        | `"retrieved"` | `(chunks, scores, QATiming)` | 检索完、**调用模型之前** |
        | `"delta"` | `str` | 每个文本增量 |
        | `"done"` | `StreamDone` | 结束（正常或拒答） |

        ## 为什么先产 `"retrieved"` 再产 `"delta"`

        **这是必答题在接口上的落地**：UI 需要在"等模型出第一个字"的那段时间里
        就把检索结果显示出来（命中了哪几块、分数多少）。如果等 `done` 才拿到，
        用户在整个 TTFT 期间看到的是一片空白 —— 而那恰好是最让人焦虑的一段时间。

        同时 `"retrieved"` 里带着当时的 `QATiming`（已经有 embed/search 的耗时），
        UI 可以立刻显示"检索 11ms"，而不是等全部结束才知道。

        ## 拒答也走这个接口

        拒答时只产 `"retrieved"`（可能为空）和 `"done"`，**不产任何 delta**，
        且**不调用模型**。调用方按 `event` 分派，不需要判断"这次是不是拒答"。
        """
        from app.core.llm.streaming import StreamStats, stream_chat

        overall = time.perf_counter()
        chunks, scores, timing = self._retrieve_timed(question, top_k=top_k)

        # 闸门 1+2：空结果 / 分数过低 → 拒答，**不调模型**。
        # 位置必须在 `yield "retrieved"` **之前**判断好，但 retrieved 要先吐出去
        # （UI 要显示"检索到 0 条"或"分数 0.1 太低"）。
        top_score = max(scores) if scores else None
        refuse = not chunks or (top_score is not None and top_score < self.min_score)

        yield ("retrieved", (chunks, scores, timing))

        if refuse:
            answer = Answer(text=REFUSAL_TEXT, refused=True, top_score=top_score)
            timing.total_ms = _ms(overall)
            yield ("done", StreamDone(answer=answer, timing=timing, contexts=[]))
            return

        started = time.perf_counter()
        context, available = build_context(chunks, max_chars=self.max_context_chars)
        timing.build_context_ms = _ms(started)

        if not available:
            answer = Answer(text=REFUSAL_TEXT, refused=True, top_score=top_score)
            timing.total_ms = _ms(overall)
            yield ("done", StreamDone(answer=answer, timing=timing, contexts=[]))
            return

        from app.rag.generation.base import ANSWER_PROMPT, SYSTEM_PROMPT

        available_ids = "、".join(f"[{c.index}]" for c in available)
        prompt = ANSWER_PROMPT.format(context=context, question=question, available=available_ids)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # TTFT 的起点是**发起请求之前**，不是收到第一个 chunk 之后 ——
        # 见 `core/llm/streaming.py` 的口径说明。
        stats = StreamStats()
        parts: list[str] = []
        async for delta in stream_chat(
            client,  # type: ignore[arg-type]
            model=model,
            messages=messages,
            stats=stats,
            extra_request=extra_request,  # type: ignore[arg-type]
        ):
            parts.append(delta)
            yield ("delta", delta)

        text = "".join(parts).strip()
        citations, invalid = resolve_citations(text, available)
        timing.ttft_ms = stats.ttft_ms
        timing.total_ms = _ms(overall)

        answer = Answer(
            text=text,
            citations=citations,
            invalid_refs=invalid,
            refused=False,
            context_chars=len(context),
            used_chunks=len(available),
            top_score=top_score,
        )
        yield (
            "done",
            StreamDone(answer=answer, timing=timing, contexts=_contexts_of(answer, chunks)),
        )


@dataclass
class StreamDone:
    """流式问答的结束事件载荷。

    **为什么不直接返回 `Answer`**：`Answer` 是 `generation` 层的数据类，
    它的职责是"答案 + 引用"，不该被塞进"检索耗时、上下文块"这些编排层的信息。
    加字段进去会让 CLI（不用流式）也背上无关字段。

    所以编排层有自己的产物类型：`QAResult`（非流式）与 `StreamDone`（流式结束）。
    两者都指向同一个 `Answer`，差别只在**额外携带什么**。
    """

    answer: Answer
    timing: QATiming
    contexts: list[Chunk] = field(default_factory=list)
    """真正拼进上下文的块（按编号顺序）。UI 用它渲染"参考来源"原文。"""


def _contexts_of(answer: Answer, chunks: Sequence[Chunk]) -> list[Chunk]:
    """从 `Answer.citations` 反查出对应的 `Chunk`。

    **按引用取，不是按传入顺序取**：`Answer.citations` 只含"被答案真正引用了的"
    那几条（按编号升序），而 `chunks` 是全部检索结果。UI 的"参考来源"应该
    只列被引用到的 —— 列一堆没被用到的块会让人以为"这些也是依据"。

    找不到对应块时跳过（不抛）：那说明引用编号与上下文不一致，
    该由 `invalid_refs` 报出来，而不是在这里崩掉整个渲染。
    """
    by_index = {index + 1: chunk for index, chunk in enumerate(chunks)}
    return [by_index[c.index] for c in answer.citations if c.index in by_index]


__all__ = ["QAEngine", "QAResult", "QATiming", "StreamDone"]
