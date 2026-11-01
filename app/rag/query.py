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
    REFUSAL_TEXT,
    Answer,
    build_context,
    resolve_citations,
)
from app.rag.refusal import Refusal, RefusalKind, decide
from app.rag.schema import Chunk
from app.rag.settings import RagConfig, get_rag_config
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

    refusal: Refusal | None = None
    """拒答的完整说明（含理由与建议）；答出来了就是 None。

    **与 `answer.refused` 的分工**：`answer.refused` 是"有没有拒"这个布尔，
    这个对象回答"为什么拒、用户接下来该做什么"。只给布尔的话，
    CLI/UI 只能说"没找到"，而用户无法区分"文档没导入"和"问法不对"。
    """

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
        top_k: int | None = None,
        max_context_chars: int | None = None,
        min_score: float | None = None,
        config: RagConfig | None = None,
    ) -> None:
        # **参数来源收敛到 RagConfig**（W5 周三）。
        # 每个参数都保留显式覆盖，因为 CLI/UI 确实会传；但"不传时取什么"
        # 只有一个出处 —— 否则"默认 top_k 是几"又会变回四个答案。
        self.config = config or get_rag_config()
        if top_k is not None:
            self.config = self.config.with_overrides(top_k=top_k)
        if max_context_chars is not None:
            self.config = self.config.with_overrides(max_context_chars=max_context_chars)
        if min_score is not None:
            self.config = self.config.with_overrides(score_threshold=min_score)

        if self.config.top_k <= 0:
            raise ValueError("top_k 必须为正数")
        self.embedder = embedder
        self.store = store
        self.top_k = self.config.top_k
        self.max_context_chars = self.config.max_context_chars
        # 阈值：显式参数 > 配置值。
        #
        # ## 这里曾经有一个把精标结果吃掉的 bug（W5 周日发现）
        #
        # 原逻辑是"**配置值 ≠ 模块默认值时**才用配置"，本意是区分
        # "用户设了 RAG_SCORE_THRESHOLD 还是没设"。但实际行为是：
        #
        #     精标把 DEFAULT_SCORE_THRESHOLD 从 0.40 改成 0.575
        #     → `RagConfig()` 的 score_threshold 也是 0.575
        #     → 两者相等 → 判定"用户没设过" → 回退 embedder 的 0.40
        #
        # 结果 **精标后的 0.575 完全没有生效**。实测
        # `QAEngine(embedder=bge).min_score == 0.40`，而配置写的是 0.575。
        #
        # 根因是"用值相等推断意图"。改成只看**显式参数是否给**：
        # 显式传 → 用它；否则 → 用 `config.score_threshold`
        # （`get_rag_config()` 已处理环境变量覆盖与回退默认值）。
        #
        # **embedder 的推荐值不再自动生效**：它是"按模型反推的量级"，
        # 不是可直接用于生产的阈值（bge 的 0.40 在 50 条集上拒答只有 0/10）。
        # 想用它的人可以显式传 `min_score=embedder.recommended_min_score`。
        self.min_score = min_score if min_score is not None else self.config.score_threshold

        # ## 配置里的**单一数字**跨 embedder 不可移植 → 挡一道
        #
        # 实测：`hashing` 的相关分数落在 0~0.48（它与无关分布**重叠**，
        # 见 `embedding/hashing.py`），而精标后的 bge 阈值是 0.575。
        # 拿 0.575 去跑 hashing，**所有问题都会被拒** —— 而且是静默的。
        #
        # `recommended_min_score == 0.0` 是 hashing 的"分布无可分性"标记，
        # 此时配置阈值对它没有意义，回退到 0.0。
        #
        # **只在"没显式传阈值"时挡**：显式传 `min_score` 是调用方的明确决定
        # （测试要构造拒答场景、CLI 的 `--min-score` 也是），不该被这条保护覆盖。
        # 第一版写成无条件覆盖，直接把 `min_score=0.99` 抹成 0.0，
        # 导致三条拒答测试失败 —— 那是"保护"变成了"越权"。
        if min_score is None:
            if embedder.recommended_min_score == 0.0 and self.min_score > 0.0:
                logger.warning(
                    "%s 的分数分布无可分性（推荐阈值 0.0），忽略配置里的 %.3f 并关掉分数闸门",
                    embedder.name,
                    self.min_score,
                )
                self.min_score = 0.0
            elif self.min_score > embedder.recommended_min_score + 0.3:
                # **只在"差得离谱"时警告**：推荐值是模型级的量级参考
                # （bge 0.40 来自 4+4 条标注对），配置值是在具体语料上精标的
                # （0.575 来自 50 条集）。后者高于前者是**正常的**。
                # 高出 0.3 以上才说明分数分布可能根本不匹配。
                logger.warning(
                    "阈值 %.3f 比 %s 的推荐值 %.3f 高出 %.2f，"
                    "可能大量误拒；确认该 embedder 的分数分布与调阈值时用的模型一致",
                    self.min_score,
                    embedder.name,
                    embedder.recommended_min_score,
                    self.min_score - embedder.recommended_min_score,
                )

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

        # **闸门走 `refusal.decide`，不是只比一个分数**（W5 周三）。
        # 顺序：空 → 低分 → 证据形态。第三种是今天新加的，理由见
        # `refusal.py` 模块 docstring：分数挡不住"散射/平局"那类无答案。
        verdict = decide(chunks, scores, threshold=self.min_score, top_k=self.top_k)
        refusal = None if verdict.kind is RefusalKind.ANSWER else verdict

        started = time.perf_counter()
        if refusal is not None:
            # 拒答**不调模型**。
            # `answer.text` 恒等于 `REFUSAL_TEXT`（固定字符串），针对性建议走
            # `verdict`（`QAResult.refusal`）—— 理由见 `refusal.py` 模块说明最后一节：
            # 拒答文案进对话历史，必须可等值比较；建议属于展示层。
            answer = Answer(
                text=REFUSAL_TEXT,
                refused=True,
                top_score=verdict.top_score,
            )
            timing.build_context_ms = 0
        else:
            answer = await generate_answer(
                question,
                chunks,
                client=client,
                model=model,
                max_context_chars=self.max_context_chars,
                min_score=self.min_score,
                scores=scores,
            )
            timing.build_context_ms = _ms(started)
        timing.total_ms = _ms(overall)

        return QAResult(
            answer=answer,
            timing=timing,
            contexts=_contexts_of(answer, chunks),
            top_score=answer.top_score,
            refusal=refusal,
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

        # 闸门：走同一份 `refusal.decide`，**不在这里另写一份判据**。
        # 两条路径（非流式 / 流式）各写一遍闸门是这类系统最典型的漂移源：
        # CLI 拒了、UI 答了，而两边都"跑通了"。
        verdict = decide(chunks, scores, threshold=self.min_score, top_k=self.top_k)
        refusal = None if verdict.kind is RefusalKind.ANSWER else verdict

        yield ("retrieved", (chunks, scores, timing))

        if refusal is not None:
            answer = Answer(text=REFUSAL_TEXT, refused=True, top_score=verdict.top_score)
            timing.total_ms = _ms(overall)
            yield ("done", StreamDone(answer=answer, timing=timing, contexts=[], refusal=refusal))
            return

        started = time.perf_counter()
        context, available = build_context(chunks, max_chars=self.max_context_chars)
        timing.build_context_ms = _ms(started)

        if not available:
            # `build_context` 会强制放一条，所以这个分支基本走不到。
            # 保留它是为了"绝不把空上下文交给模型"这条不变量。
            answer = Answer(text=REFUSAL_TEXT, refused=True, top_score=verdict.top_score)
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
            top_score=verdict.top_score,
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

    refusal: Refusal | None = None
    """拒答说明（含理由与建议）；答出来了就是 None。与 `QAResult.refusal` 同义。"""


def _contexts_of(answer: Answer, chunks: Sequence[Chunk]) -> list[Chunk]:
    """从 `Answer.citations` 反查出对应的 `Chunk`。

    **按引用取，不是按传入顺序取**：`Answer.citations` 只含"被答案真正引用了的"
    那几条（按编号升序），而 `chunks` 是全部检索结果。UI 的"参考来源"应该
    只列被引用到的 —— 列一堆没被用到的块会让人以为"这些也是依据"。

    ## 配对用 `chunk.id`，不用编号，也不用位置

    W4 收尾时这里原来是 `by_index = {index + 1: chunk for index, chunk in enumerate(chunks)}`，
    也就是说"引用 [n] 对应 `chunks[n-1]`"——一个**位置配对**。它能跑对，
    但前提是 `build_context` 与这里对"编号"的理解完全一致。而这个前提
    在预算截断处最脆：`build_context` 里**被预算挡掉的块不占编号**
    （见那边 `index = len(citations) + 1`），所以 `chunks[n-1]` 与引用的第 n 条
    在"有块被丢弃"时并不总是同一个块。

    现在的做法是拿 `citation.chunk_id`（内容派生 id）去 `chunks` 里找。
    它同时给了三件事：
    ① 配对不依赖两处对编号的隐式约定；
    ② `citation.chunk_id` 能直接用于 `store.get_by_content_id()` 回查（双向校验）；
    ③ 找不到时能明确区分两种坏情况（见下）。

    ## 两种"找不到"，都要能看见

    | 情况 | 含义 | 这里怎么做 |
    | --- | --- | --- |
    | 编号越界 | 模型编了 `[7]` 而只有 3 条 | 由 `Answer.invalid_refs` 报，不在这里崩 |
    | 编号在界内但 id 对不上 | 引用与块快照错配 | 跳过该条（渲染少一条），不能伪造一个块 |

    找不到时**返回长度可能小于 `citations`**，UI 侧必须按 id 而非 `zip()` 渲染 ——
    否则一条错配会让后面全部错位，显示的原文与编号整片对不上。
    """
    by_id = {chunk.id: chunk for chunk in chunks}
    matched = [by_id[c.chunk_id] for c in answer.citations if c.chunk_id in by_id]
    if len(matched) < len(answer.citations):
        missing = [c.index for c in answer.citations if c.chunk_id not in by_id]
        logger.warning("引用与块快照错配，无法渲染的编号：%s", missing)
    return matched


__all__ = ["QAEngine", "QAResult", "QATiming", "StreamDone"]
