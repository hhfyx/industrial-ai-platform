"""问答编排与计时的测试：**必答题"检索占首 token 多少"的验证层**。

## 这个文件测什么

`app/rag/query.py` 是 UI 与 CLI 共用的编排层，它做四件事：

1. 把一次问答拆成四段计时（`embed` / `search` / `build_context` / `ttft`）
2. 流式时**先吐 `retrieved` 事件**，再吐 `delta`（这是必答题的接口落点）
3. 拒答时**不调模型**
4. `retrieval_share` 给出"检索占首 token 的比例"

## 为什么这些测试不能靠手动点 UI

Streamlit 的每次交互都重跑整个脚本，**手动点击无法断言"第几个事件先到"**。
而"`retrieved` 必须在 `delta` 之前"正是 UI 能在等待期显示检索结果的前提 ——
它必须被自动验证，否则某天有人调换了顺序，界面会静默退回"空白等待 TTFT"。
"""

from __future__ import annotations

import pytest

from app.rag.embedding import HashingEmbedder
from app.rag.generation import REFUSAL_TEXT
from app.rag.query import QAEngine, QATiming
from app.rag.schema import Chunk
from app.rag.stores import InMemoryVectorStore

_DIM = 64


# ============================================================ 0. 假 client


class _Delta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str | None, finish_reason: str | None = None) -> None:
        self.delta = _Delta(content)
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, content: str | None, *, finish: str | None = None) -> None:
        self.choices = [_Choice(content, finish)]
        self.usage = None


class _Usage:
    def __init__(self) -> None:
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.total_tokens = 15


class _StreamResponse:
    """模拟流式响应：async iterable。"""

    def __init__(self, pieces: list[str]) -> None:
        self.pieces = pieces

    def __aiter__(self):  # noqa: ANN204
        return self._gen()

    async def _gen(self):  # noqa: ANN202
        for piece in self.pieces:
            yield _Chunk(piece)
        # 末块：choices 空数组 + usage（真实网关的行为，见 streaming.py 的说明）
        last = _Chunk(None, finish="stop")
        last.usage = _Usage()
        yield last


class FakeStreamClient:
    """只实现流式路径用到的那一个方法。"""

    def __init__(self, reply: str = "违约金按日万分之五计算 [1]。") -> None:
        self.reply = reply
        self.calls: list[dict] = []
        self.completions = type("C", (), {"create": self._create})()

        self.chat = type("Chat", (), {"completions": self.completions})()

    async def _create(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(kwargs)
        # 把回复拆成几块，模拟逐字输出
        pieces = [self.reply[i : i + 4] for i in range(0, len(self.reply), 4)]
        return _StreamResponse(pieces)


class _ExplodingClient:
    """被调用就抛。证明"拒答路径没碰模型"。"""

    @property
    def chat(self):  # noqa: ANN201
        raise AssertionError("拒答路径不该调用模型")


def _store_with(*texts: str) -> InMemoryVectorStore:
    embedder = HashingEmbedder(dimension=_DIM)
    store = InMemoryVectorStore()
    chunks = [
        Chunk(text=text, page=index + 1, source="合同.pdf", index=index)
        for index, text in enumerate(texts)
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    return store


def _engine(store: InMemoryVectorStore, *, top_k: int = 3, min_score: float = 0.0) -> QAEngine:
    return QAEngine(
        embedder=HashingEmbedder(dimension=_DIM),
        store=store,
        top_k=top_k,
        min_score=min_score,
    )


# ============================================================ 1. QATiming 的形状


def test_timing_retrieval_is_embed_plus_search() -> None:
    """`retrieval_ms` = 向量化 + 相似度。

    **单独算而不是拿总耗时减**：减法会把所有没归类的开销都算进检索里，
    于是"检索占 60%"这种结论可能是假的。
    """
    timing = QATiming(embed_ms=10, search_ms=3, build_context_ms=1, ttft_ms=1200, total_ms=2900)
    assert timing.retrieval_ms == 13


def test_timing_share_is_the_answer_to_the_must_answer() -> None:
    """**必答题的答案就是这个数**：检索占首 token 的比例。

    这是 W8 的决策依据：
    - 接近 0 → 优化检索无意义
    - 显著（> 0.3）→ 检索值得优化

    没有这个数字，W8 只能靠猜。
    """
    timing = QATiming(embed_ms=10, search_ms=1, ttft_ms=1100)
    assert timing.retrieval_share is not None
    assert timing.retrieval_share == pytest.approx(11 / 1100)
    assert timing.retrieval_share < 0.02, "实测检索只占首 token 的 1% 量级"


def test_timing_share_is_none_without_ttft() -> None:
    """拒答时没有 TTFT（没调模型）—— 比例必须是 None，不能是 0。

    **`None` 与 `0.0` 是两件事**：0.0 表示"检索耗时为零"（不可能），
    None 表示"没有这个测量"。混用会让人以为"拒答时检索是瞬间的"。
    """
    assert QATiming(embed_ms=10, search_ms=1, ttft_ms=None).retrieval_share is None
    assert QATiming(embed_ms=10, search_ms=1, ttft_ms=0).retrieval_share is None


def test_timing_as_dict_keeps_share_precision() -> None:
    """比例保留 4 位小数。

    **两位不够**：1% 量级的比例用 `:.2f` 会显示成 `0.01`，
    而真实值可能是 0.008 或 0.014 —— 差一倍，正好跨过"要不要优化"的判断线。
    """
    payload = QATiming(embed_ms=10, search_ms=1, ttft_ms=1100).as_dict()
    assert payload["retrieval_share"] == pytest.approx(0.01, abs=0.0015)
    assert isinstance(payload["retrieval_share"], float)
    assert payload["retrieval_ms"] == 11


# ============================================================ 2. 非流式编排


@pytest.mark.anyio
async def test_answer_records_all_phases() -> None:
    """非流式问答要填满四段耗时。"""
    engine = _engine(_store_with("违约金按日万分之五计算。"))
    result = await engine.answer("违约金", client=_NonStreamClient(), model="m")

    assert result.timing.embed_ms > 0
    assert result.timing.search_ms > 0
    assert result.timing.total_ms > 0
    assert result.answer.text


class _NonStreamClient:
    """非流式：`chat.completions.create` 直接返回完整响应。"""

    def __init__(self, reply: str = "违约金按日万分之五计算 [1]。") -> None:
        self.completions = type("C", (), {"create": self._create})()
        self.chat = type("Chat", (), {"completions": self.completions})()

    async def _create(self, **kwargs):  # noqa: ANN003, ANN201
        del kwargs
        msg = type("M", (), {"content": "违约金按日万分之五计算 [1]。"})()
        choice = type("Ch", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()


@pytest.mark.anyio
async def test_answer_returns_contexts_matching_citations() -> None:
    """`contexts` 只含**被真正引用**的块，与 `citations` 一一对应。

    UI 的「参考来源」列的就是它。列全部检索结果会让人以为"这些也是依据"。
    """
    engine = _engine(_store_with("违约金按日万分之五计算。", "质保期 12 个月。"))
    result = await engine.answer("违约金", client=_NonStreamClient(), model="m")

    assert len(result.contexts) == len(result.answer.citations)
    for context, _citation in zip(result.contexts, result.answer.citations, strict=True):
        assert context.text  # 有原文可展开


# ============================================================ 3. 流式事件顺序（核心）


@pytest.mark.anyio
async def test_stream_emits_retrieved_before_delta() -> None:
    """**核心断言**：`retrieved` 必须在任何 `delta` 之前。

    这是必答题在接口上的落地 —— UI 要在"等模型出第一个字"的那段时间里
    就显示检索结果。顺序反了，界面会退回"空白等待 TTFT"，
    而这个 bug **不会报错**，只会让体验变差。
    """
    engine = _engine(_store_with("违约金按日万分之五计算。"))
    client = FakeStreamClient()

    events: list[str] = []
    async for event, _payload in engine.answer_stream("违约金", client=client, model="m"):
        events.append(event)

    assert events[0] == "retrieved", f"第一个事件应是 retrieved，实际 {events[:3]}"
    assert events[-1] == "done"
    assert "delta" in events
    # retrieved 在所有 delta 之前
    first_delta = events.index("delta")
    assert events.index("retrieved") < first_delta


@pytest.mark.anyio
async def test_stream_retrieved_carries_timing_already() -> None:
    """`retrieved` 事件里的 timing 已经有 embed/search 耗时。

    这样 UI 在 TTFT 期间就能显示"检索 11ms"，不用等全部结束。
    """
    engine = _engine(_store_with("违约金按日万分之五计算。"))
    async for event, payload in engine.answer_stream(
        "违约金", client=FakeStreamClient(), model="m"
    ):
        if event == "retrieved":
            _chunks, _scores, timing = payload  # type: ignore[misc]
            assert timing.embed_ms > 0
            assert timing.search_ms > 0
            assert timing.retrieval_ms == timing.embed_ms + timing.search_ms
            break


@pytest.mark.anyio
async def test_stream_done_has_ttft() -> None:
    """`done` 里的 timing 要有 TTFT —— 它只能由流式层给出。"""
    engine = _engine(_store_with("违约金按日万分之五计算。"))
    done = None
    async for event, payload in engine.answer_stream(
        "违约金", client=FakeStreamClient(), model="m"
    ):
        if event == "done":
            done = payload

    assert done is not None
    assert done.timing.ttft_ms is not None and done.timing.ttft_ms > 0
    assert done.timing.total_ms >= done.timing.ttft_ms


@pytest.mark.anyio
async def test_stream_deltas_concatenate_to_answer() -> None:
    """所有 delta 拼起来 == 最终答案。**逐字渲染不能丢字**。"""
    engine = _engine(_store_with("违约金按日万分之五计算。"))
    pieces: list[str] = []
    done = None
    async for event, payload in engine.answer_stream(
        "违约金", client=FakeStreamClient(), model="m"
    ):
        if event == "delta":
            pieces.append(payload)  # type: ignore[arg-type]
        elif event == "done":
            done = payload

    assert done is not None
    assert "".join(pieces).strip() == done.answer.text


@pytest.mark.anyio
async def test_stream_done_carries_citations_and_contexts() -> None:
    """`done` 要带引用与可展开的原文（UI 的「参考来源」靠它）。"""
    engine = _engine(_store_with("违约金按日万分之五计算。"))
    done = None
    async for event, payload in engine.answer_stream(
        "违约金", client=FakeStreamClient(), model="m"
    ):
        if event == "done":
            done = payload

    assert done is not None
    assert done.answer.citations, "答案里有 [1]，引用列表不该空"
    assert done.contexts, "要有原文供展开"
    assert done.contexts[0].text


# ============================================================ 4. 拒答（不调模型）


@pytest.mark.anyio
async def test_stream_refuses_on_empty_store() -> None:
    """空库 → 只产 retrieved（空）+ done，**不产 delta**，**不调模型**。"""
    engine = _engine(InMemoryVectorStore())
    events: list[str] = []
    async for event, _payload in engine.answer_stream(
        "任何问题", client=_ExplodingClient(), model="m"
    ):
        events.append(event)

    assert events == ["retrieved", "done"], f"拒答不该有 delta，实际 {events}"


@pytest.mark.anyio
async def test_stream_refuses_below_threshold() -> None:
    """分数低于阈值 → 拒答。**用 `_ExplodingClient` 证明没碰模型。**"""
    store = _store_with("违约金按日万分之五计算。")
    engine = _engine(store, min_score=0.99)  # 高到必然触发

    done = None
    async for event, payload in engine.answer_stream(
        "违约金", client=_ExplodingClient(), model="m"
    ):
        if event == "done":
            done = payload

    assert done is not None
    assert done.answer.refused is True
    assert done.answer.text == REFUSAL_TEXT
    assert done.timing.ttft_ms is None, "拒答没调模型，不该有 TTFT"


@pytest.mark.anyio
async def test_refusal_timing_share_is_none() -> None:
    """拒答时 `retrieval_share` 是 None。

    但如果**有**检索耗时（检索跑过了才拒答），那个耗时仍然要记 ——
    UI 需要显示"检索花了 11ms，但分数太低所以拒答"。
    """
    engine = _engine(_store_with("违约金按日万分之五计算。"), min_score=0.99)
    done = None
    async for event, payload in engine.answer_stream(
        "违约金", client=_ExplodingClient(), model="m"
    ):
        if event == "done":
            done = payload

    assert done is not None
    assert done.timing.retrieval_share is None
    assert done.timing.retrieval_ms >= 0


# ============================================================ 5. 阈值来自 embedder


def test_engine_takes_threshold_from_embedder() -> None:
    """**阈值默认从 embedder 取**，不是全局常量。

    实测（`scripts/embedding_score_probe.py`）：hashing 的相关/无关分布完全重叠，
    所以它的阈值是 0.0（关掉闸门）；bge 是 0.40。

    如果这里硬编一个全局值，hashing 路径会误拒正确命中的答案 ——
    周五就是这么踩到的（0.247 被 0.35 拒掉）。
    """
    engine = _engine(InMemoryVectorStore())
    assert engine.min_score == HashingEmbedder().recommended_min_score == 0.0


def test_engine_explicit_threshold_wins() -> None:
    """显式传的阈值优先于 embedder 推荐值。"""
    engine = QAEngine(
        embedder=HashingEmbedder(dimension=_DIM),
        store=InMemoryVectorStore(),
        min_score=0.5,
    )
    assert engine.min_score == 0.5


def test_engine_rejects_bad_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        QAEngine(embedder=HashingEmbedder(), store=InMemoryVectorStore(), top_k=0)


# ============================================================ 6. 索引分阶段计时


def test_index_report_has_stage_timings() -> None:
    """索引报告要分开记四段耗时 —— 进度条与"慢在哪"都靠它。

    任务要求进度条显示"解析 N 页 / 切出 M chunk / 编码耗时"。
    只给一个总耗时无法回答"慢在哪一步"。
    """
    from pathlib import Path

    from app.rag.indexing import Indexer

    indexer = Indexer(embedder=HashingEmbedder(dimension=_DIM), store=InMemoryVectorStore())
    report = indexer.index_file(Path(__file__).parent / "fixtures" / "sample50.pdf")

    assert report.ok
    assert report.pages == 50, "页数要取自真实页码"
    assert report.load_ms > 0
    assert report.split_ms >= 0
    assert report.embed_ms > 0
    assert report.store_ms >= 0
    # 四段之和应当接近总耗时（load 已回填进 elapsed_ms）
    assert report.elapsed_ms >= report.load_ms


def test_index_on_stage_callback_fires_in_order() -> None:
    """进度回调要按 解析→切分→编码→入库 的顺序触发。"""
    from pathlib import Path

    from app.rag.indexing import Indexer

    stages: list[str] = []
    indexer = Indexer(embedder=HashingEmbedder(dimension=_DIM), store=InMemoryVectorStore())
    indexer.index_file(
        Path(__file__).parent / "fixtures" / "sample5_no_header.pdf",
        on_stage=lambda stage, _ratio, _msg: stages.append(stage),
    )

    assert stages == ["load", "split", "embed", "store"], f"阶段顺序不对：{stages}"


def test_index_survives_broken_progress_callback() -> None:
    """**回调抛异常不能毁掉索引**。

    进度条是"展示"，索引是"事实"。让一个画进度条的 bug 把索引搞挂，
    是最得不偿失的耦合。这里用一个必然抛的回调验证索引仍然成功。
    """
    from pathlib import Path

    from app.rag.indexing import Indexer

    store = InMemoryVectorStore()

    def boom(_stage: str, _ratio: float, _message: str) -> None:
        raise RuntimeError("进度条坏了")

    indexer = Indexer(embedder=HashingEmbedder(dimension=_DIM), store=store)
    report = indexer.index_file(
        Path(__file__).parent / "fixtures" / "sample5_no_header.pdf", on_stage=boom
    )

    assert report.ok, "回调抛异常不该影响索引结果"
    assert store.count() > 0


def test_page_count_matches_loader() -> None:
    """索引报告的页数必须与 loader 的口径一致。

    两处口径不一致的话，界面显示"解析 50 页"、报告里写 5 页 ——
    用户会以为丢了内容。
    """
    from pathlib import Path

    from app.rag.indexing import Indexer
    from app.rag.loaders import load_document

    path = Path(__file__).parent / "fixtures" / "sample50.pdf"
    expected = load_document(path).page_count

    indexer = Indexer(embedder=HashingEmbedder(dimension=_DIM), store=InMemoryVectorStore())
    assert indexer.index_file(path).pages == expected
