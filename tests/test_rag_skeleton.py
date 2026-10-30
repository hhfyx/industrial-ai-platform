"""W4 骨架测试：结构契约 + 三段链路。

**这个文件为什么重点测"结构"而不是"效果"**：

效果（召回率、答案准确率）在当前 embedder 下**没有意义** ——
`HashingEmbedder` 只保证字面相近，不保证语义相近（见那个模块的 docstring）。
拿它跑出来的召回率去调参，等于在优化噪声。

但结构是可以、而且必须现在就锁死的：
- `Chunk.id` 是否真的内容派生（幂等入库全靠它）
- `page` 是否真的从第一步就带下来（事后补不回来）
- 三段是不是真的共享同一个 embedder / store（配错了不会报错，只会答案变差）
- 引用编号和引用列表是否一一对应（断链接比没有引用更糟）

这些性质一旦被破坏，将来换成真模型时**你无法判断问题出在模型还是骨架**。
所以这批测试的名字里几乎都有 "structure" 或具体的不变量，而不是 "works"。
"""

from __future__ import annotations

import pytest

from app.rag import Chunk, Document
from app.rag.embedding import HashingEmbedder, get_embedder
from app.rag.embedding.base import Embedder
from app.rag.generation import build_context, generate_answer
from app.rag.indexing import Indexer, index_paths
from app.rag.pipeline import RagPipeline, build_pipeline
from app.rag.retrieval import Retriever
from app.rag.schema import restore_location
from app.rag.splitters import ChunkSize, RecursiveSplitter, Splitter
from app.rag.stores import InMemoryVectorStore, VectorStore, get_store

# ============================================================ 0. 假 client


class _Msg:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str | None) -> None:
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content: str | None) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(kwargs)
        return _Resp("违约金按合同总额计算 [1]。")


class FakeClient:
    """只实现链路真正用到的那一个方法。鸭子类型在这里是对的。"""

    def __init__(self) -> None:
        self.completions = _Completions()
        self.chat = type("Chat", (), {"completions": self.completions})()


def _doc(text: str, *, page: int | None = 1, source: str = "t.pdf") -> Document:
    return Document(source=source, page=page, text=text)


# ============================================================ 1. 数据结构


def test_chunk_id_is_content_derived_and_stable() -> None:
    """同样的 (source, page, text) → 同样的 id，且**跨构造调用**一致。

    这是幂等入库的全部基础。若 id 里混进时间戳/随机数/自增序号，
    "重跑一次索引"就会让库里条数翻倍 —— 而检索看起来还挺正常。
    """
    a = Chunk(text="违约金 1‰", page=7, source="合同.pdf")
    b = Chunk(text="违约金 1‰", page=7, source="合同.pdf")
    assert a.id == b.id
    assert len(a.id) == 16

    # 内容变一个字就该是新 id（否则"改了内容但 id 不变"会让旧向量继续被命中）
    assert Chunk(text="违约金 2‰", page=7, source="合同.pdf").id != a.id
    # 页码或来源变化同样是新记录：同一段文字出现在两份文档里必须能分别溯源
    assert Chunk(text="违约金 1‰", page=8, source="合同.pdf").id != a.id
    assert Chunk(text="违约金 1‰", page=7, source="另一份.pdf").id != a.id


def test_chunk_id_ignores_position_index() -> None:
    """**index 不参与 id 计算**：改切分参数导致块序号平移时，id 必须不变。

    否则"只调了 chunk_chars"会表现成"整库 id 全部失效"，幂等性被参数调优打穿。
    """
    a = Chunk(text="同一段文字", page=1, source="s.pdf", index=0)
    b = Chunk(text="同一段文字", page=1, source="s.pdf", index=9)
    assert a.id == b.id


def test_chunk_page_is_carried_through_splitting() -> None:
    """`page` 必须在结构里 —— 这是"不能事后补"的那一条。

    断言的不是"page 有值"，而是页码信息一路传到了块上。
    """
    docs = [_doc("第一页内容" * 20, page=1), _doc("第二页内容" * 20, page=2)]
    chunks = RecursiveSplitter(ChunkSize(chunk_chars=60, overlap_chars=0)).split(docs)
    assert chunks, "应当切出块"
    for chunk in chunks:
        assert chunk.page in (1, 2), "块丢了页码"
        assert chunk.pages, "块丢了覆盖页码"
    assert {c.page for c in chunks} == {1, 2}


def test_chunk_keeps_none_page_for_non_paged_sources() -> None:
    """Markdown 没有页的概念，`page=None` 必须**一路传下去**，不能伪造 page=1。"""
    chunks = RecursiveSplitter(ChunkSize(chunk_chars=50, overlap_chars=0)).split(
        [_doc("没有页码的文档内容" * 5, page=None, source="notes.md")]
    )
    assert chunks
    assert all(c.page is None for c in chunks)
    assert all(c.pages == () for c in chunks)
    assert all(c.location == "notes.md" for c in chunks)


def test_chunk_location_covers_multiple_pages() -> None:
    """跨页块的位置要能读成 `p7-9`，而不是只报起始页。"""
    assert Chunk(text="x", page=7, source="a.pdf", pages=(7, 8, 9)).location == "a.pdf:p7-9"
    assert Chunk(text="x", page=7, source="a.pdf", pages=(7,)).location == "a.pdf:p7"


def test_metadata_roundtrip_through_store_encoding() -> None:
    """扁平化往返：`to_payload` 编码的 page/-1 与 pages 必须能被 `restore_location` 还原。

    编码只出现一次、解码也只出现一次，两边不会各自漂移。
    """
    chunk = Chunk(text="跨页内容", page=7, source="a.pdf", pages=(7, 8), index=3)
    payload = chunk.to_payload()
    page, pages = restore_location(payload["metadata"])
    assert page == 7
    assert pages == (7, 8)
    assert payload["metadata"]["chunk_index"] == 3

    page2, pages2 = restore_location(
        Chunk(text="无页", page=None, source="n.md").to_payload()["metadata"]
    )
    assert page2 is None
    assert pages2 == ()


def test_document_still_rejects_bad_page() -> None:
    """老约束不能因为迁移文件而丢：page=0 是"从 0 开始"这类混淆的信号。"""
    with pytest.raises(ValueError, match="page 必须"):
        Document(source="x.pdf", page=0, text="t")


# ============================================================ 2. 切分


def test_split_respects_chunk_size() -> None:
    """每块不得超过 `chunk_chars` —— 超了 embedding 请求会变贵甚至被静默截断。"""
    text = "这是一句话。" * 300  # 1800 字
    chunks = RecursiveSplitter(ChunkSize(chunk_chars=200, overlap_chars=20)).split([_doc(text)])
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 200, f"块超长：{len(chunk.text)}"


def test_split_prefers_sentence_boundary() -> None:
    """优先在句号处断 —— 切出"半句话"是检索质量最直接的下水道。"""
    text = "第一句话讲付款。第二句话讲违约。第三句话讲验收。" * 20
    chunks = RecursiveSplitter(ChunkSize(chunk_chars=100, overlap_chars=0)).split([_doc(text)])
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert chunk.text.endswith("。"), f"没有在句子边界切：{chunk.text[-20:]!r}"


def test_split_overlap_makes_neighbour_blocks_share_text() -> None:
    """重叠要真的产生 —— 它是跨块信息断裂的唯一缓解手段。"""
    text = "ABCDEFGHIJ" * 30
    chunks = RecursiveSplitter(ChunkSize(chunk_chars=100, overlap_chars=20)).split([_doc(text)])
    assert len(chunks) > 1
    # 后一块的开头应当能在前一块的尾部找到（重叠生效）
    assert chunks[1].text[:20] in chunks[0].text


def test_split_skips_empty_documents() -> None:
    """空 Document 不产出空块：空块会污染索引，且"少了/多了"都难查。"""
    assert RecursiveSplitter().split([]) == []
    assert RecursiveSplitter().split([_doc("   ")]) == []


def test_chunk_size_rejects_non_converging_overlap() -> None:
    """overlap >= chunk_chars 时步进为 0，切分要么死循环、要么静默挤成一块。

    这条约束必须挂在结构上，不能靠调用方自觉。
    """
    with pytest.raises(ValueError, match="overlap_chars"):
        ChunkSize(chunk_chars=100, overlap_chars=100)
    with pytest.raises(ValueError, match="chunk_chars"):
        ChunkSize(chunk_chars=0)


def test_splitter_satisfies_protocol() -> None:
    """装配要能被验证 —— 这是 `runtime_checkable` 的唯一用途。"""
    assert isinstance(RecursiveSplitter(), Splitter)


# ============================================================ 3. 向量化


def test_hashing_embedder_is_deterministic() -> None:
    """同样的文本必须得到同样的向量。

    **跨进程也要一致** —— 所以实现里用 blake2b 而不是内置 `hash()`（带随机盐）。
    这里只能测同进程，但实现的选择理由是跨进程的。
    """
    embedder = HashingEmbedder(dimension=64)
    a = embedder.embed(["违约金条款"])[0]
    b = embedder.embed(["违约金条款"])[0]
    assert a == b
    assert len(a) == 64


def test_embedding_keeps_input_order() -> None:
    """批量返回的顺序必须与入参严格一致。

    调用方是按位置把向量和块配对的；一旦顺序被改（排序/去重），
    向量就会静默错配到别的块上 —— 最难查的一类 bug。
    """
    embedder = HashingEmbedder(dimension=32)
    vectors = embedder.embed(["甲", "乙", "丙"])
    assert len(vectors) == 3
    assert vectors[0] == embedder.embed(["甲"])[0]
    assert vectors[2] == embedder.embed(["丙"])[0]


def test_embedding_is_l2_normalized() -> None:
    """归一化之后余弦可以直接用点积算，短文本不会因为"字少"天然吃亏。"""
    vector = HashingEmbedder(dimension=64).embed(["合同的违约责任条款"])[0]
    norm = sum(v * v for v in vector) ** 0.5
    assert abs(norm - 1.0) < 1e-9


def test_embedding_empty_text_gives_zero_vector_not_exception() -> None:
    """整页扫描件（无文本）是正常状态，不该炸掉整批索引。"""
    vector = HashingEmbedder(dimension=16).embed([""])[0]
    assert vector == [0.0] * 16


def test_embedder_is_swappable_by_name() -> None:
    assert isinstance(get_embedder("hashing"), Embedder)
    with pytest.raises(ValueError, match="未知的 embedder"):
        get_embedder("openai")


# ============================================================ 4. 向量库（内存实现）


def test_store_upsert_is_idempotent() -> None:
    """**核心不变量**：同一批块重复 upsert，条数与 id 都不变。

    upsert 语义（而不是 insert）是内容派生 id 存在的意义所在。
    若这里退化成 insert，"重跑一次索引"就会让库里条数翻倍，
    而检索结果看起来还挺正常 —— 这种坏法最难发现。
    """
    store = InMemoryVectorStore()
    chunks = [
        Chunk(text="块一", page=1, source="a.pdf"),
        Chunk(text="块二", page=2, source="a.pdf"),
    ]
    vectors = [[1.0, 0.0], [0.0, 1.0]]

    assert store.upsert(chunks, vectors) == 2
    assert store.count() == 2
    first_ids = store.ids()

    assert store.upsert(chunks, vectors) == 2
    assert store.count() == 2, "重复入库后条数翻倍了"
    assert store.ids() == first_ids


def test_store_rejects_mismatched_vectors() -> None:
    """数量不一致必须炸，不能截断到较短的 —— 静默截断会把"漏了一条"
    伪装成"索引成功，但检索就是差一点"。"""
    store = InMemoryVectorStore()
    with pytest.raises(ValueError, match="数量不一致"):
        store.upsert([Chunk(text="一", page=1, source="a")], [])


def test_store_search_returns_top_k_in_descending_order() -> None:
    store = InMemoryVectorStore()
    store.upsert(
        [
            Chunk(text="甲", page=1, source="a"),
            Chunk(text="乙", page=2, source="a"),
            Chunk(text="丙", page=3, source="a"),
        ],
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]],
    )
    hits = store.search([1.0, 0.0, 0.0], top_k=2)
    assert len(hits) == 2
    assert hits[0].chunk.text == "甲"
    assert hits[0].score >= hits[1].score


def test_store_search_on_empty_store_returns_empty() -> None:
    """空库返回空列表而不是抛异常：第一次提问时库本来就可能是空的。"""
    assert InMemoryVectorStore().search([1.0, 0.0], top_k=3) == []


def test_store_clear_removes_everything() -> None:
    """重建索引前必须能清干净，否则旧向量会和新向量混在一起。"""
    store = InMemoryVectorStore()
    store.upsert([Chunk(text="x", page=1, source="a")], [[1.0, 0.0]])
    store.clear()
    assert store.count() == 0


def test_store_rejects_dimension_mismatch() -> None:
    """维度不一致要炸：那是索引与查询用了不同 embedder 的**结构性错误**，
    把它伪装成"没命中"会让人在错误的层面查问题。"""
    store = InMemoryVectorStore()
    store.upsert([Chunk(text="x", page=1, source="a")], [[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="维度不一致"):
        store.search([1.0, 0.0], top_k=1)


def test_store_is_swappable_by_name() -> None:
    assert isinstance(get_store("memory"), VectorStore)
    with pytest.raises(ValueError, match="未知的向量库"):
        get_store("milvus")


# ============================================================ 5. 索引段


def test_index_report_numbers_are_self_consistent() -> None:
    """`ok` 断言四步计数自洽：切出多少块就该有多少向量、写进多少条。

    向量与块是**按位**对应的，数量一旦不一致，检索就会把 A 的向量配上 B 的正文。
    """
    store = InMemoryVectorStore()
    indexer = Indexer(embedder=HashingEmbedder(dimension=32), store=store)
    report = indexer.index_documents([_doc("合同正文" * 100, page=1)], source="a.pdf")

    assert report.documents == 1
    assert report.chunks > 0
    assert report.chunks == report.vectors == report.stored
    assert report.store_total == report.chunks
    assert report.ok is True


def test_index_empty_document_reports_zero_but_does_not_raise() -> None:
    """用户上传纯扫描件时，得到一个"0 块"的报告比得到 500 更容易理解。"""
    indexer = Indexer(embedder=HashingEmbedder(), store=InMemoryVectorStore())
    report = indexer.index_documents([], source="scanned.pdf")
    assert report.chunks == 0
    assert report.ok is False


def test_index_requires_store() -> None:
    """没装配 store 就必须炸：静默"索引成功但没落库"会让下一个人查半天。"""
    indexer = Indexer(embedder=HashingEmbedder())
    with pytest.raises(RuntimeError, match="store"):
        indexer.index_documents([_doc("x")], source="a")


def test_index_carries_loader_warnings_into_report() -> None:
    """ "有 3 页没文本"必须跟着索引报告走 —— 否则报告全绿，
    而用户永远不知道那 3 页查不到。"""
    indexer = Indexer(embedder=HashingEmbedder(), store=InMemoryVectorStore())
    report = indexer.index_documents(
        [_doc("正文", page=1)],
        source="a.pdf",
        warnings=["有 3 页没有可提取文本"],
    )
    assert report.warnings == ["有 3 页没有可提取文本"]
    assert report.to_dict()["warnings"] == ["有 3 页没有可提取文本"]


def test_index_real_pdf_keeps_page_attribution() -> None:
    """真实 50 页 PDF：块必须带着真实页码，且页码落在 1..50 之内。

    只断言"切出块"是不够的 —— 页码全丢也能过。
    """
    from pathlib import Path

    store = InMemoryVectorStore()
    indexer = Indexer(embedder=HashingEmbedder(dimension=64), store=store)
    report = indexer.index_file(Path(__file__).parent / "fixtures" / "sample50.pdf")

    assert report.documents == 50
    assert report.chunks > 0
    assert report.ok is True

    for text in ("第 7 章",):
        hits = store.search(HashingEmbedder(dimension=64).embed([text])[0], top_k=1)
        assert hits, "检索不到真实 PDF 的内容"
        assert hits[0].chunk.page is not None
        assert 1 <= hits[0].chunk.page <= 50


def test_index_paths_tolerates_one_bad_file() -> None:
    """一份坏文件不该让其他好文件白跑（与 loader 的"能读的读出来"同一原则）。"""
    from pathlib import Path

    fixtures = Path(__file__).parent / "fixtures"
    reports = index_paths(
        [fixtures / "sample.md", fixtures / "does-not-exist.pdf"],
        embedder=HashingEmbedder(),
        store=InMemoryVectorStore(),
    )
    assert len(reports) == 2
    assert reports[0].ok is True
    assert reports[1].ok is False
    assert any("索引失败" in w for w in reports[1].warnings)


# ============================================================ 6. 检索段


def test_retrieve_returns_empty_for_blank_query() -> None:
    """空 query 不调 embedder：零向量会让"分数全是 0"看起来像检索成功，
    实际是随便挑的。让它在源头就返回空，问题才能被看见。"""
    store = InMemoryVectorStore()
    store.upsert([Chunk(text="x", page=1, source="a")], [[1.0, 0.0]])
    retriever = Retriever(embedder=HashingEmbedder(dimension=2), store=store)
    assert retriever.retrieve("   ").is_empty


def test_retrieve_respects_top_k_override() -> None:
    store = InMemoryVectorStore()
    embedder = HashingEmbedder(dimension=16)
    chunks = [Chunk(text=f"第 {i} 段内容", page=i, source="a.pdf") for i in range(1, 6)]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))

    retriever = Retriever(embedder=embedder, store=store, top_k=2)
    assert len(retriever.retrieve("段内容").hits) == 2
    assert len(retriever.retrieve("段内容", top_k=4).hits) == 4


def test_retrieve_rejects_bad_top_k() -> None:
    retriever = Retriever(embedder=HashingEmbedder(), store=InMemoryVectorStore())
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("问题", top_k=0)


def test_retrieval_result_exposes_chunks_without_scores() -> None:
    """分数跨库不可比，所以对外主要给块；分数单独取。"""
    store = InMemoryVectorStore()
    embedder = HashingEmbedder(dimension=16)
    chunk = Chunk(text="违约责任", page=3, source="合同.pdf")
    store.upsert([chunk], embedder.embed([chunk.text]))

    result = Retriever(embedder=embedder, store=store, top_k=1).retrieve("违约责任")
    assert result.chunks[0].page == 3
    assert len(result.scores()) == 1


# ============================================================ 7. 生成段


def _hits_for_generation() -> list[Chunk]:
    return [
        Chunk(text="违约金按合同总额的千分之一计算。", page=7, source="合同.pdf"),
        Chunk(text="逾期超过三十日，甲方有权解除合同。", page=9, source="合同.pdf"),
    ]


def test_build_context_numbers_blocks_from_one() -> None:
    context, citations = build_context(_hits_for_generation())
    assert "[1]" in context
    assert "[2]" in context
    assert [c.index for c in citations] == [1, 2]
    assert citations[0].page == 7
    assert citations[0].label == "合同.pdf 第 7 页"


def test_build_context_drops_whole_block_instead_of_truncating() -> None:
    """超预算时**整块丢弃**：截断一半的块会让模型读到残句，
    然后照着残句给出一个看起来有依据的错答案 —— 比少给一块糟得多。"""
    chunks = _hits_for_generation()
    context, citations = build_context(chunks, max_chars=60)
    assert len(citations) == 1
    assert "解除合同" not in context
    # 保留的那一块必须是**完整**的（结尾标点还在），不是被截半的残句
    assert "千分之一计算。" in context


def test_build_context_keeps_one_block_when_budget_tiny() -> None:
    """预算比单块还小时必须放一条，否则模型收到空资料 —— 那必然导致编造。"""
    context, citations = build_context(_hits_for_generation(), max_chars=10)
    assert len(citations) == 1
    assert context


def test_build_context_rejects_bad_budget() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        build_context([], max_chars=0)


@pytest.mark.anyio
async def test_generate_answer_short_circuits_on_empty_context() -> None:
    """一条都没命中时**不调模型**：让模型对着空上下文回答，它唯一能做的就是编。

    省这次调用不是目的，避免一次必然的幻觉才是。
    """
    client = FakeClient()
    answer = await generate_answer("违约金怎么算", [], client=client, model="m")
    assert answer.text == "资料中没有提到相关内容。"
    assert answer.citations == []
    assert client.completions.calls == [], "空上下文时不该发起模型调用"


@pytest.mark.anyio
async def test_generate_answer_returns_citations_matching_prompt_numbers() -> None:
    """引用编号必须和 prompt 里的编号一一对应 —— 断链接比没有引用更糟。"""
    client = FakeClient()
    answer = await generate_answer("违约金怎么算", _hits_for_generation(), client=client, model="m")

    assert answer.citations[0].index == 1
    assert answer.text
    assert answer.used_chunks == 2
    assert answer.context_chars > 0

    prompt = client.completions.calls[0]["messages"][1]["content"]
    for citation in answer.citations:
        assert f"[{citation.index}]" in prompt, "prompt 里的编号与引用列表对不上"


@pytest.mark.anyio
async def test_generate_answer_uses_zero_temperature() -> None:
    """问答要可复现：同样的资料 + 同样的问题应该给同样的答案。"""
    client = FakeClient()
    await generate_answer("问题", _hits_for_generation(), client=client, model="m")
    assert client.completions.calls[0]["temperature"] == 0.0


def test_citation_label_without_page() -> None:
    """非分页来源的引用标签只有文件名 —— 不能显示"第 None 页"。"""
    _, citations = build_context([Chunk(text="内容", page=None, source="notes.md")])
    assert citations[0].label == "notes.md"


# ============================================================ 8. 三段装配


def test_pipeline_shares_embedder_and_store_between_phases() -> None:
    """**装配的唯一正确接法**：索引段与检索段必须共用同一个 embedder 和 store。

    配错了不会报错，只会答案变差 —— 所以这条必须由测试锁住，
    而不是靠注释提醒。
    """
    pipeline = RagPipeline()
    assert pipeline.indexer.embedder is pipeline.retriever.embedder
    assert pipeline.indexer.store is pipeline.retriever.store
    assert pipeline.indexer.embedder is pipeline.embedder


def test_pipeline_end_to_end_offline() -> None:
    """三段串起来能跑通（离线、不用 key、不花钱）。"""
    pipeline = RagPipeline(top_k=2)
    report = pipeline.index_text("违约金按合同总额的千分之一计算。", source="合同")
    assert report.ok is True

    retrieved = pipeline.retrieve("违约金")
    assert retrieved.hits, "索引进去的内容检索不到"
    assert retrieved.chunks[0].source == "合同"


@pytest.mark.anyio
async def test_pipeline_answer_returns_cited_answer() -> None:
    pipeline = RagPipeline(top_k=2)
    pipeline.index_text("违约金按合同总额的千分之一计算。", source="合同")
    answer = await pipeline.answer("违约金怎么算", client=FakeClient(), model="m")
    assert answer.text
    assert answer.citations


def test_pipeline_index_text_has_no_fake_page() -> None:
    """一段粘贴文本没有页的概念，硬给 page=1 会让"第 1 页"变成谎言。"""
    pipeline = RagPipeline(top_k=1)
    pipeline.index_text("粘贴进来的一段话", source="clipboard")
    hits = pipeline.retrieve("粘贴").hits
    assert hits
    assert hits[0].chunk.page is None
    assert hits[0].chunk.location == "clipboard"


def test_build_pipeline_by_name() -> None:
    assert isinstance(build_pipeline().store, InMemoryVectorStore)
