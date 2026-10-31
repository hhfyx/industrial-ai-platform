"""ChromaStore 测试：持久化、`distances` 口径、where 过滤、与内存实现的一致性。

## 这个文件要钉死的三件事

### 一、`distances` 的方向（必答题）

**cosine 距离 = 1 - 相似度，越小越像。** 搞反了阈值逻辑就全错，
而且不会报错，只会"筛出来的是最不像的"。

所以 `test_distance_is_one_minus_similarity` 用三个**构造向量**
（同向 / 正交 / 反向）把这层关系算出来，误差必须是 0：
不是"大致成立"，是 `distance = 1 - cosine` 这个定义式。

并且 `ChromaStore.search()` 必须把 distance **换算成相似度**再返回 ——
`test_search_returns_similarity_not_distance` 守着这条。

### 二、持久化（"数据怎么丢了"的正解）

`chromadb.Client()` 是内存/临时，`PersistentClient(path=...)` 才是持久化。
`test_data_survives_reopen` 用"写 → 关 → 重开"证明它真的落盘了。

### 三、两个实现给同一种分数

`test_both_stores_agree_on_score_direction` 用同一批数据、同一个查询，
断言两个 store 返回的分数与顺序**完全一致**。这条防的是
"内存里测通过、Chroma 上行为不同"——那种不一致最难查。

## 测试用 tmp_path

每个测试一个独立的 `tmp_path` 目录，**不碰 `data/chroma`**。
理由：测试跑在仓库里会留下 `data/` 垃圾，而且并行跑时互相污染。
"""

from __future__ import annotations

import math

import pytest

from app.rag.embedding import HashingEmbedder
from app.rag.rag_availability import chroma_available
from app.rag.schema import Chunk, Document
from app.rag.stores import InMemoryVectorStore, VectorStore
from app.rag.stores.chroma import (
    DEFAULT_COLLECTION,
    DEFAULT_PERSIST_PATH,
    DISTANCE_SPACE,
    ChromaStore,
)

pytestmark = pytest.mark.skipif(not chroma_available(), reason="需要 chromadb")

_DIM = 64


def _norm(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(x * x for x in vector))
    return [x / length for x in vector] if length else vector


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(_norm(a), _norm(b), strict=True))


def _chunk(text: str, *, page: int | None = 1, source: str = "合同A.pdf", index: int = 0) -> Chunk:
    return Chunk(text=text, page=page, source=source, index=index)


def _store(tmp_path, *, name: str = "test_collection") -> ChromaStore:
    """每个测试一个独立目录 + 独立 collection 名。"""
    return ChromaStore(tmp_path / "chroma", collection_name=name)


# ============================================================ 1. distances 语义（必答题）


def test_distance_is_one_minus_similarity(tmp_path) -> None:
    """**必答题**：cosine 空间下 `distance = 1 - 相似度`，误差 0。

    用三个构造向量把关系算到底：
    - A 与查询同向  → 余弦  1.0 → distance 0.0（**最像，排最前**）
    - B 与查询正交  → 余弦  0.0 → distance 1.0
    - C 与查询反向  → 余弦 -1.0 → distance 2.0（**最不像，排最后**）

    两个必须记住的结论：
    1. **distance 越小越像**（与相似度方向相反）
    2. **distance 范围是 [0, 2]**，不是 [0, 1] —— 因为余弦可以是负数。
       按 `distance < 0.8` 当阈值的人，实际表达的是"相似度 > 0.2"，宽松得多。
    """
    import chromadb

    store_dir = tmp_path / "probe"
    store_dir.mkdir()
    client = chromadb.PersistentClient(path=str(store_dir))
    collection = client.get_or_create_collection(
        name="distance_probe", metadata={"hnsw:space": "cosine"}
    )

    query = _norm([1.0, 0.0, 0.0])
    same = _norm([1.0, 0.0, 0.0])
    orthogonal = _norm([0.0, 1.0, 0.0])
    opposite = _norm([-1.0, 0.0, 0.0])
    collection.add(
        ids=["same", "orthogonal", "opposite"],
        embeddings=[same, orthogonal, opposite],
        documents=["same", "orthogonal", "opposite"],
        metadatas=[{"k": "v"} for _ in range(3)],
    )

    raw = collection.query(query_embeddings=[query], n_results=3, include=["distances"])
    order = raw["ids"][0]
    distances = raw["distances"][0]

    # 返回顺序是**距离升序** → 越像越靠前
    assert order == ["same", "orthogonal", "opposite"], f"顺序不对：{order}"

    expected = {"same": 1.0, "orthogonal": 0.0, "opposite": -1.0}
    for doc_id, distance in zip(order, distances, strict=True):
        similarity = expected[doc_id]
        assert distance == pytest.approx(1.0 - similarity, abs=1e-6), (
            f"{doc_id}: distance={distance} 不等于 1 - {similarity}"
        )
    # 范围是 [0, 2] 而不是 [0, 1]
    assert max(distances) == pytest.approx(2.0, abs=1e-6)


def test_search_returns_similarity_not_distance(tmp_path) -> None:
    """`ChromaStore.search()` 必须把 distance **换算成相似度**再返回。

    这是必答题的落地。如果直接把 distance 当 score 交出去：
    - 排序会反（调用方按 score 降序 → 最不像的排第一）
    - 与 `InMemoryVectorStore` 的分数不可比（一个越大越像、一个越小越像）

    探针：查询与已知块**完全同向**，相似度应为 1.0。
    若返回 0.0，说明它把 distance 直接交出去了。
    """
    store = _store(tmp_path)
    vector = _norm([1.0, 0.0] + [0.0] * (_DIM - 2))
    store.upsert([_chunk("完全同向的块")], [vector])

    hits = store.search(vector, top_k=1)
    assert hits
    assert hits[0].score == pytest.approx(1.0, abs=1e-5), (
        f"同向向量的相似度应为 1.0，实际 {hits[0].score}（0.0 说明返回的是 distance）"
    )


def test_score_is_higher_for_more_similar(tmp_path) -> None:
    """**分数方向**：越像分数越大。这条与上一条互补 ——
    上一条证明"同向 = 1.0"，这条证明"更像的排更前"。"""
    embedder = HashingEmbedder(dimension=_DIM)
    store = _store(tmp_path)

    chunks = [
        _chunk("违约金按日万分之五计算", page=1, source="A.pdf"),
        _chunk("完全无关的一句话讲天气", page=2, source="A.pdf"),
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))

    query = embedder.embed_query(["违约金"])[0]
    hits = store.search(query, top_k=2)
    assert hits[0].score > hits[1].score
    assert "违约金" in hits[0].chunk.text


# ============================================================ 2. 持久化


def test_data_survives_reopen(tmp_path) -> None:
    """**必答题之外最重要的一条**：`PersistentClient(path=...)` 真的落盘。

    `chromadb.Client()`（不带 path）是内存/临时，重启就没了 ——
    这是 W4 最高频的"数据怎么丢了"。这里用"写 → 关 → 重开"证明路径有效。
    """
    path = tmp_path / "chroma"
    embedder = HashingEmbedder(dimension=_DIM)

    first = ChromaStore(path, collection_name="persist_test")
    chunks = [_chunk("违约金条款", page=1), _chunk("付款条款", page=2)]
    first.upsert(chunks, embedder.embed([c.text for c in chunks]))
    assert first.count() == 2
    del first  # 模拟进程退出

    second = ChromaStore(path, collection_name="persist_test")
    assert second.count() == 2, "重启后数据丢了 —— PersistentClient 的 path 没生效"

    hits = second.search(embedder.embed_query(["违约金"])[0], top_k=1)
    assert hits and hits[0].chunk.text == "违约金条款"
    # metadata 也要活下来（页码是最容易在往返中丢的）
    assert hits[0].chunk.page == 1


def test_collection_metadata_records_cosine_space(tmp_path) -> None:
    """collection 必须显式声明 `hnsw:space=cosine`。

    Chroma 的默认是 **l2**。用 l2 时 distance 范围是 [0,4]，
    拿它当相似度用会得到一堆无法解释的数字 —— 而且不会报错。
    """
    store = _store(tmp_path)
    assert store._collection.metadata is not None
    assert store._collection.metadata.get("hnsw:space") == DISTANCE_SPACE == "cosine"


def test_persist_path_default_is_explicit_not_temp() -> None:
    """默认 path 必须是显式常量，不能靠 Chroma 的默认行为。

    这条是**静态检查**（不建库）：只要 `DEFAULT_PERSIST_PATH` 是个非空字符串，
    就说明代码里给出了显式路径。Chroma 在 `path=None` 时会用临时目录，
    那样"重启后数据丢了"是必然的。
    """
    from pathlib import Path

    assert isinstance(DEFAULT_PERSIST_PATH, str) and DEFAULT_PERSIST_PATH
    # 相对项目根，不是绝对路径、不是 ~/.cache
    assert not Path(DEFAULT_PERSIST_PATH).is_absolute()
    assert DEFAULT_PERSIST_PATH.startswith("data/")


# ============================================================ 3. 幂等与删除


def test_upsert_is_idempotent_on_store_id(tmp_path) -> None:
    """同一批块重复 upsert，条数与主键都不变。"""
    store = _store(tmp_path)
    embedder = HashingEmbedder(dimension=_DIM)
    chunks = [_chunk("块一", page=1, index=0), _chunk("块二", page=2, index=1)]
    vectors = [embedder.embed([c.text])[0] for c in chunks]

    store.upsert(chunks, vectors)
    ids_first = sorted(store.ids())
    store.upsert(chunks, vectors)

    assert store.count() == 2
    assert sorted(store.ids()) == ids_first


def test_upsert_same_position_overwrites_content(tmp_path) -> None:
    """**位置派生主键的另一面**：同 (doc, page, chunk_index) 的新内容**覆盖**旧内容。

    这是有意的取舍（见 `schema.Chunk.store_id`）：同一个位置应该只有一条。
    代价是"改切分参数后 id 会指向不同内容"—— 所以重建索引前必须
    `delete_by_source`，不能指望 id 帮你发现不一致。
    """
    store = _store(tmp_path)
    embedder = HashingEmbedder(dimension=_DIM)

    old = _chunk("旧版本的违约金条款", page=7, index=2)
    new = _chunk("新版本的违约金条款", page=7, index=2)
    assert old.store_id == new.store_id, "位置相同的两块 store_id 必须相同"

    store.upsert([old], embedder.embed([old.text]))
    store.upsert([new], embedder.embed([new.text]))

    assert store.count() == 1, "同位置的两块应当互相覆盖，而不是并存"
    hits = store.search(embedder.embed_query(["违约金"])[0], top_k=5)
    assert hits[0].chunk.text == "新版本的违约金条款"


def test_delete_by_source_removes_only_that_document(tmp_path) -> None:
    """按 source 删除：只删指定文档，别的不受影响。

    这是**重建单份文档**的标准流程的第一步 —— 不删的话，
    "旧文档 10 块、新文档 8 块"会留下 2 条孤儿，一直被检索到。
    """
    store = _store(tmp_path)
    embedder = HashingEmbedder(dimension=_DIM)
    chunks = [
        _chunk("A 的块一", page=1, source="A.pdf", index=0),
        _chunk("A 的块二", page=2, source="A.pdf", index=1),
        _chunk("B 的块一", page=1, source="B.pdf", index=0),
    ]
    store.upsert(chunks, [embedder.embed([c.text])[0] for c in chunks])

    removed = store.delete_by_source("A.pdf")
    assert removed == 2
    assert store.count() == 1

    hits = store.search(embedder.embed_query(["块"])[0], top_k=5)
    assert all(h.chunk.source == "B.pdf" for h in hits)


def test_delete_by_source_on_missing_returns_zero(tmp_path) -> None:
    """删不存在的 source 返回 0，不抛 —— 幂等删除才能安全重跑。"""
    store = _store(tmp_path)
    assert store.delete_by_source("不存在.pdf") == 0


def test_clear_keeps_collection_and_metadata(tmp_path) -> None:
    """清空数据但**保留 collection 与它的 cosine 配置**。

    如果实现是"删 collection 再重建"，重建时忘了传 `hnsw:space=cosine`
    就会被悄悄改成默认的 l2 —— 那是静默的口径变更，比数据丢了更难查。
    """
    store = _store(tmp_path)
    embedder = HashingEmbedder(dimension=_DIM)
    store.upsert([_chunk("x", page=1)], [embedder.embed(["x"])[0]])
    store.clear()

    assert store.count() == 0
    assert store._collection.metadata.get("hnsw:space") == "cosine", "清空后口径被改了"


# ============================================================ 4. where 过滤


def _filter_fixture(tmp_path):
    """3 份文档 × 3 页的固定语料，专供过滤测试。"""
    store = _store(tmp_path)
    embedder = HashingEmbedder(dimension=_DIM)
    chunks = []
    for source in ("合同A.pdf", "合同B.pdf", "合同C.pdf"):
        for page in (1, 2, 3):
            chunks.append(
                _chunk(f"{source} 第 {page} 页的条款内容", page=page, source=source, index=page)
            )
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    return store, embedder


def test_where_filters_by_source(tmp_path) -> None:
    store, embedder = _filter_fixture(tmp_path)
    hits = store.search(embedder.embed_query(["条款"])[0], top_k=20, where={"source": "合同B.pdf"})

    assert hits, "按 source 过滤没命中"
    assert all(h.chunk.source == "合同B.pdf" for h in hits), "过滤返回了别的文档"
    assert len(hits) == 3, f"合同B 有 3 条，实际 {len(hits)}"


def test_where_filters_by_page(tmp_path) -> None:
    store, embedder = _filter_fixture(tmp_path)
    hits = store.search(embedder.embed_query(["条款"])[0], top_k=20, where={"page": 2})

    assert hits
    assert all(h.chunk.page == 2 for h in hits)


def test_search_in_pages_range(tmp_path) -> None:
    """页码区间过滤（含两端）。"""
    store, embedder = _filter_fixture(tmp_path)
    hits = store.search_in_pages(
        embedder.embed_query(["条款"])[0], page_from=2, page_to=3, top_k=20
    )

    assert hits
    assert all(2 <= (h.chunk.page or 0) <= 3 for h in hits)
    assert len(hits) == 6, f"3 份文档 × 2 页 = 6 条，实际 {len(hits)}"


def test_search_in_pages_combines_with_where(tmp_path) -> None:
    """页码区间 + source 同时过滤。"""
    store, embedder = _filter_fixture(tmp_path)
    hits = store.search_in_pages(
        embedder.embed_query(["条款"])[0],
        page_from=1,
        page_to=2,
        top_k=20,
        where={"source": "合同A.pdf"},
    )

    assert hits
    assert all(h.chunk.source == "合同A.pdf" for h in hits)
    assert all((h.chunk.page or 0) in (1, 2) for h in hits)
    assert len(hits) == 2


def test_search_in_pages_rejects_reversed_range(tmp_path) -> None:
    """`page_from > page_to` 必须抛，不能静默返回空 ——
    空结果是"这段没命中"，参数颠倒是"我写错了"，两者必须区分。"""
    store, embedder = _filter_fixture(tmp_path)
    with pytest.raises(ValueError, match="page_from"):
        store.search_in_pages(embedder.embed_query(["x"])[0], page_from=5, page_to=1)


def test_where_rejects_nested_values(tmp_path) -> None:
    """`where` 的值必须是扁平标量。在入口拦住比让 Chroma 抛内部错误好定位。"""
    store, embedder = _filter_fixture(tmp_path)
    with pytest.raises(TypeError, match="扁平标量|str/int"):
        store.search(embedder.embed_query(["x"])[0], where={"k": {"nested": 1}})  # type: ignore[dict-item]


def test_where_on_missing_field_returns_empty(tmp_path) -> None:
    """查不存在的字段返回空列表，不报错（Chroma 的行为，我们依赖它）。"""
    store, embedder = _filter_fixture(tmp_path)
    hits = store.search(embedder.embed_query(["条款"])[0], where={"nonexistent": "x"})
    assert hits == []


# ============================================================ 5. 两个实现的一致性


def test_both_stores_agree_on_score_direction(tmp_path) -> None:
    """**内存实现与 Chroma 必须给出同一组分数和同一个顺序。**

    防的是"内存里测通过、Chroma 上行为不同"。分数口径不同（一个越大越像、
    一个越小越像）时**不会报错**，只会让阈值逻辑在换库那天突然全错。

    **用 `tmp_path` 而不是 `tempfile.TemporaryDirectory()`**：Windows 上
    后者的清理会失败（`PermissionError: [WinError 32] 另一个程序正在使用此文件`），
    因为 Chroma 的 sqlite 句柄没释放。`tmp_path` 由 pytest 管理，
    失败也只是留下临时目录，不会让一条本来通过的测试变成 ERROR。
    """
    embedder = HashingEmbedder(dimension=_DIM)
    chunks = [
        _chunk("违约金按日万分之五计算", page=1, source="A.pdf", index=0),
        _chunk("质保期 12 个月", page=2, source="A.pdf", index=1),
        _chunk("杭州仲裁委员会", page=3, source="B.pdf", index=0),
    ]
    vectors = embedder.embed([c.text for c in chunks])
    query = embedder.embed_query(["违约金"])[0]

    memory = InMemoryVectorStore()
    memory.upsert(chunks, vectors)
    memory_hits = memory.search(query, top_k=3)

    chroma = ChromaStore(tmp_path / "chroma", collection_name="agree_test")
    chroma.upsert(chunks, vectors)
    chroma_hits = chroma.search(query, top_k=3)

    assert len(memory_hits) == len(chroma_hits)
    for memory_hit, chroma_hit in zip(memory_hits, chroma_hits, strict=True):
        assert memory_hit.chunk.text == chroma_hit.chunk.text, "两个 store 的排序不同"
        assert memory_hit.score == pytest.approx(chroma_hit.score, abs=1e-6), (
            f"分数不一致：memory={memory_hit.score} chroma={chroma_hit.score}"
        )


def test_both_stores_implement_protocol(tmp_path) -> None:
    """两个都必须过 `VectorStore` 的 runtime_checkable 检查。"""
    assert isinstance(InMemoryVectorStore(), VectorStore)
    assert isinstance(_store(tmp_path), VectorStore)


def test_memory_store_supports_same_filters() -> None:
    """内存实现也要支持 where / search_in_pages —— 否则一半测试只能跑在 Chroma 上。

    两个实现的过滤结果必须相同，否则"内存里测通过、Chroma 上过滤不到"
    会变成一类无法复现的 bug。
    """
    embedder = HashingEmbedder(dimension=_DIM)
    chunks = [
        _chunk("A 第 1 页", page=1, source="A.pdf", index=0),
        _chunk("B 第 2 页", page=2, source="B.pdf", index=1),
    ]
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    query = embedder.embed_query(["页"])[0]

    assert len(store.search(query, where={"source": "A.pdf"})) == 1
    assert len(store.search_in_pages(query, page_from=2, page_to=3)) == 1
    with pytest.raises(ValueError, match="page_from"):
        store.search_in_pages(query, page_from=3, page_to=1)


# ============================================================ 6. 与索引链路接起来


def test_indexing_into_chroma_via_pipeline(tmp_path) -> None:
    """`RagPipeline(store=ChromaStore(...))` 能跑通 —— 换库只改装配那一行。

    **用 `tmp_path` 而不是 `tempfile.TemporaryDirectory()`**：
    Windows 上 Chroma 的 sqlite 句柄不释放，上下文管理器退出时清理会
    `PermissionError: [WinError 32]`（实测）。
    """
    from app.rag.pipeline import RagPipeline

    store = ChromaStore(tmp_path / "chroma", collection_name="pipeline_test")
    pipeline = RagPipeline(embedder=HashingEmbedder(dimension=_DIM), store=store, top_k=2)

    report = pipeline.index_text("违约金按日万分之五计算。", source="合同")
    assert report.ok, f"索引失败：{report.to_dict()}"
    assert store.count() == 1

    hits = pipeline.retrieve("违约金").hits
    assert hits and hits[0].chunk.source == "合同"


def test_document_roundtrip_keeps_page_attribution(tmp_path) -> None:
    """**页码必须活着穿过 Chroma 的 metadata 往返**。

    页码在 `to_payload()` 里被压成 int（None → -1），
    在 `_query()` 里被 `restore_location` 还原。这条测的是那条往返没丢信息，
    包括"无页"这个状态。
    """
    store = _store(tmp_path)
    embedder = HashingEmbedder(dimension=_DIM)
    chunks = [
        _chunk("有页码的块", page=7, source="A.pdf"),
        _chunk("无页码的块", page=None, source="notes.md"),
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))

    query = embedder.embed_query(["块"])[0]
    by_text = {h.chunk.text: h.chunk for h in store.search(query, top_k=5)}

    assert by_text["有页码的块"].page == 7
    assert by_text["有页码的块"].pages == (7,)
    assert by_text["无页码的块"].page is None, "无页码的块被还原成了别的值"
    assert by_text["无页码的块"].source == "notes.md"


def test_pages_tuple_roundtrip(tmp_path) -> None:
    """跨页块的 `pages` 元组也要穿过 metadata 往返。"""
    store = _store(tmp_path)
    embedder = HashingEmbedder(dimension=_DIM)
    crosspage = Chunk(text="跨页的一段话", page=6, source="A.pdf", pages=(6, 7), index=0)
    store.upsert([crosspage], embedder.embed([crosspage.text]))

    hit = store.search(embedder.embed_query(["跨页"])[0], top_k=1)[0]
    assert hit.chunk.pages == (6, 7), f"pages 丢了：{hit.chunk.pages}"


def test_content_id_survives_roundtrip(tmp_path) -> None:
    """内容派生的 `id` 要能在库里读到（引用去重靠它）。"""
    store = _store(tmp_path)
    embedder = HashingEmbedder(dimension=_DIM)
    chunk = _chunk("带 id 的块", page=3, source="A.pdf")
    store.upsert([chunk], embedder.embed([chunk.text]))

    hit = store.search(embedder.embed_query(["带 id"])[0], top_k=1)[0]
    assert hit.chunk.id == chunk.id, "从库里读回来的 id 与写入时不同"


# ============================================================ 7. 误用防线


def test_auto_embedding_is_blocked(tmp_path) -> None:
    """Chroma 想自己算 embedding 时必须炸。

    不设守卫的话，`query_texts=` 这种写法会让 Chroma 用**它自带的模型**算向量，
    而库里存的是 bge 的向量 —— 两份向量在不同空间里，检索结果无法解释，
    且**不会报错**。
    """
    store = _store(tmp_path)
    embedder = HashingEmbedder(dimension=_DIM)
    store.upsert([_chunk("x", page=1)], [embedder.embed(["x"])[0]])

    with pytest.raises(RuntimeError, match="试图自己算 embedding"):
        store._collection.query(query_texts=["x"], n_results=1)


def test_upsert_rejects_length_mismatch(tmp_path) -> None:
    """chunks 与 vectors 数量不一致必须炸，不能截断。"""
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="数量不一致"):
        store.upsert([_chunk("一", page=1)], [])


def test_upsert_rejects_duplicate_store_ids(tmp_path) -> None:
    """同一 (doc, page, chunk_index) 出现两次必须炸。

    这是**真实的错误**（切分器给了重复 index），
    保留哪个都是猜 —— 所以宁可报错。
    """
    store = _store(tmp_path)
    embedder = HashingEmbedder(dimension=_DIM)
    duplicates = [
        _chunk("第一次", page=1, source="A.pdf", index=0),
        _chunk("第二次", page=1, source="A.pdf", index=0),
    ]
    assert duplicates[0].store_id == duplicates[1].store_id
    with pytest.raises(ValueError, match="store_id 重复"):
        store.upsert(duplicates, embedder.embed([c.text for c in duplicates]))


def test_empty_upsert_is_noop(tmp_path) -> None:
    store = _store(tmp_path)
    assert store.upsert([], []) == 0
    assert store.count() == 0


def test_search_on_empty_store_returns_empty(tmp_path) -> None:
    """空库检索返回空列表，不抛 —— 第一次提问时库本来就可能是空的。"""
    store = _store(tmp_path)
    assert store.search([0.1] * _DIM, top_k=3) == []
    assert store.search_in_pages([0.1] * _DIM, page_from=1, page_to=5) == []


def test_store_id_is_human_readable() -> None:
    """`store_id` 格式 `{doc_id}_{page}_{chunk_index}`，打开库就能看出是哪块。"""
    chunk = Chunk(text="x", page=7, source="合同.pdf", index=3)
    parts = chunk.store_id.split("_")
    assert len(parts) == 3
    assert parts[1] == "7"
    assert parts[2] == "3"
    assert len(parts[0]) == 10

    # 无页码时用 -1，与 to_payload 的编码一致
    assert Chunk(text="x", page=None, source="n.md", index=0).store_id.endswith("_-1_0")


def test_doc_id_is_stable_per_source() -> None:
    """同一个 source 的不同块共用 `doc_id`；不同 source 不共用。"""
    a1 = Chunk(text="块一", page=1, source="A.pdf", index=0)
    a2 = Chunk(text="块二", page=2, source="A.pdf", index=1)
    b1 = Chunk(text="块一", page=1, source="B.pdf", index=0)

    assert a1.doc_id == a2.doc_id
    assert a1.doc_id != b1.doc_id


def test_store_id_survives_content_change() -> None:
    """内容变了但位置没变 → `store_id` 不变，`id` 变。

    这正是两者分工的证据：`store_id` 管"覆盖哪个位置"，`id` 管"是不是同一内容"。
    """
    old = Chunk(text="旧内容", page=7, source="A.pdf", index=2)
    new = Chunk(text="新内容", page=7, source="A.pdf", index=2)
    assert old.store_id == new.store_id
    assert old.id != new.id


def test_default_collection_name_is_valid() -> None:
    """collection 名必须满足 Chroma 的规则：3-512 字符、首尾字母数字。

    实测 `"t"` 会被直接拒绝（`ValueError: Expected a name containing 3-512 characters`）。
    """
    assert 3 <= len(DEFAULT_COLLECTION) <= 512
    assert DEFAULT_COLLECTION[0].isalnum() and DEFAULT_COLLECTION[-1].isalnum()


def test_get_store_by_name() -> None:
    from app.rag.stores import describe_implementations, get_store

    assert isinstance(get_store("memory"), InMemoryVectorStore)
    assert "chroma" in describe_implementations()
    with pytest.raises(ValueError, match="未知的向量库"):
        get_store("milvus")


def test_unused_import_guard() -> None:
    """`Document` 在这个文件里只用于类型感知（roundtrip 测试里没直接用），
    但保留导入是为了让读者看到"Chroma 存的是 Chunk，Chunk 来自 Document"这条链路。
    这里显式用一下，避免 F401 与"忘了删"混淆。"""
    assert Document(source="a.pdf", page=1, text="x").page == 1


# ============================================================ 8. 回归：index_text 的覆盖语义


def test_index_text_default_overwrites_same_source(tmp_path) -> None:
    """**默认是"这段就是该 source 的全部内容"**：再入一次会替换掉旧的。

    这是"重新上传同一份文件"的正确语义 —— 旧版本必须消失，
    否则"改了合同再上传"会让新旧两版同时被检索到。
    """
    from app.rag.pipeline import RagPipeline

    store = ChromaStore(tmp_path / "chroma", collection_name="overwrite_test")
    pipeline = RagPipeline(embedder=HashingEmbedder(dimension=_DIM), store=store, top_k=5)

    pipeline.index_text("旧版本的违约金条款", source="合同")
    assert store.count() == 1

    pipeline.index_text("新版本的违约金条款", source="合同")
    assert store.count() == 1, "同一 source 重新索引应当替换，不是叠加"

    hits = pipeline.retrieve("违约金").hits
    assert hits[0].chunk.text == "新版本的违约金条款"


def test_index_text_append_accumulates(tmp_path) -> None:
    """`append=True` 时追加多个块。

    **这个开关来自一个真实 bug**：原来 `index_text` 没有 append，
    而三条不同条款共用同一个 `source` 时，后一条会**静默覆盖**前一条 ——
    连续入 3 条，库里只剩 1 条，而每次 `IndexReport.ok` 都是 `True`。

    发现它的是一次"看起来与 Chroma 无关"的 bge 端到端测试：
    它想入 3 条再检索，结果命中不对。**换 store 时跑全套测试的价值就在这里。**
    """
    from app.rag.pipeline import RagPipeline

    store = ChromaStore(tmp_path / "chroma", collection_name="append_test")
    pipeline = RagPipeline(embedder=HashingEmbedder(dimension=_DIM), store=store, top_k=5)

    for text in ("第一条 付款条款", "第二条 违约条款", "第三条 质保条款"):
        report = pipeline.index_text(text, source="合同", append=True)
        assert report.ok

    assert store.count() == 3, f"append=True 应当累积 3 条，实际 {store.count()}"


def test_index_report_ok_does_not_guarantee_accumulation(tmp_path) -> None:
    """**`IndexReport.ok` 只看"这一批写成功了"，不保证"历史数据还在"。**

    这条记录一个认知：`ok=True` 与"库里有 N 条"是两件事。
    排查"数据怎么少了"时必须 `count()`，不能只看报告。
    """
    from app.rag.pipeline import RagPipeline

    store = ChromaStore(tmp_path / "chroma", collection_name="ok_test")
    pipeline = RagPipeline(embedder=HashingEmbedder(dimension=_DIM), store=store)

    first = pipeline.index_text("第一条", source="合同")
    second = pipeline.index_text("第二条", source="合同")

    assert first.ok is True
    assert second.ok is True
    # 两次都"成功"，但库里只有一条 —— 覆盖语义下这是正确的，但容易被误读成"累积"
    assert store.count() == 1
