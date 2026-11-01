"""`app/rag/filters.py` 的契约测试：**重点是"写错条件不能静默返回空"**。

这个模块的存在理由就是一个具体的静默故障（见它的模块 docstring）：
`where` 写错时向量库返回空列表，与"确实没有匹配"无法区分。
所以这里每条测试都对着一种"写错了但看起来正常"的情形。
"""

from __future__ import annotations

import pytest

from app.rag import Document
from app.rag.chunking import ChunkingConfig, build_strategy
from app.rag.embedding import HashingEmbedder
from app.rag.filters import (
    FilterKeyNotPopulated,
    UnknownFilterKey,
    UnknownFilterValue,
    build_schema,
    guard_where,
    resolve_section_filter,
    schema_from_store,
)
from app.rag.schema import Chunk
from app.rag.stores.memory import InMemoryVectorStore

DIM = 32


def _chunk(text: str, *, source: str, index: int, metadata: dict | None = None) -> Chunk:
    return Chunk(
        text=text,
        page=None,
        source=source,
        index=index,
        metadata=metadata or {"block": "text"},
    )


def _store_with(chunks: list[Chunk]) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    embedder = HashingEmbedder(dimension=DIM)
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    return store


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        _chunk("付款计划：分期支付。", source="合同A.md", index=0),
        _chunk("验收标准：书面确认。", source="合同A.md", index=1),
        _chunk("违约条款：按日万分之五。", source="合同B.md", index=0, metadata={"block": "text"}),
        _chunk(
            "| 金额 | 期次 |\n| --- | --- |",
            source="合同B.md",
            index=1,
            metadata={"block": "table"},
        ),
    ]


@pytest.fixture
def schema(chunks: list[Chunk]):  # noqa: ANN201
    return build_schema(chunks)


# --------------------------------------------------------------------------- schema


def test_schema_lists_actual_keys(schema) -> None:  # noqa: ANN001
    """schema 只包含库里真实写过的键 —— 这正是"拼错键名"能被发现的前提。"""
    assert "source" in schema.keys
    assert "block" in schema.keys
    # 这个键从没写过
    assert "section_path" not in schema.keys


def test_schema_counts_core_keys_from_payload(chunks: list[Chunk]) -> None:  # noqa: ANN001
    """`Chunk.to_payload()` 统一写入的那几个键必须都在 schema 里。

    它们是任何策略、任何 loader 都会有的，所以按它们过滤永远安全。
    """
    schema = build_schema(chunks)
    for key in ("source", "page", "pages", "chunk_index", "content_id"):
        assert key in schema.keys, f"{key} 是 to_payload 写入的核心键，schema 里必须有"


def test_schema_from_store_matches_build_schema(chunks: list[Chunk]) -> None:  # noqa: ANN001
    """两条入口必须给出同一份 schema。

    漂移的表现是"用 chunks 算的 schema 通过校验，但 store 那边不认" ——
    而这种不一致只在某个调用路径上暴露。
    """
    store = _store_with(chunks)
    from_chunks = build_schema(chunks)
    from_store = schema_from_store(store)
    assert set(from_store.keys) == set(from_chunks.keys)
    for key in from_chunks.keys:
        assert from_store.values[key] == from_chunks.values[key]


def test_store_without_describe_metadata_raises() -> None:
    """不支持的 store 要明确说，不能静默返回空 schema。

    空 schema 会让**所有**过滤都抛 `UnknownFilterKey`，
    报错信息指向"键不存在"，而真正的原因是"这个 store 不支持"。
    """

    class DumbStore:
        pass

    with pytest.raises(TypeError, match="describe_metadata"):
        schema_from_store(DumbStore())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- 三种失败


def test_unknown_key_raises_with_available_keys(schema) -> None:  # noqa: ANN001
    """键名拼错 → 抛，且错误信息里列出可用键。

    **这是本模块最重要的一条**：没有它，`{"blockk": "table"}` 返回空列表，
    调用方会以为"库里没有表格"。
    """
    with pytest.raises(UnknownFilterKey) as exc_info:
        guard_where(schema, {"blockk": "table"})
    message = str(exc_info.value)
    assert "blockk" in message
    assert "block" in message  # 可用键里要有正确的那个


def test_unknown_key_suggests_close_match(schema) -> None:  # noqa: ANN001
    """拼错一个字母时，错误信息直接给出正确的键。"""
    with pytest.raises(UnknownFilterKey) as exc_info:
        guard_where(schema, {"sourc": "合同A.md"})
    assert "source" in str(exc_info.value)


def test_unknown_value_raises_with_candidates(schema) -> None:  # noqa: ANN001
    """文档名写错 → 抛，且给出接近的候选。

    没有它时，`source="合同AA.md"` 返回 0 条，用户会以为"这份文档没导入"。
    """
    with pytest.raises(UnknownFilterValue) as exc_info:
        guard_where(schema, {"source": "合同AA.md"})
    message = str(exc_info.value)
    assert "合同AA.md" in message
    assert "合同A.md" in message or "合同B.md" in message


def test_non_populated_key_names_the_missing_strategy() -> None:
    """字段存在但全为空（`section_path` 在 recursive 下）→ 说清需要哪种策略。

    这一条对应的不是笔误，而是**真实的接口陷阱**：默认策略 recursive
    根本不写 `section_path`，所以任何按章节过滤的调用都静默返回空。
    """
    chunks = [
        # 模拟 heading 策略：键存在，但这一批全是空值
        _chunk("正文一", source="a.md", index=0, metadata={"block": "text", "section_path": ""}),
    ]
    schema = build_schema(chunks)
    assert "section_path" in schema.keys  # 键在
    with pytest.raises(FilterKeyNotPopulated) as exc_info:
        guard_where(schema, {"section_path": "架构"})
    message = str(exc_info.value)
    assert "heading" in message


def test_empty_where_is_allowed(schema) -> None:  # noqa: ANN001
    """不传过滤是正常的，不是错误。"""
    assert guard_where(schema, None) == {}
    assert guard_where(schema, {}) == {}


def test_valid_where_passes_through_unchanged(schema) -> None:  # noqa: ANN001
    """合法的过滤条件原样返回（**不改类型**）。

    改成 str 会让 `where={"page": "3"}` 传给 Chroma 时匹配不到 int 型的 3 ——
    而校验通过、检索为空，正是最难查的那种组合。
    """
    result = guard_where(schema, {"source": "合同A.md", "block": "table"})
    assert result == {"source": "合同A.md", "block": "table"}


def test_integer_value_accepted_for_integer_metadata(chunks: list[Chunk]) -> None:  # noqa: ANN001
    """`page` 在库里是 int（`-1` 表示 None），传 int 要能过校验。"""
    schema = build_schema(chunks)
    assert guard_where(schema, {"page": -1}) == {"page": -1}


# --------------------------------------------------------------------------- 章节过滤


def test_section_filter_requires_heading_strategy(schema) -> None:  # noqa: ANN001
    """recursive 下按章节过滤要立刻抛，并指明需要 heading。"""
    with pytest.raises(FilterKeyNotPopulated, match="heading"):
        resolve_section_filter(schema, section="架构", strategy="recursive")


def test_section_filter_picks_shortest_matching_path() -> None:
    """多个路径都包含关键词时取**最短**的那条（最靠近"某一章"）。"""
    chunks = [
        _chunk(
            "正文",
            source="a.md",
            index=0,
            metadata={"block": "text", "section_path": "环境搭建 > 架构"},
        ),
        _chunk(
            "正文二",
            source="a.md",
            index=1,
            metadata={"block": "text", "section_path": "环境搭建 > 架构 > 目录"},
        ),
    ]
    schema = build_schema(chunks)
    resolved = resolve_section_filter(schema, section="架构", strategy="heading")
    assert resolved == {"section_path": "环境搭建 > 架构"}


def test_section_filter_rejects_unknown_section() -> None:
    chunks = [
        _chunk("正文", source="a.md", index=0, metadata={"block": "text", "section_path": "架构"}),
    ]
    schema = build_schema(chunks)
    with pytest.raises(UnknownFilterValue, match="架构"):
        resolve_section_filter(schema, section="安全", strategy="heading")


# --------------------------------------------------------------------------- 真实链路


def test_filter_actually_narrows_real_retrieval(chunks: list[Chunk]) -> None:  # noqa: ANN001
    """过滤在真实 store 上确实生效（不只是校验通过）。"""
    store = _store_with(chunks)
    embedder = HashingEmbedder(dimension=DIM)
    query = embedder.embed_query(["付款"])[0]
    schema = schema_from_store(store)

    everything = store.search(query, top_k=10)
    only_a = store.search(query, top_k=10, where=guard_where(schema, {"source": "合同A.md"}))
    assert len(only_a) < len(everything)
    assert all(hit.chunk.source == "合同A.md" for hit in only_a)


def test_table_filter_selects_only_tables(chunks: list[Chunk]) -> None:  # noqa: ANN001
    """`block=table` 只返回表格块 —— 这是"按内容类型过滤"的验收形式。"""
    store = _store_with(chunks)
    embedder = HashingEmbedder(dimension=DIM)
    query = embedder.embed_query(["金额"])[0]
    schema = schema_from_store(store)

    tables = store.search(query, top_k=10, where=guard_where(schema, {"block": "table"}))
    assert tables
    assert all(hit.chunk.metadata.get("block") == "table" for hit in tables)


# --------------------------------------------------------------------------- 与切分策略的联动


def test_section_path_absent_under_default_strategy() -> None:
    """**这条是今天最重要的一个事实**：默认策略切出来的块没有 `section_path`。

    它把"按章节过滤在默认配置下不可用"从'我记得好像是这样'
    变成了一条可执行的断言。要按章节过滤，索引必须改用 heading。
    """
    doc = Document(
        source="x.md",
        page=None,
        text="# 第一章\n\n正文内容一。\n\n## 1.1 小节\n\n正文内容二。",
        metadata={"block": "text"},
    )
    cfg = ChunkingConfig(chunk_chars=500, overlap_chars=0)

    recursive_chunks = build_strategy("recursive", cfg=cfg).chunk([doc], cfg)
    assert all("section_path" not in c.metadata for c in recursive_chunks)

    heading_chunks = build_strategy("heading", cfg=cfg).chunk([doc], cfg)
    assert any("section_path" in c.metadata for c in heading_chunks)


def test_heading_schema_accepts_section_filter() -> None:
    """heading 策略的索引上，按章节过滤能过校验。"""
    doc = Document(
        source="x.md",
        page=None,
        text="# 第一章\n\n正文内容一。\n\n## 1.1 小节\n\n正文内容二。",
        metadata={"block": "text"},
    )
    cfg = ChunkingConfig(chunk_chars=500, overlap_chars=0)
    chunks = build_strategy("heading", cfg=cfg).chunk([doc], cfg)
    schema = build_schema(chunks)
    resolved = resolve_section_filter(schema, section="第一章", strategy="heading")
    assert "第一章" in str(resolved["section_path"])
