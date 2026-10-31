"""RAG 问答链路测试：切分、metadata 继承、引用解析、拒答。

## 这个文件重点测四件事

### 一、中文切分的边界质量（对比 LangChain 的实测差异）

`test_chunk_boundaries_end_at_sentence_marks` 断言每块以句末标点收尾。
**这条在 LangChain 上会失败** —— 实测它切出 `…支付合同价**程**`（4/5 块是半个词）。
差异的根源在合并策略：LangChain 为了填满 `chunk_size` 会切割原子片，
本项目只合并完整原子片、装不下就开新块。

### 二、metadata 继承（页码抽查）

切分最容易丢的就是 `page` —— 它一旦在这里丢了，**下游再也补不回来**。
`test_every_chunk_inherits_its_group_fields` 逐条核对 source/page/doc_id。

### 三、引用解析的三种情况

| 输入 | 期望 |
| --- | --- |
| `"事实 [1]。"` | citations=[1] |
| `"事实 [1][2]。"` | citations=[1,2]，按编号升序 |
| `"事实 [1][9]。"`（9 不存在） | citations=[1]，**invalid_refs=[9]** |

第三条是重点：模型引用了不存在的编号说明它在编，**必须可见**，不能静默丢弃。

### 四、拒答是代码的决定，不是模型的

`test_refuses_below_threshold` 用 `_ExplodingClient`（被调用就抛）证明
拒答路径**根本没碰模型**。这比"断言话术对不对"更有力 ——
话术可以改，但"没调模型"是一条结构性事实。
"""

from __future__ import annotations

import pytest

from app.rag import Document
from app.rag.embedding import HashingEmbedder
from app.rag.embedding.vector import cosine
from app.rag.generation import (
    DEFAULT_MIN_SCORE,
    REFUSAL_TEXT,
    build_context,
    extract_refs,
    generate_answer,
    resolve_citations,
)
from app.rag.schema import Chunk
from app.rag.splitters import ChunkSize, DelimiterSplitter, RecursiveSplitter

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
    def __init__(self, reply: str) -> None:
        self.calls: list[dict] = []
        self.reply = reply

    async def create(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(kwargs)
        return _Resp(self.reply)


class FakeClient:
    def __init__(self, reply: str = "违约金按日万分之五计算 [1]。") -> None:
        self.completions = _Completions(reply)
        self.chat = type("Chat", (), {"completions": self.completions})()


class _ExplodingClient:
    """被调用就抛。用来证明"拒答路径没碰模型"。"""

    @property
    def chat(self):  # noqa: ANN201
        raise AssertionError("拒答路径不该调用模型 —— 说明闸门位置错了")


def _chunk(text: str, *, page: int = 1, source: str = "合同.pdf", index: int = 0) -> Chunk:
    return Chunk(text=text, page=page, source=source, index=index)


# ============================================================ 1. 中文切分


# 2128 字中文合同，含 5 条条款。用于对比两种切分的边界质量。
_LONG_CN = (
    "第五条 付款计划。甲方按下列里程碑分期支付合同价款，验收合格后十五个工作日内付款。"
    "付款方式为银行转账，账户信息以合同附件为准。乙方须在收款前开具增值税专用发票。"
    "第六条 违约责任。乙方逾期交付的，按日万分之五支付违约金，累计不超过合同总额的百分之十。"
    "违约金可直接从应付款项中扣除。甲方逾期付款的，按日万分之三支付逾期利息。"
    "第七条 验收标准。每一期交付物须经甲方书面确认验收合格，方可进入付款流程。"
    "验收不通过的，乙方须在十个工作日内完成整改并重新提交验收申请。"
    "第八条 知识产权。本项目产生的软件著作权归甲方所有，乙方保留通用组件的复用权。"
    "第九条 保密义务。双方对合同履行中知悉的商业秘密负有保密义务，保密期三年。"
) * 7


def _split(text: str, *, size: int = 500, overlap: int = 50) -> list[Chunk]:
    return DelimiterSplitter(ChunkSize(chunk_chars=size, overlap_chars=overlap)).split(
        [Document(source="合同.pdf", page=1, text=text)]
    )


def test_chinese_text_actually_splits() -> None:
    """中文必须真的被切开。

    **这条不是废话**：LangChain 的默认 `separators` 是 `["\\n\\n", "\\n", " ", ""]` ——
    全英文。用它切中文，等于只在空格处断，而中文段落几乎没有空格，
    结果是**整篇挤成一块**（实测：304 字的中文段落用默认分隔符切，一块都不切，
    直到 2128 字时才因为超过 chunk_size 被迫硬切）。
    """
    chunks = _split(_LONG_CN)
    assert len(chunks) > 3, f"2128 字只切出 {len(chunks)} 块，可能没按中文标点切"
    for chunk in chunks:
        assert len(chunk.text) <= 500


def test_chunk_boundaries_end_at_sentence_marks() -> None:
    """**边界质量的硬指标**：每块以句末标点收尾。

    实测对比（同一份 2128 字语料、同样 500/50 参数）：

    | | 块尾字符 | 全落在句末标点 |
    | --- | --- | :---: |
    | LangChain | `程` `款` `准` `划` `。` | ❌ 4/5 是半个词 |
    | 本项目 | `。` `。` `。` `。` `。` | ✅ |

    差异根源：LangChain 为了填满 `chunk_size` 会**切割原子片**；
    本项目只合并完整原子片、装不下就开新块。代价是块略短（实测平均 470/500），
    换来的是"读起来是完整句子"。
    """
    chunks = _split(_LONG_CN)
    for chunk in chunks:
        assert chunk.text.rstrip()[-1] in "。！？；", f"块尾不是句末标点：{chunk.text[-20:]!r}"


def test_chinese_punctuation_separators_are_used() -> None:
    """中文分隔符必须在优先级表里，且顺序是"段落 → 句末 → 逗号"。

    顺序错了不会报错，只会让块边界变差 —— 例如把 `，` 排在 `。` 前面，
    就会优先在逗号处断，块尾变成半句话。
    """
    from app.rag.splitters import DEFAULT_SEPARATORS

    assert "\n\n" in DEFAULT_SEPARATORS
    assert "。" in DEFAULT_SEPARATORS
    assert "！" in DEFAULT_SEPARATORS
    assert "？" in DEFAULT_SEPARATORS
    assert "，" in DEFAULT_SEPARATORS
    # 段落边界必须优先于句末标点（否则同一节会被句号切散）
    assert DEFAULT_SEPARATORS.index("\n\n") < DEFAULT_SEPARATORS.index("。")
    assert DEFAULT_SEPARATORS.index("。") < DEFAULT_SEPARATORS.index("，")


def test_overlap_makes_neighbour_chunks_share_text() -> None:
    """**必答题的落地检查**：重叠要真的产生。

    overlap 解决的是"关键信息正好被切在边界上断开"。前提是它**真的生效** ——
    如果实现里算了 tail 却没拼上，配置看起来对、行为完全没有，
    而这种 bug 不会报错。
    """
    chunks = _split(_LONG_CN, overlap=50)
    assert len(chunks) > 1
    for left, right in zip(chunks, chunks[1:], strict=False):
        # 前一块的尾部应当出现在后一块的开头
        tail = left.text[-30:]
        assert tail in right.text, f"重叠没生效：{tail!r} 不在下一块开头"


def test_overlap_zero_means_no_redundancy() -> None:
    """`overlap_chars=0` 时相邻块不共享文字 —— 用来量化"重叠的代价"。"""
    with_overlap = _split(_LONG_CN, overlap=50)
    without = _split(_LONG_CN, overlap=0)

    total_with = sum(len(c.text) for c in with_overlap)
    total_without = sum(len(c.text) for c in without)
    assert total_with > total_without, "有重叠的总字数应当更多（这就是索引冗余）"


def test_splitter_alias_is_the_same_class() -> None:
    """`RecursiveSplitter` 是 `DelimiterSplitter` 的别名，**不是子类**。

    子类会让 `isinstance` 与 `name` 出现两种答案，而它们本该是同一件事。
    """
    assert RecursiveSplitter is DelimiterSplitter


def test_legacy_import_path_still_works() -> None:
    """旧 import 路径必须还能用（周三的代码/文档引用的是它）。"""
    from app.rag.splitters.recursive import RecursiveSplitter as Legacy

    assert Legacy is DelimiterSplitter


# ============================================================ 2. metadata 继承


def test_every_chunk_inherits_its_group_fields() -> None:
    """**页码抽查**：每个 chunk 都要带 source / page / pages / doc_id。

    这是切分最核心的职责 —— `page` 一旦在这里丢了，下游再也补不回来
    （见 `schema.py` 模块 docstring）。所以逐字段核对，不是"抽查有值就算过"。
    """
    doc = Document(
        source="合同A.pdf", page=7, text=_LONG_CN, pages=(7,), metadata={"block": "text"}
    )
    chunks = DelimiterSplitter(ChunkSize(chunk_chars=500, overlap_chars=50)).split([doc])

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.source == "合同A.pdf"
        assert chunk.page == 7, "页码丢了"
        assert chunk.pages == (7,), "覆盖页码丢了"
        # `doc_id` 由 source 派生（10 位短哈希），不来自 Document —— 即使 loader 没填也一定有值
        assert len(chunk.doc_id) == 10
        assert chunk.doc_id == chunks[0].doc_id, "同一文档的不同块必须共用 doc_id"
        assert chunk.metadata.get("block") == "text"
        assert chunk.metadata.get("splitter") == "delimiter"


def test_chunks_from_same_page_share_page_value() -> None:
    """同一页切出的多个 chunk **共享同一个 `page`** —— 它们确实来自同一页。

    "这一块覆盖了哪几页"要看 `pages`，不是 `page`。
    """
    doc = Document(source="a.pdf", page=3, text="短句。" * 400, pages=(3,))
    chunks = DelimiterSplitter(ChunkSize(chunk_chars=200, overlap_chars=20)).split([doc])
    assert len(chunks) > 1
    assert {c.page for c in chunks} == {3}


def test_crosspage_document_propagates_all_pages() -> None:
    """跨页段落的 `pages=(6,7)` 要传给每一块。

    只记 `page=6` 是"半个真话"：用户点进 p6 找不到后半截，
    点进 p7 又找不到前半截 —— 这正是 demo 现场最容易被点穿的地方。
    """
    doc = Document(source="a.pdf", page=6, text=_LONG_CN, pages=(6, 7))
    chunks = DelimiterSplitter(ChunkSize(chunk_chars=500, overlap_chars=50)).split([doc])
    for chunk in chunks:
        assert chunk.page == 6
        assert chunk.pages == (6, 7), f"跨页范围丢了：{chunk.pages}"


def test_table_is_not_split() -> None:
    """表格整张一块，哪怕超预算 —— 切碎后行列关系无法还原。"""
    wide_table = "| 里程碑 | 时间 | 金额 |\n| --- | --- | --- |\n" + "\n".join(
        f"| 第{i}期 | 2026-{i:02d}-01 | {i * 1000} |" for i in range(1, 60)
    )
    doc = Document(source="t.pdf", page=1, text=wide_table, metadata={"block": "table"})
    chunks = DelimiterSplitter(ChunkSize(chunk_chars=200, overlap_chars=20)).split([doc])
    assert len(chunks) == 1, "表格被切开了"
    assert chunks[0].text == wide_table.strip()


def test_empty_document_produces_no_chunk() -> None:
    assert DelimiterSplitter().split([Document(source="a.md", page=None, text="   ")]) == []


def test_no_fake_page_for_unpaged_source() -> None:
    """无页码来源切出的块 `page=None`、`pages=()` —— 不伪造页。"""
    chunks = DelimiterSplitter().split([Document(source="notes.md", page=None, text=_LONG_CN)])
    for chunk in chunks:
        assert chunk.page is None
        assert chunk.pages == ()


# ============================================================ 3. 引用抽取


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("违约金按日万分之五计算 [1]。", [1]),
        ("事实一 [1]，事实二 [2]。", [1, 2]),
        ("连续引用 [1][2][3]。", [1, 2, 3]),
        ("逗号写法 [1,2]。", [1, 2]),
        ("全角逗号 [1，2]。", [1, 2]),
        ("区间写法 [1-3]。", [1, 2, 3]),
        ("重复引用 [1][1]。", [1]),
        ("无序引用 [2][1]。", [2, 1]),
        ("没有引用。", []),
    ],
)
def test_extract_refs(text: str, expected: list[int]) -> None:
    """`[n]` 的各种写法都要认。

    **顺序规则**：按出现位置去重保序，不是升序 ——
    正文里先写 `[2]` 后写 `[1]` 是正常的行文，抽取阶段不该重排
    （重排发生在 `resolve_citations`，那里是引用列表，才需要升序）。
    """
    assert extract_refs(text) == expected


def test_extract_refs_ignores_absurd_range() -> None:
    """`[1-9999]` 不展开 —— 模型发疯时不该把内存打满。"""
    assert extract_refs("看 [1-9999] 和 [2]。") == [2]


def test_resolve_citations_maps_to_source_and_page() -> None:
    """编号 → Citation → source + page。这是"带引用"的实质。"""
    chunks = [
        _chunk("违约金条款", page=7, source="合同.pdf"),
        _chunk("质保条款", page=9, source="合同.pdf"),
    ]
    _, available = build_context(chunks)
    citations, invalid = resolve_citations("违约金见 [1]，质保见 [2]。", available)

    assert invalid == []
    assert [c.index for c in citations] == [1, 2]
    assert citations[0].label == "合同.pdf 第 7 页"
    assert citations[1].label == "合同.pdf 第 9 页"


def test_resolve_citations_flags_out_of_range() -> None:
    """**越界编号必须被记录，不能静默丢弃。**

    模型引用 `[9]` 而上下文只有 2 条 —— 那说明它在编。
    静默丢弃会让答案里留一个悬空的 `[9]`，而用户以为那是有依据的。
    """
    _, available = build_context([_chunk("只有一个", page=1)])
    citations, invalid = resolve_citations("事实 [1]，另有 [9]。", available)

    assert [c.index for c in citations] == [1]
    assert invalid == [9], "越界编号没被记录"


def test_resolve_citations_dedups_and_sorts() -> None:
    """引用列表按编号升序且去重 —— 它是"阅读地图"，不是正文顺序。"""
    _, available = build_context([_chunk(f"块{i}", page=i) for i in range(1, 4)])
    citations, _ = resolve_citations("[3][1][3][2]", available)
    assert [c.index for c in citations] == [1, 2, 3]


# ============================================================ 4. 生成与引用约束


def _two_chunks() -> list[Chunk]:
    return [
        _chunk("违约金按合同总额的千分之一计算。", page=7, source="合同.pdf"),
        _chunk("逾期超过三十日，甲方有权解除合同。", page=9, source="合同.pdf"),
    ]


@pytest.mark.anyio
async def test_answer_carries_numbered_citations() -> None:
    """回答里带编号 → `Answer.citations` 按编号映射回 source+page。"""
    client = FakeClient("违约金按合同总额的千分之一计算 [1]。")
    answer = await generate_answer(
        "违约金怎么算", _two_chunks(), client=client, model="m", scores=[0.9]
    )

    assert answer.refused is False
    assert [c.index for c in answer.citations] == [1]
    assert answer.citations[0].source == "合同.pdf"
    assert answer.citations[0].page == 7
    assert "[1]" in answer.text


@pytest.mark.anyio
async def test_prompt_tells_model_which_numbers_are_allowed() -> None:
    """**引用约束必须写进 prompt**：明确的"只允许引用给定编号"。

    只靠"请标注引用"是不够的 —— 模型会自己发明编号。
    所以 prompt 里要**枚举当前可用的编号**，让它没有发明空间。
    """
    client = FakeClient()
    await generate_answer("问题", _two_chunks(), client=client, model="m", scores=[0.9])

    prompt = client.completions.calls[0]["messages"][1]["content"]
    assert "[1]" in prompt and "[2]" in prompt
    assert "只允许引用" in prompt
    # 每个事实句后标注的要求也要在
    assert "每个事实句后" in prompt


@pytest.mark.anyio
async def test_answer_records_invalid_refs() -> None:
    """模型编了一个不存在的编号 → `Answer.invalid_refs` 非空。"""
    client = FakeClient("违约金见 [1]，另见 [9]。")
    answer = await generate_answer("违约金", _two_chunks(), client=client, model="m", scores=[0.9])

    assert [c.index for c in answer.citations] == [1]
    assert answer.invalid_refs == [9]


@pytest.mark.anyio
async def test_zero_temperature_for_reproducibility() -> None:
    client = FakeClient()
    await generate_answer("问题", _two_chunks(), client=client, model="m", scores=[0.9])
    assert client.completions.calls[0]["temperature"] == 0.0


# ============================================================ 5. 拒答


@pytest.mark.anyio
async def test_refuses_on_empty_retrieval() -> None:
    """检索无结果 → 拒答，**且不调模型**。"""
    client = _ExplodingClient()
    answer = await generate_answer("知识库里没有的问题", [], client=client, model="m")

    assert answer.refused is True
    assert answer.text == REFUSAL_TEXT
    assert answer.citations == []


@pytest.mark.anyio
async def test_refuses_below_score_threshold() -> None:
    """最高分低于阈值 → 拒答，**且不调模型**。

    用 `_ExplodingClient` 证明"根本没碰模型"，比断言话术更有力 ——
    话术可以改，但"没调模型"是一条结构性事实。
    """
    client = _ExplodingClient()
    answer = await generate_answer(
        "无关问题",
        _two_chunks(),
        client=client,
        model="m",
        min_score=0.5,
        scores=[0.2, 0.1],
    )

    assert answer.refused is True
    assert answer.text == REFUSAL_TEXT
    assert answer.top_score == pytest.approx(0.2), "拒答时要给出分数，便于排查"


@pytest.mark.anyio
async def test_threshold_equal_is_not_refused() -> None:
    """分数**等于**阈值时不算拒答（`<` 而不是 `<=`）。

    边界条件要明确：等于阈值说明"刚好达到下限"，那不是"没找到"。
    """
    client = FakeClient()
    answer = await generate_answer(
        "问题", _two_chunks(), client=client, model="m", min_score=0.5, scores=[0.5]
    )
    assert answer.refused is False
    assert len(client.completions.calls) == 1


@pytest.mark.anyio
async def test_threshold_zero_disables_gate() -> None:
    """`min_score=0.0` 显式关掉闸门（给"只想测引用逻辑"的场景留口子）。"""
    client = FakeClient()
    answer = await generate_answer(
        "问题", _two_chunks(), client=client, model="m", min_score=0.0, scores=[0.01]
    )
    assert answer.refused is False
    assert len(client.completions.calls) == 1


@pytest.mark.anyio
async def test_no_scores_skips_gate() -> None:
    """不传 `scores` 时跳过分数闸门（空结果闸门仍然生效）。

    **这是给测试留的口子，生产路径必须传 scores。**
    两条闸门分开测，是为了让"哪条生效"永远可分辨。
    """
    client = FakeClient()
    answer = await generate_answer("问题", _two_chunks(), client=client, model="m")
    assert answer.refused is False
    assert answer.top_score is None


def test_refusal_text_points_to_the_right_action() -> None:
    """拒答话术要引导正确的下一步：**检查文档有没有导入**。

    不能只说"没找到" —— 用户会以为"再问一遍试试"。
    也不能说成"模型认为资料里没有" —— 那是把责任推给模型，
    而实际是**我们的检索没找到**。
    """
    assert "知识库" in REFUSAL_TEXT
    assert "导入" in REFUSAL_TEXT


# ============================================================ 6. 阈值必须挂在 embedder 上


def test_threshold_is_embedder_specific() -> None:
    """**阈值不能是全局常量**：不同 embedder 的分数分布完全不同。

    实测（`scripts/embedding_score_probe.py`）：

    | embedder | 相关对 | 无关对 |
    | --- | --- | --- |
    | bge | 0.577 / 0.667 / 0.693 | 0.246 |
    | hashing | 0.148 / 0.273 / 0.387 | 0.190 |

    hashing 的相关与无关**完全重叠**，所以它的阈值必须是 0.0（关掉闸门）。
    这条测试就是防止有人"统一阈值"把 hashing 的路径改成会误拒。
    """
    from app.rag.embedding import get_embedder

    hashing = get_embedder("hashing")
    assert hashing.recommended_min_score == 0.0, (
        "hashing 的分数分布无可分界线，阈值必须为 0（关掉闸门）"
    )

    bge_class = __import__("app.rag.embedding.bge", fromlist=["BGEEmbedder"]).BGEEmbedder
    assert bge_class.recommended_min_score == pytest.approx(DEFAULT_MIN_SCORE)
    assert bge_class.recommended_min_score > 0.0


def test_hashing_relevant_and_irrelevant_overlap() -> None:
    """**用数据证明"hashing 不能设阈值"**，而不是嘴上说。

    取一组"真正相关"的问答对和一组"无关"的，算余弦。
    如果相关的**最小值**低于无关的**最大值**，两个分布就重叠了 ——
    任何单一阈值都会同时误杀和漏放。
    """
    embedder = HashingEmbedder(dimension=64)
    relevant = [
        ("违约金怎么算", "第六条 违约责任：按日万分之五支付违约金。"),
        ("什么时候付钱", "第五条 付款计划：验收合格后十五个工作日内付款。"),
        ("质保多久", "第十三条 质保期：12 个月。"),
    ]
    irrelevant = [
        ("违约金怎么算", "附件二 培训计划：为运维人员提供 3 天培训。"),
        ("违约金怎么算", "第十二条 交付地点：杭州市余杭区文一西路。"),
    ]

    rel_scores = [cosine(embedder.embed_query([q])[0], embedder.embed([p])[0]) for q, p in relevant]
    irr_scores = [
        cosine(embedder.embed_query([q])[0], embedder.embed([p])[0]) for q, p in irrelevant
    ]

    assert min(rel_scores) < max(irr_scores), (
        f"预期分布重叠（相关最低 {min(rel_scores):.3f} < 无关最高 {max(irr_scores):.3f}），"
        f"若不重叠说明这个测试语料不再有代表性"
    )
