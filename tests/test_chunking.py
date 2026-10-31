"""三策略切分的契约测试：统一签名、metadata 继承、失效方式。

## 这个文件测什么（按重要性）

### 一、三种策略**都**必须满足的契约

- `page` / `pages` 从 `Document` 继承，不重新推断（跨页 `(6,7)` 不许被吞掉）
- 表格整张不切
- 空块不产出
- `metadata["splitter"]` 等于策略名

这三条不满足，下游（引用溯源、表格标记）会**静默出错** ——
不会抛异常，只是块的归属变错。

### 二、每种策略各自的**失效方式**必须与文档一致

这是本文件最特别的部分。三种策略的失效不是"对/错"，而是
**"在什么条件下会变质"**（`docs/chunking-predictions.md` 的理论 vs 实测）：

| 策略 | 断言 | 为什么这条值得测 |
| --- | --- | --- |
| heading | PDF 上必须标 `heading_fallback="page"` | 不标的话，"标题策略在 PDF 上表现如何"测的是分页器 |
| heading | 纯标题节必须被合并进下一节 | 否则产出"27 字只有标题"的块（判据三） |
| semantic | 缺 embedder 时必须**拒绝构造** | 给默认值会让"用哪个模型切的"变成隐式状态 |
| semantic | 并列短句不得切成一堆碎片 | 预判里说它"在合同上会切过碎"，要有对策且可验证 |
| recursive | 块尾必须落句末标点 | 这是它相对 LangChain 的唯一实质优势 |

### 三、工厂与配置校验

`build_strategy` 的报错必须**说人话**（缺 embedder 时不能是 `TypeError`），
配置非法必须在装配期炸而不是切到一半才炸。
"""

from __future__ import annotations

import pytest

from app.rag.chunking import (
    DEFAULT_STRATEGY,
    STRATEGIES,
    ChunkingConfig,
    ChunkingStrategy,
    HeadingChunking,
    RecursiveChunking,
    SemanticChunking,
    build_strategy,
    from_config,
    split_sentences,
    validate_config,
)
from app.rag.embedding import HashingEmbedder
from app.rag.schema import Chunk, Document

# ============================================================ 0. 样本


def _doc(
    text: str,
    *,
    page: int | None = 1,
    source: str = "合同.pdf",
    pages: tuple[int, ...] = (),
    metadata: dict | None = None,
) -> Document:
    return Document(source=source, page=page, text=text, metadata=metadata or {}, pages=pages)


_LONG_CN = (
    "第五条 付款计划。甲方按下列里程碑分期支付合同价款，验收合格后十五个工作日内付款。"
    "付款方式为银行转账，账户信息以合同附件为准。乙方须在收款前开具增值税专用发票。"
    "第六条 违约责任。乙方逾期交付的，按日万分之五支付违约金，累计不超过合同总额的百分之十。"
    "违约金可直接从应付款项中扣除。甲方逾期付款的，按日万分之三支付逾期利息。"
) * 6


def _md_docs() -> list[Document]:
    """模拟 Markdown loader 的产物：标题行 + 正文段，**每段都带 heading_level**。

    这条 fixture 本身是实测教训：第一版实现假设"有 heading_level 就是节开头"，
    而 loader 的真实语义是"这一段属于哪一级的节"。fixture 必须复现那个语义，
    否则测试会绿而线上会错。
    """
    return [
        _doc(
            "# 平台说明\n本文档描述平台总体架构。",
            page=None,
            source="说明.md",
            metadata={"block": "text", "heading_level": 1},
        ),
        _doc(
            "## 一、整体架构",
            page=None,
            source="说明.md",
            metadata={"block": "heading", "heading_level": 2},
        ),
        _doc(
            "平台由接入层、调度层、模型层三部分组成。接入层负责协议转换与限流。",
            page=None,
            source="说明.md",
            metadata={"block": "text", "heading_level": 2},
        ),
        _doc(
            "## 二、数据流",
            page=None,
            source="说明.md",
            metadata={"block": "heading", "heading_level": 2},
        ),
        _doc(
            "原始文档经 loader 解析后进入切分器，再按 token 预算分批送入模型。",
            page=None,
            source="说明.md",
            metadata={"block": "text", "heading_level": 2},
        ),
    ]


def _heading_only_docs() -> list[Document]:
    """父标题下面紧跟子标题 —— 父节**没有正文**。用来测孤儿标题合并。"""
    return [
        _doc(
            "## P0-2 · extract 延迟要处理",
            page=None,
            source="清单.md",
            metadata={"block": "heading", "heading_level": 2},
        ),
        _doc(
            "### 为什么周报里漏了它",
            page=None,
            source="清单.md",
            metadata={"block": "heading", "heading_level": 3},
        ),
        _doc(
            "W3 只盯着摘要，但实测四类任务的延迟里 extract 的 P95 是 40.8 秒。",
            page=None,
            source="清单.md",
            metadata={"block": "text", "heading_level": 3},
        ),
    ]


# ============================================================ 1. 统一契约


@pytest.mark.parametrize("strategy_name", ["recursive", "heading", "semantic"])
def test_all_strategies_satisfy_protocol(strategy_name: str) -> None:
    """三种策略都必须是 `ChunkingStrategy` 实例（工厂装配结果可直接验证）。"""
    strategy = build_strategy(strategy_name, embedder=HashingEmbedder())
    assert isinstance(strategy, ChunkingStrategy)
    assert strategy.name == strategy_name


@pytest.mark.parametrize("strategy_name", ["recursive", "heading", "semantic"])
def test_all_strategies_keep_page_attribution(strategy_name: str) -> None:
    """**跨页的 `pages=(6,7)` 必须活下来**，三种策略都不例外。

    这条是 2026-10-31 那个缺口的回归测试：页码链在 `Document` → `Chunk` 这一段
    一旦断掉，后面"抽查任意 chunk 回查原文"就做不成。
    """
    docs = [
        _doc(_LONG_CN, page=6, pages=(6, 7)),
        _doc("另一段的短内容。", page=7, pages=(7,)),
    ]
    strategy = build_strategy(strategy_name, embedder=HashingEmbedder())
    chunks = strategy.chunk(docs, ChunkingConfig())

    assert chunks, "没有产出任何块"
    # 逐文档核对继承：第一段跨页 (6,7)，第二段单页 (7,)
    # **不能笼统要求"每个块都含 6"** —— 那样断言的是我的测试预期而不是契约。
    crosspage = [c for c in chunks if "第五条" in c.text or "第六条" in c.text]
    single = [c for c in chunks if "另一段" in c.text]
    assert crosspage, f"{strategy_name} 第一段一个块都没产出"
    assert single, f"{strategy_name} 第二段一个块都没产出"

    for chunk in crosspage:
        assert chunk.pages == (6, 7), f"{strategy_name} 丢了跨页范围：{chunk.pages}"
        assert chunk.page == 6
    for chunk in single:
        assert chunk.pages == (7,), f"{strategy_name} 单页段的范围错了：{chunk.pages}"
        assert chunk.page == 7
    # 所有块的 pages 都非空 —— 没有块丢掉出处
    assert all(chunk.pages for chunk in chunks), "有块丢了 pages"


@pytest.mark.parametrize("strategy_name", ["recursive", "heading", "semantic"])
def test_all_strategies_keep_table_whole(strategy_name: str) -> None:
    """**表格整张一块**，不许被切开。

    一张 20 行的合同金额表被切成两半，行列关系就永久丢失 ——
    而"这张表的第三列是什么"是合同问答里最常见的问题。
    """
    table = "| 里程碑 | 时间 | 金额 |\n| --- | --- | --- |\n" + "\n".join(
        f"| 第{i}期 | 2026-{i:02d}-01 | {i * 1000} |" for i in range(1, 30)
    )
    docs = [_doc(table, page=3, pages=(3,), metadata={"block": "table"})]
    chunks = build_strategy(strategy_name, embedder=HashingEmbedder()).chunk(
        docs, ChunkingConfig(chunk_chars=100)
    )

    assert len(chunks) == 1, f"{strategy_name} 把表格切成了 {len(chunks)} 块"
    assert chunks[0].text == table.strip()


@pytest.mark.parametrize("strategy_name", ["recursive", "heading", "semantic"])
def test_all_strategies_skip_blank_blocks(strategy_name: str) -> None:
    """空白 `Document` 不产出空块。**下游会忘掉这个过滤**，所以在这里挡住。"""
    docs = [_doc("   \n\n  "), _doc("有内容的段落。", page=2, pages=(2,))]
    chunks = build_strategy(strategy_name, embedder=HashingEmbedder()).chunk(docs, ChunkingConfig())
    assert all(chunk.text.strip() for chunk in chunks)
    assert len(chunks) >= 1


@pytest.mark.parametrize("strategy_name", ["recursive", "heading", "semantic"])
def test_all_strategies_empty_input(strategy_name: str) -> None:
    """空输入返回空列表，不抛异常（契约明写）。"""
    assert (
        build_strategy(strategy_name, embedder=HashingEmbedding()).chunk([], ChunkingConfig()) == []
    )


def HashingEmbedding():  # noqa: N802 - 测试内的构造辅助
    return HashingEmbedder()


@pytest.mark.parametrize("strategy_name", ["recursive", "heading", "semantic"])
def test_splitter_metadata_is_strategy_name(strategy_name: str) -> None:
    """`metadata["splitter"]` 必须等于策略名。

    它是"这批向量是哪版策略切的"的唯一线索。混入别的值（比如 `"delimiter"`
    或 `"table"`）会让按策略分组统计失效。
    """
    chunks = build_strategy(strategy_name, embedder=HashingEmbedder()).chunk(
        [_doc("一段正文，足够成块。", page=1, pages=(1,))], ChunkingConfig()
    )
    assert chunks
    assert all(chunk.metadata.get("splitter") == strategy_name for chunk in chunks)


# ============================================================ 2. recursive


def test_recursive_ends_at_sentence_marks() -> None:
    """**块尾必须落在句末标点** —— 这是递归字符相对 LangChain 的唯一实质优势。

    LangChain 实测 4/5 块停在半个词上（`…验收标` / `…支付合同价`），
    因为它为了填满 `chunk_size` 会切割原子片。
    """
    chunks = RecursiveChunking().chunk([_doc(_LONG_CN, page=1, pages=(1,))], ChunkingConfig())
    assert len(chunks) > 1, "样本太短，测不到切分"
    for chunk in chunks:
        assert chunk.text.rstrip().endswith(("。", "！", "？", "；", "\n")), (
            f"块尾不是句末标点：{chunk.text[-12:]!r}"
        )


def test_recursive_packs_parallel_clauses_into_one_block() -> None:
    """**递归字符的固有缺陷**：并列条款被塞进同一块。

    这条**不是 bug 断言，是特性记录**。预判里写了"递归字符会在并列条款上输"，
    这里把它固定成一个可验证的事实 —— 周六实测若发现"它其实会分开"，
    说明这条预判错了，需要改的是预判而不是实现。

    短条款（每句 20 字）加起来远小于 500，所以必然被合并。
    """
    clauses = "".join(f"{i}. 本条第 {i} 项约定见附件第 {i} 页。\n" for i in range(1, 8))
    chunks = RecursiveChunking().chunk([_doc(clauses, page=1, pages=(1,))], ChunkingConfig())

    assert len(chunks) == 1
    # 一个块里出现了 7 个独立条款 —— 检索"第 3 项"时会带出全部 7 项
    assert chunks[0].text.count("本条第") == 7


# ============================================================ 3. heading


def test_heading_merges_paragraphs_of_same_section() -> None:
    """同一节的多个段落必须**合并**成一块（结构优先，不按段切）。

    回归测试：第一版按"有 `heading_level`"分组，结果 md 切出 44 块 = 段数。
    """
    chunks = HeadingChunking().chunk(_md_docs(), ChunkingConfig())

    # 5 段 md 结构 → 标题节 + 两个二级节，合并后应显著少于此前的 5
    assert len(chunks) <= 4, f"没有合并：{len(chunks)} 块 / 5 段"
    # 二级节的标题与正文必须在同一块
    architecture = [c for c in chunks if "整体架构" in c.text]
    assert architecture, "「一、整体架构」这一节不见了"
    assert "接入层负责协议转换" in architecture[0].text, "标题与正文被分开了"


def test_heading_pdf_degrades_to_page_and_says_so() -> None:
    """**PDF 无标题 → 按页切，且必须标 `heading_fallback="page"`。**

    不标这条，"标题策略在 PDF 上表现如何"这个实验结论测的是分页器。
    """
    docs = [_doc(f"第 {i} 页正文，讨论第 {i} 项交付物。", page=i, pages=(i,)) for i in range(1, 4)]
    chunks = HeadingChunking().chunk(docs, ChunkingConfig())

    assert len(chunks) == 3, "按页降级应每页一块"
    for chunk in chunks:
        assert chunk.metadata.get("heading_fallback") == "page", f"降级没被标注：{chunk.metadata}"


def test_heading_merges_heading_only_section() -> None:
    """**纯标题节并进下一节** —— 判据三（标题与内容是否分离）的正面处理。

    实测：`## P0-2` 下面是 `### 为什么…`，父节只有标题 → 27 字只有标题的块。
    检索"P0-2 是什么"时命中的是它，真正的 518 字内容在下一块且没进上下文。
    """
    chunks = HeadingChunking().chunk(_heading_only_docs(), ChunkingConfig())

    # 父标题不带正文 → 与子节合并，不应剩下"只有标题"的块
    heading_only = [c for c in chunks if c.text.strip().count("\n") == 0]
    assert not heading_only, f"仍有纯标题块：{[c.text[:30] for c in heading_only]}"

    merged = [c for c in chunks if "P0-2" in c.text]
    assert merged
    assert "P95 是 40.8 秒" in merged[0].text, "父标题没有并进子节正文"


def test_heading_splits_oversized_section_and_keeps_path() -> None:
    """超长节在节内二次切，后续片补标题路径（`heading_keep_path`）。

    不补路径的话，第 2、3 片**不再以标题开头**，丢掉了"我在哪一节"。
    """
    docs = [
        _doc(
            "## 很长的节",
            page=None,
            source="长.md",
            metadata={"block": "heading", "heading_level": 2},
        ),
        _doc(
            _LONG_CN,
            page=None,
            source="长.md",
            metadata={"block": "text", "heading_level": 2},
        ),
    ]
    chunks = HeadingChunking().chunk(docs, ChunkingConfig(chunk_chars=300))

    assert len(chunks) > 1, "超长节没有被二次切"
    # 第一片以标题开头；后续片补了**完整 section_path**
    assert chunks[0].text.startswith("## 很长的节")
    for chunk in chunks[1:]:
        assert chunk.text.startswith("【很长的节】"), f"后续片没补路径：{chunk.text[:30]!r}"
        # **二次切的每一片都要继承 section_path**（本周验收项之一）
        assert chunk.metadata.get("section_path") == "很长的节"
        assert chunk.metadata.get("section_part") is not None


def test_heading_respects_max_level() -> None:
    """`heading_max_level` 之后的标题不开新节：它们只是上一节内部的分点。"""
    docs = [
        _doc(
            "## 大节",
            page=None,
            metadata={"block": "heading", "heading_level": 2},
        ),
        _doc(
            "### 分点一",
            page=None,
            metadata={"block": "heading", "heading_level": 3},
        ),
        _doc(
            "分点一的正文。",
            page=None,
            metadata={"block": "text", "heading_level": 3},
        ),
    ]
    cfg = ChunkingConfig(heading_max_level=2)
    chunks = HeadingChunking().chunk(docs, cfg)

    # 只按 level<=2 分节 → 全部落在同一节
    assert len(chunks) == 1, f"level=3 的标题不该开新节：{len(chunks)} 块"
    assert "分点一的正文。" in chunks[0].text


# ============================================================ 4. semantic


def test_semantic_requires_embedder() -> None:
    """**缺 embedder 必须拒绝构造，并说人话**（不是 `TypeError` 或事后 AttributeError）。"""
    with pytest.raises(ValueError, match="embedder"):
        build_strategy("semantic")


def test_semantic_splits_sentences() -> None:
    """句子切分器：句末标点保留在前一句，空句不产出。"""
    sentences = split_sentences("第一句。第二句！第三句？")
    assert sentences == ["第一句。", "第二句！", "第三句？"]


@pytest.mark.parametrize(
    "text",
    [
        "第一行。\n第二行。\n第三行。",
        "# 标题\n> 引用正文。\n\n下一行。",
        "无标点的一段话",
        "第一句。第二句！第三句？",
        # Markdown 表格：行内句末标点**不许**把表格切碎
        "| 主题 | 要读的 |\n| --- | --- |\n| A | 它会不会重排？（我想控制） |",
        "说明一句。\n| A | 降不降分？ |\n表格后面的正文。",
    ],
)
def test_semantic_sentence_split_preserves_every_character(text: str) -> None:
    """**切分不丢字符**：`"".join(split_sentences(t)) == t.strip()`。

    这条不变量是"块覆盖了原文"的可验证形式。它被实测打穿了**四次**，
    每次都是同一类错：局部 strip / 局部丢弃，在没有 Markdown 结构的纯文本上看不出来。

    | 版本 | 写法 | 丢了什么 |
    | --- | --- | --- |
    | 一 | `buffer.strip()` 落袋 | 行尾换行全丢 → `# 标题> 正文` 一坨 |
    | 二 | `strip(" \\t")` | `"。"` 之后的 `\n` 成空白新句被丢弃 |
    | 三 | 空白片段并入上一句 | **连带丢掉换行前的正文**：覆盖率 72% |
    | 四 | 空白不落袋、留在 buffer 里往下带 | 对；但表格行被句末标点切碎（177→176） |
    | 五（当前） | 表格行内不收句 | 对 |

    第 5 条参数化用例专门守最后一版：表格行 `| A | 它会不会重排？（我想控制） |`
    不能被 `？` 切成两段，否则行尾 `|` 会被当作空片段吞掉。
    """
    sentences = split_sentences(text)
    assert "".join(sentences) == text.strip(), f"切分丢了字符：{[repr(s) for s in sentences]}"
    assert all(s.strip() for s in sentences), "产生了空白句"


def test_semantic_table_rows_are_not_split_at_punctuation() -> None:
    """表格行内的句末标点**不是断点**：整行是一整个单元格。

    实测：`docs/w4-checklist.md` 末段的表格行
    `| bge 模型卡 | … | 前缀写错会不会静默降分？ |`
    被 `？` 切开后，行尾孤立的 `|` 在下一步被当空片段丢弃 ——
    整行少了收尾管道符，Markdown 表格从此渲染不出来。
    """
    row = "| bge 模型卡 | 相似度度量 | 前缀写错会不会静默降分？ |"
    sentences = split_sentences(row)
    assert sentences == [row], f"表格行被切开了：{sentences}"


def test_semantic_merges_parallel_clauses_not_to_fragments() -> None:
    """**并列条款不许被切成一堆碎片** —— 预判里"语义策略切过碎"的对策。

    每句一个主题时，句间相似度全都低，每句都会被判成断点。
    `semantic_min_chars` 必须把过短的段并回去。
    """
    clauses = "".join(f"{i}. 本条第 {i} 项约定见附件第 {i} 页。" for i in range(1, 9))
    chunks = SemanticChunking(HashingEmbedder(dimension=64)).chunk(
        [_doc(clauses, page=1, pages=(1,))], ChunkingConfig()
    )

    assert len(chunks) < 8, f"每句一块 = 切过碎：{len(chunks)} 块 / 8 句"
    assert all(len(chunk.text) >= 50 for chunk in chunks), (
        f"出现了碎片块：{[len(c.text) for c in chunks]}"
    )


def test_semantic_forces_break_in_single_topic() -> None:
    """**单主题长段必须被强制断开**（`semantic_max_chars`）。

    没有这条，语义策略在"一整节都在讲同一个概念"的文档上会一路合并，
    退化成一个**代价更高的字数切分器** —— 付了 embedding 的钱，拿递归字符的效果。
    """
    single_topic = "本节讨论交付物验收标准。" * 120  # 约 1200 字，同主题重复
    chunks = SemanticChunking(HashingEmbedder(dimension=64)).chunk(
        [_doc(single_topic, page=1, pages=(1,))],
        ChunkingConfig(semantic_max_chars=400, chunk_chars=500),
    )

    assert len(chunks) > 1, "单主题长段没有被强制断开"
    assert all(len(chunk.text) <= 400 + 60 for chunk in chunks), (
        f"超过 semantic_max_chars 还没断：{[len(c.text) for c in chunks]}"
    )


def test_semantic_records_embedder_name() -> None:
    """块上要记 `embedder` —— 语义切分的结论**完全由它决定**。

    不记的话，一份 hashing 切出来的样本与一份 bge 切出来的样本无法区分，
    而 hashing 下的语义边界基本是噪声。
    """
    chunks = SemanticChunking(HashingEmbedder(dimension=64)).chunk(
        [_doc("第一句。第二句。第三句。", page=1, pages=(1,))], ChunkingConfig()
    )
    assert all(chunk.metadata.get("embedder") == "hashing" for chunk in chunks)


# ============================================================ 5. 工厂与配置


def test_factory_unknown_strategy_lists_available() -> None:
    """未知策略名要报出可用列表 —— 只说"未知"等于让人去翻代码。"""
    with pytest.raises(ValueError, match="recursive"):
        build_strategy("whatever")


def test_factory_default_is_recursive() -> None:
    """默认策略必须是**最不容易静默变差**的那个（只依赖字符）。"""
    assert DEFAULT_STRATEGY == "recursive"
    assert isinstance(build_strategy(), RecursiveChunking)


def test_factory_accepts_instance() -> None:
    """`from_config` 收实例（测试注入替身用），与收字符串是同一个入口。"""
    strategy = HeadingChunking()
    assert from_config(strategy) is strategy


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"chunk_chars": 0}, "chunk_chars"),
        ({"overlap_chars": -1}, "overlap_chars"),
        ({"chunk_chars": 100, "overlap_chars": 100}, "overlap"),
        ({"heading_max_level": 9}, "heading_max_level"),
        ({"semantic_min_chars": 500, "semantic_max_chars": 400}, "semantic_min"),
    ],
)
def test_validate_config_rejects_illegal(kwargs: dict, match: str) -> None:
    """非法配置在**装配期**就炸，而不是切到一半才炸。"""
    with pytest.raises(ValueError, match=match):
        validate_config(ChunkingConfig(**kwargs))


def test_build_strategy_validates_config() -> None:
    """工厂要连带校验配置：这是"装配期失败"而不是"运行期失败"的落点。"""
    with pytest.raises(ValueError, match="overlap"):
        build_strategy("recursive", cfg=ChunkingConfig(chunk_chars=100, overlap_chars=100))


def test_registered_strategies_all_buildable() -> None:
    """注册表里的每个名字都必须能装配出来 —— 否则注册表就是在撒谎。"""
    for name in STRATEGIES:
        strategy = build_strategy(name, embedder=HashingEmbedder())
        assert strategy.name == name


def test_strategy_names_are_unique() -> None:
    """`metadata["splitter"]` 靠名字区分策略，重名会让统计合并成一项。"""
    names = {build_strategy(name, embedder=HashingEmbedder()).name for name in STRATEGIES}
    assert len(names) == len(STRATEGIES)


def test_chunks_have_index_in_order() -> None:
    """块序号必须连续递增 —— 它进 `store_id`（位置派生主键）。"""
    chunks = RecursiveChunking().chunk(
        [_doc(_LONG_CN, page=1, pages=(1,)), _doc("另一段。", page=2, pages=(2,))],
        ChunkingConfig(),
    )
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert len({chunk.store_id for chunk in chunks}) == len(chunks), "store_id 撞了"
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
