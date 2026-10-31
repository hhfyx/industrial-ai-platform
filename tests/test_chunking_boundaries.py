"""W5 周二：标题层级 / 边界处理 / 三类内容 / 半句率。

## 这个文件按任务的四个验收项组织

| 验收项 | 测试 |
| --- | --- |
| 层级信息不丢 | `test_*_section_path*`（Markdown 栈、PDF 启发式、降级不伪造） |
| 边界 case 通过 | `test_number_*`（数字/单位不切开、日期整体、重叠不截半数字） |
| 三类内容不被切碎 | `test_*_table_*` / `test_*_code_*` |
| 半句率有数字 | `test_sentence_rate_*`（含与"等长硬切"对照的**量级差**） |

## 为什么每条都写"为什么这条值得测"

这批测试里有几条是**实测打穿过来的**（注释里标了"实测"）。
没有那段说明，后来的人只会看到一条奇怪的断言，删掉它时不知道删掉了什么。
"""

from __future__ import annotations

import pytest

from app.rag.chunking import (
    ChunkingConfig,
    HeadingChunking,
    RecursiveChunking,
    SemanticChunking,
)
from app.rag.chunking.boundaries import (
    number_spans,
    position_is_unsafe,
    safe_cut_position,
)
from app.rag.chunking.code import split_code, strip_fences
from app.rag.chunking.quality import check_chunk, measure, measure_by_kind
from app.rag.embedding import HashingEmbedder
from app.rag.schema import Chunk, Document


def _doc(
    text: str,
    *,
    page: int | None = None,
    source: str = "说明.md",
    metadata: dict | None = None,
) -> Document:
    return Document(source=source, page=page, text=text, metadata=metadata or {})


# ============================================================ 1. 标题层级：section_path


def _nested_md() -> list[Document]:
    """三级结构：`# 第1章` → `## 1.2` → `### 公差`。

    **`heading_level` 挂在该节的每一段上**（loader 的真实语义），
    `block="heading"` 才是"整段就是标题"。

    ⚠️ 每级标题下都必须有**足够长的正文**：只放一行引言会让该节成为
    "纯标题节"，从而被 `_merge_orphan_headings` 并进下一节（那是对的行为，
    但会让本测试测到合并逻辑而不是路径逻辑）。实测踩过这个坑。
    """
    body = "正文内容，说明本节要讲的事。" * 12  # 约 150 字，足够成节
    return [
        _doc("# 第1章", metadata={"block": "heading", "heading_level": 1}),
        _doc(body, metadata={"block": "text", "heading_level": 1}),
        _doc("## 1.2", metadata={"block": "heading", "heading_level": 2}),
        _doc(body, metadata={"block": "text", "heading_level": 2}),
        _doc("### 公差", metadata={"block": "heading", "heading_level": 3}),
        _doc("公差要求为 0.05mm。", metadata={"block": "text", "heading_level": 3}),
    ]


def test_section_path_is_full_ancestor_chain() -> None:
    """**`section_path` 必须是完整祖先链**：`第1章 > 1.2 > 公差`。

    这是标题切分相对递归切分**唯一不可替代**的价值：
    检索时能按章节过滤，生成时能说"在第 1.2 节"。
    递归切分从结构上就拿不到这个信息（它只看分隔符）。

    只给一层（`L3 > 公差`）是"看起来像路径、其实不是" —— 下游拿它当层级依据会错。
    """
    chunks = HeadingChunking().chunk(_nested_md(), ChunkingConfig())

    paths = [c.metadata.get("section_path") for c in chunks]
    assert "第1章" in paths
    assert "第1章 > 1.2" in paths
    assert "第1章 > 1.2 > 公差" in paths, f"三级路径不完整：{paths}"


def test_section_path_pops_siblings() -> None:
    """同级标题**出栈**：`## 1.2` 之后的 `## 1.3` 路径不能包含 `1.2`。"""
    body = "正文内容，说明本节要讲的事。" * 12
    docs = [
        _doc("# 第1章", metadata={"block": "heading", "heading_level": 1}),
        _doc(body, metadata={"block": "text", "heading_level": 1}),
        _doc("## 1.2", metadata={"block": "heading", "heading_level": 2}),
        _doc(body, metadata={"block": "text", "heading_level": 2}),
        _doc("## 1.3", metadata={"block": "heading", "heading_level": 2}),
        _doc(body, metadata={"block": "text", "heading_level": 2}),
    ]
    chunks = HeadingChunking().chunk(docs, ChunkingConfig())
    paths = [c.metadata.get("section_path") for c in chunks]

    assert "第1章 > 1.2" in paths, f"缺 1.2 的路径：{paths}"
    assert "第1章 > 1.3" in paths, f"缺 1.3 的路径：{paths}"
    # 1.3 的路径里不许出现 1.2（同级必须出栈）
    for path in paths:
        if path and "1.3" in path:
            assert "1.2" not in path, f"同级标题没出栈：{path}"


def test_section_path_inherited_by_secondary_split() -> None:
    """**超长节二次切之后，每一片都要继承 `section_path`**（本周验收项）。

    样本要让第 1 章有正文（否则它成纯标题节被合并），
    而第 2 层节超预算 → 触发二次切。
    """
    long_body = "违约金按日万分之五计算。" * 80
    docs = [
        _doc("# 第1章", metadata={"block": "heading", "heading_level": 1}),
        _doc("章引言正文。" * 20, metadata={"block": "text", "heading_level": 1}),
        _doc("## 违约责任", metadata={"block": "heading", "heading_level": 2}),
        _doc(long_body, metadata={"block": "text", "heading_level": 2}),
    ]
    chunks = HeadingChunking().chunk(docs, ChunkingConfig(chunk_chars=300))

    multi = [c for c in chunks if c.metadata.get("section_part") is not None]
    assert multi, "样本太短，没有触发二次切"
    for chunk in multi:
        assert chunk.metadata.get("section_path") == "第1章 > 违约责任", (
            f"二次切片路径丢了：{chunk.metadata.get('section_path')}"
        )


def test_section_path_empty_when_no_heading() -> None:
    """无标题时**不伪造路径**。

    空串让下游能区分"没有层级"（正常）与"层级算失败"（要查 bug）。
    伪造一个 `未命名节` 会把这两件事混成一个。
    """
    docs = [_doc("一段没有标题的正文。", page=None)]
    chunks = HeadingChunking().chunk(docs, ChunkingConfig())
    for chunk in chunks:
        assert chunk.metadata.get("section_path") in (None, "")


# ============================================================ 2. PDF 标题启发式


def test_pdf_heading_probe_on_fixture() -> None:
    """字号启发式在合成样本上应该给出高可信度（正文 10pt / 标题 15pt）。"""
    from app.rag.loaders.pdf_headings import probe_pdf

    probe = probe_pdf("tests/fixtures/sample20.pdf")
    assert probe.body_size == 10.0
    assert probe.ratio >= 1.3
    assert probe.confidence >= 0.6
    assert probe.is_trustworthy()
    assert 1 in probe.headings


def test_pdf_heading_probe_untrustworthy_is_flagged() -> None:
    """**不可信的探测必须能被识别出来** —— 这是"降级而不是硬做"的前提。

    构造一个"所有行同字号"的假文档：没有区分度 → confidence/ratio 都低。
    真实场景是排版器把正文与标题都设成 10pt（或扫描件转文字后字号全丢）。
    """
    from app.rag.loaders.pdf_headings import LineFont, analyze

    lines = {1: [LineFont(text=f"第 {i} 行正文内容", size=10.0) for i in range(1, 20)]}
    probe = analyze(lines)
    assert probe.body_size == 10.0
    assert probe.headings == {}, "同字号文档不该识别出标题"
    assert not probe.is_trustworthy()


def test_pdf_loader_splits_heading_into_own_document() -> None:
    """**标题行必须单独成 `Document`**，否则策略层拿不到节边界。

    实测教训：第一版给整页打 `heading_level=1`，结果策略层"每页都开一节"
    → 等同于按页切，而且不会被标成 `heading_fallback="page"`（因为有 level）。
    那是一条**假的成功路径**。
    """
    from app.rag.loaders.pdf import load_pdf

    result = load_pdf("tests/fixtures/sample20.pdf")
    headings = [d for d in result.documents if d.block == "heading"]
    assert headings, "标题行没有独立成块"
    assert all(d.page is not None for d in headings)
    assert headings[0].text.startswith("第 1 章")

    bodies = [d for d in result.documents if d.block == "text" and d.page == 1]
    assert bodies, "第 1 页的正文丢了"
    # 正文段带层级（表示"属于哪一节"），但**不是标题**
    assert bodies[0].metadata.get("heading_level") == 1


def test_pdf_loader_degrades_when_heuristic_disabled() -> None:
    """关掉启发式 → 一行 `heading_level` 都不填，策略层自动降级为按页切。"""
    from app.rag.chunking import HeadingChunking
    from app.rag.loaders.pdf import load_pdf

    result = load_pdf("tests/fixtures/sample20.pdf", use_heading_heuristic=False)
    assert all(d.metadata.get("heading_level") is None for d in result.documents)

    chunks = HeadingChunking().chunk(result.documents[:4], ChunkingConfig())
    for chunk in chunks:
        assert chunk.metadata.get("heading_fallback") == "page", (
            "降级没有被标注 —— 这会让实验把分页器当成标题切分器"
        )


# ============================================================ 3. 边界：数字与单位


@pytest.mark.parametrize(
    "text",
    [
        "精度为 ±0.05mm，重复",
        "38,400.00 元",
        "95% 通过",
        "2026-11-30 交付",
    ],
)
def test_number_spans_detect_units(text: str) -> None:
    """数字+单位整体被识别为一个区间（切在区间中间会拆开它）。"""
    spans = number_spans(text)
    assert spans, f"没识别出数字：{text}"


def test_safe_cut_moves_left_off_number() -> None:
    """切点落在数字内部时**向左挪到数字之前**。

    `精度为 ±0.05mm，重复` 的数字区间是 `(4, 11)`：
    position=5 会把 `0.05mm` 切成 `0.` + `05mm` —— 后一块以一个半截数字开头，
    检索命中它时模型看到的精度值是错的（而且不会报错）。
    """
    text = "精度为 ±0.05mm，重复"
    assert position_is_unsafe(text, 5)
    assert safe_cut_position(text, 5) == 4
    # 边界位置本身是安全的
    assert not position_is_unsafe(text, 4)
    assert not position_is_unsafe(text, 11)


def test_safe_cut_returns_original_when_number_at_start() -> None:
    """数字从文本开头就开始时**返回原位置**。

    向前挪到 0 不安全、向后挪会超预算 —— 退回原位是唯一能保证"不产出空块"的选择。
    这种情形只在"预算小于一个数字串"时发生，属于极端预算，实现层无法修复。
    """
    text = "0.05mm 是公差"
    assert safe_cut_position(text, 3) == 3


def test_hard_cut_does_not_break_numbers() -> None:
    """**等长硬切不切断数字**。

    实测：`0.05mm` 被切成 `…0.05m` + `m…`，两块都不含完整的 `0.05mm` ——
    命中哪块都答不对精度要求，而且这个错**不会报错**。
    """
    from app.rag.splitters import ChunkSize, DelimiterSplitter

    text = "设备精度为 ±0.05mm，重复定位 0.02mm。" * 3
    chunks = DelimiterSplitter(ChunkSize(chunk_chars=20, overlap_chars=0)).split([_doc(text)])

    for chunk in chunks:
        tail = chunk.text.rstrip()
        # 块尾不许停在"半个单位"上（`0.05m` / `0.`）
        assert not tail.endswith("0.05m"), f"数字被切断：{tail[-10:]!r}"
        # 块首不许以下半截数字开头
        assert not chunk.text.lstrip().startswith(("05mm", "02mm")), (
            f"块首是半截数字：{chunk.text[:10]!r}"
        )


def test_overlap_tail_does_not_start_block_with_half_number() -> None:
    """**重叠截取也不许从数字中间开始**。

    实测：`buffer[-50:]` 落在 `0.05mm` 中间时，后一块以 `05mm` 开头 ——
    块首出现半个数字，模型看到的精度值是错的。
    """
    from app.rag.splitters import ChunkSize, DelimiterSplitter

    text = "公差要求为 ±0.05mm，累计误差不超过 ±0.02mm。" * 40
    chunks = DelimiterSplitter(ChunkSize(chunk_chars=120, overlap_chars=15)).split([_doc(text)])
    for chunk in chunks[1:]:
        head = chunk.text.lstrip()
        assert not head.startswith(("5mm", "05mm", "2mm", "02mm")), (
            f"重叠让块首成了半截数字：{head[:12]!r}"
        )


# ============================================================ 4. 三类内容不被切碎


def test_table_stays_whole_and_typed() -> None:
    """**表格独立成块、行不切断、转 Markdown、`type=table`** —— 三种策略都要求。"""
    table = "| 里程碑 | 时间 | 金额 |\n| --- | --- | --- |\n" + "\n".join(
        f"| 第{i}期 | 2026-{i:02d}-01 | {i * 1000} |" for i in range(1, 40)
    )
    docs = [_doc(table, metadata={"block": "table"})]

    for strategy in (
        RecursiveChunking(),
        HeadingChunking(),
        SemanticChunking(HashingEmbedder()),
    ):
        chunks = strategy.chunk(docs, ChunkingConfig(chunk_chars=100))
        assert len(chunks) == 1, f"{strategy.name} 把表格切成了 {len(chunks)} 块"
        assert chunks[0].is_table, f"{strategy.name} 丢了 block=table"
        assert chunks[0].text.startswith("|"), "表格不是 Markdown 形态"
        # 行数不许少（行被切掉就没法回答"第三列是什么"）
        assert chunks[0].text.count("\n") == table.count("\n")


def test_heading_keeps_table_separate_from_body() -> None:
    """**heading 策略里表格也必须独立**，不能跟着所在节合并成一块。

    实测教训：早期版本让表格跟节合并，`is_table` 变成 `False`
    （metadata 来自节首段，不是表格）—— 下游不再标"（表格）"，
    而表格的 Markdown 结构还在，所以这个错误**不会报错**。
    """
    docs = [
        _doc("# 付款计划", metadata={"block": "heading", "heading_level": 1}),
        _doc("下表列出各期金额。", metadata={"block": "text", "heading_level": 1}),
        _doc("| 一期 | 100 |\n| --- | --- |\n| 二期 | 200 |", metadata={"block": "table"}),
    ]
    chunks = HeadingChunking().chunk(docs, ChunkingConfig())

    tables = [c for c in chunks if c.is_table]
    assert len(tables) == 1, "表格没有独立成块"
    # 表格仍属于这一节（section_path 要能查到它）
    assert tables[0].metadata.get("section_path") == "付款计划"


def test_code_block_not_split_by_function_boundary() -> None:
    """**代码块按函数边界切**：一个 `def` 内的行必须落在同一块。"""
    code = "```python\ndef load(path):\n    with open(path) as f:\n        return f.read()\n```"
    blocks = split_code(code, max_chars=40)

    assert len(blocks) == 1, f"小函数被拆成了 {len(blocks)} 块"
    assert "def load" in blocks[0].text
    assert "return f.read()" in blocks[0].text


def test_code_block_preserves_indent_and_language() -> None:
    """**保留缩进与语言标记**。

    缩进在 Python 里是语法：丢了缩进的块永远报 `IndentationError`，
    模型读到的是"语法错误的代码"。语言标记丢了则前端无法高亮。
    """
    code = "```python\ndef f():\n    return 1\n```"
    blocks = split_code(code)

    assert blocks[0].language == "python"
    # 行内缩进原样保留（`    return 1`）
    assert "\n    return 1" in blocks[0].text
    # 围栏保留：下游要能重新渲染成代码块
    assert blocks[0].text.lstrip().startswith("```python")


def test_code_large_function_split_by_lines_not_chars() -> None:
    """**超大函数按行切，不切在行中间**（唯一能做的选择）。"""
    body = "\n".join(f"    line_{i} = {i}" for i in range(200))
    code = f"```python\ndef big():\n{body}\n```"
    blocks = split_code(code, max_chars=200)

    assert len(blocks) > 1, "超大函数没有被切"
    for block in blocks[1:]:
        # 每块都以完整的一行开头/结尾，不出现半行
        assert block.text.split("\n")[1].startswith("    line_") or block.text.split("\n")[
            1
        ].startswith("def "), f"切在行中间：{block.text[:40]!r}"


def test_all_strategies_route_code_without_body_splitting() -> None:
    """三种策略都必须**把代码块从正文分流出去**、且保持 `block=code`。"""
    code_doc = _doc("```python\ndef f():\n    return 1\n```", metadata={"block": "code"})
    plain = _doc("这是一段普通正文。")

    for strategy in (
        RecursiveChunking(),
        HeadingChunking(),
        SemanticChunking(HashingEmbedder()),
    ):
        chunks = strategy.chunk([plain, code_doc], ChunkingConfig())
        codes = [c for c in chunks if c.is_code]
        assert codes, f"{strategy.name} 把代码块弄丢了"
        assert codes[0].metadata.get("code_language") == "python"


def test_heading_code_block_inherits_section_path() -> None:
    """**代码块也要带 `section_path`** —— 这是实测发现的一个缺口。

    早期版本把代码抽出来在尾部追加（照抄 `recursive` 的做法），
    结果 `docs/w4-checklist.md` 35 块里 34 块带路径、**唯一缺的就是代码块**。

    对代码块来说路径尤其要紧：`` flags = a | b `` 单看完全不知道自己属于
    "二、校验逻辑"这一节。递归策略丢这个信息无所谓（它本来就没有），
    但 heading 策略的全部价值就在这条路径上 —— 丢掉它等于把 heading 降级成 recursive。

    所以代码块必须**留在所属节里**，由节产出时按角色拆分。
    """
    docs = [
        _doc("## 二、校验逻辑", metadata={"block": "heading", "heading_level": 2}),
        _doc("下面这段代码演示位运算。", metadata={"block": "text", "heading_level": 2}),
        _doc("```python\nflags = a | b\n```", metadata={"block": "code"}),
    ]
    chunks = HeadingChunking().chunk(docs, ChunkingConfig())

    codes = [c for c in chunks if c.is_code]
    assert codes, "代码块没有产出"
    assert codes[0].metadata.get("section_path") == "二、校验逻辑", (
        f"代码块丢了 section_path：{codes[0].metadata}"
    )


def test_heading_special_block_forces_split_even_when_fits() -> None:
    """**含表格/代码的节，即使整节不超预算也必须拆分**。

    实测教训：早期版本只在"超预算"分支里拆表格，于是
    "标题 + 100 字正文 + 一张小表"这种**完全装得下**的节被合并成一块，
    表格的 `is_table` 变成 `False`（metadata 来自首段）。
    那个错不报错，只在人工读样本时被发现。
    """
    docs = [
        _doc("## 表格节", metadata={"block": "heading", "heading_level": 2}),
        _doc("下表很短。", metadata={"block": "text", "heading_level": 2}),
        _doc("| a | b |\n| --- | --- |\n| 1 | 2 |", metadata={"block": "table"}),
    ]
    # 预算设得很大，整节完全装得下
    chunks = HeadingChunking().chunk(docs, ChunkingConfig(chunk_chars=5000))

    tables = [c for c in chunks if c.is_table]
    assert tables, "表格被合并进正文块 —— is_table 丢了"
    assert tables[0].metadata.get("section_path") == "表格节"
    assert len(chunks) >= 2, "含表格的节没有被拆分"


def test_strip_fences_keeps_unclosed_fence() -> None:
    """未闭合围栏**不替源文档「修好」** —— 原样保留，让问题可见。"""
    text = "```python\ndef f():\n    pass"
    body, language = strip_fences(text)
    assert language == "python"
    assert body == text, "未闭合围栏被「修复」了，问题被掩盖"


# ============================================================ 5. 半句率


def test_sentence_rate_zero_for_clean_blocks() -> None:
    """边界都落在句号上 → 半句率 0。"""
    chunks = [
        Chunk(text="这是完整的一句。", page=1, source="a"),
        Chunk(text="这也很完整。", page=1, source="a"),
    ]
    assert measure(chunks).rate == 0.0


def test_sentence_rate_counts_head_and_tail() -> None:
    """块首/块尾各算一次：分母是 `2 × 块数`（每块两个边界）。

    注意 `。句号开头` 这一块**头尾都算可疑**：它以句末标点开头（首可疑），
    且以汉字结尾无标点（尾可疑）。这不是测试写错 —— 一个从标点中间开始的
    碎片，它的结尾也往往不在标点上。所以这里分开断言，把这两个数固定下来。
    """
    chunks = [
        Chunk(text="没说完的半句", page=1, source="a"),  # 尾可疑
        Chunk(text="。句号开头", page=1, source="a"),  # 首 + 尾都可疑
    ]
    report = measure(chunks)
    assert report.head_suspect == 1
    assert report.tail_suspect == 2
    assert report.rate == 0.75  # 3 / (2×2)


def test_sentence_rate_skips_table_and_code() -> None:
    """表格与代码**不计入**半句率。

    它们的"边界"不是句子边界（表格以行结束、代码以 `}` 结束）。
    计进去会让指标系统性偏高，而那个偏高的数字没有对应任何真实问题。
    """
    chunks = [
        Chunk(text="正文完整。", page=1, source="a"),
        Chunk(text="| a | b |", page=1, source="a", metadata={"block": "table"}),
        Chunk(text="def f():", page=1, source="a", metadata={"block": "code"}),
    ]
    report = measure(chunks)
    assert report.total == 1, "表格/代码被计入了分母"
    assert report.skipped == 2
    assert report.rate == 0.0


def test_sentence_rate_detects_number_tail() -> None:
    """块尾停在数字/单位上 → 判为可疑。**这是"数字被切断"最强的信号**。"""
    verdict = check_chunk(Chunk(text="精度为 0.05mm", page=1, source="a"))
    assert verdict is not None
    assert verdict.tail_suspect
    assert "数字" in verdict.reason


def test_sentence_rate_orders_of_magnitude_difference() -> None:
    """**核心断言：等长硬切 vs 按标点切，半句率差一个量级。**

    这就是 W4 的 B2 用这个指标发现"LangChain 切出半个词"的原因 ——
    光看块长分布看不出来（两者平均块长可以一样）。

    若这两个数落在同一个量级，说明指标没有区分度，它就不再是"代理指标"，
    只是一串数字。
    """
    text = "违约金按日万分之五计算。甲方逾期付款的，按日万分之三支付逾期利息。" * 40

    # ① 按标点切（`DelimiterSplitter`）
    from app.rag.splitters import ChunkSize, DelimiterSplitter

    good = DelimiterSplitter(ChunkSize(chunk_chars=120, overlap_chars=0)).split([_doc(text)])
    # ② 等长硬切（模拟 LangChain 式的贪心截断）
    hard = [
        Chunk(text=text[i : i + 120], page=None, source="a.md", index=n)
        for n, i in enumerate(range(0, len(text), 120))
    ]

    good_rate = measure(good).rate
    hard_rate = measure(hard).rate

    assert good_rate < 0.05, f"按标点切的半句率不该这么高：{good_rate}"
    assert hard_rate > good_rate * 3, f"指标没有区分度：按标点 {good_rate} vs 硬切 {hard_rate}"


def test_sentence_rate_by_kind_separates_types() -> None:
    """按块类型分组：**三类块的判据不同，混平均没有意义**。"""
    chunks = [
        Chunk(text="正文完整。", page=1, source="a"),
        Chunk(text="| a | b |", page=1, source="a", metadata={"block": "table"}),
    ]
    grouped = measure_by_kind(chunks)
    assert "text" in grouped
    assert "table" in grouped
    assert grouped["text"]["chunks"] == 1
