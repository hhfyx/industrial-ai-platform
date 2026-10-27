"""Loader 测试（W4 复用基础）。离线，用 `tests/fixtures/` 里的生成样本。

**fixture 是脚本生成的**（`tests/fixtures/make_fixtures.py`），不是二进制塞进仓库。
好处：review 时能读到"这份 50 页 PDF 每页有什么"，而不是面对一个无法审查的二进制。

覆盖的重点全是"不完美输入"，happy path 只是顺便验证：
- 页码必须对得上（分页格式的核心价值就在这）
- 页眉页脚 / 页码行要被剔除，但**正文一行都不能少**
- 扫描件/空页记 warning 而不崩
- 跨页断句要留下"没说完"的标记
- 三种格式输出同一个 `Document` 模型
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag import Document, LoadResult
from app.rag.loaders import LOADERS, SUPPORTED_EXTENSIONS, load_document
from app.rag.loaders.pdf import (
    _looks_unfinished,
    _strip_page_numbers,
    _strip_repeated_lines,
    normalize_text,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ============================================================ 1. 统一模型


def test_document_rejects_bad_page() -> None:
    """页码必须 >= 1 —— 0 或负数是"从 0 开始"这类混淆的信号，早拦。"""
    with pytest.raises(ValueError, match="page 必须"):
        Document(source="x.pdf", page=0, text="t")


def test_document_location_is_human_readable() -> None:
    assert Document(source="a.pdf", page=3, text="x").location == "a.pdf:p3"
    # 非分页格式没有页码，location 就是文件名
    assert Document(source="a.md", page=None, text="x").location == "a.md"


def test_load_result_summary() -> None:
    result = LoadResult(
        documents=[
            Document(source="a", page=1, text="12345"),
            Document(source="a", page=2, text="678"),
        ],
        warnings=["某页无文本"],
    )
    assert result.total_chars == 8
    assert result.page_count == 2
    assert result.summary()["warnings"] == 1


# ============================================================ 2. 文本清洗


def test_normalize_collapses_whitespace_but_keeps_paragraphs() -> None:
    """压缩多余空格与空行，但**保留段落边界** —— 全压成一行会让摘要串味。"""
    text = "第一段   有  多余空格\n\n\n\n第二段"
    out = normalize_text(text)
    assert out == "第一段 有 多余空格\n\n第二段"


def test_normalize_handles_fullwidth_space_and_crlf() -> None:
    assert normalize_text("a\u3000b\r\nc") == "a b\nc"


def test_normalize_empty() -> None:
    assert normalize_text("") == ""
    assert normalize_text("   \n\t  ") == ""


# ============================================================ 3. 跨页断句启发式


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        ("这句话说完了。", False),
        ("This sentence ends.", False),
        ("问题解决了？", False),
        ("注意到这一点", True),  # 汉字结尾，很可能被切断
        ("the sentence was cut", True),  # 字母结尾
        ("前面有逗号，", True),  # 逗号结尾 = 没说完
        ("", False),
    ],
)
def test_looks_unfinished(tail: str, expected: bool) -> None:
    assert _looks_unfinished(tail) is expected


# ============================================================ 4. 页眉页脚 / 页码行剔除


def test_strip_repeated_header_footer_keeps_body() -> None:
    """**核心断言**：页眉页脚要删，正文一行不能少。

    这正是我踩过的坑 —— 测试数据若每页正文相同，就无法区分
    "正常去页眉"和"丢了正文"。这里显式断言正文仍在。
    """
    pages = [
        ["机密文件", "正文甲 唯一", "正文乙 唯一", "第 1 页"],
        ["机密文件", "正文丙 唯一", "正文丁 唯一", "第 2 页"],
        ["机密文件", "正文戊 唯一", "正文己 唯一", "第 3 页"],
    ]
    cleaned = _strip_repeated_lines(pages)

    for original, result in zip(pages, cleaned, strict=True):
        body = list(original[1:-1])
        for line in body:
            assert line in result, f"正文行被误删：{line}，结果 {result}"
        assert "机密文件" not in result, "页眉应被剔除"


def test_strip_disabled_for_tiny_documents() -> None:
    """少于 3 页不做统计：2 页重复恰好 100%，会把正文当页脚删掉。"""
    pages = [["唯一内容 A"], ["唯一内容 A"]]
    assert _strip_repeated_lines(pages) == pages


def test_page_number_lines_are_stripped_by_pattern() -> None:
    """页码每页都不同，频率统计抓不到 —— 必须用模式识别。

    但**只在首尾位置删**：正文里的孤立数字（金额）是正常内容。
    """
    assert _strip_page_numbers(["第 3 页 / 共 50 页", "正文", "看重"]) == ["正文", "看重"]
    assert _strip_page_numbers(["正文", "第 3 页"]) == ["正文"]
    # 正文中间的数字不动
    assert _strip_page_numbers(["金额 128000 元", "另一行"]) == ["金额 128000 元", "另一行"]


# ============================================================ 5. PDF 真实样本


def test_pdf_page_numbers_are_sequential_and_correct() -> None:
    """**验收核心**：50 页 PDF，页码必须 1..50 连续且对得上。"""
    result = load_document(FIXTURES / "sample50.pdf")
    pages = [d.page for d in result.documents]
    assert pages == list(range(1, 51))


def test_pdf_page_content_matches_its_page_number() -> None:
    """每页内容里的"第 N 章"必须与该页 `page=N` 一致 —— 这是页码正确的实质证明。

    只断言"有 50 个 Document"是不够的：页码全错也能过。
    """
    result = load_document(FIXTURES / "sample50.pdf")
    for doc in result.documents:
        assert doc.page is not None
        assert f"第 {doc.page} 章" in doc.text, f"第 {doc.page} 页内容对不上：{doc.text[:40]}"


def test_pdf_header_and_page_number_are_removed() -> None:
    """页眉文字和页码行都不该出现在正文里（否则 50 页重复 50 次，白烧 token）。"""
    result = load_document(FIXTURES / "sample50.pdf")
    body = "\n".join(d.text for d in result.documents)
    assert "严格保密" not in body, "页眉应被剔除"
    assert "共 50 页" not in body, "页码行应被剔除"


def test_pdf_body_lines_are_preserved() -> None:
    """**反向断言**：剔除要精准，正文一行都不能少。

    每页应有 3 行正文（标题 + 3 句）。只断言行数下限，不锁死具体字数，
    否则改 fixture 就会误报。
    """
    result = load_document(FIXTURES / "sample50.pdf")
    for doc in result.documents:
        lines = [ln for ln in doc.text.split("\n") if ln.strip()]
        assert len(lines) >= 3, f"第 {doc.page} 页正文行数不足：{lines}"


def test_pdf_without_header_also_works() -> None:
    """没有页眉页脚的 PDF 不能被误剥离。"""
    result = load_document(FIXTURES / "sample5_no_header.pdf")
    assert len(result.documents) == 5
    assert result.warnings == []
    for doc in result.documents:
        assert f"第 {doc.page} 章" in doc.text


# ============================================================ 6. 扫描件 / 空页


def test_scanned_pdf_records_warning_not_crash(tmp_path: Path) -> None:
    """整份是"扫描件"（无文本层）：返回空结果 + 明确 warning，**不抛异常**。

    这是本格最重要的行为约定 —— 处理失败 ≠ 内容缺失。
    """
    from reportlab.pdfgen import canvas

    path = tmp_path / "scanned.pdf"
    pdf = canvas.Canvas(str(path))
    for _ in range(3):
        # 只画一个方块，没有任何文字 → 抽不出文本
        pdf.rect(100, 100, 200, 200, fill=1)
        pdf.showPage()
    pdf.save()

    result = load_document(path)

    assert result.documents == []
    assert any("OCR" in w for w in result.warnings), result.warnings


def test_corrupt_pdf_raises_clear_error(tmp_path: Path) -> None:
    """坏文件是**真失败**，必须抛 —— 但错误信息要能看懂。"""
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a real pdf at all")

    with pytest.raises(ValueError, match="无法打开 PDF"):
        load_document(path)


# ============================================================ 7. DOCX


def test_docx_extracts_paragraphs_and_tables() -> None:
    """**表格必须抽出来**：合同里的金额/时间表通常在表格里，只抽段落会丢掉核心信息。"""
    result = load_document(FIXTURES / "sample.docx")
    body = "\n".join(d.text for d in result.documents)

    assert "技术服务合同" in body
    assert "合同编号：HT-2026-001" in body
    # 表格内容
    assert "一期交付" in body
    assert "38,400.00" in body


def test_docx_has_no_page_numbers() -> None:
    """DOCX 没有物理页概念，`page` 必须是 None 而不是硬编 1。

    硬编页码会让"第 3 页"变成谎言（Word 的页是渲染结果，文件里没存）。
    """
    result = load_document(FIXTURES / "sample.docx")
    assert all(d.page is None for d in result.documents)


def test_docx_table_is_rendered_as_pipe_rows() -> None:
    """表格转成 `| a | b |` 文本行：模型对这种格式最熟悉，摘要时能看出是数据行。"""
    result = load_document(FIXTURES / "sample.docx")
    table_docs = [d for d in result.documents if "|" in d.text]
    assert table_docs, "没有找到表格转出的文本"
    assert any("里程碑" in d.text for d in table_docs)


# ============================================================ 8. Markdown


def test_markdown_splits_by_headings() -> None:
    """按标题切块：每个块是语义单元，块首即小标题。"""
    result = load_document(FIXTURES / "sample.md")
    assert len(result.documents) >= 4

    first = result.documents[0]
    assert first.text.startswith("# 工业 AI 平台设计说明")
    # 标题本身**保留**：它是最浓缩的信息，丢掉等于扔掉最便宜的摘要来源
    assert all("#" in d.text for d in result.documents if d.text.startswith("#"))


def test_markdown_code_fence_does_not_split_on_hash() -> None:
    """代码块里的 `#` 是注释，不是标题 —— 必须在围栏内跳过标题识别。"""
    result = load_document(FIXTURES / "sample.md")
    code_docs = [d for d in result.documents if "```" in d.text]
    assert code_docs, "代码块没有被保留"
    # 代码块内不应出现"被切成两块"的情况：整块里既有 def 又有 return
    joined = code_docs[0].text
    assert "def handler" in joined
    assert "return " in joined


def test_markdown_utf8_and_gbk(tmp_path: Path) -> None:
    """Windows 上 GBK 编码的 .md 也要能读，且给出明确 warning。"""
    path = tmp_path / "gbk.md"
    path.write_bytes("# 标题\n\n正文内容。".encode("gbk"))

    result = load_document(path)
    assert result.documents
    assert any("GBK" in w for w in result.warnings), result.warnings


# ============================================================ 9. 注册表


def test_registry_covers_three_formats() -> None:
    assert set(SUPPORTED_EXTENSIONS) >= {".pdf", ".docx", ".md"}


def test_unsupported_extension_raises() -> None:
    with pytest.raises(ValueError, match="不支持的格式"):
        load_document(FIXTURES / "sample.txt")


def test_extension_is_case_insensitive(tmp_path: Path) -> None:
    """`.PDF` 大写也要认 —— Windows 上用户经常拿到大写后缀的文件。"""
    src = FIXTURES / "sample5_no_header.pdf"
    dst = tmp_path / "UPPER.PDF"
    dst.write_bytes(src.read_bytes())
    assert load_document(dst).documents


def test_missing_file_raises() -> None:
    with pytest.raises(ValueError, match="文件不存在"):
        load_document(FIXTURES / "nope.pdf")


def test_all_loaders_return_same_model() -> None:
    """**三格式统一输出的证明**：下游完全不需要知道源文件是什么格式。"""
    for name in ("sample50.pdf", "sample.docx", "sample.md"):
        result = load_document(FIXTURES / name)
        for doc in result.documents:
            assert isinstance(doc, Document)
            assert isinstance(doc.text, str) and doc.text
            assert hasattr(doc, "page")
        assert isinstance(result, LoadResult)


def test_loader_registry_is_the_only_place_to_add_formats() -> None:
    """加格式只改 `LOADERS` 一处。这条测试守的是"分发集中"这个设计。"""
    assert set(LOADERS) == set(SUPPORTED_EXTENSIONS)
