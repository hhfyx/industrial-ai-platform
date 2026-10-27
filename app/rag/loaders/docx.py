"""DOCX loader：python-docx 抽段落 + 表格，统一成 `Document`。

**DOCX 没有"页"，所以 `page=None`。** 这不是偷懒，是诚实：
Word 的"页"是**渲染结果**，同一份文件在不同打印机/字号下页数不同，
文件里根本没有存页码。硬编一个 `page=1` 会让"第 3 页"变成谎言。

**段落和表格都要抽**：合同、报告里的关键信息（金额、条款、时间表）经常在表格里。
只抽 `paragraphs` 会丢掉大部分有价值的结构化内容 —— 这是最常见的 DOCX 抽取失误。

表格转成**文本行**（`单元格1 | 单元格2`）而不是保留结构：
下游的切分/摘要都按纯文本处理，保留二维结构反而要多一层模型转换，
而那层转换会引入幻觉风险。
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.rag import Document, LoadResult
from app.rag.loaders.pdf import normalize_text

logger = logging.getLogger(__name__)


def _table_to_lines(table: object) -> list[str]:
    """把一张表转成文本行。

    用 `|` 分隔单元格，与 Markdown 表格接近 —— 模型对这种格式最熟悉，
    摘要时也更容易看出"这是一行数据"而不是一句话。
    """
    lines: list[str] = []
    for row in table.rows:  # type: ignore[attr-defined]
        cells = []
        for cell in row.cells:
            value = normalize_text(cell.text).replace("\n", " ")
            cells.append(value)
        # 全空行没有信息量，跳过（合并单元格会产生大量空行）
        if any(c.strip() for c in cells):
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def load_docx(path: str | Path) -> LoadResult:
    """加载 DOCX。

    输出策略：**每个非空段落一个 `Document`**，表格则整张作为一个 `Document`。

    **为什么一段一个 Document 而不是整篇一个**：
    下游要按 token 预算切分，如果 loader 已经给出"段"这个天然边界，
    切分就能尽量不破坏段落。整篇一个 Document 的话，切分器只能按字数硬切，
    容易把一句话砍成两半。
    """
    from docx import Document as DocxDocument

    file_path = Path(path)
    result = LoadResult()

    try:
        doc = DocxDocument(str(file_path))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法打开 DOCX {file_path.name}：{type(exc).__name__}: {exc}") from exc

    paragraph_count = 0
    for para in doc.paragraphs:
        text = normalize_text(para.text or "")
        if not text:
            continue
        paragraph_count += 1
        result.documents.append(Document(source=file_path.name, page=None, text=text))

    table_count = 0
    for table in doc.tables:
        lines = _table_to_lines(table)
        text = normalize_text("\n".join(lines))
        if not text:
            continue
        table_count += 1
        result.documents.append(Document(source=file_path.name, page=None, text=text))

    if not result.documents:
        result.warnings.append(f"{file_path.name} 没有可提取的文本或表格 —— 可能是空文档或纯图片")
    else:
        logger.info(
            "DOCX 加载完成：%s（%d 段 + %d 表格）",
            file_path.name,
            paragraph_count,
            table_count,
        )

    return result


__all__ = ["load_docx"]
