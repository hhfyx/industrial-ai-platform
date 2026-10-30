"""Markdown loader：按 `#` 分节，标题层级与代码块都进 metadata。

**Markdown 有天然的语义边界：标题。** 不用它，就得按字数硬切 ——
那会把"## 付款条款"和"## 违约条款"混成一段，摘要时模型会串台。

切分策略：**遇到标题就开新块**。每个块是一个语义单元，且块的第一行就是它的
小标题，下游摘要时不至于"摘要了一段不知道在讲什么的文字"。

**保留标题本身**：很多天真实现会把标题当"格式"丢掉，只留正文。
但标题恰恰是**最浓缩的信息**（"违约责任"四个字概括了整节内容），
丢掉它等于扔掉最便宜的摘要来源。

**代码块识别必须做**：Markdown 代码里的 `#` 是注释，不是标题。
不区分围栏的实现会把一段 Python 切开，然后在切开的缝里插入一个新"章节" ——
这种错在 review 时很难看出来，因为输出仍然是"合法的块"。

**表格单独成块**：Markdown 表格行以 `|` 开头，与 PDF/DOCX 那边统一走
`block="table"`，让切分器一并按"表格不切碎"处理。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from app.rag import Document, LoadResult
from app.rag.loaders.clean import normalize_text

logger = logging.getLogger(__name__)

# ATX 风格标题：`# ` 到 `###### `（前面允许最多 3 个空格缩进）
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")

# 围栏：``` 或 ~~~
_FENCE = re.compile(r"^\s*(```|~~~)")

# Markdown 表格行：以 | 开头（允许前导空格）
_TABLE_ROW = re.compile(r"^\s*\|")


def split_by_headings(text: str) -> list[tuple[int, str]]:
    """按标题切块，返回 `[(标题级别, 块文本), ...]`。

    标题级别用 0 表示"文档开头、第一个标题之前的内容"。

    **围栏内的标题不切**：代码块的 `#` 是注释。这条不是可选的优化，
    见模块 docstring。
    """
    blocks: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_level = 0
    in_fence = False

    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
            current_lines.append(line)
            continue

        match = None if in_fence else _HEADING.match(line)
        if match:
            if current_lines:
                blocks.append((current_level, "\n".join(current_lines)))
            current_level = len(match.group(1))
            current_lines = [line]  # 标题行本身保留
        else:
            current_lines.append(line)

    if current_lines:
        blocks.append((current_level, "\n".join(current_lines)))

    return blocks


def _split_table_blocks(block: str, level: int) -> list[tuple[int, str, str]]:
    """把一块内容里的**表格行**摘出来单独成块。

    返回 `[(level, block_type, text), ...]`，`block_type` 是 `text` / `table`。

    **为什么不直接把整块都算表格**：Markdown 的一节通常"一段正文 + 一张表 + 一段正文"。
    把整节都标成表格，切分器就不会拆它 —— 但正文本来该按语义边界切。
    所以只把连续的 `|` 行摘出来。

    **围栏内的 `|` 不是表格**：代码块里出现 `if a | b:` 或 `|` 位运算非常常见。
    不识别围栏的话，一段代码会被标成"表格"然后禁止切分 ——
    结果是超大块进 embedding，或者表格语义被错误挂到代码上。
    这与 `split_by_headings` 里"围栏内的 `#` 不是标题"是同一条原则。
    """
    pieces: list[tuple[int, str, str]] = []
    buffer: list[str] = []
    table: list[str] = []
    in_fence = False

    def flush_text() -> None:
        nonlocal buffer
        if buffer:
            pieces.append((level, "text", "\n".join(buffer)))
            buffer = []

    def flush_table() -> None:
        nonlocal table
        if table:
            pieces.append((level, "table", "\n".join(table)))
            table = []

    for line in block.split("\n"):
        if _FENCE.match(line):
            # 围栏开关本身属于正文，而且必须先把待收的表格收掉
            flush_table()
            in_fence = not in_fence
            buffer.append(line)
            continue
        if not in_fence and _TABLE_ROW.match(line):
            flush_text()
            table.append(line)
        else:
            flush_table()
            buffer.append(line)
    flush_table()
    flush_text()
    return pieces


def load_markdown(path: str | Path) -> LoadResult:
    """加载 Markdown 文件。

    `page=None`：Markdown 没有页的概念。
    """
    file_path = Path(path)
    result = LoadResult()

    try:
        raw = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # 尝试 GBK（Windows 上常见）
        try:
            raw = file_path.read_text(encoding="gbk")
            result.warnings.append(f"{file_path.name} 不是 UTF-8，已按 GBK 解码")
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"无法解码 {file_path.name}：{exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法读取 {file_path.name}：{exc}") from exc

    blocks = split_by_headings(raw.replace("\r\n", "\n"))

    table_count = 0
    heading_count = 0
    for level, block in blocks:
        for piece_level, block_type, piece in _split_table_blocks(block, level):
            text = normalize_text(piece)
            if not text:
                continue

            metadata: dict[str, str | int | float | bool] = {"block": block_type}
            if block_type == "table":
                table_count += 1
                metadata["page_chars"] = len(text)
            elif piece_level > 0:
                # 该节以小标题开头：层级记下来（与 DOCX 的 heading_level 同名，
                # 下游可以完全不区分源格式）。标题块额外标成 heading。
                metadata["heading_level"] = piece_level
                # **只有"整块就是一行标题"才标 heading**。块里若还有正文，
                # 标成 heading 会让下游误以为整段都是标题 —— 那是元数据说谎，
                # 比没有元数据更糟（下游会据此做决策）。
                body_lines = [ln for ln in piece.split("\n") if ln.strip()][1:]
                if piece.lstrip().startswith("#") and not body_lines:
                    metadata["block"] = "heading"
                    heading_count += 1

            result.documents.append(
                Document(source=file_path.name, page=None, text=text, metadata=metadata)
            )

    if not result.documents:
        result.warnings.append(f"{file_path.name} 没有可提取的内容")

    logger.info(
        "Markdown 加载完成：%s（%d 块，其中 %d 标题 / %d 表格）",
        file_path.name,
        len(result.documents),
        heading_count,
        table_count,
    )
    return result


__all__ = ["load_markdown", "split_by_headings"]
