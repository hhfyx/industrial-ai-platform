"""Markdown loader：按标题层级切成逻辑块。

**Markdown 有天然的语义边界：标题。** 不用它，就得按字数硬切 ——
那会把"## 付款条款"和"## 违约条款"混成一段，摘要时模型会串台。

切分策略：**遇到标题就开新块**。这样每个块都是一个语义单元，
而且块的第一行就是它的小标题，下游摘要时不至于"摘要了一段不知道在讲什么的文字"。

**保留标题本身**：很多天真实现会把标题当"格式"丢掉，只留正文。
但标题恰恰是**最浓缩的信息**（"违约责任"四个字概括了整节内容），
丢掉它等于扔掉最便宜的摘要来源。

**不处理表格的合并/对齐**：Markdown 表格在纯文本里已经是 `| a | b |` 形式，
与 DOCX 的处理保持一致，下游统一按文本对待。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from app.rag import Document, LoadResult
from app.rag.loaders.pdf import normalize_text

logger = logging.getLogger(__name__)

# ATX 风格标题：`# ` 到 `###### `（前面允许有缩进）
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")

# 保留这些块，且不把整个文件当代码块
_FENCE = re.compile(r"^\s*(```|~~~)")


def split_by_headings(text: str) -> list[tuple[int, str]]:
    """按标题切块，返回 `[(标题级别, 块文本), ...]`。

    标题级别用 0 表示"文档开头、第一个标题之前的内容"。
    """
    blocks: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_level = 0
    in_fence = False

    for line in text.split("\n"):
        # 代码块里的 `#` 是注释，不是标题 —— 必须在围栏内跳过标题识别
        if _FENCE.match(line):
            in_fence = not in_fence
            current_lines.append(line)
            continue

        match = None if in_fence else _HEADING.match(line)
        if match:
            # 收束上一块
            if current_lines:
                blocks.append((current_level, "\n".join(current_lines)))
            current_level = len(match.group(1))
            current_lines = [line]  # 标题行本身保留
        else:
            current_lines.append(line)

    if current_lines:
        blocks.append((current_level, "\n".join(current_lines)))

    return blocks


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

    for _level, block in blocks:
        text = normalize_text(block)
        if not text:
            continue
        result.documents.append(Document(source=file_path.name, page=None, text=text))

    if not result.documents:
        result.warnings.append(f"{file_path.name} 没有可提取的内容")

    logger.info("Markdown 加载完成：%s（%d 块）", file_path.name, len(result.documents))
    return result


__all__ = ["load_markdown", "split_by_headings"]
