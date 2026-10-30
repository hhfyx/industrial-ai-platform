"""RAG 包入口：只 re-export **结构级契约**，不放实现。

**为什么这里只有两个结构体**：`app/rag/` 下每个子目录都是一个可替换的实现
（loaders / splitters / embedding / stores / generation），它们共用 `Document`
与 `Chunk` 这两个"货币"。入口文件如果还顺手导出 `chunk_documents` 之类的函数，
调用方就会形成"import app.rag 拿到全部能力"的习惯 —— 那条路径一多，
将来换切分策略就得改所有调用点。

所以规矩是：**结构从这里拿，实现从子模块拿**。

    from app.rag import Document, Chunk             # 契约
    from app.rag.splitters import split_documents   # 实现

---

本文件在 W4 之前承担"统一文档模型"的角色（`Document` / `LoadResult` 都写在这里）。
现在 `Document` 已迁到 [`schema.py`](schema.py)，因为它必须和 `Chunk` 放在一起、
一起演进（理由见那个模块的 docstring）。`LoadResult` 留在原地 ——
它是 **loader 的返回类型**，只被加载链路用到，不算跨阶段契约。

**为什么 `page` 对非分页格式是 `None` 而不是 1**：
Markdown / DOCX 没有物理页，强行给 `page=1` 会让"第 3 页"这类概念在这个模型里
变成谎言。分页格式（PDF）给真实页码，非分页格式给 `None`，表示"这个源没有页的概念"。
下游要分页时自己决定"按多少字符算一页"，那是切分策略的事，不该由 loader 假装。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.rag.schema import Chunk, Document, Metadata, MetadataValue, restore_location

__all__ = [
    "Chunk",
    "Document",
    "LoadResult",
    "Metadata",
    "MetadataValue",
    "restore_location",
]


@dataclass
class LoadResult:
    """一次加载的完整结果。

    **为什么需要 `warnings` 而不是直接抛异常**：
    "某一页是扫描件所以没文本"是**文档的正常状态**，不是加载失败。
    如果为此抛异常，一份 50 页的 PDF 只要有 1 页是图片就整个加载不了 ——
    那显然不对。所以：能读的读出来，读不到的记为 warning，情报不丢。
    """

    documents: list[Document] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(len(d.text) for d in self.documents)

    @property
    def page_count(self) -> int:
        """页数：有页码的取最大页码，否则为文档段数。"""
        pages = [d.page for d in self.documents if d.page is not None]
        return max(pages) if pages else len(self.documents)

    def extend(self, other: LoadResult) -> None:
        self.documents.extend(other.documents)
        self.warnings.extend(other.warnings)

    def summary(self) -> dict[str, Any]:
        """给 CLI / API 返回的结构化摘要。"""
        return {
            "documents": len(self.documents),
            "pages": self.page_count,
            "total_chars": self.total_chars,
            "warnings": len(self.warnings),
        }
