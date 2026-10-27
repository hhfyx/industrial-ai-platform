"""统一文档模型：所有 loader 的唯一输出格式。

**为什么必须有这个"最小公约数"**：PDF / DOCX / Markdown 的结构差异极大，
但下游（切分、摘要、检索）只关心三件事：**来自哪个文件、第几页、这段文字是什么**。

只要所有 loader 都吐 `Document`，下游就完全不需要知道源文件是什么格式 ——
这正是"加载"和"处理"能解耦的原因，也是 W4 直接复用这套东西的前提。

**为什么 `page` 对非分页格式也要有值**：
Markdown / DOCX 没有物理页，但强行给 `page=1` 会让"第 3 页"这类概念在这个模型里
变成谎言。所以 `page` 的类型是 `int | None`：
- 分页格式（PDF）：真实页码，从 1 开始
- 非分页格式：`None`，表示"这个源没有页的概念"

下游要分页时自己决定"按多少字符算一页"，那是切分策略的事，不该由 loader 假装。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """一段来自某个源的文本（loader 的唯一产物）。

    不可变（frozen）：loader 产出后就该是只读的。
    下游如果要做清洗，应该产出**新的** `Document`，而不是就地修改 ——
    否则"原始文本"和"清洗后文本"的边界会消失，排查时无法回答
    "到底是模型读错了还是我们切错了"。
    """

    source: str
    """来源标识：文件名（不含目录），便于在日志/前端显示。"""

    page: int | None
    """页码，从 1 开始；非分页格式为 None。"""

    text: str
    """该页/该段的正文。已做**最小**清洗（见 `normalize_text`），不做语义加工。"""

    def __post_init__(self) -> None:
        if self.page is not None and self.page < 1:
            raise ValueError(f"page 必须 >= 1 或为 None，收到 {self.page}")

    @property
    def location(self) -> str:
        """人类可读的位置描述，用于日志与报错。"""
        return f"{self.source}:p{self.page}" if self.page is not None else self.source


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


__all__ = ["Document", "LoadResult"]
