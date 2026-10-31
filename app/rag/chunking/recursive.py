"""策略一：递归字符切分（`RecursiveChunking`）。

## 它依赖什么、在哪退化

依赖 **分隔符分布**：段落 → 换行 → 句末 → 逗号 → 硬切，逐级降级。
它不看结构、不看语义，所以**任何文本都能跑**，这是它作为默认策略的全部理由。

退化点（预判写在 `docs/chunking-predictions.md`，这里是实现上的对策）：

| 文档形态 | 失效方式 | 本实现的对策 |
| --- | --- | --- |
| 表格密集 | 表格被切开，行列关系丢失 | `doc.is_table` 整张不切（契约要求） |
| 并列条款（每条一行、都很短） | **把 N 个独立条款塞进同一块** | 无 —— 这是本策略的固有缺陷，留给实测暴露 |
| 无标点的 OCR 残页 | 逐级降级到硬切 | 硬切兜底（块宽 = `chunk_chars`） |

## 与 W4 `DelimiterSplitter` 的关系

**不是替代，是包装**。`DelimiterSplitter` 已经是"中文分隔符优先级 + 只合并完整原子片"
的成熟实现，且被 40 多条测试锁住（块尾落在句末标点、冗余率 9.4% 等）。
这里只做三件事：把配置翻译成 `ChunkSize`、补 `splitter` 元数据、
统一 `pages` 继承。**重写一遍只会丢掉那些已经验证过的性质。**

> 这正是"接口可以提前定、实现不要提前写"的另一面：实现已经验证过了，
> 新的统一签名应该**包住它**，而不是推倒它。
"""

from __future__ import annotations

from app.rag.chunking.base import ChunkingConfig
from app.rag.schema import Chunk, Document
from app.rag.splitters import ChunkSize, DelimiterSplitter


class RecursiveChunking:
    """按分隔符优先级递归切分。"""

    name = "recursive"

    def __init__(self, *, splitter: DelimiterSplitter | None = None) -> None:
        """`splitter` 可注入：测试要能塞一个"每次都硬切"的破坏性实现来验证契约。"""
        self._splitter = splitter

    def chunk(self, docs: list[Document], cfg: ChunkingConfig) -> list[Chunk]:
        """切分。`pages` 继承与表格保护由 `DelimiterSplitter` 内部完成。"""
        if not docs:
            return []

        splitter = self._splitter or DelimiterSplitter(
            ChunkSize(chunk_chars=cfg.chunk_chars, overlap_chars=cfg.overlap_chars)
        )
        raw = splitter.split(docs)

        # **重建一遍而不是直接返回 raw**：`DelimiterSplitter` 写入的
        # `metadata["splitter"]` 是 `"delimiter"` / `"table"`，而本策略对外叫
        # `"recursive"`。不统一的话，`metadata["splitter"]` 在策略之间不可比，
        # "这批向量是哪版策略切的"这个排查线索就失效了。
        #
        # 表格块保留 `"table"` 标记：那个值说明的不是"哪个策略"，
        # 而是"这块不参与切分"。混成一个字段是原实现的既成事实，
        # 这里用 `chunk_strategy` 额外记一份，让两者互不干扰。
        rebuilt: list[Chunk] = []
        for position, chunk in enumerate(raw):
            rebuilt.append(
                Chunk(
                    text=chunk.text,
                    page=chunk.page,
                    source=chunk.source,
                    metadata={
                        **chunk.metadata,
                        "splitter": self.name,
                        # 表格块的真实来源标记另存，下游（引用"（表格）"）照旧可用
                        "chunk_strategy": "table" if chunk.is_table else self.name,
                    },
                    index=position,
                    pages=chunk.pages,
                    id=chunk.id,
                )
            )
        return rebuilt


__all__ = ["RecursiveChunking"]
