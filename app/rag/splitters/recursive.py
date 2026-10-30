"""递归切分：优先在**语义边界**断，实在没有才硬切。

**为什么不是"每 500 字切一刀"**：那样切出来的块，第一个字常常是上个句子
的尾巴、最后一个字常常是下个句子的开头。检索命中了这种块，模型拿到的是
两段半截话 —— 答案要么错，要么得靠猜。

分层边界的顺序是刻意排的：

    段落（\\n\\n）  →  换行（\\n）  →  句末标点（。！？；）  →  空格  →  硬切

越靠前越"语义完整"，越靠后越"只是字符"。这跟 `summarize._split_long_text`
的思路一脉相承，但那边只需要"别把一句话劈成两半"，
这边还要输出能入库的稳定块，所以多做了两件事：

1. **继承页码**：一个块跨了 p.3–p.5，`Chunk.pages` 要记全，`Chunk.page` 记起始页。
   （见 `schema.py` 模块 docstring：`page` 事后补不回来。）
2. **重叠**：相邻块共享 `overlap_chars` 个字符。重叠取的是**前一块的尾部**，
   于是"条件句被切断"时，下一块开头仍能读到条件。

**已知取舍：重叠会产生近重复块**。检索时可能同时命中 A 和 A 的尾部（=B 的开头），
浪费上下文窗口。这里不做特殊处理 —— 用 `Chunk.id` 去重就够了，
而"为了省这点窗口去做语义去重"会引入远超收益的复杂度。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from app.rag import Document
from app.rag.schema import Chunk
from app.rag.splitters.base import ChunkSize

logger = logging.getLogger(__name__)

# 语义边界，从"最完整"到"最破碎"排列。
# 正则用 `re.split` 的捕获组保留分隔符 —— 去掉标点的块会让模型读起来不自然。
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ", " ")


class RecursiveSplitter:
    """按分隔符优先级递归切块，块间带重叠。"""

    name = "recursive"

    def __init__(self, size: ChunkSize | None = None) -> None:
        self.size = size or ChunkSize()

    def split(self, documents: Sequence[Document]) -> list[Chunk]:
        """主入口：逐 Document 切块，页码随块继承。"""
        chunks: list[Chunk] = []
        for doc in documents:
            text = doc.text.strip()
            if not text:
                # 空 Document 直接跳过：产出一个空块会污染索引，
                # 而"为什么这批少了一条"比"为什么多了一条空的"难查得多。
                continue
            pieces = self._split_text(text)
            for piece in pieces:
                chunks.append(
                    Chunk(
                        text=piece,
                        page=doc.page,
                        source=doc.source,
                        # metadata 里放的是**可过滤**的事实，不是调试信息
                        metadata={**doc.metadata, "splitter": self.name},
                        # 全局递增：排序和调试都要一个稳定序号。
                        # 但**它不是主键** —— 主键是内容派生的 `Chunk.id`（见 schema.py）。
                        index=len(chunks),
                        pages=(doc.page,) if doc.page is not None else (),
                    )
                )
        return chunks

    # ------------------------------------------------------------------ 内部

    def _split_text(self, text: str) -> list[str]:
        """把单段文本切成若干块，带重叠。

        用"递归下降"而不是"先按段落分再拼接"：段落常常短于块长
        （合同的一个条款就是一段），拼接是必要的；但拼接时必须记得
        **哪里是可以断的** —— 递归先把文本按边界切碎，再往回贪心合并，
        边界信息就始终存在。
        """
        atoms = self._atomize(text, 0)
        return self._merge(atoms)

    def _atomize(self, text: str, level: int) -> list[str]:
        """按第 `level` 级分隔符把文本切到"每片都 <= chunk_chars"。"""
        if len(text) <= self.size.chunk_chars:
            return [text]
        if level >= len(_SEPARATORS):
            # 所有分隔符都用尽了还没切小：只可能是"一整段没有标点的长文"
            # （例如代码块或 OCR 出来的残页）。硬切是唯一出路。
            return self._hard_cut(text)

        sep = _SEPARATORS[level]
        parts = self._split_keep_sep(text, sep)
        if len(parts) == 1:
            # 这一级分隔符在文本里不存在，直接降级，别做无用功
            return self._atomize(text, level + 1)

        atoms: list[str] = []
        for part in parts:
            atoms.extend(self._atomize(part, level + 1))
        return atoms

    @staticmethod
    def _split_keep_sep(text: str, sep: str) -> list[str]:
        """按分隔符切分并**把分隔符留在前一片末尾**。

        丢掉句号会让块看起来像被截断的残句；留在末尾则读起来是完整的句子。
        代价是块尾多一个字符，可以忽略。
        """
        if sep not in text:
            return [text]
        pieces: list[str] = []
        buffer = ""
        i = 0
        while i < len(text):
            if text.startswith(sep, i):
                buffer += text[i : i + len(sep)]
                pieces.append(buffer)
                buffer = ""
                i += len(sep)
            else:
                buffer += text[i]
                i += 1
        if buffer:
            pieces.append(buffer)
        return [p for p in pieces if p]

    def _hard_cut(self, text: str) -> list[str]:
        """最后手段：等长硬切。**只在没有任何分隔符时发生**。"""
        step = self.size.chunk_chars
        return [text[i : i + step] for i in range(0, len(text), step)]

    def _merge(self, atoms: list[str]) -> list[str]:
        """把原子片贪心合并到 `chunk_chars` 以内，并加尾部重叠。"""
        budget = self.size.chunk_chars
        joined: list[str] = []
        buffer = ""
        for atom in atoms:
            if buffer and len(buffer) + len(atom) > budget:
                joined.append(buffer)
                # 重叠：上一块尾部 overlap_chars 字符作为下一块的开头。
                # **注意这里不能直接拼上去**：拼上去再和 atom 合并会让新块
                # 一开场就超预算，必须让重叠计入本块长度。
                tail = buffer[-self.size.overlap_chars :] if self.size.overlap_chars else ""
                buffer = tail + atom
            else:
                buffer += atom
        if buffer:
            joined.append(buffer)

        # 合并后可能因"尾部重叠 + 原子片"而超预算，统一裁一次。
        # 裁到 chunk_chars 而不是不裁：超长块会让 embedding 请求变贵且可能被截断，
        # 那是"静默降级"，比这里多一个循环糟得多。
        return [piece[:budget].strip() for piece in joined if piece.strip()]


__all__ = ["RecursiveSplitter"]
