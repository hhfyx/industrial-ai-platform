"""策略二：按标题层级切分（`HeadingChunking`）。

## 它依赖什么、在哪退化

依赖 **结构标记**：Markdown 的 `#` / DOCX 的 `Heading N` 样式。
本项目的 loader 已经把两者统一成 `metadata["heading_level"]`（1-6），
所以这一层**完全不区分源格式** —— 那是 loader 的功劳，不该在这里再判一次。

退化点（预判见 `docs/chunking-predictions.md`）：

| 文档形态 | 失效方式 | 本实现的行为 |
| --- | --- | --- |
| **PDF** | 文件里没有"标题"概念，`heading_level` 一律缺失 | **按页分组**（每页一节），并标 `heading_fallback="page"` |
| 标题粒度极不均匀 | 3000 字节 + 30 字节并存 | 大节在**节内二次切**（此时才用 `overlap_chars`）；小结保持完整 |
| 标题只作装饰 | "第 1 节""Part A"，标题里无主题词 | 仍然切、仍然保留标题行，但**块间没有可区分信息**（固有缺陷） |

## 为什么必须显式暴露"降级到按页切"

PDF 走标题策略时，它不是"按标题切"，它是"按页切" —— **两件事**。
如果 `metadata` 里不写这一条，下游看到的都是 `splitter="heading"` 的块，
于是"标题策略在 PDF 上表现如何"这个实验结论就是假的：
你测的是分页器，不是标题切分器。

所以降级时写 `heading_fallback="page"`，评测脚本按这个字段**分组统计**，
而不是把两种完全不同的行为混进同一个平均值。

## `heading_keep_path` 真正做的事

块被二次切碎之后，第 2、3 片**不再以标题开头** —— 它们丢掉了"我在哪一节"。
`heading_keep_path=True` 时给这些后续片补一行 `【L2 > 标题】`，
让每一片都能自报所属章节。这不是格式美化：`"接入层负责协议转换与限流"`
单独进检索时，不知道自己在讲哪一节。

---

## 这个文件里的三条判据都是**实测逼出来的**，不是设计出来的

第一版实现跑 `docs/w4-checklist.md` 得到 **44 块 = 文档段数** ——
也就是"按标题切分"完全没有合并。三个连续的问题：

| # | 症状 | 根因 | 修法 |
| ---: | --- | --- | --- |
| 1 | 44 块，一段一块 | 把"有 `heading_level`"当成"节的开头" | `_is_section_start`：只认**标题行** |
| 2 | `## 优先级总览` 被拆成"标题 / 表格 / `---`" | 遇到表格就 `current = []`，把节切断 | 表格**不切断节**，只由 `_emit` 保证整张不切 |
| 3 | 27 字的块只有标题 | 父标题下面紧跟子标题，父节没正文 | `_merge_orphan_headings`：纯标题节并进下一节 |

第 3 条正对应 `docs/chunking-quality.md` 的判据三（标题与内容是否分离），
而且**它是可自动判定的** —— 这验证了 `chunking-predictions.md` 里那个附带预判。
"""

from __future__ import annotations

import re

from app.rag.chunking.base import ChunkingConfig, inherited_pages, make_chunk
from app.rag.schema import Chunk, Document
from app.rag.splitters import ChunkSize, DelimiterSplitter

_ATX_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")


class HeadingChunking:
    """按标题层级切分：结构优先，超长才在节内二次切。"""

    name = "heading"

    def chunk(self, docs: list[Document], cfg: ChunkingConfig) -> list[Chunk]:
        """切分。

        ## 分组规则（一条路径 = 一节）

        1. `block == "heading"` 或**首行是 ATX 标题**的段，视为**节的开头**；
           它之后直到下一个同级/更高级标题之前的所有段都属于这一节。
        2. **完全没有标题行**（PDF / 纯文本）→ 按页分组，标 `heading_fallback="page"`。
        3. 本节正文 <= `cfg.chunk_chars` → **整节一块**。
           > 这是本策略相对递归字符的核心优势：不该切的节，一个字都不切。
        4. 超了 → 节内用 `DelimiterSplitter` 二次切（这时才用 `overlap_chars`），
           并按 `cfg.heading_keep_path` 给后续片补标题路径。
        """
        if not docs:
            return []

        chunks: list[Chunk] = []
        for section in self._sections(docs, cfg):
            chunks.extend(self._emit(section, cfg, index_base=len(chunks)))
        return chunks

    # ------------------------------------------------------------------ 分组

    def _sections(self, docs: list[Document], cfg: ChunkingConfig) -> list[list[Document]]:
        """把平铺的 `Document` 列表分组。**跨文档也合并**：一节常被 loader 拆成多段。"""
        if not _has_headings(docs):
            # 降级：没有标题行 → 按页分组（PDF 的实际路径）
            return _group_by_page(docs)

        sections: list[list[Document]] = []
        current: list[Document] = []
        for doc in docs:
            if not doc.text.strip():
                continue
            # **表格不切断节**：它只是"节内一块特殊内容"，节还在继续。
            # 早期版本在这里 `current = []`，把 `## 优先级总览` 拆成
            # "标题行 / 表格 / 表格后的 `---`"，而那个 `---` 成了**无主孤儿块**。
            # 表格的"整张不切"由 `_emit` 保证，与"它属于哪一节"是两件事。
            if _is_section_start(doc, cfg) and current:
                sections.append(current)
                current = [doc]
            else:
                current.append(doc)
        if current:
            sections.append(current)
        return _merge_orphan_headings(sections)

    # ------------------------------------------------------------------ 产出

    def _emit(
        self, section: list[Document], cfg: ChunkingConfig, *, index_base: int
    ) -> list[Chunk]:
        """把一节变成一或多块。"""
        first = section[0]
        joined = "\n".join(doc.text.strip() for doc in section).strip()
        if not joined:
            return []

        # 一节的覆盖范围 = 各段 `pages` 的并集（跨页节同时覆盖 p6、p7）
        section_pages = _union_pages(section)
        extra: dict[str, object] = {
            "section_chars": len(joined),
            "section_docs": len(section),
        }
        title = _heading_title(first)
        level = _heading_level(first)
        if title:
            extra["heading_title"] = title
        if level:
            extra["heading_level"] = level
        fallback = _page_fallback(section)
        if fallback:
            extra["heading_fallback"] = fallback

        # 不超预算（或整节就是一张表）：**整节一块，不做任何二次切**
        if len(joined) <= cfg.chunk_chars or _is_only_tables(section):
            return [
                make_chunk(
                    text=joined,
                    doc=first,
                    index=index_base,
                    strategy=self.name,
                    pages=section_pages,
                    extra_metadata=extra,
                )
            ]

        # 超预算：节内二次切。复用已验证的 `DelimiterSplitter` ——
        # 节内切分面对的问题（中文分隔符、块尾落句末标点）与它完全同一个。
        splitter = DelimiterSplitter(
            ChunkSize(chunk_chars=cfg.chunk_chars, overlap_chars=cfg.overlap_chars)
        )
        # 整节包成一个虚拟 Document 再切，让重叠逻辑走同一份代码。
        synthetic = Document(
            source=first.source,
            page=first.page,
            text=joined,
            metadata=first.metadata,
            pages=section_pages,
        )
        pieces = splitter.split([synthetic])

        out: list[Chunk] = []
        for offset, piece in enumerate(pieces):
            text = piece.text
            # 后续片补标题路径：它们不再以标题开头，需要自报所属章节
            if offset > 0 and cfg.heading_keep_path and title:
                text = f"【{_path_label(level, title)}】\n{text}"
            out.append(
                make_chunk(
                    text=text,
                    doc=first,
                    index=index_base + offset,
                    strategy=self.name,
                    pages=section_pages,
                    extra_metadata={**extra, "section_part": offset},
                )
            )
        return out


# --------------------------------------------------------------------------- 工具


def _is_section_start(doc: Document, cfg: ChunkingConfig) -> bool:
    """这一段是不是**新一节的开头**。

    ## 这个判据错了会让整个策略静默失效（实测第一次就错在这里）

    第一版写的是"`heading_level` 存在且 <= `heading_max_level`"。
    结果 md 切出 **44 块 = 文档段数**，一点都没合并。

    原因在 loader 的语义：**`heading_level` 表示"这一段属于哪一级的节"**，
    它挂在该节的**每一段**上，而不是只挂在标题行上。实测
    （`docs/w4-checklist.md`，44 段）：

        idx 0  level=1  block=text     '# W4 开工清单'
        idx 1  level=2  block=heading  '## 优先级总览'
        idx 2  level=None block=table  '| # | 事项 | …'
        idx 3  level=2  block=text     '---'
        idx 4  level=3  block=text     '### 为什么必须现在就定'

    按"有 level 就开新节"分组 → **每一段都是节开头** → 等于没分组。

    ## 正确判据：标题**行**

    三选一，全部指向"这一行的字面事实"：
    ① `block == "heading"`（loader 判定"整段就是标题"）
    ② 首行是 ATX 标题（`# …`，覆盖"标题 + 正文同一块"的情况）
    ③ 其余情况不算节开头

    `level` 只用来筛"够不够深"（`> heading_max_level` 的标题只是上一节的分点）。
    """
    if doc.is_table:
        return False

    level = _heading_level(doc)
    too_deep = level is not None and level > cfg.heading_max_level

    # ① loader 明确标成"整段是标题"
    if doc.metadata.get("block") == "heading":
        return not too_deep

    # ② 首行是 ATX 标题（"标题 + 正文"合在一个块里）
    first_line = doc.text.strip().split("\n", 1)[0]
    if _ATX_HEADING.match(first_line):
        return not too_deep

    return False


def _has_headings(docs: list[Document]) -> bool:
    """这一批文档里是否有**标题行**。决定走标题分组还是按页降级。"""
    return any(
        doc.metadata.get("block") == "heading"
        or _ATX_HEADING.match(doc.text.strip().split("\n", 1)[0])
        for doc in docs
        if not doc.is_table
    )


def _group_by_page(docs: list[Document]) -> list[list[Document]]:
    """降级分组：每页各成一组，表格与无页文档各成一组。

    **为什么按页而不是把全部并成一节**：并成一节会让 50 页 PDF 变成
    "一个 5000 字的节"再被二次切 —— 结果与递归字符几乎一样，
    但 `splitter="heading"` 这个标签会让实验数据失真。

    无页文档（Markdown 但恰好没有标题行）每段各成一组：它们本来就没有
    "节"的概念，按段是最接近的最小结构单位。
    """
    groups: list[list[Document]] = []
    current_page: object = object()  # sentinel：与任何真实页号都不同

    for doc in docs:
        if not doc.text.strip():
            continue
        if doc.is_table or doc.page is None:
            groups.append([doc])
            current_page = object()
            continue
        if doc.page != current_page:
            groups.append([doc])
            current_page = doc.page
        else:
            groups[-1].append(doc)

    return groups


def _merge_orphan_headings(sections: list[list[Document]]) -> list[list[Document]]:
    """把"只有标题、没有正文"的节**并进下一节**。

    ## 实测暴露的问题（判据三的正面处理）

    `docs/w4-checklist.md` 第一次跑出这种块：

        ─── [8] chars=27 ───
        ## P0-2 · `extract` 延迟同样要处理

    `## P0-2` 下面**紧跟着** `### 为什么周报里漏了它`（深一层）。
    因为 `heading_max_level=3` 把两层都当节边界，父标题那一节就只剩标题行 ——
    **一个 27 字的块，除了标题什么都不含**。检索后果很具体：
    用户问"P0-2 是什么"，命中的是这个 27 字块，模型只能答出标题本身；
    真正的 518 字内容在下一块，**没进上下文**。

    ## 为什么"并进下一节"而不是"并进上一节"

    父标题在语义上**统辖**它下面的子节，所以 `## P0-2` + 子节内容合成一块时，
    块首是父标题、内容来自子节 —— 读者得到"P0-2 的为什么"，这是对的。
    并进上一节会得到"P1-4 的正文 + P0-2 的标题"，那是错的。
    """
    if len(sections) <= 1:
        return sections

    merged: list[list[Document]] = []
    pending: list[Document] = []

    for section in sections:
        if pending:
            section = pending + section
            pending = []
        if _is_heading_only(section):
            pending = section
            continue
        merged.append(section)

    if pending:
        # 最后一节没有"下一节"可并 → 并进上一节（总比留一个只有标题的块强）
        if merged:
            merged[-1] = merged[-1] + pending
        else:
            merged.append(pending)

    return merged


def _is_heading_only(section: list[Document]) -> bool:
    """这一节是否**只有标题行**、没有真实正文。

    逐段去掉标题行，看剩下的有没有非空、非分隔线（`---` / `***`）内容。
    表格算正文（表格是真实内容，不是标题的附属）。
    """
    for doc in section:
        if doc.is_table:
            return False
        if doc.metadata.get("block") == "heading":
            continue
        lines = doc.text.strip().split("\n")
        body_lines = lines[1:] if _ATX_HEADING.match(lines[0]) else lines
        for line in body_lines:
            stripped = line.strip()
            if stripped and not _is_rule(stripped):
                return False
    return True


def _is_rule(line: str) -> bool:
    """Markdown 分隔线（`---` / `***` / `___`）。**它不算正文**。

    一份文档里 `---` 到处都是（章节分隔），把它当正文会让每个纯标题节
    都"看起来有内容"，孤儿标题就永远合并不了。
    """
    return len(line) >= 3 and len(set(line)) == 1 and line[0] in {"-", "*", "_"}


def _heading_level(doc: Document) -> int | None:
    """从 metadata 取标题级别。loader 统一填 `heading_level`（1-6）。"""
    raw = doc.metadata.get("heading_level")
    if raw is None:
        return None
    try:
        level = int(raw)
    except (TypeError, ValueError):
        return None
    return level if 1 <= level <= 6 else None


def _heading_title(doc: Document) -> str:
    """块的第一行若是标题，取出来。**保留标题行本身**（它是最浓缩的主题词）。"""
    first_line = doc.text.strip().split("\n", 1)[0].strip()
    if not first_line:
        return ""
    if first_line.startswith("#"):
        return first_line.lstrip("#").strip()
    return first_line if doc.metadata.get("block") == "heading" else ""


def _path_label(level: int | None, title: str) -> str:
    """二次切后续片补的路径标签。**简版：只有标题本身 + 层级**。

    **为什么不拼完整祖先链**：`Document` 是 loader 按标题切开的**平铺列表**，
    每段只记得自己的 `heading_level`，不记得父标题是谁 ——
    拼 `A > B > C` 必须维护跨块的标题栈。

    完整路径在这里补不了，所以**不伪造**：只给 `L2 > 标题`。
    写一个看起来像完整路径、其实只有一层的字段，比没有更糟 ——
    下游会拿它当层级依据。
    """
    return f"L{level} > {title}" if level else title


def _union_pages(section: list[Document]) -> tuple[int, ...]:
    """一节覆盖的页范围 = 各段 `pages` 的并集。跨页节同时覆盖 p6、p7。"""
    collected: set[int] = set()
    for doc in section:
        collected.update(inherited_pages(doc))
    return tuple(sorted(collected))


def _is_only_tables(section: list[Document]) -> bool:
    """这一节是否**全部是表格**。是的话整节一块（表格必须整张）。"""
    return all(doc.is_table for doc in section if doc.text.strip())


def _page_fallback(section: list[Document]) -> str:
    """这一节是"按标题分的"还是"降级成按页/按段分的"。

    `heading_fallback="page"` 表示**这一节的边界来自页码而不是标题** ——
    评测时必须与真正的标题切分分开统计，否则"标题策略在 PDF 上表现如何"
    这个结论测的是分页器。见模块 docstring。
    """
    if any(
        doc.metadata.get("block") == "heading"
        or _ATX_HEADING.match(doc.text.strip().split("\n", 1)[0])
        for doc in section
        if not doc.is_table
    ):
        return ""
    return "page" if any(doc.page is not None for doc in section) else "none"


__all__ = ["HeadingChunking"]
