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

        # **代码块不在这里分流。**
        #
        # 早期版本把代码抽出来在尾部追加（照抄 `recursive` 的做法），
        # 结果代码块**丢了 `section_path`** —— 实测 `docs/w4-checklist.md`
        # 35 块里 34 块带路径，唯一缺的就是代码块。
        #
        # 而对代码块来说路径尤其要紧：`` flags = a | b `` 单看完全不知道自己
        # 属于"二、校验逻辑"这一节。递归策略丢这个信息无所谓（它本来就没有），
        # 但 heading 策略的全部价值就在这条路径上 —— 丢掉它等于把 heading
        # 降级成 recursive，还多付了结构解析的力气。
        #
        # 所以：代码块**留在所属节里**，由 `_emit` 按"三类内容"拆分产出。
        chunks: list[Chunk] = []
        for section, path in self.sections_with_path(docs, cfg):
            chunks.extend(self._emit(section, cfg, index_base=len(chunks), section_path=path))
        return chunks

    # ------------------------------------------------------------------ 分组

    def _sections(self, docs: list[Document], cfg: ChunkingConfig) -> list[list[Document]]:
        """把平铺的 `Document` 列表分组。**跨文档也合并**：一节常被 loader 拆成多段。"""
        return [section for section, _path in self.sections_with_path(docs, cfg)]

    def sections_with_path(
        self, docs: list[Document], cfg: ChunkingConfig
    ) -> list[tuple[list[Document], str]]:
        """分组，并同时算出每组的 **section_path**。

        ## 为什么分组和路径必须一次遍历算完

        路径是**标题栈**的产物：`## 1.2` 的完整路径是 `第1章 > 1.2`，
        而"哪一个是祖先"取决于遍历到那儿时的栈状态。
        如果分两趟（先分组、再单独扫一遍拼路径），第二趟就必须
        **重放每个 section 起始处的栈** —— 那是把同一次遍历拆成两半，
        任何一处边界判断漂移（比如空段跳过规则不同），路径就会与分组错位。
        实测这种错位的表现是"块 3 的 section_path 属于块 4 的节"，
        不报错、只在人工读样本时被发现。

        所以一次遍历：边分组边维护栈，每开一个新 section 就把当时的栈快照下来。

        ## 降级路径也要有 path

        无标题时按页分组（PDF）。这时 `section_path` 用 **`第 N 页`** 占位 ——
        不是伪造章节名，而是"这一节的可读定位就是页码"。留空串会让
        下游无法区分"没有路径"与"路径算失败"。
        """
        if not _has_headings(docs):
            return [(section, _page_path(section)) for section in _group_by_page(docs)]

        raw_sections: list[tuple[list[Document], str]] = []
        stack: list[tuple[int, str]] = []
        current: list[Document] = []
        current_path = ""

        def close_current() -> None:
            nonlocal current, current_path
            if current:
                raw_sections.append((current, current_path))
                current = []

        for doc in docs:
            if not doc.text.strip():
                continue
            # **表格不切断节**：它只是"节内一块特殊内容"，节还在继续。
            # 早期版本在这里 `current = []`，把 `## 优先级总览` 拆成
            # "标题行 / 表格 / 表格后的 `---`"，而那个 `---` 成了**无主孤儿块**。
            # 表格的"整张不切"由 `_emit` 保证，与"它属于哪一节"是两件事。
            if _is_section_start(doc, cfg):
                level = _heading_level(doc) or 1
                title = _heading_title(doc)
                if title:
                    # 弹栈：同级或更深的标题出栈，栈里只留祖先
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, title))
                if current:
                    close_current()
                current_path = _join_path(stack)
                current = [doc]
            else:
                current.append(doc)
        close_current()

        merged = _merge_orphan_headings([section for section, _path in raw_sections])
        # 孤儿合并会**减少组数**（纯标题节被并进下一节），但不会产生新组。
        # 所以按"合并后每组的第一段身份"去原始分组里查路径：
        # 合并只会把 B 段并到后面的 C 组，C 组的第一段身份不变 —— 查得到。
        path_by_first_doc: dict[int, str] = {id(section[0]): path for section, path in raw_sections}
        results: list[tuple[list[Document], str]] = []
        for section in merged:
            results.append((section, path_by_first_doc.get(id(section[0]), "")))
        return results

    # ------------------------------------------------------------------ 产出

    def _emit(
        self,
        section: list[Document],
        cfg: ChunkingConfig,
        *,
        index_base: int,
        section_path: str = "",
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
            # **`section_path` 是标题切分相对递归切分唯一不可替代的产物**：
            # 检索时可以按"章节"过滤（`where={"section_path": …}` 走 Chroma 的扁平元数据），
            # 生成时可以告诉用户"在第 1.2 节"而不是"在第 7 页"。
            # 其余差异（块长、块数）都能被参数弥补，只有这条不能 ——
            # 递归切分从结构上就拿不到层级信息。
            **({"section_path": section_path} if section_path else {}),
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

        # **表格/代码存在时必须走拆分路径**（即使整节不超预算）。
        #
        # 否则一个"标题 + 100 字正文 + 一张小表"的节会合并成一块，
        # 表格的 `is_table` 变成 `False` —— 那个错不报错、只在人工读样本时才被发现。
        # 早期版本只在"超预算"分支里做拆分，正是漏了这条。
        # **含表格/代码的节要特殊处理，不能整节二次切。**

        # 二次切用的是 `DelimiterSplitter`（中文句读），它会把表格行撕开、
        # 把 `def` 与它的 `return` 分到两块 —— 那正是本周要防的
        # "三类内容被切碎"。实测踩过：早期版本只在"超预算"分支里拆表格，
        # 于是"标题 + 100 字正文 + 一张小表"的节被整节合并，
        # 表格的 `is_table` 变成 `False`，而那个错不会报错。
        has_special = any(doc.is_table or doc.is_code for doc in section)
        if has_special or _is_only_tables(section):
            return self._emit_with_specials(
                section=section,
                cfg=cfg,
                extra=extra,
                section_pages=section_pages,
                index_base=index_base,
                section_path=section_path,
            )

        # 纯正文且不超预算：**整节一块，不做任何二次切**
        if len(joined) <= cfg.chunk_chars:
            return self._split_tables_out(
                section=section,
                text=joined,
                extra=extra,
                section_pages=section_pages,
                index_base=index_base,
                section_path=section_path,
            )

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
            # 后续片补标题路径：它们不再以标题开头，需要自报所属章节。
            # **用完整 `section_path` 而不是 `L2 > 标题`** —— 后者只有一层，
            # 下游拿它当层级依据会错。
            if offset > 0 and cfg.heading_keep_path and section_path:
                text = f"【{section_path}】\n{text}"
            out.append(
                make_chunk(
                    text=text,
                    doc=first,
                    index=index_base + offset,
                    strategy=self.name,
                    pages=section_pages,
                    extra_metadata={
                        **extra,
                        # 二次切的每一片都要带 `section_part`，且**路径在 extra 里已继承** ——
                        # 这是"二次切分的每块继承 section_path"这条验收的可验证形式。
                        "section_part": offset,
                    },
                )
            )
        return out

    def _emit_with_specials(
        self,
        *,
        section: list[Document],
        cfg: ChunkingConfig,
        extra: dict[str, object],
        section_pages: tuple[int, ...],
        index_base: int,
        section_path: str,
    ) -> list[Chunk]:
        """含表格/代码的节：**表格整张一块、代码按函数边界切、正文按预算二次切**。

        ## 为什么不能只做"整节不切"或"整节二次切"

        两个极端都是错的：

        | 做法 | 后果 |
        | --- | --- |
        | 整节合一块 | 表格/代码的 `block` 被首段的 `text` 覆盖 → 下游认不出它们 |
        | 整节二次切 | 中文句读把表格行撕开、把函数拆散 —— 正是本周要防的 |

        正确做法是**按角色分流**：表格与代码各自独立产出，
        夹在它们之间的正文段落按"是否超预算"决定整块还是二次切。

        ## 顺序保住了吗

        产出的顺序与原文顺序一致（`buffer` 在遇到特殊块时先落袋）。
        这一点重要：`section_part` 与引用编号都依赖顺序。
        """
        from app.rag.chunking.code import split_code

        out: list[Chunk] = []
        buffer: list[Document] = []

        def flush_text() -> None:
            nonlocal buffer
            if not buffer:
                return
            body = "\n".join(doc.text.strip() for doc in buffer).strip()
            buffer = []
            if not body:
                return
            if len(body) <= cfg.chunk_chars:
                out.append(
                    make_chunk(
                        text=body,
                        doc=section[0] if not out else _first_text_doc(section),
                        index=index_base + len(out),
                        strategy=self.name,
                        pages=section_pages,
                        extra_metadata=extra,
                    )
                )
                return
            # 超预算的正文段：用已验证的中文分隔符切（这是它擅长的场景）
            splitter = DelimiterSplitter(
                ChunkSize(chunk_chars=cfg.chunk_chars, overlap_chars=cfg.overlap_chars)
            )
            synthetic = Document(
                source=section[0].source,
                page=section[0].page,
                text=body,
                metadata=section[0].metadata,
                pages=section_pages,
            )
            for offset, piece in enumerate(splitter.split([synthetic])):
                out.append(
                    make_chunk(
                        text=piece.text,
                        doc=section[0],
                        index=index_base + len(out),
                        strategy=self.name,
                        pages=section_pages,
                        extra_metadata={**extra, "section_part": offset},
                    )
                )

        for doc in section:
            if doc.is_table:
                flush_text()
                out.append(
                    make_chunk(
                        text=doc.text.strip(),
                        doc=doc,
                        index=index_base + len(out),
                        strategy=self.name,
                        pages=section_pages,
                        # **表格自带 section_path**：独立成块但仍属于这一节
                        extra_metadata=extra,
                    )
                )
            elif doc.is_code:
                flush_text()
                for block in split_code(doc.text):
                    out.append(
                        make_chunk(
                            text=block.text,
                            doc=doc,
                            index=index_base + len(out),
                            strategy=self.name,
                            pages=section_pages,
                            extra_metadata={
                                **extra,
                                # **覆盖 block**：代码块必须保持 code，否则
                                # 下游会把它当正文再递归切一次
                                "block": "code",
                                "code_language": block.language,
                            },
                        )
                    )
            else:
                buffer.append(doc)
        flush_text()
        return out

    def _split_tables_out(
        self,
        *,
        section: list[Document],
        text: str,
        extra: dict[str, object],
        section_pages: tuple[int, ...],
        index_base: int,
        section_path: str,
    ) -> list[Chunk]:
        """整节一块时，**把表格抽成独立块**，其余正文按原顺序保留。

        ## 为什么必须在产出时拆（这是一个实测出来的错）

        早期版本让表格跟着它所在的节合并成一块，导致 `is_table` 变成 `False`：
        `make_chunk` 的 metadata 来自该节**首段**，首段是标题或正文，不是表格。

        后果是链条性的：
        - `is_table=False` → 下游不会标"（表格）"，引用只剩文件名页码
        - 表格行与正文混在一个块里，检索命中时模型看到的是"一段文字 + 一张表"
          挤在一起，而它无法判断哪部分是可引用的条文
        - **表格的 Markdown 结构还在**（`|` 分隔），所以这个问题不会报错，
          只在人工读样本时被发现 —— 正是判据二（拦腰切）会漏、判据三会抓到的那类

        ## 归属：表格仍在这一节的 `section_path` 里

        拆出去不等于脱钩：表格块的 `section_path` 沿用所在节的路径，
        **只是它自己是独立块**。这是"属于哪一节"与"是否独立成块"两件事
        的正确分离（与 `_sections` 里"表格不切断节"是同一条原则的两面）。

        ## 代码块走同一条路（但用 `split_code` 而不是原样一块）

        代码也需要独立成块（不参与正文切分），但它**可能很长**，
        所以不能像表格那样"整张一块"，而是按函数边界切（`split_code`）。
        切完的每一块都继承 `section_path` —— 这就是上面那个
        "代码块丢了路径"实测的修法。
        """
        special = [doc for doc in section if doc.is_table or doc.is_code]
        if not special:
            return [
                make_chunk(
                    text=text,
                    doc=section[0],
                    index=index_base,
                    strategy=self.name,
                    pages=section_pages,
                    extra_metadata=extra,
                )
            ]

        from app.rag.chunking.code import split_code

        out: list[Chunk] = []
        buffer: list[Document] = []

        def flush_text() -> None:
            nonlocal buffer
            if not buffer:
                return
            body = "\n".join(doc.text.strip() for doc in buffer).strip()
            if body:
                out.append(
                    make_chunk(
                        text=body,
                        # **用 buffer[0] 而不是 section[0]**：metadata 来自这一组的首段，
                        # 这样"标题 + 正文"那一组的 `block` 是 `heading`/`text`，
                        # 不会把表格/代码的 `block` 错挂到正文块上。
                        doc=buffer[0],
                        index=index_base + len(out),
                        strategy=self.name,
                        pages=section_pages,
                        extra_metadata=extra,
                    )
                )
            buffer = []

        for doc in section:
            if doc.is_table:
                flush_text()
                out.append(
                    make_chunk(
                        text=doc.text.strip(),
                        doc=doc,
                        index=index_base + len(out),
                        strategy=self.name,
                        pages=section_pages,
                        # **表格块自己也带 section_path**：它独立成块，
                        # 但仍属于这一节 —— 检索按章节过滤时要能查得到它。
                        extra_metadata=extra,
                    )
                )
            elif doc.is_code:
                flush_text()
                for block in split_code(doc.text):
                    out.append(
                        make_chunk(
                            text=block.text,
                            doc=doc,
                            index=index_base + len(out),
                            strategy=self.name,
                            pages=section_pages,
                            extra_metadata={
                                **extra,
                                "block": "code",
                                "code_language": block.language,
                            },
                        )
                    )
            else:
                buffer.append(doc)
        flush_text()
        return out


# --------------------------------------------------------------------------- 工具


def _first_text_doc(section: list[Document]) -> Document:
    """这一节里第一个非表格、非代码的文档。给 metadata 取 `block="text"` 用。"""
    for doc in section:
        if not doc.is_table and not doc.is_code:
            return doc
    return section[0]


def _join_path(stack: list[tuple[int, str]]) -> str:
    """把标题栈拼成 `第1章 > 1.2 > 公差` 形式的路径。

    **用 ` > ` 而不是 `/`**：章节标题本身常含 `/`（`HTTP / SSE`），
    用 `/` 分隔会让路径无法反解。` > ` 在中文标题里极少出现。

    栈为空（这一节前面没有标题）时返回空串 —— **不伪造一个"未命名"路径**，
    空串让下游能区分"没有层级"与"层级算失败"。
    """
    return " > ".join(title for _level, title in stack if title)


def _page_path(section: list[Document]) -> str:
    """降级（按页切）时的 `section_path`：`第 7 页`。

    不是伪造章节名，而是"这一节的可读定位就是页码"。留空串会让下游
    无法区分"没有路径"与"路径算失败" —— 而这两件事的处理完全不同
    （前者正常，后者要查 bug）。
    """
    pages = [doc.page for doc in section if doc.page is not None]
    if not pages:
        return ""
    first, last = min(pages), max(pages)
    return f"第 {first} 页" if first == last else f"第 {first}-{last} 页"


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
