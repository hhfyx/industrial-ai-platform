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

# Markdown 图片语法：`![alt](path)`
# 用于**整行只有图片**时丢弃该行（见 `_drop_image_only_lines`）
_IMAGE_ONLY = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$")

# Markdown 目录锚点链接：`[文字](#锚点)`
# 用于识别"目录块"（见 `_is_toc_block`）
_ANCHOR_LINK = re.compile(r"\[[^\]\n]+\]\(#[^)\n]+\)")

#: 目录块判据：锚点链接占非空行数的比例超过这个值，就认为是目录。
#:
#: ## 为什么用"锚点占比"而不是"有没有『目录』两个字"
#:
#: 目录的标题可能写成「目录」「Contents」「Table of Contents」，
#: 甚至**没有标题**（直接就是链接列表）。靠关键词匹配会漏。
#:
#: 而 `[数据集](#1-数据集)` 这种**带井号的锚点链接**是 Markdown 目录的
#: 结构性特征 —— 正文里几乎不会出现（正文的链接是 `[文字](url)`，
#: 指向外部地址，不是 `#` 锚点）。
#:
#: 取 **0.5** 的理由：实测那份报告里目录块是 8 条链接 / 9 行非空 = 0.89；
#: 而正文里偶尔出现 1 个锚点链接的段落（如"详见[第3节](#3-模型架构)"）
#: 通常 1/10 = 0.10，远低于阈值。
TOC_ANCHOR_RATIO = 0.5

#: 短于此长度的块会被合并进相邻块（单位：字符）。
#:
#: ## 这个数字是**实测标定的**，不是拍的
#:
#: 2026-11-04 用用户那份《工业缺陷生成模型全链路报告》实测：
#: 105 个块里有 **17 个短于 40 字符**，它们全都是这类东西：
#:
#:     ---                              (3 字符，水平分割线)
#:     **训练命令**：                    (9 字符，引导语，代码在下一块)
#:     **预期训练进度**：                 (11 字符)
#:     ### 3.3 LoRA 低秩适配              (28 字符，纯标题)
#:
#: 这些块**信息量接近零**，但在字面检索里是**顶配候选**：
#: `**训练命令**：` 这 4 个实词与"训练怎么跑"的问题字面完全重合，
#: 而一段 500 字的正文因为被更多无关字稀释，反而排到后面。
#:
#: 实测后果：`训练损失怎么变化` 的 Top-5 第 1、2 名是
#: `**训练命令**：`(9 字符) 和 `**预期训练进度**：`(11 字符) —— 真正的
#: 「5.3 训练损失曲线」排第 4。
#:
#: 取 **40** 的理由：能覆盖上面这些标签行（最长 28），
#: 又不会误并到正常段落（正文段的中位数在 150+ 字符）。
MIN_CHUNK_CHARS = 40


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
    """把一块内容里的**表格行**与**代码围栏**各摘出来单独成块。

    返回 `[(level, block_type, text), ...]`，`block_type` 是 `text` / `table` / `code`。

    **为什么不直接把整块都算表格**：Markdown 的一节通常"一段正文 + 一张表 + 一段正文"。
    把整节都标成表格，切分器就不会拆它 —— 但正文本来该按语义边界切。
    所以只把连续的 `|` 行摘出来。

    **围栏内的 `|` 不是表格**：代码块里出现 `if a | b:` 或 `|` 位运算非常常见。
    不识别围栏的话，一段代码会被标成"表格"然后禁止切分 ——
    结果是超大块进 embedding，或者表格语义被错误挂到代码上。
    这与 `split_by_headings` 里"围栏内的 `#` 不是标题"是同一条原则。

    ## 代码块为什么单独标成 `block="code"`

    代码与正文的切分边界完全不同（见 `chunking/code.py`）：
    正文按中文句读切，代码按 `def`/`class` 等结构边界切，且**必须保留缩进**。
    如果代码留在 `block="text"` 里，正文切分器会把
    `def handler(request)` 与 `return {"ok": True}` 分到两块 ——
    命中的块读不出任何逻辑。

    所以这里把**整个围栏（含首尾围栏行）**摘成一条 `block="code"` 的 Document。
    围栏行保留在文本里：下游重新渲染（`markdown` → 高亮）需要它，
    而且**语言标记就在围栏行上**。
    """
    pieces: list[tuple[int, str, str]] = []
    buffer: list[str] = []
    table: list[str] = []
    code: list[str] = []
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

    def flush_code() -> None:
        nonlocal code
        if code:
            pieces.append((level, "code", "\n".join(code)))
            code = []

    for line in block.split("\n"):
        if _FENCE.match(line):
            flush_table()
            if not in_fence:
                # 开围栏：先把之前的正文落袋，代码从这里开始收
                flush_text()
                in_fence = True
                code.append(line)
            else:
                # 闭围栏：收尾（**围栏行留在 code 里**，语言标记在它上面）
                code.append(line)
                in_fence = False
                flush_code()
            continue
        if in_fence:
            code.append(line)
            continue
        if _TABLE_ROW.match(line):
            flush_text()
            table.append(line)
        else:
            flush_table()
            buffer.append(line)
    # 未闭合的围栏：按代码块收（**不替源文档"修好"它**，只是不让它混进正文）
    flush_code()
    flush_table()
    flush_text()
    return pieces


def _is_toc_block(text: str) -> bool:
    """这一块是不是**目录**（Table of Contents）。

    ## 为什么目录块必须从检索里去掉

    目录是"章节标题的列表"，它**字面上覆盖了全文每一个主题词**：

        1. [数据集](#1-数据集)
        2. [环境配置](#2-环境配置)
        3. [模型架构](#3-模型架构)
        ...

    所以在字面检索里它**必然排第一** —— 不管你问什么，"数据集""模型架构"
    "生成结果"这些词全都在目录里出现过。

    实测（2026-11-04，那份全链路报告）：

        问题「用了什么基座模型」
          #1  目录块（0.259）           ← 全是章节名，没有答案
          #5  #### 3.2.3 Text Encoder（CLIP ViT-L/14）  ← 真正的答案

    这是"越泛的块排得越前"的又一个变体 —— 目录是**最泛**的一块。

    ## 但目录**不能直接丢**

    它承载了文档结构信息，对"文档分几章"这类问题有用。
    所以采取的策略是：**丢链接、留标题**：

        1. [数据集](#1-数据集)   ->   1. 数据集

    这样目录块变成一行行纯文字章节名，字面匹配能力大幅下降
    （不再有 `#` 锚点这种强特征），但仍保留了结构信息。

    ## 判据

    见 `TOC_ANCHOR_RATIO` —— 用"锚点链接占非空行的比例"，
    而不是"含不含『目录』二字"（那个词可能不存在或是别的语言）。
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return False
    anchors = len(_ANCHOR_LINK.findall(text))
    return anchors > 0 and anchors / len(lines) >= TOC_ANCHOR_RATIO


def _strip_toc_links(text: str) -> str:
    """把目录里的锚点链接降级成纯文字：`[数据集](#1-数据集)` -> `数据集`。

    ## 为什么是"降级"而不是"删除整块"

    删掉整块会让"这份文档讲了哪几章"这类问题失去依据。
    降级之后目录仍在，只是**不再带 `#` 锚点这个强字面特征**。

    ## 为什么连序号也保留

    `1. 数据集` 里的 `1.` 是结构信息（章节顺序），删了反而丢内容。
    只把 Markdown 的链接语法 `[文字](#锚点)` 展开成 `文字`。

    ## ⚠ 实测结论：光降级**不够**（2026-11-04）

    只做这一步之后，实测同一个问题「用了什么基座模型」：

        修复前：#1 目录块 score=0.259
        只降级：#1 目录块 score=0.255   ← 排名没动，分数几乎没变

    原因：`#` 锚点符号本身不参与分词（`normalize_text` 之后基本被吃掉），
    真正拉高分的**是那一串章节名本身** ——「模型架构」「生成结果」
    这些词跟问题里的「模型」高度重合。

    **所以目录块必须整体不参与检索**，而不是"降级后继续检索"。
    见 `_is_toc_block` 和 `load_markdown` 里的处理。
    """
    return _ANCHOR_LINK.sub(lambda m: m.group(0)[1:].split("]")[0], text)


def _drop_image_only_lines(text: str) -> str:
    """删掉**整行只有图片语法**的行（`![alt](path)`）。

    ## 为什么图片引用必须从检索里去掉

    图片对问答毫无价值 —— 模型没法看图，`report_figures/fig3_2_loss_curve.png`
    这个路径不包含任何可回答的语义。

    但它在**字面检索**里是有害的：`![图3-2 训练损失曲线](...)` 这一行含
    「训练损失曲线」五个实词，跟"损失怎么变化"这类问题高度重合，
    于是它会挤进 Top-5，占掉一个本该给正文的位置。

    实测（2026-11-04，用户那份全链路报告）：`### 5.3 训练损失曲线` 这个
    图片块排在 `缺陷有哪些` 的**第 5 名**（score 0.089）。

    ## 只删"整行只有图片"的行

    `见图 3-2：![曲线](x.png)` 这种**混在句子里**的不动 —— 图片是句子的
    一部分，删掉会破坏语义。判据是"这一行除了图片什么都没有"。
    """
    kept = [line for line in text.split("\n") if not _IMAGE_ONLY.match(line)]
    return "\n".join(kept)


def _is_too_short(block_type: str, text: str) -> bool:
    """这一块是否短到"不该独立成为检索候选"。

    ## 判据为什么不是"看它是不是标题"

    第一版写成"纯标题块合并"，实测**不够** —— 用户文档里还有大量
    `**训练命令**：` 这种"加粗引导语"块（9 字符），它们不是标题，
    但同样是 0 信息量的噪音，而且字面匹配度极高。

    所以判据改成**纯长度**：短于 `MIN_CHUNK_CHARS` 就并。
    这个判据不假设块的**语法形态**（标题/引导语/分割线/表格碎片），
    只看"它短得不足以承载一个完整意思"。

    ## 例外

    - **表格块**：表格天生短（一个只有两行的表可能 30 字符），
      但它语义完整，合并会破坏"表格不切碎"的约定。
      所以表格只看是否短到荒谬（< 10 字符）。
    - **代码块**：同理，一段 3 行的 shell 命令可能不到 40 字符，
      但它本身就是完整可执行单元。
    """
    stripped = text.strip()
    if not stripped:
        return True
    if block_type in ("table", "code"):
        return len(stripped) < 10  # 只拦"荒谬短"（如 `|---|` 残片）
    return len(stripped) < MIN_CHUNK_CHARS


def _merge_short_blocks(
    pieces: list[tuple[int, str, str]],
) -> list[tuple[int, str, str]]:
    """把**过短的块**合并到相邻块，避免它们成为独立的检索候选。

    ## 这是本文件最重要的一处修复（2026-11-04 线上实测驱动）

    用户那份 Markdown 报告（《工业缺陷生成模型全链路报告》，21.9 KB）
    被切成 113 块，其中 **17 块短于 40 字符**，全是这类：

        ---                      (3 字符，水平分割线)
        **训练命令**：            (9 字符，真正的命令在下一块)
        **预期训练进度**：         (11 字符)
        ## 附录：项目文件索引      (12 字符，纯标题)

    ## 为什么这些短块是灾难（不是"稍微有点噪音"）

    字面检索（hashing / BM25）算的是**字符重叠比例**。一个 9 字符的块
    `**训练命令**：` 里，4 个实词占了绝大比重；而一个 500 字符的正文块，
    同样的 4 个词被大量无关字稀释。

    于是**短块天然赢**。实测：

        问题「训练损失怎么变化」
          #1  **训练命令**：          (9 字符)
          #2  **预期训练进度**：       (11 字符)
          #4  ### 5.3 训练损失曲线     ← 真正的答案，被挤到第 4

    这不是"排序略有偏差"，是**系统性倒挂**：越没内容的块排得越前。

    ## 合并方向：向后优先，末尾则向前

    - 绝大多数情况短块是"引导语"，正文在**后面** -> 向后合并
    - 文档末尾的短块（如 `*文档生成日期：2026-06-15*`）没有后面 -> 向前合并
    - **合并后可能仍然短**（两个 20 字符的块并成 40）-> 再并一轮，直到稳定

    ## 为什么不用"直接丢掉短块"

    丢掉会损失真实内容 —— `**训练命令**：` 后面的那条命令是有效的，
    只是被 `_split_table_blocks` 摘成了独立的代码块。丢掉引导语会让
    那个代码块**没有说明文字**，模型看到一段裸命令不知道它是干什么的。
    """
    if not pieces:
        return []

    merged = list(pieces)

    # 反复合并直到稳定。**必须有轮数上限**：如果整篇都是短块
    # （极端情况：全是 `---`），无上限会死循环。
    for _ in range(10):
        changed = False
        out: list[tuple[int, str, str]] = []
        i = 0
        while i < len(merged):
            level, block_type, text = merged[i]
            if not _is_too_short(block_type, text):
                out.append((level, block_type, text))
                i += 1
                continue

            # ## 过短的块：并入最近的**普通文本块**，跳过表格/代码
            #
            # 这条是实测踩出来的两轮教训：
            #
            # 第 1 版：短块无条件吸收后一块 -> 把
            #   `### 1.3 缺陷类型分布`（14 字符）和后面那张表**拼在一起**，
            #   破坏了"表格独立成块"的老约定（下游靠 `block="table"` 决定
            #   "这张表不许切碎"），旧测试立刻红。
            #
            # 第 2 版：看到后一块是表格就**放弃合并** -> 短块全留下了，
            #   真实文档从 51 块回退到 80 块、短块从 0 涨到 29。
            #   "因为旁边是表格所以不并" 让大量短标题无法处理。
            #
            # 现在：**向后找最近的普通文本块**（跳过中间的表格/代码），
            # 把短块并到它前面。表格/代码块自身**不被吸收内容**，
            # 保持独立身份不变。
            target = None
            for j in range(i + 1, len(merged)):
                if merged[j][1] not in ("table", "code"):
                    target = j
                    break

            if target is not None:
                n_level, n_type, n_text = merged[target]
                text = f"{text}\n{n_text}"
                # **取更浅的层级**（数字更小 = 更高层）。
                #
                # 实测踩到：`# 付款计划`（level 1）是短标题，被并进
                # `## 一、里程碑`（level 2）后层级变成 2，于是
                # `test_markdown_heading_levels_go_into_metadata` 里
                # `assert 1 in levels` 失败。
                #
                # 语义上取更浅的才对：合并后的块代表"付款计划下的里程碑"，
                # 它的**主标题**是 level 1。取深的会让下游误以为整块都在
                # 二级标题之下，摘要时丢失文档的顶层结构。
                level = min(level, n_level) if level > 0 and n_level > 0 else level or n_level
                # 中间跳过的表格/代码块原样保留，位置不变
                out.extend(merged[i + 1 : target])
                out.append((level, n_type if n_type != "text" else block_type, text))
                i = target + 1
                changed = True
                continue

            # 后面没有普通文本块（只剩表格/代码，或已到末尾）-> 向前合并
            can_merge_back = out and out[-1][1] not in ("table", "code")
            if can_merge_back:
                p_level, p_type, p_text = out.pop()
                out.append((level, p_type, f"{p_text}\n{text}"))
                changed = True
                i += 1
                continue

            # 前后都不行（整篇只有它 / 邻居都是表格）：保留。
            # 这种块很少（`---` 夹在两张表之间），留着比乱并安全。
            out.append((level, block_type, text))
            i += 1

        merged = out
        if not changed:
            break

    return merged


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

    # 先把所有 piece 收集成 `(level, block_type, text)`，**再统一合并** ——
    # 合并需要看到完整序列（纯标题要跨到下一块去），不能边遍历边决定。
    pieces: list[tuple[int, str, str]] = []
    image_lines_dropped = 0
    toc_count = 0
    toc_titles: list[str] = []

    for level, block in blocks:
        for piece_level, block_type, piece in _split_table_blocks(block, level):
            # ① 先去掉"整行只有图片"的行（对字面检索是纯噪音，见函数 docstring）
            image_lines_dropped += len([ln for ln in piece.split("\n") if _IMAGE_ONLY.match(ln)])
            piece = _drop_image_only_lines(piece)

            text = normalize_text(piece)
            if not text:
                continue

            # ② 目录块：**直接跳过，不进索引**
            #
            # 必须在 normalize_text 之后判断（判据依赖"非空行数"）。
            #
            # ## 为什么不是"降级后继续索引"（实测证明不够）
            #
            # 第一版只把 `[数据集](#1-数据集)` 降级成 `数据集`，结果：
            #
            #     修复前：#1 目录块 score=0.259
            #     只降级：#1 目录块 score=0.255   <- 排名没动
            #
            # 因为拉高分的是**那一串章节名本身**，不是 `#` 符号。
            # 目录块必然是最"泛"的一块 —— 不看内容只看章节名，
            # 任何涉及主题词的问题都能在它里面找到匹配。
            #
            # ## 丢了什么（诚实记录）
            #
            # "这份文档讲了哪几章"这类问题失去依据。但代价可接受：
            # 那种问题在 RAG 场景里罕见，而目录污染是**每一个**问题都在受损。
            if _is_toc_block(text):
                toc_count += 1
                toc_titles.append(_strip_toc_links(text))
                continue

            pieces.append((piece_level, block_type, text))

    # ② 把过短的块合并到相邻块。
    #
    # 这是 2026-11-04 线上实测驱动的修复，详见 `_merge_short_blocks`：
    # 短块在字面检索里是"顶配关键词集合"，会把真正文挤出 Top-K，
    # 表现为"越没内容的块排得越前"。
    pieces = _merge_short_blocks(pieces)

    table_count = 0
    heading_count = 0
    for piece_level, block_type, text in pieces:
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
            body_lines = [ln for ln in text.split("\n") if ln.strip()][1:]
            if text.lstrip().startswith("#") and not body_lines:
                metadata["block"] = "heading"
                heading_count += 1

        result.documents.append(
            Document(source=file_path.name, page=None, text=text, metadata=metadata)
        )

    if not result.documents:
        result.warnings.append(f"{file_path.name} 没有可提取的内容")

    # **目录被跳过这件事要说出来**。
    #
    # 静默丢弃是危险的：用户看到"块数比预期少"会以为是 bug，
    # 或者反过来 —— 问"文档分几章"得不到答案，却不知道为什么。
    # 一条 warning 让这个行为可见，UI 会把它显示在"当前知识库"下面。
    if toc_count:
        result.warnings.append(
            f"{file_path.name}：跳过了 {toc_count} 个目录块（目录的字面匹配度过高，"
            "会挤掉正文；如需回答'文档分几章'这类问题，请把目录写成正文段落）"
        )

    logger.info(
        "Markdown 加载完成：%s（%d 块，其中 %d 标题 / %d 表格；"
        "跳过 %d 个目录块、丢弃 %d 行纯图片）",
        file_path.name,
        len(result.documents),
        heading_count,
        table_count,
        toc_count,
        image_lines_dropped,
    )
    return result


__all__ = ["load_markdown", "split_by_headings"]
