"""PDF loader：逐页抽文本（pypdf）+ 表格转 Markdown（pdfplumber）。

## 页码为什么在这里就落到每一段上

**因为切分之后再也猜不出来。** 这是本格最重要的一条认知，写在最前面：

    错误做法：全篇抽成一坨文本 → 按字数切 → 事后把块"对回"页码
    结果    ：块 3 明明横跨 p6/p7，却只能标一个页码；或者按"平均每页多少字"折算

后者在两种常见文档上必然错位：
- **页密度不均匀**（第 1 页是封面只有 3 个字，后面的页 1200 字）
- **有页眉页脚被剔过**（每页字符数被人为削掉一截，折算基准整体漂了）

所以正确的顺序是**反过来**：解析阶段就用 pypdf 的逐页接口，
把 `page` 落在每个 `Document` 上；切分只做"继承"，不重新推断。
`Chunk.pages` 因此永远来自真实页码，而不是估算。

## 表格为什么单独抽、且不进文字流

pypdf 抽表格是**按阅读顺序铺平**的：一行单元格会变成好几行文本
（实测：`里程碑 / 时间 / 金额 / 一期交付 / 2026-11-30 …`），
行列关系彻底丢失。而"这张表第三列是什么"是合同问答里最常见的问题。

所以用 pdfplumber 单独抽表格，转成 Markdown 表格，作为**独立的 Document**
（`metadata["block"]="table"`）。同时必须把这份表格从文字流里**摘掉**，
否则同一份内容会进库两次：一次是铺平的文字垃圾，一次是结构化表格 ——
检索时两条都命中，上下文里出现两份互相矛盾的版本。

## 扫描件为什么必须打 warning

"抽不出文本"和"这一页是空白"在上层看起来完全一样，但处理方式完全不同：
前者要 OCR，后者什么都不用做。静默跳过 = 用户以为这份文档没内容。
判据取 `< 20 字符/页`（见 `SCAN_PAGE_CHARS`），理由在常量旁边。
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.rag import Document, LoadResult
from app.rag.loaders.clean import (
    can_join,
    merge_wrapped_lines,
    needs_space,
    normalize_text,
    strip_page_numbers,
    strip_repeated_lines,
)

logger = logging.getLogger(__name__)

# 一页少于这么多字符 → 疑似扫描件（无文本层）。
# 取 20 是因为：正常 A4 正文一页最少也有几百字；能抽出 20 字以下的，
# 通常只有"页码 + 页眉"被剔除后的残渣，或者一个孤立的图注。
# 宁可多报（多打一条 warning），不可漏报 —— 漏报的后果是用户以为文档空的。
SCAN_PAGE_CHARS: int = 20


def _make_reader(path: Path):  # noqa: ANN202
    """打开 PDF。坏文件是**真失败**，必须抛清晰错误。"""
    from pypdf import PdfReader

    try:
        return PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001 - 坏文件/加密/格式不支持
        raise ValueError(f"无法打开 PDF {path.name}：{type(exc).__name__}: {exc}") from exc


def extract_all_tables(path: Path, page_count: int) -> dict[int, list[list[list[str | None]]]]:
    """一次性抽出整份 PDF 的表格，返回 `{页码: [表格][行][单元格]}`。

    **必须只 open 一次**：`pdfplumber.open()` 会把整份文档的对象树读进来。
    按页 open 的话，50 页文档要 open 50 次 —— 实测那样做单页 3.3 ms、
    整份 50 页多花一倍时间，纯属自己给自己加负担。

    **失败必须降级成"没有表格"而不是抛**：pdfplumber 对复杂排版
    （旋转文本、矢量图、损坏的 xref）比 pypdf 敏感得多。它挂了不代表这份
    PDF 读不了，只代表这一页没有可用的表格。这与"坏文件要抛"不矛盾：
    打不开文件是**真失败**，抽不出表格是**这份文档没有表格**。
    """
    try:
        import pdfplumber
    except ImportError:  # pragma: no cover - 依赖缺失是部署问题，不是运行分支
        logger.warning("未安装 pdfplumber，跳过表格抽取：%s", path.name)
        return {}

    tables_by_page: dict[int, list[list[list[str | None]]]] = {}
    try:
        with pdfplumber.open(str(path)) as pdf:
            for index in range(min(page_count, len(pdf.pages))):
                page = pdf.pages[index]
                try:
                    # find_tables() 给出 Table 对象（带 bbox），extract() 取行列值。
                    # 用这条路径而不是 extract_tables()：两者耗时实测相同，
                    # 但 find_tables 额外给出 bbox —— 将来"按区域裁表格"要用到。
                    found = page.find_tables()
                    rows = [table.extract() for table in found]
                except Exception as exc:  # noqa: BLE001 - 单页表格失败不该毁整份
                    logger.warning(
                        "第 %d 页表格抽取失败（%s），按无表格处理：%s",
                        index + 1,
                        type(exc).__name__,
                        path.name,
                    )
                    continue
                if rows:
                    tables_by_page[index + 1] = rows
    except Exception as exc:  # noqa: BLE001 - 整份打不开也只是"没有表格"
        logger.warning("表格抽取整体失败（%s），按无表格处理：%s", type(exc).__name__, path.name)
        return {}
    return tables_by_page


def extract_tables(page_number: int, path: Path) -> list[list[list[str | None]]]:
    """抽单页表格（给调试/联调用）。批量场景请用 `extract_all_tables`。"""
    return extract_all_tables(path, page_number).get(page_number, [])


def table_to_markdown(rows: list[list[str | None]], *, header: bool = True) -> str:
    """把 pdfplumber 的行列数据转成 Markdown 表格。

    **为什么转 Markdown 而不是保留二维结构**：下游（切分、embedding、生成）
    全部按纯文本处理。Markdown 表格是纯文本里"结构信息保留得最完整"的形式，
    而且模型对它最熟 —— 它会知道 `| 里程碑 | 时间 | 金额 |` 后面跟的是数据行，
    而不是一段散文。

    **空单元格保留为空**而不是删掉列：删列会让"这一格是空的"和
    "这张表少一列"变得无法区分，而合并单元格产生的空格恰恰是有含义的。
    """
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    lines: list[str] = []
    for index, row in enumerate(rows):
        cells = [
            (c if c is not None else "").replace("\n", " ").replace("|", "／").strip() for c in row
        ]
        cells += [""] * (width - len(cells))
        lines.append("| " + " | ".join(cells) + " |")
        if index == 0 and header:
            # 分隔行：Markdown 表格的语法要求，也让人眼一眼看出这是表头
            lines.append("| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)


def _cell_values(rows: list[list[str | None]]) -> list[str]:
    """把表格里所有非空单元格按出现次数摊平。用于"从文字流里摘掉表格"。"""
    values: list[str] = []
    for row in rows:
        for cell in row:
            if cell and str(cell).strip():
                values.append(str(cell).strip())
    return values


def _remove_table_lines(lines: list[str], rows: list[list[str | None]]) -> tuple[list[str], int]:
    """从 pypdf 的文字行里摘掉表格区域。

    **为什么按"多重集相等"匹配，而不是删掉所有长度一致的包含行**：
    简单做法是"只要这一行等于某个单元格就删"，但那会误删正文 ——
    正文里出现 `金额` 这个词是完全正常的。

    这里要求：找到一段**连续**的行，它们（去掉空行后）的多重集合
    **恰好等于**该表格全部非空单元格的多重集合。表格在 pypdf 的输出里
    通常正好是连续的若干行，所以这个判据既准确又简单。
    找不到就一个都不删（返回 0），让"表格没被摘干净"表现为重复内容，
    而不是"正文被误删" —— 前者难看，后者是数据损坏。
    """
    from collections import Counter

    if not rows:
        return lines, 0
    wanted = Counter(_cell_values(rows))
    if not wanted:
        return lines, 0

    for start in range(len(lines)):
        acc: Counter[str] = Counter()
        index = start
        while index < len(lines):
            value = lines[index].strip()
            if not value:
                index += 1
                continue
            if wanted[value] == 0 or acc[value] >= wanted[value]:
                break
            acc[value] += 1
            index += 1
            if acc == wanted:
                removed = sum(acc.values())
                return lines[:start] + lines[index:], removed
    return lines, 0


def load_pdf(path: str | Path) -> LoadResult:
    """逐页加载 PDF。

    返回的 `Document` **一页一个**（表格另起一条），`page` 从 1 开始
    （与人类习惯一致，不是 0）。

    处理顺序是有讲究的，顺序换了结果就错：

        抽原始行 → 去页眉页脚（要跨页统计）→ 去页码行（按形态）
        → 摘表格（要按行匹配）→ 合并断行 → 跨页拼接 → 产 Document

    **去页眉必须在合并断行之前**：合并会把多行粘成一段，
    粘完之后"首行/末行"这个概念就没有了，重复行统计再也做不了。
    """
    file_path = Path(path)
    result = LoadResult()
    reader = _make_reader(file_path)

    # ---------------------------------------------------------------- ① 逐页抽原始行
    raw_pages: list[list[str]] = []
    failed_pages: list[int] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = normalize_text(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - 单页解析失败不该毁掉整份文档
            result.warnings.append(
                f"第 {index} 页解析失败（{type(exc).__name__}），已跳过：{file_path.name}"
            )
            failed_pages.append(index)
            raw_pages.append([])
            continue
        raw_pages.append(text.split("\n") if text else [])

    # ---------------------------------------------------------------- ② 表格（按页先抽好）
    # 表格抽取放在清洗**之前**，因为摘除表格要用原始行去匹配；
    # 清洗会挪动行，清洗完再匹配就找不到边界了。
    # 一次 open 抽完整份文档的表格（见 extract_all_tables 的 docstring）。
    page_tables = extract_all_tables(file_path, len(raw_pages))

    # ---------------------------------------------------------------- ③ 去页眉页脚
    cleaned_pages = strip_repeated_lines(raw_pages)

    # ---------------------------------------------------------------- ④ 逐页：去页码 → 摘表格 → 合并断行
    page_lines: dict[int, list[str]] = {}
    table_blocks: list[tuple[int, str]] = []
    for index, lines in enumerate(cleaned_pages, start=1):
        lines = strip_page_numbers([ln for ln in lines if ln.strip()])
        for rows in page_tables.get(index, []):
            lines, removed = _remove_table_lines(lines, rows)
            markdown = table_to_markdown(rows)
            if markdown:
                table_blocks.append((index, markdown))
            if removed == 0:
                # 没摘干净：说明表格文字和正文交错在 pypdf 的行流里。
                # 这时**保留文字流**（宁可重复也不要丢内容），并记 warning 让人知道。
                result.warnings.append(
                    f"第 {index} 页表格未能在文字流中完整摘除，可能出现重复内容：{file_path.name}"
                )
        page_lines[index] = merge_wrapped_lines(lines)
    # ---------------------------------------------------------------- ⑤ 跨页拼接
    # 断句跨页：上一页最后一段没说完，且下一页第一段是续行 → 拼接。
    # **这是"页码会不会错位"的第二个关键点**：拼接后的段落横跨两页，
    # 所以它的 `pages` 必须同时记两页，`page` 只记起始页。
    span_of: dict[int, tuple[int, ...]] = {index: (index,) for index in page_lines}
    sorted_indexes = sorted(page_lines)
    for position, index in enumerate(sorted_indexes[:-1]):
        following = sorted_indexes[position + 1]
        current = page_lines.get(index) or []
        nxt_lines = page_lines.get(following) or []

        # 取"当前页最后一段非空行"与"下一页第一条非空行"
        tail_pos = next((i for i in range(len(current) - 1, -1, -1) if current[i].strip()), None)
        head_pos = next((i for i in range(len(nxt_lines)) if nxt_lines[i].strip()), None)
        if tail_pos is None or head_pos is None:
            continue

        tail, head = current[tail_pos], nxt_lines[head_pos]
        if not can_join(tail, head):
            continue

        separator = " " if needs_space(tail, head) else ""
        current[tail_pos] = tail.rstrip() + separator + head.lstrip()
        # 下一页的首段已被拼走，从它自己的页里删掉（**只删这一行**，其余不动）
        page_lines[following] = nxt_lines[:head_pos] + nxt_lines[head_pos + 1 :]
        page_lines[index] = current
        # 拼接后的段落真实覆盖了两页 —— `pages` 记全，`page` 只记起始页
        span_of[index] = tuple(sorted(set(span_of[index]) | {following}))

    # ---------------------------------------------------------------- ⑥ 产出 Document
    # 顺序按页码排：某页的正文段 → 该页的表格 → 下一页。表格跟着它所在的页走，
    # 摘要/检索时"这段表格属于哪一节"不会前后颠倒。
    empty_pages: list[int] = []
    scan_pages: list[int] = []
    spaced_pages: dict[int, tuple[int, ...]] = span_of
    tables_by_page: dict[int, list[str]] = {}
    for table_page, markdown in table_blocks:
        tables_by_page.setdefault(table_page, []).append(markdown)

    # 有表格但完全没有正文的页（整页就是一张表）也要出现在输出里
    all_pages = sorted(set(sorted_indexes) | set(tables_by_page))

    for index in all_pages:
        lines = [ln for ln in (page_lines.get(index) or []) if ln.strip()]
        text = normalize_text("\n".join(lines))
        if text:
            if len(text) < SCAN_PAGE_CHARS:
                scan_pages.append(index)
            result.documents.append(
                Document(
                    source=file_path.name,
                    page=index,
                    text=text,
                    metadata={"block": "text", "page_chars": len(text)},
                    pages=spaced_pages.get(index, (index,)),
                )
            )
        elif index not in failed_pages and index not in tables_by_page:
            empty_pages.append(index)

        for markdown in tables_by_page.get(index, []):
            result.documents.append(
                Document(
                    source=file_path.name,
                    page=index,
                    text=markdown,
                    # `block="table"` 是切分器"表格必须单独成块"的依据
                    metadata={"block": "table", "page_chars": len(markdown)},
                    pages=(index,),
                )
            )

    # ---------------------------------------------------------------- ⑦ warning
    if empty_pages:
        preview = "、".join(str(p) for p in empty_pages[:10])
        more = f" 等 {len(empty_pages)} 页" if len(empty_pages) > 10 else ""
        result.warnings.append(
            f"有 {len(empty_pages)} 页没有可提取文本（扫描件或空白页）：第 {preview}{more} 页"
        )
    if scan_pages:
        # **单独一条**：这页有字但极少，说明文本层可能残缺（部分扫描）
        preview = "、".join(str(p) for p in scan_pages[:10])
        more = f" 等 {len(scan_pages)} 页" if len(scan_pages) > 10 else ""
        result.warnings.append(
            f"有 {len(scan_pages)} 页可提取字符少于 {SCAN_PAGE_CHARS}（疑似扫描页，可能需要 OCR）："
            f"第 {preview}{more} 页"
        )
    if not result.documents:
        result.warnings.append(
            f"{file_path.name} 整份文档都没有可提取文本 —— 很可能是纯扫描件，需要 OCR 才能处理"
        )

    return result


__all__ = [
    "SCAN_PAGE_CHARS",
    "extract_tables",
    "load_pdf",
    "normalize_text",
    "table_to_markdown",
]
