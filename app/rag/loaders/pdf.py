"""PDF loader：pypdf 逐页抽文本，保留页码。

**这一格的难点全在"不完美输入"上**，而不是"怎么调 pypdf"：

| 情况 | 天真写法会怎样 | 这里怎么处理 |
| --- | --- | --- |
| 扫描件（整页是图片） | 抽出空字符串，下游以为那一页没内容 | 记 warning，**跳过空页**但保留页码信息 |
| 空页（真的空白页） | 同上 | 同上 —— 无法区分"空白"和"扫描件"，统一记为"无文本页" |
| 跨页断句 | 一句话被切成两半，分属两页 | **在两页之间补一个连接符**，让下游切分时能粘回来 |
| 页眉页脚重复 | 每页都重复同样几行，浪费 token | 检测重复次数过多的首尾行并剔除 |

**为什么跨页断句要在这里处理**：
`pypdf` 是按页返回的，一页的末尾可能正好断在词中间（PDF 里没有"句"的概念，
只有"排版位置"）。如果下游按页独立处理，那句话就永远粘不回去了。
所以 loader 输出时要带一个信号：**这页结尾看起来是"没说完"的**。

这里用最简单的启发式：页末没有句末标点，且以字母/汉字结尾 → 认为是被截断的。
不够完美，但比"什么都不做"好得多，而且**代价是零**（不算模型调用）。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from app.rag import Document, LoadResult

logger = logging.getLogger(__name__)

# 页末出现这些标点，视为"这句话说完了"
_SENTENCE_END = re.compile(r"[。！？.!?:：；;】）)\"'」』]\s*$")

# 页末是这些字符，视为"话没说完"（字母/汉字/逗号/顿号）
_UNFINISHED_END = re.compile(r"[A-Za-z0-9\u4e00-\u9fff,，、]\s*$")

# 连续空白压缩。PDF 抽出来的文本常有多余空白与硬换行。
_MULTI_WS = re.compile(r"[ \t\u3000]+")
# 连续的 3 个以上换行压成 2 个（保留段落感）
_MULTI_NL = re.compile(r"\n{3,}")


def normalize_text(raw: str) -> str:
    """最小清洗：压缩空白、统一换行、去首尾空白。

    **刻意不做的事**：不删页眉页脚（那是 `_strip_repeated_lines` 的活）、
    不修断词、不做任何需要上下文的重排 —— loader 的职责是"忠实地取出文字"，
    加工属于下游。这条边界一旦模糊，出问题时无法判断是谁改坏的。
    """
    if not raw:
        return ""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ")  # 全角空格
    text = _MULTI_WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    # 逐行去首尾空白，但保留空行（段落边界）
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _looks_unfinished(text: str) -> bool:
    """页末是否"话没说完"。

    两步判断：
    1. 有句末标点 → 说完了 → False
    2. 以字母/数字/汉字/逗号结尾 → 很可能是被分页切断的 → True

    这个启发式会在"页末正好是个短句但没加标点"时误判，
    但成本极低（只是多补一个空格），误判后果远小于"句子被永久切断"。
    """
    tail = text.rstrip()
    if not tail:
        return False
    if _SENTENCE_END.search(tail):
        return False
    return bool(_UNFINISHED_END.search(tail))


def _strip_repeated_lines(
    pages: list[list[str]], *, min_repeat_ratio: float = 0.6
) -> list[list[str]]:
    """去掉**每页都出现**的首尾行（页眉 / 页脚 / 水印）。

    算法：统计所有页的**首行与末行**里，哪些行在超过 `min_repeat_ratio` 的页上出现。
    出现得够多就认定是页眉页脚，逐页从两端剥掉。

    **为什么要做这件事**：一份 50 页的报告，页眉是"XX 公司机密 第 N 页"，
    每页重复 —— 那是 50 份无意义的 token，而且会干扰摘要（模型可能反复提"公司机密"）。

    **只取首尾各 1 行，是踩过坑的**：
    最初写成"首 2 行 + 末 2 行"，测试用的 4 行页面直接覆盖了**全部**行 ——
    正文三行全被当成候选并删掉，每页只剩标题。那种 bug 很隐蔽：
    结果看起来"有内容"（标题还在），只是内容少了一大截。
    真正的页眉页脚就在首/末行，取 1 行足够，且不会吃掉正文。

    **必须排除"含页码的行"**：`第 1 页` / `第 2 页` 这种行有**每页都不同**的特征，
    但按"计数"统计会漏掉它们（每行只出现 1 次）。所以另外用一个正则识别
    "页码行"，单独剔除 —— 这是两种不同的模式，不能混在一个计数器里。
    """
    if len(pages) < 3:
        # 页数太少，统计没有意义（2 页重复恰好 100%，会误删正文）
        return pages

    threshold = max(2, int(len(pages) * min_repeat_ratio))

    # ① 逐页收集"首行"和"末行"
    first_lines: dict[str, int] = {}
    last_lines: dict[str, int] = {}
    for lines in pages:
        non_empty = [ln.strip() for ln in lines if ln.strip()]
        if not non_empty:
            continue
        for bucket, line in ((first_lines, non_empty[0]), (last_lines, non_empty[-1])):
            if len(line) < 2:
                continue
            bucket[line] = bucket.get(line, 0) + 1

    # 只有**同时**反复出现在首行或末行的，才算页眉页脚
    drop: set[str] = set()
    for bucket in (first_lines, last_lines):
        for line, count in bucket.items():
            if count >= threshold:
                drop.add(line)

    cleaned: list[list[str]] = []
    for lines in pages:
        kept = list(lines)
        # 从两端剥掉命中 drop 的行（只在两端剥，正文中间的重复不动）
        while kept and kept[0].strip() and kept[0].strip() in drop:
            kept.pop(0)
        while kept and kept[-1].strip() and kept[-1].strip() in drop:
            kept.pop()
        cleaned.append(kept)

    if drop:
        logger.info("剔除疑似页眉页脚 %d 行：%s", len(drop), list(drop)[:3])
    return cleaned


# 页码行：`第 3 页` / `- 3 -` / `Page 3 of 50` / 纯数字
_PAGE_NUMBER = re.compile(
    r"^\s*(?:第\s*\d+\s*页(?:\s*[/／]?\s*共\s*\d+\s*页)?|[-–—]\s*\d+\s*[-–—]|page\s+\d+(?:\s+of\s+\d+)?|\d{1,4})\s*$",
    re.IGNORECASE,
)


def _strip_page_numbers(lines: list[str]) -> list[str]:
    """剔除页码行。

    **为什么要单独处理**：页码每页都不同（"第 1 页"、"第 2 页"…），
    按"重复计数"永远识别不出来，但它的 token 成本和干扰效果与页眉一样。
    所以要用**模式匹配**而不是频率统计。

    只剥首尾各 1 行的位置：正文里出现一个孤立的数字
    （比如金额"128000"）是正常内容，不能动。
    """
    if not lines:
        return lines
    kept = list(lines)
    if kept and _PAGE_NUMBER.match(kept[0]):
        kept.pop(0)
    if kept and _PAGE_NUMBER.match(kept[-1]):
        kept.pop()
    return kept


def load_pdf(path: str | Path) -> LoadResult:
    """逐页加载 PDF。

    返回的 `Document` **一页一个**，`page` 从 1 开始（与人类习惯一致，不是 0）。

    读不到的页：
    - 整页空 → 跳过并记 warning（通常是扫描件或纯图片页）
    - 所有页都空 → 仍然返回空 documents + 明确 warning，**不抛异常**
      （调用方自己决定这算不算错误）
    """
    from pypdf import PdfReader

    file_path = Path(path)
    result = LoadResult()

    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:  # noqa: BLE001 - 坏文件/加密/格式不支持
        # **这里必须抛**：连文件都打不开是真正的失败，不是"某页没文本"。
        # 让调用方看到明确错误，而不是拿到一个空结果。
        raise ValueError(f"无法打开 PDF {file_path.name}：{type(exc).__name__}: {exc}") from exc

    raw_pages: list[list[str]] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = normalize_text(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - 单页解析失败不该毁掉整份文档
            result.warnings.append(
                f"第 {index} 页解析失败（{type(exc).__name__}），已跳过：{file_path.name}"
            )
            raw_pages.append([])
            continue
        raw_pages.append(text.split("\n") if text else [])

    # ① 去页眉页脚（按"重复频率"识别）
    cleaned_pages = _strip_repeated_lines(raw_pages)

    empty_pages: list[int] = []
    for index, lines in enumerate(cleaned_pages, start=1):
        # ② 去页码行（按"模式"识别 —— 页码每页都不同，频率统计抓不到）
        lines = _strip_page_numbers(lines)

        text = normalize_text("\n".join(lines))
        if not text:
            empty_pages.append(index)
            continue

        # 跨页断句：在页末补一个连接符，表示"这句话在两页之间被切断了"。
        # 下游切分时可以据此知道"这里不该断"。
        if _looks_unfinished(text):
            text = text + " "

        result.documents.append(Document(source=file_path.name, page=index, text=text))

    if empty_pages:
        # 一条汇总 warning 比每页一条更有用（50 页扫描件不该刷 50 行日志）
        preview = "、".join(str(p) for p in empty_pages[:10])
        more = f" 等 {len(empty_pages)} 页" if len(empty_pages) > 10 else ""
        result.warnings.append(
            f"有 {len(empty_pages)} 页没有可提取文本（扫描件或空白页）：第 {preview}{more} 页"
        )

    if not result.documents:
        result.warnings.append(
            f"{file_path.name} 整份文档都没有可提取文本 —— 很可能是纯扫描件，需要 OCR 才能处理"
        )

    return result


__all__ = ["load_pdf", "normalize_text"]
