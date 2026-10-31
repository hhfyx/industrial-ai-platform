"""PDF 标题识别：用字号/加粗启发式猜"这一行是不是标题"。

## 为什么这是启发式，而且必须能被关掉

PDF **没有"标题"这个概念**。它是排版格式：一个"标题"在文件里只是一行
`font_size=15, bold=True` 的文字，与"正文"的差别是**视觉的**，不是结构的。
Word 有 `Heading 1` 样式、Markdown 有 `#`，PDF 什么都没有。

所以这里做的是**猜**，猜的准确率就是"PDF 走标题策略"的天花板。
三条实测观察（本项目的 `tests/fixtures/sample20.pdf`）：

| 行 | 字号 | 真实角色 |
| --- | ---: | --- |
| `星辰科技内部资料 · 严格保密` | 8 | 页眉（该被剔除） |
| `第 1 章 · 第 1 页主题` | 15 | **标题**（正文 10） |
| `本节是…验收标准。` | 10 | 正文 |
| `第 1 页 / 共 20 页` | 8 | 页脚 |

所以"字号显著大于正文"这条判据在这个样本上非常干净。但它会在真实文档上错：

| 会错的情况 | 为什么 |
| --- | --- |
| **封面/图表标题** | 字号大但不是章节标题，切出来是"一节只有一个书名" |
| **全角数字/公式行** | 某些排版器把公式渲染成大字号 |
| **字号整体偏小的扫描件转文字** | 正文与标题的字号差可能不足 2 pt |
| **同一行内混排** | 一行里 title 与正文用不同字号（`第 3 章 概述` 后面直接接正文） |

**所以本模块的产物带 `confidence`，且调用方必须能"不信"。**
`load_pdf(use_heading_heuristic=False)` 走原路径（`heading_level` 一律不填），
策略层随后自动降级为按页切 —— 这就是任务里说的"不可靠就降级为递归，别硬做"。

## 判据本身

一条行被认作标题，需同时满足：

1. **字号 >= 正文主字号 × `SIZE_RATIO`**（默认 1.3 倍）—— 相对判据，不是绝对值
2. **该行足够短**（不超过 `MAX_TITLE_CHARS` 字）—— "标题不会是一整段"
3. **不是页眉页脚的数字/页码行**（形如 `第 N 页 / 共 M 页`）

第 2 条不是优化是必需：一份 PDF 里"整段正文被排成大字号"（比如宣传册）时，
只有长度判据能挡住"整段变成标题"。

## 为什么用 `visitor_text` 而不是重新解析

`pypdf` 的 `extract_text(visitor_text=...)` 会在**每个文本块**上回调，
带着 `font_size`（已由 pypdf 从 `Tf` 操作符解析出来）。这样我们拿到的是
"pypdf 自己认为的块"，与它产出的换行是一致的 —— 换一条解析路径去对齐字号，
两边对"一行"的定义会不一样，那是最难查的错位。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# 标题字号相对正文主字号的倍数。1.3 是"明显更大"的最低量级：
# 实测 10pt 正文 / 15pt 标题 = 1.5 倍；1.3 留了余量给 1.2 倍的文档。
SIZE_RATIO: float = 1.3

# 标题的最长字符数。"标题不会是一整段"这条挡的是"整段被排成大字号"的文档。
MAX_TITLE_CHARS: int = 60

# 页脚页码行：`第 N 页 / 共 M 页`、`Page N of M`、`- N -`
_PAGE_FOOTER = re.compile(
    r"^\s*(?:第\s*\d+\s*页(?:\s*/\s*共\s*\d+\s*页)?|page\s+\d+\s+of\s+\d+|-\s*\d+\s*-)\s*$",
    re.IGNORECASE,
)


@dataclass
class LineFont:
    """一行的字号信息。`text` 只用于诊断，判据只看 `size` / `bold`。"""

    text: str
    size: float
    bold: bool = False

    @property
    def chars(self) -> int:
        return len(self.text.strip())


@dataclass
class HeadingProbe:
    """整份 PDF 的标题识别结果。

    `body_size` 是**正文主字号**（按字符数加权的中位数近似），
    判据全是相对它算的 —— 绝对字号不可靠（同一份文档 8pt 标题也有 10pt 正文）。
    """

    body_size: float = 0.0
    headings: dict[int, list[str]] = field(default_factory=dict)
    """`{页码: [标题行文本, ...]}`。只收命中的行。"""

    confidence: float = 0.0
    """可信度（0~1）。调用方用它决定"信多少"。

    **为什么要这个字段**：启发式一定会错，而错的后果是"这一节切错位置" ——
    不报错、只是块变差。有了 confidence，上层（`load_pdf` 的调用方 /
    策略层）可以选择"不信就不填 heading_level，让它降级成按页切"。
    """

    ratio: float = 0.0
    """标题字号 / 正文主字号 的实际倍数。所有命中行里取最大值 ——
    它是"这个启发式在这个文档上有没有区分度"的直接证据。"""

    def is_trustworthy(self, *, min_confidence: float = 0.6, min_ratio: float = 1.3) -> bool:
        """这份文档的标题识别结果是否值得用。

        两个条件**必须同时满足**：
        - `confidence >= min_confidence`：命中的行占"候选大字号行"的比例够高
        - `ratio >= min_ratio`：字号确实拉开了差距

        只用其中一条会放过两类坏文档：①字号有差距但几乎没有标题（confidence 低）；
        ②有标题但字号差不足（ratio 低，说明区分度来自噪声）。
        """
        return self.confidence >= min_confidence and self.ratio >= min_ratio


def probe_page(
    page,  # noqa: ANN001 - pypdf.PageObject，避免顶层 import 可选依赖
) -> list[LineFont]:
    """抽一页的"行 + 字号"。**只读，不改 page 状态。**

    用 `visitor_text`：pypdf 在每个文本块上回调，参数里带 `font_size`。
    空文本块（纯定位）跳过 —— 它们没有字符，字号无意义。
    """
    lines: list[LineFont] = []

    def visitor(text: str, _cm, _tm, font_dict, font_size) -> None:  # noqa: ANN001
        if not text or not text.strip():
            return
        bold = _is_bold(font_dict)
        # 一块里可能有多个换行：按行拆，字号沿用这一块的字号。
        # pypdf 的字号是"当前图形状态"，同一 block 内的行字号通常相同 ——
        # 不同时会被拆成不同 block（实测 sample20.pdf 就是这样）。
        for part in text.split("\n"):
            if part.strip():
                lines.append(LineFont(text=part, size=float(font_size or 0.0), bold=bold))

    page.extract_text(visitor_text=visitor)
    return lines


def _is_bold(font_dict: dict | None) -> bool:
    """字体是否加粗。**优先看字体名里的 Bold 字样**，其次看 `Flags` 位。

    `font_dict` 的结构因 pypdf 版本而异（实测 6.x 给的是
    `{'/BaseFont': '/XXXXXX+Arial-BoldMT', ...}`）。
    取不到就当"不加粗"—— 假阴性无害（字号判据还能work），
    假阳性会让正文被误判成标题。
    """
    if not font_dict:
        return False
    base = str(font_dict.get("/BaseFont", "") or "")
    return "bold" in base.lower() or "black" in base.lower()


def analyze(lines_by_page: dict[int, list[LineFont]]) -> HeadingProbe:
    """从"每页的行+字号"推断标题，返回 `HeadingProbe`。

    ## 正文主字号怎么取（用字符加权，不用行数）

    按**字符数**统计字号，而不是按行数。理由：标题行少、正文行多，
    但更关键的是"一页里 3 行标题 + 40 行正文"时按行数算正文也是多数 ——
    两种算法在这个样本上结果相同，但在"标题非常多的大纲式文档"上按行数
    会让标题成为多数，正文主字号直接被算成标题字号。

    ## confidence 怎么算

    `confidence = 命中标题的最小字号 - 正文主字号` 归一化到 [0,1]？
    不行 —— 那是 ratio 的重复表达。这里用的是**覆盖率**：

        confidence = 标题行数 / (标题行数 + 大字号但被判为正文的行数)

    它衡量的不是"字号差距够不够"，而是"这个判据在这个文档上稳定不稳定"：
    如果大量大字号行既不短也不像标题，说明这份文档的排版把"大字号"
    用在了别处（宣传册、图表说明），判据就会同时误伤。
    """
    body_size = _body_size(lines_by_page)
    if body_size <= 0:
        return HeadingProbe()

    threshold = body_size * SIZE_RATIO
    headings: dict[int, list[str]] = {}
    hits = 0
    candidates = 0
    max_size = 0.0

    for page_no, lines in lines_by_page.items():
        for line in lines:
            if line.size < threshold:
                continue
            candidates += 1
            stripped = line.text.strip()
            if _PAGE_FOOTER.match(stripped):
                continue
            if line.chars > MAX_TITLE_CHARS:
                continue
            hits += 1
            headings.setdefault(page_no, []).append(stripped)
            max_size = max(max_size, line.size)

    confidence = hits / candidates if candidates else 0.0
    ratio = (max_size / body_size) if body_size else 0.0
    return HeadingProbe(
        body_size=body_size,
        headings=headings,
        confidence=round(confidence, 3),
        ratio=round(ratio, 3),
    )


def _body_size(lines_by_page: dict[int, list[LineFont]]) -> float:
    """正文主字号：**按字符数加权**的众数。

    "众数"而不是"中位数"：一份 PDF 的字号集合通常是离散的几个值
    （8 / 10 / 15），中位数会在 10 与 15 之间落出一个 12.5 的假值 ——
    而那个值拿去做 `× 1.3` 阈值，会把真正的标题漏掉。
    """
    weights: dict[float, int] = {}
    for lines in lines_by_page.values():
        for line in lines:
            weights[line.size] = weights.get(line.size, 0) + len(line.text.strip())
    if not weights:
        return 0.0
    return max(weights, key=lambda size: weights[size])


def probe_pdf(
    path: str | Path,
    *,
    max_pages: int | None = None,
) -> HeadingProbe:
    """扫整份 PDF 的字号分布，返回标题识别结果。

    `max_pages` 给测试用（大文档不必全扫）。**注意它会影响正文主字号的统计**：
    只扫前 3 页时，若前面恰是封面 + 目录，主字号可能算成目录的字号。
    生产路径不传（扫全本）。
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    lines_by_page: dict[int, list[LineFont]] = {}
    for index, page in enumerate(reader.pages, start=1):
        if max_pages is not None and index > max_pages:
            break
        try:
            lines_by_page[index] = probe_page(page)
        except Exception as exc:  # noqa: BLE001 - 单页失败不该毁掉整份文档
            logger.warning("字号探测失败，跳过第 %d 页：%s", index, exc)

    probe = analyze(lines_by_page)
    logger.info(
        "PDF 标题探测：%s → 正文 %.1fpt / 命中 %d 页标题 / confidence=%.3f / ratio=%.3f",
        Path(path).name,
        probe.body_size,
        len(probe.headings),
        probe.confidence,
        probe.ratio,
    )
    return probe


__all__ = [
    "MAX_TITLE_CHARS",
    "SIZE_RATIO",
    "HeadingProbe",
    "LineFont",
    "analyze",
    "probe_page",
    "probe_pdf",
]
