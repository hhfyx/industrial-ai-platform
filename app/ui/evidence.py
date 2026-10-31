"""引用定位：把"答案里那句话"在 chunk 原文里找出来并高亮。

## 为什么需要这一层

UI 只做"整段展开 + 页码标注"也能交差，但那样用户还得**自己在那段文字里找**。
一段 500 字的条款里找"按日万分之五"这五个字，人眼要扫三秒 ——
而这三秒恰恰是"引用可不可信"的关键体验：找不到，用户就不信。

所以这里做一件小而具体的事：**用答案里的文字去反查原文，把命中的那句高亮**。

## 为什么不用"PDF 内跳页"

任务明确说"简单版：整段展开 + 页码标注即可，不做 PDF 内跳页"。
而不做的理由不只是工作量大 —— 真实原因是**我们没有 PDF 的坐标信息**：
`loaders/pdf.py` 只存了文本，没存每个字在页面上的 xy。
要做精确跳页得重新解析并保存坐标，那是另一格的事。

## 匹配策略：为什么不用 `in` 直接查

朴素做法是拿整句答案去 `chunk.text` 里 `in` 一下。**几乎必然失败**：
- 模型会把"第十五工作日"改写成"15 个工作日"
- 会省略主语、会调整语序
- 会加"根据"之类的连接词

所以这里用**最长公共子串（LCS 的连续版本）**：
找出答案句与原文之间最长的连续共同片段。够长（≥ 阈值）就认为"这句有原文依据"，
把它在原文里的位置标出来。

**这个算法还有一个副作用很有价值**：它同时给出了"这句答案有多少字与原文逐字相同"——
那是一个**弱化的引用校验**。如果模型说的每句在原文里都找不到 8 个连续字，
那它基本是在自由发挥。UI 可以据此提示"这条引用是弱关联"。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 认定"这句话有原文依据"所需的最短连续匹配长度（字符）。
#
# **8 是怎么来的**：中文里 8 个连续字几乎不可能偶然相同
# （"按日万分之五支付违约金" 是 10 字，""违约金按日万分之五" 是 9 字）。
# 取更小的值（如 4）会让"合同"、"甲方"这种通用词造成大量假命中；
# 取更大（如 15）则模型稍作改写就匹配不上。
# **这是一个经验阈值，不是调优结果** —— 文档里不许说它"最佳"。
MIN_MATCH_CHARS = 8

# 句子切分：中文句末标点 + 换行。**不用 `split("。")` 然后丢掉标点** ——
# 高亮时要把标点也包含进去，否则"按日万分之五支付违约金"后面少了句号，
# 看起来像被截断。
_SENTENCE = re.compile(r"[^。！？；\n]+[。！？；]?")


@dataclass(frozen=True, slots=True)
class Span:
    """原文里的一段高亮区间：`[start, end)`。"""

    start: int
    end: int
    matched: str
    """命中的那段文字（原文里的，不是答案里的）。"""

    @property
    def length(self) -> int:
        return self.end - self.start


def longest_common_substring(a: str, b: str) -> tuple[int, int]:
    """最长**连续**公共子串。返回 `(在 b 中的起始位置, 长度)`。

    用动态规划 O(n·m)。**不追求更快**：一段答案 ~200 字、一块原文 ~500 字，
    10 万次整数比较是微秒级；换后缀自动机之类只会让代码难懂、
    且这类算法最容易写错而**错得不明显**（结果偏一两个字符，高亮标错位置）。

    ## 第一版为什么写错了

    初版为了省内存把 `a`/`b` 按长度交换，然后用一个 `swapped` 标志把位置"翻译回去"。
    那是个不该有的复杂度：位置语义在函数内部被交换过，调用方只能靠
    "记得检查标志位"。我用一个三元组返回 `(start, length, swapped)`，
    还写了一段注释解释"只在没交换时用位置" —— **那是设计缺陷的自白**。

    正确做法：**保持参数顺序不变**，只在内部把较短的那个用作 DP 的列。
    位置始终相对于 `b`，没有歧义。
    """
    if not a or not b:
        return (0, 0)

    previous = [0] * (len(b) + 1)
    best_length = 0
    best_end_in_b = 0
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        char_a = a[i - 1]
        for j in range(1, len(b) + 1):
            if char_a == b[j - 1]:
                current[j] = previous[j - 1] + 1
                if current[j] > best_length:
                    best_length = current[j]
                    best_end_in_b = j
        previous = current

    return (best_end_in_b - best_length, best_length)


def _split_sentences(text: str) -> list[str]:
    """切成句子，保留标点，丢掉空白句。"""
    return [s.strip() for s in _SENTENCE.findall(text) if s.strip()]


def find_evidence_spans(
    answer_text: str, chunk_text: str, *, min_chars: int = MIN_MATCH_CHARS
) -> list[Span]:
    """在 `chunk_text` 里找出所有"能支撑 `answer_text`"的片段。

    ## 算法

    1. 把答案切成句子
    2. 每句去掉引用编号 `[n]`（那是我们的标记，原文里没有）
    3. 对每句求它与原文的最长连续公共子串
    4. 够长（≥ `min_chars`）就记一个 `Span`
    5. **合并重叠区间** —— 两句可能落在同一段原文里，合成一个高亮区
       比画两个相邻色块更清楚

    ## 返回空列表意味着什么

    意味着**这段引用是弱关联**：模型说的那句在原文里找不到连续 8 个字。
    不一定是编造（可能是改写幅度大），但用户应该知道。
    UI 会据此显示"⚠️ 未能在原文中定位依据"。
    """
    if not answer_text or not chunk_text:
        return []

    spans: list[Span] = []
    for sentence in _split_sentences(answer_text):
        # 去掉 `[1]` / `[1,2]` / `[1-3]` 这类引用标记
        cleaned = re.sub(r"\[\d+(?:\s*[-,，]\s*\d+)*\]", "", sentence).strip()
        if len(cleaned) < min_chars:
            # 太短的句子（"是。"、"共 3 条。"）匹配上没有意义：
            # 它们与原文的任意几个字都可能相同，会造成大量假高亮。
            continue

        start, length = longest_common_substring(cleaned, chunk_text)
        if length < min_chars:
            continue
        spans.append(
            Span(start=start, end=start + length, matched=chunk_text[start : start + length])
        )

    if not spans:
        return []
    return _merge_spans(spans)


def _merge_spans(spans: list[Span]) -> list[Span]:
    """合并重叠或相邻的区间（按 start 排序）。"""
    ordered = sorted(spans, key=lambda span: span.start)
    merged: list[Span] = [ordered[0]]
    for span in ordered[1:]:
        last = merged[-1]
        if span.start <= last.end:
            merged[-1] = Span(
                start=last.start,
                end=max(last.end, span.end),
                matched=last.matched if last.end >= span.end else span.matched,
            )
        else:
            merged.append(span)
    return merged


def highlight(
    chunk_text: str, spans: list[Span], *, marker_start: str = "**", marker_end: str = "**"
) -> str:
    """把原文按 `spans` 插上高亮标记，返回可直接渲染的字符串。

    **为什么用 `**…**` 而不是 HTML `<mark>`**：Streamlit 的 `st.markdown`
    默认**不渲染原始 HTML**（除非 `unsafe_allow_html=True`，而那会带来
    XSS 面 —— 文档内容来自用户上传，不该信任）。用 Markdown 粗体是
    零风险的等价效果，而且在纯文本环境下也能看懂。

    **区间必须已排序且不重叠**（`find_evidence_spans` 的输出保证这点）。
    这里再排一次是防御性的：手工传入乱序区间时不该渲染出乱码。
    """
    if not spans:
        return chunk_text

    ordered = sorted(spans, key=lambda span: span.start)
    parts: list[str] = []
    cursor = 0
    for span in ordered:
        if span.start < cursor:
            # 与上一个区间重叠 —— 跳过，别重复插入标记
            continue
        parts.append(chunk_text[cursor : span.start])
        parts.append(f"{marker_start}{chunk_text[span.start : span.end]}{marker_end}")
        cursor = span.end
    parts.append(chunk_text[cursor:])
    return "".join(parts)


__all__ = [
    "MIN_MATCH_CHARS",
    "Span",
    "find_evidence_spans",
    "highlight",
    "longest_common_substring",
]
