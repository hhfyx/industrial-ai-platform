"""引用定位测试：把"答案里那句话"在原文里找出来。

## 为什么这件事需要测试

高亮的**位置**一旦算错，渲染出来仍然"像个正常的高亮块" ——
只是标在了错误的句子上。用户看到黄色标记，会以为那句就是依据。

**这是本项目里最"错得不明显"的一类 bug**：不报错、不空白、看起来还挺专业。
所以这里对位置做逐字符断言，而不是"有高亮就算过"。

## 三个必须区分的情况

| 情况 | 期望 |
| --- | --- |
| 答案与原文有长公共片段 | 返回 Span，位置精确 |
| 答案被大幅改写（无 8 字连续相同） | **返回空** —— 这是弱关联，要能让 UI 提示 |
| 答案很短（"是。"） | **返回空** —— 否则会和原文任意几个字假命中 |
"""

from __future__ import annotations

import pytest

from app.ui.evidence import (
    MIN_MATCH_CHARS,
    Span,
    find_evidence_spans,
    highlight,
    longest_common_substring,
)

# 一段真实形态的合同条款（500 字以内的常见长度）
CHUNK = (
    "第五条 付款计划：甲方按下列里程碑分期支付合同价款，验收合格后十五个工作日内付款。"
    "第六条 违约责任：乙方逾期交付的，按日万分之五支付违约金，累计不超过合同总额的百分之十。"
)


# ============================================================ 1. 最长公共子串


@pytest.mark.parametrize(
    ("a", "b", "expected_text"),
    [
        ("按日万分之五支付违约金", CHUNK, "按日万分之五支付违约金"),
        # **别用中文构造"完全无公共字符"的输入** —— 我两次都失败了：
        #   "完全不存在的内容" 里的"不" → 命中 CHUNK 的"不超过"
        #   "天干地支子丑寅卯" 里的"支" → 命中 CHUNK 的"支付"
        # 中文常用字在 200 字以上的文本里几乎必然出现。
        # "零命中"的情况用 `test_lcs_returns_zero_for_truly_disjoint_text`
        # 单独测（用符号而非汉字），不塞进 parametrize 里假装是普通数据。
        ("", CHUNK, ""),
        ("违约金", CHUNK, "违约金"),
    ],
)
def test_longest_common_substring_finds_the_right_text(a: str, b: str, expected_text: str) -> None:
    """返回的位置必须能**准确切出**那段公共文字。

    断言的是 `b[start:start+length] == 期望文字`，而不是只断言 length > 0 ——
    位置偏一个字符也能让 length 正确，但高亮就标错地方了。
    """
    start, length = longest_common_substring(a, b)
    assert b[start : start + length] == expected_text


def test_lcs_returns_zero_for_truly_disjoint_text() -> None:
    """**完全无公共字符**时必须返回 length=0。

    这条单独写、不用 parametrize，是因为构造"真的完全不重叠"的输入很反直觉：
    中文里常见的字几乎都会在长文本里出现。
    """
    start, length = longest_common_substring("①②③④⑤⑥⑦⑧⑨", CHUNK)
    assert length == 0, f"应当 0 命中，实际命中 {CHUNK[start : start + length]!r}"


def test_longest_common_substring_position_is_in_b() -> None:
    """**位置必须相对于 `b`**，不是 `a`。

    第一版为了省内存交换了 a/b，然后用一个 `swapped` 标志"翻译回去"。
    那让调用方必须记得检查标志位 —— 设计缺陷。现在参数顺序固定，
    位置语义唯一。

    这条测试用两个长度差异很大的输入，专门暴露"交换后位置算错了"的情况。
    """
    short = "万分之五"
    long_text = "前" * 100 + "万分之五" + "后" * 100
    start, length = longest_common_substring(short, long_text)
    assert long_text[start : start + length] == "万分之五"
    assert start == 100, f"位置应是 100，实际 {start}（说明位置算到了错误的字符串上）"


def test_longest_common_substring_is_symmetric_in_result() -> None:
    """交换参数顺序，命中的**文字**不变（只有位置的含义变）。"""
    text = "违约金按日万分之五"
    start1, len1 = longest_common_substring("按日万分之五", text)
    start2, len2 = longest_common_substring(text, "按日万分之五")
    assert len1 == len2
    # 一个是在 text 里的位置，一个是在短串里的位置
    assert text[start1 : start1 + len1] == "按日万分之五"


# ============================================================ 2. 找依据


def test_finds_evidence_when_answer_quotes_original() -> None:
    """答案直接引用原文 → 能找到，且位置精确。"""
    answer = "乙方逾期交付的，按日万分之五支付违约金 [2]。"
    spans = find_evidence_spans(answer, CHUNK)

    assert spans, "应当能找到依据"
    joined = "".join(span.matched for span in spans)
    assert "按日万分之五支付违约金" in joined

    # 位置正确性：用返回的区间去切原文，必须与 matched 一致
    for span in spans:
        assert CHUNK[span.start : span.end] == span.matched


def test_returns_empty_for_heavily_rewritten_answer() -> None:
    """**大幅改写的答案找不到连续 8 字 → 返回空**。

    这不是 bug，是特性：它让 UI 能提示"未能在原文中定位依据"。
    如果这里返回一个"勉强凑出来"的 Span，用户会误以为有依据。

    用完全不同的措辞构造：
    """
    answer = "如果乙方拖延工期，需要赔付百分之零点零五 [1]。"
    spans = find_evidence_spans(answer, CHUNK)
    assert spans == [], f"不该找到依据，实际命中：{[s.matched for s in spans]}"


def test_returns_empty_for_very_short_answer() -> None:
    """太短的句子不参与匹配 —— 否则"是。"会与原文任意几个字假命中。"""
    assert find_evidence_spans("是。", CHUNK) == []
    assert find_evidence_spans("共 3 条。", CHUNK) == []


def test_ignores_citation_markers_when_matching() -> None:
    """引用编号 `[2]` 是我们加的标记，原文里没有 —— 匹配前必须去掉。

    不去掉的后果：`"按日万分之五支付违约金 [2]。"` 的最长公共子串
    会把 `[2]` 也算进去，位置整体偏移。
    """
    with_marker = find_evidence_spans("按日万分之五支付违约金 [2]。", CHUNK)
    without_marker = find_evidence_spans("按日万分之五支付违约金。", CHUNK)

    assert with_marker and without_marker
    assert [(s.start, s.end) for s in with_marker] == [(s.start, s.end) for s in without_marker]


def test_merges_overlapping_spans() -> None:
    """两句答案落在同一段原文里 → 合并成一个高亮区。

    不合并的话会渲染出两个相邻色块，中间夹一段没高亮的字 ——
    看起来像"这段没依据"，实际是同一句。
    """
    # 两句都指向"按日万分之五支付违约金"这片区域
    answer = "按日万分之五支付违约金。按日万分之五支付违约金，累计不超过合同总额。"
    spans = find_evidence_spans(answer, CHUNK)

    assert len(spans) >= 1
    # 合并后区间两两不重叠
    for left, right in zip(spans, spans[1:], strict=False):
        assert left.end <= right.start, f"区间重叠了：{left} {right}"


def test_empty_inputs_return_empty() -> None:
    assert find_evidence_spans("", CHUNK) == []
    assert find_evidence_spans("按日万分之五支付违约金", "") == []


# ============================================================ 3. 高亮渲染


def test_highlight_wraps_only_the_matched_part() -> None:
    """高亮只包住命中片段，前后原文原样保留。"""
    spans = find_evidence_spans("按日万分之五支付违约金。", CHUNK)
    rendered = highlight(CHUNK, spans)

    assert "**" in rendered
    # 去掉标记后必须还原成原文（一个字都不能改）
    assert rendered.replace("**", "") == CHUNK


def test_highlight_preserves_text_exactly() -> None:
    """**关键不变量**：去掉高亮标记后必须与原文完全一致。

    这保证"高亮"只是加视觉标记，绝不改写内容 ——
    用户是拿这段去对照 PDF 的，改一个字就搜不到。
    """
    spans = [Span(start=10, end=20, matched=CHUNK[10:20])]
    rendered = highlight(CHUNK, spans)
    assert rendered.replace("**", "") == CHUNK
    assert rendered[10:12] == "**"


def test_highlight_without_spans_returns_original() -> None:
    assert highlight(CHUNK, []) == CHUNK


def test_highlight_ignores_overlapping_spans() -> None:
    """手工传入重叠区间时不能渲染出乱码（防御性）。"""
    spans = [
        Span(start=5, end=15, matched="x"),
        Span(start=10, end=20, matched="y"),  # 与上一个重叠
    ]
    rendered = highlight(CHUNK, spans)
    assert rendered.replace("**", "") == CHUNK
    assert rendered.count("**") == 2, "重叠部分只应有一对标记"


def test_highlight_uses_markdown_not_html() -> None:
    """用 Markdown `**` 而不是 HTML `<mark>`。

    **理由不是审美**：`st.markdown` 默认不渲染原始 HTML，
    要开 `unsafe_allow_html=True` —— 而文档内容来自用户上传，
    开那个开关等于给自己开一个 XSS 面。
    """
    rendered = highlight(CHUNK, [Span(start=0, end=3, matched=CHUNK[:3])])
    assert "<mark>" not in rendered
    assert rendered.startswith("**")


# ============================================================ 4. 阈值


def test_min_match_threshold_is_documented_and_reasonable() -> None:
    """阈值 8 是**经验值**，不是调优结果。

    太小（4）会让"合同"、"甲方"这类通用词造成大量假命中；
    太大（15）则模型稍作改写就匹配不上。
    这条测试只保证它没被随手改成明显不合理的值。
    """
    assert 6 <= MIN_MATCH_CHARS <= 12, f"阈值 {MIN_MATCH_CHARS} 偏离合理范围"


def test_threshold_is_boundary_checked() -> None:
    """恰好 8 个字能命中，7 个字不行 —— 边界要明确。"""
    eight = "按日万分之五支付"  # 8 字
    seven = "按日万分之五支"  # 7 字
    assert find_evidence_spans(eight + "。", CHUNK), "8 字应当命中"
    assert find_evidence_spans(seven + "。", CHUNK) == [], "7 字不该命中"
