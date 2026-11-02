"""错例归因判据的测试。**核心是锁住"② 不再是默认分支"。**

## 这个文件防的是什么

旧版 `classify()` 的最后一个 `return` 是 `candidate = "② 术语/同义词未命中"` ——
**那是默认分支**，不是判断。于是 30 条错例里 15 条（50%）被标成"术语未命中"，
而 W6 周一的数据显示：那 15 条里 **0 条**含稀有标识符。

后果：W6 周一做 BM25 是为了救 ② 类，而 ② 类实际上不存在。
**一个占 50% 的错误标签，比一个占 5% 的贵十倍** —— 它会把整周的改进方向带偏。

## 锁的五件事

| 锁什么 | 防什么 |
| --- | --- |
| ② 只在含稀有标识符时出现 | 回退成默认分支 |
| 零交集归 ⑥ | 又被塞进 ② |
| 高频词共享归 ⑦ | 又被塞进 ② |
| 表格块归 ⑤（且带重叠提示） | 归成"断语义"或"术语" |
| Markdown 分隔线不算断句 | ① 类虚增 |
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import attribute_errors as ae  # noqa: E402


class _Chunk:
    """最小 Chunk 替身：只需 `text` 与 `metadata`。"""

    def __init__(self, text: str, *, block: str = "prose") -> None:
        self.text = text
        self.metadata = {"block": block}


class _Item:
    def __init__(
        self,
        question: str,
        *,
        is_refusal: bool = False,
        kind: str = "fact",
        gold_answer: str = "",
    ) -> None:
        self.question = question
        self.is_refusal = is_refusal
        self.type = kind
        self.gold_answer = gold_answer
        self.anchors: list[dict[str, object]] = []


class _Outcome:
    def __init__(self, ranks: tuple[int | None, ...] = (None,)) -> None:
        self.anchor_ranks = ranks


# --------------------------------------------------------------------------- ② 不再是默认分支


def test_term_category_requires_rare_identifier() -> None:
    """**② 只在 query 与 gold 共享稀有标识符时才出现。**

    这是 W6 周日重写判据的核心。旧版把"不是多跳/不是表格/不是断句"的
    全部塞进 ②，于是 15/30 都是"术语未命中"。
    """
    # 含稀有标识符：`invalid_input` 匹配 _RARE_TOKEN（含下划线）
    item = _Item("INVALID_INPUT 返回什么错误码")
    chunk = _Chunk("正文超长返回 422 INVALID_INPUT，属于校验失败。")
    result = ae.classify(
        item=item,
        outcome=_Outcome(),
        gold_chunks=[chunk],
        top50_rank=None,
        top1_score=0.5,
        threshold=0.575,
    )
    assert result["candidate"] == "② 术语未命中"
    assert any("稀有标识符" in s for s in result["signals"])


def test_high_frequency_shared_tokens_do_not_become_term_category() -> None:
    """**只共享高频普通词时，必须归 ⑦，不能归 ②。**

    实测那 15 条里 12 条是这种情况（共享「模型/输入/上限/文档」）。
    这些词全库到处都是、IDF 低，**加词典/改分词一点用都没有**。

    注：`_tokens` 对中文是**按字**切（与 compress.py 同一套规则），
    所以这里要构造"字面上真能对上"的句子 —— 否则会落到 ⑥ 零交集。
    """
    item = _Item("模型的输入长度上限是多少")
    chunk = _Chunk("模型的输入长度上限的说明如下，具体数值见下方表格。")
    result = ae.classify(
        item=item,
        outcome=_Outcome(),
        gold_chunks=[chunk],
        top50_rank=25,
        top1_score=0.58,
        threshold=0.575,
    )
    assert result["candidate"] == "⑦ 语义区分度不足", (
        f"共享高频词不该被判成术语问题，实际判成 {result['candidate']}"
    )
    assert result["candidate"] != "② 术语未命中"


def test_zero_overlap_goes_to_no_literal_overlap() -> None:
    """**零公共 token 归 ⑥** —— 字面完全没对上，BM25/词典再强也没用。

    这里要举一个**真正零交集**的例子。注意中文按字切之后，
    「首字延迟」与「TTFT……时间」其实共享一个「指」字（"指标" vs "指"），
    所以那条会被判 ⑦（有交集但无区分度）—— 那也是对的。

    用纯英文与纯中文对照，才构造得出真正的零交集。
    """
    item = _Item("TTFT 是什么意思")
    chunk = _Chunk("首字等待时间指从请求发出到第一个字出现。")
    result = ae.classify(
        item=item,
        outcome=_Outcome(),
        gold_chunks=[chunk],
        top50_rank=None,
        top1_score=0.52,
        threshold=0.575,
    )
    assert result["candidate"] == "⑥ 字面零交集", (
        f"纯英文 query 对纯中文块应当零交集，实际判成 {result['candidate']}"
    )
    assert any("零公共 token" in s for s in result["signals"])


# --------------------------------------------------------------------------- 表格块


def test_table_chunk_is_long_chunk_not_truncation() -> None:
    """**表格块归 ⑤，不能归 ①（断语义）。**

    表格不以句号结尾是正常的 —— 它不是句子。W5 的第一版归因
    把 7 条判成 ①，其中 5 条其实是表格，那导错了改进方向。
    """
    item = _Item("对外暴露的端点有哪些")
    chunk = _Chunk(
        "| 项目 | 值 |\n| --- | --- |\n| 端点 | `extract` · `chat` |",
        block="table",
    )
    result = ae.classify(
        item=item,
        outcome=_Outcome(),
        gold_chunks=[chunk],
        top50_rank=None,
        top1_score=0.5,
        threshold=0.575,
    )
    assert result["candidate"] == "⑤ 长块被挤掉"
    assert any("表格" in s for s in result["signals"])


def test_table_chunk_with_zero_overlap_flags_both_possibilities() -> None:
    """表格块常常同时"零交集"（实测 8 条里 6 条）。

    判 ⑤ 但**把 ⑥ 的可能写进 signals** —— 让复核的人知道有第二种解释。
    """
    item = _Item("四组入口分别是什么")
    chunk = _Chunk("| 端点 | `a` · `b` |", block="table")
    result = ae.classify(
        item=item,
        outcome=_Outcome(),
        gold_chunks=[chunk],
        top50_rank=None,
        top1_score=0.5,
        threshold=0.575,
    )
    assert result["candidate"] == "⑤ 长块被挤掉"
    assert any("⑥" in s or "零交集" in s for s in result["signals"])


# --------------------------------------------------------------------------- ① 断句判据


def test_markdown_divider_is_not_truncation() -> None:
    """**块尾是 Markdown 分隔线不算断句。**

    实测：f-08 / f-10 / f-27 的块尾是 `\\n\\n---`，第一版把前两条误判成 ①，
    使 ① 从 2 条虚增到 4 条。分隔线不是句子内容 ——
    块尾落在分隔线上恰恰说明**切在段落边界上**，那是正确行为。
    """
    assert ae.looks_truncated("这是一句完整的话。\n\n---") is False
    assert ae.looks_truncated("这是一句完整的话。\n\n***") is False
    assert ae.looks_truncated("这是一句完整的话。\n\n___") is False


def test_real_truncation_is_still_detected() -> None:
    """真的半句结尾仍然要检出 —— 别把判据放松成永远为 False。"""
    assert ae.looks_truncated("这句话没有说完，后面还有") is True
    assert ae.looks_truncated("块首是半句") is True
    assert ae.looks_truncated("正常结束。") is False


def test_only_divider_chunk_is_not_truncated() -> None:
    """整块只有分隔线时不算断句（它没有语义内容）。"""
    assert ae.looks_truncated("---\n") is False
    assert ae.looks_truncated("   \n\n---\n") is False


# --------------------------------------------------------------------------- 分类完备性


def test_refusal_and_multi_hop_keep_their_own_categories() -> None:
    """拒答与多跳的判据没被改动 —— 它们本来就有硬信号。"""
    refusal = ae.classify(
        item=_Item("xxx", is_refusal=True),
        outcome=_Outcome(),
        gold_chunks=[],
        top50_rank=None,
        top1_score=0.9,
        threshold=0.575,
    )
    assert refusal["candidate"] == "refusal_false_accept"

    multi = ae.classify(
        item=_Item("两处都要找", kind="multi_hop"),
        outcome=_Outcome((3, None)),
        gold_chunks=[_Chunk("块一。"), _Chunk("块二。")],
        top50_rank=3,
        top1_score=0.6,
        threshold=0.575,
    )
    assert multi["candidate"] == "③ 多跳需跨块"


def test_classify_always_returns_a_known_category() -> None:
    """**任何输入都必须落到已知类别**，不能返回 `None`。

    旧版有个隐患：`candidate` 初值是 `None`，某些路径可能漏赋值。
    现在每个分支都显式 return。
    """
    known = {
        "① 切分断语义",
        "② 术语未命中",
        "③ 多跳需跨块",
        "⑤ 长块被挤掉",
        "⑥ 字面零交集",
        "⑦ 语义区分度不足",
        "refusal_false_accept",
    }
    cases = [
        (_Item("随机问题"), _Chunk("随机内容。")),
        (_Item("带 INVALID_INPUT 的问题"), _Chunk("含 INVALID_INPUT 的块。")),
        (_Item("零交集问题"), _Chunk("完全不同的内容。")),
        (_Item("表格问题"), _Chunk("| a | b |", block="table")),
    ]
    for item, chunk in cases:
        result = ae.classify(
            item=item,
            outcome=_Outcome(),
            gold_chunks=[chunk],
            top50_rank=None,
            top1_score=0.5,
            threshold=0.575,
        )
        assert result["candidate"] in known, f"{item.question!r} 落到未知类别 {result['candidate']}"


def test_rare_token_detection() -> None:
    """稀有标识符判据本身要正确 —— 它是 ② 类的唯一门槛。"""
    assert ae._is_rare("invalid_input") is True  # 含下划线
    assert ae._is_rare("bge-small-zh") is True  # 含连字符
    assert ae._is_rare("v0.4.1") is True  # 含点
    assert ae._is_rare("retrieval") is True  # 长度 ≥6 的英文
    assert ae._is_rare("模型") is False  # 中文
    assert ae._is_rare("token") is False  # 5 字符英文，太短
