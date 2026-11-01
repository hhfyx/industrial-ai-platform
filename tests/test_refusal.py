"""`app/rag/refusal.py` 的契约测试。

**这组测试记录了一个被撤销的设计。** 今天先加了"证据形态"判据
（文档太散 / 分差太小 → 判 weak_evidence），实测把它否掉了：

    纯阈值          误拒 0/20   误放 4/5
    阈值 + 形态     误拒 13/20  误放 0/5

数据在 `scripts/refusal_demo.json`。所以判据回到"空 → 低分"两条，
而这里除了测它们，还**测"形态判据没有被重新加回来"**——
否则下一次有人看到"rf-02 拿 0.6577 还答了"就会想再试一遍。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.generation import REFUSAL_TEXT
from app.rag.refusal import RefusalKind, decide, describe_kinds
from app.rag.schema import Chunk


def _chunks(n: int, *, source: str | list[str] = "合同A.md") -> list[Chunk]:
    """造 n 块。`source` 传字符串则全部同源，传列表则按序分配。"""
    sources = [source] * n if isinstance(source, str) else list(source)
    return [Chunk(text=f"正文第{i}段。", page=None, source=sources[i], index=i) for i in range(n)]


# --------------------------------------------------------------------------- 三条判定


def test_empty_hits_refuses_with_import_advice() -> None:
    """一条都没命中 → 拒答，第一条建议必须是"确认文档已导入"。

    这条建议排第一是因为它是该状态下**唯一确定有意义**的动作：
    没有命中时，用户唯一可能做错的事就是"文档没导进来"。
    """
    verdict = decide([], [], threshold=0.4)
    assert verdict.kind is RefusalKind.EMPTY
    assert verdict.refused is True
    assert "导入" in verdict.advice[0]


def test_low_score_refuses_and_cites_the_numbers() -> None:
    """分数不够 → 拒答，且文案里有实际分数与阈值。

    不给数字时用户只能反复试；给了数字，至少能看出"差得远"还是"差一点"。
    """
    verdict = decide(_chunks(3), [0.31, 0.30, 0.20], threshold=0.4)
    assert verdict.kind is RefusalKind.LOW_SCORE
    assert "0.310" in verdict.text
    assert "0.40" in verdict.text
    assert verdict.top_score == pytest.approx(0.31)


def test_pass_above_threshold() -> None:
    verdict = decide(_chunks(2), [0.7, 0.5], threshold=0.4)
    assert verdict.kind is RefusalKind.ANSWER
    assert verdict.refused is False
    assert verdict.text == ""


def test_boundary_equal_to_threshold_passes() -> None:
    """`score == threshold` 算通过（判据是 `score < threshold` 才拒）。

    边界写反的表现是"刚好达标的永远被拒"，而差一分的错误在对数轴上
    几乎看不出来（0.40 vs 0.4000001）。
    """
    assert decide(_chunks(1), [0.4], threshold=0.4).kind is RefusalKind.ANSWER


def test_low_score_uses_max_not_mean() -> None:
    """看的是**最高分**，不是平均分。

    与 `generation.generate_answer` 的判据必须一致（它用 `max(scores)`）。
    不一致的表现是"离线调好的阈值上线行为不同"，而两边看起来都对。
    """
    # 平均 0.375 < 0.4，但最高 0.45 >= 0.4 → 应当放行
    verdict = decide(_chunks(4), [0.45, 0.40, 0.35, 0.30], threshold=0.4)
    assert verdict.kind is RefusalKind.ANSWER


# --------------------------------------------------------------------------- 被撤销的形态判据


def test_scattered_sources_do_not_block_answering() -> None:
    """**命中散落在多份文档时不拒答** —— 这是撤销形态判据后的行为。

    留这条测试是为了固定那个实测结论：应拒答的 5 条里有 2 条来自 4 份文档，
    而应命中的 20 条里有 8 条也来自 4 份文档。**两者完全重叠**，
    按文档数拦会误伤 40% 的真答案。
    """
    chunks = _chunks(5, source=["a.md", "b.md", "c.md", "d.md", "e.md"])
    verdict = decide(chunks, [0.70, 0.65, 0.60, 0.55, 0.50], threshold=0.4)
    assert verdict.kind is RefusalKind.ANSWER


def test_tiny_margin_does_not_block_answering() -> None:
    """**top1 与 top2 分数接近时不拒答。**

    实测：应命中的最低分差是 0.0012（rg-14），应拒答的分差是 0.0041~0.0311 ——
    **真答案可以比假答案更"平"**。按分差拦等于随机拒掉一半真答案。
    """
    chunks = _chunks(3, source="a.md")
    verdict = decide(chunks, [0.6100, 0.6099, 0.50], threshold=0.4)
    assert verdict.kind is RefusalKind.ANSWER


def test_single_hit_does_not_trip_margin_logic() -> None:
    """单条命中没有"分差"概念，不能因此判异常。

    （旧的形态判据在这里有个真 bug 的风险：`scores[-2]` 在一条时取到自己，
    分差恒为 0，于是所有单条命中都会被判成平局。）
    """
    verdict = decide(_chunks(1), [0.9], threshold=0.4)
    assert verdict.kind is RefusalKind.ANSWER


def test_refusal_module_has_no_shape_criteria_left() -> None:
    """**形态判据不许被悄悄加回来。**

    这条断言读源码而不是行为，因为要防的是"以后有人加一个阈值常量"：
    加回来会让误拒率从 0% 跳到 65%，而那种回归只有跑 25 条标注才能发现。

    ## 只查代码不查 docstring

    模块文档里**必须**保留 `weak_evidence` 的来龙去脉（那是撤销记录，
    删掉它下次还会有人再试），所以不能全文匹配那个词。
    精确的检查是：**枚举成员**与**判据常量**不存在。

        RefusalKind.WEAK_EVIDENCE  —— 会被当成一种结果去渲染
        MAX_SOURCE_SPREAD / MIN_SCORE_MARGIN —— 会被当成一条判据去用
    """
    source = (Path(__file__).resolve().parents[1] / "app" / "rag" / "refusal.py").read_text(
        encoding="utf-8"
    )
    # 枚举成员：形如 `    WEAK_EVIDENCE = "weak_evidence"`
    assert "WEAK_EVIDENCE =" not in source
    # 判据常量：形如 `DEFAULT_MAX_SOURCE_SPREAD: int = 3`
    assert "DEFAULT_MAX_SOURCE_SPREAD" not in source
    assert "DEFAULT_MIN_SCORE_MARGIN" not in source
    # 剩下的枚举必须恰好是这三种
    assert {kind.value for kind in RefusalKind} == {"answer", "low_score", "empty"}


# --------------------------------------------------------------------------- 文案契约


def test_refusal_text_starts_with_generation_constant() -> None:
    """`Refusal.text` 必须以 `REFUSAL_TEXT` 开头。

    而 `Answer.text` 那边要求**恒等**（见 `query.py`）——
    数据层可等值比较，展示层可加建议。
    """
    for scores in ([], [0.1]):
        verdict = decide(_chunks(len(scores) or 1), scores, threshold=0.4)
        if verdict.refused:
            assert verdict.text.startswith(REFUSAL_TEXT)


def test_advice_is_structured_for_ui() -> None:
    """建议必须是**列表**，不能只有一段文字。

    UI 要把它渲染成按钮，CLI 要逐行打印 —— 一段文字会让两边各自去切分，
    而切分逻辑迟早不一致。
    """
    verdict = decide([], [], threshold=0.4)
    assert isinstance(verdict.advice, tuple)
    assert len(verdict.advice) >= 2
    assert all(line and not line.startswith("·") for line in verdict.advice)


def test_headline_is_short_for_ui_title() -> None:
    """`headline` 只取第一句，供 UI 当标题 —— 不要整段塞进标题栏。"""
    verdict = decide([], [], threshold=0.4)
    assert verdict.headline == REFUSAL_TEXT.split("\n")[0]
    assert verdict.headline in verdict.text


def test_diagnostics_carry_numbers_for_debugging() -> None:
    """拒答必须带诊断数据。没有它，"为什么这条被拒"只能靠重跑。"""
    verdict = decide(_chunks(3), [0.31, 0.30, 0.2], threshold=0.4)
    assert verdict.diagnostics["threshold"] == 0.4
    assert verdict.diagnostics["top_score"] == 0.31
    assert len(verdict.diagnostics["scores"]) == 3


def test_every_kind_has_a_description() -> None:
    """三种结果都必须有图例说明 —— 否则 UI 只能显示一个英文枚举值。"""
    descriptions = describe_kinds()
    assert set(descriptions) == {kind.value for kind in RefusalKind}
    assert all(descriptions.values())
