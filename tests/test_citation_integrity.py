"""去重与双向校验的离线测试（W8 周二）。

验收逐条：
1. 去重有计数（exact_removed / near_removed / kept）
2. 去重后编号从 1 重排、连续无空洞
3. 双向校验三条断言（越界 / 回查 / 未引用）
4. 越界 → 重答一次 → 仍越界 → degraded

全部离线，复用 `test_rag_query_chain.py` 的 FakeClient 模式。
"""

from __future__ import annotations

import pytest

from app.rag.generation.base import (
    Citation,
    build_context,
    resolve_and_verify,
    verify_citations,
)
from app.rag.generation.dedupe import dedupe_chunks
from app.rag.schema import Chunk


def _chunk(text: str, *, page: int = 1, source: str = "合同.pdf", index: int = 0) -> Chunk:
    return Chunk(text=text, page=page, source=source, index=index)


# ============================================================ 1. 去重


class TestExactDedupe:
    def test_removes_duplicate_chunk_id(self) -> None:
        """chunk_id 相同（同文件重复索引）→ 只留一条。"""
        a = _chunk("违约金按千分之一计算。", page=7)
        b = _chunk("违约金按千分之一计算。", page=7)  # 同 source+page+text → 同 id
        report = dedupe_chunks([a, b], [0.9, 0.5])
        assert report.exact_removed == 1
        assert report.near_removed == 0
        assert len(report.chunks) == 1
        assert report.removed == 1

    def test_keeps_higher_score_duplicate(self) -> None:
        """重复条目保留分数高者（原地替换，位置不变）。"""
        a = _chunk("违约金按千分之一计算。", page=7)
        b = _chunk("违约金按千分之一计算。", page=7)
        report = dedupe_chunks([a, b], [0.5, 0.95])  # 第二条分数更高
        assert len(report.chunks) == 1
        assert report.scores == (0.95,), "必须保留分数高的副本"

    def test_empty_input(self) -> None:
        """空入参不报错，计数全零。"""
        report = dedupe_chunks([], [])
        assert report.chunks == ()
        assert report.removed == 0
        assert report.as_dict() == {"input": 0, "exact_removed": 0, "near_removed": 0, "kept": 0}


class TestNearDedupe:
    def test_overlap_tail_is_merged(self) -> None:
        """overlap 造成的近重复（大量共享文字）→ 合并，保留高分先出现者。"""
        base = "第五条 付款计划。甲方按里程碑分期支付合同价款，验收合格后十五个工作日内付款。" * 5
        first = _chunk(base + "新增的唯一尾部内容甲乙双方权利义务。", page=7)
        second = _chunk(" 开头少量重叠。" + base, page=8)  # 与 first 共享 base 主体
        report = dedupe_chunks([first, second], [0.9, 0.8])
        assert report.near_removed == 1
        assert report.exact_removed == 0
        assert len(report.chunks) == 1

    def test_distinct_chunks_not_merged(self) -> None:
        """同文档不同条款（真实相邻块）→ 不误伤。

        真实 overlap（50/500 = 10% 共享）的 ratio 远低于阈值——
        这条测试锁住"去重不吞正文"的边界。
        """
        c1 = _chunk(
            "违约金按合同总额的千分之一计算，甲方逾期付款按日万分之三支付逾期利息。" * 3, page=7
        )
        c2 = _chunk(
            "验收标准：每一期交付物须经甲方书面确认验收合格，方可进入付款流程。" * 3, page=8
        )
        report = dedupe_chunks([c1, c2], [0.9, 0.8])
        assert report.removed == 0
        assert len(report.chunks) == 2

    def test_no_scores_treated_as_equal(self) -> None:
        """不传 scores → 同分处理，保先出现（测试专用口径）。"""
        a = _chunk("违约金按千分之一计算。", page=7)
        b = _chunk("违约金按千分之一计算。", page=7)
        report = dedupe_chunks([a, b])
        assert report.exact_removed == 1
        assert report.chunks[0].id == a.id


# ============================================================ 2. 编号连续


class TestRenumbering:
    def test_numbers_continuous_after_dedupe(self) -> None:
        """**本周最易漏**：去重后编号必须从 1 重排，连续无空洞。

        build_context 用 len(citations)+1 分配编号 —— 去重砍掉中间块后，
        后面的块编号自动前移。空洞（1,3,5）会让模型以为 2、4 可引用。
        """
        c1 = _chunk("第一条内容甲。", page=1)
        c2 = _chunk("第二条内容乙。", page=2)
        dup = _chunk("第二条内容乙。", page=2)  # 与 c2 同 id
        c3 = _chunk("第三条内容丙。", page=3)
        report = dedupe_chunks([c1, c2, dup, c3], [0.9, 0.8, 0.7, 0.6])
        assert report.removed == 1

        context, citations = build_context(report.chunks)
        assert [c.index for c in citations] == [1, 2, 3], "编号必须连续"
        for n in (1, 2, 3):
            assert f"[{n}]" in context
        # 被去重的块的旧位置内容不该出现在上下文里
        assert "第二条内容乙" in context  # 保留的那条还在
        assert citations[1].page == 2  # 编号 2 现在对应的是原 c2

    def test_near_dedupe_renumbers_too(self) -> None:
        """近重复去重同样触发重排。"""
        base = "第五条 付款计划。甲方按里程碑分期支付合同价款。" * 5
        c1 = _chunk(base, page=1)
        c2 = _chunk("独立块内容。", page=2)
        near = _chunk(" " + base, page=3)  # 近重复
        report = dedupe_chunks([c1, c2, near], [0.9, 0.8, 0.7])
        assert report.near_removed == 1
        _, citations = build_context(report.chunks)
        assert [c.index for c in citations] == [1, 2]


# ============================================================ 3. 双向校验三条断言


def _available(*pages: int) -> list[Citation]:
    chunks = [_chunk(f"第{n}块正文。", page=n) for n in pages]
    _, citations = build_context(chunks)
    return citations


def _chunks_by_id(citations: list[Citation]) -> dict[str, Chunk]:
    return {c.chunk_id: _chunk(f"第{c.index}块正文。", page=c.page or 1) for c in citations}


class TestBidirectionalVerification:
    def test_assertion1_body_refs_subset_of_provided(self) -> None:
        """断言①：正文编号 ⊆ 提供编号。越界编号被抓出来。"""
        available = _available(1, 2, 3)
        verdict = verify_citations("违约金见 [1]，另见 [7]。", available)
        assert verdict["body_refs"] == [1, 7]
        assert verdict["out_of_bounds"] == [7]

    def test_assertion1_in_bounds_passes(self) -> None:
        available = _available(1, 2, 3)
        verdict = verify_citations("见 [1] 和 [3]。", available)
        assert verdict["out_of_bounds"] == []

    def test_assertion2_citation_resolves_to_chunk(self) -> None:
        """断言②：引用列表每项能按 chunk_id 回查到块。"""
        available = _available(1, 2)
        by_id = _chunks_by_id(available)
        verdict = resolve_and_verify("见 [1]。", available, by_id)
        assert verdict["unresolvable_refs"] == []
        assert len(verdict["citations"]) == 1
        assert verdict["citations"][0].chunk_id in by_id

    def test_assertion2_broken_wiring_is_caught(self) -> None:
        """断言②反向：引用列表的 chunk_id 不在本次检索块里 → 断链被抓。

        模拟"编号对、块错位"的接线事故（W4 demo 的 [2] 显示成 [3] 原文）。
        """
        available = _available(1, 2)
        verdict = resolve_and_verify("见 [1]。", available, {})  # 空映射 = 块全丢
        assert verdict["unresolvable_refs"] == [1]

    def test_assertion3_unreferenced_numbers_recorded(self) -> None:
        """断言③：提供的编号没被正文引用 → 记录，不算错误。"""
        available = _available(1, 2, 3)
        verdict = verify_citations("只用了 [2]。", available)
        assert verdict["unreferenced"] == [1, 3]
        assert verdict["out_of_bounds"] == []

    def test_extract_refs_range_form(self) -> None:
        """`[1-3]` 区间展开后参与双向校验。"""
        available = _available(1, 2, 3)
        verdict = verify_citations("见 [1-3]。", available)
        assert verdict["body_refs"] == [1, 2, 3]
        assert verdict["out_of_bounds"] == []


# ============================================================ 4. 越界重答与降级


class _ScriptedCompletions:
    """按剧本回放：第 1 次调用返回 replies[0]，第 2 次返回 replies[1]……"""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def create(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(kwargs)
        reply = self.replies.pop(0) if self.replies else "违约金按千分之一计算 [1]。"
        msg = type("M", (), {"content": reply})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()


class _ScriptedClient:
    def __init__(self, replies: list[str]) -> None:
        self.completions = _ScriptedCompletions(replies)
        self.chat = type("Chat", (), {"completions": self.completions})()


_TWO = [
    _chunk("违约金按合同总额的千分之一计算。", page=7),
    _chunk("逾期超过三十日，甲方有权解除合同。", page=9),
]


@pytest.mark.anyio
async def test_out_of_bounds_triggers_one_retry_then_fixes() -> None:
    """首答越界 [9] → 回灌重答 → 重答只引用 [1] → degraded=False，1 次额外调用。"""
    client = _ScriptedClient(["违约金见 [1] 和 [9]。", "违约金按千分之一计算 [1]。"])
    from app.rag.generation.base import generate_answer

    answer = await generate_answer("违约金", _TWO, client=client, model="m", scores=[0.9, 0.8])

    assert len(client.completions.calls) == 2, "必须恰好重答一次"
    # 回灌的 prompt 里必须点名越界编号
    retry_prompt = client.completions.calls[1]["messages"][1]["content"]
    assert "[9]" in retry_prompt and "只允许引用" in retry_prompt
    assert answer.invalid_refs == []
    assert answer.degraded is False
    assert [c.index for c in answer.citations] == [1]


@pytest.mark.anyio
async def test_retry_still_out_of_bounds_degrades() -> None:
    """重答后**仍然**越界 → 接受答案 + degraded=True。"""
    client = _ScriptedClient(["见 [1] [9]。", "还是见 [1] [9]。"])
    from app.rag.generation.base import generate_answer

    answer = await generate_answer("违约金", _TWO, client=client, model="m", scores=[0.9, 0.8])

    assert len(client.completions.calls) == 2, "只重答一次，不无限重试"
    assert answer.degraded is True
    assert answer.invalid_refs == [9]
    assert answer.text, "降级时正文保留"


@pytest.mark.anyio
async def test_in_bounds_answer_no_retry() -> None:
    """首答就干净 → 只调一次模型，degraded=False。"""
    client = _ScriptedClient(["违约金按千分之一计算 [1]。"])
    from app.rag.generation.base import generate_answer

    answer = await generate_answer("违约金", _TWO, client=client, model="m", scores=[0.9, 0.8])
    assert len(client.completions.calls) == 1
    assert answer.degraded is False
    assert answer.dedupe["kept"] == 2


@pytest.mark.anyio
async def test_unreferenced_recorded_in_answer() -> None:
    """断言③的产物进 Answer.unreferenced —— 预算浪费可对账。"""
    client = _ScriptedClient(["违约金按千分之一计算 [1]。"])
    from app.rag.generation.base import generate_answer

    answer = await generate_answer("违约金", _TWO, client=client, model="m", scores=[0.9, 0.8])
    assert answer.unreferenced == [2]


# ============================================================ 5. 去重计数进 Answer


@pytest.mark.anyio
async def test_dedupe_counter_flows_into_answer() -> None:
    """去重计数随 Answer.dedupe 出来 —— "去重了几条"可对账。"""
    dup = _chunk("违约金按合同总额的千分之一计算。", page=7)  # 与 _TWO[0] 同 id
    client = _ScriptedClient(["违约金按千分之一计算 [1]。"])
    from app.rag.generation.base import generate_answer

    answer = await generate_answer(
        "违约金", [*_TWO, dup], client=client, model="m", scores=[0.9, 0.8, 0.7]
    )
    assert answer.dedupe == {"input": 3, "exact_removed": 1, "near_removed": 0, "kept": 2}
    assert answer.used_chunks == 2
