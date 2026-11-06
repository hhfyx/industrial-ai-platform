"""改写层的离线测试：三条边界 + 归一化细节。

计划要求的验收是"离线测试绿（不发请求）"，三条边界逐条锁住：

1. **首轮不改写** —— 空历史不调模型、原样返回。这是"最容易做错的一件事"：
   改写对首轮生效会让 W7 的单轮回归题跑出不同结果，回归变红且很难归因。
2. **改写失败降级** —— 模型输出坏 / 调用失败时用原问，检索链路不挂。
3. **指代歧义反问** —— needs_clarification=true 时回填原问 + 带出 missing，
   硬猜会让检索走向错误方向且无法归因。

全部离线：用 `tests/fake_llm.py` 的 `FakeLLMClient` 按剧本回放，
patch 位置与 `test_structured.py` 相同（patch structured 模块里
已 import 进来的 `get_llm_client`）。异步用 `@pytest.mark.anyio`。
"""

from __future__ import annotations

import pytest

from app.core.llm import structured as S
from app.rag import query_rewrite as Q
from app.rag.query_rewrite import RewriteOutcome, rewrite
from tests.fake_llm import FakeLLMClient, json_reply

# 两个模型的假 client 各写一份 —— patch 的是不同模块的 get_llm_client，
# 但 FakeLLMClient 本体同构。这个 fixture 与 test_structured.patch_client 同款。


@pytest.fixture
def patch_rewrite_client(monkeypatch: pytest.MonkeyPatch):
    """把 structured_call 内部的 client 换成 FakeLLMClient。"""

    def _apply(fake: FakeLLMClient) -> FakeLLMClient:
        monkeypatch.setattr(S, "get_llm_client", lambda *a, **k: fake)
        return fake

    return _apply


HISTORY = [
    {"role": "user", "content": "介绍一下 HX-200 型伺服电机"},
    {"role": "assistant", "content": "HX-200 是 200W 伺服电机，额定扭矩 0.64 N·m。"},
]


# ============================================================ 边界①：首轮不改写


@pytest.mark.anyio
async def test_first_turn_skips_model_entirely(patch_rewrite_client) -> None:
    """空历史 → 原样返回，**一次模型调用都不发生**。

    不只是"返回原问"，是 client 根本不该被碰 —— W7 的 200 条单轮题
    跑在"当前系统"上必须得到与昨天逐位相同的结果。
    """
    fake = patch_rewrite_client(FakeLLMClient(replies=[]))  # 没有剧本：被调就炸剧本
    outcome = await rewrite("伺服电机的公差标准是什么？", [])

    assert outcome == RewriteOutcome(standalone_question="伺服电机的公差标准是什么？")
    assert outcome.rewrote is False
    assert outcome.degraded is False
    assert fake.calls == [], "首轮绝不能调模型"


@pytest.mark.anyio
async def test_none_history_treated_as_first_turn(patch_rewrite_client) -> None:
    """`history=None` 与空列表同义 —— 调用方（CLI/eval）最容易传的就是 None。"""
    fake = patch_rewrite_client(FakeLLMClient(replies=[]))
    outcome = await rewrite("违约金怎么算？", None)
    assert outcome.standalone_question == "违约金怎么算？"
    assert fake.calls == []


@pytest.mark.anyio
async def test_history_with_only_invalid_entries_is_first_turn(patch_rewrite_client) -> None:
    """全是非法项的历史（缺 role / role 不认识）等同空历史。"""
    fake = patch_rewrite_client(FakeLLMClient(replies=[]))
    outcome = await rewrite(
        "公差是多少？", [{"role": "system", "content": "x"}, {"content": "没 role"}]
    )
    assert outcome.standalone_question == "公差是多少？"
    assert fake.calls == []


# ============================================================ 边界②：降级用原问


@pytest.mark.anyio
async def test_degrades_to_original_on_bad_model_output(patch_rewrite_client) -> None:
    """模型输出非法 JSON 且修复轮也失败 → fallback 用原问，不抛异常。

    `structured_call(fallback=...)` 承诺"给了就返回不再抛"；
    这里验证的是**改写层把 fallback 组装对了**（standalone_question=原问），
    以及降级在产物上可见（degraded=True，可归因）。
    """
    patch_rewrite_client(FakeLLMClient(replies=["这不是 JSON", "还是不是 JSON", "依然不是"]))
    outcome = await rewrite("它的公差是多少？", HISTORY)
    assert outcome.degraded is True
    assert outcome.standalone_question == "它的公差是多少？"
    assert outcome.needs_clarification is False


@pytest.mark.anyio
async def test_empty_standalone_question_degrades(patch_rewrite_client) -> None:
    """模型输出合法 JSON 但 standalone_question 是空白 → 降级用原问。

    空串进检索层是静默错误（空 query 查全库），必须在改写层拦下。
    """
    patch_rewrite_client(
        FakeLLMClient(
            replies=[
                json_reply(
                    standalone_question="  ",
                    needs_clarification=False,
                    missing=[],
                    rolling_summary="",
                )
            ]
        )
    )
    outcome = await rewrite("它的公差是多少？", HISTORY)
    assert outcome.degraded is True
    assert outcome.standalone_question == "它的公差是多少？"


@pytest.mark.anyio
async def test_success_clears_missing_and_marks_rewrote(patch_rewrite_client) -> None:
    """正常改写：missing 清空（模型顺手写"缺上下文"要拦）、rewrote 可判。"""
    patch_rewrite_client(
        FakeLLMClient(
            replies=[
                json_reply(
                    standalone_question="HX-200 型伺服电机的公差是多少？",
                    needs_clarification=False,
                    missing=["缺上下文"],  # 模型坏习惯：改写成功还填 missing
                    rolling_summary="",
                )
            ]
        )
    )
    outcome = await rewrite("它的公差是多少？", HISTORY)
    assert outcome.degraded is False
    assert outcome.rewrote is True
    assert outcome.standalone_question == "HX-200 型伺服电机的公差是多少？"
    assert outcome.missing == [], "needs_clarification=False 时 missing 必须被清空"
    # 模型的 rolling_summary（本轮历史 ≤3 轮，值是模型给的，原样带回供调用方判空落盘）
    assert isinstance(outcome.rolling_summary, str)


@pytest.mark.anyio
async def test_unchanged_question_reports_not_rewrote(patch_rewrite_client) -> None:
    """改写结果与原问相同时 rewrote=False —— 对照实验的分母靠它。"""
    patch_rewrite_client(
        FakeLLMClient(
            replies=[
                json_reply(
                    standalone_question="它的公差是多少？",
                    needs_clarification=False,
                    missing=[],
                    rolling_summary="",
                )
            ]
        )
    )
    outcome = await rewrite("它的公差是多少？", HISTORY)
    assert outcome.rewrote is False


# ============================================================ 边界③：歧义反问


@pytest.mark.anyio
async def test_ambiguous_referral_returns_original_with_flag(patch_rewrite_client) -> None:
    """needs_clarification=true → 原问回填 + missing 带出，不硬猜。

    UI 路径拿 missing 渲染反问话术；CLI/eval 路径按 needs_clarification
    记标记跳过改写。两层语义都在这一个产物上。
    """
    patch_rewrite_client(
        FakeLLMClient(
            replies=[
                json_reply(
                    standalone_question="上面那个型号的防护等级呢？",
                    needs_clarification=True,
                    missing=["指的是 HX-200 还是 HX-300"],
                    rolling_summary="",
                )
            ]
        )
    )
    outcome = await rewrite("上面那个型号的防护等级呢？", HISTORY)
    assert outcome.needs_clarification is True
    assert outcome.standalone_question == "上面那个型号的防护等级呢？"
    assert outcome.missing == ["指的是 HX-200 还是 HX-300"]
    assert outcome.rewrote is False, "歧义样本不计入改写收益的分母"


# ============================================================ 裁剪与 sink


@pytest.mark.anyio
async def test_history_truncated_from_oldest(patch_rewrite_client) -> None:
    """字符预算超限 → 从最旧截起，最近的（指代先行词所在）必须保留。

    prompt 里应出现最新一轮的内容，不应出现被裁掉的最旧内容。
    """
    fake = patch_rewrite_client(
        FakeLLMClient(
            replies=[
                json_reply(
                    standalone_question="Q",
                    needs_clarification=False,
                    missing=[],
                    rolling_summary="",
                )
            ]
        )
    )
    long_old = "旧" * 500
    await rewrite(
        "它的公差是多少？",
        [
            {"role": "user", "content": long_old},
            {"role": "assistant", "content": "旧答复" * 100},
            {"role": "user", "content": "最新轮里提到的 HX-200 型号"},
            {"role": "assistant", "content": "HX-200 的扭矩是 0.64 N·m"},
        ],
        max_history_chars=120,
    )
    assert fake.calls, "非空历史必须调模型"
    sent = fake.calls[0].messages[0]["content"]
    assert "HX-200 的扭矩是 0.64 N·m" in sent, "最新轮必须保留"
    assert long_old not in sent, "最旧内容必须被裁掉"


@pytest.mark.anyio
async def test_rewrite_sink_receives_record(patch_rewrite_client) -> None:
    """sink 收到完整归因记录 —— W7 纪律：改了什么/为什么没改必须可回放。"""
    records: list[dict] = []
    Q.set_rewrite_sink(records.append)
    try:
        patch_rewrite_client(
            FakeLLMClient(
                replies=[
                    json_reply(
                        standalone_question="HX-200 的公差是多少？",
                        needs_clarification=False,
                        missing=[],
                        rolling_summary="",
                    )
                ]
            )
        )
        await rewrite("它的公差是多少？", HISTORY)
        assert len(records) == 1
        rec = records[0]
        assert rec["rewrote"] is True
        assert rec["degraded"] is False
        assert rec["rounds"] == 1
        assert rec["question"] == "它的公差是多少？"
        assert rec["standalone_question"] == "HX-200 的公差是多少？"

        # 首轮路径也落盘（rewrote=False），这样 eval 的每一行都有出处
        records.clear()
        patch_rewrite_client(FakeLLMClient(replies=[]))
        await rewrite("全新问题", [])
        assert len(records) == 1 and records[0]["rewrote"] is False and records[0]["rounds"] == 0
    finally:
        Q.set_rewrite_sink(None)


def test_normalize_history_drops_display_fields() -> None:
    """citations/timing 等展示字段不进 prompt —— 只留 role/content。"""
    messy = [
        {
            "role": "user",
            "content": "介绍一下 HX-200",
            "citations": [{"index": 1}],
            "timing": {"total_ms": 999},
            "contexts": [{"text": "很长很长的块"}],
        }
    ]
    cleaned = Q._normalize_history(messy)
    assert cleaned == [{"role": "user", "content": "介绍一下 HX-200"}]
