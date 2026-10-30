"""RAG 层测试：loader 之外的 map-reduce 与 API 端点。

**为什么 map-reduce 的测试要格外小心**：
它涉及多次 LLM 调用，天然慢且贵。所以这里**全部用假 client**，
只验证逻辑（分块、层级汇总、失败容忍），真实调用留给 CLI 压测。

重点断言的是"出错时的行为"，因为那才是真跑起来才会遇到的：
- 某一段摘要失败，整篇不能因此失败
- 段数超阈值时走层级汇总
- 空文档给出明确结论而不是崩
"""

from __future__ import annotations

import json

import pytest

from app.rag import Document
from app.rag.summarize import (
    HIERARCHICAL_THRESHOLD,
    REDUCE_BATCH_SIZE,
    SummarySegment,
    segment_documents,
    summarize_document,
)

# ============================================================ 假 client


class _Msg:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str | None) -> None:
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content: str | None) -> None:
        self.choices = [_Choice(content)]


class FakeCompletions:
    def __init__(self, *, fail_indices: set[int] | None = None) -> None:
        self.calls: list[dict] = []
        self.fail_indices = fail_indices or set()

    async def create(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(kwargs)
        index = len(self.calls)
        if index in self.fail_indices:
            raise RuntimeError(f"模拟第 {index} 次调用失败")
        # 从 prompt 里摘一点信息，让每次回复可区分
        user = kwargs["messages"][1]["content"]
        return _Resp(f"摘要#{index}（基于 {len(user)} 字输入）")


class FakeClient:
    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        self.completions = FakeCompletions(**kwargs)
        self.chat = type("Chat", (), {"completions": self.completions})()


def _docs(*texts: str) -> list[Document]:
    return [Document(source="t", page=i + 1, text=t) for i, t in enumerate(texts)]


# ============================================================ 1. 分块


def test_chunk_respects_token_budget() -> None:
    """预算 100 tokens ≈ 142 字符（CHARS_PER_TOKEN=0.7）。分块后每块不该远超它。"""
    docs = _docs(*["x" * 100 for _ in range(10)])
    chunks = segment_documents(docs, chunk_tokens=100)

    assert len(chunks) > 1
    for chunk in chunks:
        # 允许最后一块略微超出（贪心装箱的正常结果）
        assert len(chunk.text) < 400, f"分块过大：{len(chunk.text)}"


def test_chunk_records_page_range() -> None:
    """页码信息要跟着段走 —— 摘要里要能回答"这部分对应原文哪里"。"""
    docs = _docs(*["内容" * 50 for _ in range(6)])
    chunks = segment_documents(docs, chunk_tokens=60)

    for chunk in chunks:
        assert chunk.pages, "分块丢了页码信息"
        assert all(p >= 1 for p in chunk.pages)


def test_chunk_location_is_readable() -> None:
    one = SummarySegment(index=0, text="x", pages=(3,))
    assert one.location == "第 3 页"
    many = SummarySegment(index=0, text="x", pages=(3, 4, 5))
    assert many.location == "第 3–5 页"
    none = SummarySegment(index=0, text="x", pages=())
    assert none.location == "未知位置"


def test_chunk_prefers_not_splitting_documents() -> None:
    """**关键取舍**：尽量按 Document 边界装，而不是按字符硬切。

    段落被切断 = 两半各缺一半语义 = 摘要质量的最大杀手。
    """
    docs = _docs("短段落一", "短段落二", "短段落三")
    chunks = segment_documents(docs, chunk_tokens=3000)  # 预算足够大
    assert len(chunks) == 1, "预算足够时不该拆开"
    assert chunks[0].text.count("短段落") == 3


def test_chunk_splits_long_text_at_sentence_boundary() -> None:
    """超长单段落要被切开，且**优先在句号处切**，不是硬截断。"""
    sentence = "这是一句话。" * 200  # 1200 字
    docs = [Document(source="t", page=1, text=sentence)]
    chunks = segment_documents(docs, chunk_tokens=100)  # 预算 142 字

    assert len(chunks) > 1
    # 除最后一块外，其余应以句号结尾（在句子边界切开）
    for chunk in chunks[:-1]:
        assert chunk.text.endswith("。"), f"没有在句子边界切：...{chunk.text[-20:]!r}"


def test_chunk_empty_documents() -> None:
    assert segment_documents([]) == []
    assert segment_documents([Document(source="t", page=1, text="   ")]) == []


def test_chunk_rejects_bad_budget() -> None:
    with pytest.raises(ValueError, match="chunk_tokens"):
        segment_documents(_docs("x"), chunk_tokens=0)


# ============================================================ 2. map-reduce


@pytest.mark.anyio
async def test_small_document_single_reduce() -> None:
    """段数 <= 阈值：map 各段 + 一次汇总。

    注意 `chunk_tokens` 是 token 数，实际字符预算 = tokens / 0.7。
    这里给足预算（3000 tokens ≈ 4285 字符），3 个短段落必须装进一段。
    """
    client = FakeClient()
    docs = _docs(*[f"第 {i} 段内容" * 5 for i in range(3)])  # 每段约 40 字
    result = await summarize_document(docs, client=client, model="m", chunk_tokens=3000)

    assert result.chunk_count == 1, "预算够大，应只有一段"
    assert result.hierarchical is False
    # 1 次 map + 1 次 reduce
    assert result.api_calls == 2
    assert result.reduce_rounds == 1
    assert result.summary


@pytest.mark.anyio
async def test_many_chunks_use_hierarchical_reduce() -> None:
    """**验收点**：段数 > 10 时启用层级汇总，而不是一次性喂给模型。

    构造足够多的独立 Document，且每段都短于预算 ——
    这样每个 Document 都自成一"块"（贪心装箱：装不下就开新段）。
    """
    client = FakeClient()
    # 每段约 20 字，预算 30 tokens ≈ 42 字符 → 每段恰好装两个 Document。
    # 因此要 > 10 段就需要 > 20 个 Document（之前用 20 个正好装成 10 段，
    # 卡在阈值上不触发层级汇总 —— 这是测试自己算错，不是实现问题）。
    docs = _docs(
        *[f"第 {i:02d} 句话内容共二十字左右。" for i in range(HIERARCHICAL_THRESHOLD * 2 + 4)]
    )
    result = await summarize_document(docs, client=client, model="m", chunk_tokens=30)

    assert result.chunk_count > HIERARCHICAL_THRESHOLD, (
        f"应有 > {HIERARCHICAL_THRESHOLD} 段，实际 {result.chunk_count}"
    )
    assert result.hierarchical is True
    assert result.reduce_rounds > 1, "层级汇总应该有不止一轮"
    assert result.summary


@pytest.mark.anyio
async def test_partial_failure_does_not_fail_whole_summary() -> None:
    """**核心韧性**：某一段摘要失败，整篇摘要仍要产出。

    50 页文档里坏 1 段，剩下 49 段的信息依然有价值 ——
    因为一个段失败就让整篇失败，是最差的产品行为。
    """
    client = FakeClient(fail_indices={2})  # 第二次调用（第 2 段的 map）失败
    docs = _docs(*[f"第 {i} 段内容" * 30 for i in range(4)])
    result = await summarize_document(docs, client=client, model="m", chunk_tokens=150)

    assert result.summary, "整篇摘要不该因为单段失败而变空"
    assert any("失败" in s or "摘要#" in s for s in result.chunk_summaries)
    # 失败的那段被替换成占位文字，而不是抛出去
    assert all(isinstance(s, str) and s for s in result.chunk_summaries)


@pytest.mark.anyio
async def test_empty_document_returns_explicit_message() -> None:
    """没有可摘要文本时给出**明确结论**，而不是空字符串（空字符串会被误当成功）。"""
    client = FakeClient()
    result = await summarize_document([], client=client, model="m")

    assert result.chunk_count == 0
    assert "没有可摘要" in result.summary
    assert client.completions.calls == [], "空文档不该调用模型"


@pytest.mark.anyio
async def test_chunk_summaries_are_numbered_in_reduce_prompt() -> None:
    """reduce 的 prompt 里给各段摘要编号 —— 让模型知道"这是第几部分"，更容易保序。"""
    client = FakeClient()
    docs = _docs(*[f"第 {i} 段。" * 20 for i in range(3)])
    await summarize_document(docs, client=client, model="m", chunk_tokens=150)

    reduce_prompt = client.completions.calls[-1]["messages"][1]["content"]
    assert "[1]" in reduce_prompt
    assert "摘要#" in reduce_prompt


@pytest.mark.anyio
async def test_temperature_is_zero_for_summarization() -> None:
    """摘要要稳定可复现（尽管 temp=0 也不是绝对，见 learn/llm/01_params.md §5）。"""
    client = FakeClient()
    docs = _docs("内容" * 50)
    await summarize_document(docs, client=client, model="m", chunk_tokens=3000)

    assert all(call["temperature"] == 0.0 for call in client.completions.calls)


def test_reduce_batch_size_is_bounded() -> None:
    """每批汇总的段数必须有上限，否则又回到"输入太长"。"""
    assert 1 <= REDUCE_BATCH_SIZE <= 10


# ============================================================ 3. 端点契约（不联网）


def test_summarize_endpoint_registered() -> None:
    from app.main import app

    paths = app.openapi()["paths"]
    assert "/api/v1/summarize/stream" in paths


def test_extract_contract_still_untouched() -> None:
    """**保线**：本周加了一堆端点，v1 的 /extract 必须一字未改。"""
    from app.main import app

    op = app.openapi()["paths"]["/api/v1/extract"]["post"]
    schema = op["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema["$ref"].endswith("ExtractResult")


def test_sse_frame_format() -> None:
    from app.api.v1.summarize import done_frame, sse_frame

    frame = sse_frame({"delta": "杭"})
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    assert json.loads(frame[6:-2]) == {"delta": "杭"}
    assert done_frame() == "data: [DONE]\n\n"


def test_cors_middleware_configured() -> None:
    """CORS 必须配好，否则 file:// 打开的前端会被浏览器拦下。"""
    from app.main import app

    cors = [m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware"]
    assert cors, "没有配置 CORSMiddleware"
    assert cors[0].kwargs.get("allow_origins") == ["*"]


def test_static_ui_is_mounted() -> None:
    """前端的静态页挂在 /ui，且 index.html 存在。"""
    from pathlib import Path

    static = Path(__file__).resolve().parents[1] / "static" / "index.html"
    assert static.exists(), "static/index.html 缺失"
    content = static.read_text(encoding="utf-8")
    # 前端必须用 fetch + ReadableStream（EventSource 不能 POST）
    assert "ReadableStream" in content or "getReader" in content
    assert "AbortController" in content, "中断按钮需要 AbortController"
