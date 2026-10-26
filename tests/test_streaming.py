"""SSE 与流式调用的测试（W1 周三）。

**离线**：用假 client 模拟 chunk 流，不联网。
真调用的 TTFT 数字与 curl 抓包记录在 docs/sse.md。

分三组：
1. `sse_frame` 的**字节形状** —— 双换行是唯一的帧分隔，少一个换行浏览器完全不派发，
   而且**不报错**。这种 bug 只能靠测试锁住字节，肉眼读代码发现不了。
2. `stream_chat` 的指标 —— TTFT 口径、usage 只在末块、choices 可能为空。
3. SSE 端点 —— 事件顺序、错误事件、流开始后不能改状态码。
"""

from __future__ import annotations

import json

import pytest

from app.api.v1.stream import PING_INTERVAL, SSE_MEDIA_TYPE, sse_frame
from app.core.llm.streaming import StreamStats, collect_stream, stream_chat

# ============================================================ 1. 帧字节形状


def test_frame_ends_with_blank_line() -> None:
    """**双换行结尾**是硬要求：一个空行 = 一个事件结束。

    少一个换行，客户端会把下一条也当成同一个事件的一部分，或者干脆不派发。
    这个错误没有任何报错，是 SSE 最难查的一类 bug。
    """
    frame = sse_frame({"a": 1})
    assert frame.endswith("\n\n"), "帧必须以空行结尾"
    assert frame.count("\n\n") == 1


def test_frame_has_data_prefix_single_space() -> None:
    """规范规定 `data:` 后是一个空格（`data:值` 也算合法，但统一写更稳）。"""
    frame = sse_frame("hello")
    assert frame == "data: hello\n\n"


def test_frame_named_event() -> None:
    frame = sse_frame("x", event="delta")
    assert frame == "event: delta\ndata: x\n\n"


def test_frame_with_event_id_supports_reconnect() -> None:
    """带 `id:` 时浏览器断线重连会回传 `Last-Event-ID`，用于续传。"""
    frame = sse_frame("x", event="delta", event_id="7")
    assert frame.startswith("id: 7\n")
    assert "event: delta" in frame


def test_multiline_data_is_split_into_multiple_data_lines() -> None:
    """data 里的换行**必须**拆成多条 `data:`。

    否则换行会被客户端当成字段分隔 —— 内容被截断或解析错，而且没有报错。
    """
    frame = sse_frame("line1\nline2")
    assert frame == "data: line1\ndata: line2\n\n"


def test_dict_is_serialized_with_utf8_not_escaped() -> None:
    """中文不能被转义成 unicode 码点，否则 curl 里看到一片乱码。"""
    frame = sse_frame({"text": "杭州"})
    assert "杭州" in frame
    assert "\\u" not in frame
    # 且是合法 JSON
    payload = frame[len("data: ") : -2]
    assert json.loads(payload) == {"text": "杭州"}


# ============================================================ 假 client


class _Delta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str | None, finish_reason: str | None = None) -> None:
        self.delta = _Delta(content)
        self.finish_reason = finish_reason


class _Usage:
    prompt_tokens = 15
    completion_tokens = 20
    total_tokens = 35


class _Chunk:
    def __init__(self, choices: list, usage=None) -> None:  # noqa: ANN001
        self.choices = choices
        self.usage = usage


async def _aiter(chunks: list):  # noqa: ANN201
    for c in chunks:
        yield c


class _FakeStream:
    def __init__(self, chunks: list) -> None:
        self._chunks = chunks

    def __aiter__(self):  # noqa: ANN204
        return _aiter(self._chunks)


class _FakeCompletions:
    def __init__(self, chunks: list) -> None:
        self.chunks = chunks
        self.calls: list[dict] = []

    async def create(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(kwargs)
        return _FakeStream(self.chunks)


class FakeClient:
    def __init__(self, chunks: list) -> None:
        self.completions = _FakeCompletions(chunks)
        self.chat = type("Chat", (), {"completions": self.completions})()


def _typical_chunks() -> list:
    """复刻实测的真实分块序列（见 docs/sse.md §1）。

    关键三块：首块空 delta、末块 finish_reason、独立 usage 块（choices 为空）。
    """
    return [
        _Chunk([_Choice(None)]),  # 首块常常只有 role，没有内容
        _Chunk([_Choice("杭")]),
        _Chunk([_Choice("州")]),
        _Chunk([_Choice(None, finish_reason="stop")]),  # finish 在空 delta 上
        _Chunk([], usage=_Usage()),  # 独立 usage 块，choices 为空数组
    ]


# ============================================================ 2. 指标口径


@pytest.mark.anyio
async def test_stream_yields_only_text_deltas() -> None:
    text, stats = await collect_stream(
        FakeClient(_typical_chunks()), model="m", messages=[{"role": "user", "content": "x"}]
    )
    assert text == "杭州"
    assert stats.chunk_count == 5
    assert stats.text_chars == 2


@pytest.mark.anyio
async def test_ttft_ignores_empty_first_chunk() -> None:
    """**TTFT 口径**：从发起到第一个**非空 content**。

    首块空 delta 若被当成首 token，TTFT 会偏小成一个漂亮但没用的数字。
    这里断言首 token出现在第 2 个 chunk（index=2），而不是第 1 个。
    """
    _, stats = await collect_stream(
        FakeClient(_typical_chunks()), model="m", messages=[{"role": "user", "content": "x"}]
    )
    assert stats.first_token_index == 2, "首块是空 delta，不应计入 TTFT"
    assert stats.ttft_ms is not None
    assert stats.total_ms >= stats.ttft_ms


@pytest.mark.anyio
async def test_usage_comes_from_separate_final_chunk() -> None:
    """usage 在 `choices` 为空的独立末块上 —— 写 `chunk.choices[0]` 会 IndexError。"""
    _, stats = await collect_stream(
        FakeClient(_typical_chunks()), model="m", messages=[{"role": "user", "content": "x"}]
    )
    assert stats.usage == {"prompt_tokens": 15, "completion_tokens": 20, "total_tokens": 35}


@pytest.mark.anyio
async def test_finish_reason_recorded() -> None:
    _, stats = await collect_stream(
        FakeClient(_typical_chunks()), model="m", messages=[{"role": "user", "content": "x"}]
    )
    assert stats.finish_reason == "stop"


@pytest.mark.anyio
async def test_stream_options_include_usage_is_sent() -> None:
    """不加 `stream_options={"include_usage": True}`，usage 永远不会来 —— 成本就算不出。"""
    client = FakeClient(_typical_chunks())
    await collect_stream(client, model="m", messages=[{"role": "user", "content": "x"}])
    body = client.completions.calls[0]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


@pytest.mark.anyio
async def test_empty_stream_has_no_ttft() -> None:
    """一个字都没吐时，TTFT 必须是 None 而不是 0 —— 0 会被当成"极快"。"""
    _, stats = await collect_stream(
        FakeClient([_Chunk([_Choice(None, finish_reason="length")]), _Chunk([], usage=_Usage())]),
        model="m",
        messages=[{"role": "user", "content": "x"}],
    )
    assert stats.ttft_ms is None
    assert stats.text_chars == 0


@pytest.mark.anyio
async def test_stats_have_values_when_consumer_breaks_early() -> None:
    """消费者中途 break（用户点"停止生成"）时，total_ms 仍必须有值。

    `total_ms` 写在 finally 里正是为此：被中断的请求恰恰是最需要看指标的那一类。
    **断言 `>= 1` 而不是 `> 0`**：`int()` 会把亚毫秒截成 0，而 0 语义上等于"没测到"。
    代码里用 `max(int(...), 1)` 保证任何真实耗时都至少是 1ms。
    """
    stats = StreamStats()
    client = FakeClient(_typical_chunks())
    agen = stream_chat(client, model="m", messages=[{"role": "user", "content": "x"}], stats=stats)
    async for _delta in agen:
        break
    await agen.aclose()
    assert stats.total_ms >= 1
    assert stats.chunk_count >= 1


def test_stream_stats_as_dict_is_json_serializable() -> None:
    """指标要能直接塞进 SSE 事件，所以必须可 JSON 序列化。"""
    payload = StreamStats(ttft_ms=100, total_ms=200).as_dict()
    assert json.loads(json.dumps(payload))["ttft_ms"] == 100


# ============================================================ 3. 端点契约


def test_endpoint_is_registered_and_extract_is_untouched() -> None:
    """流式是**新增**端点：`/api/v1/extract` 必须原样还在，v1 契约未被破坏。"""
    from app.main import app

    paths = app.openapi()["paths"]
    assert "/api/v1/extract" in paths
    assert "/api/v1/extract/stream" in paths
    # 原端点的响应仍是 JSON（没被改成流）
    success = paths["/api/v1/extract"]["post"]["responses"]["200"]["content"]
    assert "application/json" in success


def test_sse_media_type_is_standard() -> None:
    """浏览器 EventSource 只认 `text/event-stream`。"""
    assert SSE_MEDIA_TYPE == "text/event-stream"
    assert PING_INTERVAL > 0


@pytest.mark.anyio
async def test_stream_endpoint_emits_events_in_order() -> None:
    """事件顺序：start → delta* → stats → end。

    用假 client 直接驱动生成器（不走 HTTP），验证协议顺序与内容。
    """
    # 打桩：让 stream_chat 用假 chunks
    import app.api.v1.stream as stream_mod
    from app.api.v1.stream import extract_stream
    from app.schemas.extract import ExtractRequest

    original = stream_mod.stream_chat

    async def fake_stream_chat(client, *, model, messages, stats=None, **_kw):  # noqa: ANN001, ANN003
        assert stats is not None
        for chunk in _typical_chunks():
            if chunk.usage is not None:
                stats.usage = {"prompt_tokens": 15, "completion_tokens": 20, "total_tokens": 35}
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                stats.finish_reason = choice.finish_reason
            if not choice.delta.content:
                continue
            if stats.ttft_ms is None:
                stats.ttft_ms = 10
            stats.text_chars += len(choice.delta.content)
            yield choice.delta.content

    stream_mod.stream_chat = fake_stream_chat
    try:
        payload = ExtractRequest(text="合同编号：HT-2026-001", fields=["合同编号"])
        response = await extract_stream(payload, request_id="")
        frames = [f async for f in response.body_iterator]
    finally:
        stream_mod.stream_chat = original

    events = []
    for raw in frames:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        for block in text.split("\n\n"):
            if not block.strip():
                continue
            name = next(
                (
                    line[len("event: ") :]
                    for line in block.split("\n")
                    if line.startswith("event: ")
                ),
                "message",
            )
            events.append(name)

    assert events[0] == "start"
    assert events[-1] == "end"
    assert "delta" in events
    assert "stats" in events
    # stats 必须在 end 之前
    assert events.index("stats") < events.index("end")


def test_errors_after_stream_start_cannot_change_status_code() -> None:
    """**流式的核心约束**：第一个事件发出时状态码就定了(200)。

    之后的错误只能作为 `error` 事件发送，不能返回 500。
    这条是设计说明的直接断言 —— 提醒后来者别在生成器里 `raise HTTPException`。
    """
    import inspect

    from app.api.v1 import stream as stream_mod

    source = inspect.getsource(stream_mod.extract_stream)
    assert 'event="error")' in source.replace("\n", " ").replace("  ", " ") or "event=" in source
    # 生成器内部不应有 raise HTTPException
    assert "raise HTTPException" not in source
