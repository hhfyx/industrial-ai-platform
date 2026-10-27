"""SSE 聊天端点的测试（W1 周五）。离线，用假 client 驱动。

覆盖四组：

1. **帧字节形状** —— 双换行、多行 data 拆分、心跳是注释行、
   `[DONE]` 终止。这些错了**不会报错**，只能靠测试锁字节。
2. **响应头** —— `Cache-Control: no-cache` 与 `X-Accel-Buffering: no`。
   任务里点名的"最易漏"，缺了会在 Nginx 后面表现为一片空白。
3. **事件顺序** —— delta* → stats → [DONE]；异常路径 → error → [DONE]
   （错误后也要发 [DONE]，否则客户端会一直等）。
4. **断开检测** —— `DisconnectWatcher` 能拿到 `http.disconnect`，
   且 `stop()` 不留悬挂任务。

注：`is_disconnected()` 在 StreamingResponse 里失效这件事，
在 `app/core/sse.py` 的 docstring 里有实测推导；这里测的是替代方案。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.api.v1.chat import (
    DONE_MARKER,
    PING_INTERVAL,
    SSE_HEADERS,
    SSE_MEDIA_TYPE,
    build_messages,
    done_frame,
    ping_frame,
    sse_data_frame,
)
from app.core.sse import DisconnectWatcher
from app.schemas.chat import ChatStreamRequest

# ============================================================ 1. 帧字节形状


def test_data_frame_ends_with_blank_line() -> None:
    """**双换行是硬要求**：一个空行 = 一个事件结束。少一个都不行。"""
    frame = sse_data_frame({"delta": "hi"})
    assert frame.endswith("\n\n")
    assert frame.startswith("data: ")
    assert frame.count("\n\n") == 1


def test_data_frame_keeps_chinese_unescaped() -> None:
    """中文不能被转成 unicode 转义，否则 curl 里一片乱码。"""
    frame = sse_data_frame({"delta": "杭州"})
    assert "杭州" in frame
    assert "\\u" not in frame


def test_json_payload_with_newline_is_escaped_not_split() -> None:
    """**JSON 路径的换行不需要拆帧**：`json.dumps` 已把换行转义成 `\\n` 两个字符。

    这条是修掉一个误实现的证据。早期版本无条件 `payload.split("\\n")`，
    看起来更"小心"，但 JSON 那条路径根本不会出现字面换行 —— 那段代码是死代码。
    测试用 `{"delta": "line1\\nline2"}` 断言"只有一条 data 行"，
    把"我以为的规范要求"和"实际发生的字节"对齐了。
    """
    frame = sse_data_frame({"delta": "line1\nline2"})
    data_lines = [ln for ln in frame.split("\n") if ln.startswith("data: ")]
    assert len(data_lines) == 1, frame
    # 内容里的换行被转义成 \\n（字面反斜杠+n），仍然只有一行
    assert r"\n" in frame
    # 且这一行仍是合法 JSON
    assert json.loads(data_lines[0][len("data: ") :]) == {"delta": "line1\nline2"}


def test_plain_text_payload_splits_on_newline() -> None:
    """**纯文本路径才需要拆**：这时 data 里真的会出现字面换行。

    不拆的话，客户端会把第二行当成"未知字段"丢弃 —— 内容被静默截断。
    """
    frame = sse_data_frame("line1\nline2")
    assert frame == "data: line1\ndata: line2\n\n"
    data_lines = [ln for ln in frame.split("\n") if ln.startswith("data: ")]
    assert len(data_lines) == 2


def test_string_payload_passes_through() -> None:
    assert sse_data_frame("hello") == "data: hello\n\n"


def test_done_frame_exact() -> None:
    """`[DONE]` 是 OpenAI 生态约定，客户端按它判结束。"""
    assert done_frame() == "data: [DONE]\n\n"
    assert DONE_MARKER == "[DONE]"


def test_ping_frame_is_comment_line() -> None:
    """心跳必须以 `:` 开头（注释行），客户端会忽略但连接被保活。"""
    assert ping_frame() == ": ping\n\n"
    assert ping_frame().startswith(":")


def test_sse_constants() -> None:
    assert SSE_MEDIA_TYPE == "text/event-stream"
    assert PING_INTERVAL > 0


# ============================================================ 2. 响应头（最易漏）


def test_cache_control_header_present() -> None:
    """**缺了会被代理缓存整个响应**，前端表现为"最后一次性全出"。"""
    assert SSE_HEADERS["Cache-Control"] == "no-cache"


def test_x_accel_buffering_header_present() -> None:
    """**任务点名的最易漏一行**：没有它，Nginx 会攒够 buffer 才吐。

    症状和"代码没写流式"完全一样，但真凶是反向代理 —— 能查一整天。
    """
    assert SSE_HEADERS["X-Accel-Buffering"] == "no"


def test_headers_are_case_insensitive_lookup_safe() -> None:
    """HTTP 头名大小写不敏感，但我们的 dict 用规范写法，便于测试稳定断言。"""
    keys = {k.lower() for k in SSE_HEADERS}
    assert {"cache-control", "x-accel-buffering"} <= keys


# ============================================================ 3. 消息构造


def test_build_messages_maps_roles() -> None:
    payload = ChatStreamRequest(
        messages=[
            {"role": "user", "content": " 你好 "},
            {"role": "assistant", "content": "你好！"},
        ]
    )
    messages = build_messages(payload)
    assert messages == [
        {"role": "user", "content": "你好"},  # 已 strip
        {"role": "assistant", "content": "你好！"},
    ]


def test_rejects_empty_and_overlong_content() -> None:
    """入参校验：空消息、超长消息都该 422。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChatStreamRequest(messages=[{"role": "user", "content": "  "}])

    with pytest.raises(ValidationError):
        ChatStreamRequest(messages=[{"role": "user", "content": "x" * 9000}])


def test_rejects_unknown_role() -> None:
    """**刻意不收 `tool` role**：工具调用需要 tool_call_id 配对与注册表执行，
    那是 tool_loop 的职责，不该让聊天接口的调用方注入任意 tool 消息。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChatStreamRequest(messages=[{"role": "tool", "content": "x"}])


def test_rejects_unknown_top_level_key() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChatStreamRequest(messages=[{"role": "user", "content": "x"}], temperature=0.5)


# ============================================================ 4. 断开检测


class _FakeRequest:
    """模拟 Starlette 的 `Request.receive()`。

    真实实现里，`receive` 会在客户端断开时返回 `{"type": "http.disconnect"}`；
    这里用队列按脚本投递，验证 watcher 的行为。
    """

    def __init__(self, messages: list[dict]) -> None:  # noqa: ANN001
        self._messages = list(messages)

    async def receive(self) -> dict:
        if self._messages:
            return self._messages.pop(0)
        # 队列空了就一直等（模拟"客户端还在，没有新事件"）
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}


@pytest.mark.anyio
async def test_watcher_detects_disconnect() -> None:
    """核心：拿到 `http.disconnect` 后 `disconnected` 变 True。"""
    from typing import cast

    request = cast("object", _FakeRequest([{"type": "http.disconnect"}]))
    watcher = DisconnectWatcher(request)  # type: ignore[arg-type]

    assert watcher.disconnected is False
    watcher.start()
    try:
        # 给它几次事件循环机会去读 receive
        for _ in range(20):
            if watcher.disconnected:
                break
            await asyncio.sleep(0.01)
        assert watcher.disconnected is True
    finally:
        await watcher.stop()


@pytest.mark.anyio
async def test_watcher_ignores_non_disconnect_messages() -> None:
    """`http.request` 之类消息不该被误判为断开。"""
    from typing import cast

    request = cast(
        "object",
        _FakeRequest([{"type": "http.request"}, {"type": "http.request"}]),
    )
    watcher = DisconnectWatcher(request)  # type: ignore[arg-type]
    watcher.start()
    try:
        await asyncio.sleep(0.05)
        assert watcher.disconnected is False
    finally:
        await watcher.stop()


@pytest.mark.anyio
async def test_watcher_stop_cancels_task() -> None:
    """`stop()` 必须真正结束后台任务，否则它会一直挂在 receive() 上。"""
    from typing import cast

    request = cast("object", _FakeRequest([]))  # 永远不返回
    watcher = DisconnectWatcher(request)  # type: ignore[arg-type]
    watcher.start()

    task = watcher._task
    assert task is not None and not task.done()

    await watcher.stop()
    assert watcher._task is None
    # 给事件循环一次机会让 cancel 生效
    await asyncio.sleep(0.01)
    assert task.done()


@pytest.mark.anyio
async def test_watcher_start_is_idempotent() -> None:
    """重复 start() 不该起第二个任务。"""
    from typing import cast

    watcher = DisconnectWatcher(cast("object", _FakeRequest([])))  # type: ignore[arg-type]
    watcher.start()
    first = watcher._task
    watcher.start()
    assert watcher._task is first
    await watcher.stop()


@pytest.mark.anyio
async def test_watcher_wait_returns_false_on_timeout() -> None:
    from typing import cast

    watcher = DisconnectWatcher(cast("object", _FakeRequest([])))  # type: ignore[arg-type]
    watcher.start()
    try:
        assert await watcher.wait(0.05) is False
    finally:
        await watcher.stop()


@pytest.mark.anyio
async def test_watcher_survives_receive_exception() -> None:
    """`receive()` 抛异常时，监听器不能崩 —— 它只负责置位。"""

    class _Broken:
        async def receive(self) -> dict:
            raise RuntimeError("connection lost")

    watcher = DisconnectWatcher(_Broken())  # type: ignore[arg-type]
    watcher.start()
    try:
        for _ in range(20):
            if watcher.disconnected:
                break
            await asyncio.sleep(0.01)
        # 抛异常意味着连接有问题，按"已断开"处理是安全的选择
        assert watcher.disconnected is True
    finally:
        await watcher.stop()


# ============================================================ 5. 端到端（假 client 驱动生成器）


@pytest.mark.anyio
async def test_endpoint_streams_deltas_then_done() -> None:
    """用假 client 驱动真实路由，验证事件顺序与 [DONE]。"""
    import app.api.v1.chat as chat_mod

    async def fake_stream_chat(client, *, model, messages, stats=None, **_kw):  # noqa: ANN001, ANN003
        assert stats is not None
        stats.ttft_ms = 100
        stats.usage = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        stats.finish_reason = "stop"
        for piece in ("杭", "州"):
            stats.text_chars += len(piece)
            yield piece

    original = chat_mod.stream_chat
    chat_mod.stream_chat = fake_stream_chat
    try:
        payload = ChatStreamRequest(messages=[{"role": "user", "content": "介绍一下杭州"}])
        from typing import cast

        request = cast("object", _FakeRequest([]))  # 永不返回 receive → 不断开
        response = await chat_mod.chat_stream(payload, request)  # type: ignore[arg-type]

        frames = [
            f.decode("utf-8") if isinstance(f, bytes) else f async for f in response.body_iterator
        ]
        await response.body_iterator.aclose() if hasattr(response.body_iterator, "aclose") else None
    finally:
        chat_mod.stream_chat = original

    joined = "".join(frames)
    # 顺序：delta(delta=杭) → delta(delta=州) → stats → [DONE]
    assert '{"delta": "杭"}' in joined
    assert '{"delta": "州"}' in joined
    assert "ttft_ms" in joined
    assert joined.rstrip().endswith(f"data: {DONE_MARKER}")
    # stats 必须在 [DONE] 之前，否则客户端收不到
    assert joined.index("ttft_ms") < joined.index(DONE_MARKER)


@pytest.mark.anyio
async def test_endpoint_emits_error_frame_then_done() -> None:
    """流开始后出错：发 error 帧 + [DONE]，绝不 raise（状态码已经是 200 了）。"""
    import app.api.v1.chat as chat_mod

    async def boom(*_a, **_kw):  # noqa: ANN002, ANN003
        raise RuntimeError("upstream exploded")
        yield  # pragma: no cover

    original = chat_mod.stream_chat
    chat_mod.stream_chat = boom
    try:
        payload = ChatStreamRequest(messages=[{"role": "user", "content": "x"}])
        from typing import cast

        request = cast("object", _FakeRequest([]))
        response = await chat_mod.chat_stream(payload, request)  # type: ignore[arg-type]
        frames = [
            f.decode("utf-8") if isinstance(f, bytes) else f async for f in response.body_iterator
        ]
    finally:
        chat_mod.stream_chat = original

    joined = "".join(frames)
    assert "STREAM_ERROR" in joined
    assert "upstream exploded" in joined
    # 错误之后仍要发 [DONE]，否则客户端会一直等
    assert joined.rstrip().endswith(f"data: {DONE_MARKER}")


def test_endpoint_registered_and_extract_untouched() -> None:
    """**保线**：新增 /chat/stream 的同时，/api/v1/extract 一字未动。"""
    from app.main import app

    paths = app.openapi()["paths"]
    assert "/api/v1/chat/stream" in paths
    assert "/api/v1/extract" in paths
    # 原端点仍是 JSON 响应，没被改成流
    assert "application/json" in paths["/api/v1/extract"]["post"]["responses"]["200"]["content"]


def test_openapi_documents_chat_endpoint() -> None:
    """`/docs` 里能看到这个端点，且描述里点明了响应头要求。"""
    from app.main import app

    op = app.openapi()["paths"]["/api/v1/chat/stream"]["post"]
    desc = op.get("description", "")
    assert "Cache-Control" in desc
    assert "X-Accel-Buffering" in desc
    assert DONE_MARKER in desc


def test_stats_frame_is_valid_json_after_unwrapping() -> None:
    """运维数据帧必须能被客户端解析（我们只是没保证所有人都解析它）。"""
    frame = sse_data_frame({"usage": {}, "ttft_ms": 100})
    payload = frame[len("data: ") : -2]
    assert json.loads(payload)["ttft_ms"] == 100
