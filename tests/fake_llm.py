"""可复用的假 LLM client：让"离线测试"成为默认体验，而不是特例。

**为什么把它放在 `tests/` 而不是 `app/`**：
它是测试基础设施，不该进生产包。放在 `tests/fake_llm.py`，
任何测试文件 `from tests.fake_llm import FakeLLMClient` 即可复用。

**为什么要有这个文件**（而不是每个测试各写一个假 client）：
之前每个测试自己定义一个 `_FakeCompletions`，结果同一件事写了五遍，
而且它们的行为**悄悄不一致**（有的支持 exc、有的不支持；有的记录 messages、有的不记录）。
一旦"假 client"的行为和真 client 差太多，测试就在测一个不存在的东西。

所以这里把假 client 做成**有明确契约**的东西：
- 支持固定回复、按脚本回复、抛异常
- 记录每次调用（messages / kwargs），供断言"回灌内容对不对"
- 支持流式（返回固定 chunk 序列），供 SSE/streaming 测试
- 行为与 OpenAI SDK 的最小表面一致：`client.chat.completions.create(...)`

**它不是 mock，是 stub**：不验证"你调了几次哪个参数"，
而是"给定这个输入，我按剧本返回这个输出"。测试因此能断言**业务行为**
（降级了吗、回灌内容对不对），而不是断言实现细节。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

# ============================================================ 响应对象
# 这些类的形状刻意模仿 OpenAI SDK 的返回对象，
# 让 LlmExtractor 不用为测试做任何特殊处理（它读 .choices[0].message.content）。


@dataclass
class FakeMessage:
    content: str | None = None
    role: str = "assistant"


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str = "stop"


@dataclass
class FakeUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class FakeResponse:
    choices: list[FakeChoice]
    usage: FakeUsage = field(default_factory=FakeUsage)
    model: str = "fake-model"


# ============================================================ 流式 chunk


@dataclass
class FakeDelta:
    content: str | None = None


@dataclass
class FakeStreamChoice:
    delta: FakeDelta
    finish_reason: str | None = None


@dataclass
class FakeChunk:
    """一个流式块。`choices` 可为空（末块只有 usage，这是真实网关的行为）。"""

    choices: list[FakeStreamChoice] = field(default_factory=list)
    usage: FakeUsage | None = None


def stream_chunks(text: str, *, chunk_size: int = 1) -> list[FakeChunk]:
    """把文本切成"逐字/逐块"的 FakeChunk 序列，末尾附 finish + usage 块。

    复刻真实网关的分块结构（learn/llm/docs/sse.md 实测）：
        [首块: 空 delta] → [若干 delta] → [finish_reason] → [choices=[], usage]

    **首块带空 delta**是刻意的：真实网关常这样，而 TTFT 口径必须跳过它。
    """
    chunks: list[FakeChunk] = [FakeChunk(choices=[FakeStreamChoice(delta=FakeDelta(None))])]
    for i in range(0, len(text), chunk_size):
        chunks.append(
            FakeChunk(choices=[FakeStreamChoice(delta=FakeDelta(text[i : i + chunk_size]))])
        )
    chunks.append(
        FakeChunk(choices=[FakeStreamChoice(delta=FakeDelta(None), finish_reason="stop")])
    )
    chunks.append(
        FakeChunk(
            choices=[],
            usage=FakeUsage(
                prompt_tokens=10, completion_tokens=len(text), total_tokens=10 + len(text)
            ),
        )
    )
    return chunks


class FakeAsyncStream:
    """可 `async for` 的流对象。"""

    def __init__(self, chunks: Sequence[FakeChunk]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk

        return gen()


# ============================================================ 调用记录


@dataclass
class FakeCall:
    """一次被记录的调用。测试用它断言"回灌了什么"。"""

    messages: list[dict[str, Any]]
    kwargs: dict[str, Any]

    @property
    def user_content(self) -> str:
        """最后一条 user 消息的内容 —— 回灌断言最常用的入口。"""
        for message in reversed(self.messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    def all_texts(self) -> str:
        """所有消息拼起来，用于"错误信息是否回灌"这类模糊断言。"""
        return "\n".join(str(m.get("content", "")) for m in self.messages)


# ============================================================ 假 client


class FakeLLMClient:
    """假 client：按剧本返回固定 JSON / 固定流，或抛指定异常。

    用法：

        # ① 永远返回同一段输出
        client = FakeLLMClient(replies=['{"甲方": "张三"}'])
        extractor = LlmExtractor(client, model="fake")

        # ② 第一次坏、第二次好（测回灌重试）
        client = FakeLLMClient(replies=['{"甲方": 123}', '{"甲方": "张三"}'])

        # ③ 直接抛异常（测降级）
        client = FakeLLMClient(exc=TimeoutError("boom"))

        # ④ 流式（测 streaming / SSE）
        client = FakeLLMClient(replies=["ignored"], stream_text="杭州很美")

    剧本行为：
    - `replies` 按顺序消费；用完后**重复最后一条**（不是抛错 —— 大多数测试
      不关心"多调了第 N 次"，它们关心前几次的结果）
    - `exc` 优先：每次调用都抛（除非 `exc_once=True`，只抛第一次）
    """

    def __init__(
        self,
        replies: Iterable[str] | None = None,
        *,
        exc: BaseException | None = None,
        exc_once: bool = False,
        stream_text: str | None = None,
        stream_chunk_size: int = 1,
        model: str = "fake-model",
    ) -> None:
        self.replies: list[str] = list(replies) if replies is not None else []
        self.exc = exc
        self.exc_once = exc_once
        self.stream_text = stream_text
        self.stream_chunk_size = stream_chunk_size
        self.model = model
        self.calls: list[FakeCall] = []
        self._used = 0

        self.chat = type("FakeChat", (), {"completions": FakeCompletions(self)})()

    # ------------------------------------------------------------ 供断言
    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_call(self) -> FakeCall | None:
        return self.calls[-1] if self.calls else None

    def _next_reply(self) -> str:
        if not self.replies:
            reply = "{}"
        elif self._used < len(self.replies):
            reply = self.replies[self._used]
            self._used += 1
        else:
            # 用完后重复最后一条：让"多调一次"不会把测试搞崩
            reply = self.replies[-1]
        return reply


class FakeCompletions:
    """`client.chat.completions` 的实现。"""

    def __init__(self, parent: FakeLLMClient) -> None:
        self._parent = parent

    async def create(self, **kwargs: Any) -> Any:
        parent = self._parent
        parent.calls.append(
            FakeCall(messages=list(kwargs.get("messages") or []), kwargs=dict(kwargs))
        )

        # ---- 异常剧本 ----
        if parent.exc is not None:
            if parent.exc_once:
                exc, parent.exc = parent.exc, None
                raise exc
            raise parent.exc

        # ---- 流式剧本 ----
        if kwargs.get("stream"):
            text = parent.stream_text if parent.stream_text is not None else "流式回复"
            return FakeAsyncStream(stream_chunks(text, chunk_size=parent.stream_chunk_size))

        # ---- 普通剧本 ----
        content = parent._next_reply()
        return FakeResponse(
            choices=[FakeChoice(message=FakeMessage(content=content))],
            usage=FakeUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
            model=parent.model,
        )


# ============================================================ 便捷构造


def json_reply(**fields: Any) -> str:
    """一行"模型输出 JSON"的便捷写法：`json_reply(甲方="张三")`。

    注意值传 `None` 会序列化成 `null`（正确），传 Python 字符串会带引号（也正确）。
    """
    return json.dumps(fields, ensure_ascii=False)


def wrapped_reply(**fields: Any) -> str:
    """带 `{"data": {...}}` 外层包装的输出 —— 测"容忍旧格式"用。"""
    return json.dumps({"data": fields}, ensure_ascii=False)


__all__ = [
    "FakeAsyncStream",
    "FakeCall",
    "FakeChoice",
    "FakeChunk",
    "FakeDelta",
    "FakeLLMClient",
    "FakeMessage",
    "FakeResponse",
    "FakeStreamChoice",
    "FakeUsage",
    "json_reply",
    "stream_chunks",
    "wrapped_reply",
]
