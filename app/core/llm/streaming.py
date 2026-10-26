"""流式调用：把 `stream=True` 的 chunk 流变成可观测的增量事件。

**这一层的价值不是"能逐字打印"，而是把"慢"变成可测量、可中断的东西。**

同步调用的体验问题是：用户要等**全部**生成完才能看到任何东西。
实测同一请求：非流式要等约 3.2 s 才出现第一个字；流式的 **TTFT（首 token 时间）
只有 1.46 s** —— 同样的总耗时，体感差了 2 倍。这就是流式唯一真正的收益。

三个必须处理对的细节（都是实测踩过的）：

1. **usage 默认不返回**。流式下多数网关只在最后一块带 usage，且需要
   `stream_options={"include_usage": True}` 显式要求。不加这个参数，
   你永远算不出成本 —— 而成本是流式最主要的副作用（token 一样烧，只是早看到）。
2. **最后一块的 `choices` 可能是空数组**。带 usage 的那一块
   `choices: []`，如果代码写 `chunk.choices[0]` 就会 IndexError。
3. **`finish_reason` 出现在 delta 为空的那一块上**。
   它是"这一轮生成结束"的唯一信号，不是靠 `delta.content is None` 判断
   （最后一个有内容的块之后还会再来两块：finish + usage）。

**TTFT 的口径要写死**：从**发起请求**到**第一个非空 `delta.content`**。
不是"第一个 chunk 到达"—— 首块常常只有 `role` 或空 delta，
用它算 TTFT 会得到一个漂亮但没用（偏小）的数字。
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Final

from openai import AsyncOpenAI

# 首块常只有 role / 空 delta，不算"首 token"。口径见模块 docstring。
DEFAULT_MAX_TOKENS: Final[int] = 1024


@dataclass
class StreamStats:
    """一次流式调用的实测指标。

    这些是**运维数据**，不是契约字段 —— 所以它们留在这个对象里，
    不塞进 `ExtractResult`（契约 §2.1 只有 4 个字段）。
    """

    ttft_ms: int | None = None
    total_ms: int = 0
    chunk_count: int = 0
    first_token_index: int | None = None
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    text_chars: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            "chunk_count": self.chunk_count,
            "first_token_index": self.first_token_index,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "text_chars": self.text_chars,
        }


async def stream_chat(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    extra_request: dict[str, Any] | None = None,
    stats: StreamStats | None = None,
) -> AsyncIterator[str]:
    """逐块产出文本增量。**只 yield 文本**，指标写进 `stats`。

    为什么把 stats 做成"传入的可变对象"而不是返回值：
    async generator 没法同时 yield 文本又 return 汇总（`return value` 拿不到）。
    传一个对象进去、调用方读它，是这类"边产出边统计"最省事的方式。

    调用方必须在 `aclose()` 或自然结束后读 `stats`：
    生成器被中途 `break` 时，后面的统计代码不会执行 —— 这是 async generator 的固有行为。
    """
    if stats is None:
        stats = StreamStats()

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        # **不加这个参数，usage 永远不会来**（实测：加了才有末块 usage）
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": temperature,
        **(extra_request or {}),
    }

    started = time.perf_counter()
    response = await client.chat.completions.create(**payload)

    try:
        async for chunk in response:
            stats.chunk_count += 1

            # ① 末块：choices 是空数组，只有 usage。**不能**写 chunk.choices[0]。
            if chunk.usage is not None:
                stats.usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }

            if not chunk.choices:
                continue

            choice = chunk.choices[0]

            # ② finish_reason 在 delta 为空的那块上；记录下来，不中断循环
            #    （后面还可能有 usage 块）
            if choice.finish_reason:
                stats.finish_reason = choice.finish_reason

            delta = choice.delta.content
            if not delta:
                continue

            # ③ TTFT 只认第一个**非空 content**
            if stats.ttft_ms is None:
                # max(..., 1)：int() 会把 0.4ms 截成 0，而 0 看起来像"根本没测到"。
                # 任何真实发生过的耗时至少记 1ms，语义才不含糊。
                stats.ttft_ms = max(int((time.perf_counter() - started) * 1000), 1)
                stats.first_token_index = stats.chunk_count

            stats.text_chars += len(delta)
            yield delta
    finally:
        # `total_ms` 放在 finally：调用方提前 break（用户点"停止生成"）时也要有值，
        # 否则统计里会缺掉最需要看的那一类请求（被中断的请求）。
        stats.total_ms = max(int((time.perf_counter() - started) * 1000), 1)


async def collect_stream(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> tuple[str, StreamStats]:
    """把流式结果拼成完整文本 + 指标。用于"要完整答案但想拿到 TTFT"的场景。

    ⚠️ **这不是给用户看的路径**（拼完才返回，等于退化成非流式）。
    它存在的意义是测试与基准：能一次拿到 `text` 和 `stats` 做断言。
    """
    stats = StreamStats()
    parts: list[str] = []
    async for delta in stream_chat(client, model=model, messages=messages, stats=stats, **kwargs):
        parts.append(delta)
    return "".join(parts), stats


__all__ = ["DEFAULT_MAX_TOKENS", "StreamStats", "collect_stream", "stream_chat"]
