"""`POST /api/v1/extract/stream` —— v1 契约之外的**新增**流式端点。

**为什么是新增端点，而不是改 `/extract`**：
`docs/api-spec.md` §4 第 2 条写着"新增只能是可选字段，且先改本文件"。
把已有端点改成流式会**破坏所有现有调用方** —— 他们拿到的不再是 JSON。
所以流式是**新路径**，`/extract` 一个字都没动，契约冻住的状态保持不变。

**返回的不是 JSON，是 SSE**（`text/event-stream`）。这意味着：
- OpenAPI 里没法描述它的响应体（SSE 不是 JSON schema 能表达的），
  所以这里的手写文档就是它的全部契约 —— 这也解释了为什么要有 `docs/sse.md`。
- 一旦开始推事件，HTTP 状态码就已经发出去了（200）。**后续任何错误都只能用事件表达**，
  不能再返回 500。这是流式接口最反直觉的一点：错误处理要提前到流开始之前。

事件协议（本端点自己定的，见 docs/sse.md）：

    event: start     data: {"request_id": "...", "model": "..."}
    event: delta     data: {"text": "杭州"}          # 增量，可能很多条
    event: stats     data: {"ttft_ms": 1457, "total_ms": 3195, "usage": {...}}
    event: end       data: {"finished": true, "confidence": 0.9}
    event: error     data: {"code": "...", "message": "..."}   # 只在流开始后出错时出现

为什么不复用 OpenAI 的 `data: [DONE]`：那是**上游厂商的约定，不是 SSE 规范**。
自己定 `event: end` 可以带结构化信息（token 用量、置信度），
而 `[DONE]` 只能说"完了"。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.core.llm.client import get_llm_client
from app.core.llm.streaming import StreamStats, stream_chat
from app.schemas.extract import ExtractRequest

router = APIRouter(prefix="/api/v1", tags=["extract"])

# SSE 的标准媒体类型。浏览器 EventSource 认的就是这个。
SSE_MEDIA_TYPE = "text/event-stream"

# 心跳间隔（秒）。见 docs/sse.md：长时间静默会被代理判定空闲而掐断连接。
PING_INTERVAL = 15.0


def sse_frame(data: Any, *, event: str | None = None, event_id: str | None = None) -> str:
    """拼一个 SSE 帧。**双换行结尾是硬性要求。**

    抽成独立函数（而不是散在生成器里拼字符串）是为了可单测：
    `data:` 与空行之间少一个换行，浏览器就完全不派发事件，而且**不报任何错**。
    这种 bug 靠肉眼读代码发现不了，只能靠测试固定字节形状。
    """
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    parts: list[str] = []
    if event_id is not None:
        parts.append(f"id: {event_id}")
    if event is not None:
        parts.append(f"event: {event}")
    # data 内部的换行必须拆成多条 `data:`，否则换行会被客户端当成字段分隔
    for line in payload.split("\n"):
        parts.append(f"data: {line}")
    return "\n".join(parts) + "\n\n"


def _build_prompt(text: str, fields: list[str]) -> list[dict[str, str]]:
    """构造抽取 prompt。与 `LlmExtractor` 用同一套系统提示词。

    ⚠️ 这里**没有**做原文回指校验（`LlmExtractor._ground_value`）——
    因为流式输出的中间态没法逐字校验，只能在最后一次性校验。
    所以本端点的结果**可能比 `/extract` 更容易出现模型原文里的措辞**，
    不适合直接入库；它是"给人看"的路径，不是"给程序用"的路径。
    这个差异必须写进文档，否则会有人拿它当数据管道。
    """
    from app.services.llm_extractor import SYSTEM_PROMPT

    listed = json.dumps(fields, ensure_ascii=False)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"待抽取的字段（按此顺序返回全部字段）：{listed}\n\n"
                f'输出格式：{{"data": {{"字段名": "值或 null"}}}}\n\n'
                f"正文：\n{text}"
            ),
        },
    ]


@router.post(
    "/extract/stream",
    summary="流式抽取（SSE）",
    description=(
        "与 `/api/v1/extract` **同一套 prompt 与字段语义**，但以 SSE 逐字返回。\n\n"
        "**这不是 v1 契约的一部分**（`/extract` 未改动）。SSE 的响应体无法用 JSON Schema 表达，"
        "本说明与 `docs/sse.md` 一起构成它的契约。\n\n"
        "事件顺序：`start` → 若干 `delta` → `stats` → `end`；"
        "流开始后若出错则发 `error` 并结束（此时 HTTP 状态已是 200，无法再改）。\n\n"
        "**注意**：本路径不做原文回指校验，适合给人看，不适合当数据管道入库。"
    ),
    response_class=StreamingResponse,
)
async def extract_stream(
    payload: ExtractRequest,
    request_id: Annotated[str, Depends(lambda: "")],
) -> StreamingResponse:
    """流式抽取。

    `request_id` 这里用不到 —— 真正的 id 由中间件写进响应头。
    但**保留参数签名里的 request 上下文**是有意的：将来要在事件里带 request_id 时，
    不用改路由签名。
    """
    settings = get_settings()
    llm = settings.llm
    client = get_llm_client(settings)
    extra: dict[str, Any] = {}
    if settings.llm.model and "glm" in llm.model.lower():
        # 本机网关特性：推理模式会先吐一大段 reasoning，TTFT 会变得毫无意义。
        # 只在需要时补偿，换云端时这段自然不触发。
        extra["reasoning_effort"] = "none"

    messages = _build_prompt(payload.text, payload.fields)

    async def event_stream() -> AsyncIterator[str]:
        stats = StreamStats()
        started = time.perf_counter()
        yield sse_frame({"model": llm.model, "fields": payload.fields}, event="start")

        try:
            # 用 asyncio.wait_for 给"首字节"设上限：如果模型迟迟不开口，
            # 连接会一直挂着，客户端什么也收不到。流式接口必须能"失败得快"。
            async def _guarded() -> AsyncIterator[str]:
                async for delta in stream_chat(
                    client,
                    model=llm.model,
                    messages=messages,
                    max_tokens=1024,
                    extra_request=extra,
                    stats=stats,
                ):
                    yield delta

            last_ping = time.perf_counter()
            async for delta in _guarded():
                yield sse_frame({"text": delta}, event="delta")
                # 心跳：正常出字时不需要，但一旦模型卡住，客户端会以为断线
                if time.perf_counter() - last_ping > PING_INTERVAL:
                    yield ": ping\n\n"
                    last_ping = time.perf_counter()

            # 流正常结束：把实测指标一并交出去（这些不进契约，是运维数据）
            yield sse_frame(
                {
                    **stats.as_dict(),
                    # 总耗时包含我们自己拼帧的开销，所以单独给一个端到端值
                    "server_ms": int((time.perf_counter() - started) * 1000),
                },
                event="stats",
            )
            yield sse_frame({"finished": True}, event="end")

        except asyncio.CancelledError:
            # 客户端断开（关标签页 / 点停止）。**必须 re-raise**，
            # 否则会把这个协程当成正常结束，日志里看不到"用户主动中断"。
            raise
        except Exception as exc:  # noqa: BLE001 - 流开始后只能用事件报告错误
            # 这里已经发不出 500 了（状态码早在第一个 yield 时就定了）。
            # 只能把错误当成一条事件交给客户端 —— 这是流式接口的核心约束。
            yield sse_frame(
                {
                    "code": "STREAM_ERROR",
                    "message": f"{type(exc).__name__}: {exc}",
                    "hint": "流式接口一旦开始推事件，状态码就已确定(200)，错误只能走事件",
                },
                event="error",
            )

    return StreamingResponse(
        event_stream(),
        media_type=SSE_MEDIA_TYPE,
        headers={
            # 关掉各级缓冲，否则事件会被攒住 —— 实测 curl 不带 -N 就会这样，
            # 服务端行为和客户端/代理的缓冲是两件事，两边都要关。
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["PING_INTERVAL", "SSE_MEDIA_TYPE", "router", "sse_frame"]
