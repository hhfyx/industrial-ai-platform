"""`POST /api/v1/chat/stream` —— 通用聊天流式接口（SSE）。

**这是今天的主题格。它和周三的 `/extract/stream` 是两个不同的东西：**

| | `/extract/stream` | `/chat/stream` |
| --- | --- | --- |
| 用途 | 按字段抽取的结构化结果 | 开放式对话 |
| 入参 | `text` + `fields` | `messages`（多轮） |
| 终止 | `event: end` + 结构化 stats | **`data: [DONE]`**（与 OpenAI 生态一致） |
| 帧格式 | 命名事件 `event: delta` | 纯 `data: {json}` |

**为什么这个端点用 `[DONE]` 而抽取端点不用**：`[DONE]` 是 OpenAI 生态的既成事实，
凡是"套壳聊天"的客户端（各种 SDK、开源前端）都按它判结束；
对接这类客户端时用 `event: end` 会让他们一直等下去。
反过来，抽取端点是自研协议，用能携带结构化信息的 `event: end` 更划算。

**两件事必须做对，否则前面所有工作都白费：**

1. **`Cache-Control: no-cache` + `X-Accel-Buffering: no`**
   —— 任务里点名的"最易漏"。没有这两行，请求经过 Nginx 时会被整段缓冲：
   服务端明明在逐字推，前端却一直空白，等生成完了才"一次全出"。
   现象和"代码没写流式"完全一样，但真凶是反向代理。**这种 bug 能查一整天。**

2. **客户端断开检测** —— 用户关掉页面后，生成还在继续烧 token。
   **注意：`await request.is_disconnected()` 在这里是失效的**，
   实测在生成器里永远返回 False（详见 `app/core/sse.py` 的推导）。
   本实现用 `DisconnectWatcher` 监听 ASGI `receive`，才能真正拿到断开事件。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.core.llm.client import get_llm_client
from app.core.llm.streaming import StreamStats, stream_chat
from app.core.sse import DisconnectWatcher
from app.schemas.chat import MAX_TOKENS_LIMIT, ChatStreamRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])

SSE_MEDIA_TYPE = "text/event-stream"

# 心跳间隔。15s 足够短，避免被 30s/60s 空闲超时的代理掐断；
# 又足够长，不会在正常出字时污染流。
PING_INTERVAL = 15.0

# OpenAI 生态约定：流结束发这一行。**不是 SSE 规范的一部分**，
# 但对客户端兼容性至关重要（见模块 docstring）。
DONE_MARKER = "[DONE]"

# 这些响应头是 SSE 能正确工作的前提，集中在一处定义便于测试断言。
SSE_HEADERS = {
    # 中间代理不许缓存整个响应
    "Cache-Control": "no-cache",
    # 关掉 nginx 的 proxy_buffering —— **最易漏的一行**。
    # 没有它，nginx 会攒够 buffer 才吐，前端表现为"一直空白，最后一次性全出"，
    # 而服务端日志显示流完全正常。这个 bug 极难查。
    "X-Accel-Buffering": "no",
    # SSE 本就是长连接，显式声明避免被中间层改写
    "Connection": "keep-alive",
}


def sse_data_frame(payload: Any) -> str:
    """一帧纯数据事件：`data: {json}\\n\\n`。

    **双换行是硬要求**：一个空行 = 一个事件结束。少一个换行，
    客户端会把下一条也当成同一事件的一部分，或者根本不派发 —— 而且不报错。

    关于"多行 data"：SSE 规范要求 data 里的换行必须拆成多条 `data:` 行。
    但这里有个容易误解的点 ——

    * `json.dumps()` 会把字符串里的换行**转义**成 `\\n`（反斜杠 + n，两个字符），
      所以**用 JSON 承载内容时，帧里根本不会出现字面换行**，不需要拆。
    * 只有当 payload 本身是**多行字符串**（`isinstance(payload, str)`）时，
      才真的会出现字面换行，必须按行拆开。

    早期版本的实现无条件 `payload.split("\\n")`，看起来"更小心"，实则没有作用 ——
    因为 JSON 那条路径永远不会触发它。测试用 `{"delta": "line1\\nline2"}`
    抓到了这个误解：断言拆成 2 行会失败。**这是好事**：说明测试真的在检查字节。
    """
    if isinstance(payload, str):
        body = payload
        # 纯文本：字面换行必须拆成多条 data:
        lines = "\n".join(f"data: {line}" for line in body.split("\n"))
        return f"{lines}\n\n"

    # JSON 路径：换行已被 json.dumps 转义成 \n 两个字符，不会有字面换行
    body = json.dumps(payload, ensure_ascii=False)
    return f"data: {body}\n\n"


def done_frame() -> str:
    """终止帧。`data: [DONE]\\n\\n`。"""
    return f"data: {DONE_MARKER}\n\n"


def ping_frame() -> str:
    """心跳。以 `:` 开头的注释行是合法 SSE 帧，客户端会忽略但连接被保活。"""
    return ": ping\n\n"


def build_messages(payload: ChatStreamRequest) -> list[dict[str, str]]:
    """把请求消息转成 SDK 需要的 dict 列表。

    抽成独立函数是为了可单测：字段映射错了（如 role 传成 `"human"`）
    会让模型完全不按预期回答，而这种错不会报错。
    """
    return [{"role": m.role, "content": m.content} for m in payload.messages]


@router.post(
    "/chat/stream",
    summary="聊天流式接口（SSE）",
    description=(
        "按 OpenAI 风格逐块返回聊天增量。\n\n"
        "**帧格式**：每块 `data: {json}\\n\\n`，结束发 `data: [DONE]\\n\\n`；"
        "长时间静默时发 `: ping` 注释行保活（客户端会忽略）。\n\n"
        "**为什么用 `[DONE]` 而不是自定义事件**：套壳聊天客户端普遍按 `[DONE]` 判结束，"
        "用它才能直接对接现成 SDK。\n\n"
        "**响应头**带 `Cache-Control: no-cache` 与 `X-Accel-Buffering: no` —— "
        "少了这两行，经过 Nginx 时会被整段缓冲，前端表现为「一直空白，最后一次性全出」。\n\n"
        "**客户端断开时服务端会停止生成**，避免用户关掉页面后继续烧 token。\n\n"
        "错误说明：一旦流开始，HTTP 状态码已固定为 200，后续错误只能以 "
        "`data: {error}` 帧表达，不能再返回 5xx。"
    ),
    response_class=StreamingResponse,
)
async def chat_stream(payload: ChatStreamRequest, request: Request) -> StreamingResponse:
    """聊天流式入口。

    **所有可能失败的准备都在第一个 yield 之前完成**：
    `payload` 已由 FastAPI 校验（失败会返回 422 的 JSON 契约错误体），
    配置在这一行读取。这样"能返回 422"仍然成立 ——
    一旦开始推事件，状态码就再也改不了了。
    """
    settings = get_settings()
    llm = settings.llm
    client = get_llm_client(settings)

    extra: dict[str, Any] = {}
    # 本机网关特性：推理模式先吐一大段思维链，TTFT 会失去意义。
    # 只在检测到 glm 系模型时补偿，换云端时自然不触发。
    if "glm" in llm.model.lower():
        extra["reasoning_effort"] = "none"

    messages = build_messages(payload)

    async def event_stream() -> AsyncIterator[str]:
        stats = StreamStats()
        started = time.perf_counter()
        last_ping = started
        aborted = False

        # 断开监听必须在这里启动（而不是路由函数体里）：
        # 它和这条流是同一个生命周期，流结束就该停。
        watcher = DisconnectWatcher(request)
        watcher.start()

        try:
            async for delta in stream_chat(
                client,
                model=llm.model,
                messages=messages,
                max_tokens=MAX_TOKENS_LIMIT,
                extra_request=extra,
                stats=stats,
            ):
                # **非阻塞**查标志：可以每块都查，开销近似为零。
                # 这就是"关掉页面立刻停止烧 token"的实现。
                if watcher.disconnected:
                    aborted = True
                    logger.info(
                        "chat/stream 客户端断开，停止生成（已发送 %d 块，%d 字，%d ms）",
                        stats.chunk_count,
                        stats.text_chars,
                        int((time.perf_counter() - started) * 1000),
                    )
                    break

                now = time.perf_counter()
                if now - last_ping >= PING_INTERVAL:
                    last_ping = now
                    yield ping_frame()

                yield sse_data_frame({"delta": delta})

            if not aborted:
                # 正常结束：推送运维数据作为普通数据帧（客户端可以不理会）。
                # 必须放在 [DONE] **之前** —— [DONE] 之后就不会再有数据了。
                yield sse_data_frame(
                    {
                        "usage": stats.usage,
                        "ttft_ms": stats.ttft_ms,
                        "total_ms": stats.total_ms,
                        "finish_reason": stats.finish_reason,
                    }
                )
                yield done_frame()
            else:
                # 被中断时不发 [DONE]：客户端已经走了，发了没人收；
                # 而若误发，某些客户端会把半截答案当成完整答案。
                logger.info("chat/stream 已中止，未发送 [DONE]")

        except asyncio.CancelledError:
            # 客户端断开时，Starlette 会取消响应 body 的迭代任务 —— 这里就是那条路径。
            # **必须 re-raise**：吞掉会让 asyncio 认为协程正常结束，
            # 上层（StreamingResponse）也就不会做它该做的清理。
            #
            # 日志放在 finally 里统一打（见下），这样 watcher 路径和取消路径
            # 都只有一条记录，不会重复也不会漏。
            logger.info("chat/stream 被取消（通常是客户端断开）")
            raise
        except Exception as exc:  # noqa: BLE001 - 流已开始，只能用帧报告错误
            # 状态码早在第一个 yield 时就定了(200)，现在只能把错误当数据发。
            logger.exception("chat/stream 流式过程出错")
            with contextlib.suppress(Exception):
                yield sse_data_frame(
                    {
                        "error": {
                            "code": "STREAM_ERROR",
                            "message": f"{type(exc).__name__}: {exc}",
                            "hint": "流已开始，HTTP 状态码无法再更改，错误只能走数据帧",
                        }
                    }
                )
                yield done_frame()
        finally:
            # ---- 断开统计的唯一出口 ----
            # 为什么放在 finally：断开可能经由两条完全不同的路径被感知 ——
            #   ① watcher 置位（主流主动 break）
            #   ② Starlette 取消 body 迭代（CancelledError）
            # 实测两条都可能先到。把日志收在一处，就不会"看起来没检测到断开"。
            if watcher.disconnected:
                logger.info(
                    "chat/stream 客户端断开，停止生成（已发送 %d 块，%d 字，%d ms）",
                    stats.chunk_count,
                    stats.text_chars,
                    int((time.perf_counter() - started) * 1000),
                )
            await watcher.stop()

    return StreamingResponse(
        event_stream(),
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS,
    )


__all__ = [
    "DONE_MARKER",
    "PING_INTERVAL",
    "SSE_HEADERS",
    "SSE_MEDIA_TYPE",
    "build_messages",
    "chat_stream",
    "done_frame",
    "ping_frame",
    "router",
    "sse_data_frame",
]
