"""请求上下文中间件：生成 / 回显 request_id，并把耗时与契约版本写进响应头。

**为什么 request_id 要由中间件生成、而不是各个 handler 自己造**：
日志、错误体、响应头三处的 id 必须是同一个，否则"报 request_id 查不到日志"。
中间件是唯一能同时看到"请求刚进来"和"响应要出去"的地方。

**边界（实测）**：中间件对 2xx/4xx 都能补头，但**未捕获异常不行** ——
`Exception` handler 在 `ServerErrorMiddleware` 层，比所有用户中间件都外，
中间件的「后半」根本不执行。所以 `errors.py` 里那个 handler 自己写头，两者配合才全覆盖。
"""

from __future__ import annotations

import logging
import os
import random
import time
import uuid

from fastapi import FastAPI, Request, Response

from app.core.errors import CONTRACT_HEADER, CONTRACT_VERSION, REQUEST_ID_ATTR, REQUEST_ID_HEADER
from app.core.logging import set_request_id
from app.core.tracing import TRACE_ID_HEADER, trace_scope

logger = logging.getLogger(__name__)

# 响应头里的契约字段；与 api-spec.md §2.4 一一对应
MODEL_HEADER = "X-Extract-Model"
ELAPSED_HEADER = "X-Elapsed-Ms"


async def request_context_middleware(request: Request, call_next):
    """给每个请求挂一个 id，并在响应上补契约头。

    客户端传了 `X-Request-ID` 就沿用 —— 跨服务追踪时上游的 id 才有意义，
    自己重新造一个等于把链路切断。没传才生成。

    W13 周一：trace 与请求同生命周期。trace_id = request_id（同值两份头，
    显式回写让"观测头"不依赖巧合），整条请求就是一条 trace——包括
    4xx/5xx：错误请求也要有 trace，错误是排查时最先要看的那些数据。
    """
    incoming = request.headers.get(REQUEST_ID_HEADER)
    request_id = incoming.strip() if incoming and incoming.strip() else str(uuid.uuid4())
    setattr(request.state, REQUEST_ID_ATTR, request_id)
    # 同步进 contextvar：全链路的 JSON 日志都带这个 id（W8 周日，request_id 贯穿）
    set_request_id(request_id)

    start = time.perf_counter()
    with trace_scope(
        request_id,
        root_name="http.request",
        root_attrs={"method": request.method, "path": request.url.path},
    ) as root:
        response: Response = await call_next(request)
        cost_ms = (time.perf_counter() - start) * 1000
        # HTTP 状态码是 span 的第一公民：4xx/5xx 让这条 trace 被判为错误
        # （100% 采样），排查时按 trace_id 一次捞全。
        root["attrs"]["status_code"] = response.status_code
        if response.status_code >= 400:
            root["status"] = "error"

    response.headers[REQUEST_ID_HEADER] = request_id
    response.headers[TRACE_ID_HEADER] = request_id
    response.headers[CONTRACT_HEADER] = CONTRACT_VERSION
    # `X-Elapsed-Ms` 由路由自己写（= body 里的 elapsed_ms，契约 §2.4 要求两者相等）。
    # 中间件这里只兜底：路由没写（如 404/405）且是 200 时才补，避免两个数各测各的、
    # 差个 1ms 就变成"契约说是同一个值，实际不是"。实测踩过：中间件测的含序列化开销，
    # 会比服务层多 1–2ms，两者对不上。
    if response.status_code == 200 and ELAPSED_HEADER not in response.headers:
        response.headers[ELAPSED_HEADER] = str(int(cost_ms))

    logger.info(
        "%s %s -> %s | %.1fms | %s",
        request.method,
        request.url.path,
        response.status_code,
        cost_ms,
        request_id,
    )
    return response


def register_middlewares(app: FastAPI) -> None:
    """注册中间件。后注册的在**内层**（starlette 语义），这里只有一个，顺序无关。"""
    app.middleware("http")(request_context_middleware)


async def chaos_middleware(request: Request, call_next):
    """演练注入开关（W13 周三）：CHAOS_DELAY_MS / CHAOS_ERROR_RATE 环境变量。

    慢请求与 5xx 注入是告警演练的唯一诚实手段——不注入就只能在
    生产等事故。默认关（变量不设 = 零开销两行判断）；开了也只是
    演练环境的事，生产配置不设这两个变量。
    """
    import asyncio as _asyncio

    delay = os.environ.get("CHAOS_DELAY_MS", "")
    if delay.isdigit() and int(delay) > 0:
        await _asyncio.sleep(int(delay) / 1000)
    rate = os.environ.get("CHAOS_ERROR_RATE", "")
    try:
        if rate and random.random() < float(rate):
            from app.core.errors import AppError

            raise AppError(
                code="CHAOS_INJECTED",
                message="演练注入的 5xx（CHAOS_ERROR_RATE 开关）",
                status_code=500,
            )
    except ValueError:
        pass
    return await call_next(request)


def register_chaos_middleware(app: FastAPI) -> None:
    """演练注入层。register_middlewares 之后调用（更内层，能包住业务异常）。"""
    if os.environ.get("CHAOS_DELAY_MS", "") or os.environ.get("CHAOS_ERROR_RATE", ""):
        app.middleware("http")(chaos_middleware)


__all__ = [
    "ELAPSED_HEADER",
    "MODEL_HEADER",
    "chaos_middleware",
    "register_chaos_middleware",
    "register_middlewares",
    "request_context_middleware",
]
