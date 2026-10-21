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
import time
import uuid

from fastapi import FastAPI, Request, Response

from app.core.errors import CONTRACT_HEADER, CONTRACT_VERSION, REQUEST_ID_ATTR, REQUEST_ID_HEADER

logger = logging.getLogger(__name__)

# 响应头里的契约字段；与 api-spec.md §2.4 一一对应
MODEL_HEADER = "X-Extract-Model"
ELAPSED_HEADER = "X-Elapsed-Ms"


async def request_context_middleware(request: Request, call_next):
    """给每个请求挂一个 id，并在响应上补契约头。

    客户端传了 `X-Request-ID` 就沿用 —— 跨服务追踪时上游的 id 才有意义，
    自己重新造一个等于把链路切断。没传才生成。
    """
    incoming = request.headers.get(REQUEST_ID_HEADER)
    request_id = incoming.strip() if incoming and incoming.strip() else str(uuid.uuid4())
    setattr(request.state, REQUEST_ID_ATTR, request_id)

    start = time.perf_counter()
    response: Response = await call_next(request)
    cost_ms = (time.perf_counter() - start) * 1000

    response.headers[REQUEST_ID_HEADER] = request_id
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


__all__ = [
    "ELAPSED_HEADER",
    "MODEL_HEADER",
    "register_middlewares",
    "request_context_middleware",
]
