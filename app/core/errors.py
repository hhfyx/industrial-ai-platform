"""全局异常处理：让**所有非 2xx 都是同一个 4 字段形状**（api-spec.md §3）。

难点不是写一个 handler，而是默认行为有三套不同形状：

    RequestValidationError → {"detail": [...]}       FastAPI 的字段校验
    StarletteHTTPException → {"detail": "..."}      starlette 的 404/405
    Exception              → 纯文本 "Internal Server Error"

这里用 5 个 handler 全部焊到 `ErrorResponse` 上：

1. `AppError`            → 自定义，能带 code（业务层抛它）
2. `RequestValidationError` → 422 INVALID_INPUT 或 400 INVALID_JSON（按错误类型分流）
3. `StarletteHTTPException` → 兜住 404/405，同样进契约体
4. `ResponseValidationError` → 500 RESPONSE_CONTRACT_VIOLATION
5. `Exception`          → 500 INTERNAL_ERROR，对外只说一句话

**`Exception` handler 在最外层（ServerErrorMiddleware），绕开所有用户中间件**，
所以响应头 `X-Request-ID` 必须由它自己写 —— 中间件的「后半」在这种情况下根本不执行。
统一出口是 `_error_response()`，body 形状和响应头都只在那里定义一次。
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas.extract import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)

# 请求 id 在 request.state 上的键名，中间件写、handler 读（同一条链路）
REQUEST_ID_ATTR = "request_id"
REQUEST_ID_HEADER = "X-Request-ID"
CONTRACT_HEADER = "X-API-Contract"
CONTRACT_VERSION = "v1"

# 校验错误里 loc 的来源前缀，前端拿到这些前缀对不上表单字段，必须剥掉
_LOC_PREFIXES = ("body", "query", "path", "header", "cookie")


class AppError(Exception):
    """业务异常：带稳定的 code 和 HTTP 状态码。

    不用 `HTTPException` 的原因很实际：它的 `detail` 只能塞字符串，
    而契约要求 `code` 可被程序判断（`INVALID_INPUT` vs `INTERNAL_ERROR`）。
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 422,
        details: list[ErrorDetail] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or []


def _request_id_of(request: Request) -> str:
    """取请求 id：中间件没跑（如 `Exception` 在中间件之外）就现场生成一个。

    现场生成的意义是**报错响应永远有 request_id**，哪怕中间件没执行到。
    代价是它不会出现在访问日志里，所以这是兜底而非常规路径。
    """
    rid = getattr(request.state, REQUEST_ID_ATTR, None)
    if rid:
        return str(rid)
    rid = str(uuid.uuid4())
    request.state.request_id = rid
    return rid


def _error_response(
    *,
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: list[ErrorDetail] | None = None,
) -> JSONResponse:
    """统一错误响应的唯一出口：body 形状与响应头都在这里定。"""
    request_id = _request_id_of(request)
    payload = ErrorResponse(
        code=code,
        message=message,
        request_id=request_id,
        details=details or [],
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(),
        headers={
            # Exception handler 在中间件之外，头只能自己补（api-spec.md §2.4）
            REQUEST_ID_HEADER: request_id,
            CONTRACT_HEADER: CONTRACT_VERSION,
        },
    )


def _strip_loc_prefix(loc: tuple[object, ...]) -> str:
    """把 FastAPI 的 `("body", "fields", 2)` 变成 `fields.2`。

    剥掉来源前缀后前端才能直接 `document.querySelector([name=fields.2])`。
    跨字段校验的 loc 是空元组，归一成 `_model`（前端应归到表单顶部）。
    """
    parts = [str(p) for p in loc]
    if parts and parts[0] in _LOC_PREFIXES:
        parts = parts[1:]
    return ".".join(parts) if parts else "_model"


def _translate_validation_errors(exc: RequestValidationError) -> list[ErrorDetail]:
    """把 pydantic 的错误列表翻成契约里的 `details`。

    一个坑：`fields.2` 报"字符串太短"时，用户看到的是 `fields.2` 而不是字段名，
    因为此时该位置的值本身不合法，取不到内容 —— 这是可接受的，路径足够定位。
    """
    details: list[ErrorDetail] = []
    for error in exc.errors():
        field = _strip_loc_prefix(tuple(error.get("loc", ())))
        reason = str(error.get("msg", "invalid value"))
        # 自定义 model_validator 抛的 ValueError 会带上 "Value error, " 前缀，去掉更干净
        if reason.startswith("Value error, "):
            reason = reason[len("Value error, ") :]
        details.append(ErrorDetail(field=field, reason=reason))
    return details


async def app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """业务异常 → 契约错误体。"""
    assert isinstance(exc, AppError)  # noqa: S101 - handler 只由异常类型表驱动
    _log_handled(request, exc.status_code, exc.code, exc.message)
    return _error_response(
        request=request,
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """结构合法但字段不合规 → 422 INVALID_INPUT；**JSON 都解析不了 → 400 INVALID_JSON**。

    实测踩出来的坑（FastAPI 0.143 / starlette 1.8）：畸形 JSON **不会**以
    `JSONDecodeError` 冒到 handler，框架先把它包成 `RequestValidationError`，
    错误 `type` 是 `json_invalid`。所以 400/422 的分流必须在这里按错误类型做，
    单独注册一个 `JSONDecodeError` handler 是**永远走不到**的死代码。

    这条分界线（400 = 读不懂请求，422 = 读懂了但字段不对）决定了调用方
    能不能自动重试：400 重试无意义，422 改字段即可。混成一个码就等于把判断推给人工。
    """
    assert isinstance(exc, RequestValidationError)  # noqa: S101
    errors = exc.errors()

    if any(error.get("type") == "json_invalid" for error in errors):
        _log_handled(request, 400, "INVALID_JSON", "请求体不是合法 JSON")
        return _error_response(
            request=request,
            status_code=400,
            code="INVALID_JSON",
            message="请求体不是合法的 JSON",
            details=[],
        )

    details = _translate_validation_errors(exc)
    _log_handled(request, 422, "INVALID_INPUT", f"{len(details)} 处校验失败")
    return _error_response(
        request=request,
        status_code=422,
        code="INVALID_INPUT",
        message="请求体校验失败，详见 details",
        details=details,
    )


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """starlette 的 404/405 等 → 同样进契约错误体。

    **必须注册 starlette 版**：路由不存在时是框架抛 starlette 的 `HTTPException`，
    只注册 FastAPI 那个接不住，响应会退回 `{"detail": "..."}`，统一体就漏了一个口子。

    code 的映射刻意只留两个值（api-spec.md §3.1 的「只追加不改旧值」）：
    4xx → `INVALID_INPUT`，5xx → `INTERNAL_ERROR`。契约不认识的 404 复用
    `INVALID_INPUT`，而不是新增一个没有文档的码 —— 调用方少一个分支要写。
    """
    assert isinstance(exc, StarletteHTTPException)  # noqa: S101
    code = "INVALID_INPUT" if 400 <= exc.status_code < 500 else "INTERNAL_ERROR"
    _log_handled(request, exc.status_code, code, str(exc.detail))
    return _error_response(
        request=request,
        status_code=exc.status_code,
        code=code,
        message=str(exc.detail),
        details=[],
    )


async def response_validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """出参不符合契约 = 服务端 bug → 500 RESPONSE_CONTRACT_VIOLATION（api-spec.md §3.1）。

    单独一个 code 的价值：线上看到它就知道**不是调用方的错**，不用去翻请求体。
    """
    assert isinstance(exc, ResponseValidationError)  # noqa: S101
    logger.exception("响应体不符合契约: %s %s", request.method, request.url.path)
    return _error_response(
        request=request,
        status_code=500,
        code="RESPONSE_CONTRACT_VIOLATION",
        message="服务端生成的响应不符合接口契约，请联系维护者",
        details=[],
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底：对外一句话，对内全堆栈（api-spec.md §3.1 INTERNAL_ERROR）。"""
    logger.exception("未捕获异常: %s %s", request.method, request.url.path)
    return _error_response(
        request=request,
        status_code=500,
        code="INTERNAL_ERROR",
        message="服务端处理失败，请携带 request_id 联系维护者",
        details=[],
    )


def _log_handled(request: Request, status_code: int, code: str, message: str) -> None:
    """已处理的错误用 warning（不打堆栈）：它们大多是调用方的问题，堆栈只会淹没日志。"""
    logger.warning(
        "handled %s %s -> %s %s | %s",
        request.method,
        request.url.path,
        status_code,
        code,
        message,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """挂全部 handler。顺序无关（FastAPI 按异常类型精确查找）。"""
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(ResponseValidationError, response_validation_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)


__all__ = [
    "CONTRACT_HEADER",
    "CONTRACT_VERSION",
    "REQUEST_ID_ATTR",
    "REQUEST_ID_HEADER",
    "AppError",
    "register_exception_handlers",
]
