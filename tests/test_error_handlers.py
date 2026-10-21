"""全局异常处理的测试（契约 §3 的"统一"是靠这些分支撑起来的）。

为什么单独一个文件：这几条路径**在正常业务里跑不到**（404、未捕获异常、
出参违约），但每一条都在契约里承诺了形状。不测 = 承诺没有证据。

为什么自己造一个 app 而不是往线上 `app` 里塞 `/debug/boom`：
测试用的路由不该出现在生产 `/docs` 里，那会变成"线上有一个永远 500 的接口"。
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import ResponseValidationError

from app.core.errors import (
    AppError,
    _error_response,
    _request_id_of,
    register_exception_handlers,
    response_validation_error_handler,
    unhandled_exception_handler,
)
from app.core.middleware import register_middlewares
from app.schemas.extract import ErrorResponse

ERROR_KEYS = {"code", "message", "request_id", "details"}


def _build_app() -> FastAPI:
    """与 main.py 同款接线，但路由是测试专用的"必然出错"路由。"""
    test_app = FastAPI()
    register_middlewares(test_app)
    register_exception_handlers(test_app)
    # 注册异常处理器后，`Exception` 的兜底会被 starlette 的 ServerErrorMiddleware 接住；
    # 测试里要让异常冒出来变成响应，而不是把测试进程打崩。
    test_app.add_exception_handler(Exception, unhandled_exception_handler)

    @test_app.get("/api/v1/boom")
    async def boom() -> None:
        raise RuntimeError("故意的崩溃")

    @test_app.get("/api/v1/business-error")
    async def business_error() -> None:
        raise AppError("NOT_FOUND", "文档不存在", status_code=404)

    return test_app


@pytest.fixture
async def boom_client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=_build_app(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.mark.anyio
async def test_unknown_path_404_uses_unified_body(boom_client: httpx.AsyncClient) -> None:
    """路由不存在时也必须进契约体。

    这条最容易漏：框架抛的是 **starlette** 的 `HTTPException`，
    只注册 FastAPI 那个版本接不住，响应会退回 `{"detail": "..."}`。
    """
    response = await boom_client.get("/api/v1/no-such-endpoint")

    assert response.status_code == 404
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["request_id"] == response.headers["x-request-id"]


@pytest.mark.anyio
async def test_unhandled_exception_returns_500_without_leaking_details(
    boom_client: httpx.AsyncClient,
) -> None:
    """未捕获异常：对外只有一句话，**不能泄露堆栈或异常原文**。

    `X-Request-ID` 必须由这个 handler 自己写 —— 它在中间件之外，
    中间件的"后半"根本不会执行（实测确认，写错的话报错响应就没有这个头了）。
    """
    response = await boom_client.get("/api/v1/boom")

    assert response.status_code == 500
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["code"] == "INTERNAL_ERROR"
    assert "故意的崩溃" not in body["message"]
    assert response.headers["x-request-id"] == body["request_id"]
    assert response.headers["x-api-contract"] == "v1"


@pytest.mark.anyio
async def test_app_error_maps_code_and_status(boom_client: httpx.AsyncClient) -> None:
    """业务异常用 `AppError` 才能带 code；`HTTPException` 的 detail 只能塞字符串。"""
    response = await boom_client.get("/api/v1/business-error")

    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "NOT_FOUND"
    assert body["message"] == "文档不存在"


@pytest.mark.anyio
async def test_response_validation_error_is_a_distinct_code() -> None:
    """出参违约单独一个 code：看到它就知道**不是调用方的错**，不用去翻请求体。

    直接调 handler 而不是造一条"违约路由"：这条分支一旦触发就是服务端 bug，
    给它做一条可用路由反而是给线上留了个坏接口。
    """
    request = Request({"type": "http", "method": "POST", "path": "/api/v1/extract", "headers": []})
    exc = ResponseValidationError(
        errors=[{"loc": (), "msg": "bad", "type": "value_error", "input": None}],
        body={},
    )

    response = await response_validation_error_handler(request, exc)
    payload = ErrorResponse.model_validate_json(response.body)

    assert response.status_code == 500
    assert payload.code == "RESPONSE_CONTRACT_VIOLATION"
    assert payload.request_id  # 中间件没跑也能兜出一个 id
    assert payload.request_id in response.headers["x-request-id"]


@pytest.mark.anyio
async def test_request_id_is_generated_when_middleware_did_not_run() -> None:
    """兜底路径：`request.state` 上没有 id 时现场生成一个，保证报错响应永远有 id。"""
    request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})

    first = _request_id_of(request)
    second = _request_id_of(request)

    assert first
    # 生成了就存回 state，同一请求内后续读取必须一致（否则日志对不上）
    assert first == second


@pytest.mark.anyio
async def test_error_response_always_carries_contract_headers() -> None:
    """统一出口只此一处：body 形状 + 两个响应头都在 `_error_response` 里定义。"""
    request = Request({"type": "http", "method": "GET", "path": "/x", "headers": []})

    response = _error_response(
        request=request, status_code=418, code="INVALID_INPUT", message="我是茶壶"
    )

    assert response.status_code == 418
    assert response.headers["x-api-contract"] == "v1"
    assert set(ErrorResponse.model_validate_json(response.body).model_dump()) == (
        ErrorResponse.model_fields.keys() | set()
    )
