"""可观测基础的测试（W8 周日）：request_id 贯穿 + JSON 日志 + 错误分类计数。

W1 的 D7 题落地验收：中间件生成的 request_id，在**请求链路任意深处**的
日志里都出现——用 contextvar 贯穿，参数透传是侵入式改造（见 logging.py 说明）。
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from app.core.logging import (
    configure_logging,
    count_error,
    error_counts,
    get_request_id,
    set_request_id,
)
from app.main import app

ENDPOINT = "/api/v1/extract"


@pytest.fixture
async def client() -> httpx.AsyncClient:
    """与 test_api.py 同款：ASGI transport 直连，不起端口。"""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


# ============================================================ contextvar 贯穿


def test_request_id_contextvar_roundtrip() -> None:
    set_request_id("trace-xyz")
    assert get_request_id() == "trace-xyz"


@pytest.mark.anyio
async def test_request_id_flows_to_deep_log(client: httpx.AsyncClient) -> None:
    """请求链路深处发的日志带中间件的 request_id（JSON 格式验证贯穿）。"""
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    configure_logging(force=True, fmt="json")
    capture = _Capture()
    logging.getLogger("app").addHandler(capture)
    try:
        # 触发一个会走到服务层深处（含拒答路径日志）的请求
        resp = await client.post(ENDPOINT, json={"text": "合同编号：HT-1", "fields": ["金额"]})
        assert resp.status_code == 200
        rid = resp.headers["x-request-id"]

        # record 上没有 request_id 属性（contextvar 在 formatter 里读）——
        # 用与生产相同的 _JsonFormatter 格式化后解析 JSON 验证贯穿
        from app.core.logging import _JsonFormatter

        fmt = _JsonFormatter()
        rids = []
        for r in records:
            try:
                rids.append(json.loads(fmt.format(r)).get("request_id", ""))
            except Exception:  # noqa: BLE001
                continue
        assert rid in rids, f"请求链路的日志必须带 request_id（contextvar 贯穿），实际：{rids}"
    finally:
        logging.getLogger("app").removeHandler(capture)


@pytest.mark.anyio
async def test_json_log_format(client: httpx.AsyncClient) -> None:
    """JSON 格式：每条日志一行可解析的 JSON，含 time/level/message/request_id。"""
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    configure_logging(force=True, fmt="json")
    capture = _Capture()
    logging.getLogger("app").addHandler(capture)
    try:
        await client.get("/health")
        assert records, "至少有一条日志"
        formatter_line = json.loads(json.dumps({"m": records[0].getMessage()}))  # sanity
        assert "m" in formatter_line
        # 用与 _JsonFormatter 相同的字段抽取验证
        from app.core.logging import _JsonFormatter

        line = _JsonFormatter().format(records[0])
        parsed = json.loads(line)
        assert set(parsed) >= {"time", "level", "logger", "message", "request_id"}
    finally:
        logging.getLogger("app").removeHandler(capture)


# ============================================================ 错误分类计数


@pytest.mark.anyio
async def test_error_counts_by_code(client: httpx.AsyncClient) -> None:
    """422 的 INVALID_INPUT 被按码计数（统一出口 = 唯一计数点）。"""
    before = error_counts().get("INVALID_INPUT", 0)
    resp = await client.post("/api/v1/extract", json={"text": "", "fields": ["金额"]})
    assert resp.status_code == 422
    after = error_counts().get("INVALID_INPUT", 0)
    assert after == before + 1


@pytest.mark.anyio
async def test_metrics_endpoint_exposes_counts(client: httpx.AsyncClient) -> None:
    """/metrics 暴露错误分类计数。"""
    await client.post("/api/v1/extract", json={"text": "", "fields": ["金额"]})
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["errors"]["INVALID_INPUT"] >= 1


def test_count_error_direct() -> None:
    before = error_counts().get("GPU_OOM", 0)
    count_error("GPU_OOM")
    assert error_counts()["GPU_OOM"] == before + 1
