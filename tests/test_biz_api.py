"""biz API 契约测试（docs/module-contract-approval.md §四）。

沿用项目既有契约纪律（tests/test_api.py / test_error_handlers.py 同款）：

- 所有非 2xx 都是 `{code, message, request_id, details}` 4 字段；
- `X-Request-ID` 每个响应都回显；
- 非法状态迁移 / 版本冲突 → 409，形状不偏离统一错误体。

这些用例**不需要 MySQL**：mock 掉 ContractService 的依赖（dependency_overrides），
只验证 HTTP 层的形状 —— 数据正确性是上面并发/幂等测试的职责，分层各有测点。
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.v2.contracts import get_contract_service_dep
from app.core.errors import AppError
from app.main import app


class FakeService:
    """可编程的假 service：路由层的形状测试不需要真库。"""

    def __init__(self) -> None:
        self.next_result: dict | Exception = {"id": 1, "status": "draft"}

    async def create_customer(self, payload, *, idempotency_key=None):
        if isinstance(self.next_result, Exception):
            raise self.next_result
        return {**self.next_result, "name": payload.name}, False

    async def create_contract(self, payload, *, idempotency_key=None):
        if isinstance(self.next_result, Exception):
            raise self.next_result
        return {**self.next_result, "contract_no": payload.contract_no}, False

    async def submit(self, contract_id: int, actor: str):
        if isinstance(self.next_result, Exception):
            raise self.next_result
        return {**self.next_result, "id": contract_id, "status": "pending"}

    async def approve(self, contract_id: int, payload):
        if isinstance(self.next_result, Exception):
            raise self.next_result
        return {**self.next_result, "id": contract_id, "status": "approved"}


@pytest.fixture()
def fake_service(monkeypatch) -> Iterator[FakeService]:
    """把路由依赖指到假 service（FastAPI dependency_overrides 官方注入点）。"""
    svc = FakeService()
    app.dependency_overrides[get_contract_service_dep] = lambda: svc
    yield svc
    app.dependency_overrides.pop(get_contract_service_dep, None)


def _client() -> TestClient:
    return TestClient(app)


def test_error_shape_is_uniform_on_invalid_transition(fake_service) -> None:
    """非法状态迁移 → 409 + 4 字段错误体 + X-Request-ID 回显。"""
    fake_service.next_result = AppError(
        code="INVALID_STATE_TRANSITION",
        message="非法状态迁移：当前是 approved，不能 submit",
        status_code=409,
    )
    resp = _client().post("/api/v2/contracts/1/submit")
    assert resp.status_code == 409
    body = resp.json()
    assert set(body.keys()) == {"code", "message", "request_id", "details"}
    assert body["code"] == "INVALID_STATE_TRANSITION"
    assert "approved" in body["message"]
    assert resp.headers.get("X-Request-ID")


def test_error_shape_on_version_conflict(fake_service) -> None:
    """版本冲突 → 409 VERSION_CONFLICT，形状不变。"""
    fake_service.next_result = AppError(
        code="VERSION_CONFLICT",
        message="该合同已被他人处理",
        status_code=409,
    )
    resp = _client().post(
        "/api/v2/contracts/1/approve",
        json={"comment": "x", "expected_version": 0, "actor": "a"},
    )
    assert resp.status_code == 409
    assert resp.json()["code"] == "VERSION_CONFLICT"


def test_validation_error_is_422_uniform(fake_service) -> None:
    """缺 expected_version → 422（Pydantic 挡在路由前），仍是统一错误体。"""
    resp = _client().post("/api/v2/contracts/1/approve", json={"actor": "a"})
    assert resp.status_code == 422
    body = resp.json()
    assert set(body.keys()) == {"code", "message", "request_id", "details"}


def test_request_id_echo(fake_service) -> None:
    """带 X-Request-ID 的请求，响应原样回显（全项目排障链路的锚点）。"""
    rid = str(uuid.uuid4())
    resp = _client().post("/api/v2/contracts/1/submit", headers={"X-Request-ID": rid})
    assert resp.headers.get("X-Request-ID") == rid


def test_customer_created_contract_created(fake_service) -> None:
    """2xx 形状：201 建客户、201 建合同。"""
    fake_service.next_result = {"id": 7, "status": "draft", "version": 0}
    client = _client()
    r1 = client.post("/api/v2/customers", json={"name": "某建工"})
    assert r1.status_code == 201
    assert r1.json()["id"] == 7
    r2 = client.post(
        "/api/v2/contracts",
        json={
            "contract_no": "HT-20261201-001",
            "customer_id": 7,
            "title": "供货合同",
            "amount_cents": 128_000_000,
        },
    )
    assert r2.status_code == 201
    assert r2.json()["contract_no"] == "HT-20261201-001"


def test_amount_must_be_positive(fake_service) -> None:
    """amount_cents <= 0 → 422（Pydantic gt=0 挡在入口）。"""
    fake_service.next_result = {"id": 1, "status": "draft"}
    resp = _client().post(
        "/api/v2/contracts",
        json={
            "contract_no": "HT-BAD-1",
            "customer_id": 1,
            "title": "零金额",
            "amount_cents": 0,
        },
    )
    assert resp.status_code == 422
