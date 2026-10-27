"""`degraded` 字段与降级路径的 API 级测试（W2 收尾）。

**这一组测的是"契约新增可选字段"是否真的向后兼容。**

关键断言（对应 api-spec §2.5 的三条规则）：

1. **正常结果里没有 `degraded` 键** —— 老的 4 字段客户端零影响
2. **降级结果里才有 `degraded: true`**
3. 降级时 `X-Extract-Model` 带 `+fallback(...)` 后缀，与 body 里的布尔同步

这三条如果只写在文档里，就只是承诺；这里把它们变成可执行的断言。
"""

from __future__ import annotations

import httpx
import pytest

from app.core.deps import get_extractor
from app.main import app
from app.services.extractor import ExtractionOutput, RuleExtractor

ENDPOINT = "/api/v1/extract"
CONTRACT_TEXT = "合同编号：HT-2026-001，甲方：张三，签订日期：2026-10-20"

SUCCESS_KEYS_WITHOUT_DEGRADED = {"request_id", "data", "confidence", "elapsed_ms"}


class _DegradingExtractor:
    """永远返回"已降级"结果的假抽取器。"""

    model_name = "fake+damp"

    async def extract(self, *, text: str, fields: list[str]) -> ExtractionOutput:
        data = dict.fromkeys(fields)
        return ExtractionOutput(
            data=data,
            confidence=0.0,
            model="fake-model+fallback(ValueError)",
            elapsed_ms=1,
            missing=list(fields),
            degraded=True,
        )


class _NormalExtractor:
    """返回正常（未降级）结果的假抽取器。"""

    model_name = "fake-ok"

    async def extract(self, *, text: str, fields: list[str]) -> ExtractionOutput:
        return ExtractionOutput(
            data={name: f"v-{i}" for i, name in enumerate(fields)},
            confidence=1.0,
            model="fake-ok",
            elapsed_ms=5,
            degraded=False,
        )


@pytest.fixture
async def client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


# ============================================================ 1. 向后兼容


@pytest.mark.anyio
async def test_normal_result_has_no_degraded_key(client: httpx.AsyncClient) -> None:
    """**正常结果必须是 4 字段**，不能出现 `degraded: null`。

    这是"新增可选字段"的全部意义。如果序列化成 `"degraded": null`，
    老客户端断言"恰好 4 个键"就会失败 —— 那就是破坏性变更。
    """
    app.dependency_overrides[get_extractor] = lambda: _NormalExtractor()
    try:
        response = await client.post(ENDPOINT, json={"text": CONTRACT_TEXT, "fields": ["合同编号"]})
    finally:
        app.dependency_overrides.pop(get_extractor, None)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == SUCCESS_KEYS_WITHOUT_DEGRADED, f"正常结果不该有额外键：{set(body)}"
    assert "degraded" not in body


@pytest.mark.anyio
async def test_degraded_result_has_the_flag(client: httpx.AsyncClient) -> None:
    """降级结果里才出现 `degraded: true`。"""
    app.dependency_overrides[get_extractor] = lambda: _DegradingExtractor()
    try:
        response = await client.post(ENDPOINT, json={"text": CONTRACT_TEXT, "fields": ["合同编号"]})
    finally:
        app.dependency_overrides.pop(get_extractor, None)

    assert response.status_code == 200, "降级不是错误，仍是 200"
    body = response.json()
    assert body["degraded"] is True
    assert set(body) == SUCCESS_KEYS_WITHOUT_DEGRADED | {"degraded"}


@pytest.mark.anyio
async def test_degraded_flag_syncs_with_model_header(client: httpx.AsyncClient) -> None:
    """body 里的 `degraded` 与响应头的 `+fallback(...)` 必须同步。

    头里是**原因**、body 里是**布尔事实** —— 两者不同步会让排查自相矛盾。
    """
    app.dependency_overrides[get_extractor] = lambda: _DegradingExtractor()
    try:
        response = await client.post(ENDPOINT, json={"text": CONTRACT_TEXT, "fields": ["合同编号"]})
    finally:
        app.dependency_overrides.pop(get_extractor, None)

    model_header = response.headers["x-extract-model"]
    body = response.json()
    assert body["degraded"] is True
    assert "+fallback(" in model_header, f"降级时头里应有 fallback 标记，实际 {model_header}"


@pytest.mark.anyio
async def test_normal_result_header_has_no_fallback(client: httpx.AsyncClient) -> None:
    app.dependency_overrides[get_extractor] = lambda: _NormalExtractor()
    try:
        response = await client.post(ENDPOINT, json={"text": CONTRACT_TEXT, "fields": ["合同编号"]})
    finally:
        app.dependency_overrides.pop(get_extractor, None)

    assert "fallback" not in response.headers["x-extract-model"]


# ============================================================ 2. 真实规则版路径


@pytest.mark.anyio
async def test_rule_extractor_never_marks_degraded(client: httpx.AsyncClient) -> None:
    """规则版**自己**不是降级 —— 它是正常实现，只是能力有限。

    这个区别很重要：把"用规则版"和"降级到规则版"混为一谈，
    会让所有规则版请求都带上 degraded 标记，那标记就失去意义了。
    """
    app.dependency_overrides[get_extractor] = lambda: RuleExtractor()
    try:
        response = await client.post(
            ENDPOINT, json={"text": CONTRACT_TEXT, "fields": ["合同编号", "甲方"]}
        )
    finally:
        app.dependency_overrides.pop(get_extractor, None)

    body = response.json()
    assert "degraded" not in body, "正常用规则版不应标记降级"
    assert body["data"]["合同编号"] == "HT-2026-001"


# ============================================================ 3. 契约文档一致性


def test_api_spec_documents_degraded() -> None:
    """新增字段必须**先写进契约文档**（api-spec §4 规则 2）。"""
    from pathlib import Path

    spec = (Path(__file__).resolve().parents[1] / "docs" / "api-spec.md").read_text(
        encoding="utf-8"
    )
    assert "degraded" in spec
    assert "v1.1" in spec, "新增字段应记入变更记录"
    assert "只在 `true` 时出现" in spec, "必须写明'只在 true 时出现'这条兼容性规则"


def test_openapi_documents_degraded() -> None:
    """`/docs` 里也要能看到这个可选字段，否则调用方不知道它的存在。"""
    schema = app.openapi()
    props = schema["components"]["schemas"]["ExtractResult"]["properties"]
    assert "degraded" in props
    assert "degraded" not in schema["components"]["schemas"]["ExtractResult"]["required"]
