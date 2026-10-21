"""API 集成测试（任务 15:30–16:30）：200 与 422 两条路径。

**这一批测试就是 `docs/api-spec.md` 的可执行形态。**
W2 把规则版换成 LLM 时，这个文件**一行都不改**，仍然必须全绿 ——
"改了实现测试还绿"才证明契约真的稳，否则契约只是文档里的一段话。

为什么用 `httpx.AsyncClient(transport=ASGITransport(app=app))` 而不是 `TestClient`：
- 路由是 `async def`，ASGI 传输走的是**同一个事件循环**，测到的就是线上那条路径；
- `TestClient` 是同步包装（starlette 1.x 里还带 `httpx` 弃用提示），
  中间件、异常处理器在它下面多隔了一层，出问题时不好判断是哪层的锅。
"""

from __future__ import annotations

import httpx
import pytest

from app.main import app

ENDPOINT = "/api/v1/extract"

CONTRACT_TEXT = "合同编号：HT-2026-001，甲方：张三，签订日期：2026-10-20，合同金额：¥128,000.00"

# 契约 §3：所有非 2xx 都必须是这 4 个键，一个不多一个不少
ERROR_KEYS = {"code", "message", "request_id", "details"}
# 契约 §2.1：成功响应恒为这 4 个键
SUCCESS_KEYS = {"request_id", "data", "confidence", "elapsed_ms"}


@pytest.fixture
async def client() -> httpx.AsyncClient:
    """每个用例一个独立客户端；transport 直连 ASGI app，不起端口。"""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


# ============================================================ 200 路径
@pytest.mark.anyio
async def test_extract_returns_200_with_contracted_shape(client: httpx.AsyncClient) -> None:
    """验收命令里的那条请求：返回 request_id 且 data.合同编号 == HT-2026-001。"""
    response = await client.post(
        ENDPOINT,
        json={"text": CONTRACT_TEXT, "fields": ["合同编号", "甲方"]},
    )

    assert response.status_code == 200
    body = response.json()

    # 1) 形状：4 个键，不多不少（多一个键 = 服务层偷偷改了契约）
    assert set(body) == SUCCESS_KEYS
    # 2) 验收断言
    assert body["data"]["合同编号"] == "HT-2026-001"
    assert body["data"]["甲方"] == "张三"
    assert body["request_id"]


@pytest.mark.anyio
async def test_request_id_is_consistent_between_header_and_body(
    client: httpx.AsyncClient,
) -> None:
    """`request_id` / `X-Request-ID` 必须是**同一个值**，否则报障对不上日志。"""
    response = await client.post(ENDPOINT, json={"text": CONTRACT_TEXT, "fields": ["合同编号"]})

    assert response.headers["x-request-id"] == response.json()["request_id"]


@pytest.mark.anyio
async def test_incoming_request_id_is_echoed(client: httpx.AsyncClient) -> None:
    """客户端传了 `X-Request-ID` 就沿用 —— 跨服务追踪时上游的 id 才有意义。"""
    response = await client.post(
        ENDPOINT,
        json={"text": CONTRACT_TEXT, "fields": ["合同编号"]},
        headers={"X-Request-ID": "trace-from-upstream"},
    )

    assert response.headers["x-request-id"] == "trace-from-upstream"
    assert response.json()["request_id"] == "trace-from-upstream"


@pytest.mark.anyio
async def test_data_keys_follow_request_field_order(client: httpx.AsyncClient) -> None:
    """`data` 的键顺序 = 请求 `fields` 顺序，抽不到的字段也占位为 null。"""
    fields = ["甲方", "合同编号", "合同中不存在的字段"]
    response = await client.post(ENDPOINT, json={"text": CONTRACT_TEXT, "fields": fields})

    body = response.json()
    assert list(body["data"]) == fields
    assert body["data"]["合同中不存在的字段"] is None
    # 3 个字段命中 2 个 → 2/3 * 0.75 = 0.5
    assert body["confidence"] == 0.5


@pytest.mark.anyio
async def test_contract_headers_are_present(client: httpx.AsyncClient) -> None:
    """契约版本头必须出现：调用方靠它判断兼容性，不靠猜。"""
    response = await client.post(ENDPOINT, json={"text": CONTRACT_TEXT, "fields": ["甲方"]})

    assert response.headers["x-api-contract"] == "v1"
    assert int(response.headers["x-elapsed-ms"]) == response.json()["elapsed_ms"]


# ============================================================ 422 路径
@pytest.mark.anyio
async def test_empty_text_returns_422_with_unified_error_body(
    client: httpx.AsyncClient,
) -> None:
    """空正文 → 422 INVALID_INPUT，且错误体是契约的 4 字段形状。"""
    response = await client.post(ENDPOINT, json={"text": "", "fields": ["合同编号"]})

    assert response.status_code == 422
    body = response.json()

    assert set(body) == ERROR_KEYS
    assert body["code"] == "INVALID_INPUT"
    assert body["request_id"] == response.headers["x-request-id"]
    # details 必须指到具体字段，前端才能高亮输入框
    assert body["details"][0]["field"] == "text"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("payload", "expected_field"),
    [
        ({"text": "x" * 20001, "fields": ["合同编号"]}, "text"),
        ({"text": CONTRACT_TEXT, "fields": []}, "fields"),
        ({"text": CONTRACT_TEXT, "fields": ["甲方", "甲方"]}, "_model"),
        ({"text": CONTRACT_TEXT, "fields": ["甲方"], "extra": 1}, "extra"),
    ],
    ids=["text-too-long", "fields-empty", "fields-duplicated", "unknown-key"],
)
async def test_invalid_payloads_return_422(
    client: httpx.AsyncClient, payload: dict, expected_field: str
) -> None:
    """四类字段级校验失败，都是 422 + 能定位到字段的 details。

    `unknown-key` 一条尤其重要：请求体多一个键必须报错，
    否则拼错的 `texts` 会被静默忽略，调用方拿到一堆 null 还以为是抽取失败。
    """
    response = await client.post(ENDPOINT, json=payload)

    assert response.status_code == 422
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["code"] == "INVALID_INPUT"
    assert expected_field in {d["field"] for d in body["details"]}


@pytest.mark.anyio
async def test_malformed_json_returns_400_not_422(client: httpx.AsyncClient) -> None:
    """畸形 JSON → 400 INVALID_JSON（契约 §3.1 的 400/422 界线）。

    实测坑：FastAPI 把这个错误包成 `RequestValidationError`（type=`json_invalid`），
    单独注册 `JSONDecodeError` handler 是永远走不到的死代码，分流只能在校验 handler 里做。
    """
    response = await client.post(
        ENDPOINT,
        content=b'{"text": "abc", "fields": [',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["code"] == "INVALID_JSON"
    assert body["details"] == []


@pytest.mark.anyio
async def test_error_body_has_request_id_even_for_malformed_json(
    client: httpx.AsyncClient,
) -> None:
    """400 也要带 request_id：报障时这是唯一能查日志的线索。"""
    response = await client.post(
        ENDPOINT, content=b"not-json", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 400
    assert response.json()["request_id"] == response.headers["x-request-id"]


# ============================================================ 契约的"可替换"证明
@pytest.mark.anyio
async def test_openapi_declares_contract_shapes() -> None:
    """`/openapi.json` 里必须能看到 4 字段响应与 3 个错误响应。

    这条测试守的是"文档与实现同源"：`Field(description=...)` 写的东西
    如果没进 openapi，那 `/docs` 就是过期的，调用方照着写就会错。
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.get("/openapi.json")

    schema = response.json()
    operation = schema["paths"][ENDPOINT]["post"]

    success_ref = operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    assert success_ref.endswith("ExtractResult")
    for status in ("400", "422", "500"):
        error_ref = operation["responses"][status]["content"]["application/json"]["schema"]["$ref"]
        assert error_ref.endswith("ErrorResponse"), status


@pytest.mark.anyio
async def test_service_is_swappable_without_touching_the_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**这是本课最重要的测试**：换掉抽取实现，路由与响应形状完全不变。

    用 `dependency_overrides` 换一个假的 LLM 版抽取器 —— 它不碰正则，直接返回固定值，
    模拟 W2 的模型输出。断言的是"契约还在"：路由没改一行，200 照样成立。
    没有这条，`docs/api-spec.md` 的 §4.7 就只是个愿望。
    """
    from app.core.deps import get_extractor
    from app.services.extractor import ExtractionOutput

    class FakeLlmExtractor:
        model_name = "fake-llm-v2"

        async def extract(self, *, text: str, fields: list[str]) -> ExtractionOutput:
            data = {name: f"LLM::{name}" for name in fields}
            return ExtractionOutput(
                data=data, confidence=0.91, model=self.model_name, elapsed_ms=42
            )

    app.dependency_overrides[get_extractor] = lambda: FakeLlmExtractor()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            response = await ac.post(
                ENDPOINT, json={"text": "任意正文", "fields": ["合同编号", "甲方"]}
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert set(body) == SUCCESS_KEYS
    assert body["data"] == {"合同编号": "LLM::合同编号", "甲方": "LLM::甲方"}
    assert body["confidence"] == 0.91
    assert body["elapsed_ms"] == 42
