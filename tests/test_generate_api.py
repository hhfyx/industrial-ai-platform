"""`POST /api/v1/generate` 的契约测试（项目一 · W8 周三）。

**这一批测试就是 api-spec.md 项目一段的可执行形态。**
FakeGenerator 注入（W2 的离线测试纪律）：不碰 GPU、不碰网络，
四类失败路径用"按剧本抛错的假实现"逐条复现 ——
断网 pytest 全绿的验收本体在这里。

真实现（SdLoraGenerator）不在本文件测：它现在抛
GeneratorUnavailableError 是**装配态的正常行为**（有自己的测试），
推理内核的验证属于真机部署，不属于契约测试。
"""

from __future__ import annotations

import httpx
import pytest

from app.core.deps import get_generator, reset_generator_cache
from app.main import app
from app.services.generator import (
    GenerationOOMError,
    GenerationOutput,
    GenerationTimeoutError,
    Generator,
    GeneratorUnavailableError,
)

ENDPOINT = "/api/v1/generate"

# W1 契约 §3：所有非 2xx 都是这 4 个键，一个不多一个不少
ERROR_KEYS = {"code", "message", "request_id", "details"}


@pytest.fixture
async def client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.fixture(autouse=True)
def _default_fake_generator():
    """每个用例重置装配缓存并保证默认拿到 FakeGenerator（离线纪律）。

    环境变量 GENERATOR_IMPL 若被外层污染，这里显式拨回 fake。
    """
    import os

    old = os.environ.pop("GENERATOR_IMPL", None)
    reset_generator_cache()
    yield
    if old is not None:
        os.environ["GENERATOR_IMPL"] = old
    reset_generator_cache()


class _ScriptedGenerator:
    """按剧本抛错/返回的假实现 —— 四类失败的复现装置。"""

    def __init__(self, *, exc: Exception | None = None, images: int = 1) -> None:
        self.model_name = "scripted-generator"
        self._exc = exc
        self._images = images
        self.calls: list[dict] = []

    async def generate(self, **kwargs) -> GenerationOutput:  # noqa: ANN003, ANN201
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return GenerationOutput(
            images=[
                {
                    "path": f"data/generated/x_{i}.png",
                    "size": kwargs.get("size", 512),
                    "seed": 42 + i,
                }
                for i in range(self._images)
            ],
            model=self.model_name,
            elapsed_ms=3,
        )


def _inject(gen: Generator) -> None:
    app.dependency_overrides[get_generator] = lambda: gen


def _clear_inject() -> None:
    app.dependency_overrides.pop(get_generator, None)


# ============================================================ 200 路径


@pytest.mark.anyio
async def test_generate_returns_contracted_shape(client: httpx.AsyncClient) -> None:
    """成功响应：request_id + images[count] + elapsed_ms + model。"""
    gen = _ScriptedGenerator(images=2)
    _inject(gen)
    try:
        response = await client.post(
            ENDPOINT,
            json={"defect_class": "划痕", "count": 2, "size": 512, "seed": 7},
        )
    finally:
        _clear_inject()

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"]
    assert body["model"] == "scripted-generator"
    assert body["elapsed_ms"] >= 0
    assert len(body["images"]) == 2
    assert body["images"][0]["path"].endswith(".png")
    assert body["images"][0]["seed"] in (42, 43)
    # 参数透传：服务层收到的是校验后的值
    assert gen.calls[0] == {"defect_class": "划痕", "count": 2, "size": 512, "seed": 7}


@pytest.mark.anyio
async def test_request_id_echo_and_header_consistency(client: httpx.AsyncClient) -> None:
    """request_id 与 X-Request-ID 同源；客户端传了就沿用（W1 中间件纪律）。"""
    gen = _ScriptedGenerator()
    _inject(gen)
    try:
        response = await client.post(
            ENDPOINT,
            json={"defect_class": "划痕"},
            headers={"X-Request-ID": "trace-gen-1"},
        )
    finally:
        _clear_inject()

    assert response.headers["x-request-id"] == "trace-gen-1"
    assert response.json()["request_id"] == "trace-gen-1"
    assert response.headers["x-generator-model"] == "scripted-generator"


# ============================================================ 422 参数上限


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"defect_class": "划痕", "count": 9}, "count"),  # 超上限
        ({"defect_class": "划痕", "count": 0}, "count"),
        ({"defect_class": "划痕", "size": 600}, "size"),  # 非法档位
        ({"defect_class": "划痕", "size": 1024}, "size"),
        ({"defect_class": "划痕", "seed": -1}, "seed"),
        ({"defect_class": ""}, "defect_class"),  # 空类别
    ],
)
async def test_param_limits_are_422_with_error_contract(
    client: httpx.AsyncClient, payload: dict, field: str
) -> None:
    """没有上限的生成接口等于自杀 —— 上限在契约层拦截，且错误体是 4 字段。"""
    response = await client.post(ENDPOINT, json=payload)

    assert response.status_code == 422
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["code"] == "INVALID_INPUT"
    assert any(d["field"] == field for d in body["details"]), body["details"]


@pytest.mark.anyio
async def test_unknown_top_level_key_rejected(client: httpx.AsyncClient) -> None:
    """多传一个键（negative_prompt）→ 422，不被静默忽略。"""
    response = await client.post(
        ENDPOINT,
        json={"defect_class": "划痕", "negative_prompt": "blur"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_INPUT"


@pytest.mark.anyio
async def test_unknown_defect_class_422_with_dedicated_code(client: httpx.AsyncClient) -> None:
    """表外类别 → 422 UNKNOWN_DEFECT_CLASS：宁可报错，不静默用基座模型生成。"""
    response = await client.post(ENDPOINT, json={"defect_class": "异物"})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "UNKNOWN_DEFECT_CLASS"
    assert body["details"][0]["field"] == "defect_class"


# ============================================================ 四类失败（服务层异常映射）


@pytest.mark.anyio
async def test_weights_not_loaded_maps_to_503(client: httpx.AsyncClient) -> None:
    """权重未加载 → 503 MODEL_NOT_LOADED。"""
    _inject(_ScriptedGenerator(exc=GeneratorUnavailableError("权重未装配")))
    try:
        response = await client.post(ENDPOINT, json={"defect_class": "划痕"})
    finally:
        _clear_inject()

    assert response.status_code == 503
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["code"] == "MODEL_NOT_LOADED"


@pytest.mark.anyio
async def test_oom_maps_to_503_with_retry_after(client: httpx.AsyncClient) -> None:
    """OOM → 503 GPU_OOM，带 Retry-After（"等一等能过"的错要有退避提示）。"""
    _inject(_ScriptedGenerator(exc=GenerationOOMError("CUDA out of memory")))
    try:
        response = await client.post(ENDPOINT, json={"defect_class": "划痕", "count": 8})
    finally:
        _clear_inject()

    assert response.status_code == 503
    assert response.json()["code"] == "GPU_OOM"
    assert "retry-after" in {k.lower() for k in response.headers}


@pytest.mark.anyio
async def test_timeout_maps_to_504(client: httpx.AsyncClient) -> None:
    """超时 → 504 GENERATION_TIMEOUT。"""
    _inject(_ScriptedGenerator(exc=GenerationTimeoutError("180s 超时")))
    try:
        response = await client.post(ENDPOINT, json={"defect_class": "划痕"})
    finally:
        _clear_inject()

    assert response.status_code == 504
    assert response.json()["code"] == "GENERATION_TIMEOUT"


@pytest.mark.anyio
async def test_unexpected_error_maps_to_500_internal() -> None:
    """未预期异常 → 全局 handler 兜底 500 INTERNAL_ERROR（形状不变）。

    未捕获异常在 ServerErrorMiddleware 层处理，**必须用
    `raise_app_exceptions=False`** 的 transport 让异常变成响应
    （与 test_error_handlers.py 的 boom_client 同款）——
    默认 transport 会把异常原样抛给测试进程。
    """
    _inject(_ScriptedGenerator(exc=RuntimeError("内部炸了")))
    try:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            response = await ac.post(ENDPOINT, json={"defect_class": "划痕"})
    finally:
        _clear_inject()

    assert response.status_code == 500
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["code"] == "INTERNAL_ERROR"


# ============================================================ 默认装配（离线纪律）


@pytest.mark.anyio
async def test_default_assembly_is_fake_and_offline(client: httpx.AsyncClient, tmp_path) -> None:
    """默认装配 = FakeGenerator：真实写盘、秒回、不碰 GPU —— 断网全绿的本体。"""
    response = await client.post(ENDPOINT, json={"defect_class": "锈蚀", "count": 1, "size": 768})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "fake-generator"
    assert body["images"][0]["size"] == 768
    # FakeGenerator 真实写盘（项目内 data/，不是系统 Temp）
    from pathlib import Path

    assert Path(body["images"][0]["path"]).exists()


@pytest.mark.anyio
async def test_fake_generator_is_valid_generator_protocol() -> None:
    """装配结果 isinstance Generator（runtime_checkable 协议的用法）。"""
    from app.services.generator import FakeGenerator

    assert isinstance(FakeGenerator(), Generator)
