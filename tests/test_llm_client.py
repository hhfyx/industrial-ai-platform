"""LLM 接入层测试（W1 周一：配置层 + client 封装）。

**这里一行都不联网**：`_build_client` 只构造对象不发请求，所以"依赖注入拿得到
client"这件事可以纯离线验证。真调用放 `scripts/smoke_llm.py`，它需要真密钥，
不能进 CI。

覆盖的四个断言，正好对应四类"必须不能犯的错"：
1. `Depends(get_llm_client)` 真的能解析出 `AsyncOpenAI` —— 装配没写错
2. 两次解析拿到的**是同一个对象** —— 连接池没被每请求重建
3. `dependency_overrides` 能整条替换 —— 测试可替换性
4. `import` 阶段不解析配置 —— 模块级实例化的反例
"""

from __future__ import annotations

import importlib
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from app.core.config import LLMSettings, Settings, get_settings
from app.core.llm.client import LLMClientDep, _build_client, get_llm_client


@pytest.fixture
def llm_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """构造一份不依赖真实 .env 的配置。

    `_env_file=None` 是必须的：`LLMSettings` 的 model_config 指向项目根的 .env，
    不显式关掉的话，测试结果会随开发者本机的 .env 变化 —— 那种测试是最坏的，
    在别人机器上红、在自己机器上绿。
    """
    monkeypatch.setenv("LLM_API_KEY", "sk-test-not-used-offline")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT", "12.5")
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    return Settings(_env_file=None)


def test_settings_exposes_nested_llm(llm_settings: Settings) -> None:
    """验收点：`Settings().llm.api_key` 能读到，且明文不出现在 repr 里。"""
    llm = llm_settings.llm
    assert llm.api_key.get_secret_value() == "sk-test-not-used-offline"
    assert llm.model == "test-model"
    assert llm.timeout == 12.5
    assert llm.max_retries == 1
    # SecretStr 的核心价值：repr / masked 里没有明文
    assert "sk-test-not-used-offline" not in repr(llm)
    assert llm.api_key_masked == "**********"


def test_base_url_trailing_slash_and_blank_key() -> None:
    """两个解析期规整：末尾斜杠去掉；空 key 报错而不是跑到 401。"""
    settings = LLMSettings(
        _env_file=None,
        api_key="sk-x",
        base_url="https://llm.example.test/v1/",
    )
    assert settings.base_url == "https://llm.example.test/v1"

    with pytest.raises(ValueError, match="LLM_API_KEY 不能为空字符串"):
        LLMSettings(_env_file=None, api_key="   ", base_url="https://x/v1")


def test_get_llm_client_returns_async_openai(llm_settings: Settings) -> None:
    """`get_llm_client(settings)` 直接调用也能拿到 client，且参数被正确注入。"""
    client = get_llm_client(llm_settings)
    assert isinstance(client, AsyncOpenAI)
    # base_url 在 SDK 内部会被规范化为带尾斜杠的 URL 对象
    assert str(client.base_url).rstrip("/") == "https://llm.example.test/v1"
    assert client.timeout == 12.5


def test_depends_resolves_and_is_singleton(llm_settings: Settings) -> None:
    """**E3 的考点**：Depends 拿得到 client，且同一配置下两次拿到同一个实例。

    用真实的 FastAPI 应用走一遍依赖树，而不是直接调函数 ——
    直接调函数验证不了 `Annotated[..., Depends(...)]` 这套装配有没有写对。

    **必须用模块级的 `LLMClientDep`，不能在测试函数里现造一个 `Annotated`**：
    本文件有 `from __future__ import annotations`，所有注解都是字符串；
    FastAPI 解析字符串注解时按**函数所在模块的全局命名空间**求值，
    函数内局部定义的别名它根本看不见 —— 结果是把 `client` 当成 query 参数，
    返回 422 "Field required"。这个坑实测踩到了，所以这里刻意写成模块级引用。
    """
    app = FastAPI()

    seen: list[AsyncOpenAI] = []

    @app.get("/probe")
    def probe(client: LLMClientDep) -> dict[str, str]:
        seen.append(client)
        return {"base_url": str(client.base_url)}

    app.dependency_overrides[get_settings] = lambda: llm_settings

    with TestClient(app) as http:
        first = http.get("/probe")
        second = http.get("/probe")

    assert first.status_code == 200, first.text
    assert first.json()["base_url"].rstrip("/") == "https://llm.example.test/v1"
    assert second.status_code == 200
    # 进程级单例：两次请求必须复用同一个连接池对象
    assert seen[0] is seen[1]


def test_dependency_overrides_can_replace_client(llm_settings: Settings) -> None:
    """整条依赖可替换：这是"不能被模块级实例化锁死"的直接收益。"""
    app = FastAPI()
    fake = AsyncOpenAI(api_key="sk-fake", base_url="https://fake.example/v1")

    @app.get("/probe")
    def probe(client: LLMClientDep) -> dict[str, bool]:
        return {"is_fake": client is fake}

    app.dependency_overrides[get_llm_client] = lambda: fake

    with TestClient(app) as http:
        assert http.get("/probe").json() == {"is_fake": True}


def test_import_does_not_build_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """**反例回归**：import 时不许构造 client，也不许解析配置。

    先把 KEY 清掉再 import —— 如果模块顶层有 `AsyncOpenAI(...)` 或 `Settings()`，
    这一步就会抛 ValidationError / 连接错误。能 import 成功就是设计成立。
    """
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    # 清掉缓存与登记表，模拟"进程刚起来第一次 import"
    _build_client.cache_clear()
    for name in [m for m in sys.modules if m.startswith("app.core.llm")]:
        del sys.modules[name]

    module = importlib.import_module("app.core.llm.client")
    assert hasattr(module, "get_llm_client")
    # 缓存里没有任何 client 实例 = 没有在 import 阶段构造
    assert _build_client.cache_info().currsize == 0


def test_client_does_not_trust_system_proxy(llm_settings: Settings) -> None:
    """**P0 阻断的回归测试**：client 必须显式 `trust_env=False`。

    背景（W6 周日）：本机装了 Clash，系统代理是 `127.0.0.1:7897`。
    openai SDK 底层 httpx 默认 `trust_env=True`，会把**本地回环请求**
    （`http://127.0.0.1:8326`）也扔进代理，网关返回：

        502 {"message": "未配置凭据，请先登录"}

    这个错误信息把排查带偏了整整一周 —— 看起来是账号问题，实际是
    本地回环地址走了系统代理。修法是 `_build_client` 显式传
    `http_client=httpx.AsyncClient(trust_env=False)`。

    这条测试锁住它：如果哪天有人把 `trust_env=False` 去掉，
    `_client._client`（httpx 底层）的 `trust_env` 会变回 True，这里就红。
    """
    _build_client.cache_clear()
    client = _build_client(
        base_url="http://127.0.0.1:8326/v1",
        api_key="sk-x",
        timeout=10.0,
        max_retries=1,
    )
    # AsyncOpenAI 把 httpx 客户端挂在 `_client` 上；直接断言它的 trust_env。
    # 只测"构造时到底有没有注入 trust_env=False"，不发起任何网络请求。
    inner = getattr(client, "_client", None)
    assert inner is not None, "AsyncOpenAI 内部没有 _client（SDK 结构变了？）"
    assert inner.trust_env is False, (
        "client 的 trust_env 必须是 False —— 否则本地回环请求会走系统代理，"
        "复现 W6 周日的「502 未配置凭据」假象"
    )
