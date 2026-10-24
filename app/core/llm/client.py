"""LLM 接入层：进程级 client 工厂 + FastAPI 依赖。

**这个文件的核心考点只有一个：为什么不能模块级实例化。**

反面写法（下面这行一旦出现在模块顶层，就是错的）：

    client = AsyncOpenAI(api_key=...)          # ❌ 禁止

它错在三个地方，且都不是"风格问题"：

1. **import 即副作用**。`import app.main` 会连带 import 到这里，
   于是"跑一个单元测试"或"生成一次 openapi.json"都要先有合法密钥和网络可达。
   测试里想换一个假 client，就得去改环境变量、驱逐模块缓存 —— 绕不过去。

2. **密钥在 import 时被固化**。配置优先级是"环境变量 > .env > 默认值"，
   而 import 发生在进程最早期；之后 `monkeypatch.setenv` 改的任何东西都不会生效，
   因为 client 早就把旧 key 焊死了。12-factor 的配置注入就此失效。

3. **连接池无法关闭**。`AsyncOpenAI` 内部持有一个 httpx 连接池。
   模块级实例没有明确的生命周期，进程退出时靠 GC 收尾会打
   "Unclosed client session" 一类告警；在有 lifespan 的应用里更是没人负责 `aclose()`。

正面写法：**工厂 + 依赖注入**。`get_llm_client()` 是个普通函数，
由 FastAPI 在**每次请求解析依赖时**调用；真正的构造被 `_build_client` 的
`lru_cache` 收成进程级单例 —— 既满足"只建一个连接池"，又保证"第一次需要它时才建"。

缓存键为什么是四个原始值而不是 `Settings` 对象：
pydantic 的 `BaseModel` 默认不可哈希，拿 `settings.llm` 当 lru_cache 的键会
`TypeError: unhashable type`。用原始值做键还顺带带来一个好处 ——
测试里 `monkeypatch.setenv` 改了 base_url 之后缓存键就变了，
会自然建出**新的** client，不用手动 `cache_clear()` 就能验证配置真的生效。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from openai import AsyncOpenAI

from app.core.config import Settings, get_settings

# 已创建的 client 登记表：`aclose_llm_client()` 要拿到实例才能真正关连接池。
# lru_cache 不提供"列出所有缓存值"的接口，所以必须自己登记。
_OPEN_CLIENTS: set[AsyncOpenAI] = set()


@lru_cache(maxsize=8)
def _build_client(
    *,
    base_url: str,
    api_key: str,
    timeout: float,
    max_retries: int,
) -> AsyncOpenAI:
    """真正构造 client 的地方，只被 `get_llm_client` 调用。

    参数全用关键字、且都是原始值：既是 lru_cache 的哈希键，
    也让调用点不可能按位置传错。超时/重试一律从这里注入，
    **不在调用点临时覆盖** —— 否则同一进程里不同代码路径的超时行为不一致，
    排查问题时会非常难。

    注意 `timeout` 只约束**单次 HTTP 往返**；`max_retries` 次重试的
    最坏总耗时约是它的 `1 + max_retries` 倍。要约束"整个调用"必须在业务层
    套 `asyncio.timeout(...)`，这一点 W2 的流式接口会踩到。
    """
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    _OPEN_CLIENTS.add(client)
    return client


def get_llm_client(settings: Annotated[Settings, Depends(get_settings)]) -> AsyncOpenAI:
    """FastAPI 依赖：拿到本进程共用的 AsyncOpenAI 实例。

    路由里的用法：

        LlmClientDep = Annotated[AsyncOpenAI, Depends(get_llm_client)]

        @router.post("/x")
        async def handler(client: LlmClientDep) -> ...:
            ...

    参数写成 `Depends(get_settings)` 而不是函数体里直接调 `get_settings()`：
    这样测试里可以用 `app.dependency_overrides[get_settings]` 注入测试配置，
    整条依赖链（settings → client）一起被替换，不用碰环境变量也不用清缓存。
    """
    config = settings.llm
    return _build_client(
        base_url=config.base_url,
        api_key=config.api_key.get_secret_value(),  # 明文只在这一处出现
        timeout=config.timeout,
        max_retries=config.max_retries,
    )


async def aclose_llm_client() -> None:
    """关闭所有连接池并清空缓存。挂到 lifespan 的 shutdown 阶段。

    两件事都要做，缺一不可：
    - 逐个 `close()`：真正释放 httpx 连接池里的 socket；
    - `cache_clear()`：否则下次 `get_llm_client()` 会拿到刚被关掉的实例，
      首次使用时报 "Cannot send a request, as the client has been closed".

    顺序也不能反：先清缓存再 close 依然拿得到旧实例，但如果有并发请求
    在这中间拿到新 client，清缓存就会把新实例也丢掉并重复建池。
    先 close 再 clear，窗口期只有"清缓存"这一瞬间，且被 clear 的实例已经关闭，
    下次调用会老老实实重建。
    """
    for client in _OPEN_CLIENTS:
        await client.close()
    _OPEN_CLIENTS.clear()
    _build_client.cache_clear()


LLMClientDep = Annotated[AsyncOpenAI, Depends(get_llm_client)]


__all__ = [
    "LLMClientDep",
    "aclose_llm_client",
    "get_llm_client",
]
