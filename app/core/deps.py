"""依赖注入：把"用哪个抽取器"变成一行装配代码。

**这个文件是 W2 换实现的唯一开关。**
路由只声明 `ExtractorDep`，不知道底下是正则还是 LLM；换实现时只改这里的一行，
路由、schema、契约文档、测试全都不用动。

`get_extractor` 为什么是 `lru_cache` 的单例：抽取器的构造（W2 时是模型 client）
很贵，绝不能每个请求造一个。

**W2 已切换（2026-10-26）**：`get_extractor` 默认返回 `LlmExtractor`。
证据是 `tests/test_api.py` **一行没改**仍然全绿 —— 那才是"契约冻住"的可执行证明，
而不是文档里的一句承诺。

`EXTRACTOR_IMPL=rule` 保留一键回退：线上把 LLM 打挂了，运维改一个环境变量重启即可，
不用紧急发版。**应急开关必须是改配置，不是改代码** —— 改代码要过评审、要构建、
要发版，那段时间线上一直在 500。
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated, Protocol, runtime_checkable

from fastapi import Depends, Request

from app.core.config import get_settings
from app.core.errors import REQUEST_ID_ATTR
from app.core.llm.client import get_llm_client
from app.services.extractor import ExtractionOutput, RuleExtractor
from app.services.llm_extractor import LlmExtractor

# 实现选择开关：默认 llm；设成 rule 退回规则版。
EXTRACTOR_IMPL_ENV = "EXTRACTOR_IMPL"

# 本机网关默认开推理模式，思维链会吃光 max_tokens 导致 content=null
# （实测见 learn/llm/01_params.md §3.1）。设成 0 可关掉这个补偿。
DISABLE_REASONING_ENV = "LLM_DISABLE_REASONING"


@runtime_checkable
class Extractor(Protocol):
    """抽取器的**唯一接口**（api-spec.md §4.7）。

    W1 时这里还没写 Protocol，注释里写的是"W2 有了第二个实现再抽"。
    现在两个实现都在了 —— 那正是抽接口的时机，早一天都是为不存在的需求付设计成本。

    `runtime_checkable` 让 `isinstance(extractor, Extractor)` 能验证装配结果，
    测试里不用 `hasattr` 手搓。
    """

    model_name: str

    async def extract(self, *, text: str, fields: list[str]) -> ExtractionOutput: ...


@lru_cache(maxsize=1)
def _build_llm_extractor() -> LlmExtractor:
    """构造 LLM 抽取器（进程级单例）。

    `get_llm_client(get_settings())` 直接调用而不是 `Depends(...)`：
    `lru_cache` 装饰的函数**不是** FastAPI 依赖，参数不会被依赖系统解析。
    但因为 `get_llm_client` 自己就是转发到 `_build_client`（lru_cache 单例），
    这里拿到的仍是**同一个 client** —— 连接池没有被复制。
    这一点很关键：复制连接池会让并发上限凭空翻倍，也浪费 fd。
    """
    settings = get_settings()
    extra: dict[str, object] = {}
    if os.environ.get(DISABLE_REASONING_ENV, "1") == "1":
        extra["reasoning_effort"] = "none"
    return LlmExtractor(
        get_llm_client(settings),
        model=settings.llm.model,
        extra_request=extra,
    )


@lru_cache(maxsize=1)
def get_extractor() -> Extractor:
    """当前生效的抽取器（进程级单例）。

    返回类型标注成 `Protocol` 而不是具体类：路由只关心 `extract()`，
    这是"接口不变、实现可换"在类型层面的表达。
    """
    impl = os.environ.get(EXTRACTOR_IMPL_ENV, "llm").strip().lower()
    if impl == "rule":
        return RuleExtractor()
    return _build_llm_extractor()


def reset_extractor_cache() -> None:
    """清掉单例缓存。测试里切换实现后必须调用，否则拿到的是旧实例。"""
    get_extractor.cache_clear()
    _build_llm_extractor.cache_clear()


ExtractorDep = Annotated[Extractor, Depends(get_extractor)]


def get_request_id(request: Request) -> str:
    """读中间件写进 `request.state` 的 id。

    **不要在这里生成**：生成会跟日志里的 id 对不上（中间件那次才是被记录的）。
    中间件一定先跑，所以这里直接取；取不到说明中间件被摘了，属于装配错误。
    """
    return str(getattr(request.state, REQUEST_ID_ATTR, ""))


RequestIdDep = Annotated[str, Depends(get_request_id)]


__all__ = [
    "DISABLE_REASONING_ENV",
    "EXTRACTOR_IMPL_ENV",
    "Extractor",
    "ExtractorDep",
    "RequestIdDep",
    "get_extractor",
    "get_request_id",
    "reset_extractor_cache",
]
