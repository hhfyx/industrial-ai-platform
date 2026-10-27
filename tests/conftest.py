"""pytest 全局配置。

## 这个文件是"测试能离线跑绿"的总开关

W2 的灵魂要求是：**拔掉网线，`pytest -q` 必须全绿**。
这件事之所以重要，是因为"能离线测绿" ≈ "实现真的可替换" ——
如果测试依赖真实模型，那它测的就不是**你的代码**，而是**供应商的可用性**。

实测教训（2026-10-27）：网关凭据过期返回 `502 未配置凭据`，
`LlmExtractor` 于是降级到规则版，响应体多出 `degraded` 键 ——
**契约测试 `test_api.py` 直接挂了**，而那个测试本身一行都没改、也没写错。
这说明：任何依赖外部服务的测试都是定时炸弹。

## 做法：用 FastAPI 的 dependency_overrides 注入离线抽取器

**为什么不能用 monkeypatch 换 `deps.get_extractor`**：
路由用的是 `Annotated[Extractor, Depends(get_extractor)]`，FastAPI 在**注册路由时**
就把依赖函数对象存下来了。之后替换模块属性不会影响已注册的依赖 —— 实测过，无效。

正确机制是 `app.dependency_overrides[get_extractor] = ...`，
这是 FastAPI 官方提供的测试注入点，也正是"依赖注入"这个设计存在的理由之一。

## 效果

- 不联网、不需要 API key、不花钱、结果确定
- `test_api.py` **一行不用改**仍然全绿（它测正常路径的 4 字段，离线规则版正好走这条路）
- 测 LLM 路径的测试自己构造 `LlmExtractor(FakeLLMClient(...))`，不经过这个 fixture
- 测降级的测试显式覆盖，并断言响应多出 `degraded: true`
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.core.deps import get_extractor
from app.main import app
from app.services.extractor import RuleExtractor


@pytest.fixture
def anyio_backend() -> str:
    """把 anyio 的异步测试固定到 asyncio 后端。

    为什么不用 `pytest-asyncio`：本项目依赖拉取要过 PyPI，实测不通；
    而 `anyio` 是 starlette 的既有依赖，自带 pytest 插件，能直接跑 `@pytest.mark.anyio`。
    **不引入新依赖**也是这条测试的意义：契约的验证不该依赖额外的第三方包。
    """
    return "asyncio"


@pytest.fixture(autouse=True)
def offline_extractor() -> Iterator[None]:
    """**所有走 FastAPI 的测试默认用离线规则版**。

    `autouse` 是刻意的：让"离线"成为默认，而不是每个测试都要记得声明。
    忘记声明的代价是"CI 突然因为供应商挂了而红" —— 那种失败最浪费时间。
    """
    app.dependency_overrides[get_extractor] = lambda: RuleExtractor()
    yield
    app.dependency_overrides.pop(get_extractor, None)
