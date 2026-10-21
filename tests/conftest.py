from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """把 anyio 的异步测试固定到 asyncio 后端。

    为什么不用 `pytest-asyncio`：本项目依赖拉取要过 PyPI，实测不通；
    而 `anyio` 是 starlette 的既有依赖，自带 pytest 插件，能直接跑 `@pytest.mark.anyio`。
    **不引入新依赖**也是这条测试的意义：契约的验证不该依赖额外的第三方包。
    """
    return "asyncio"
