"""FastAPI 依赖注入适配层 —— **只有 FastAPI 项目会导入这里**。

## 为什么要单独一个文件

`app/core/llm/client.py` 是**双运行时共用**的：
Streamlit UI（`app/ui/`）和脚本都直接 `get_llm_client(get_settings())`，
而 Streamlit 的部署依赖（`app/ui/requirements.txt`）**刻意不含 fastapi**。

如果 `client.py` 顶层写 `from fastapi import Depends`，那么
Streamlit 一 import 就崩。实测线上就是这个报错：

    File "/mount/src/industrial-ai-platform/app/core/llm/client.py", line 39
        from fastapi import Depends
    ModuleNotFoundError

## 试过但**行不通**的折中方案（都试过，记在这里省得再试）

| 方案 | 为什么不行 |
| --- | --- |
| `Depends` 放在 `TYPE_CHECKING` 里 | `Annotated[x, Depends(f)]` 要**真实对象**，运行时求值 |
| 模块级 `__getattr__` 惰性暴露 `LLMClientDep` | `from app.core.llm import LLMClientDep` 会真的取属性；更要命的是 `dep` 是**每次新建**的对象，`dependency_overrides` 依赖函数对象身份，失败 |
| `globals()` 缓存后再 `dependency_overrides[get_llm_client]` | 覆盖的是 `get_llm_client`，而实际依赖是内层包装函数，**对象不同，覆盖不到** |

结论：**"惰性"和"可被 `dependency_overrides` 覆盖"无法同时满足** ——
后者要求真正被 `Depends()` 包住的那个函数对象在导入时就是固定的、可寻址的。

所以把 fastapi 相关的东西**整体搬到一个只在 FastAPI 场景导入的模块**：
`client.py` 保持纯净（零 fastapi），本文件负责装配。

## 用法（与原来完全一致）

```python
from app.core.llm.deps import LLMClientDep

@router.post("/x")
async def handler(client: LLMClientDep) -> ...:
    ...
```

`dependency_overrides` 也照旧可用，因为这里的 `Depends` 目标就是
模块级的真实函数对象：

    app.dependency_overrides[get_llm_client] = lambda: fake
    app.dependency_overrides[get_settings] = lambda: test_settings
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from openai import AsyncOpenAI

from app.core.config import Settings, get_settings
from app.core.llm.client import get_llm_client


def _dep_get_llm_client(
    settings: Settings = Depends(get_settings),  # noqa: B008 - FastAPI 惯例
) -> AsyncOpenAI:
    """把 `Depends(get_settings)` 解析出的配置交给纯函数 `get_llm_client`。

    ## 为什么包这一层（两个坑都是实测踩到的）

    ### 坑 1：不能直接 `Depends(get_llm_client)`

    `get_llm_client` 的形参是 pydantic 模型 `Settings`。
    FastAPI 看到"单个 pydantic 模型参数"会判定为**请求体**，
    要求调用方 POST 一整个 Settings 上来 —— 显然不是意图。

    ### 坑 2：不能用 `Annotated[Settings, Depends(get_settings)]` 的注解写法

    本模块有 `from __future__ import annotations`，
    **注解是字符串**。FastAPI 解析不到 `Depends`，会退化成
    "必填 query 参数"，运行时报：

        {'loc': ['query', 'settings'], 'msg': 'Field required'}

    所以这里用 `Depends(...)` 的**显式默认值**写法 ——
    默认值在定义时就是真实对象，不依赖注解求值。

    ## 为什么依赖要写成 `Depends(get_settings)` 而不是函数体里调

    这样 `app.dependency_overrides[get_settings]` 能替换整条链，
    测试里注入测试配置不用碰环境变量、不用清 `lru_cache`。
    """
    return get_llm_client(settings)


#: `Annotated[AsyncOpenAI, Depends(...)]` —— 路由签名里直接用它。
#:
#: **必须是模块级常量**：FastAPI 按"函数所在模块的全局命名空间"解析注解，
#: 函数内局部定义的别名它看不见（会退化成 query 参数 -> 422）。
LLMClientDep = Annotated[AsyncOpenAI, Depends(_dep_get_llm_client)]

__all__ = ["LLMClientDep"]
