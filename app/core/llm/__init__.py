"""LLM 接入层。

只放"怎么连到模型"这一件事，不含任何业务语义：

- `client.py`  `get_llm_client(settings)`：进程级 AsyncOpenAI 单例（**零 fastapi**）
- `deps.py`    `LLMClientDep`：FastAPI 依赖注入用的类型别名（**只有 FastAPI 场景导入**）

## ⚠ 这个包**不能**在 import 时要求 fastapi 存在

`app/ui/streamlit_app.py` 会 `from app.core.llm.client import get_llm_client`。
Streamlit 的部署依赖（`app/ui/requirements.txt`）**刻意不含 fastapi** ——
UI 用不到 Web 框架，装上只是白占安装时间和磁盘。

实测踩到：线上直接起不来

    File "/mount/src/.../app/core/llm/client.py", line 39
        from fastapi import Depends
    ModuleNotFoundError: No module named 'fastapi'

修法不是"把 fastapi 加进部署依赖"，而是**把 fastapi 隔离到 `deps.py`**：
`client.py` 保持纯净，需要 `Depends` 的人显式 `from app.core.llm.deps import ...`。

**注意这里不要再 `from app.core.llm.deps import LLMClientDep`** ——
那会让"import 本包"重新依赖 fastapi，等于把坑挖回去。
"""

from __future__ import annotations

from app.core.llm.client import aclose_llm_client, get_llm_client, is_loopback_base

__all__ = [
    "aclose_llm_client",
    "get_llm_client",
    "is_loopback_base",
]
