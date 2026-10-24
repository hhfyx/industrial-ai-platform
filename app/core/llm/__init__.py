"""LLM 接入层。

只放"怎么连到模型"这一件事，不含任何业务语义：

- `client.py`  `get_llm_client()`：进程级 AsyncOpenAI 单例，供 `Depends` 注入

W2 的 `LlmExtractor` 会 import 这里，但**不 import fastapi** ——
服务层不该知道依赖注入的存在，依赖注入是路由层的事。
"""

from __future__ import annotations

from app.core.llm.client import LLMClientDep, aclose_llm_client, get_llm_client

__all__ = ["LLMClientDep", "aclose_llm_client", "get_llm_client"]
