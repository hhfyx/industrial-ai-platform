"""四类模板库。

目录约定（见 `app/prompts/loader.py` 的 docstring）：

    app/prompts/{qa,summarize,classify,extract}/v1.jinja2   模板
    app/prompts/{qa,summarize,classify,extract}/schema.py   该类模板的输出 Pydantic 模型
    app/prompts/loader.py                                   load_prompt(name, version)

本 `__init__.py` 只做**转发**，不在此处定义模型 ——
模型定义必须留在各自的 `schema.py`，否则"每类一个 schema.py"的规范就形同虚设。
"""

from __future__ import annotations

from app.prompts.loader import (
    PromptNotFoundError,
    clear_cache,
    list_versions,
    load_prompt,
    load_schema,
)

__all__ = [
    "PromptNotFoundError",
    "clear_cache",
    "list_versions",
    "load_prompt",
    "load_schema",
]
