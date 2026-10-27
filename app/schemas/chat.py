"""对外的聊天接口契约（`POST /api/v1/chat/stream`）。

**为什么单独一个文件、而不是塞进 `schemas/extract.py`**：
抽取和聊天是两条独立的接口线。`extract` 的 4 字段响应是**冻结的 v1 契约**，
聊天是新增能力 —— 混在一个文件里，改聊天就可能碰到冻结的表单，
那种"改 A 坏 B"正是契约管理要避免的。

和 `extract` 的取舍不同：这里**允许**较宽的自由度（多轮消息、可选参数），
因为它没有"必须字节级不变"的历史包袱。
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

MAX_MESSAGE_LENGTH = 8000
MAX_MESSAGE_COUNT = 40
MAX_TOKENS_LIMIT = 4096

# 入参严格：多传一个键就报错，拼错的字段名不会被静默忽略。
# 出参也是流，没有 JSON 出参模型（SSE 无法用 JSON Schema 表达）。
STRICT_CONFIG = ConfigDict(extra="forbid", str_strip_whitespace=True)

Role = Literal["user", "assistant", "system"]

ContentText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_LENGTH),
]


class ChatMessage(BaseModel):
    """一条对话消息。

    **只收 `user` / `assistant` / `system` 三种 role**，刻意不收 `tool`：
    工具调用需要 `tool_call_id` 配对、需要注册表执行，那是 `tool_loop` 的职责，
    不该通过一个聊天接口让调用方注入任意 tool 消息 —— 那等于把执行边界交给客户端。
    """

    model_config = STRICT_CONFIG

    role: Role = Field(description="消息角色：user / assistant / system")
    content: ContentText = Field(description=f"消息正文，去空白后 1–{MAX_MESSAGE_LENGTH} 字符")


class ChatStreamRequest(BaseModel):
    """聊天流式请求体。

    刻意保持窄：**没有** `temperature` / `max_tokens` / `model` 这类实现参数。
    理由与 `extract` 一致 —— 调用方不该为了换模型而改请求。
    需要按请求调参的场景，应该走另一个明确标注"实验性"的端点，
    而不是往这个稳定的契约里加字段。
    """

    model_config = STRICT_CONFIG

    messages: list[ChatMessage] = Field(
        min_length=1,
        max_length=MAX_MESSAGE_COUNT,
        description=f"对话历史，1–{MAX_MESSAGE_COUNT} 条；最后一条应为 user",
    )


__all__ = [
    "MAX_MESSAGE_COUNT",
    "MAX_MESSAGE_LENGTH",
    "MAX_TOKENS_LIMIT",
    "ChatMessage",
    "ChatStreamRequest",
    "Role",
]
