"""问答模板（`v1.jinja2`）的输出 schema。

## 为什么"不确定"是独立字段而不是空答案

`答案=""` 和 `不确定=True` 在路由层的处理**完全不同**：
- 前者可能是输出被截断、格式坏了 —— 要重试或降级
- 后者是模型明确表态"上下文里没有依据" —— 直接回用户"没找到依据"即可，**不该重试**

把两者混成一个空字符串，上游就永远分不清该不该重试。

## 引用编号为什么是 `list[int]` 而不是 `str`

下游要做的是"按编号取回原文段落"，是**索引**不是文本。
让模型输出 `"第1段、第3段"` 这种自然语言，还得再正则拆一遍，且拆错就是静默错误。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QAAnswer(BaseModel):
    """问答结果。"""

    答案: str = Field(description="基于给定上下文的回答；上下文没有就说明缺什么")
    引用编号: list[int] = Field(
        default_factory=list,
        description="答案依据的上下文段落编号；必须是给定编号的子集，无依据时为空数组",
    )
    不确定: bool = Field(
        default=False,
        description="上下文不足以回答时为 true，此时 答案 应说明缺什么信息",
    )
