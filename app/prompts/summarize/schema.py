"""摘要模板（`v1.jinja2`）的输出 schema。

## 字数为什么让模型自报

`字数` 是模型自己数的，**不能当唯一依据**（模型数错很常见）。
它的作用是**双重用途**：
1. 让模型在生成时"意识到"长度约束（写字段名的动作本身就是提醒）；
2. 与本地 `len(摘要)` 交叉校验 —— 两者差太多说明模型在执行约束上不稳定，
   这个信号比"字数超了"本身更有价值。

真正的硬约束仍然在本地：`loader` 不管长度，调用方用 `len(result.摘要)` 判。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SummaryResult(BaseModel):
    """摘要结果。"""

    摘要: str = Field(description="保留数字与专有名词的长度受控摘要")
    字数: int = Field(ge=0, description="摘要的实际字数，由模型自报，用于与本地计数交叉校验")
    覆盖要点: list[str] = Field(
        default_factory=list,
        description="摘要覆盖的关键点，每条不超过 15 字",
    )
