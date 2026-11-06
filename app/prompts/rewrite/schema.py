"""改写模板（`v1.jinja2`）的输出 schema。

## 指代消解为什么是"改写"而不是"补全"

用户说「它的公差是多少」，历史里"它"指某个型号。检索层要的是一条
**能独立拿去向量化的 query** —— 不是"把代词填回去的原文"。
所以输出字段叫 `standalone_question`：**脱离对话历史单看这一句，
仍然能检索到正确内容**。这是 W8 对照实验的判分口径，也是改写层
唯一的产出契约。

## `needs_clarification` 与 `missing` 的分工

| 字段 | 语义 | 下游行为 |
| --- | --- | --- |
| `needs_clarification=true` | 指代歧义到**无法确定改写方向**（"上面那个型号"前面出现过两个型号） | UI 反问用户；CLI/eval 用原问 + 记标记 |
| `missing=[]` 且 `needs_clarification=false` | 改写成功 | 用 `standalone_question` 检索 |

`missing` 只在 `needs_clarification=true` 时有内容（缺什么信息才能消解歧义）。
两者都为"正常"时 `missing` 必须是空数组 —— 模型容易在改写成功时也顺手
写"缺上下文"，校验层要拦这种（见 `query_rewrite.py`）。

## `rolling_summary` 为什么是常驻字段而不是条件字段

计划原文是"历史 > 3 轮时 schema 加 rolling_summary 字段"。实现上**没有**
做成条件 schema，理由有三：

1. loader 的 `load_schema()` 按"一个模块恰好一个 BaseModel"发现 schema
   （loader.py:359-375），条件 schema 意味着两个模型类，破坏这个约定。
2. `structured_call(prompt, schema)` 的 schema 参数是运行时传的，
   条件切换 = 调用方按轮数选类 —— 三条离线边界测试就要写两套断言。
3. `default=""` 的常驻字段在"历史 ≤ 3 轮"时让模型输出空串即可，
   token 成本可忽略（一个字段名 + 一个空串）。

**语义**：历史 ≤ 3 轮时输出 `""`；超过时输出一段把更早对话压进两三句话的
摘要，供下一轮裁剪策略使用（"最近 3 轮原文 + 更早的摘要"）。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RewriteResult(BaseModel):
    """改写层的输出。"""

    standalone_question: str = Field(
        description="改写后的独立问题：不含代词，脱离对话历史单看仍能检索到正确内容；"
        "若 needs_clarification 为 true，此字段填原问"
    )
    needs_clarification: bool = Field(
        default=False,
        description="指代歧义到无法确定改写方向时为 true，此时应向用户澄清而不是硬猜",
    )
    missing: list[str] = Field(
        default_factory=list,
        description="needs_clarification 为 true 时列出还需要用户补充什么；否则为空数组",
    )
    rolling_summary: str = Field(
        default="",
        description="对话历史的滚动摘要；历史不超过 3 轮时输出空字符串，"
        "超过时用两三句话概括更早轮次的主题与已确认的关键信息",
    )
