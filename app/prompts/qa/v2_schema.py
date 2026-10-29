"""问答模板 v2 的输出 schema。

## 相对 v1（`QAAnswer`）的两处结构改动

| | v1 | v2 |
| --- | --- | --- |
| 能否回答 | `不确定: bool`（反向语义） | `能否回答: "能"/"不能"`（正向，更直观） |
| 缺什么 | 混在 `答案` 的正文里 | `缺少信息: list[str]`（结构化） |

## 为什么要独立的 `缺少信息` 字段

v1 的问题：模型说"无法确定"时，具体缺什么只能从 `答案` 的**自然语言**里读，
下游无法程序化处理（比如"缺日期"要不要触发追问、"缺政策"要不要转人工，判不出来）。

v2 把它结构化成列表，路由层可以直接 `if "付款日期" in result.缺少信息: 追问用户`。

## `引用编号` 保留为 list[int]

不变 —— 索引比自然语言文本更适合下游"按编号取回原文段落"。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QAAnswerV2(BaseModel):
    """v2 问答结果：能答/不能答 + 结构化缺失信息。"""

    能否回答: Literal["能", "不能"] = Field(description="上下文信息是否足以回答问题")
    答案: str = Field(description="基于给定上下文的回答；不能回答时说明缺什么")
    引用编号: list[int] = Field(
        default_factory=list,
        description="答案依据的段落编号，必须是给定编号的子集；有相关段落时应非空",
    )
    缺少信息: list[str] = Field(
        default_factory=list,
        description="要回答问题还缺哪些信息，每项一句话；能回答时为空数组",
    )
