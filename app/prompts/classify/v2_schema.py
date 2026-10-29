"""分类模板 v2 的输出 schema。

## 相对 v1（`ClassifyLabel`）的改动

| | v1 | v2 |
| --- | --- | --- |
| 可解释性 | `理由`（为什么是这个） | 加 `排除类别`（为什么不是别的） |
| 类别集合 | 写死 5 个 Literal | 保持不变（类别体系是业务契约，不随模板版本变） |

## 为什么 `类别` 的 Literal 不变

类别集合是本项目的**业务契约**，不是模板的写法问题。v2 改的是"怎么判"，
不是"能判成什么"。改了 Literal 就等于改了业务口径，历史实验数据全部失效。

如果真要换类别体系，正确做法是新增 `v3` + 新的 schema 文件，
并**重新标注 gold** —— 不能复用旧的 gold。

## `排除类别` 为什么是 `list[str]` 而不是 Literal

它记录的是"模型排除掉的候选"，本质上是一个**诊断字段**：
- 允许为空（模型可能直接定位到答案）
- 不参与业务判定
- 允许它写错（只影响可解释性，不影响正确性）

用 Literal 会给它加上不该有的约束 —— 一旦模型写了个不在集合里的词，
整条结果就被 Pydantic 拒绝、触发修复轮，白白多花一次调用。
诊断字段就该宽松。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClassifyV2(BaseModel):
    """v2 分类结果：唯一类别 + 排除理由。"""

    类别: Literal["故障报修", "功能咨询", "费用争议", "投诉建议", "其他"] = Field(
        description="只能选一个最贴近的类别，不允许同时给多个"
    )
    置信度: float = Field(ge=0.0, le=1.0, description="对本次分类的把握，0-1；低于 0.5 建议转人工")
    理由: str = Field(description="一句话说明为什么归到这个类别，不得超过 30 字")
    排除类别: list[str] = Field(
        default_factory=list,
        description="明确排除掉的候选类别名；用于诊断边界混淆，不参与业务判定",
    )
