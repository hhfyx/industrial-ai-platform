"""抽取模板 v2 的输出 schema：支持任意字段清单。

## 为什么用 `dict[str, str | None]` 而不是动态 Pydantic 模型

可以用 `create_model` 按调用方给的字段名生成一个真·模型（每个字段一个属性），
但那样有两个问题：

1. **schema 就不再是"唯一真源"** —— 它变成运行时拼出来的，`load_schema` 拿不到它，
   缓存键也没法稳定计算（同一个逻辑 schema 每次生成的对象不同）。
2. **调试变难** —— 报错里的字段路径来自动态模型，人读起来比字典直观。

用 `dict[str, str | None]` 换来的是：
- 结构固定、可缓存、报错路径明确（`fields.版本`）
- 值类型仍然被约束（`str | None`，嵌套对象会被 Pydantic 拦下）
- 代价是**键名漂移**不受 schema 保护（多一个或少一个键都合法），
  所以模板里明写"键必须逐字一致"，调用方按原键取、取不到记为缺失。

## 与 v1 的区别

| | v1 (`schema.py`) | v2（本文件） |
| --- | --- | --- |
| 字段槽位 | 写死 `主体`/`时间` | 任意，运行时传入 |
| 适用 | 工单场景 | 通用 |
| 键名漂移风险 | 无（槽位固定） | 有（靠提示约束） |

v1 保持不动 —— 历史实验都是在 v1 口径下测的，改它等于让那些数据无法复现。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractV2(BaseModel):
    """v2 抽取结果：字段值装在 `fields` 字典里。"""

    fields: dict[str, str | None] = Field(
        default_factory=dict,
        description="字段名 → 抽取到的值；原文没有的字段值为 null，键必须与调用方给的清单一致",
    )
    缺失字段: list[str] = Field(
        default_factory=list,
        description="原文中找不到、因此填了 null 的字段名列表",
    )
    置信度: float = Field(ge=0.0, le=1.0, description="对整体抽取完整度的把握，0-1")
