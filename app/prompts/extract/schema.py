"""抽取模板（`v1.jinja2`）的输出 schema。

## "缺失填 null，不许编"怎么落到类型上

所有业务字段的类型都是 `str | None`，**不是 `str`**。这一条是刻意的：

- 若写 `str`，Pydantic 会把缺失当成"必填未提供"而报错，模型被迫编一个值来通过校验
  —— 正是我们要禁止的行为；
- 写 `str | None`，`None` 是**合法值**，模型没有"编一个"的压力。

同时 `缺失字段` 让上层能区分两种 `None`：

| 情况 | `主体` | `缺失字段` | 含义 |
| --- | --- | --- | --- |
| 原文确实没写主体 | `None` | `["主体"]` | 抽取成功，数据本来就没有 |
| 抽取本身失败 | `None` | `["主体"]` | 上一种情况的混淆项，靠 `置信度` 区分 |

## 字段名为什么不写死在这里

`主体` / `时间` 是本项目的示例字段。真实业务里字段会变（合同、工单、病历各不相同），
所以**模板通过 `fields` 参数接收字段清单**，schema 提供对应数量的槽位。

W4 换文档类型时，正确做法是新增 `v2.jinja2` + 新 schema 文件，
而不是改 `v1` —— 这正是版本目录存在的理由。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractResult(BaseModel):
    """抽取结果。"""

    主体: str | None = Field(default=None, description="原文中出现的对象主体；没有则 null")
    时间: str | None = Field(default=None, description="原文中出现的时间，YYYY-MM-DD；没有则 null")
    缺失字段: list[str] = Field(
        default_factory=list,
        description="原文中找不到、因此填了 null 的字段名列表，便于上层区分「确实没有」和「抽取失败」",
    )
    置信度: float = Field(ge=0.0, le=1.0, description="对整体抽取完整度的把握，0-1")
