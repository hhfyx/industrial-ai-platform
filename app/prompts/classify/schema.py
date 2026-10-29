"""分类模板（`v1.jinja2`）的输出 schema。

## "不能选两个"怎么落到类型上

任务点名要「不能选两个」。这一条**不能只写在 prompt 里**（那是建议，不是约束），
必须在类型层面封死：

`类别: Literal[...]` 是**单值**，不是 `list[str]`。模型想给两个类别时，
只有两条路：① 塞进 `理由`（而 `理由` 不参与业务判定，等于无效）；
② 输出非法 JSON / 数组，被 Pydantic 当场拦下并触发修复轮。

所以"只能选一个"在这里是**结构保证**，不是措辞提醒。

## 类别枚举是本模板唯一的业务耦合点

这 5 个类别是**本项目的业务词**，写死在 `Literal` 里是**必须的**——因为
Pydantic 的 `Literal` 就是用来锁定取值集合的。这与"模板里不硬编码业务词"的纪律不冲突：

- **模板（.jinja2）里不写业务词**：类别列表通过 `categories` 参数传入，W4 换一套类别不用改模板；
- **schema.py 里的 Literal 是本项目的契约**：换业务就换一个 schema 文件（v2），
  这正是版本目录存在的意义。

一句话：**模板参数化，schema 显式化。**
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClassifyLabel(BaseModel):
    """分类结果。"""

    类别: Literal["故障报修", "功能咨询", "费用争议", "投诉建议", "其他"] = Field(
        description="只能选一个最贴近的类别，不允许同时给多个"
    )
    置信度: float = Field(ge=0.0, le=1.0, description="对本次分类的把握，0-1；低于 0.5 建议转人工")
    理由: str = Field(description="一句话说明为什么归到这个类别，不得超过 30 字")
