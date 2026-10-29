"""摘要模板 v2 的输出 schema。

## 相对 v1（`SummaryResult`）的两处改动

| | v1 | v2 |
| --- | --- | --- |
| 中间结果 | 不可见 | `分段要点: list[str]`（先提炼再合并） |
| must_keep 验证 | 只能事后由调用方查 | `保留清单` 模型自报 + 调用方复核 |

## `保留清单` 为什么要存在

它不是"让模型保证"，而是**制造一个可检查的矛盾点**：

模型说"我保留了 1,240 万元"（在 `保留清单` 里），但如果摘要里没有这个串，
就说明**模型的自我报告与实际输出不一致** —— 这类错误比"直接漏了"更值得抓，
因为它会让人误以为约束生效了。

调用方的正确用法：拿 `保留清单` 与 `摘要` 实际内容做交叉校验，
两者不一致即判该条失败。**不能只看 `保留清单` 就信。**
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SummaryV2(BaseModel):
    """v2 摘要结果：中间结果 + 自报保留清单。"""

    分段要点: list[str] = Field(
        default_factory=list,
        description="先提炼的语义要点，每条保留涉及的数字与专有名词原始写法",
    )
    摘要: str = Field(description="合并后的最终摘要，数字与专有名词逐字保留")
    字数: int = Field(ge=0, description="摘要的实际字数，由模型自报，与本地计数交叉校验")
    保留清单: list[str] = Field(
        default_factory=list,
        description="模型自报已保留的 must_keep 项；调用方须与摘要实际内容交叉校验",
    )
