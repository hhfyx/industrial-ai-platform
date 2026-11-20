"""金额阈值路由：决定一张合同的审批链有几级。纯函数，无 DB 依赖。

规则（docs/module-contract-approval.md §4.1）：

    amount_cents <  10_000_000（10 万元）→ 1 个节点：manager
    amount_cents >= 10_000_000          → 2 个节点：manager + director

阈值单位是「分」：10 万元 = 10_000_000 分。金额全程整数，
展示层才除以 100 —— FLOAT 的 0.1+0.2≠0.3 误差在合同场景不可接受，
DECIMAL 又要小心两侧语言映射，「分 + 整数」是最省心且可讲清的选择。

这条规则独立成文件的原因和状态机一样：它是**业务规则**不是数据访问，
单独可测（3 档测试毫秒级），改阈值时改这一个常量即可。
"""

from __future__ import annotations

from app.core.errors import AppError

# 10 万元（分）。>= 该值走两级审批。
TWO_LEVEL_THRESHOLD_CENTS: int = 10_000_000

ROLE_MANAGER = "manager"
ROLE_DIRECTOR = "director"


def plan_nodes(amount_cents: int) -> tuple[str, ...]:
    """按金额返回审批链（有序，seq 从 1 开始由调用方分配）。

    金额 <= 0 直接 422：零金额合同没有业务意义，宁可在入口挡掉
    也不要生成一张永远审批不完的合同。
    """
    if amount_cents <= 0:
        raise AppError(
            code="INVALID_INPUT",
            message=f"金额必须大于 0（收到 {amount_cents} 分）",
            status_code=422,
        )
    if amount_cents < TWO_LEVEL_THRESHOLD_CENTS:
        return (ROLE_MANAGER,)
    return (ROLE_MANAGER, ROLE_DIRECTOR)


__all__ = ["ROLE_DIRECTOR", "ROLE_MANAGER", "TWO_LEVEL_THRESHOLD_CENTS", "plan_nodes"]
