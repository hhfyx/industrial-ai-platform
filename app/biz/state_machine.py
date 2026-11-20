"""合同审批状态机：纯函数，不 import 任何 DB（docs/module-contract-approval.md §三）。

5 个状态、7 条合法迁移。所有迁移知识集中在 `transition()` 一个函数里，
非法迁移抛 `AppError("INVALID_STATE_TRANSITION", ..., 409)`，message 写清
「当前是 X，不能 Y」——调用方（service.py / 测试）不需要复制任何 if 分支。

这张表是模块里唯一的状态真相源：

    #   from → to     触发                前置/参数
    1   draft → pending      submit
    2   pending → pending    approve（非末节点，is_last=False）
    3   pending → approved   approve（末节点，is_last=True）
    4   pending → rejected   reject
    5   draft → cancelled    cancel
    6   pending → cancelled  cancel
    7   rejected → draft     revise（重置节点，version+1 由调用方做）

为什么状态机不碰 DB：它是「规则」，不是「数据」。规则不依赖数据源才能
毫秒级测试（23 条纯函数测试无容器），也才能在换 ORM、换数据库时原样复用。
这和项目里「契约冻结」是同一条纪律：规则层冻结，实现层可换。
"""

from __future__ import annotations

from app.core.errors import AppError

STATUSES: tuple[str, ...] = ("draft", "pending", "approved", "rejected", "cancelled")
ACTIONS: tuple[str, ...] = ("submit", "approve", "reject", "cancel", "revise")


def _invalid(from_status: str, action: str) -> AppError:
    """非法迁移的唯一出口：code 稳定、409、message 可读。

    message 里带「当前是 X，不能 Y」是给调用方（前端/脚本）的排障信息，
    也是面试时展示错误设计的细节：错误不仅要有 code，还要能直接回答"为什么"。
    """
    return AppError(
        code="INVALID_STATE_TRANSITION",
        message=f"非法状态迁移：当前是 {from_status}，不能 {action}",
        status_code=409,
    )


def transition(from_status: str, action: str, *, is_last: bool = True) -> str:
    """从 from_status 执行 action，返回目标状态；非法组合抛 409。

    `is_last` 只对 `pending + approve` 有意义：
    - True  = 没有 more waiting 节点 → 合同 approved（迁移 3）
    - False = 还有下一级节点 → 留在 pending，当前节点 approved（迁移 2）
    其它组合忽略该参数（默认 True，因为那 5 条迁移都一步到位）。
    """
    if from_status not in STATUSES:
        raise _invalid(from_status, action)
    if action not in ACTIONS:
        raise _invalid(from_status, action)

    if from_status == "draft" and action == "submit":
        return "pending"
    if from_status == "pending" and action == "approve":
        return "approved" if is_last else "pending"
    if from_status == "pending" and action == "reject":
        return "rejected"
    if from_status in ("draft", "pending") and action == "cancel":
        return "cancelled"
    if from_status == "rejected" and action == "revise":
        return "draft"

    # 剩下的组合全部非法：approved/cancelled 是终态；rejected 只能 revise。
    raise _invalid(from_status, action)


__all__ = ["ACTIONS", "STATUSES", "transition"]
