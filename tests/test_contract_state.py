"""状态机测试：7 条合法迁移 + 非法迁移（docs/module-contract-approval.md §七）。

纯函数测试：无 DB、毫秒级、拔网线必绿。非法迁移断言两件事 ——
code 稳定（INVALID_STATE_TRANSITION）和 409（调用方按 code 分支，不是按 message）。
"""

from __future__ import annotations

import pytest

from app.biz.state_machine import STATUSES, transition
from app.core.errors import AppError


# ---------------------------------------------------------------- 合法迁移（7 条）
def test_submit_draft_to_pending() -> None:
    assert transition("draft", "submit") == "pending"


def test_approve_non_last_stays_pending() -> None:
    # 迁移 2：非末节点通过 → 留在 pending（两级审批的中间态）
    assert transition("pending", "approve", is_last=False) == "pending"


def test_approve_last_to_approved() -> None:
    # 迁移 3：末节点通过 → approved
    assert transition("pending", "approve", is_last=True) == "approved"


def test_reject_pending_to_rejected() -> None:
    assert transition("pending", "reject") == "rejected"


def test_cancel_draft() -> None:
    assert transition("draft", "cancel") == "cancelled"


def test_cancel_pending() -> None:
    assert transition("pending", "cancel") == "cancelled"


def test_revise_rejected_to_draft() -> None:
    assert transition("rejected", "revise") == "draft"


# ---------------------------------------------------------------- 非法迁移
def _expect_invalid(from_status: str, action: str) -> None:
    with pytest.raises(AppError) as exc_info:
        transition(from_status, action)
    assert exc_info.value.code == "INVALID_STATE_TRANSITION"
    assert exc_info.value.status_code == 409
    # message 必须能直接回答"为什么"（方案 §三的要求）
    assert from_status in exc_info.value.message
    assert action in exc_info.value.message


def test_cannot_submit_approved() -> None:
    _expect_invalid("approved", "submit")


def test_cannot_submit_pending() -> None:
    _expect_invalid("pending", "submit")


def test_cannot_approve_draft() -> None:
    _expect_invalid("draft", "approve")


def test_cannot_reject_draft() -> None:
    _expect_invalid("draft", "reject")


def test_cannot_reject_approved() -> None:
    # 终态不可再操作 —— 这是「改合同等于改历史」要防的第一件事
    _expect_invalid("approved", "reject")


def test_cannot_cancel_approved() -> None:
    # 已批的合同不能撤，只能走线下流程 —— 业务规则的硬约束
    _expect_invalid("approved", "cancel")


def test_cannot_cancel_cancelled() -> None:
    _expect_invalid("cancelled", "cancel")


def test_cannot_revise_draft() -> None:
    _expect_invalid("draft", "revise")


def test_cannot_revise_pending() -> None:
    _expect_invalid("pending", "revise")


def test_cannot_submit_cancelled() -> None:
    _expect_invalid("cancelled", "submit")


def test_unknown_status_rejected() -> None:
    _expect_invalid("flying", "submit")


def test_unknown_action_rejected() -> None:
    _expect_invalid("draft", "delete")


# ---------------------------------------------------------------- 全覆盖巡检
def test_all_terminal_state_actions_are_invalid() -> None:
    """终态（approved/cancelled）× 全部动作 = 全部非法。

    这条测试是「状态机完备性」的证明：终态永远不可能被任何动作拉起来，
    未来加新动作时如果它应该对终态生效，这里会先红 —— 提醒人改规则而不是悄悄放行。
    """
    for status in ("approved", "cancelled"):
        for action in ("submit", "approve", "reject", "cancel", "revise"):
            _expect_invalid(status, action)


def test_status_set_is_exactly_five() -> None:
    assert STATUSES == ("draft", "pending", "approved", "rejected", "cancelled")
