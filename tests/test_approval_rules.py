"""金额阈值路由测试：3 档边界（docs/module-contract-approval.md §七）。

纯函数、毫秒级。边界值直接写在用例名里 —— 阈值改动时哪个用例红了，
文档就该改哪行。
"""

from __future__ import annotations

import pytest

from app.biz.approval_rules import ROLE_DIRECTOR, ROLE_MANAGER, plan_nodes
from app.core.errors import AppError


def test_below_threshold_single_level() -> None:
    # 9.9999 万元 → 一级（manager）
    assert plan_nodes(9_999_999) == (ROLE_MANAGER,)


def test_at_threshold_two_levels() -> None:
    # 恰好 10 万 → 两级。>= 是含等号的 —— 边界落在谁身上必须有测试钉住
    assert plan_nodes(10_000_000) == (ROLE_MANAGER, ROLE_DIRECTOR)


def test_above_threshold_two_levels() -> None:
    assert plan_nodes(128_000_000) == (ROLE_MANAGER, ROLE_DIRECTOR)


def test_order_is_manager_first() -> None:
    # 审批链有序：manager 在前，director 在后（seq 由调用方按序分配）
    roles = plan_nodes(10_000_000)
    assert roles[0] == ROLE_MANAGER
    assert roles[1] == ROLE_DIRECTOR


@pytest.mark.parametrize("amount", [0, -1, -128_000_000])
def test_zero_or_negative_rejected(amount: int) -> None:
    with pytest.raises(AppError) as exc_info:
        plan_nodes(amount)
    assert exc_info.value.code == "INVALID_INPUT"
    assert exc_info.value.status_code == 422
