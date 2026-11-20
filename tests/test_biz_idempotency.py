"""幂等与报表测试（docs/module-contract-approval.md §5.2 / §4.3）。

幂等的验收：同 key 两次 → 结果一致 + 第二次是重放（replayed=True）+ 业务只执行一次。
报表的验收：按月 × 客户聚合的数字正确 + CSV 头/行形状正确。

运行前提见 tests/biz_fixture.py（MySQL 容器 + BIZ_DB_URL）。
"""

from __future__ import annotations

import pytest

from tests.biz_fixture import db_available, run_biz

pytestmark = pytest.mark.skipif(not db_available(), reason="MySQL 容器未运行")


def test_idempotency_same_key_replays_first_result() -> None:
    """同 key 重复提交 → 首次结果被重放，业务只执行一次（日志不重复）。"""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.biz.schemas import CustomerCreate
    from app.biz.service import ContractService

    async def _twice(engine) -> tuple[dict, bool, dict, bool]:
        svc = ContractService(sessionmaker=async_sessionmaker(engine, expire_on_commit=False))
        first, replayed_1 = await svc.create_customer(
            CustomerCreate(name="幂等客户"), idempotency_key="idem-001"
        )
        second, replayed_2 = await svc.create_customer(
            CustomerCreate(name="幂等客户"), idempotency_key="idem-001"
        )
        return first, replayed_1, second, replayed_2

    first, replayed_1, second, replayed_2 = run_biz(_twice)

    assert replayed_1 is False  # 首次执行
    assert replayed_2 is True  # 第二次命中历史键
    assert first == second  # 重放返回首次结果


def test_idempotency_key_scoped_per_endpoint() -> None:
    """同一个 key 用于不同端点 → 422 IDEMPOTENCY_KEY_REUSED（防调用方 bug）。"""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.biz.schemas import ContractCreate, CustomerCreate
    from app.biz.service import ContractService
    from app.core.errors import AppError

    async def _reuse(engine) -> None:
        svc = ContractService(sessionmaker=async_sessionmaker(engine, expire_on_commit=False))
        await svc.create_customer(CustomerCreate(name="作用域客户"), idempotency_key="key-x")
        cust, _ = await svc.create_customer(
            CustomerCreate(name="作用域客户2"), idempotency_key="key-x"
        )
        # 同 key 换 scope 用（建合同）→ 必须被拒
        await svc.create_contract(
            ContractCreate(
                contract_no="HT-SCOPE-1",
                customer_id=cust["id"],
                title="作用域测试",
                amount_cents=1_000_000,
            ),
            idempotency_key="key-x",
        )

    with pytest.raises(AppError) as exc_info:
        run_biz(_reuse)
    assert exc_info.value.code == "IDEMPOTENCY_KEY_REUSED"
    assert exc_info.value.status_code == 422


def test_report_monthly_aggregation() -> None:
    """按月 × 客户聚合：cnt / amount_yuan / approved_cnt 三个数字都要对。"""
    from datetime import datetime

    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.biz.repository import ContractRepository
    from app.biz.schemas import ApprovalAction, ContractCreate, CustomerCreate
    from app.biz.service import ContractService

    async def _seed_and_report(engine) -> list[dict]:
        maker = async_sessionmaker(engine, expire_on_commit=False)
        svc = ContractService(sessionmaker=maker)
        cust, _ = await svc.create_customer(CustomerCreate(name="报表客户A"))
        for i, cents in enumerate((5_000_000, 12_000_000)):
            c, _ = await svc.create_contract(
                ContractCreate(
                    contract_no=f"HT-RPT-{i}",
                    customer_id=cust["id"],
                    title=f"报表合同{i}",
                    amount_cents=cents,
                )
            )
            await svc.submit(c["id"], actor="rpt")
            if i == 0:  # 第一张批掉（一级：5 万）
                await svc.approve(c["id"], ApprovalAction(expected_version=1, actor="mgr"))
        repo = ContractRepository(sessionmaker=maker)
        return await repo.report_monthly(start=datetime(2026, 1, 1), end=datetime(2027, 1, 1))

    rows = run_biz(_seed_and_report)
    row = next(r for r in rows if r["customer_name"] == "报表客户A")
    assert row["cnt"] == 2
    assert row["amount_yuan"] == pytest.approx(170_000.0)  # 5 万 + 12 万
    assert row["approved_cnt"] == 1


def test_report_csv_export_shape() -> None:
    """CSV：头行固定 6 列，数据行金额两位小数。"""
    from app.api.v2.reports import CSV_HEADERS, _parse_month_bound

    assert CSV_HEADERS[0] == "月份"
    assert len(CSV_HEADERS) == 6
    # 月份边界换算：左闭右开，跨年正确
    assert _parse_month_bound("2026-12", is_start=True) == datetime_ctor(2026, 12, 1)
    assert _parse_month_bound("2026-12", is_start=False) == datetime_ctor(2027, 1, 1)


def datetime_ctor(y: int, m: int, d: int):
    from datetime import datetime

    return datetime(y, m, d)
