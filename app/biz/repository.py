"""读路径封装：列表、详情、报表聚合 SQL（docs/module-contract-approval.md §4.3）。

报表 SQL 是 JD「写基础 SQL」的直接证据：

    GROUP BY 客户 × 月份，SUM(CASE WHEN status='approved') 统计已批笔数，
    DATE_FORMAT('%Y-%m') 出月份桶 —— 全部在数据库聚合，不把明细拉回 Python 数。

只读不写，所以不开显式事务；sessionmaker 自动提交读事务。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.biz.models import ApprovalLog, ApprovalNode, Contract, Customer
from app.biz.service import ContractService  # 复用 _contract_dict 的序列化形状
from app.core.errors import AppError

VALID_STATUSES = ("draft", "pending", "approved", "rejected", "cancelled")


class ContractRepository:
    """查询对象。与 ContractService 共享序列化函数，避免响应形状分叉。"""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession] | None = None) -> None:
        self._sm = sessionmaker or get_sessionmaker_lazy()

    async def list_contracts(
        self,
        *,
        status: str | None = None,
        customer_id: int | None = None,
        page: int = 1,
        size: int = 20,
    ) -> tuple[list[dict], int]:
        """分页列表。过滤参数全部可选；status 非法值 422（不静默返回空页）。"""
        if status is not None and status not in VALID_STATUSES:
            raise AppError(
                code="INVALID_INPUT",
                message=f"status 必须是 {'/'.join(VALID_STATUSES)} 之一，收到 {status}",
                status_code=422,
            )

        async with self._sm() as session:
            conditions = []
            if status is not None:
                conditions.append(Contract.status == status)
            if customer_id is not None:
                conditions.append(Contract.customer_id == customer_id)

            total = await session.scalar(
                select(func.count()).select_from(Contract).where(*conditions)
            )
            rows = (
                await session.scalars(
                    select(Contract)
                    .where(*conditions)
                    .order_by(Contract.id.desc())
                    .offset((page - 1) * size)
                    .limit(size)
                )
            ).all()
            return (
                [ContractService._contract_dict(c, nodes=[]) for c in rows],
                int(total or 0),
            )

    async def get_detail(self, contract_id: int) -> dict:
        """详情 = 合同 + 节点 + 完整审计日志（§4.3 的 GET /contracts/{id}）。"""
        async with self._sm() as session:
            contract = await session.get(Contract, contract_id)
            if contract is None:
                raise AppError(
                    code="CONTRACT_NOT_FOUND",
                    message=f"合同不存在：id={contract_id}",
                    status_code=404,
                )
            # AsyncSession 懒加载是非法 IO（MissingGreenlet）：
            # 节点与日志都显式查，不碰 relationship 属性
            nodes = list(
                (
                    await session.scalars(
                        select(ApprovalNode)
                        .where(ApprovalNode.contract_id == contract_id)
                        .order_by(ApprovalNode.seq)
                    )
                ).all()
            )
            logs = list(
                (
                    await session.scalars(
                        select(ApprovalLog)
                        .where(ApprovalLog.contract_id == contract_id)
                        .order_by(ApprovalLog.id)
                    )
                ).all()
            )
            data = ContractService._contract_dict(contract, nodes=nodes)
            data["logs"] = [
                {
                    "id": log.id,
                    "action": log.action,
                    "actor": log.actor,
                    "from_status": log.from_status,
                    "to_status": log.to_status,
                    "comment": log.comment,
                    "created_at": log.created_at.isoformat() if log.created_at else None,
                }
                for log in logs
            ]
            return data

    async def report_monthly(
        self, *, start: datetime | None = None, end: datetime | None = None
    ) -> list[dict]:
        """按月 × 客户聚合（报表 SQL 原样翻译成 Core select）。

        SUM(c.status='approved') 是 MySQL 惯用法（布尔当 0/1 加），
        SQLAlchemy 里写成 SUM(CASE WHEN ...) 才是跨方言写法 —— 这里目标就是
        MySQL，保留等价的 CASE 写法让 SQL 意图显式。
        """
        month = func.date_format(Contract.created_at, "%Y-%m").label("month")
        approved_cnt = func.sum(func.if_(Contract.status == "approved", 1, 0)).label("approved_cnt")

        async with self._sm() as session:
            q = (
                select(
                    Contract.customer_id,
                    Customer.name.label("customer_name"),
                    month,
                    func.count().label("cnt"),
                    (func.sum(Contract.amount_cents) / 100.0).label("amount_yuan"),
                    approved_cnt,
                )
                .join(Customer, Customer.id == Contract.customer_id)
                .group_by(Contract.customer_id, Customer.name, month)
                .order_by(func.sum(Contract.amount_cents).desc())
                .limit(100)
            )
            if start is not None:
                q = q.where(Contract.created_at >= start)
            if end is not None:
                q = q.where(Contract.created_at < end)

            result = await session.execute(q)
            rows = result.all()
            return [
                {
                    "customer_id": r.customer_id,
                    "customer_name": r.customer_name,
                    "month": r.month,
                    "cnt": int(r.cnt),
                    "amount_yuan": float(r.amount_yuan or 0),
                    "approved_cnt": int(r.approved_cnt or 0),
                }
                for r in rows
            ]

    async def logs_for_export(self, contract_id: int | None = None) -> list[dict]:
        """审计日志平铺（CSV 导出用）。contract_id=None 导出全部。"""
        async with self._sm() as session:
            q = (
                select(ApprovalLog, Contract.contract_no)
                .join(Contract, Contract.id == ApprovalLog.contract_id)
                .order_by(ApprovalLog.id)
                .limit(10000)
            )
            if contract_id is not None:
                q = q.where(ApprovalLog.contract_id == contract_id)
            rows = (await session.execute(q)).all()
            return [
                {
                    "id": log.id,
                    "contract_no": contract_no,
                    "contract_id": log.contract_id,
                    "action": log.action,
                    "actor": log.actor,
                    "from_status": log.from_status,
                    "to_status": log.to_status,
                    "comment": log.comment,
                    "created_at": log.created_at.isoformat() if log.created_at else None,
                }
                for log, contract_no in rows
            ]


def get_sessionmaker_lazy() -> async_sessionmaker[AsyncSession]:
    """延迟 import 的 sessionmaker 获取器。

    为什么不直接 `from app.biz.db import get_sessionmaker`：repository 也可能被
    纯函数测试 import（收集模块元数据），而 db.py 构造引擎需要环境变量。
    延迟到实例化时才碰 db，import 侧保持零副作用。
    """
    from app.biz.db import get_sessionmaker

    return get_sessionmaker()


__all__ = ["VALID_STATUSES", "ContractRepository"]
