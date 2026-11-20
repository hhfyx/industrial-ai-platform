"""事务边界全部在这里（docs/module-contract-approval.md §五）。

## 审批事务的形状（一个事务改三张表）

    approval_nodes（当前节点 approved）
  + contracts（status / version+1）
  + approval_logs（追加一条审计）

少一张都不行：节点过了但主表还是 pending、日志记了但状态没改，
都是并发下真实会出现（且面试官会追问）的脏数据。

## 锁策略：乐观为主，行锁为辅

- `expected_version`（请求体必填）是**乐观锁**：审批低频，不想让事务持行锁
  等用户思考；冲突时 409 让用户刷新重试。
- `with_for_update` 是**辅助**：乐观锁的校验和写入之间有一个微小窗口，
  两个并发请求可能都读到 version=0 并都通过校验 —— 行锁把「读版本」和
  「写版本」放进同一把锁里，杜绝双写。理论上只靠 UPDATE ... WHERE version=?
  的影响行数判断也够（影响 0 行即冲突），两种手段这里叠加，注释即面试答案。

## 幂等（复用 W8 投递幂等）

`Idempotency-Key` 头可选。同一 key 重复提交 → 直接返回**首次**的响应（200），
不重复执行。key_hash = sha256(key)，scope 隔离不同端点的同名 key。
幂等键的写入与业务在**同一个事务**里：业务回滚时键也回滚，
不会出现「键占了坑、业务没做成」的假命中。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.biz.approval_rules import plan_nodes
from app.biz.db import get_sessionmaker
from app.biz.models import ApprovalLog, ApprovalNode, Contract, Customer, IdempotencyKey
from app.biz.schemas import (
    ApprovalAction,
    ContractCreate,
    CustomerCreate,
    ReviseAction,
)
from app.biz.state_machine import transition
from app.core.errors import AppError

# 幂等键响应体的序列化场景标签：json.dumps 传给 default 的兜底（datetime 等）


def _key_hash(key: str) -> str:
    """sha256(key)。表里绝不存明文 key（客户端可能拿业务 id 拼它）。"""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _now() -> datetime:
    """统一用 UTC。MySQL DATETIME 不带时区，进程内一律 UTC 才不会混出两套时间。"""
    return datetime.now(UTC).replace(tzinfo=None)


class ContractService:
    """写路径的服务对象。读路径（列表/详情/报表）在 repository.py。

    每个公开方法都自带事务（`async with session.begin()`）——
    事务边界在 service 层，路由层不碰 session，这是分层纪律的一部分。
    """

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession] | None = None) -> None:
        # 测试可以注入自己的 sessionmaker（连测试库）；生产走 db.py 的单例。
        self._sm = sessionmaker or get_sessionmaker()

    # ---------------------------------------------------------------- 幂等
    async def _run_idempotent(
        self,
        *,
        idempotency_key: str | None,
        scope: str,
        body: Callable[[AsyncSession], Awaitable[dict]],
    ) -> tuple[dict, bool]:
        """带幂等保护地执行一个事务体。

        返回 (响应, replayed)。replayed=True 表示命中了历史键，
        响应是首次执行的存档 —— 调用方（路由）据此不打多余的日志/指标。

        实现要点：SELECT 键 → 执行业务 → INSERT 键 在**同一个事务**里。
        两个并发同 key 请求同时过 SELECT？第二个会撞主键（ IntegrityError），
        对幂等来说撞主键 = 首次还在执行中，回 409 即可（极少发生，不为其加锁）。
        """
        if not idempotency_key:
            async with self._sm() as session, session.begin():
                return await body(session), False

        digest = _key_hash(idempotency_key)
        async with self._sm() as session, session.begin():
            existing = await session.get(IdempotencyKey, digest)
            if existing is not None and existing.scope == scope:
                # 命中：返回首次结果。response_json 为 None 说明首次事务
                # 还没提交完（理论不可达：提交与写键同事务），按冲突处理。
                if existing.response_json is None:
                    raise AppError(
                        code="IDEMPOTENCY_IN_PROGRESS",
                        message="相同的幂等键仍在处理中，请稍后重试",
                        status_code=409,
                    )
                return json.loads(existing.response_json), True
            if existing is not None:
                # 同 hash 不同 scope：客户端复用了 key。这是调用方 bug，422。
                raise AppError(
                    code="IDEMPOTENCY_KEY_REUSED",
                    message="幂等键已被其它接口使用，请更换 Idempotency-Key",
                    status_code=422,
                )

            result = await body(session)
            session.add(
                IdempotencyKey(
                    key_hash=digest,
                    scope=scope,
                    response_json=json.dumps(result, ensure_ascii=False, default=str),
                )
            )
            return result, False

    # ---------------------------------------------------------------- 客户
    async def create_customer(
        self, payload: CustomerCreate, *, idempotency_key: str | None = None
    ) -> tuple[dict, bool]:
        async def _do(session: AsyncSession) -> dict:
            dup = await session.scalar(select(Customer.id).where(Customer.name == payload.name))
            if dup is not None:
                raise AppError(
                    code="DUPLICATE_CUSTOMER",
                    message=f"客户名已存在：{payload.name}",
                    status_code=409,
                )
            customer = Customer(name=payload.name, tax_no=payload.tax_no, contact=payload.contact)
            session.add(customer)
            await session.flush()
            return {
                "id": customer.id,
                "name": customer.name,
                "tax_no": customer.tax_no,
                "contact": customer.contact,
            }

        return await self._run_idempotent(
            idempotency_key=idempotency_key, scope="customer", body=_do
        )

    # ---------------------------------------------------------------- 合同创建
    async def create_contract(
        self, payload: ContractCreate, *, idempotency_key: str | None = None
    ) -> tuple[dict, bool]:
        """建合同（draft）并**预生成审批节点**（方案 §三迁移 1 的前置条件）。

        节点在建单时生成而不是提交时：金额 → 节点数是创建时确定的规则，
        提交时才生成会让「先改金额后提交」的合同带着旧节点链。
        """

        async def _do(session: AsyncSession) -> dict:
            customer = await session.get(Customer, payload.customer_id)
            if customer is None:
                raise AppError(
                    code="CUSTOMER_NOT_FOUND",
                    message=f"客户不存在：id={payload.customer_id}",
                    status_code=404,
                )
            dup = await session.scalar(
                select(Contract.id).where(Contract.contract_no == payload.contract_no)
            )
            if dup is not None:
                raise AppError(
                    code="DUPLICATE_CONTRACT_NO",
                    message=f"合同号已存在：{payload.contract_no}",
                    status_code=409,
                )

            contract = Contract(
                contract_no=payload.contract_no,
                customer_id=payload.customer_id,
                title=payload.title,
                amount_cents=payload.amount_cents,
                status="draft",
                version=0,
                created_by="api",  # 演示无用户体系；真实系统取认证主体
            )
            session.add(contract)
            await session.flush()  # 拿 contract.id

            roles = plan_nodes(payload.amount_cents)  # 金额 ≤ 0 在这里抛 422
            for seq, role in enumerate(roles, start=1):
                session.add(ApprovalNode(contract_id=contract.id, seq=seq, role=role))
            await session.flush()

            nodes = await self._nodes_of(session, contract.id)
            return self._contract_dict(contract, nodes=nodes)

        return await self._run_idempotent(
            idempotency_key=idempotency_key, scope="contract", body=_do
        )

    # ---------------------------------------------------------------- 状态动作
    async def submit(self, contract_id: int, actor: str) -> dict:
        """draft → pending。节点已预生成，直接过状态机。"""

        async def _do(session: AsyncSession) -> dict:
            contract = await self._locked_contract(session, contract_id, expected_version=None)
            old = contract.status
            new = transition(old, "submit")
            return await self._apply(
                session, contract, action="submit", old=old, new=new, actor=actor, comment=None
            )

        return (await self._run_idempotent(idempotency_key=None, scope="noop", body=_do))[0]

    async def approve(self, contract_id: int, payload: ApprovalAction) -> dict:
        """pending → pending(下一节点) 或 pending → approved(末节点)。乐观锁在这里生效。"""

        async def _do(session: AsyncSession) -> dict:
            contract = await self._locked_contract(
                session, contract_id, expected_version=payload.expected_version
            )
            is_last = await self._decide_is_last(session, contract)
            old = contract.status
            new = transition(old, "approve", is_last=is_last)
            return await self._apply(
                session,
                contract,
                action="approve",
                old=old,
                new=new,
                actor=payload.actor,
                comment=payload.comment,
            )

        return (await self._run_idempotent(idempotency_key=None, scope="noop", body=_do))[0]

    async def reject(self, contract_id: int, payload: ApprovalAction) -> dict:
        """pending → rejected。余下 waiting 节点全部 skipped。"""

        async def _do(session: AsyncSession) -> dict:
            contract = await self._locked_contract(
                session, contract_id, expected_version=payload.expected_version
            )
            old = contract.status
            new = transition(old, "reject")
            result = await self._apply(
                session,
                contract,
                action="reject",
                old=old,
                new=new,
                actor=payload.actor,
                comment=payload.comment,
            )
            await self._skip_waiting_nodes(session, contract.id)
            return result

        return (await self._run_idempotent(idempotency_key=None, scope="noop", body=_do))[0]

    async def cancel(self, contract_id: int, payload: ApprovalAction) -> dict:
        """draft|pending → cancelled。pending 撤回要求「未有人审批」（方案迁移 6）。"""

        async def _do(session: AsyncSession) -> dict:
            contract = await self._locked_contract(
                session, contract_id, expected_version=payload.expected_version
            )
            if contract.status == "pending":
                decided = await session.scalar(
                    select(ApprovalNode.id)
                    .where(ApprovalNode.contract_id == contract.id)
                    .where(ApprovalNode.status != "waiting")
                    .limit(1)
                )
                if decided is not None:
                    raise AppError(
                        code="ALREADY_DECIDED",
                        message="已有审批人决策，不能撤回；请走驳回流程",
                        status_code=409,
                    )
            old = contract.status
            new = transition(old, "cancel")
            result = await self._apply(
                session,
                contract,
                action="cancel",
                old=old,
                new=new,
                actor=payload.actor,
                comment=payload.comment,
            )
            await self._skip_waiting_nodes(session, contract.id)
            return result

        return (await self._run_idempotent(idempotency_key=None, scope="noop", body=_do))[0]

    async def revise(self, contract_id: int, payload: ReviseAction) -> dict:
        """rejected → draft。重置审批节点为 waiting，等待改单后重新 submit（迁移 7）。"""

        async def _do(session: AsyncSession) -> dict:
            contract = await self._locked_contract(session, contract_id, expected_version=None)
            old = contract.status
            new = transition(old, "revise")
            result = await self._apply(
                session,
                contract,
                action="revise",
                old=old,
                new=new,
                actor=payload.actor,
                comment=None,
            )
            # 重置整条审批链（此前可能有 approved/rejected/skipped 节点）
            nodes = (
                await session.scalars(
                    select(ApprovalNode).where(ApprovalNode.contract_id == contract.id)
                )
            ).all()
            for node in nodes:
                node.status = "waiting"
                node.approver = None
                node.decided_at = None
            return result

        return (await self._run_idempotent(idempotency_key=None, scope="noop", body=_do))[0]

    # ---------------------------------------------------------------- 内部工具
    async def _locked_contract(
        self, session: AsyncSession, contract_id: int, *, expected_version: int | None
    ) -> Contract:
        """行锁读合同 + 版本校验。

        `with_for_update` 的作用：把「读 version → 校验 → 写 version」放进同一把行锁，
        杜绝两个并发请求同时通过校验。它不是悲观锁的「锁住等用户」——
        事务只持锁几毫秒（校验 + 三表写入），用户思考时间发生在事务之外。
        """
        contract = await session.get(Contract, contract_id, with_for_update=True)
        if contract is None:
            raise AppError(
                code="CONTRACT_NOT_FOUND",
                message=f"合同不存在：id={contract_id}",
                status_code=404,
            )
        if expected_version is not None and contract.version != expected_version:
            raise AppError(
                code="VERSION_CONFLICT",
                message=f"该合同已被他人处理（当前 version={contract.version}，"
                f"你提交的是 {expected_version}），请刷新后重试",
                status_code=409,
            )
        return contract

    async def _decide_is_last(self, session: AsyncSession, contract: Contract) -> bool:
        """approve 是否命中末节点：不存在「seq > 当前最小 waiting 节点」的 waiting 节点。"""
        waiting = (
            await session.scalars(
                select(ApprovalNode)
                .where(ApprovalNode.contract_id == contract.id)
                .where(ApprovalNode.status == "waiting")
                .order_by(ApprovalNode.seq)
            )
        ).all()
        if not waiting:
            raise AppError(
                code="NO_PENDING_NODE",
                message="没有待审批节点（数据不一致），请联系维护者",
                status_code=500,
            )
        # 只剩一个 waiting 节点 = 当前就是末节点 → 通过即 approved。
        return len(waiting) == 1

    async def _apply(
        self,
        session: AsyncSession,
        contract: Contract,
        *,
        action: str,
        old: str,
        new: str,
        actor: str,
        comment: str | None,
    ) -> dict:
        """状态动作的公共落库：当前节点 + 主表 + 审计日志（一个事务的三张表）。

        并发测试的验收锚点在这：日志与状态变更同事务，
        19 个失败的事务回滚后 approval_logs 只多 1 条。
        """
        # 当前节点（最小 seq 的 waiting）标记为 approved —— reject/cancel/revise
        # 不动当前节点（它没有"通过"语义），skip_waiting_nodes 处理余下节点。
        if action == "approve":
            waiting = (
                await session.scalars(
                    select(ApprovalNode)
                    .where(ApprovalNode.contract_id == contract.id)
                    .where(ApprovalNode.status == "waiting")
                    .order_by(ApprovalNode.seq)
                )
            ).all()
            if waiting:
                current = waiting[0]
                current.status = "approved"
                current.approver = actor
                current.decided_at = _now()

        contract.status = new
        contract.version += 1
        session.add(
            ApprovalLog(
                contract_id=contract.id,
                from_status=old,
                to_status=new,
                action=action,
                actor=actor,
                comment=comment,
            )
        )
        await session.flush()
        nodes = await self._nodes_of(session, contract.id)
        return self._contract_dict(contract, nodes=nodes)

    async def _nodes_of(self, session: AsyncSession, contract_id: int) -> list[ApprovalNode]:
        """显式查询节点并按 seq 排序。

        为什么不访问 `contract.nodes` 关系：AsyncSession 的**懒加载是非法 IO**
        （MissingGreenlet），关系必须在查询时 selectinload 或干脆显式查。
        这里选显式查 —— 节点数量 ≤ 2，多一次往返可忽略，换来的是不用
        在每个读路径上记得挂 selectinload。
        """
        return list(
            (
                await session.scalars(
                    select(ApprovalNode)
                    .where(ApprovalNode.contract_id == contract_id)
                    .order_by(ApprovalNode.seq)
                )
            ).all()
        )

    async def _skip_waiting_nodes(self, session: AsyncSession, contract_id: int) -> None:
        """余下 waiting 节点 → skipped（reject / cancel 的副作用）。"""
        nodes = (
            await session.scalars(
                select(ApprovalNode)
                .where(ApprovalNode.contract_id == contract_id)
                .where(ApprovalNode.status == "waiting")
            )
        ).all()
        for node in nodes:
            node.status = "skipped"

    # ---------------------------------------------------------------- 序列化
    @staticmethod
    def _contract_dict(contract: Contract, *, nodes: list[ApprovalNode]) -> dict:
        """行 → 响应 dict。json.dumps 存幂等键、路由返回响应共用同一形状。

        `nodes` 必须由调用方显式查出（见 _nodes_of 的 MissingGreenlet 注释）；
        列表接口传 `nodes=[]`（列表不展示节点）。
        """
        data: dict = {
            "id": contract.id,
            "contract_no": contract.contract_no,
            "customer_id": contract.customer_id,
            "title": contract.title,
            "amount_cents": contract.amount_cents,
            "amount_yuan": round(contract.amount_cents / 100, 2),
            "status": contract.status,
            "version": contract.version,
            "created_by": contract.created_by,
            "created_at": contract.created_at.isoformat() if contract.created_at else None,
        }
        data["nodes"] = [
            {
                "seq": n.seq,
                "role": n.role,
                "status": n.status,
                "approver": n.approver,
                "decided_at": n.decided_at.isoformat() if n.decided_at else None,
            }
            for n in sorted(nodes, key=lambda n: n.seq)
        ]
        return data


__all__ = ["ContractService"]
