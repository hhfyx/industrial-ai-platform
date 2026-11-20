"""SQLAlchemy 2.0 ORM：合同审批模块的 5 张表（docs/module-contract-approval.md §二）。

表清单与职责（每张表都对应一个面试可讲的设计决定）：

    customers         客户，外键目标，name 唯一
    contracts         合同主表：status + version（乐观锁）+ amount_cents（分）
    approval_nodes    审批节点：一张合同 N 个节点（金额阈值路由生成）
    approval_logs     审计日志：只 INSERT 不 UPDATE/DELETE
    idempotency_keys  幂等键：sha256(key) 为主键，存首次响应

两个最值得讲的设计（详见方案文档 §2.2）：

- `amount_cents BIGINT`：金额用「分」存整数。FLOAT 有 0.1+0.2≠0.3 的精度误差，
  合同金额不能错；DECIMAL 可用但要小心跨语言映射。「分 + 整数」展示层再除 100。
- `version INT`：乐观锁。两个审批人同时点通过，先提交的 `UPDATE ... WHERE version=0`
  成功；后提交的影响 0 行 → 409。不用悲观锁的原因：审批低频，
  不想让事务持行锁等用户思考。`with_for_update` 只是防止同一瞬间两个请求
  都通过版本校验（见 service.py 的注释）。

主键统一用 MySQL 方言的 `BIGINT`（unsigned）：SQLAlchemy 2.1 起通用类型
不再接受 `unsigned=True` 参数，而本模块**只跑 MySQL**（方案文档提醒 1），
用方言类型是显式声明这一前提，不是偷懒。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CHAR,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.mysql import BIGINT as MySQL_BIGINT
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    """模块级 Base。与主项目（无 DB）隔离，biz 的表不混进别的元数据。"""


class Customer(Base):
    """客户表：外键目标。name 唯一 —— 重名客户靠 tax_no 区分，不要靠放宽约束。"""

    __tablename__ = "customers"
    # eager_defaults：created_at 由服务端生成，flush 后立刻随 INSERT 取回。
    # 不开它 = 属性过期，访问时触发懒加载 SELECT = AsyncSession 下 MissingGreenlet。
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[int] = mapped_column(MySQL_BIGINT, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    tax_no: Mapped[str | None] = mapped_column(String(32), nullable=True)
    contact: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(3), nullable=False, server_default=func.now()
    )

    contracts: Mapped[list[Contract]] = relationship(back_populates="customer")


class Contract(Base):
    """合同主表。status + version 是并发正确性的全部载体。"""

    __tablename__ = "contracts"
    # 同 Customer：created_at/updated_at 服务端生成，flush 后立即取回（eager_defaults）
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[int] = mapped_column(MySQL_BIGINT, primary_key=True, autoincrement=True)
    contract_no: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)
    customer_id: Mapped[int] = mapped_column(
        MySQL_BIGINT, ForeignKey("customers.id"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    # 金额用「分」存 BIGINT 整数，不用 FLOAT/DECIMAL —— 见模块 docstring。
    amount_cents: Mapped[int] = mapped_column(MySQL_BIGINT, nullable=False)
    status: Mapped[str] = mapped_column(
        Enum(
            "draft",
            "pending",
            "approved",
            "rejected",
            "cancelled",
            name="contract_status",
        ),
        nullable=False,
        default="draft",
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_by: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(3), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(3),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    customer: Mapped[Customer] = relationship(back_populates="contracts")
    nodes: Mapped[list[ApprovalNode]] = relationship(
        back_populates="contract",
        cascade="all, delete-orphan",
        order_by="ApprovalNode.seq",
    )
    logs: Mapped[list[ApprovalLog]] = relationship(
        back_populates="contract", order_by="ApprovalLog.id"
    )


class ApprovalNode(Base):
    """审批节点。一张合同 N 个节点，seq 从 1 连续递增；(contract_id, seq) 唯一。

    status: waiting（未轮到）/ approved（已通过）/ rejected（驳回时本节点保留原状，
    余下 waiting 节点置 skipped）/ skipped（驳回或撤回时未决策的节点）。
    """

    __tablename__ = "approval_nodes"
    __table_args__ = (UniqueConstraint("contract_id", "seq", name="uk_nodes_contract_seq"),)

    id: Mapped[int] = mapped_column(MySQL_BIGINT, primary_key=True, autoincrement=True)
    contract_id: Mapped[int] = mapped_column(
        MySQL_BIGINT, ForeignKey("contracts.id"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(
        Enum("waiting", "approved", "rejected", "skipped", name="node_status"),
        nullable=False,
        default="waiting",
    )
    approver: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(3), nullable=True)

    contract: Mapped[Contract] = relationship(back_populates="nodes")


class ApprovalLog(Base):
    """审计日志：只 INSERT，永不 UPDATE/DELETE（docs §2.2③）。

    塞进主表 JSON 等于「改合同就等于改历史」；独立成表才能只增不改、
    按 (contract_id, created_at) 独立查询与索引。
    """

    __tablename__ = "approval_logs"

    id: Mapped[int] = mapped_column(MySQL_BIGINT, primary_key=True, autoincrement=True)
    contract_id: Mapped[int] = mapped_column(
        MySQL_BIGINT, ForeignKey("contracts.id"), nullable=False, index=True
    )
    from_status: Mapped[str] = mapped_column(String(16), nullable=False)
    to_status: Mapped[str] = mapped_column(String(16), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    actor: Mapped[str] = mapped_column(String(64), nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(3), nullable=False, server_default=func.now()
    )

    contract: Mapped[Contract] = relationship(back_populates="logs")


class IdempotencyKey(Base):
    """幂等键表：sha256(key) 为主键，scope 记录业务作用域（如 approve:123）。

    复用 W8 投递幂等的思路：同一 key 重复提交 → 返回首次结果，不重复执行。
    前端网络重试、用户连点两次的兜底 —— 与乐观锁互补：
    乐观锁解决「两个不同的人同时审」，幂等键解决「同一个人点了两次」。
    """

    __tablename__ = "idempotency_keys"

    key_hash: Mapped[str] = mapped_column(CHAR(64), primary_key=True)
    scope: Mapped[str] = mapped_column(String(32), nullable=False)
    response_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(3), nullable=False, server_default=func.now()
    )


__all__ = [
    "ApprovalLog",
    "ApprovalNode",
    "Base",
    "Contract",
    "Customer",
    "IdempotencyKey",
]
