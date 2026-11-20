"""Pydantic v2 请求/响应模型（docs/module-contract-approval.md §四）。

契约原则与 `app/schemas/extract.py` 一致：

- 响应模型就是契约：字段名、可选性在这里定死，OpenAPI 从它生成；
- 金额对外一律 `amount_yuan`（float，元）+ `amount_cents`（int，分）双字段，
  展示层不再各自除 100 —— 换算只发生在序列化这一个地方；
- 审批动作的 `expected_version` 必填：让「读-改-写」竞态在客户端就暴露出来
  （方案文档 §4.2），而不是服务端悄悄覆盖。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- 请求体
class CustomerCreate(BaseModel):
    """POST /api/v2/customers。"""

    name: str = Field(min_length=1, max_length=128)
    tax_no: str | None = Field(default=None, max_length=32)
    contact: str | None = Field(default=None, max_length=64)


class ContractCreate(BaseModel):
    """POST /api/v2/contracts。amount_cents 是「分」，必须 > 0（规则在 plan_nodes）。"""

    contract_no: str = Field(min_length=1, max_length=32)
    customer_id: int = Field(gt=0)
    title: str = Field(min_length=1, max_length=200)
    amount_cents: int = Field(gt=0, description="金额（分），10 万元 = 10_000_000")


class ApprovalAction(BaseModel):
    """approve / reject / cancel 的公共请求体。

    `expected_version` 必填（乐观锁的客户端侧一半）：
    请求带的版本 != 表里 version → 409 VERSION_CONFLICT。
    """

    comment: str | None = Field(default=None, max_length=500)
    expected_version: int = Field(ge=0, description="客户端读到的 version，乐观锁校验用")
    actor: str = Field(min_length=1, max_length=64, description="审批人（演示用，不接用户体系）")


class ReviseAction(BaseModel):
    """rejected → draft 的重提请求。revise 由创建人发起，重置审批链。"""

    actor: str = Field(min_length=1, max_length=64)


# ---------------------------------------------------------------- 响应体
class NodeOut(BaseModel):
    """审批节点（响应里 seq/role/status 三件套，decided 时带人与时间）。"""

    seq: int
    role: str
    status: str
    approver: str | None = None
    decided_at: datetime | None = None


class LogOut(BaseModel):
    """审计日志条目：谁、何时、什么动作、从什么态到什么态。"""

    id: int
    action: str
    actor: str
    from_status: str
    to_status: str
    comment: str | None = None
    created_at: datetime


class CustomerOut(BaseModel):
    """201 响应体（§4.1）。"""

    id: int
    name: str
    tax_no: str | None = None
    contact: str | None = None


class ContractOut(BaseModel):
    """合同详情/列表条目。nodes 随详情返回，列表接口不带（省查询）。"""

    id: int
    contract_no: str
    customer_id: int
    title: str
    amount_cents: int
    amount_yuan: float
    status: str
    version: int
    created_by: str
    created_at: datetime
    nodes: list[NodeOut] = Field(default_factory=list)


class ContractDetailOut(ContractOut):
    """GET /api/v2/contracts/{id}：详情 = 列表条目 + 完整审计日志。"""

    logs: list[LogOut] = Field(default_factory=list)


class ReportRowOut(BaseModel):
    """按月 × 客户的汇总行（报表 SQL 的直接映射，§4.3）。"""

    customer_id: int
    customer_name: str
    month: str
    cnt: int
    amount_yuan: float
    approved_cnt: int


class ReportOut(BaseModel):
    """GET /api/v2/reports/contracts 的 JSON 形状。"""

    rows: list[ReportRowOut] = Field(default_factory=list)
    total_cnt: int = 0
    total_amount_yuan: float = 0.0


__all__ = [
    "ApprovalAction",
    "ContractCreate",
    "ContractDetailOut",
    "ContractOut",
    "CustomerCreate",
    "CustomerOut",
    "LogOut",
    "NodeOut",
    "ReportOut",
    "ReportRowOut",
    "ReviseAction",
]
