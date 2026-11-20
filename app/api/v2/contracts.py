"""合同审批的 5 类端点（docs/module-contract-approval.md §四）。

契约沿用项目既有约定：

- `X-API-Key` 鉴权与双层限流：v2 的 `require_api_key` 依赖，一行声明；
- 错误一律 `AppError` → 统一 4 字段错误体 `{code, message, request_id, details}`；
- `Idempotency-Key` 请求头可选：命中历史键时响应头带
  `X-Idempotent-Replay: true`，调用方能区分"新执行"和"重放"。

路由层不做业务判断：所有状态/版本/幂等逻辑在 ContractService，
这里只做 HTTP 形状（状态码、响应模型、头）。service 通过 `Depends`
注入（ServiceDep）—— 测试用 dependency_overrides 换假实现，
这正是依赖注入存在的理由之一（conftest.py 同款机制）。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.biz.schemas import (
    ApprovalAction,
    ContractCreate,
    ContractDetailOut,
    ContractOut,
    CustomerCreate,
    CustomerOut,
    ReviseAction,
)
from app.biz.service import ContractService
from app.core.deps import RequestIdDep
from app.core.security import require_api_key

router = APIRouter(
    prefix="/api/v2",
    tags=["contracts"],
    dependencies=[Depends(require_api_key)],
)


def get_contract_service() -> ContractService:
    """service 装配（惰性）。

    为什么不在 import 时构造：service 构造会触碰 db.get_sessionmaker()
    → get_engine()，后者要求 BIZ_DB_URL 已设置。import app.main 时
    （conftest 收集离线测试）绝不能因缺数据库配置而炸 —— 装配推迟到
    第一个请求，配置错误挡在启动时而不是 import 时。
    与 app.core.deps.get_extractor 同一条纪律。
    """
    return ContractService()


def get_contract_service_dep() -> ContractService:
    """FastAPI 依赖入口。单层包装让 dependency_overrides 有稳定的覆盖键。"""
    return get_contract_service()


ServiceDep = Annotated[ContractService, Depends(get_contract_service_dep)]


def _idem_header(request: Request, replayed: bool) -> dict[str, str]:
    """命中幂等键时显式回一个头 —— 重放与新执行对调用方可区分（W8 纪律）。"""
    return {"X-Idempotent-Replay": "true"} if replayed else {}


@router.post("/customers", status_code=201, response_model=CustomerOut)
async def create_customer(
    payload: CustomerCreate, request: Request, request_id: RequestIdDep, service: ServiceDep
) -> CustomerOut:
    """建客户（§4.1）。重名 409 DUPLICATE_CUSTOMER。"""
    data, replayed = await service.create_customer(
        payload, idempotency_key=request.headers.get("Idempotency-Key")
    )
    return JSONResponse(status_code=201, content=data, headers=_idem_header(request, replayed))


@router.post("/contracts", status_code=201, response_model=ContractOut)
async def create_contract(
    payload: ContractCreate, request: Request, request_id: RequestIdDep, service: ServiceDep
) -> ContractOut:
    """建合同（draft，预生成审批节点）。客户不存在 404，单号重复 409。"""
    data, replayed = await service.create_contract(
        payload, idempotency_key=request.headers.get("Idempotency-Key")
    )
    return JSONResponse(status_code=201, content=data, headers=_idem_header(request, replayed))


@router.post("/contracts/{contract_id}/submit", response_model=ContractOut)
async def submit_contract(contract_id: int, request: Request, service: ServiceDep) -> ContractOut:
    """draft → pending。提交人从请求体拿不到（无 body），用头或默认。"""
    actor = request.headers.get("X-Actor", "system")
    data = await service.submit(contract_id, actor)
    return JSONResponse(status_code=200, content=data)


@router.post("/contracts/{contract_id}/approve", response_model=ContractOut)
async def approve_contract(
    contract_id: int, payload: ApprovalAction, service: ServiceDep
) -> ContractOut:
    """通过当前节点。非末节点 → 仍 pending；末节点 → approved。版本冲突 409。"""
    data = await service.approve(contract_id, payload)
    return JSONResponse(status_code=200, content=data)


@router.post("/contracts/{contract_id}/reject", response_model=ContractOut)
async def reject_contract(
    contract_id: int, payload: ApprovalAction, service: ServiceDep
) -> ContractOut:
    """驳回：pending → rejected，余下节点 skipped。"""
    data = await service.reject(contract_id, payload)
    return JSONResponse(status_code=200, content=data)


@router.post("/contracts/{contract_id}/cancel", response_model=ContractOut)
async def cancel_contract(
    contract_id: int, payload: ApprovalAction, service: ServiceDep
) -> ContractOut:
    """撤回：draft|pending → cancelled。pending 已有人决策则 409。"""
    data = await service.cancel(contract_id, payload)
    return JSONResponse(status_code=200, content=data)


@router.post("/contracts/{contract_id}/revise", response_model=ContractOut)
async def revise_contract(
    contract_id: int, payload: ReviseAction, service: ServiceDep
) -> ContractOut:
    """重提：rejected → draft，审批链重置。"""
    data = await service.revise(contract_id, payload)
    return JSONResponse(status_code=200, content=data)


@router.get("/contracts", response_model=list[ContractOut])
async def list_contracts(
    status: str | None = None,
    customer_id: int | None = None,
    page: int = 1,
    size: int = 20,
) -> list[dict]:
    """分页列表（§4.3）。过滤非法 status → 422（在 repository 里抛）。"""
    from app.biz.repository import ContractRepository

    rows, _total = await ContractRepository().list_contracts(
        status=status, customer_id=customer_id, page=page, size=size
    )
    return rows


@router.get("/contracts/{contract_id}", response_model=ContractDetailOut)
async def get_contract(contract_id: int) -> dict:
    """详情：合同 + 节点 + 审计日志。"""
    from app.biz.repository import ContractRepository

    return await ContractRepository().get_detail(contract_id)


__all__ = ["ServiceDep", "get_contract_service", "get_contract_service_dep", "router"]
