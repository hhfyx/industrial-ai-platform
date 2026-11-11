"""v2 路由：鉴权 + 限流下的契约展示端点（W12 周一）。

## v2 与 v1 的关系（必答问题的工程答案）

v1 一行不改（docs/api-spec.md 冻结规则）。v2 是**新前缀 + 必填鉴权 + 限流头**，
端点先选 extract（最稳定、测试最全的契约）作为 v2 的样板：

- 请求/响应体与 v1 **完全相同** —— v2 改变的是"谁、以多大频率、以什么凭证"调用，
  不是"抽什么"。body 契约复用 v1 的 schema 类（`ExtractRequest`/`ExtractResponse`），
  不复制类定义 —— 复制类是契约漂移的起点。
- 新增的是**响应头**：`X-RateLimit-Remaining`（剩余配额）。
  头是可增量添加的（api-spec §4 规则 2：新增只能是可选字段）。

## 为什么只挂 extract 不挂全部

W12 周一的任务是"把鉴权限流的契约立住"，不是把六个端点全部翻新。
样板端点立住契约（401/422/429 全覆盖、可复现），其余端点按同一模式
迁移是机械工作，等规范 v2 定稿后逐个搬 —— 与 W8"同步版先行"同一策略：
先用最简单的路径把状态机、错误码、单测立住，再复制。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response
from fastapi import status as http_status

from app.core.deps import ExtractorDep, RequestIdDep
from app.core.security import remaining_header_for, require_api_key
from app.schemas.extract import ExtractRequest
from app.schemas.extract import ExtractResult as ExtractResponse

router = APIRouter(
    prefix="/api/v2",
    tags=["v2"],
    dependencies=[Depends(require_api_key)],
)

# 与 extract.py 同款的错误响应声明（OpenAPI 挂出统一错误形状）
ERROR_RESPONSES = {
    401: {"description": "缺少或无效的 X-API-Key（UNAUTHORIZED）"},
    422: {"description": "字段校验失败（INVALID_INPUT）"},
    429: {"description": "请求频率超限（RATE_LIMITED），带 Retry-After 头"},
    500: {"description": "服务端错误（INTERNAL_ERROR）"},
}

RATE_LIMIT_HEADER = "X-RateLimit-Remaining"


async def _write_rate_limit_header(request: Request, response: Response) -> None:
    """把剩余配额写进响应头。鉴权未启用时 security 返回 None，跳过。"""
    remaining = remaining_header_for(request)
    if remaining is not None:
        response.headers[RATE_LIMIT_HEADER] = str(remaining)


@router.post(
    "/extract",
    response_model=ExtractResponse,
    status_code=http_status.HTTP_200_OK,
    summary="结构化抽取（v2：鉴权 + 限流样板端点）",
    description=(
        "请求/响应体与 v1 `POST /api/v1/extract` 完全一致；本端点新增：\n\n"
        "- **鉴权**：请求头 `X-API-Key`（必填），失败 401 `UNAUTHORIZED`；\n"
        "- **限流**：单 key 配额 + 全局 QPS 双层令牌桶，超限 429 `RATE_LIMITED` "
        "带 `Retry-After`；成功响应带 `X-RateLimit-Remaining`；\n"
        "- 限流为**单实例内存桶**，多实例部署需 Redis（见 docs/api-spec.md v2 节）。"
    ),
    responses=ERROR_RESPONSES,
    response_model_exclude_none=True,
)
async def extract_v2(
    payload: ExtractRequest,
    extractor: ExtractorDep,
    request_id: RequestIdDep,
    request: Request,
    response: Response,
) -> ExtractResponse:
    """v2 抽取入口。业务路径与 v1 路由完全一致（同一 extractor 协议、同一异常策略）。

    异常不在这里接（与 v1 extract 同一条纪律）：服务层抛什么就走全局 handler，
    路由里写 try/except 会把"统一错误体"的责任从一处扩散到每一条路由。
    """
    output = await extractor.extract(text=payload.text, fields=payload.fields)

    await _write_rate_limit_header(request, response)
    return ExtractResponse(
        request_id=request_id,
        data=output.data,
        confidence=output.confidence,
        elapsed_ms=output.elapsed_ms,
        degraded=True if output.degraded else None,
    )


__all__ = ["router", "RATE_LIMIT_HEADER"]
