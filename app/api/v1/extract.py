"""`POST /api/v1/extract` 路由（任务 13:30–14:30）。

**这一层只做三件事**：解析出参、调服务、包统一响应。任何"怎么抽"的逻辑都不许进来，
否则 W2 换实现时就要改动路由 —— 那正是 api-spec.md §4 明令禁止的。

路由的签名就是契约的证明：`ExtractRequest` 进、`ExtractResult` 出，
中间只经过 `ExtractorDep.extract(text, fields)` 这一个抽象。
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.core.deps import ExtractorDep, RequestIdDep
from app.core.middleware import ELAPSED_HEADER, MODEL_HEADER
from app.schemas.extract import ErrorResponse, ExtractRequest, ExtractResult

router = APIRouter(prefix="/api/v1", tags=["extract"])

# 在 /docs 里把错误形状挂出来。不挂的话 Swagger 只显示默认的 HTTPValidationError，
# 调用方照着它写代码就会拿到 `{detail: [...]}` —— 而契约其实是 4 字段的错误体。
ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "请求体不是合法 JSON（INVALID_JSON）"},
    422: {"model": ErrorResponse, "description": "字段校验失败（INVALID_INPUT），详见 details"},
    500: {"model": ErrorResponse, "description": "服务端错误（INTERNAL_ERROR 等）"},
}


@router.post(
    "/extract",
    response_model=ExtractResult,
    status_code=status.HTTP_200_OK,
    summary="从正文中抽取指定字段",
    description=(
        "按 `fields` 顺序抽取，返回 `data` 的键与请求完全一致；抽不到的字段为 `null`，"
        "整体置信度随之下降（不会报错）。\n\n"
        "**契约见 `docs/api-spec.md`，v1 冻结**：实现（规则版 → W2 的 LLM）可换，契约不变。\n\n"
        "错误形状统一为 `{code, message, request_id, details}`；"
        "400 = JSON 都没解析出来，422 = 解析成功但字段不合规。"
    ),
    responses=ERROR_RESPONSES,
)
async def extract_fields(
    payload: ExtractRequest,
    extractor: ExtractorDep,
    request_id: RequestIdDep,
    response: Response,
) -> ExtractResult:
    """抽取入口。

    唯一的业务动作是一次 `await extractor.extract(...)` —— 它内部有 100ms 的异步延迟，
    真实系统里那是模型调用的往返时间，这里让 async 语义提前成立。

    异常不在这里接：服务层抛什么就走全局 handler（`errors.py`），
    路由里写 try/except 会把"统一错误体"的责任从一处扩散到每一条路由。
    """
    output = await extractor.extract(text=payload.text, fields=payload.fields)

    # 耗时与实现标识走响应头（契约 §2.4）：头里的数和 body 里的必须是同一个
    # `elapsed_ms`，所以**由这里写、中间件不重复算** —— 中间件测的含序列化开销，
    # 会比服务层多 1–2ms，两个数对不上就等于契约自相矛盾。
    response.headers[ELAPSED_HEADER] = str(output.elapsed_ms)
    response.headers[MODEL_HEADER] = output.model

    return ExtractResult(
        request_id=request_id,
        data=output.data,
        confidence=output.confidence,
        elapsed_ms=output.elapsed_ms,
    )


__all__ = ["router"]
