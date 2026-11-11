"""`POST /api/v1/generate` —— 缺陷图像生成（同步版，项目一 · W8 周三）。

**这一层只做四件事**（与 extract.py 同一条规矩）：校验白名单、调服务、
映射已知异常、包统一响应。任何"怎么画"的逻辑都不许进来。

## 为什么先做同步版（必答问题）

异步版的复杂度在"任务状态机 + 查询接口 + 产物清理"，其中每一个失败路径
都会长出自己的错误格式 —— 先用同步版把**契约、错误码、单测**立住，
异步版只是给同一份契约套一层 202 + task_id。
反过来先做异步，等于在错误格式还没定的时候就把它们冻结进状态机。

## 异常映射表（21:15 格的四类，全部可复现于离线测试）

| 异常 | HTTP | code | 调用方动作 |
| --- | --- | --- | --- |
| `ValueError`（白名单外类别） | 422 | `UNKNOWN_DEFECT_CLASS` | 改类别，不重试 |
| `GeneratorUnavailableError` | 503 | `MODEL_NOT_LOADED` | 联系维护者装配权重 |
| `GenerationOOMError` | 503 | `GPU_OOM` | 按 Retry-After 退避 |
| `GenerationTimeoutError` | 504 | `GENERATION_TIMEOUT` | 减小 count/size 后重试 |
| 其余 | 500 | `INTERNAL_ERROR` | 全局 handler 兜底 |

映射在路由层做而不在服务层抛 AppError：services 层不 import fastapi
（分层纪律），异常的 HTTP 语义归路由。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Response, status
from pydantic import BaseModel, ConfigDict, Field

from app.core.deps import GeneratorDep, RequestIdDep
from app.core.errors import AppError
from app.schemas.extract import ErrorDetail, ErrorResponse
from app.schemas.generate import GeneratedImage, GenerateRequest, GenerateResult
from app.services.generation_queue import (
    QueueFullError,
    get_generation_queue,
)
from app.services.generator import (
    GenerationOOMError,
    GenerationTimeoutError,
    GeneratorUnavailableError,
    validate_defect_class,
)

router = APIRouter(prefix="/api/v1", tags=["generate"])

# /docs 里挂出统一错误形状（与 extract.py 的 ERROR_RESPONSES 同款，加 503/504）
ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "请求体不是合法 JSON（INVALID_JSON）"},
    422: {
        "model": ErrorResponse,
        "description": "字段校验失败（INVALID_INPUT / UNKNOWN_DEFECT_CLASS）",
    },
    503: {"model": ErrorResponse, "description": "生成器不可用（MODEL_NOT_LOADED / GPU_OOM）"},
    504: {"model": ErrorResponse, "description": "生成超时（GENERATION_TIMEOUT）"},
    500: {"model": ErrorResponse, "description": "服务端错误（INTERNAL_ERROR）"},
}

MODEL_HEADER = "X-Generator-Model"
ELAPSED_HEADER = "X-Elapsed-Ms"


@router.post(
    "/generate",
    response_model=GenerateResult,
    status_code=status.HTTP_200_OK,
    summary="生成缺陷图像（同步版）",
    description=(
        "按缺陷类别生成指定数量的工业缺陷图像（SD1.5 + LoRA）。\n\n"
        "**同步版**：请求挂起直到全部图生成完毕。大批量请等异步版（返回 task_id）。\n\n"
        "**参数上限是契约的一部分**：count ≤ 8、size ∈ {512, 768}、"
        "defect_class 必须在白名单内 —— 超界一律 422，服务端不会静默放宽。\n\n"
        "错误形状统一为 `{code, message, request_id, details}`（与 /extract 同一张错误码表 + 本端点新增码）。"
    ),
    responses=ERROR_RESPONSES,
)
async def generate_images(
    payload: GenerateRequest,
    generator: GeneratorDep,
    request_id: RequestIdDep,
    response: Response,
) -> GenerateResult:
    """生成入口。

    白名单校验在这里而不在 schema 的 Literal：类别清单是**配置**（随权重目录
    增长），Literal 会让加类别变成改契约。校验失败的 ValueError 在下一行
    立刻翻译成 AppError —— 服务层永远收不到表外类别。
    """
    try:
        validate_defect_class(payload.defect_class)
    except ValueError as exc:
        raise AppError(
            code="UNKNOWN_DEFECT_CLASS",
            message=str(exc),
            status_code=422,
            details=[ErrorDetail(field="defect_class", reason=str(exc))],
        ) from exc

    try:
        output = await generator.generate(
            defect_class=payload.defect_class,
            count=payload.count,
            size=payload.size,
            seed=payload.seed,
        )
    except GeneratorUnavailableError as exc:
        raise AppError(
            code="MODEL_NOT_LOADED",
            message="生成模型权重未加载或设备不可用，请稍后再试或联系维护者",
            status_code=503,
        ) from exc
    except GenerationOOMError as exc:
        # Retry-After 跟着异常走：异常一抛，路由自己的 response 写头代码不执行
        # （AppError.headers 由全局 handler 透传，见 errors.py）
        raise AppError(
            code="GPU_OOM",
            message="显存不足，请减小 count 或 size 后按 Retry-After 重试",
            status_code=503,
            headers={"Retry-After": "30"},
        ) from exc
    except GenerationTimeoutError as exc:
        raise AppError(
            code="GENERATION_TIMEOUT",
            message="生成超时，请减小 count 或 size 后重试",
            status_code=504,
        ) from exc

    response.headers[MODEL_HEADER] = output.model
    response.headers[ELAPSED_HEADER] = str(output.elapsed_ms)

    return GenerateResult(
        request_id=request_id,
        images=[GeneratedImage(**img) for img in output.images],  # type: ignore[arg-type]
        elapsed_ms=output.elapsed_ms,
        model=output.model,
    )


# ============================================================ 异步版（W8 周四）
#
# 与同步版共享同一份请求契约（GenerateRequest）与同一张错误码表；
# 差别只有提交方式（202 + task_id 立刻返回）与查询接口。
# 同步版保留不删：它仍是"契约 + 错误码"的参考实现，且小请求（count=1）
# 走同步版少一次轮询，两边各司其职。


@dataclass(slots=True)
class AsyncTaskAccepted:
    """202 响应体：提交成功但未完成。"""

    request_id: str
    task_id: str
    status: str
    reused: bool = False
    """true = 幂等命中已 succeeded 的相同参数，直接复用结果。"""


class AsyncTaskStatus(BaseModel):
    """任务查询响应体。status ∈ {queued, running, succeeded, failed}。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(description="任务 id")
    status: str = Field(description="queued / running / succeeded / failed")
    progress: str = Field(default="", description="人读进度（如 3/4）")
    result: dict[str, Any] | None = Field(
        default=None,
        description="succeeded 时：{images[], elapsed_ms, model}，形状与同步响应一致",
    )
    error: dict[str, Any] | None = Field(
        default=None,
        description="failed 时：{code, message}，错误码与同步版同源",
    )
    created_at: float = Field(description="任务创建时间（Unix 秒）")


@router.post(
    "/generate/async",
    response_model=AsyncTaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="提交异步生成任务（立刻返回 task_id）",
    description=(
        "请求体与 `/generate` 完全相同；区别是**不等待出图**，立刻返回 task_id。\n\n"
        "用 `GET /generate/tasks/{task_id}` 轮询状态（queued → running → succeeded）。\n\n"
        "**单 worker 串行**是刻意设计：GPU 独占，并发 worker 只会互相抢显存。\n\n"
        "幂等：相同 (defect_class, count, size, seed) 且已有成功结果时直接复用（reused=true），"
        "不重复占用 GPU。seed 不传则每次都是新任务。\n\n"
        "队列深度超限返回 429 + Retry-After。"
    ),
    responses={
        202: {"model": AsyncTaskAccepted, "description": "任务已接受"},
        422: {"model": ErrorResponse, "description": "字段校验失败（含 UNKNOWN_DEFECT_CLASS）"},
        429: {"model": ErrorResponse, "description": "队列已满（QUEUE_FULL）"},
        500: {"model": ErrorResponse, "description": "服务端错误"},
    },
)
async def generate_images_async(
    payload: GenerateRequest,
    generator: GeneratorDep,
    request_id: RequestIdDep,
    response: Response,
) -> AsyncTaskAccepted:
    """异步提交入口。

    白名单校验在**提交时**就做（不等 worker 跑）—— 表外类别应该立刻 422，
    而不是进了队列再 failed，那会让调用方轮询一圈才发现拼错了词。
    """
    try:
        validate_defect_class(payload.defect_class)
    except ValueError as exc:
        raise AppError(
            code="UNKNOWN_DEFECT_CLASS",
            message=str(exc),
            status_code=422,
            details=[ErrorDetail(field="defect_class", reason=str(exc))],
        ) from exc

    queue = get_generation_queue(generator)
    try:
        task, reused = await queue.submit(
            defect_class=payload.defect_class,
            count=payload.count,
            size=payload.size,
            seed=payload.seed,
        )
    except QueueFullError as exc:
        # W12 周一：Retry-After 用队列估算的动态值（"预计等待 X 秒"），
        # 不再回静态 60s —— 静态值让客户端按它退避后再吃一个 429。
        raise AppError(
            code="QUEUE_FULL",
            message=str(exc),
            status_code=429,
            headers={"Retry-After": str(exc.retry_after_s)},
        ) from exc

    response.headers["Location"] = f"/api/v1/generate/tasks/{task.task_id}"
    return AsyncTaskAccepted(
        request_id=request_id,
        task_id=task.task_id,
        status=task.status.value,
        reused=reused,
    )


@router.get(
    "/generate/tasks/{task_id}",
    response_model=AsyncTaskStatus,
    status_code=status.HTTP_200_OK,
    summary="查询异步生成任务",
    description=(
        "返回任务状态与结果。succeeded 后 result 的形状与同步响应一致；"
        "failed 时 error.code 与同步版错误码同源。\n\n"
        "任务结果保留 1 小时（TASK_TTL），过期清理 —— 不要把任务 id 当永久句柄。"
    ),
    responses={404: {"model": ErrorResponse, "description": "任务不存在或已过期清理"}},
)
async def get_generation_task(
    task_id: str,
    generator: GeneratorDep,
) -> AsyncTaskStatus:
    task = await get_generation_queue(generator).get(task_id)
    if task is None:
        raise AppError(
            code="TASK_NOT_FOUND",
            message="任务不存在或已过期清理",
            status_code=404,
        )
    return AsyncTaskStatus(
        task_id=task.task_id,
        status=task.status.value,
        progress=task.progress,
        result=task.result,
        error=task.error,
        created_at=task.created_at,
    )


__all__ = ["router"]
