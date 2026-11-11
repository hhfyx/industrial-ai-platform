"""`/api/v2/agent-tasks`：Agent 任务的服务化接口（W12 周三 09:00–10:00）。

## 周三验收的三件事，各自的语义

| 端点 | 验收点 | 关键语义 |
| --- | --- | --- |
| `POST /agent-tasks` | 接口可用 | 提交 → 202 + task_id，执行协程后台跑图 |
| `GET /agent-tasks/{id}` | 六字段进度 | **不存在的 id 返回 404**，不是 200+空 |
| `DELETE /agent-tasks/{id}` | 可取消 | running = 协作式取消（标记，执行协程确认终态）；queued = 直接 cancelled |

## 六字段进度（周计划原文）

`{status, current_node, steps, waiting_human, partial_answer, degraded}`
—— 与 `scripts/query_graph_state.py` 的 `query_progress()` 同源同义
（current_node/steps_done/waiting_human/partial），degraded 来自 W11 的
预算优雅退出。真相在 checkpointer 的 get_state，任务表存执行协程
每步回写的快照（内存 O(1) 轮询，重查询不打 sqlite）。

## 执行协程怎么跑图

`build_agent_graph(client)` 编译图（当前无 checkpointer —— interrupt
HTTP 化是 W13 的活，本接口先交付"提交→跑→查进度→取消"）。
client 走 `get_llm_client`（与 CLI/评测同一装配）。执行协程里每步
结束后调 `registry.snapshot()` 回写；图状态可从 `result_from_state`
组装的轨迹里拿 steps/final_answer/degraded。

图没有"每步回调"钩子，快照从哪来：**轮询**。执行协程用
`asyncio.create_task` 跑图，同时起一个看护协程每 0.5s 从
`app.get_state`（无 checkpointer 时从任务的中间变量不可行）——
所以这里改成：**把 steps 增量写进任务表**的轻量做法是在图外包装
executor（`execute_agent_tool` 的入口在图内闭包，包装点选择在
`executor` 参数 —— 图支持注入自定义 executor，W9 的注入测试正是走这里）。
executor 包装是每步必经之路，快照由它回写，零轮询。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import get_settings
from app.core.deps import RequestIdDep
from app.core.errors import AppError
from app.core.llm.agent_graph import build_agent_graph, result_from_state
from app.core.llm.agent_registry import execute_agent_tool
from app.core.llm.client import get_llm_client
from app.core.security import require_api_key
from app.schemas.extract import ErrorResponse
from app.services.agent_tasks import (
    AgentTaskRegistry,
    AgentTaskStatus,
    get_agent_registry,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v2",
    tags=["v2-agent"],
    dependencies=[Depends(require_api_key)],
)

ERROR_RESPONSES = {
    404: {"model": ErrorResponse, "description": "任务不存在或已过期清理（TASK_NOT_FOUND）"},
    409: {"model": ErrorResponse, "description": "任务已完结，取消无效（TASK_FINISHED）"},
    422: {"model": ErrorResponse, "description": "字段校验失败（INVALID_INPUT）"},
    500: {"model": ErrorResponse, "description": "服务端错误（INTERNAL_ERROR）"},
}

# 取消检查间隔（秒）：executor 包装层每步调用时顺带检查标记，这里是
# 步间 sleep（协作式取消的退让窗口，模拟真实工具调用的 IO 间隙）。
CANCEL_CHECK_S = 0.0


class AgentTaskSubmitRequest(BaseModel):
    """提交体。instruction 是任务原文（与 20 任务集的 instruction 同字段）。"""

    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=2, max_length=2000, description="任务指令")
    model: str | None = Field(default=None, description="模型名；不传用配置默认")


class AgentTaskAccepted(BaseModel):
    task_id: str
    status: str
    location: str = Field(description="查询路径（Location 头同值）")


class AgentTaskView(BaseModel):
    """进度查询响应体：周计划的六字段 + 生命周期字段。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: str = Field(description="queued / running / succeeded / failed / cancelled")
    current_node: list[str] = Field(default_factory=list, description="当前/下一节点")
    steps: int = Field(default=0, description="已完成步数")
    waiting_human: bool = Field(default=False, description="是否挂起等人工批复")
    partial_answer: str | None = Field(default=None, description="半成品答案（挂起/预算退出时）")
    degraded: bool = Field(default=False, description="预算退出/降级收场")
    final_answer: str | None = Field(default=None, description="终答（succeeded 时）")
    error: dict[str, Any] | None = Field(default=None, description="failed 时的结构化错误")
    created_at: float


class AgentTaskCancelView(BaseModel):
    task_id: str
    status: str
    cancelled: bool = Field(description="是否已到 cancelled 终态（running 时为 false，等协程确认）")


async def _run_agent_task(
    registry: AgentTaskRegistry, task_id: str, instruction: str, model: str
) -> None:
    """后台执行协程：建图 → 跑到终态 → 回写。异常必须被接住转 FAILED ——
    后台任务的异常没人 await，静默吞掉会让任务永远停在 running。"""
    registry_task = await registry.get(task_id)
    if registry_task is None:
        return
    settings = get_settings()
    client = get_llm_client(settings)

    # executor 包装：每步必经之路，回写快照 + 检查取消标记。
    base_executor = execute_agent_tool
    current_step = {"no": 0}

    async def instrumented_executor(tool: str, args: dict) -> dict:
        # 协作式取消：步开始时检查标记，请求了就抛出以中断图执行
        task = await registry.get(task_id)
        if task is not None and task._cancel_requested:
            raise asyncio.CancelledError("cancel requested")
        current_step["no"] += 1
        await registry.snapshot(task_id, current_node=["tools"], steps_done=current_step["no"])
        result = await base_executor(tool, args)
        return result

    try:
        app = build_agent_graph(client, executor=instrumented_executor)
        # messages 初值与评测 harness（run_eval_agent_graph.py）同一形状：
        # system prompt + user instruction —— 图的 model_step 消费 messages，
        # 不给它就 400 "messages must be a non-empty array"。
        from app.core.llm.react_loop import REACT_SYSTEM_PROMPT

        state = await app.ainvoke(
            {
                "instruction": instruction,
                "model": model,
                "messages": [
                    {"role": "system", "content": REACT_SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                ],
            },
            config={"recursion_limit": 50},
        )
        trace = result_from_state(task_id, "service", model, instruction, state)
        await registry.finish(
            task_id,
            final_answer=trace.get("final_answer", ""),
            degraded=bool(
                trace.get("budget_exit") or state.get("stopped_reason", "completed") != "completed"
            ),
            error=None,
        )
    except asyncio.CancelledError:
        await registry.finish(
            task_id,
            final_answer="（已取消，执行到此步为止）",
            degraded=True,
        )
    except Exception as exc:  # noqa: BLE001 - 后台任务兜底：标记 failed，不让任务停在 running
        logger.exception("Agent 任务执行失败 %s", task_id)
        await registry.finish(
            task_id,
            final_answer="",
            degraded=False,
            error={"code": "INTERNAL_ERROR", "message": f"{type(exc).__name__}: {exc}"},
        )


@router.post(
    "/agent-tasks",
    response_model=AgentTaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="提交 Agent 任务（异步）",
    description=(
        "提交一条 Agent 任务（ReAct 图执行），立刻返回 task_id。\n\n"
        "用 `GET /agent-tasks/{task_id}` 轮询六字段进度；"
        "`DELETE /agent-tasks/{task_id}` 请求取消（协作式）。\n\n"
        "**单实例内存任务表**：重启丢任务状态（与项目一同款约束，"
        "写入部署说明）。"
    ),
    responses={422: {"model": ErrorResponse, "description": "字段校验失败"}},
)
async def submit_agent_task(
    payload: AgentTaskSubmitRequest,
    request_id: RequestIdDep,
    response: Response,
) -> AgentTaskAccepted:
    registry = get_agent_registry()
    model = payload.model or get_settings().llm.model
    task = await registry.submit(instruction=payload.instruction, model=model)
    await registry.mark_running(task.task_id)
    asyncio.get_running_loop().create_task(
        _run_agent_task(registry, task.task_id, payload.instruction, model)
    )
    location = f"/api/v2/agent-tasks/{task.task_id}"
    response.headers["Location"] = location
    return AgentTaskAccepted(task_id=task.task_id, status=task.status.value, location=location)


@router.get(
    "/agent-tasks/{task_id}",
    response_model=AgentTaskView,
    status_code=status.HTTP_200_OK,
    summary="查询 Agent 任务进度",
    description=(
        "返回六字段进度：status / current_node / steps / waiting_human / "
        "partial_answer / degraded。\n\n"
        "**不存在的 id 返回 404**（TASK_NOT_FOUND），不是 200+空 —— "
        "轮询方按状态码分支，200+空会把'任务没了'伪装成'还没开始'。"
    ),
    responses=ERROR_RESPONSES,
)
async def get_agent_task(task_id: str) -> AgentTaskView:
    task = await get_agent_registry().get(task_id)
    if task is None:
        raise AppError(
            code="TASK_NOT_FOUND",
            message="任务不存在或已过期清理",
            status_code=404,
        )
    public = task.as_public_dict()
    return AgentTaskView(**public)


@router.delete(
    "/agent-tasks/{task_id}",
    response_model=AgentTaskCancelView,
    summary="取消 Agent 任务",
    description=(
        "queued → 直接 cancelled；running → 协作式取消（下一个工具步边界退出，"
        "终态由执行协程确认并保留 partial_answer）；已完结 → 409（幂等语义的"
        "显式拒绝：对已完结任务返回 200 会让调用方误以为取消生效了）。"
    ),
    responses=ERROR_RESPONSES,
)
async def cancel_agent_task(task_id: str) -> AgentTaskCancelView:
    task = await get_agent_registry().request_cancel(task_id)
    if task is None:
        raise AppError(
            code="TASK_NOT_FOUND",
            message="任务不存在或已过期清理",
            status_code=404,
        )
    if task.status is AgentTaskStatus.CANCELLED:
        # 已取消的任务再取消：要么上次请求的幂等回显（cancelled=True），要么
        # 显式 409。这里选**幂等回显**：DELETE 的目标状态已达成，200+cancelled
        # 比 409 更诚实 —— 409 说"任务已完结，取消无效"，但取消明明生效了。
        # 与生成面（generate.py）的 409 不同：那边 SUCCEEDED/FAILED 时取消
        # 确实没生效。两个面的语义都由"取消是否真的达成了目标状态"决定。
        return AgentTaskCancelView(task_id=task.task_id, status=task.status.value, cancelled=True)
    if task.status in (AgentTaskStatus.SUCCEEDED, AgentTaskStatus.FAILED):
        # 已成功/已失败：取消没赶上，显式拒绝（200 会让调用方误以为取消生效）
        raise AppError(
            code="TASK_FINISHED",
            message=f"任务已完结（{task.status.value}），取消无效",
            status_code=409,
        )
    return AgentTaskCancelView(
        task_id=task.task_id,
        status=task.status.value,
        cancelled=task.status is AgentTaskStatus.CANCELLED,
    )


__all__ = ["router"]
