"""Agent 任务注册表：HTTP 任务（W12 周三）与图执行之间的内存任务表。

## 为什么内存表（与项目一 GenerationQueue 同一决定，同一代价）

W12 周一冻结清单核对时确认过：项目一的生成任务状态在内存、重启即丢，
写进压测报告前提。项目二的 Agent 任务此刻做了同样的选择 ——
checkpointer（sqlite）里存的是**图的执行状态**（thread 级），而
"任务"这层（提交→执行→终态的生命周期）没有落库。重启丢任务 =
已知约束，与项目一对齐，等"已部署"周统一解决（都迁 sqlite）。

## 与 checkpointer 的分工

- 任务表（本文件）：生命周期 + 进度快照。GET /tasks/{id} 读它；
- checkpointer（langgraph）：图的中间状态（thread_id = task_id）。
  waiting_human 与 current_node 的**真相**在 checkpointer 的 get_state，
  任务表存的是最后一次同步过来的快照。

分两层的理由：get_state 是重查询（读 sqlite + 反序列化全 state），
进度轮询是高频操作 —— 轮询打任务表（内存 O(1)），只在"任务终态/
挂起事件"发生时同步一次快照。快照过时 <1s（执行协程每步结束回写）。

## 不用 GenerationQueue 的原因

不是偷懒：Agent 任务与生成任务的**执行模型不同** —— 生成是单 worker
串行队列（GPU 独占），Agent 任务是并发的 LLM 调用（无独占资源），
排队没有收益。共用的只有"提交返回 id + 状态机"这个形状，
把形状抽成基类是 W13 的事（两处实现先各自长，第二个实现出现时
才抽 —— 与 Extractor 协议同一时机哲学）。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

logger = logging.getLogger(__name__)

# 已完结任务的保留窗口（秒）。与项目一 TASK_TTL_S 同口径。
TASK_TTL_S: Final[float] = 3600.0


class AgentTaskStatus(StrEnum):
    """Agent 任务状态机。合法迁移：queued → running → succeeded / failed / cancelled。"""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


VALID_TRANSITIONS: Final[dict[AgentTaskStatus, set[AgentTaskStatus]]] = {
    AgentTaskStatus.QUEUED: {AgentTaskStatus.RUNNING, AgentTaskStatus.CANCELLED},
    AgentTaskStatus.RUNNING: {
        AgentTaskStatus.SUCCEEDED,
        AgentTaskStatus.FAILED,
        AgentTaskStatus.CANCELLED,
    },
    AgentTaskStatus.SUCCEEDED: set(),
    AgentTaskStatus.FAILED: set(),
    AgentTaskStatus.CANCELLED: set(),
}


@dataclass(slots=True)
class AgentTask:
    """一个 Agent 任务的注册表条目。"""

    task_id: str
    instruction: str
    model: str
    status: AgentTaskStatus = AgentTaskStatus.QUEUED
    # 进度快照（执行协程每步回写；真相在 checkpointer）
    current_node: list[str] = field(default_factory=list)
    steps_done: int = 0
    waiting_human: bool = False
    partial_answer: str | None = None
    degraded: bool = False
    # 终态载荷
    final_answer: str | None = None
    error: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    _cancel_requested: bool = field(default=False, repr=False)
    """协作式取消标记：执行协程每步开始时检查（asyncio 任务不可安全硬杀）。"""

    def transition(self, to: AgentTaskStatus) -> None:
        if to not in VALID_TRANSITIONS[self.status]:
            raise RuntimeError(
                f"非法状态迁移 {self.status.value} → {to.value}（task {self.task_id}）"
            )
        self.status = to
        if to in (
            AgentTaskStatus.SUCCEEDED,
            AgentTaskStatus.FAILED,
            AgentTaskStatus.CANCELLED,
        ):
            self.finished_at = time.time()

    def as_public_dict(self) -> dict[str, Any]:
        """GET /tasks/{id} 的返回体（周计划指定的六字段 + 生命周期字段）。"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "current_node": list(self.current_node),
            "steps": self.steps_done,
            "waiting_human": self.waiting_human,
            "partial_answer": self.partial_answer,
            "degraded": self.degraded,
            "final_answer": self.final_answer,
            "error": self.error,
            "created_at": self.created_at,
        }


class AgentTaskRegistry:
    """进程级任务表。提交、查询、取消、快照回写都在这里。"""

    def __init__(self) -> None:
        self._tasks: dict[str, AgentTask] = {}
        self._lock = asyncio.Lock()

    async def submit(self, *, instruction: str, model: str) -> AgentTask:
        task = AgentTask(task_id=uuid.uuid4().hex[:16], instruction=instruction, model=model)
        async with self._lock:
            self._cleanup_expired()
            self._tasks[task.task_id] = task
            return task

    async def get(self, task_id: str) -> AgentTask | None:
        async with self._lock:
            return self._tasks.get(task_id)

    async def request_cancel(self, task_id: str) -> AgentTask | None:
        """请求取消。running 任务走**协作式**取消（标记 + 执行协程自查），
        queued 任务直接迁移 cancelled。返回任务或 None（不存在）。"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if task.status is AgentTaskStatus.QUEUED:
                task.transition(AgentTaskStatus.CANCELLED)
            elif task.status is AgentTaskStatus.RUNNING:
                # 不在这里强行迁移：终态由执行协程确认（它知道自己停在哪一步，
                # 能如实把 partial_answer 留下）—— "取消"是请求，不是状态。
                task._cancel_requested = True
            # 终态任务：no-op（幂等），返回原样
            return task

    async def mark_running(self, task_id: str) -> AgentTask | None:
        """queued → running（提交路径用；本项目无排队收益，见模块 docstring）。"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is not None and task.status is AgentTaskStatus.QUEUED:
                task.transition(AgentTaskStatus.RUNNING)
            return task

    async def snapshot(self, task_id: str, **progress: Any) -> AgentTask | None:
        """执行协程回写进度快照（current_node / steps_done / waiting_human /
        partial_answer / degraded 的任意子集）。"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            for key, value in progress.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            return task

    async def finish(
        self, task_id: str, *, final_answer: str, degraded: bool, error: dict | None = None
    ) -> AgentTask | None:
        """终态回写。取消请求在终态确认时兑现。"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if task._cancel_requested and task.status is AgentTaskStatus.RUNNING:
                task.transition(AgentTaskStatus.CANCELLED)
                task.partial_answer = final_answer
                return task
            if error is not None:
                task.transition(AgentTaskStatus.FAILED)
                task.error = error
            else:
                task.transition(AgentTaskStatus.SUCCEEDED)
                task.final_answer = final_answer
                task.degraded = degraded
                task.partial_answer = None  # 有终答就不需要半成品
            return task

    def _cleanup_expired(self) -> None:
        """清过 TTL 的已完结任务（提交路径上调用，与项目一同款哲学）。"""
        now = time.time()
        expired = [
            tid
            for tid, t in self._tasks.items()
            if t.status
            in (
                AgentTaskStatus.SUCCEEDED,
                AgentTaskStatus.FAILED,
                AgentTaskStatus.CANCELLED,
            )
            and t.finished_at is not None
            and now - t.finished_at > TASK_TTL_S
        ]
        for tid in expired:
            del self._tasks[tid]


# 进程级单例
_registry: AgentTaskRegistry | None = None


def get_agent_registry() -> AgentTaskRegistry:
    global _registry
    if _registry is None:
        _registry = AgentTaskRegistry()
    return _registry


def reset_agent_registry() -> None:
    global _registry
    _registry = None


__all__ = [
    "TASK_TTL_S",
    "AgentTask",
    "AgentTaskRegistry",
    "AgentTaskStatus",
    "VALID_TRANSITIONS",
    "get_agent_registry",
    "reset_agent_registry",
]
