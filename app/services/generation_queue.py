"""异步生成服务：任务模型 + 单 worker 队列 + 幂等复用 + 压力保护（W8 周四）。

## 为什么"单 worker"反而是正确设计（必答问题）

GPU 是**独占资源**：两个 worker 同时跑推理，CUDA 上下文各自占一份显存，
批处理互相抢卡 —— 实测表现不是"快一倍"，而是双双 OOM 或总吞吐反降。
所以这里与 W1 的 A7/A8（CPU 任务用并发有利）是**同一件事的反面**：
CPU 任务要并发，GPU 任务要**串行 + 排队**。

"并发高"的正确实现位置在**入口**：5 个请求立刻各拿一个 task_id 返回
（事件循环毫秒级处理完），推理在队列里排队串行执行 —— 接口不阻塞、
GPU 不打架，两头都是对的。

## 推理绝不阻塞事件循环

worker 把真实推理丢进 `asyncio.to_thread`（FakeGenerator 是毫秒级，
真实现的 SD 推理是 8–15 秒级 —— 不丢线程的话整个事件循环冻结，
连 `/tasks/{id}` 的状态查询都会卡死）。这与 embedding 层的
`asyncio.to_thread` 是同一条纪律。

## 状态机（合法迁移唯一）

    queued → running → succeeded
                     ↘ failed

没有其它箭头：不允许 running → queued（重试是新任务，task_id 不同），
不允许 failed → running（重试语义由调用方重新提交）。查询接口返回的
status 永远在这四个值里，前端不需要处理别的分支。

## 幂等复用

相同 `(defect_class, count, size, seed)` 且已有 succeeded 任务 → 直接返回
那个 task_id，不再进队列。省的是真金白银的 GPU 时间。
**seed=None 不参与幂等**：不传 seed 意味着"我每次要新的"，复用它反而是错的。

## 压力保护（拒绝与清理）

- 队列深度 > `MAX_QUEUE_DEPTH` → 429 + Retry-After（排队本身就是资源，
  无界队列等于把 OOM 换成内存泄漏）
- 每次提交后清理过期任务（TTL 内的任务保留，超时清理防内存泄漏）
  —— 清理挂在提交路径上而不是定时器：没有新请求就没有增长，
  定时器反而是在没有压力时白白跑。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from app.services.generator import (
    GenerationOOMError,
    GenerationTimeoutError,
    Generator,
    GeneratorUnavailableError,
)

logger = logging.getLogger(__name__)

# 队列深度上限。超过即 429：单 worker 串行 8 张 768px 约 2 分钟/任务，
# 深度 32 ≈ 1 小时排完 —— 排队超过一小时的任务不如让调用方稍后再来。
MAX_QUEUE_DEPTH: Final[int] = 32

# 任务保留时长（秒）。succeeded/failed 的结果查询窗口；过时清理。
TASK_TTL_S: Final[float] = 3600.0

# 429 的退避提示（秒）。**只作无样本时的保守兜底**：动态估算见 QueueFullError。
# 单 worker 串行，队尾大约每任务 2 分钟。
RETRY_AFTER_S: Final[int] = 60

# 平均任务时长的保守兜底（秒）。没有历史样本时用它估算"预计等待"。
# 取 SD 口径的上沿（8–15 秒/张 × 8 张 ≈ 2 分钟），宁可高估不多估 ——
# 高估让客户端晚一点回来（多等一会儿），低估让客户端白跑一趟（再吃一个 429）。
FALLBACK_TASK_SECONDS: Final[float] = 120.0

# 平均任务时长的样本窗口。滚动均值只看最近 N 个完结任务：
# fake/SD 切换、负载变化后旧样本会拖偏估算。
AVG_SAMPLE_WINDOW: Final[int] = 20


class TaskStatus(StrEnum):
    """任务状态。合法迁移：queued → running → succeeded / failed / cancelled。"""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


# 合法迁移表：写下来而不是靠记忆 —— 状态机错了查一圈日志不如看这张表。
# W12 周三：queued/running 增加 → CANCELLED（取消，见 request_cancel）。
VALID_TRANSITIONS: Final[dict[TaskStatus, set[TaskStatus]]] = {
    TaskStatus.QUEUED: {TaskStatus.RUNNING, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.SUCCEEDED: set(),
    TaskStatus.FAILED: set(),
    TaskStatus.CANCELLED: set(),
}


@dataclass(slots=True)
class GenerationTask:
    """一个异步生成任务（状态机的载体）。"""

    task_id: str
    request: dict[str, Any]
    """提交参数 {defect_class, count, size, seed}。幂等键由它派生。"""
    status: TaskStatus = TaskStatus.QUEUED
    progress: str = ""
    """人读的进度描述（"0/4"、"3/4"）。结构化到"已完成张数"是异步 v2 的事。"""
    images_done: int = 0
    """W12 周三：真实进度 = 已完成张数（worker 每张回写）。**不是假进度**：
    不做时间估算（GPU 任务时长大方差，估算必然撒谎），只报已发生的事实。"""
    queue_position: int | None = None
    """W12 周三：排第几位（None = 不在排队中，即 running/终态）。"""
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    """{code, message} —— 与同步版错误码同源（GPU_OOM 等），查询方拿同一张表。"""
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    trace_id: str | None = None
    """W13 周一：提交请求的 trace_id。**必须显式带进 worker**——worker 是
    长驻协程，不继承请求的 contextvars；不传的话异步任务的 span 会挂到
    worker 自己的（空的）trace 上，与提交请求断链。"""
    _cancel_requested: bool = field(default=False, repr=False)
    """协作式取消标记（与 AgentTask 同一机制）：worker 在每张图之间检查。"""

    def transition(self, to: TaskStatus) -> None:
        """按迁移表跳状态。非法迁移直接抛 —— 状态机错了必须当场炸。"""
        if to not in VALID_TRANSITIONS[self.status]:
            raise RuntimeError(
                f"非法状态迁移 {self.status.value} → {to.value}（task {self.task_id}）"
            )
        self.status = to
        if to in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            self.finished_at = time.time()

    def as_dict(self) -> dict[str, Any]:
        """查询接口的返回体（不含内部字段）。"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "progress": self.progress,
            "images_done": self.images_done,
            "queue_position": self.queue_position,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
        }


def idempotency_key(request: dict[str, Any], *, model_version: str = "") -> Hashable | None:
    """幂等键。**seed=None 返回 None（不参与幂等）**——
    不传 seed 的语义是"每次要新的"，复用旧结果是错的。

    ## W12 周二：model_version 进幂等键（计划原文的"补齐 model_version"）

    不加它的坑：换模型版本（fake → sd15-lora-v0 → sd15-lora-v1）后，
    相同参数的请求会命中**旧模型**生成的结果并复用 —— 调用方以为拿到的是
    新模型的产物。`model_version` 取 `generator.model_name`（Generator 协议
    的既有字段，每个实现必须声明），由 submit 在调用点传入，实现换版本时
    幂等键自动变 —— 与"配置指纹"（result_cache.py）是同一条设计：**派生
    数据的 key 必须绑定产生它的生产者版本**。
    """
    seed = request.get("seed")
    if seed is None:
        return None
    return (request["defect_class"], request["count"], request["size"], seed, model_version)


class GenerationQueue:
    """单 worker 异步生成队列。

    ## 惰性启动

    worker 在**第一次提交任务**时启动，而不是进程启动时 ——
    没有生成请求的进程（纯问答服务）不该白养一个 worker 循环。
    """

    def __init__(self, generator: Generator) -> None:
        self._generator = generator
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: dict[str, GenerationTask] = {}
        self._worker: asyncio.Task[None] | None = None
        self._keys: dict[Hashable, str] = {}
        """幂等索引：key → succeeded task_id。"""
        self._lock = asyncio.Lock()
        """保护 _tasks/_keys 的注册表操作（提交/查询来自不同请求协程）。"""
        self._duration_samples: deque[float] = deque(maxlen=AVG_SAMPLE_WINDOW)
        """近期任务耗时样本（秒）。worker 完结时落账，动态 Retry-After 用。"""

    # ------------------------------------------------------------------ 提交与查询

    async def submit(
        self, *, defect_class: str, count: int, size: int, seed: int | None
    ) -> tuple[GenerationTask, bool]:
        """提交任务。返回 (任务, 是否幂等命中)。

        命中已 succeeded 的相同参数 → 直接复用，不进队列（省 GPU）。
        队列深度超限 → 抛 `QueueFullError`（路由层映射 429）。
        """
        req = {"defect_class": defect_class, "count": count, "size": size, "seed": seed}
        # model_version 进幂等键（W12 周二）：换模型版本 → 旧结果不复用。
        # generator.model_name 是 Generator 协议字段，fake 与 sd 各自声明。
        key = idempotency_key(req, model_version=self._generator.model_name)
        async with self._lock:
            if key is not None:
                old_id = self._keys.get(key)
                if old_id is not None:
                    old = self._tasks.get(old_id)
                    if old is not None and old.status is TaskStatus.SUCCEEDED:
                        return old, True

            pending = sum(
                1
                for t in self._tasks.values()
                if t.status in (TaskStatus.QUEUED, TaskStatus.RUNNING)
            )
            if pending >= MAX_QUEUE_DEPTH:
                retry_after = self._estimate_retry_after(pending)
                raise QueueFullError(
                    f"生成队列已满（{pending} 个任务在排队），预计等待 {retry_after} 秒，请按 Retry-After 退避",
                    retry_after_s=retry_after,
                )

            task = GenerationTask(task_id=uuid.uuid4().hex[:16], request=req)
            # trace 贯穿（W13）：提交时捕获请求的 trace_id，worker 执行时用它
            # 开 trace——异步任务的观测归属提交它的那次请求。
            from app.core.tracing import get_trace_id

            task.trace_id = get_trace_id() or None
            self._tasks[task.task_id] = task
            if key is not None:
                self._keys[key] = task.task_id
            self._cleanup_expired()
            await self._queue.put(task.task_id)
            self._ensure_worker()
            return task, False

    def _estimate_retry_after(self, pending: int) -> int:
        """估算"预计等待 X 秒"（动态 Retry-After 的 X）。

        `pending × 平均任务时长`：单 worker 串行，队尾任务的等待就是
        前面所有任务的总时长。样本不足用保守兜底（见 FALLBACK_TASK_SECONDS）。
        至少 1 秒：Retry-After: 0 是"立刻重试"，对队列已满语义是谎言。
        """
        if self._duration_samples:
            avg = sum(self._duration_samples) / len(self._duration_samples)
        else:
            avg = FALLBACK_TASK_SECONDS
        return max(1, int(pending * avg))

    async def get(self, task_id: str) -> GenerationTask | None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is not None and task.status is TaskStatus.QUEUED:
                # W12 周三：队列位置 = 排在前面的 queued 任务数（0 = 下一个就轮到）。
                # 读路径顺带算而不是 worker 维护 —— 队列变化只在出队时发生，
                # 读时现算永远是对的，维护反而会忘更新。
                task.queue_position = sum(
                    1
                    for t in self._tasks.values()
                    if t.status is TaskStatus.QUEUED and t.created_at < task.created_at
                )
            return task

    async def request_cancel(self, task_id: str) -> GenerationTask | None:
        """W12 周三：请求取消。

        - queued → 直接 cancelled（还没占 GPU，无副作用要回滚）；
        - running → 协作式：标记 `_cancel_requested`，worker 在**每张图之间**
          检查（GPU 任务不可硬杀 —— CUDA 调用没有安全的取消点，硬杀等于
          把显存留在脏状态）。已生成的图保留在结果里，partial 语义如实。
        - 终态 → 原样返回（幂等 no-op，路由层决定 409）。

        取消的任务**不进幂等索引**（submit 里按终态判断）—— 重新提交同参数
        会拿到新任务，"取消"是调用方的决定，不该被幂等键挡住。
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if task.status is TaskStatus.QUEUED:
                task.transition(TaskStatus.CANCELLED)
                # 从等待队列里摘掉：worker 取到 None 会跳过（下面 _run 有分支）
                # asyncio.Queue 无法随机删除，靠 worker 侧状态检查过滤。
            elif task.status is TaskStatus.RUNNING:
                task._cancel_requested = True
            return task

    async def wait_idle(self, timeout: float = 10.0) -> None:
        """等队列排空（全部任务到达终态）。测试与优雅关闭用。

        **不是等 worker 退出**：worker 是常驻循环（等下一个任务），
        `wait_for(queue._worker)` 永远超时 —— 那是测试写错了对象。
        """
        await asyncio.wait_for(self._queue.join(), timeout=timeout)

    def _ensure_worker(self) -> None:
        """惰性启动 worker。worker 挂了（不该发生）在下一轮提交时重建。"""
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())

    # ------------------------------------------------------------------ worker

    async def _run(self) -> None:
        """单 worker 串行消费。**绝不并发**：GPU 独占，见模块 docstring。"""
        while True:
            task_id = await self._queue.get()
            task = self._tasks.get(task_id)
            if task is None:
                self._queue.task_done()
                continue
            # 取消的 queued 任务：跳过执行（request_cancel 已迁移终态）
            if task.status is not TaskStatus.QUEUED:
                self._queue.task_done()
                continue
            task.transition(TaskStatus.RUNNING)
            task.progress = "0/{count}".format(count=task.request["count"])
            started = time.perf_counter()
            # trace 贯穿（W13）：worker 是长驻协程，用它执行时恢复提交请求的
            # trace_id——异步生成任务的 span 归属到提交它的那次请求。
            from app.core.tracing import span as trace_span
            from app.core.tracing import trace_id_var

            trace_token = trace_id_var.set(task.trace_id or "")
            try:
                with trace_span(
                    "generate.task",
                    task_id=task_id,
                    defect_class=task.request.get("defect_class"),
                    count=task.request.get("count"),
                ):
                    await self._execute(task_id, task, started)
            finally:
                trace_id_var.reset(trace_token)
                self._queue.task_done()

    async def _execute(self, task_id: str, task: GenerationTask, started: float) -> None:
        """单任务执行（从 _run 抽出，逻辑原样）：推理、终态迁移、错误映射。

        抽成方法的唯一原因：trace span 要包住"整个任务的执行 + 终态"，
        而原 try/except 里有 continue 之外的完整控制流——包在 span 里
        需要它是一个可 await 的整体。
        """
        try:
            # 推理丢线程池：SD 一跑 8–15 秒，阻塞事件循环会冻结状态查询接口
            output = await asyncio.to_thread(
                _run_sync,
                self._generator,
                task.request,
            )
            elapsed = int((time.perf_counter() - started) * 1000)
            self._duration_samples.append(elapsed / 1000.0)
            # 取消请求在整批生成期间到达：已完成的部分如实保留
            if task._cancel_requested:
                task.transition(TaskStatus.CANCELLED)
                task.images_done = len(output.images)
                task.progress = f"{task.images_done}/{task.request['count']}（已取消）"
                logger.info("任务 %s 在执行中取消（已完成 %d 张）", task_id, task.images_done)
            else:
                task.transition(TaskStatus.SUCCEEDED)
                task.images_done = len(output.images)
                task.progress = f"{task.request['count']}/{task.request['count']}"
                task.result = {
                    "images": output.images,
                    "elapsed_ms": elapsed,
                    "model": output.model,
                }
                logger.info("异步生成完成 %s（%d ms）", task_id, elapsed)
        except (GenerationOOMError, GenerationTimeoutError, GeneratorUnavailableError) as exc:
            # 已知失败 → 结构化错误（错误码与同步版同源）
            # 失败样本也落账：OOM/超时任务占的真实 GPU 时长同样是
            # "后面排队的人要等的时间"，剔除它们会低估等待。
            self._duration_samples.append(time.perf_counter() - started)
            task.transition(TaskStatus.FAILED)
            task.error = {"code": _code_of(exc), "message": str(exc)}
            logger.warning("异步生成失败 %s：%s", task_id, _code_of(exc))
        except Exception as exc:  # noqa: BLE001 - worker 兜底：任务标记 failed，worker 不死
            task.transition(TaskStatus.FAILED)
            task.error = {"code": "INTERNAL_ERROR", "message": f"{type(exc).__name__}: {exc}"}
            logger.exception("异步生成未预期失败 %s", task_id)

    # ------------------------------------------------------------------ 清理

    def _cleanup_expired(self) -> None:
        """清理过 TTL 的已完结任务（提交路径上调用，见模块 docstring）。

        只删已完结的：queued/running 是活跃资源，删了查询就 404。
        幂等索引里指向被删任务的键一并移除（否则下次幂等命中查到 None 又提交，
        键泄漏）。
        """
        now = time.time()
        expired = [
            tid
            for tid, t in self._tasks.items()
            if t.status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            and t.finished_at is not None
            and now - t.finished_at > TASK_TTL_S
        ]
        for tid in expired:
            task = self._tasks.pop(tid)
            # 清理路径用同样的 model_version 派生键 —— 队列单例按 generator
            # 实例绑定（get_generation_queue），这里的 self._generator 与
            # submit 时是同一个，版本一致，键才能对上并正确移除。
            key = idempotency_key(task.request, model_version=self._generator.model_name)
            if key is not None and self._keys.get(key) == tid:
                del self._keys[key]
        if expired:
            logger.info("清理过期任务 %d 个", len(expired))


class QueueFullError(RuntimeError):
    """队列深度超限。路由层映射 429 + Retry-After。

    ## W12 周一：retry_after 变成动态值（"预计等待 X 秒"）

    旧版回静态 `RETRY_AFTER_S=60` —— 那是"拍脑袋能通过"的值：
    队列里实际排着 30 个任务时，60 秒后重试**必然**再吃一个 429。
    动态版按 `排在你前面的任务数 × 平均任务时长` 估算：

    - `pending`：本次拒绝时正在排队/执行的任务数（submit 已算好）；
    - `avg_task_s`：近期任务的平均耗时（成功/失败样本的滚动均值，worker 落账）；
      没有样本时用保守常数（fake 毫秒级 / SD 8–15 秒/张 → 单任务 2 分钟口径）。

    Retry-After 才有"诚实"可言：客户端按它退避后**大概率**真的能进队列。
    拒绝带原因（"队列已满，预计等待 X 秒"）同时满足周一计划的验收要求。
    """

    def __init__(self, message: str, *, retry_after_s: int) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


def _run_sync(generator: Generator, request: dict[str, Any]) -> Any:  # noqa: ANN401
    """线程侧入口：把 async generator 调用桥到 to_thread 的同步世界。

    `asyncio.to_thread` 只能跑同步函数，而 Generator.generate 是 async ——
    在工作线程里**再开一个事件循环**跑它（线程内无竞争，安全）。
    真实现换成同步内核时只需改这里一处。
    """
    return asyncio.run(
        generator.generate(
            defect_class=request["defect_class"],
            count=request["count"],
            size=request["size"],
            seed=request["seed"],
        )
    )


def _code_of(exc: Exception) -> str:
    """异常 → 同步版同源错误码。"""
    if isinstance(exc, GenerationOOMError):
        return "GPU_OOM"
    if isinstance(exc, GenerationTimeoutError):
        return "GENERATION_TIMEOUT"
    if isinstance(exc, GeneratorUnavailableError):
        return "MODEL_NOT_LOADED"
    return "INTERNAL_ERROR"


# 进程级单例（与 get_generator 同一条装配纪律）。
_queue: GenerationQueue | None = None


def get_generation_queue(generator: Generator) -> GenerationQueue:
    """取当前进程的队列单例（按 generator 实例绑定 —— 测试注入不同 fake 时各自独立）。"""
    global _queue
    if _queue is None or _queue._generator is not generator:
        _queue = GenerationQueue(generator)
    return _queue


def reset_generation_queue() -> None:
    """测试用：清掉单例（连同其中的任务表）。"""
    global _queue
    _queue = None


__all__ = [
    "MAX_QUEUE_DEPTH",
    "RETRY_AFTER_S",
    "TASK_TTL_S",
    "GenerationQueue",
    "GenerationTask",
    "QueueFullError",
    "TaskStatus",
    "VALID_TRANSITIONS",
    "get_generation_queue",
    "idempotency_key",
    "reset_generation_queue",
]
