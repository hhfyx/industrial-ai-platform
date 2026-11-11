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
    """任务状态。合法迁移：queued → running → succeeded / failed。"""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# 合法迁移表：写下来而不是靠记忆 —— 状态机错了查一圈日志不如看这张表。
VALID_TRANSITIONS: Final[dict[TaskStatus, set[TaskStatus]]] = {
    TaskStatus.QUEUED: {TaskStatus.RUNNING},
    TaskStatus.RUNNING: {TaskStatus.SUCCEEDED, TaskStatus.FAILED},
    TaskStatus.SUCCEEDED: set(),
    TaskStatus.FAILED: set(),
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
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    """{code, message} —— 与同步版错误码同源（GPU_OOM 等），查询方拿同一张表。"""
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def transition(self, to: TaskStatus) -> None:
        """按迁移表跳状态。非法迁移直接抛 —— 状态机错了必须当场炸。"""
        if to not in VALID_TRANSITIONS[self.status]:
            raise RuntimeError(
                f"非法状态迁移 {self.status.value} → {to.value}（task {self.task_id}）"
            )
        self.status = to
        if to in (TaskStatus.SUCCEEDED, TaskStatus.FAILED):
            self.finished_at = time.time()

    def as_dict(self) -> dict[str, Any]:
        """查询接口的返回体（不含内部字段）。"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
        }


def idempotency_key(request: dict[str, Any]) -> Hashable | None:
    """幂等键。**seed=None 返回 None（不参与幂等）**——
    不传 seed 的语义是"每次要新的"，复用旧结果是错的。
    """
    seed = request.get("seed")
    if seed is None:
        return None
    return (request["defect_class"], request["count"], request["size"], seed)


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
        key = idempotency_key(req)
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
            return self._tasks.get(task_id)

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
                continue
            task.transition(TaskStatus.RUNNING)
            task.progress = "0/{count}".format(count=task.request["count"])
            started = time.perf_counter()
            try:
                # 推理丢线程池：SD 一跑 8–15 秒，阻塞事件循环会冻结状态查询接口
                output = await asyncio.to_thread(
                    _run_sync,
                    self._generator,
                    task.request,
                )
                elapsed = int((time.perf_counter() - started) * 1000)
                self._duration_samples.append(elapsed / 1000.0)
                task.transition(TaskStatus.SUCCEEDED)
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
            finally:
                self._queue.task_done()

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
            if t.status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED)
            and t.finished_at is not None
            and now - t.finished_at > TASK_TTL_S
        ]
        for tid in expired:
            task = self._tasks.pop(tid)
            key = idempotency_key(task.request)
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
