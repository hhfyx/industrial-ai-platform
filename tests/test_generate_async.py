"""异步生成的契约测试（W8 周四）。

验收逐条：
1. 任务模型：状态机合法迁移（非法迁移当场炸）
2. 5 并发：全部**立刻**拿到 task_id（不等待出图）；轮询看到 queued→running→succeeded
3. 幂等：相同 (class,count,size,seed) 命中复用；seed=None 不参与
4. 429：队列深度超限 + Retry-After
5. 清理：过 TTL 任务被清，活跃任务保留

全部离线：注入可控制延迟的假生成器（延迟给 worker 留出"可观察的中间态"）。
"""

from __future__ import annotations

import asyncio
import contextlib
import time

import httpx
import pytest

from app.core.deps import GeneratorDep
from app.main import app
from app.services.generation_queue import (
    MAX_QUEUE_DEPTH,
    VALID_TRANSITIONS,
    GenerationQueue,
    GenerationTask,
    QueueFullError,
    TaskStatus,
    idempotency_key,
)
from app.services.generator import (
    GenerationOutput,
    Generator,
    GeneratorUnavailableError,
)


class _SlowFakeGenerator:
    """可控制延迟的假实现：让"running 中间态"在测试里可观察。"""

    model_name = "slow-fake"

    def __init__(self, latency_s: float = 0.15) -> None:
        self._latency = latency_s

    async def generate(
        self, *, defect_class: str, count: int, size: int, seed: int | None
    ) -> GenerationOutput:
        await asyncio.sleep(self._latency)
        return GenerationOutput(
            images=[
                {
                    "path": f"data/generated/{defect_class}_{size}.png",
                    "size": size,
                    "seed": seed or 1,
                }
            ]
            * count,
            model=self.model_name,
            elapsed_ms=int(self._latency * 1000),
        )


# ============================================================ 1. 任务模型与状态机


def test_state_machine_has_no_exit_from_terminal() -> None:
    """succeeded/failed 是终态：没有出边。"""
    assert VALID_TRANSITIONS[TaskStatus.SUCCEEDED] == set()
    assert VALID_TRANSITIONS[TaskStatus.FAILED] == set()


def test_illegal_transition_raises() -> None:
    """非法迁移（queued → succeeded）当场抛 —— 状态机错了必须炸，不能静默。"""
    task = GenerationTask(
        task_id="t1", request={"defect_class": "划痕", "count": 1, "size": 512, "seed": 1}
    )
    with pytest.raises(RuntimeError, match="非法状态迁移"):
        task.transition(TaskStatus.SUCCEEDED)


def test_legal_transition_sets_finished_at() -> None:
    task = GenerationTask(task_id="t2", request={})
    task.transition(TaskStatus.RUNNING)
    task.transition(TaskStatus.SUCCEEDED)
    assert task.status is TaskStatus.SUCCEEDED
    assert task.finished_at is not None


def test_idempotency_key_none_when_seed_missing() -> None:
    """seed=None 不参与幂等 —— 不传 seed 的语义是"每次要新的"。"""
    assert idempotency_key({"defect_class": "划痕", "count": 1, "size": 512, "seed": None}) is None
    assert idempotency_key({"defect_class": "划痕", "count": 1, "size": 512, "seed": 7}) == (
        "划痕",
        1,
        512,
        7,
    )


# ============================================================ 2. 提交与并发


@pytest.mark.anyio
async def test_five_concurrent_submits_return_immediately() -> None:
    """**本周核心验收**：5 个请求全部立刻拿到 task_id —— 提交不等待出图。

    生成器延迟 0.5s × 5 任务串行 ≈ 2.5s 总推理；但提交本身必须在
    毫秒级完成。总耗时 < 1s 就证明事件循环没被推理阻塞。
    """
    queue = GenerationQueue(_SlowFakeGenerator(latency_s=0.5))
    started = time.perf_counter()

    results = await asyncio.gather(
        *[queue.submit(defect_class="划痕", count=1, size=512, seed=None) for _ in range(5)]
    )
    tasks = [(t, r) for t, r in results]
    submit_elapsed = time.perf_counter() - started

    assert submit_elapsed < 1.0, f"5 个提交花了 {submit_elapsed:.2f}s —— 推理阻塞了事件循环"
    assert all(not r for _, r in tasks)
    assert len({t.task_id for t, _ in tasks}) == 5, "5 个请求必须有 5 个不同的 task_id"
    # 等 worker 串行跑完（5 × 0.5s + 余量）
    await queue.wait_idle(timeout=10.0)
    total = time.perf_counter() - started
    assert total >= 2.5, "串行执行的总耗时应 ≈ 2.5s（这是特性不是缺陷）"
    for t, _ in tasks:
        assert t.status is TaskStatus.SUCCEEDED


@pytest.mark.anyio
async def test_status_transitions_observable() -> None:
    """轮询能看到 queued → running → succeeded 的完整流转。"""
    queue = GenerationQueue(_SlowFakeGenerator(latency_s=0.3))
    task, _ = await queue.submit(defect_class="凹坑", count=2, size=512, seed=3)
    assert task.status is TaskStatus.QUEUED

    # 提交后立刻查：worker 可能刚把它置为 running，也可能还没 —— 两态都合法，
    # 关键是不可能已经是 succeeded（推理至少 0.3s）
    snapshot = await queue.get(task.task_id)
    assert snapshot is not None
    assert snapshot.status in (TaskStatus.QUEUED, TaskStatus.RUNNING)

    await queue.wait_idle(timeout=3.0)
    done = await queue.get(task.task_id)
    assert done is not None
    assert done.status is TaskStatus.SUCCEEDED
    assert done.progress == "2/2"
    assert done.result is not None
    assert done.result["model"] == "slow-fake"
    assert len(done.result["images"]) == 2


@pytest.mark.anyio
async def test_worker_does_not_block_event_loop() -> None:
    """推理进行中，事件循环仍可响应（to_thread 纪律的执行验证）。"""
    queue = GenerationQueue(_SlowFakeGenerator(latency_s=0.4))
    await queue.submit(defect_class="划痕", count=1, size=512, seed=None)
    await asyncio.sleep(0.05)  # worker 已开跑

    # 事件循环上跑一个协程：如果 worker 阻塞了循环，这行要等 0.4s
    heartbeat = await asyncio.wait_for(asyncio.sleep(0.01), timeout=0.2)
    assert heartbeat is None  # 能到这行 = 循环没被冻结


# ============================================================ 3. 幂等复用


@pytest.mark.anyio
async def test_idempotent_hit_reuses_succeeded_result() -> None:
    """相同 (class,count,size,seed) 且已成功 → 直接复用，不进队列。"""
    queue = GenerationQueue(_SlowFakeGenerator(latency_s=0.05))
    first, reused1 = await queue.submit(defect_class="划痕", count=1, size=512, seed=42)
    await queue.wait_idle(timeout=2.0)
    assert first.status is TaskStatus.SUCCEEDED

    second, reused2 = await queue.submit(defect_class="划痕", count=1, size=512, seed=42)
    assert reused2 is True, "相同参数必须命中复用"
    assert second.task_id == first.task_id
    assert reused1 is False


@pytest.mark.anyio
async def test_no_seed_never_reuses() -> None:
    """seed=None 每次都是新任务 —— "要新的"语义不被幂等劫持。"""
    queue = GenerationQueue(_SlowFakeGenerator(latency_s=0.01))
    a, r1 = await queue.submit(defect_class="划痕", count=1, size=512, seed=None)
    await queue.wait_idle(timeout=2.0)
    b, r2 = await queue.submit(defect_class="划痕", count=1, size=512, seed=None)
    await queue.wait_idle(timeout=2.0)
    assert a.task_id != b.task_id
    assert not r1 and not r2


@pytest.mark.anyio
async def test_different_params_do_not_reuse() -> None:
    queue = GenerationQueue(_SlowFakeGenerator(latency_s=0.01))
    a, _ = await queue.submit(defect_class="划痕", count=1, size=512, seed=1)
    await queue.wait_idle(timeout=2.0)
    b, reused = await queue.submit(defect_class="划痕", count=2, size=512, seed=1)
    assert a.task_id != b.task_id
    assert not reused


@pytest.mark.anyio
async def test_queued_task_with_same_key_not_reused_yet() -> None:
    """相同参数的任务还在排队（未 succeeded）→ 提交新任务而不是复用半成品。"""
    queue = GenerationQueue(_SlowFakeGenerator(latency_s=0.4))
    a, reused = await queue.submit(defect_class="划痕", count=1, size=512, seed=9)
    assert reused is False
    # a 还在队列里没跑完
    b, reused2 = await queue.submit(defect_class="划痕", count=1, size=512, seed=9)
    assert b.task_id != a.task_id, "未完成的任务不能被复用"
    assert reused2 is False
    await queue.wait_idle(timeout=5.0)


# ============================================================ 4. 队列满（429）


@pytest.mark.anyio
async def test_queue_full_raises() -> None:
    """深度超 MAX_QUEUE_DEPTH → QueueFullError（路由映射 429 + Retry-After）。

    用"轮询释放标志"的假实现而不是裸 sleep：to_thread 的线程是**非守护线程**，
    测试结束时 pytest 要等它退完 —— 裸睡 600s 的挂死就是这么来的（实测）。
    finally 里置位释放标志，线程在百毫秒内退出。
    """

    class _Gate(Generator):
        model_name = "gated"

        def __init__(self) -> None:
            self.release = False

        async def generate(self, **kwargs):  # noqa: ANN003, ANN201
            while not self.release:
                await asyncio.sleep(0.05)
            return GenerationOutput(images=[], model=self.model_name, elapsed_ms=1)

    gen = _Gate()
    queue = GenerationQueue(gen)
    try:
        for _ in range(MAX_QUEUE_DEPTH):
            await queue.submit(defect_class="划痕", count=1, size=512, seed=None)
        with pytest.raises(QueueFullError):
            await queue.submit(defect_class="划痕", count=1, size=512, seed=None)
    finally:
        gen.release = True  # 释放 worker：让所有挂起任务跑完，线程退出
        if queue._worker is not None:  # noqa: SLF001
            with contextlib.suppress(asyncio.TimeoutError):
                await queue.wait_idle(timeout=10.0)


@pytest.mark.anyio
async def test_failed_tasks_do_not_count_toward_depth() -> None:
    """失败任务立刻终结，不占队列深度 —— 排空后深度回到 0。

    写法说明：提交满 → 等全部失败（排空）→ 再提交不再 429。
    一次性提交 37 个会在循环中途 429（前 32 个还没被 worker 消费完）——
    那测的是"提交比消费快"，不是"失败释放深度"。
    """

    class _AlwaysFail(Generator):
        model_name = "failing"

        async def generate(self, **kwargs):  # noqa: ANN003, ANN201
            raise GeneratorUnavailableError("没权重")

    queue = GenerationQueue(_AlwaysFail())
    for _ in range(MAX_QUEUE_DEPTH):
        await queue.submit(defect_class="划痕", count=1, size=512, seed=None)
    with pytest.raises(QueueFullError):
        await queue.submit(defect_class="划痕", count=1, size=512, seed=None)

    # 全部失败终结（排空）
    await queue.wait_idle(timeout=5.0)
    # 失败不占深度：现在能继续提交
    task, reused = await queue.submit(defect_class="划痕", count=1, size=512, seed=None)
    assert not reused
    await queue.wait_idle(timeout=5.0)
    done = await queue.get(task.task_id)
    assert done is not None and done.status is TaskStatus.FAILED


# ============================================================ 5. 过期清理


@pytest.mark.anyio
async def test_expired_tasks_cleaned_on_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    """过 TTL 的已完结任务被清；活跃任务保留；幂等索引同步删除。"""
    import app.services.generation_queue as gq

    monkeypatch.setattr(gq, "TASK_TTL_S", 0.05)
    queue = GenerationQueue(_SlowFakeGenerator(latency_s=0.01))
    a, _ = await queue.submit(defect_class="划痕", count=1, size=512, seed=100)
    await queue.wait_idle(timeout=2.0)
    assert a.status is TaskStatus.SUCCEEDED

    await asyncio.sleep(0.06)  # 越过 TTL
    await queue.submit(defect_class="锈蚀", count=1, size=512, seed=101)  # 触发清理

    assert await queue.get(a.task_id) is None, "过 TTL 任务必须被清"
    # 幂等键同步删除：再次提交同参数是新任务，不是查到 None 的泄漏键
    again, reused = await queue.submit(defect_class="划痕", count=1, size=512, seed=100)
    assert not reused and again.task_id != a.task_id


@pytest.mark.anyio
async def test_active_tasks_survive_cleanup() -> None:
    """queued/running 是活跃资源，清理不碰。"""
    queue = GenerationQueue(_SlowFakeGenerator(latency_s=0.3))
    task, _ = await queue.submit(defect_class="划痕", count=1, size=512, seed=None)
    queue._cleanup_expired()  # noqa: SLF001 - 直接调：验证清理的选择性
    assert await queue.get(task.task_id) is not None


# ============================================================ 6. worker 容错


@pytest.mark.anyio
async def test_worker_survives_unexpected_error() -> None:
    """任务抛怪异常 → 该任务 failed，worker 不死、后续任务照跑。"""
    calls = {"n": 0}

    class _BombThenOk(Generator):
        model_name = "bomb"

        async def generate(self, **kwargs):  # noqa: ANN003, ANN201
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("炸弹")
            return GenerationOutput(images=[], model=self.model_name, elapsed_ms=1)

    queue = GenerationQueue(_BombThenOk())
    a, _ = await queue.submit(defect_class="划痕", count=1, size=512, seed=None)
    b, _ = await queue.submit(defect_class="划痕", count=1, size=512, seed=None)
    await queue.wait_idle(timeout=3.0)

    assert a.status is TaskStatus.FAILED
    assert a.error == {"code": "INTERNAL_ERROR", "message": "ValueError: 炸弹"}
    assert b.status is TaskStatus.SUCCEEDED, "worker 必须活着继续消费"


# ============================================================ 7. HTTP 层


@pytest.fixture
async def client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.fixture(autouse=True)
def _fresh_queue_with_slow_fake():
    """注入慢速 fake + 全新队列；用例结束后复位装配（不影响其它测试）。"""
    from app.services.generation_queue import reset_generation_queue

    slow = _SlowFakeGenerator(latency_s=0.05)
    app.dependency_overrides[GeneratorDep.__metadata__[0].dependency] = lambda: slow
    reset_generation_queue()
    yield
    app.dependency_overrides.pop(GeneratorDep.__metadata__[0].dependency, None)
    reset_generation_queue()


@pytest.mark.anyio
async def test_async_endpoint_returns_202_immediately(client: httpx.AsyncClient) -> None:
    """202 + task_id + Location 头，立刻返回。"""
    response = await client.post(
        "/api/v1/generate/async",
        json={"defect_class": "划痕", "count": 1, "size": 512, "seed": 5},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["task_id"]
    assert body["status"] == "queued"
    assert body["reused"] is False
    assert response.headers["location"] == f"/api/v1/generate/tasks/{body['task_id']}"
    # 让 worker 跑完，避免任务悬挂影响其它用例
    await asyncio.sleep(0.3)


@pytest.mark.anyio
async def test_poll_until_succeeded(client: httpx.AsyncClient) -> None:
    """轮询：queued/running → succeeded，result 形状与同步响应一致。"""
    submit = await client.post(
        "/api/v1/generate/async",
        json={"defect_class": "划痕", "count": 1, "size": 768, "seed": 6},
    )
    task_id = submit.json()["task_id"]

    seen: set[str] = set()
    final = None
    for _ in range(50):
        poll = await client.get(f"/api/v1/generate/tasks/{task_id}")
        assert poll.status_code == 200
        body = poll.json()
        seen.add(body["status"])
        if body["status"] == "succeeded":
            final = body
            break
        await asyncio.sleep(0.02)

    assert final is not None, f"50 次轮询未完成，见过的状态：{seen}"
    assert final["result"]["images"][0]["size"] == 768
    assert "queued" in seen or "running" in seen, "至少要观察到中间态"


@pytest.mark.anyio
async def test_task_not_found_404(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/generate/tasks/no-such-id")
    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "TASK_NOT_FOUND"
    assert set(body) == {"code", "message", "request_id", "details"}


@pytest.mark.anyio
async def test_async_unknown_class_422_before_enqueue(client: httpx.AsyncClient) -> None:
    """表外类别在**提交时** 422 —— 不进队列，不让调用方轮询一圈才发现拼错。"""
    response = await client.post("/api/v1/generate/async", json={"defect_class": "异物"})
    assert response.status_code == 422
    assert response.json()["code"] == "UNKNOWN_DEFECT_CLASS"
