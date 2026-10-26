"""LLM 调用的重试与熔断。

**这个模块回答两个问题：什么该重试、重试为什么必须带抖动。**

### 什么该重试

| 情况 | 重试? | 理由 |
| --- | --- | --- |
| `429` 限流 | **是** | 服务端只是"现在忙"，等一会儿就好 |
| `5xx` | **是** | 服务端临时故障，重试可能换到健康实例 |
| 超时 / 连接错误 | **是** | 网络抖动，重试常能成功 |
| `400/401/403/404` | **否** | 请求本身是错的（凭证/参数/模型名），重试只会重复错 N 次，浪费时间还烧配额 |
| `422` | **否** | 同上 |
| 编排层的 bug（`TypeError` 等） | **否** | 重试不会让 bug 消失 |

**4xx 一律不重试**是最重要的一条。凭证错了重试 5 次，只会让你多花 5 次钱
去买同一个 401，并把真正的问题（key 配错了）藏在"偶发失败"里。

### 为什么必须加抖动（jitter）

这是必答问题。假设 1000 个客户端同时收到 429，都按"等 1 秒再试"：

    第 1 秒结束的瞬间 → 1000 个请求同时涌来 → 服务端再次 429
    → 又都等 2 秒 → 再次同时涌来 ...

**没有抖动，所有客户端会同步成一个节拍器**，在服务端制造周期性尖峰 ——
本来只是"暂时繁忙"，被重试策略放大成"永久繁忙"。

加抖动后，每个客户端的等待时间被随机打散，尖峰被摊平成平滑的请求流。
**这是重试策略里唯一一个"不加会更糟"的细节**，不是优化项。

### 熔断（简单版）

一个 key 连续失败 N 次后暂停一段时间。目的不是"恢复服务"，而是**止血**：
key 被封 / 配额耗尽时，继续拿它重试只会让失败堆积，
还会拖慢所有请求（每个都要等超时）。

这里实现的是**最小可用**版本（连续失败计数 + 冷却期），
不是完整熔断器（半开探测、滑动窗口）。够用的判断标准是：
**它能让"某个 key 挂了"不再表现为"整个服务挂了"**。
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Final

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)

logger = logging.getLogger(__name__)

# 这些 HTTP 状态码**值得**重试。其余 4xx 一律不重试。
RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

DEFAULT_MAX_ATTEMPTS: Final[int] = 3
DEFAULT_BASE_DELAY: Final[float] = 0.5
DEFAULT_MAX_DELAY: Final[float] = 8.0
# 抖动比例：延迟在 [delay*(1-jitter), delay*(1+jitter)] 内均匀取值。
# 0.5 意味着"最多偏 50%"，足够打散同步节拍，又不会把 1s 拉成 1.5s。
DEFAULT_JITTER: Final[float] = 0.5


class RetryExhaustedError(Exception):
    """重试用尽后抛出，携带最后一次的原始异常，便于上层判断根因。"""

    def __init__(self, attempts: int, last: BaseException) -> None:
        super().__init__(f"重试 {attempts} 次后仍失败：{type(last).__name__}: {last}")
        self.attempts = attempts
        self.last = last


def is_retryable(exc: BaseException) -> bool:
    """判定一个异常是否值得重试。

    **顺序和类型都要小心**：`APITimeoutError` 是 `APIConnectionError` 的子类，
    而 `APIStatusError` 是独立分支。写成"先判 StatusError"能覆盖 4xx/5xx，
    再判连接/超时。反过来写会把 4xx 误判成可重试。
    """
    if isinstance(exc, APIStatusError):
        return exc.status_code in RETRYABLE_STATUS
    # 网络层：连接失败、超时、读超时 —— 都可重试
    return isinstance(exc, (APITimeoutError, APIConnectionError, TimeoutError, ConnectionError))


def compute_delay(
    attempt: int,
    *,
    base: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: float = DEFAULT_JITTER,
    rng: random.Random | None = None,
) -> float:
    """第 `attempt`（从 1 开始）次失败后应等待多久，**含抖动**。

    指数退避：`base * 2^(attempt-1)`，再乘一个 [1-jitter, 1+jitter] 的随机因子。

    为什么用乘法抖动而不是"加减一个固定随机值"：
    乘法保持相对比例 —— 基数是 0.5s 时抖 ±0.25s，基数是 8s 时抖 ±4s，
    两者都恰到好处。加法抖动要么对小延迟过猛（±0.5s 让 0.5s 变成 0 或 1s），
    要么对大延迟毫无作用。
    """
    raw = min(base * (2 ** (attempt - 1)), max_delay)
    if jitter <= 0:
        return raw
    generator = rng or random
    factor = 1.0 + generator.uniform(-jitter, jitter)
    # 下限给一点点值，避免抖动把延迟压到 0 → 立刻重试 → 打服务端
    return max(raw * factor, 0.05)


@dataclass
class RetryPolicy:
    """重试策略参数。做成对象而不是散落的常量，方便单测与按场景覆盖。"""

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    base_delay: float = DEFAULT_BASE_DELAY
    max_delay: float = DEFAULT_MAX_DELAY
    jitter: float = DEFAULT_JITTER
    # 测试里要能"不睡觉"地验证重试逻辑，所以把 sleep 注入进来。
    # 生产用 asyncio.sleep，测试用 no-op —— 否则一个退避测试要跑好几秒。
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    # 记录每次等待，便于断言"确实退避了、且带抖动"
    delays: list[float] = field(default_factory=list)


async def with_retry(
    func: Callable[[], Awaitable[Any]],
    *,
    policy: RetryPolicy | None = None,
    label: str = "llm-call",
) -> Any:
    """执行 `func`，可重试失败按指数退避 + 抖动重试。

    `func` 是**零参可调用**（用 lambda 或 partial 绑好参数），
    这样重试层完全不需要知道请求长什么样 —— 它只负责"再试一次"。

    最后一个 attempt 失败后不再 sleep，直接抛 `RetryExhaustedError`：
    多睡那几秒没有任何收益，只会拖慢错误上报。
    """
    policy = policy or RetryPolicy()
    last: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await func()
        except BaseException as exc:  # noqa: BLE001 - 先捕获再判定，不重试的会原样抛出
            last = exc
            retryable = is_retryable(exc)

            if not retryable:
                # **4xx / 编程错误：立刻失败，一次都不多试**
                logger.warning(
                    "%s 第 %d 次失败且不可重试（%s），直接抛出",
                    label,
                    attempt,
                    type(exc).__name__,
                )
                raise

            if attempt >= policy.max_attempts:
                break

            delay = compute_delay(
                attempt, base=policy.base_delay, max_delay=policy.max_delay, jitter=policy.jitter
            )
            policy.delays.append(delay)
            logger.warning(
                "%s 第 %d/%d 次失败（%s），%.2fs 后重试（含抖动）",
                label,
                attempt,
                policy.max_attempts,
                type(exc).__name__,
                delay,
            )
            await policy.sleep(delay)

    assert last is not None
    raise RetryExhaustedError(policy.max_attempts, last) from last


# ============================================================ 多 key + 熔断


@dataclass
class KeyState:
    """一个 key 的健康状态。"""

    key: str
    consecutive_failures: int = 0
    open_until: float = 0.0  # 冷却期结束的单调时钟时刻；0 表示未熔断
    total_failures: int = 0
    total_successes: int = 0

    def is_open(self, now: float | None = None) -> bool:
        """是否处于冷却（不可用）状态。"""
        moment = time.monotonic() if now is None else now
        return moment < self.open_until

    def masked(self) -> str:
        """日志里只出现掩码 —— key 绝不能进日志。

        **短 key 的处理是个真问题**：演练时用了 `key-A` 这种 5 字符的假 key，
        如果统一规则是"len<=12 就显示 ***"，那三个 key 在日志里全都变成 `***`，
        **完全看不出是哪一个被熔断了** —— 而这种日志恰恰是排查熔断的唯一线索。

        所以短 key 也走"保留前 2 后 2"的规则（长度 >= 4 时），
        真正短的（< 4）才整体打码。
        """
        if len(self.key) >= 12:
            return f"{self.key[:6]}…{self.key[-4:]}"
        if len(self.key) >= 4:
            return f"{self.key[:2]}…{self.key[-2:]}"
        return "***"


class KeyRotator:
    """多 key 轮换 + 连续失败熔断。

    **这是"单点故障不扩散"的最小实现**：
    一个 key 被限流/封禁，切到下一个继续服务，而不是整个服务 500。

    熔断逻辑刻意简单：连续失败 `threshold` 次 → 冷却 `cooldown` 秒。
    不做"半开探测"—— 那需要额外状态机和探测流量，而这里的失败模式
    （限流 / 配额耗尽 / key 失效）都只需要"等一会儿再试"。
    """

    def __init__(
        self,
        keys: list[str],
        *,
        failure_threshold: int = 3,
        cooldown: float = 30.0,
        clients: dict[str, AsyncOpenAI] | None = None,
    ) -> None:
        if not keys:
            raise ValueError("至少需要一个 key")
        self._states = [KeyState(key=k) for k in keys]
        self._cursor = 0
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self._clients = clients or {}

    @property
    def states(self) -> list[KeyState]:
        return self._states

    def available(self) -> list[KeyState]:
        """当前可用的 key（未熔断）。"""
        return [s for s in self._states if not s.is_open()]

    def pick(self) -> KeyState:
        """按轮转顺序挑一个可用 key。

        全都在冷却时抛 `RetryExhaustedError`：
        **不假装有可用 key** —— 让上层明确知道"现在没有任何 key 能用"，
        而不是硬挑一个必然失败的再重试几轮。
        """
        candidates = self.available()
        if not candidates:
            raise RetryExhaustedError(
                self.failure_threshold,
                RuntimeError("所有 key 都在冷却中"),
            )
        # 从游标位置开始找第一个可用
        for offset in range(len(self._states)):
            idx = (self._cursor + offset) % len(self._states)
            state = self._states[idx]
            if not state.is_open():
                self._cursor = (idx + 1) % len(self._states)
                return state

        # 理论上到不了这里（candidates 非空）
        return candidates[0]

    def record_success(self, state: KeyState) -> None:
        """成功：清空连续失败计数。熔断只针对**连续**失败，不是累计失败。"""
        state.consecutive_failures = 0
        state.total_successes += 1

    def record_failure(self, state: KeyState, *, retryable: bool = True) -> None:
        """失败：累计连续失败，达到阈值就熔断该 key。

        **不可重试的失败（4xx）直接熔断**：401 说明 key 本身废了，
        继续用它只会每次都失败，不如立刻换。</br>
        """
        state.consecutive_failures += 1
        state.total_failures += 1
        should_open = state.consecutive_failures >= self.failure_threshold or not retryable
        if should_open:
            state.open_until = time.monotonic() + self.cooldown
            logger.warning(
                "key %s 熔断 %.0fs（连续失败 %d 次，不可重试=%s）",
                state.masked(),
                self.cooldown,
                state.consecutive_failures,
                not retryable,
            )


__all__ = [
    "DEFAULT_BASE_DELAY",
    "DEFAULT_JITTER",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_DELAY",
    "RETRYABLE_STATUS",
    "KeyRotator",
    "KeyState",
    "RetryExhaustedError",
    "RetryPolicy",
    "compute_delay",
    "is_retryable",
    "with_retry",
]
