"""重试与熔断的测试（W1 周四）。全部离线。

核心断言（对应验收要求"验证重试确实发生且只重试 1 次"）：
- 429 → 200：调用成功，且**总共只发了 2 次请求**（1 次失败 + 1 次重试）
- 401：**一次都不重试**，直接抛
- 退避带抖动：同一 attempt 多次计算得到**不同**延迟
- 熔断：连续失败 N 次后该 key 不再被选中

**为什么不用 respx（实测踩到的坑）**：openai 3.37 内部用的是 `httpx2`
（一个独立分叉版本，`httpx2 is httpx` 为 False），而 respx 只 patch `httpx`，
所以 `@respx.mock` 对 SDK 发起的请求**完全无效** —— 路由一条都拦不到，
测试会真实去连网络然后失败。改用 SDK 官方支持的 `http_client=` 注入
`httpx2.MockTransport`：直接替换传输层，绕开版本差异，也不污染全局。
"""

from __future__ import annotations

import random
import time
from typing import Any

import httpx2
import pytest
from openai import APIStatusError, AsyncOpenAI

from app.core.llm.retry import (
    DEFAULT_MAX_ATTEMPTS,
    RETRYABLE_STATUS,
    KeyRotator,
    RetryExhaustedError,
    RetryPolicy,
    compute_delay,
    is_retryable,
    with_retry,
)

BASE_URL = "https://llm.test/v1"
CHAT_PATH = "/v1/chat/completions"


# ---------------------------------------------------------------- 工具
class _NoSleepPolicy(RetryPolicy):
    """不真的 sleep 的策略：测试退避逻辑但不花时间。"""

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        super().__init__(**kwargs)

        async def _noop(_delay: float) -> None:
            return None

        self.sleep = _noop


def _ok_body(content: str = "ok") -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class _Recorder:
    """记录每次请求，用来断言"到底发了几个请求"。"""

    def __init__(self, scripted: list[Any]) -> None:
        # scripted 里的元素：httpx2.Response 或 异常实例
        self.scripted = scripted
        self.requests: list[httpx2.Request] = []

    async def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        item = self.scripted.pop(0) if self.scripted else httpx2.Response(500)
        if isinstance(item, BaseException):
            raise item
        return item

    @property
    def call_count(self) -> int:
        return len(self.requests)


def _client(recorder: _Recorder) -> AsyncOpenAI:
    """把传输层换成 MockTransport。

    `max_retries=0` 关掉 SDK 自带的内部重试 —— 否则"重试了几次"会被 SDK 的
    隐式重试搅乱，测出来的次数不是我们这一层的。
    """
    transport = httpx2.MockTransport(recorder.handler)
    return AsyncOpenAI(
        api_key="sk-test",
        base_url=BASE_URL,
        max_retries=0,
        timeout=10,
        http_client=httpx2.AsyncClient(transport=transport),
    )


async def _call(client: AsyncOpenAI) -> str:
    resp = await client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}], max_tokens=8
    )
    return resp.choices[0].message.content or ""


# ============================================================ 1. 可重试判定


def test_retryable_status_set() -> None:
    """429/5xx 可重试；其余 4xx 不可重试。"""
    for code in (429, 500, 502, 503, 504):
        assert code in RETRYABLE_STATUS, code
    for code in (400, 401, 403, 404, 422):
        assert code not in RETRYABLE_STATUS, code


def _status_error(code: int) -> APIStatusError:
    request = httpx2.Request("POST", f"{BASE_URL}/chat/completions")
    response = httpx2.Response(code, request=request, json={"error": {"message": "x"}})
    return APIStatusError("boom", response=response, body=None)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (429, True),
        (500, True),
        (503, True),
        (401, False),
        (403, False),
        (404, False),
        (422, False),
    ],
)
def test_is_retryable_by_status(code: int, expected: bool) -> None:
    assert is_retryable(_status_error(code)) is expected


def test_is_retryable_network_and_timeout() -> None:
    assert is_retryable(TimeoutError()) is True
    assert is_retryable(ConnectionError()) is True
    # 编程错误绝不重试
    assert is_retryable(ValueError("bug")) is False
    assert is_retryable(KeyError("bug")) is False


# ============================================================ 2. 退避与抖动


def test_delay_grows_exponentially() -> None:
    """指数退避：0.5 → 1 → 2 → 4（关闭抖动看骨架）。"""
    delays = [compute_delay(n, base=0.5, max_delay=8.0, jitter=0.0) for n in range(1, 5)]
    assert delays == [0.5, 1.0, 2.0, 4.0]


def test_delay_is_capped() -> None:
    """上限封顶，避免第 10 次退避等到几小时。"""
    assert compute_delay(10, base=0.5, max_delay=8.0, jitter=0.0) == 8.0


def test_delay_has_jitter_between_bounds() -> None:
    """**必答问题的可执行形式**：同一 attempt 应产生**不同**延迟。

    如果是固定 1.0s，所有客户端会同步成节拍器，同时在服务端制造尖峰。
    """
    rng = random.Random(42)
    values = [compute_delay(2, base=0.5, max_delay=8.0, jitter=0.5, rng=rng) for _ in range(50)]

    assert len(set(values)) > 1, "抖动没生效：每次算出的延迟完全相同"
    # 1.0s 的 ±50% 区间是 [0.5, 1.5]，允许极小的浮点误差
    assert min(values) >= 0.5 - 1e-9
    assert max(values) <= 1.5 + 1e-9


def test_delay_never_zero() -> None:
    """抖动不能把延迟压到 0 —— 那等于立刻重试，会打服务端。"""
    rng = random.Random(1)
    values = [compute_delay(1, base=0.05, max_delay=8.0, jitter=0.9, rng=rng) for _ in range(100)]
    assert all(v >= 0.05 for v in values), min(values)


# ============================================================ 3. MockTransport 重试 429→200


@pytest.mark.anyio
async def test_429_then_200_retries_exactly_once() -> None:
    """**验收核心**：429 → 200，总共 2 次请求（1 失败 + 1 重试）。"""
    recorder = _Recorder(
        [
            httpx2.Response(429, json={"error": {"message": "rate limited", "type": "rate_limit"}}),
            httpx2.Response(200, json=_ok_body("ok")),
        ]
    )
    policy = _NoSleepPolicy(max_attempts=3)
    client = _client(recorder)
    try:
        text = await with_retry(lambda: _call(client), policy=policy, label="test")
    finally:
        await client.close()

    assert text == "ok"
    assert recorder.call_count == 2, (
        f"应恰好 2 次请求（1 失败 + 1 重试），实际 {recorder.call_count}"
    )
    assert len(policy.delays) == 1, "只应退避一次"


@pytest.mark.anyio
async def test_401_is_not_retried() -> None:
    """**4xx 一次都不重试**：凭证错了重试 N 次只是多花 N 次钱买同一个错误。"""
    recorder = _Recorder(
        [httpx2.Response(401, json={"error": {"message": "invalid key", "type": "auth"}})]
    )
    policy = _NoSleepPolicy(max_attempts=3)
    client = _client(recorder)
    try:
        with pytest.raises(APIStatusError) as exc_info:
            await with_retry(lambda: _call(client), policy=policy)
    finally:
        await client.close()

    assert exc_info.value.status_code == 401
    assert recorder.call_count == 1, f"401 不应重试，实际请求了 {recorder.call_count} 次"
    assert policy.delays == [], "不应有任何退避"


@pytest.mark.anyio
async def test_retry_exhausted_after_max_attempts() -> None:
    """持续 503：试满 max_attempts 后抛 RetryExhaustedError，且次数准确。"""
    recorder = _Recorder(
        [httpx2.Response(503, json={"error": {"message": "unavailable"}}) for _ in range(5)]
    )
    policy = _NoSleepPolicy(max_attempts=3)
    client = _client(recorder)
    try:
        with pytest.raises(RetryExhaustedError) as exc_info:
            await with_retry(lambda: _call(client), policy=policy)
    finally:
        await client.close()

    assert exc_info.value.attempts == 3
    assert recorder.call_count == 3
    # 3 次尝试只退避 2 次（最后一次失败后不再 sleep）
    assert len(policy.delays) == 2


@pytest.mark.anyio
async def test_500_then_success_counts_requests() -> None:
    """5xx 可重试，语义与 429 一致。"""
    recorder = _Recorder(
        [
            httpx2.Response(500, json={"error": {"message": "boom"}}),
            httpx2.Response(200, json=_ok_body("recovered")),
        ]
    )
    client = _client(recorder)
    try:
        text = await with_retry(lambda: _call(client), policy=_NoSleepPolicy())
    finally:
        await client.close()

    assert text == "recovered"
    assert recorder.call_count == 2


@pytest.mark.anyio
async def test_network_error_is_retried() -> None:
    """连不上也要重试（网络抖动是最常见的临时故障）。"""
    recorder = _Recorder(
        [httpx2.ConnectError("connection refused"), httpx2.Response(200, json=_ok_body("ok"))]
    )
    client = _client(recorder)
    try:
        text = await with_retry(lambda: _call(client), policy=_NoSleepPolicy())
    finally:
        await client.close()

    assert text == "ok"
    assert recorder.call_count == 2


@pytest.mark.anyio
async def test_two_retries_then_success() -> None:
    """429 → 429 → 200：应恰好 3 次请求、2 次退避。"""
    recorder = _Recorder(
        [
            httpx2.Response(429, json={"error": {"message": "rl"}}),
            httpx2.Response(429, json={"error": {"message": "rl"}}),
            httpx2.Response(200, json=_ok_body("finally")),
        ]
    )
    policy = _NoSleepPolicy(max_attempts=3)
    client = _client(recorder)
    try:
        text = await with_retry(lambda: _call(client), policy=policy)
    finally:
        await client.close()

    assert text == "finally"
    assert recorder.call_count == 3
    assert len(policy.delays) == 2


# ============================================================ 4. 多 key + 熔断


def test_rotation_cycles_through_keys() -> None:
    rotator = KeyRotator(["k1", "k2", "k3"])
    picked = [rotator.pick().key for _ in range(6)]
    assert picked == ["k1", "k2", "k3", "k1", "k2", "k3"]


def test_circuit_opens_after_threshold() -> None:
    """连续失败 N 次 → 该 key 熔断，不再被选中。"""
    rotator = KeyRotator(["bad", "good"], failure_threshold=2, cooldown=60)
    bad_state = rotator.states[0]

    rotator.record_failure(bad_state)
    assert not bad_state.is_open(), "1 次失败还不该熔断"

    rotator.record_failure(bad_state)
    assert bad_state.is_open(), "连续 2 次失败应熔断"

    picks = [rotator.pick().key for _ in range(4)]
    assert "bad" not in picks
    assert picks == ["good"] * 4


def test_success_resets_consecutive_failures() -> None:
    """熔断看的是**连续**失败：中间成功一次就重来。"""
    rotator = KeyRotator(["k"], failure_threshold=3, cooldown=60)
    state = rotator.states[0]

    rotator.record_failure(state)
    rotator.record_failure(state)
    rotator.record_success(state)
    assert state.consecutive_failures == 0

    rotator.record_failure(state)
    assert not state.is_open(), "计数已重置，不该熔断"


def test_non_retryable_failure_opens_immediately() -> None:
    """401（key 本身废了）应当立刻熔断，不必等连续 N 次。"""
    rotator = KeyRotator(["dead"], failure_threshold=5, cooldown=60)
    state = rotator.states[0]
    rotator.record_failure(state, retryable=False)
    assert state.is_open()


def test_all_keys_open_raises() -> None:
    """全部熔断时明确报错，而不是硬选一个必然失败的 key。"""
    rotator = KeyRotator(["a", "b"], failure_threshold=1, cooldown=60)
    for state in rotator.states:
        rotator.record_failure(state, retryable=False)

    assert rotator.available() == []
    with pytest.raises(RetryExhaustedError, match="所有 key"):
        rotator.pick()


def test_cooldown_expires_and_key_returns() -> None:
    """冷却期过后 key 应重新可用（否则熔断就等于永久下线）。

    用显式时间戳而不是真的 sleep：sleep 依赖系统时钟精度，
    在 CI 上会随机失败（实测踩过 —— sleep 0.06s 后判定仍是"冷却中"）。
    """
    rotator = KeyRotator(["k"], failure_threshold=1, cooldown=0.05)
    state = rotator.states[0]
    rotator.record_failure(state, retryable=False)
    assert state.is_open()

    # 直接把 open_until 拨到过去，等价于"冷却期已过"，但不依赖真实计时
    state.open_until = time.monotonic() - 1
    assert not state.is_open()
    assert rotator.pick().key == "k"


def test_key_is_masked_in_logs() -> None:
    """key 绝不能以明文出现在日志里。"""
    rotator = KeyRotator(["sk-1234567890abcdef"])
    masked = rotator.states[0].masked()
    assert "sk-1234567890abcdef" not in masked
    assert masked.startswith("sk-123")
    assert masked.endswith("cdef")


def test_short_keys_are_still_distinguishable() -> None:
    """**短 key 也必须能分辨**：否则「哪个 key 被熔断」这条日志就白记了。

    实测踩到：演练时用了 `key-A` / `key-B` / `key-C`，
    原来的规则（len<=12 一律打 `***`）让三条熔断日志长得一模一样。
    """
    masked = [s.masked() for s in KeyRotator(["key-A", "key-B", "key-C"]).states]
    assert len(set(masked)) == 3, f"三个 key 的掩码应互不相同，实际 {masked}"
    for full, shown in zip(("key-A", "key-B", "key-C"), masked, strict=True):
        assert shown != full


def test_very_short_key_is_fully_masked() -> None:
    """短到无法部分展示时整体打码，宁可不分辨也不泄露。"""
    assert KeyRotator(["abc"]).states[0].masked() == "***"


def test_default_max_attempts_is_bounded() -> None:
    """重试次数必须有硬上限，否则一次故障能拖很久。"""
    assert 1 <= DEFAULT_MAX_ATTEMPTS <= 5
