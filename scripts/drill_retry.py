"""drill_retry：断网 / 错误端点演练，观察重试与熔断的真实日志。

**演练的目的不是"证明代码能跑"，而是看清失败时的三件事：**

1. **重试了几次、间隔多久** —— 日志里应出现 `第 1/3 次失败 … 0.42s 后重试`
2. **4xx 一次都不重试** —— 换成一个返回 401 的端点，日志里应完全没有重试行
3. **熔断何时打开** —— 多 key 轮换时，坏 key 连续失败后会被排除

三种演练场景：

    A. 错误端口（连不上）      → 连接错误，应重试 N 次后抛 RetryExhaustedError
    B. 错误 host（DNS 失败）   → 同上，但异常类型不同
    C. 多 key 轮换 + 熔断      → 观察 key 被逐个踢出

运行：

    uv run python -X utf8 scripts/drill_retry.py
    uv run python -X utf8 scripts/drill_retry.py --scenario A
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from contextlib import suppress
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI  # noqa: E402

from app.core.llm.retry import (  # noqa: E402
    DEFAULT_MAX_ATTEMPTS,
    KeyRotator,
    RetryExhaustedError,
    RetryPolicy,
    is_retryable,
    with_retry,
)


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        with suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")


def _setup_logging() -> None:
    """把重试模块的 warning 打到控制台 —— 演练要看的就是这些行。"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("    %(levelname)-7s %(message)s"))
    logger = logging.getLogger("app.core.llm.retry")
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False


async def _call(client: AsyncOpenAI) -> str:
    resp = await client.chat.completions.create(
        model="drill-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
    )
    return resp.choices[0].message.content or ""


async def scenario_bad_endpoint(label: str, base_url: str) -> None:
    """连不上的端点：观察重试次数与退避间隔。"""
    print(f"\n=== {label} ===")
    print(f"  base_url = {base_url}")
    client = AsyncOpenAI(
        api_key="sk-drill-not-real",
        base_url=base_url,
        timeout=3,  # 演练要快，别真等 30 秒
        max_retries=0,  # 关掉 SDK 内部重试，只看我们自己那一层
    )
    # 真实退避（不注入 noop sleep），这样能观察到时间流逝
    policy = RetryPolicy(max_attempts=DEFAULT_MAX_ATTEMPTS, base_delay=0.3, max_delay=2.0)
    started = time.perf_counter()
    try:
        await with_retry(lambda: _call(client), policy=policy, label=label)
        print("  ✗ 意外成功（不该发生）")
    except RetryExhaustedError as exc:
        elapsed = time.perf_counter() - started
        print(f"  → 最终异常：{type(exc.last).__name__}")
        print(f"  → 尝试 {exc.attempts} 次，总耗时 {elapsed:.2f}s")
        print(f"  → 退避序列：{[round(d, 2) for d in policy.delays]}")
        print(f"  → 可重试判定：{is_retryable(exc.last)}（应为 True）")
        # 抖动验证：同一 attempt 的延迟不应完全相同
        if len(policy.delays) >= 2:
            same = len({round(d, 6) for d in policy.delays}) < len(policy.delays)
            print(f"  → 抖发生效：{not same}（同一 attempt 延迟应随机不同）")
    finally:
        await client.close()


async def scenario_non_retryable() -> None:
    """返回 401 的端点：**一次都不该重试**。"""
    print("\n=== C. 401（凭证错误）—— 不该重试 ===")
    client = AsyncOpenAI(
        # 指向真实网关但用假 key → 稳定 401
        api_key="sk-definitely-invalid-key-000",
        base_url="http://127.0.0.1:8326/v1",
        timeout=10,
        max_retries=0,
    )
    policy = RetryPolicy(max_attempts=DEFAULT_MAX_ATTEMPTS, base_delay=0.3)
    started = time.perf_counter()
    try:
        await with_retry(lambda: _call(client), policy=policy, label="401-drill")
        print("  ✗ 意外成功")
    except BaseException as exc:  # noqa: BLE001 - 演练脚本，任何异常都要打印
        elapsed = time.perf_counter() - started
        print(f"  → 异常：{type(exc).__name__}（不是 RetryExhaustedError 就对了）")
        print(f"  → 退避次数：{len(policy.delays)}（应为 0）")
        print(f"  → 总耗时：{elapsed:.2f}s（应很短）")
        print(f"  → is_retryable={is_retryable(exc)}（应为 False）")
        status = getattr(exc, "status_code", None)
        print(f"  → status_code={status}")
    finally:
        await client.close()


def scenario_circuit_breaker() -> None:
    """多 key 轮换 + 熔断：观察坏 key 被逐个踢出。

    这一格不需要真联网 —— 熔断是对"失败事件"的反应，
    用 `record_failure` 驱动就能看清楚状态机。
    """
    print("\n=== D. 多 key 轮换 + 熔断 ===")
    rotator = KeyRotator(["key-A", "key-B", "key-C"], failure_threshold=2, cooldown=60)

    print("  初始可用：", [s.masked() for s in rotator.available()])

    # B 连续失败两次 → 应被熔断
    key_b = rotator.states[1]
    rotator.record_failure(key_b)
    print(f"  B 失败 1 次 → 可用 {len(rotator.available())} 个（未熔断）")
    rotator.record_failure(key_b)
    print(f"  B 失败 2 次 → 可用 {len(rotator.available())} 个（B 已熔断）")

    picks = [rotator.pick().masked() for _ in range(4)]
    print(f"  连续挑选 4 次：{picks}")
    print(f"  → B 是否被排除：{'B' not in ' '.join(picks)}")

    # 一个 401 就直接熔断（key 本身废了）
    key_c = rotator.states[2]
    rotator.record_failure(key_c, retryable=False)
    print(f"  C 遇到 401（不可重试）→ 立即熔断：{key_c.is_open()}")
    print(f"  剩余可用：{[s.masked() for s in rotator.available()]}")

    # 全部熔断时明确报错
    rotator.record_failure(rotator.states[0], retryable=False)
    try:
        rotator.pick()
        print("  ✗ 应抛异常")
    except RetryExhaustedError as exc:
        print(f"  全部熔断 → 抛 RetryExhaustedError：{exc}")


async def main() -> int:
    _force_utf8_stdout()
    _setup_logging()

    parser = argparse.ArgumentParser(description="重试/熔断断网演练")
    parser.add_argument("--scenario", default="all", choices=["A", "B", "C", "D", "all"])
    args = parser.parse_args()

    print("重试与熔断演练。下面的 WARNING 行就是生产环境会看到的日志。")

    if args.scenario in ("A", "all"):
        # 端口不通：最典型的"服务没起来"
        await scenario_bad_endpoint("A. 端口不通（127.0.0.1:9）", "http://127.0.0.1:9/v1")

    if args.scenario in ("B", "all"):
        # DNS 解析失败：域名写错 / 内网无 DNS
        await scenario_bad_endpoint(
            "B. 域名不存在（.invalid 保留域）", "https://no-such-host.invalid/v1"
        )

    if args.scenario in ("C", "all"):
        await scenario_non_retryable()

    if args.scenario in ("D", "all"):
        scenario_circuit_breaker()

    print("\n=== 演练结论 ===")
    print("  1. 连接/DNS 失败会重试，退避带随机抖动（同一 attempt 的延迟每次都不同）")
    print("  2. 401 一次都不重试 —— 重试只会重复买同一个错误")
    print("  3. 熔断让坏 key 不再参与轮换，且 401 会立即熔断")
    print("  4. 全部 key 熔断时明确报错，而不是硬选一个必然失败的")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
