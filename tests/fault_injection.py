"""故障注入骨架（W11 周二 21:20–21:45）：`fail_at(n, mode)`。

**只用于测试与演练，不接生产路径。** 这一句必须说清楚：一个能按"第 n 次调用"
返回假错误的工具包装器，如果出现在生产装配里，就是一颗随时会让线上随机失败
的雷。所以它放在 `tests/` 下，`app/` 里没有它的任何引用 ——
这条边界靠目录位置表达，也靠本行注释提醒后来者。

## 为什么不把它做成 `executor` 的一个参数

那样生产装配就得传一个 `fault=None` 进去，读代码的人会以为"故障注入是
系统能力的一部分"，于是迟早有人把它接到线上做"混沌工程"。放在测试里、
只能由测试显式构造，就没有这个误读空间。

## 三种模式

| mode | 行为 | 模拟什么 |
| --- | --- | --- |
| `timeout` | 返回 `timeout` 信封（retryable=True）| 查询/写盘超时 |
| `empty` | 返回 `ok=true` + 空结果 | 检索无命中、0 行 |
| `internal_error` | 返回 `internal` 信封（retryable=True）| 工具内部炸了 |

`empty` 走 `ok=true` 是刻意的：读工具的 0 行是**事实**不是故障（W9 T10 的
教训），所以"空"模式注入的是"成功但没东西"，而不是"失败"。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Final

from app.core.llm.error_contract import normalize_error_type, tool_error, tool_ok

InjectMode = Callable[[int], str]


class FaultInjector:
    """包一层 executor，按"第几次调用"注入故障。

    用法（演练里）：

        base = execute_agent_tool
        tool = FaultInjector(base, failures={1: "timeout", 2: "timeout"})
        # 第 1、2 次调用返回 timeout，第 3 次起走真实工具

    计数**按工具名各记一份**：同一次任务里 `retrieve` 失败不该让 `notify`
    的第一次调用也失败 —— 那样注入的故障就没法归因了。
    """

    def __init__(
        self,
        inner: Callable[[str, str], Awaitable[dict[str, Any]]],
        *,
        failures: dict[int, str] | None = None,
        mode: InjectMode | None = None,
        tools: tuple[str, ...] | None = None,
    ) -> None:
        self.inner = inner
        self.failures = dict(failures or {})
        # mode(n) 的优先级高于静态 failures 表：演练里常用"永远失败"。
        self.mode = mode
        self.tools = tools
        self.calls: list[tuple[str, str, str | None]] = []  # (tool, args, injected)

    def _injected_mode(self, attempt: int) -> str | None:
        if self.mode is not None:
            return self.mode(attempt)
        return self.failures.get(attempt)

    async def __call__(self, name: str, raw_arguments: str) -> dict[str, Any]:
        if self.tools is not None and name not in self.tools:
            self.calls.append((name, raw_arguments, None))
            return await self.inner(name, raw_arguments)

        attempt = sum(1 for t, _, _ in self.calls if t == name) + 1
        injected = self._injected_mode(attempt)
        self.calls.append((name, raw_arguments, injected))
        if injected is None:
            return await self.inner(name, raw_arguments)
        return make_fault_envelope(name, injected, attempt=attempt)

    def attempts(self, name: str) -> int:
        return sum(1 for t, _, _ in self.calls if t == name)

    def injected(self, name: str) -> list[str | None]:
        return [m for t, _, m in self.calls if t == name]


def make_fault_envelope(tool: str, mode: str, *, attempt: int = 1) -> dict[str, Any]:
    """按模式造一个**与真实工具同形**的信封。

    同形很重要：只有信封同形，下游（wrap_observation / 判定 / 埋点）
    走的才是真实路径。如果注入器返回一个简化 dict，那演练验证的就是
    注入器自己，而不是系统的错误处理。
    """
    if mode == "timeout":
        return tool_error(tool, "timeout", f"[注入] 第 {attempt} 次调用超时")
    if mode == "internal_error":
        return tool_error(tool, "internal", f"[注入] 第 {attempt} 次调用内部异常")
    if mode == "empty":
        data: dict[str, Any] = {"query": "注入", "chunks": [], "count": 0, "rows": []}
        return tool_ok(tool, data, empty=True)
    if mode == "not_permitted":
        return tool_error(tool, "not_permitted", f"[注入] 第 {attempt} 次调用无权限")
    if mode == "invalid_arg":
        return tool_error(tool, "invalid_arg", f"[注入] 第 {attempt} 次调用参数非法")
    raise ValueError(f"未知注入模式：{mode!r}（{sorted(INJECT_MODES)}）")


INJECT_MODES: Final[frozenset[str]] = frozenset(
    {"timeout", "internal_error", "empty", "not_permitted", "invalid_arg"}
)


def normalize_mode(mode: str) -> str:
    """把口语化的模式名规范到注入器认识的枚举（演练脚本用得上）。"""
    if mode in INJECT_MODES:
        return mode
    kind = normalize_error_type(mode)
    mapping = {
        "timeout": "timeout",
        "internal": "internal_error",
        "empty_result": "empty",
        "not_permitted": "not_permitted",
        "invalid_arg": "invalid_arg",
    }
    if kind not in mapping:
        raise ValueError(f"模式 {mode!r} 无法映射到注入模式")
    return mapping[kind]


__all__ = [
    "INJECT_MODES",
    "FaultInjector",
    "InjectMode",
    "make_fault_envelope",
    "normalize_mode",
]
