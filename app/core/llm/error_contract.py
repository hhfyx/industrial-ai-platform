"""工具错误的统一契约（W11 周一 · 错误分类与可恢复性）。

**这一层存在的理由：把"能不能重试"从模型手里拿回来。**

## 必答问题：retryable 为什么不能靠模型判断？

模型手上有的是**文本**，不是**元信息**。同一个"没查到"可以是参数写错
（改一个字就成立）、也可以是无权限（再试一万次还是拒绝），但两者在模型
眼里都只是"失败了"，区别全靠我们喂进去的措辞。于是它只能按语气猜：
写得急切它重试，写得平淡它放弃。**语气不是元信息。**

而工具这侧知道确定的答案：

| 谁判的 | 它知道什么 | 对应类 |
| --- | --- | --- |
| 校验器 | 改哪个字段就成立 | 参数错（可重试，改参即恢复）|
| 白名单 | 同一请求永远被拒 | 权限错（不可重试）|
| 执行环境 | 第几次、多久之后可以再试 | 超时（可重试 + 退避）|
| 查询结果 | 换对象才可能有结果 | 空结果（同参重试必然还空）|
| 未预期异常 | 不知道，但可以试一次 | 内部异常（有条件可重试）|

把这三件事编译成 `retryable` + `retry_after` 两个字段，循环就能用**代码**
执行纪律，而不是求模型自觉：

1. `retryable=False` 的错误，observation 直接降级成"别试了，换策略"；
2. `retryable=True` 的按 `retry_after` 退避；
3. 同类可重试错误超过预算，`apply_retry_budget` **自动**把 retryable 翻成
   False —— "超时 1 次后不再重试"是代码强制，不是提示词建议。

**一句话：hint 给模型看，retryable 给代码用。** 两个字段是两种读者的接口；
混成一个字段，必然有一方被迫读它不该读的东西（模型读布尔值读不懂意图，
代码读自然语言读不出真假）。

## 五类 × 恢复动作（本周的验收表）

| 类 | error_type | 可恢复吗 | 靠什么恢复 | retryable | retry_after |
| --- | --- | --- | --- | --- | --- |
| 参数错 | `invalid_arg` / `invalid_json` / `unknown_tool` | 能 | 改参数（按 hint 指出的字段）| True | — |
| 权限错 | `not_permitted` | **不能**（同请求永远被拒）| 换工具/换范围/如实告知无权限 | False | — |
| 超时 | `timeout` | 能 | 退避后重试，或缩小查询范围 | True | 2.0s |
| 空结果 | `not_found` / `empty_result` | 部分 | 换关键词/换对象；同参重试必然还空 | False | — |
| 内部异常 | `internal` | 有条件 | 重试 1 次；仍失败如实告知，不要编造 | True | — |
| （既有第六类）| `conflict` | 不能 | 效果已发生，直接向上报告 | False | — |

`conflict` 不是本周新增的分类，是 W8/W9 就有的幂等拦截（重复投递）。
它不属于上面五类的任何一类 —— 硬塞会污染"可恢复"的判断，所以单列。

## 空结果为什么有两个落点

"空"是不是故障，取决于工具是读还是写：

- **读工具**（检索/SQL）：0 行是**事实**，`ok=true` + `outcome="empty_result"` +
  hint。W9 的 T10 实测过：把它当错误（ok=false）模型会重试，当事实它才会
  如实转述"该批次没有这个指标"。所以这里的"空"必须留着 ok=true。
- **写工具**（报告/通知）：没有内容可写/可发是**无法交付**，
  `ok=false` + `error_type="empty_result"` + 同样的 hint 口径。

两个落点共享同一套恢复动作（换对象/去取内容，不是改参数格式），
因此统计时按 `category == "empty"` 合并 —— 见 `category_of()`。

## 结构定版

    失败: {ok: false, tool, error_type, message, hint, retryable, retry_after[, details]}
    成功: {ok: true,  tool, data, outcome: "success"|"empty_result"[, hint]}

`message` 是给**人**看的（日志、报错回溯）；`hint` 是给**模型**看的
（下一步该做什么，必须可执行）；`retryable`/`retry_after` 是给**代码**看的。
`details` 是可选明细（Pydantic 的 fields、可用工具清单），不属于定版六字段，
模型可以忽略，测试与看板靠它定位。
"""

from __future__ import annotations

import socket
import sqlite3
from dataclasses import dataclass
from typing import Any, Final, Literal

#: `error_type` 的封闭枚举。新增一类之前先回答"模型该做什么" ——
#: 答不上来就说明这一类不该存在，落回 internal。（W3 定下的规矩，W11 沿用）
ErrorType = Literal[
    "invalid_json",
    "invalid_arg",
    "unknown_tool",
    "not_found",
    "not_permitted",
    "empty_result",
    "conflict",
    "timeout",
    "internal",
]

ERROR_TYPES: Final[frozenset[str]] = frozenset(
    {
        "invalid_json",
        "invalid_arg",
        "unknown_tool",
        "not_found",
        "not_permitted",
        "empty_result",
        "conflict",
        "timeout",
        "internal",
    }
)

#: 五类（+ 既有的 conflict）。落表与统计都按这个维度，而不是按 error_type ——
#: error_type 是"具体发生了什么"，category 是"该做什么"，后者才是可靠性设计的对象。
Category = Literal["param", "permission", "timeout", "empty", "internal", "duplicate"]


@dataclass(frozen=True, slots=True)
class ErrorSpec:
    """一类的完整含义：谁该动、怎么动、代码能不能重试、多久后能。"""

    category: Category
    retryable: bool
    recovery: str
    hint: str
    retry_after: float | None = None


#: 分类表。**这里是唯一的真相来源** —— 统计、信封、文档都从这里派生，
#: 三处各写一份就一定会漂移（W10 的教训：口径分散比字段错更贵）。
ERROR_SPECS: Final[dict[str, ErrorSpec]] = {
    "invalid_json": ErrorSpec(
        category="param",
        retryable=True,
        recovery="change_args",
        hint="参数不是合法 JSON。重新生成一段**闭合**的 JSON 对象，不要带注释或尾部逗号。",
    ),
    "invalid_arg": ErrorSpec(
        category="param",
        retryable=True,
        recovery="change_args",
        hint="参数不符合工具签名。按 hint 里点名的字段改值后重新调用（不要原样重发）。",
    ),
    "unknown_tool": ErrorSpec(
        category="param",
        retryable=True,
        recovery="change_tool",
        hint="没有这个工具。改用 available_tools 里列出的工具名，或直接用自然语言回答。",
    ),
    "not_permitted": ErrorSpec(
        category="permission",
        retryable=False,
        recovery="change_approach",
        hint="没有这个权限，**同一请求重试多少次都会被拒**。"
        "换工具、换到允许的范围，或如实告知用户「这项数据无权访问」。",
    ),
    "not_found": ErrorSpec(
        category="empty",
        retryable=False,
        recovery="change_object",
        hint="查无此对象（批次/条款/记录）。换标识符或关键词再试；"
        "确认确实不存在时如实告知用户，不要编造。",
    ),
    "empty_result": ErrorSpec(
        category="empty",
        retryable=False,
        recovery="change_object",
        hint="工具执行成功但没有产出（0 行 / 无可写内容）。"
        "同参数重试必然还是空：换关键词或换对象，或如实告知「没有这项数据」。",
    ),
    "conflict": ErrorSpec(
        category="duplicate",
        retryable=False,
        recovery="stop_and_report",
        hint="该动作已经成功执行过（幂等拦截），**效果已经发生**。不要再重试，直接向上报告已完成。",
    ),
    "timeout": ErrorSpec(
        category="timeout",
        retryable=True,
        recovery="backoff_retry",
        hint="执行超时被中断。等 retry_after 秒后重试，或缩小范围（更窄的过滤条件）再试；"
        "连续超时说明这个查询本身太重，换策略。",
        retry_after=2.0,
    ),
    "internal": ErrorSpec(
        category="internal",
        retryable=True,
        recovery="retry_then_tell",
        hint="工具内部错误（不是你的参数问题）。可以重试 1 次；仍失败就如实告知用户"
        "「工具暂时不可用」，不要编造结果。",
    ),
}


def category_of(error_type: str) -> str:
    """error_type → 五类。未知类型按 internal 算（保守：可重试一次）。"""
    spec = ERROR_SPECS.get(error_type)
    return spec.category if spec else "internal"


def recovery_of(error_type: str) -> str:
    spec = ERROR_SPECS.get(error_type)
    return spec.recovery if spec else "retry_then_tell"


def is_retryable(error_type: str) -> bool:
    spec = ERROR_SPECS.get(error_type)
    return spec.retryable if spec else True


def normalize_error_type(kind: str | None) -> str:
    """把任意来源的 kind 规范到封闭枚举里。

    历史 kind（W2/W3 的 execution_error / city_not_found 等）在这里收口，
    执行层不需要各自维护映射表。
    """
    if kind in ERROR_TYPES:
        return str(kind)
    legacy = {
        "execution_error": "internal",
        "tool_timeout": "timeout",
        "city_not_found": "not_found",
        "invalid_arguments": "invalid_arg",
        "invalid_arguments_json": "invalid_json",
        "not_permitted": "not_permitted",
        "repeat_call": "conflict",
    }
    return legacy.get(kind or "", "internal")


# ============================================================ 信封构造


def tool_error(
    tool: str,
    error_type: str,
    message: str,
    *,
    hint: str | None = None,
    retry_after: float | None | object = ...,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造**失败**信封（定版六字段 + tool + 可选 details）。

    两条一致性规则（防止调用方随手写坏）：
    1. 不可重试的错误一律 `retry_after=None` —— "不能重试但可以 2 秒后再试"
       是自相矛盾的信号，模型会照着后者做；
    2. `retry_after` 只在可重试时才有意义，默认取分类表里的值。

    `details` 传 `None` 时整个键不出现（模型侧少一个噪音字段）。
    """
    kind = normalize_error_type(error_type)
    spec = ERROR_SPECS.get(kind) or ERROR_SPECS["internal"]
    retryable = spec.retryable
    if not retryable:
        backoff: float | None = None
    elif retry_after is ...:
        backoff = spec.retry_after
    else:
        backoff = None if retry_after is None else float(retry_after)  # type: ignore[arg-type]
    envelope: dict[str, Any] = {
        "ok": False,
        "tool": tool,
        "error_type": kind,
        "message": message,
        "hint": hint or spec.hint,
        "retryable": retryable,
        "retry_after": backoff,
    }
    if details:
        envelope["details"] = details
    return envelope


def tool_ok(
    tool: str,
    data: dict[str, Any],
    *,
    empty: bool = False,
    hint: str | None = None,
) -> dict[str, Any]:
    """构造**成功**信封。`empty=True` 时打上 `outcome="empty_result"`。

    成功信封不放 `retryable`：成功不需要重试，放一个 false 只会让
    "这个字段什么意思"变成新问题。空结果的恢复动作由 `hint` 承载。
    """
    envelope: dict[str, Any] = {
        "ok": True,
        "tool": tool,
        "data": data,
        "outcome": "empty_result" if empty else "success",
    }
    if empty:
        envelope["hint"] = hint or ERROR_SPECS["empty_result"].hint
    elif hint:
        envelope["hint"] = hint
    return envelope


# ============================================================ 重试预算（代码强制）


#: 同类可重试错误允许的自动重试次数。超了就把 retryable 翻成 False ——
#: 这是"能不能重试由代码定"的落地：模型想再试，observation 也不会再说"可以"。
MAX_RETRY_PER_ERROR: Final[int] = 1


def apply_retry_budget(
    envelope: dict[str, Any],
    counts: dict[str, int],
    *,
    limit: int = MAX_RETRY_PER_ERROR,
) -> dict[str, Any]:
    """按 (tool, error_type) 计数并**用尽后降级** retryable。

    `counts` 由调用方（循环/图）持有并跨步累积；本函数原地更新它。
    返回可能是新的 dict（降级时），也可能是原对象（无需降级时）。

    为什么按 (tool, error_type) 而不是按 (tool, args) 计数：模型最常见的
    错误动作是**换个措辞重试同一个坏工具**（参数微调但错误类型不变）。
    按参数计数拦不住它，按错误类型计数能。
    """
    if envelope.get("ok") or not envelope.get("retryable"):
        return envelope
    key = f"{envelope.get('tool')}|{envelope.get('error_type')}"
    counts[key] = counts.get(key, 0) + 1
    if counts[key] <= limit:
        return envelope

    downgraded = dict(envelope)
    downgraded["retryable"] = False
    downgraded["retry_after"] = None
    downgraded["retry_suppressed"] = True
    downgraded["hint"] = (
        f"重试预算已用尽：同类错误（{envelope.get('error_type')}）已重试 {limit} 次。"
        f"**不要再调用 {envelope.get('tool')} 重试同一动作** —— 换工具、"
        "换一个明显不同的做法，或如实告知用户做不到。\n" + str(envelope.get("hint", ""))
    )
    return downgraded


# ============================================================ 异常归类


def classify_exception(exc: BaseException) -> str:
    """把异常翻译成 error_type。**执行层不该各自写 except 分支的映射表。**

    - `PermissionError`（含写盘 EACCES）→ 权限错：换路径不成立，要换方案；
    - `TimeoutError`/`socket.timeout` → 超时：可退避重试；
    - `sqlite3.OperationalError` 里带 readonly/not authorized → 权限错
      （第五道闸 mode=ro 在引擎侧拒绝，这条路径以前会被误报成 internal）；
    - 其余 → internal。
    """
    if isinstance(exc, PermissionError):
        return "not_permitted"
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "timeout"
    if isinstance(exc, sqlite3.OperationalError):
        text = str(exc).lower()
        if "readonly" in text or "not authorized" in text or "permission" in text:
            return "not_permitted"
        if "locked" in text or "busy" in text:
            return "timeout"  # 锁等待 ≈ 暂时不可用，可退避重试
    return "internal"


__all__ = [
    "ERROR_SPECS",
    "ERROR_TYPES",
    "MAX_RETRY_PER_ERROR",
    "Category",
    "ErrorSpec",
    "ErrorType",
    "apply_retry_budget",
    "category_of",
    "classify_exception",
    "is_retryable",
    "normalize_error_type",
    "recovery_of",
    "tool_error",
    "tool_ok",
]
