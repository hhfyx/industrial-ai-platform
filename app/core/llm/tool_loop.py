"""单轮工具调用闭环：请求 → tool_calls → 执行 → 回灌 → 二次请求。

**必答问题的答案在这里体现得最清楚。**

整个 `run_tool_loop` 里没有任何一行是模型执行了函数。模型做且只做两件事：
① 返回 `tool_calls`（"我想调 get_weather，参数是 {city:北京}"）
② 收到结果后，用自然语言总结。

`TOOL_REGISTRY[...](**args)` 这一行是**我方代码**在跑。所以"Function Calling"
这个名字有误导性 —— 准确说法是 **Tool Use / Model-Controlled Dispatch**：
模型输出的是一个"意图数据结构"，执行权始终在我们手里。

这个边界决定了 W9 Agent 的全部设计：
- 权限、鉴权、限流、审计**必须**在注册表那一层做（模型会点名任何它见过的工具名）
- 模型可以幻觉出不存在的工具、伪造参数 —— 所以 `execute_tool` 必须防住
- "模型调用了危险函数" 这种事只有在**我们**实现得不设防时才会发生

---

**消息数组的形状**（一次并行调用 + 回灌，最完整的情况）：

    [0] {"role": "system",    "content": "..."}
    [1] {"role": "user",      "content": "北京和上海天气"}
    [2] {"role": "assistant", "content": "...", "tool_calls": [call_1, call_2]}
    [3] {"role": "tool", "tool_call_id": call_1.id, "content": "{...北京...}"}
    [4] {"role": "tool", "tool_call_id": call_2.id, "content": "{...上海...}"}
    [5] {"role": "assistant", "content": "北京 12°C 多云，上海 19.5°C 多云。"}   ← 最终答案

三个容易写错、且**不会报错**的地方：

1. `[2]` 必须原样带回 `tool_calls`（连同 `id`）。少了它，后续的 `role="tool"`
   消息就成了"没有出处的回答"，多数服务会直接 400。
2. `[3] [4]` 的 `tool_call_id` 必须**逐个**对上，不能只对一个、也不能共用同一个 id。
3. `content` 必须是字符串形式的 JSON，不能塞 dict 对象。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from openai import AsyncOpenAI

from app.core.llm.tools import (
    TOOLS_SCHEMA,
    build_tool_messages,
    execute_tool_calls,
)

# 最多允许模型连续调多少轮工具。
#
# **必须有上限**：模型可能出现"调工具 → 结果不满意 → 再调 → 再不满意"的循环，
# 每轮都是一次计费的真实请求，没有上限就是无限烧钱。3 轮够覆盖
# "查两个城市 → 算温差" 这类合理链条，再多通常是模型在打转，此时应该停下来
# 把半成品交给用户，而不是继续。
MAX_TOOL_ROUNDS: Final[int] = 3

DEFAULT_SYSTEM_PROMPT: Final[str] = (
    "你是一个严谨的助手。需要实时数据或精确计算时必须调用工具，不要凭记忆回答。"
    "工具返回失败时，看清楚错误原因再决定是改参数重试还是如实告诉用户。"
    "最终回答里不要出现 JSON 或工具名，用自然语言说清楚结论。"
)


@dataclass
class ToolLoopResult:
    """一次闭环的完整结果。

    `messages` 是**完整对话数组**，包括中间那次 assistant 的 tool_calls 消息
    和所有 tool 消息 —— 任务要求"打印完整 messages 数组"，也是排查时的全部依据。
    """

    final_message: str
    messages: list[dict[str, Any]]
    rounds: int
    tool_calls_count: int
    stopped_reason: str = "completed"  # completed / max_rounds
    usage_total: dict[str, int] = field(default_factory=dict)


def _assistant_tool_calls_message(message: Any) -> dict[str, Any]:
    """把 SDK 的 assistant 消息转成可以回灌的 dict，**保留 tool_calls 的 id**。

    为什么要显式重建而不是 `message.model_dump()`：
    `model_dump()` 会带上 `audio` / `refusal` / `function_call`（旧版）等一堆
    这个模型根本不返回的键，其中 `None` 值在被网关重新序列化时可能触发奇怪的校验。
    只挑协议要求的字段最稳。

    `content` 允许为 `None`（模型有时只给 tool_calls 不写前言），
    但**不能省略这个键** —— 有的实现会因为缺 `content` 报 400。
    """
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message.tool_calls
        ],
    }


async def run_tool_loop(
    client: AsyncOpenAI,
    *,
    model: str,
    user_message: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_rounds: int = MAX_TOOL_ROUNDS,
    max_tokens: int = 800,
    temperature: float = 0.0,
    extra_request: dict[str, Any] | None = None,
) -> ToolLoopResult:
    """跑完整个闭环，返回最终答案和完整 messages。

    `temperature=0.0` 是刻意的：工具调用要的是"稳定地选对工具、稳定地生成合法
    JSON 参数"，任何创造性都是风险。昨天实测过 temp=0 也不完全可复现，
    但它是把漂移压到最小的选择（见 learn/llm/01_params.md §5）。

    `extra_request` 用于测试/特殊网关的额外参数（如本机网关的
    `reasoning_effort="none"`），合并进每次请求体。**不写死在函数里** ——
    因为这是网关特性，不是协议的一部分，换云端就该消失。
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    rounds = 0
    tool_calls_count = 0

    while rounds < max_rounds:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": TOOLS_SCHEMA,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **(extra_request or {}),
        }
        response = await client.chat.completions.create(**payload)
        choice = response.choices[0]
        message = choice.message

        if response.usage is not None:
            usage_total["prompt_tokens"] += response.usage.prompt_tokens
            usage_total["completion_tokens"] += response.usage.completion_tokens
            usage_total["total_tokens"] += response.usage.total_tokens

        # ---- 模型没调工具：给出最终答案，结束 ----
        if not message.tool_calls:
            messages.append({"role": "assistant", "content": message.content or ""})
            return ToolLoopResult(
                final_message=message.content or "",
                messages=messages,
                rounds=rounds + 1,
                tool_calls_count=tool_calls_count,
                usage_total=usage_total,
            )

        # ---- 模型要调工具 ----
        rounds += 1
        tool_calls_count += len(message.tool_calls)

        # ① 原样回灌 assistant 的 tool_calls（带 id），这是下一轮的前提
        messages.append(_assistant_tool_calls_message(message))

        # ② 执行（顺序与 tool_calls 一致），失败也是结构化数据
        results = execute_tool_calls(message.tool_calls)

        # ③ 按 id 一一配对回灌
        messages.extend(build_tool_messages(message.tool_calls, results))

    # 轮数用尽：把当前状态交出去，不假装完成。
    return ToolLoopResult(
        final_message="（已达工具调用轮数上限，模型仍在请求工具，未给出最终答案）",
        messages=messages,
        rounds=rounds,
        tool_calls_count=tool_calls_count,
        stopped_reason="max_rounds",
        usage_total=usage_total,
    )


def render_messages(messages: list[dict[str, Any]]) -> str:
    """把完整 messages 渲染成人能读的形式（任务要求"打印完整 messages 数组"）。

    只做可读性排版，不丢信息：`tool_calls` 的 id 和 arguments 都完整保留，
    因为"id 有没有配对"正是这类 bug 的第一现场。
    """
    lines: list[str] = []
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        lines.append(f"[{idx}] role={role}")
        content = msg.get("content")
        if content:
            lines.append(f"     content: {content}")
        for call in msg.get("tool_calls") or []:
            fn = call["function"]
            lines.append(f"     tool_call id={call['id']}")
            lines.append(f"       name={fn['name']} arguments={fn['arguments']}")
        if msg.get("tool_call_id"):
            lines.append(f"     tool_call_id: {msg['tool_call_id']}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "MAX_TOOL_ROUNDS",
    "ToolLoopResult",
    "render_messages",
    "run_tool_loop",
]
