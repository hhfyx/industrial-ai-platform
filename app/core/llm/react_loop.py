"""ReAct 循环（周四/Agent 线：Thought → Action → Observation 的多步循环）。

与 W2 `tool_loop.py` 的本质区别只有一句话：
**tool_loop 的"循环"其实是协议要求的回灌闭环，步骤数由任务形状决定；
ReAct 的循环是真正的运行时决策循环，模型每一步都在重新选择下一步做什么。**

必答问题的答案直接写进了第 3 节（三个保险丝）：
**为什么"重复调用检测"必须在里面？** —— 因为模型会陷入循环：
同一个检索查三次、参数一模一样，每一轮都是一次计费的真实请求，
而模型自己"觉得在努力"。它是成本与稳定性的第一道闸，
也是 W2 那句「模型只是提议、执行在你这一侧」的直接后果 ——
既然执行权在我这侧，那"拒绝执行"也必须在我这侧。

---

今天的坑：把"思维链"和"动作"混在同一次自由文本输出里。
本实现的做法是：**用 FC 选工具、用文本说理由** ——
`message.tool_calls` 是结构化的动作（主路径），
`message.content` 是 Thought（理由）。比让模型自由写
`Action: search(args)` 再用正则抠，稳一个量级；文本解析只作降级。

---

## 循环骨架

    while step < max_steps:
        组装（system + 历史 + tools schema）
        response = await model(...)          # ← 模型输出：content=Thought，tool_calls=Action
        if 有 tool_calls:                    # FC 主路径
            逐个：重复检测 → 执行（带超时）→ 包装 observation
            回灌 assistant(tool_calls) + tool 消息（按 id 配对）
        elif 文本里有 Action: / Final Answer:  # 文本降级路径
            解析 → 执行 → 以 user 消息回灌 [Observation]
            解析失败 → 错误回灌重试一次，再失败 → 终止
        else:                                # 没调工具也没写 Action
            content 即最终答案 → completed
        步数 +1

## 三条终止路径（返回给调用方时必须如实）

| stopped_reason | 触发 | final_answer 里有什么 |
| --- | --- | --- |
| `completed` | 模型给出最终答案 | 完整答案 |
| `max_steps` | 步数耗尽，模型仍要调工具 | **部分结果**：已执行了什么 + 未完成原因 |
| `unrepairable_failure` | 同一工具同类失败连续 3 次 | **做不到，因为…**（最后一次错误原文） |
| `repeat_loop` | 相同 (tool,args) 第 3 次出现 | 做不到 + 重复调用证据 |
| `parse_failure` | 文本动作连续 2 次解析失败 | 做不到 + 原始输出样本 |
| `model_timeout` | 单步模型调用超时 | 做不到 + 已完成部分 |

除 `completed` 外都是"如实退出"：**绝不假装完成**，交出去的是半成品和原因。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Final

from openai import AsyncOpenAI

from app.core.llm.tool_loop import _assistant_tool_calls_message
from app.core.llm.tools import TOOLS_SCHEMA, build_tool_messages

# ============================================================ 常量
#
# 每个上限都是保险丝：模型不知道它们的存在，也不需要知道 ——
# 熔断时它已经没有发言权了，final_answer 是**我方代码**拼的，不是模型说的。

#: 最大步数。一步 = 一次模型请求（含其触发的全部工具执行）。
#: 12 步足够覆盖「查规范→查数据→比对→生成→通知」级别的任务；
#: 还没跑完通常说明模型在打转，该把半成品交出去了。
MAX_STEPS: Final[int] = 12

#: 单步**模型调用**超时（秒）。注意这只约束一次 HTTP 往返；
#: 连接层的重试在 client 里，这里是业务层的硬闸。
MODEL_STEP_TIMEOUT_S: Final[float] = 45.0

#: 单次**工具执行**超时（秒）。工具是同步函数，扔进线程池再用 wait_for 卡表；
#: 没有这个闸，一个卡死的工具能把整个循环挂住。
TOOL_STEP_TIMEOUT_S: Final[float] = 10.0

#: observation 进入上下文前的字符上限。
#: 3000 字的工具返回直接塞进 messages，后续每一步都要为它付 token ——
#: 循环里这是**乘以步数**的成本，必须在这里截住。
OBSERVATION_MAX_CHARS: Final[int] = 800

#: 相同 (tool, args) 连续出现的容忍次数：
#: 第 2 次 → 不执行，回灌"重复警告"逼模型换策略（强制换策略）；
#: 第 3 次 → 直接熔断终止（终止）。
REPEAT_LIMIT: Final[int] = 2

#: 同一工具同类失败连续多少次后判定"不可修复"。
#: 3 = 给模型两次自愈机会（换参数），还不行就是工具对任务根本不可用。
UNREPAIRABLE_FAILURE_LIMIT: Final[int] = 3

#: 文本动作解析连续失败多少次后终止。第 1 次失败回灌错误让它重试。
PARSE_FAILURE_LIMIT: Final[int] = 2

REACT_SYSTEM_PROMPT: Final[str] = (
    "你是一个用 ReAct 模式工作的助手，一步步完成任务。每一步：\n"
    "1. 先在回复正文用 1-3 句话写 Thought：当前已知什么、缺什么、下一步为什么这么做；\n"
    "2. 需要外部信息或精确计算时，通过 tools 发起调用（Action）；\n"
    "3. 收到工具结果（Observation）后再决定下一步。\n"
    "硬性规则：\n"
    "- 禁止用相同参数重复调用相同工具；工具报错后必须改变参数或换工具。\n"
    "- 信息足够时直接用自然语言给出最终答案，不要再调工具。\n"
    "- 无法完成时如实说明缺什么，不要编造。"
)

# 文本降级的两个锚点。greedy + DOTALL：参数 JSON 里可能有嵌套花括号，
# 匹配到**最后一个** `}` 才安全。
_ACTION_RE: Final[re.Pattern[str]] = re.compile(
    r"Action\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\{.*\})", re.DOTALL
)
# 宽松探测：只看有没有 "Action:" 意图。严格正则抠不出参数（JSON 没闭合等）
# 但宽松探测命中 → 这才是"解析失败"。两个正则合一的话，坏 JSON 会被
# 静默当成最终答案 —— 那是文本模式最隐蔽的 bug。
_ACTION_HINT_RE: Final[re.Pattern[str]] = re.compile(r"Action\s*:")
_FINAL_RE: Final[re.Pattern[str]] = re.compile(r"Final Answer\s*:\s*(.+)", re.DOTALL)


# ============================================================ 结果类型


@dataclass
class StepRecord:
    """一步的审计记录。`steps` 是排查"模型为什么走这条路"的第一现场。"""

    step: int
    mode: str  # "fc" | "text" | "final" | "parse_error"
    thought: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    # 每个 action: {"tool", "args", "ok", "kind", "chars", "executed"}
    note: str = ""


@dataclass
class ReactResult:
    """一次 ReAct 循环的完整结果。

    `stopped_reason` 是调用方判断"这答案能不能信"的第一依据：
    除 `completed` 外，`final_answer` 都是**部分结果或失败原因**，不是答案。
    """

    final_answer: str
    messages: list[dict[str, Any]]
    steps: list[StepRecord]
    stopped_reason: str  # completed / max_steps / unrepairable_failure / repeat_loop / parse_failure / model_timeout
    usage_total: dict[str, int] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return self.stopped_reason == "completed"


# ============================================================ observation 处理


def _shrink_strings(obj: Any, limit: int) -> tuple[Any, int]:
    """递归把超长字符串截成**首尾**（摘要策略：长文档块只留头尾）。

    返回 (新对象, 被截断的字符串个数)。只动 str，dict/list 结构原样保留 ——
    结构化的键值对是模型定位信息用的，砍结构比砍内容更伤。
    """
    if isinstance(obj, str) and len(obj) > limit:
        head, tail = limit // 2, limit // 2
        omitted = len(obj) - head - tail
        return f"{obj[:head]}\n...[中间省略 {omitted} 字符]...\n{obj[-tail:]}", 1
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        count = 0
        for key, value in obj.items():
            out[key], n = _shrink_strings(value, limit)
            count += n
        return out, count
    if isinstance(obj, list):
        out_list = []
        count = 0
        for item in obj:
            shrunk, n = _shrink_strings(item, limit)
            out_list.append(shrunk)
            count += n
        return out_list, count
    return obj, 0


def wrap_observation(
    raw: dict[str, Any], *, max_chars: int = OBSERVATION_MAX_CHARS
) -> dict[str, Any]:
    """把 execute_tool 的原始结果包装成结构化、有界的 observation。

    两级截断，先便宜后昂贵：
    1. **字段级**：任何 >400 字符的字符串截成首尾（保留结构，通常这一步就够）；
    2. **整体级**：序列化后仍超 max_chars → 整个 data 换成"首尾预览"。
    两级都要保证回灌的仍是**合法 JSON** —— 戁断后的乱文本会让模型把
    截断当数据解析，比不加截断更糟。

    包装形状：`{ok, tool, kind, data, truncated, chars}` ——
    `ok` 让模型第一眼分清成败，`truncated` 明示"你看到的是节选"，
    防止模型把半截文档当成完整证据下结论。
    """
    data = raw.get("result", raw.get("error", raw))
    data, shrunk_count = _shrink_strings(data, limit=400)
    text = json.dumps(data, ensure_ascii=False)
    truncated = shrunk_count > 0
    if len(text) > max_chars:
        data = {
            "preview": text[: max_chars - 200] + "\n...[截断]...\n" + text[-150:],
            "note": "数据整体超过上限，仅保留首尾预览；完整数据未进入上下文。",
        }
        truncated = True

    wrapped = {
        "ok": bool(raw.get("ok", False)),
        "tool": raw.get("tool"),
        "kind": raw.get("kind", "ok" if raw.get("ok") else "error"),
        "data": data,
        "truncated": truncated,
        "chars": len(json.dumps(data, ensure_ascii=False)),
    }
    return wrapped


def _canonical_args(raw: str) -> str:
    """参数的规范化键：重复检测按**语义**比较，不按原始字符串。

    `{"city":"北京"}` 和 `{ "city": "北京" }` 是同一次调用，
    空格差异不该骗过熔断器。解析失败时退回原始串（前 200 字）。
    """
    try:
        return json.dumps(json.loads(raw), sort_keys=True, ensure_ascii=False)
    except (json.JSONDecodeError, ValueError):
        return f"<raw>{raw.strip()[:200]}"


# ============================================================ 文本降级解析


def parse_text_action(content: str) -> dict[str, Any] | None:
    """从自由文本里抠出 Action。返回 None 表示"没有动作意图"。

    三种结果要分清（混在一起是文本模式的经典 bug）：
    - 匹配到 Action → {"name", "arguments"}（交执行层，执行层自己会校验参数合法性）
    - 匹配到 Final Answer → None，由调用方按"最终答案"处理
    - 什么都没有 → None（纯 Thought/闲聊，同样按最终答案处理）
    """
    match = _ACTION_RE.search(content)
    if match:
        return {"name": match.group(1), "arguments": match.group(2)}
    return None


def parse_final_answer(content: str) -> str | None:
    """Final Answer: 前缀的显式答案；没有前缀返回 None（由调用方兜底）。"""
    match = _FINAL_RE.search(content)
    return match.group(1).strip() if match else None


# ============================================================ 主循环


def _partial_progress(steps: list[StepRecord]) -> str:
    """把已完成的步骤压成一段可读的"半成品清单"。

    步数耗尽 / 熔断时，final_answer 必须包含它 —— "如实退出"的底线是
    让调用方知道**做到哪了、还缺什么**，而不是一句"失败了"。
    """
    lines: list[str] = []
    for rec in steps:
        if not rec.actions:
            continue
        for act in rec.actions:
            status = "ok" if act.get("ok") else f"失败({act.get('kind')})"
            lines.append(f"  step{rec.step}: {act.get('tool')}({act.get('args')}) → {status}")
        if rec.thought:
            lines.append(f"  step{rec.step} Thought: {rec.thought[:80]}")
    return "\n".join(lines) if lines else "  （还没有执行任何工具调用）"


async def run_react_loop(
    client: AsyncOpenAI,
    *,
    model: str,
    user_message: str,
    system_prompt: str = REACT_SYSTEM_PROMPT,
    tools: list[dict[str, Any]] | None = TOOLS_SCHEMA,
    executor: Any | None = None,
    max_steps: int = MAX_STEPS,
    model_timeout: float = MODEL_STEP_TIMEOUT_S,
    tool_timeout: float = TOOL_STEP_TIMEOUT_S,
    max_tokens: int = 800,
    temperature: float = 0.0,
    text_fallback: bool = True,
    observation_max_chars: int = OBSERVATION_MAX_CHARS,
    extra_request: dict[str, Any] | None = None,
) -> ReactResult:
    """跑完一个 ReAct 循环。

    `tools=None` 走"无工具"路径：循环退化为普通的带步数上限的对话，
    用于先验证骨架能正常结束（今天的第一个验收点）。

    `executor` 是可替换的执行函数 `(name, raw_args) -> dict`。**tools 和
    executor 必须成对出现** —— 这是周三实测抓到的 P0：循环体里写死了
    `execute_tool`（W2 的注册表），换成 QA 工具的 schema 后，模型看到的
    tools 是新四个、执行层查的注册表还是旧两个，全部 unknown_tool。
    schema 与注册表不同源，就是"模型按 schema 生成了参数、执行层不认"
    的变体 —— W2 在 schema 层防过这个漂移，这里在执行层又犯了一次。
    默认 None 用 W2 的 `execute_tool`（weather/calculate，保持旧行为）。

    `temperature=0.0` 与 tool_loop 同理：动作选择要的是稳定，不是创造性。
    """
    run_tool = executor if executor is not None else _execute_tool_sync
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    steps: list[StepRecord] = []

    # ---- 保险丝状态 ----
    # 重复检测按**全局出现次数**计数，不是相邻连击：实测（demo C 场景）模型会把
    # 一整批并行调用原样重放（step1 查 4 城 → step2 又查同样 4 城），批与批之间
    # 没有"相邻"关系，连击计数会被批内其他调用稀释，绕过熔断 —— 这是本次实测
    # 抓到的真漏洞。出现第 2 次（无论隔多远）→ 警告不执行；第 3 次 → 熔断。
    action_seen: dict[str, int] = {}
    prev_failure_key: str = ""  # 上一个失败 (tool, kind)，用于同类失败连击
    failure_streak: int = 0
    parse_fail_streak: int = 0

    def _accumulate_usage(response: Any) -> None:
        if response.usage is not None:
            usage_total["prompt_tokens"] += response.usage.prompt_tokens
            usage_total["completion_tokens"] += response.usage.completion_tokens
            usage_total["total_tokens"] += response.usage.total_tokens

    def _finish(reason: str, answer: str) -> ReactResult:
        messages.append({"role": "assistant", "content": answer})
        return ReactResult(
            final_answer=answer,
            messages=messages,
            steps=steps,
            stopped_reason=reason,
            usage_total=usage_total,
        )

    step_no = 0
    while step_no < max_steps:
        step_no += 1

        # ---- ① 模型调用（带单步超时）----
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **(extra_request or {}),
        }
        if tools is not None:
            payload["tools"] = tools

        try:
            async with asyncio.timeout(model_timeout):
                response = await client.chat.completions.create(**payload)
        except TimeoutError:
            return _finish(
                "model_timeout",
                f"[无法完成] 第 {step_no} 步模型调用超过 {model_timeout}s 被中断。\n"
                f"已完成部分：\n{_partial_progress(steps)}",
            )
        _accumulate_usage(response)
        message = response.choices[0].message

        # ---- ② FC 主路径：tool_calls 是结构化的 Action ----
        if message.tool_calls:
            messages.append(_assistant_tool_calls_message(message))
            thought = (message.content or "").strip()
            record = StepRecord(step=step_no, mode="fc", thought=thought)
            steps.append(record)

            terminated: str | None = None
            tool_messages: list[dict[str, Any]] = []
            for call in message.tool_calls:
                name = call.function.name
                raw_args = call.function.arguments
                key = f"{name}|{_canonical_args(raw_args)}"

                # --- 保险丝③：重复调用检测（先于执行，拒绝执行也在我这侧）---
                # 全局计数：key 的历史出现次数 + 1 = 本次序数
                action_seen[key] = action_seen.get(key, 0) + 1
                occurrence = action_seen[key]

                if occurrence > REPEAT_LIMIT:
                    # 第 3 次出现：模型无视了上一轮的换策略警告 → 熔断
                    wrapped = {
                        "ok": False,
                        "tool": name,
                        "kind": "repeat_call",
                        "data": {"message": "相同参数的重复调用已被熔断，循环终止。"},
                        "truncated": False,
                        "chars": 0,
                    }
                    record.actions.append(
                        {
                            "tool": name,
                            "args": raw_args[:120],
                            "ok": False,
                            "kind": "repeat_call",
                            "executed": False,
                        }
                    )
                    tool_messages.extend(build_tool_messages([call], [wrapped]))
                    terminated = "repeat_loop"
                    break

                if occurrence == REPEAT_LIMIT:
                    # 第 2 次出现：不执行，回灌警告逼模型换策略
                    wrapped = {
                        "ok": False,
                        "tool": name,
                        "kind": "repeat_call",
                        "data": {
                            "message": (
                                "这是你第 2 次用完全相同的参数调用相同工具（含之前的步骤）。"
                                "禁止再次重复；请改变参数、换一个工具，或直接给出最终答案。"
                            )
                        },
                        "truncated": False,
                        "chars": 0,
                    }
                    record.actions.append(
                        {
                            "tool": name,
                            "args": raw_args[:120],
                            "ok": False,
                            "kind": "repeat_call",
                            "executed": False,
                        }
                    )
                    tool_messages.extend(build_tool_messages([call], [wrapped]))
                    continue

                # --- 真正执行（带单步超时；同步工具扔线程池，别挂住事件循环）---
                started = time.perf_counter()
                try:
                    raw_result = await asyncio.wait_for(
                        asyncio.to_thread(run_tool, name, raw_args),
                        timeout=tool_timeout,
                    )
                except TimeoutError:
                    raw_result = {
                        "ok": False,
                        "tool": name,
                        "kind": "tool_timeout",
                        "error": {"message": f"工具执行超过 {tool_timeout}s 被强制中断"},
                    }
                elapsed_ms = int((time.perf_counter() - started) * 1000)

                # --- 保险丝②'：同类失败连击 → 不可修复 ---
                if raw_result.get("ok"):
                    failure_streak = 0
                    prev_failure_key = ""
                else:
                    fkey = f"{name}|{raw_result.get('kind')}"
                    failure_streak = failure_streak + 1 if fkey == prev_failure_key else 1
                    prev_failure_key = fkey
                    if failure_streak >= UNREPAIRABLE_FAILURE_LIMIT:
                        terminated = "unrepairable_failure"

                wrapped = wrap_observation(raw_result, max_chars=observation_max_chars)
                record.actions.append(
                    {
                        "tool": name,
                        "args": raw_args[:120],
                        "ok": bool(raw_result.get("ok")),
                        "kind": wrapped["kind"],
                        "chars": wrapped["chars"],
                        "ms": elapsed_ms,
                        "executed": True,
                    }
                )
                tool_messages.extend(build_tool_messages([call], [wrapped]))

            messages.extend(tool_messages)
            if terminated == "repeat_loop":
                return _finish(
                    "repeat_loop",
                    f"[无法完成] 相同 (工具,参数) 第 {occurrence} 次出现"
                    f"（全局计数，含跨步骤与并行批内重放），"
                    f"模型在收到换策略警告后仍未改变，按熔断规则终止。\n"
                    f"已完成部分：\n{_partial_progress(steps)}",
                )
            if terminated == "unrepairable_failure":
                last_fail = next((a for a in reversed(record.actions) if not a.get("ok")), None)
                return _finish(
                    "unrepairable_failure",
                    f"[无法完成] 做不到，因为：工具 {last_fail.get('tool')} 连续 "
                    f"{failure_streak} 次同类失败（{last_fail.get('kind')}），"
                    f"已超过自愈上限，按熔断规则停止。\n"
                    f"已完成部分：\n{_partial_progress(steps)}",
                )
            continue

        # ---- ③ 无 tool_calls：文本降级 或 最终答案 ----
        content = message.content or ""

        if tools is not None and text_fallback:
            action = parse_text_action(content)
            if action is not None:
                # 文本模式：observation 以 user 消息回灌（没有 tool_call_id 可配对，
                # role="tool" 会因"没有出处"被网关 400 —— 这是文本模式的协议代价）
                record = StepRecord(step=step_no, mode="text", thought="")
                steps.append(record)
                # 文本模式与 FC 模式共用同一个超时与失败连击逻辑；
                # 这里漏接 TimeoutError 会让一次工具卡死炸掉整个循环。
                try:
                    raw_result = await asyncio.wait_for(
                        asyncio.to_thread(run_tool, action["name"], action["arguments"]),
                        timeout=tool_timeout,
                    )
                except TimeoutError:
                    raw_result = {
                        "ok": False,
                        "tool": action["name"],
                        "kind": "tool_timeout",
                        "error": {"message": f"工具执行超过 {tool_timeout}s 被强制中断"},
                    }
                ok = bool(raw_result.get("ok"))
                if ok:
                    failure_streak = 0
                    prev_failure_key = ""
                else:
                    fkey = f"{action['name']}|{raw_result.get('kind')}"
                    failure_streak = failure_streak + 1 if fkey == prev_failure_key else 1
                    prev_failure_key = fkey
                wrapped = wrap_observation(raw_result, max_chars=observation_max_chars)
                record.actions.append(
                    {
                        "tool": action["name"],
                        "args": action["arguments"][:120],
                        "ok": ok,
                        "kind": wrapped["kind"],
                        "chars": wrapped["chars"],
                        "executed": True,
                    }
                )
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": f"[Observation] {json.dumps(wrapped, ensure_ascii=False)}\n"
                        "（这是文本模式下你上一次 Action 的结果。继续：Thought → Action 或 Final Answer）",
                    }
                )
                parse_fail_streak = 0
                if failure_streak >= UNREPAIRABLE_FAILURE_LIMIT:
                    return _finish(
                        "unrepairable_failure",
                        f"[无法完成] 做不到，因为：工具 {action['name']} 连续 "
                        f"{failure_streak} 次同类失败（{prev_failure_key.split('|')[-1]}），"
                        f"已超过自愈上限。\n已完成部分：\n{_partial_progress(steps)}",
                    )
                continue

            if _ACTION_HINT_RE.search(content):
                # 有 Action: 字样但抠不出来（通常是参数 JSON 没闭合）→ 解析失败
                parse_fail_streak += 1
                if parse_fail_streak >= PARSE_FAILURE_LIMIT:
                    return _finish(
                        "parse_failure",
                        f"[无法完成] 模型输出的动作连续 {parse_fail_streak} 次无法解析，"
                        f"已停止。最后一次输出片段：{content[:200]!r}",
                    )
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": "[ParseError] 你的输出里有 Action: 但无法解析出「工具名 + JSON 参数」。"
                        '请严格使用格式：Action: 工具名 {"参数": "值"}（JSON 必须闭合），'
                        "或直接用自然语言给出最终答案。",
                    }
                )
                steps.append(
                    StepRecord(
                        step=step_no,
                        mode="parse_error",
                        note=f"解析失败第 {parse_fail_streak} 次，已回灌错误要求重试",
                    )
                )
                continue

        # ---- ④ 最终答案 ----
        explicit = parse_final_answer(content)
        answer = explicit if explicit is not None else content.strip()
        steps.append(StepRecord(step=step_no, mode="final", thought=answer[:120]))
        return _finish("completed", answer)

    # ---- 步数耗尽：模型仍在要工具，交出半成品，不假装完成 ----
    return _finish(
        "max_steps",
        f"[部分结果] 已达最大步数 {max_steps}，任务未完成。\n"
        f"已完成部分：\n{_partial_progress(steps)}\n"
        f"未完成原因：步数上限触发，模型在最后一步仍在请求工具。",
    )


def _execute_tool_sync(name: str, raw_args: str) -> dict[str, Any]:
    """execute_tool 的延迟导入包装。

    延迟导入是因为 `asyncio.to_thread` 需要一个**模块级可寻址**的函数；
    直接把 tools.execute_tool 传进去也可以，但包一层让超时分支能统一
    构造同形状的错误 dict，不往循环体里塞细节。
    """
    from app.core.llm.tools import execute_tool

    return execute_tool(name, raw_args)


__all__ = [
    "MAX_STEPS",
    "MODEL_STEP_TIMEOUT_S",
    "OBSERVATION_MAX_CHARS",
    "PARSE_FAILURE_LIMIT",
    "REACT_SYSTEM_PROMPT",
    "REPEAT_LIMIT",
    "ReactResult",
    "StepRecord",
    "TOOL_STEP_TIMEOUT_S",
    "UNREPAIRABLE_FAILURE_LIMIT",
    "parse_final_answer",
    "parse_text_action",
    "run_react_loop",
    "wrap_observation",
]
