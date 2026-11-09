"""W10 周六 · 把 W9 的 run_react_loop 迁移为 LangGraph 图（主交付）。

## 迁移原则（冻结声明 docs/weekly/w10.md 的前提）

**逐行为对齐**：工具 schema、executor、system prompt、temperature、max_tokens、
extra_request、重复熔断（全局计数）、同类失败连击、文本降级、六种停止原因
全部原样保留——判据函数与轨迹导出（eval_tasks）改零行即可复用。
图版多出来的只有：checkpointer（thread_id=每任务）与状态可视化。

**为什么不是周三的分类图**：route/direct_answer/retrieve 三路分类要加一次
分类调用（prompt 未改 → 违反冻结），且 W9 基线里模型是自主选工具的。
分类图留在 learn/llm/ 作教学产物，评测图用本文件的循环图。

## 节点

    START → model_step → route_after_model ─┬→ tools → route_after_tools ─┬→ retry ──┐
                pending_mode 决定            │                │             ├→ degrade → finalize_degraded
                ├ fc/text_action → tools     │                │             └→ model_step（回环）
                └ final → finalize → END     │
                └ parse_error → parse_retry → model_step

- model_step：唯一调模型的地方（对应 W9 循环体 ①）
- tools：执行 pending_calls（重复熔断 + 失败连击 + 超时，对应 ②③）
- **retry**（W11 周二新增）：代码退避重试（timeout / internal），带 jitter
- **degrade**（W11 周二新增）：预算用尽或不可重试 → 结构化降级，**不再进 LLM**
- parse_retry：文本 Action 解析失败的回灌重试（对应 ③'）
- finalize / finalize_budget / finalize_degraded：停止原因语义
- 路由函数是纯函数（周三的纪律）；预算判定在路由里，不靠异常

## 为什么 retry 是**节点**而不是循环里的 for

W10 的 C4：局部变量在崩溃恢复后会归零。如果重试用 `for attempt in range(2)`
写在 tools 节点里，进程在两次重试之间崩掉、从 checkpoint 恢复时，
`attempt` 会重新从 0 开始 —— **重试预算在恢复后失效**，变成"重试到成功"。
把计数放进 state（`retry_counts`）并且让每一次重试都成为一次图推进，
恢复时 LangGraph 才能把它一起带回来。

同理，`retry` 节点自己不做 sleep 判断之外的决策：它读 `pending_decisions`
（tools 算好的），**照着执行**。判定与执行分开，才有"不起图也能单测路由"。

## State 里放什么（周二的纪律）

数据与轨迹（messages/steps/usage）进 state；client 通过 build_agent_graph
闭包注入。state 可序列化（messages 是纯 dict）。
"""

from __future__ import annotations

import asyncio
import json
import operator
import time
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.core.llm.agent_registry import AGENT_TOOLS_SCHEMA, execute_agent_tool
from app.core.llm.error_contract import tool_error
from app.core.llm.react_loop import (
    _ACTION_HINT_RE,
    MAX_STEPS,
    MODEL_STEP_TIMEOUT_S,
    PARSE_FAILURE_LIMIT,
    REPEAT_LIMIT,
    TOOL_STEP_TIMEOUT_S,
    UNREPAIRABLE_FAILURE_LIMIT,
    _action_record,
    _envelope_meta,
    _partial_progress,
    is_async_executor,
    parse_text_action,
    wrap_observation,
)
from app.core.llm.recovery import (
    RETRY_MAX_DELAY_S,
    build_degrade_payload,
    decide,
    render_degrade_text,
    spec_of,
)
from app.core.llm.tool_loop import _assistant_tool_calls_message
from app.core.llm.tools import build_tool_messages

_OBS_IN_TRACE = 160  # 与 W9 harness 相同的观察摘要截断


class AgentGraphState(TypedDict, total=False):
    # ---- 累积字段（reducer）----
    messages: Annotated[list, operator.add]
    steps: Annotated[list, operator.add]  # dict 形态的 StepRecord，合并后组轨迹
    usage_log: Annotated[list, operator.add]  # 每次 API 调用的 usage 增量
    # ---- 设值字段（单写者）----
    instruction: str
    model: str
    max_steps: int
    step_no: int
    pending_mode: str  # fc / text_action / parse_error / final
    pending_calls: list
    action_seen: dict  # 全局重复计数（tools 节点返回完整替换值）
    failure_streak: int
    prev_failure_key: str
    parse_fail_streak: int
    retry_counts: dict  # W11：(tool,error_type) → 已重试次数（重试预算）
    pending_decisions: list  # W11 周二：tools/retry 算好的处置决定（路由照着走）
    retry_queue: list  # W11 周二：待代码退避重试的调用（[{call, delay_s}]）
    unresolved: list  # W11 周二：确定没解决的能力（进降级 missing）
    degrade_payload: dict  # W11 周二：{answer, missing, degraded, reason}
    stopped_reason: str
    final_answer: str


# ============================================================ 小工具


def _merged_steps(state: AgentGraphState) -> list[dict]:
    """把 state 里 dict 形态的 steps 按 step 号合并（W9 是循环内聚合一处）。"""
    merged: dict[int, dict] = {}
    for rec in state.get("steps", []):
        m = merged.setdefault(
            rec["step"],
            {
                "step": rec["step"],
                "mode": rec["mode"],
                "thought": rec.get("thought", ""),
                "model_ms": rec.get("model_ms", 0),
                "tokens": rec.get("tokens", {}),
                "observation": rec.get("observation", ""),
                "actions": [],
            },
        )
        m["actions"].extend(a for a in rec.get("actions", []) if a not in m["actions"])
    return [merged[k] for k in sorted(merged)]


def _as_step_records(state: AgentGraphState) -> list:
    """_partial_progress 吃 StepRecord 对象；图版 steps 是 dict，包一层适配。"""
    from app.core.llm.react_loop import StepRecord

    return [
        StepRecord(
            step=m["step"],
            mode=m["mode"],
            thought=m.get("thought", ""),
            actions=m.get("actions", []),
        )
        for m in _merged_steps(state)
    ]


# ============================================================ 图


def _route_after_model(state: AgentGraphState) -> str:
    """模型步之后：有工具调用去 tools，解析失败去 parse_retry，否则收尾。"""
    if state.get("stopped_reason"):
        return "finalize"
    mode = state.get("pending_mode", "final")
    if mode in ("fc", "text_action"):
        return "tools"
    if mode == "parse_error":
        return "parse_retry"
    return "finalize"


def _route_after_tools(state: AgentGraphState) -> str:
    """工具步之后：预算用尽 → degrade，有排队重试 → retry，否则回模型。

    **判降级优先于判重试**：一次批内多个调用时，可能一个还能重试、
    另一个已经用尽。用尽的那个意味着任务缺了一块，先降级收场，
    不要用"继续重试"把它盖过去。
    """
    if state.get("stopped_reason"):
        return "finalize"
    decisions = state.get("pending_decisions", [])
    if any(d.get("action") == "degrade" for d in decisions):
        return "degrade"
    if state.get("retry_queue"):
        return "retry"
    if state.get("step_no", 0) >= state.get("max_steps", MAX_STEPS):
        # 业务预算在路由里判：优雅退出（部分结果），不是异常
        return "finalize_budget"
    return "model_step"


def _route_after_retry(state: AgentGraphState) -> str:
    """重试节点之后：还能再试 → retry，用尽 → degrade，否则交回模型。"""
    if state.get("stopped_reason"):
        return "finalize"
    decisions = state.get("pending_decisions", [])
    if any(d.get("action") == "degrade" for d in decisions):
        return "degrade"
    if state.get("retry_queue"):
        return "retry"
    return "model_step"


def build_agent_graph(client: Any, *, executor: Any | None = None):
    """构建评测图。client 走闭包（不在 state 里）；executor 默认 W9 注册表。"""
    run_tool = executor if executor is not None else execute_agent_tool

    async def model_step(state: AgentGraphState) -> dict:
        """对应 W9 循环体 ①：调模型、记 usage/延迟、分类输出形态。"""
        step_no = state.get("step_no", 0) + 1
        messages = list(state["messages"])
        payload: dict[str, Any] = {
            "model": state["model"],
            "messages": messages,
            "temperature": 0.0,  # 冻结：与 W9 相同
            "max_tokens": 800,  # 冻结：与 W9 相同
            "reasoning_effort": "none",  # 冻结：与 W9 harness 相同
            "tools": AGENT_TOOLS_SCHEMA,  # 冻结：同一 schema
        }
        started = time.perf_counter()
        try:
            async with asyncio.timeout(MODEL_STEP_TIMEOUT_S):
                response = await client.chat.completions.create(**payload)
        except TimeoutError:
            return {
                "step_no": step_no,
                "stopped_reason": "model_timeout",
                "final_answer": (
                    f"[无法完成] 第 {step_no} 步模型调用超过 {MODEL_STEP_TIMEOUT_S}s 被中断。\n"
                    f"已完成部分：\n{_partial_progress(_as_step_records(state))}"
                ),
            }
        model_ms = int((time.perf_counter() - started) * 1000)
        usage = (
            {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            if response.usage is not None
            else {}
        )
        message = response.choices[0].message
        content = message.content or ""

        out: dict[str, Any] = {
            "step_no": step_no,
            "usage_log": [usage] if usage else [],
            "pending_mode": "final",
            "final_answer": content.strip(),
        }

        if message.tool_calls:  # FC 主路径
            out["messages"] = [_assistant_tool_calls_message(message)]
            out["pending_mode"] = "fc"
            out["pending_calls"] = [
                {"name": c.function.name, "arguments": c.function.arguments, "id": c.id}
                for c in message.tool_calls
            ]
            out["steps"] = [
                {
                    "step": step_no,
                    "mode": "fc",
                    "thought": content.strip(),
                    "model_ms": model_ms,
                    "tokens": usage,
                    "actions": [],
                    "observation": "",
                }
            ]
            return out

        action = parse_text_action(content)
        if action is not None:  # 文本降级路径
            out["pending_mode"] = "text_action"
            out["pending_calls"] = [{"name": action["name"], "arguments": action["arguments"]}]
            out["messages"] = [{"role": "assistant", "content": content}]
            out["steps"] = [
                {
                    "step": step_no,
                    "mode": "text",
                    "thought": "",
                    "model_ms": model_ms,
                    "tokens": usage,
                    "actions": [],
                    "observation": "",
                }
            ]
            return out

        if _ACTION_HINT_RE.search(content):  # 有 Action: 字样但抠不出参数 → 解析失败
            out["pending_mode"] = "parse_error"
            out["messages"] = [{"role": "assistant", "content": content}]
            out["steps"] = [
                {
                    "step": step_no,
                    "mode": "parse_error",
                    "note": "解析失败（模型输出含 Action: 但无法抠出参数）",
                    "model_ms": model_ms,
                    "tokens": usage,
                    "actions": [],
                    "observation": "",
                }
            ]
            return out

        # 最终答案路径：不追加 assistant 消息（finalize 统一追加，对应 W9 _finish）
        out["steps"] = [
            {
                "step": step_no,
                "mode": "final",
                "thought": content.strip()[:120],
                "model_ms": model_ms,
                "tokens": usage,
                "actions": [],
                "observation": "",
            }
        ]
        return out

    async def tools_node(state: AgentGraphState) -> dict:
        """对应 W9 ②③：执行 pending_calls（重复熔断、失败连击、超时）。

        W11 周一：与 react_loop 共用同一套埋点与重试预算 ——
        两条执行路径（手写循环 / 图）必须产出**同形状的轨迹**，否则
        `scripts/error_stats.py` 只能统计一半的调用，W10 那种"口径分散"
        的坑会再犯一次。
        """
        mode = state["pending_mode"]
        calls = state.get("pending_calls", [])
        action_seen = dict(state.get("action_seen", {}))
        failure_streak = state.get("failure_streak", 0)
        prev_failure_key = state.get("prev_failure_key", "")
        retry_counts = dict(state.get("retry_counts", {}))
        step_no = state["step_no"]
        actions: list[dict] = []
        tool_messages: list[dict] = []
        decisions: list[dict] = []
        retry_queue: list[dict] = []
        unresolved: list[str] = list(state.get("unresolved", []))
        terminated: str | None = None
        occurrence = 0

        for call in calls:
            name, raw_args = call["name"], call["arguments"]
            key = f"{name}|{raw_args.strip()}"  # W9 用 canonical args；等价简化
            action_seen[key] = action_seen.get(key, 0) + 1
            occurrence = action_seen[key]

            if occurrence > REPEAT_LIMIT:  # 第 3 次出现：模型无视警告 → 熔断
                actions.append(
                    _action_record(
                        tool=name,
                        args=raw_args[:120],
                        ok=False,
                        kind="repeat_call",
                        latency_ms=0,
                        error_type="conflict",
                        retryable=False,
                        executed=False,
                    )
                )
                wrapped = {
                    "ok": False,
                    "tool": name,
                    "kind": "repeat_call",
                    "data": {"message": "相同参数的重复调用已被熔断，循环终止。"},
                    "truncated": False,
                    "chars": 0,
                }
                tool_messages.extend(build_tool_messages([_FakeCall(call)], [wrapped]))
                terminated = "repeat_loop"
                break
            if occurrence == REPEAT_LIMIT:  # 第 2 次：不执行，回灌警告逼换策略
                actions.append(
                    _action_record(
                        tool=name,
                        args=raw_args[:120],
                        ok=False,
                        kind="repeat_call",
                        latency_ms=0,
                        error_type="conflict",
                        retryable=False,
                        executed=False,
                    )
                )
                wrapped = {
                    "ok": False,
                    "tool": name,
                    "kind": "repeat_call",
                    "data": {
                        "message": "这是你第 2 次用完全相同的参数调用相同工具。"
                        "禁止再次重复；请改变参数、换工具，或直接给出最终答案。"
                    },
                    "truncated": False,
                    "chars": 0,
                }
                tool_messages.extend(build_tool_messages([_FakeCall(call)], [wrapped]))
                continue

            started = time.perf_counter()
            try:
                if is_async_executor(run_tool):
                    raw_result = await asyncio.wait_for(
                        run_tool(name, raw_args), timeout=TOOL_STEP_TIMEOUT_S
                    )
                else:
                    raw_result = await asyncio.wait_for(
                        asyncio.to_thread(run_tool, name, raw_args), timeout=TOOL_STEP_TIMEOUT_S
                    )
            except TimeoutError:
                raw_result = tool_error(
                    name, "timeout", f"工具执行超过 {TOOL_STEP_TIMEOUT_S}s 被强制中断"
                )
            elapsed_ms = int((time.perf_counter() - started) * 1000)

            if raw_result.get("ok"):
                failure_streak, prev_failure_key = 0, ""
            else:
                fkey = f"{name}|{raw_result.get('kind')}"
                failure_streak = failure_streak + 1 if fkey == prev_failure_key else 1
                prev_failure_key = fkey
                if failure_streak >= UNREPAIRABLE_FAILURE_LIMIT:
                    terminated = "unrepairable_failure"
                # W11 周二：**判定在这里做，执行交给 retry/degrade 节点**。
                # decide() 会累加 retry_counts —— 计数只在这一处发生，
                # 两条执行路径不会各数一遍。
                decision = decide(raw_result, retry_counts)
                decisions.append(
                    {
                        "action": decision.action,
                        "error_type": decision.error_type,
                        "tool": decision.tool,
                        "attempt": decision.attempt,
                        "limit": decision.limit,
                        "delay_s": decision.delay_s,
                        "reason": decision.reason,
                    }
                )
                if decision.action == "auto_retry":
                    retry_queue.append(
                        {
                            "call": {
                                "name": name,
                                "arguments": raw_args,
                                "id": call.get("id", "call_0"),
                            },
                            "delay_s": decision.delay_s,
                        }
                    )
                if decision.action in ("degrade", "skip_and_tell") and (
                    spec_of(decision.error_type).degrade_when_unresolved
                ):
                    unresolved.append(f"{name}（{decision.error_type}：{decision.reason}）")

            wrapped = wrap_observation(raw_result)
            error_type, retryable = _envelope_meta(raw_result)
            actions.append(
                _action_record(
                    tool=name,
                    args=raw_args[:120],
                    ok=bool(raw_result.get("ok")),
                    kind=wrapped["kind"],
                    latency_ms=elapsed_ms,
                    error_type=error_type,
                    retryable=retryable,
                    chars=wrapped["chars"],
                )
            )
            tool_messages.extend(build_tool_messages([_FakeCall(call)], [wrapped]))

        out: dict[str, Any] = {
            "messages": tool_messages,
            "steps": [
                {
                    "step": step_no,
                    "mode": "tools" if mode == "fc" else "tools_text",
                    "thought": "",
                    "model_ms": 0,
                    "tokens": {},
                    "actions": actions,
                    "observation": (
                        json.dumps(actions[-1], ensure_ascii=False)[:_OBS_IN_TRACE]
                        if actions
                        else ""
                    ),
                }
            ],
            "action_seen": action_seen,  # 完整替换值（dict 无累积 reducer）
            "failure_streak": failure_streak,
            "prev_failure_key": prev_failure_key,
            "retry_counts": retry_counts,
            "pending_decisions": decisions,
            "retry_queue": retry_queue,
            "unresolved": unresolved,
        }
        if terminated == "repeat_loop":
            out["stopped_reason"] = "repeat_loop"
            out["final_answer"] = (
                f"[无法完成] 相同 (工具,参数) 第 {occurrence} 次出现，模型在收到换策略"
                f"警告后仍未改变，按熔断规则终止。\n已完成部分：\n{_partial_progress(_as_step_records(state))}"
            )
        elif terminated == "unrepairable_failure":
            out["stopped_reason"] = "unrepairable_failure"
            out["final_answer"] = (
                f"[无法完成] 做不到，因为：工具连续 {failure_streak} 次同类失败，"
                f"已超过自愈上限，按熔断规则停止。\n已完成部分：\n{_partial_progress(_as_step_records(state))}"
            )
        return out

    # ============================================================ W11 周二：重试与降级节点

    async def retry_node(state: AgentGraphState) -> dict:
        """代码退避重试队列里的调用（timeout / internal）。

        **这是"代码执行重试"的唯一实现处**。三件事按顺序做：

        1. 退避等待 —— 用 `asyncio.sleep`，不是同步 sleep（同步 sleep 会堵住
           整个事件循环，20 任务的墙钟时间会假性变长）；
        2. 重新执行同一个调用（同工具同参数，模型没有参与）；
        3. 结果**按新的尝试重新判定**：还可能再降级、也可能成功。

        第 3 步是闭环：重试成功就不该进降级；重试仍失败就要么再试
        （预算允许）、要么降级。判定仍然调用 `decide()`，计数继续累加 ——
        所以"最多重试 N 次"在图被中断恢复后依然成立（计数在 state 里）。
        """
        queue = list(state.get("retry_queue", []))
        if not queue:
            return {"retry_queue": [], "pending_decisions": []}

        retry_counts = dict(state.get("retry_counts", {}))
        action_seen = dict(state.get("action_seen", {}))
        unresolved = list(state.get("unresolved", []))
        step_no = state.get("step_no", 0)
        actions: list[dict] = []
        tool_messages: list[dict] = []
        decisions: list[dict] = []
        again: list[dict] = []

        for item in queue:
            call = item["call"]
            delay = item.get("delay_s")
            if delay:
                await asyncio.sleep(min(float(delay), RETRY_MAX_DELAY_S))
            name, raw_args = call["name"], call["arguments"]
            started = time.perf_counter()
            try:
                if is_async_executor(run_tool):
                    raw_result = await asyncio.wait_for(
                        run_tool(name, raw_args), timeout=TOOL_STEP_TIMEOUT_S
                    )
                else:
                    raw_result = await asyncio.wait_for(
                        asyncio.to_thread(run_tool, name, raw_args), timeout=TOOL_STEP_TIMEOUT_S
                    )
            except TimeoutError:
                raw_result = tool_error(
                    name, "timeout", f"工具执行超过 {TOOL_STEP_TIMEOUT_S}s 被强制中断"
                )
            elapsed_ms = int((time.perf_counter() - started) * 1000)

            wrapped = wrap_observation(raw_result)
            error_type, retryable = _envelope_meta(raw_result)
            actions.append(
                _action_record(
                    tool=name,
                    args=raw_args[:120],
                    ok=bool(raw_result.get("ok")),
                    kind=wrapped["kind"],
                    latency_ms=elapsed_ms,
                    error_type=error_type,
                    retryable=retryable,
                    chars=wrapped["chars"],
                    retried=True,
                )
            )
            tool_messages.extend(build_tool_messages([_FakeCall(call)], [wrapped]))

            if raw_result.get("ok"):
                continue

            decision = decide(raw_result, retry_counts)
            decisions.append(
                {
                    "action": decision.action,
                    "error_type": decision.error_type,
                    "tool": decision.tool,
                    "attempt": decision.attempt,
                    "limit": decision.limit,
                    "delay_s": decision.delay_s,
                    "reason": decision.reason,
                }
            )
            if decision.action == "auto_retry":
                again.append({"call": call, "delay_s": decision.delay_s})
            elif decision.action in ("degrade", "skip_and_tell") and (
                spec_of(decision.error_type).degrade_when_unresolved
            ):
                unresolved.append(f"{name}（{decision.error_type}：{decision.reason}）")

        out: dict[str, Any] = {
            "messages": tool_messages,
            "steps": [
                {
                    "step": step_no,
                    "mode": "retry",
                    "thought": "",
                    "model_ms": 0,
                    "tokens": {},
                    "actions": actions,
                    "observation": (
                        json.dumps(actions[-1], ensure_ascii=False)[:_OBS_IN_TRACE]
                        if actions
                        else ""
                    ),
                }
            ],
            "retry_counts": retry_counts,
            "action_seen": action_seen,
            "retry_queue": again,
            "pending_decisions": decisions,
            "unresolved": unresolved,
        }
        return out

    def degrade_node(state: AgentGraphState) -> dict:
        """构造降级答案并终止。**不再调一次 LLM**。

        今天的坑（写在任务原文里）：把降级写成"随便给个通用回答"。
        用户看到一个看起来正常但内容错误的答案，比明确说"查不到"危险得多。
        所以这里只做一件事：把**已有事实**和**缺什么**分开放进
        `{answer, missing, degraded, reason}`，然后原样交给用户。

        不再进 LLM 是刻意的：让它"润色一下部分结论"，缺的那块就会被顺手补上。
        """
        decisions = list(state.get("pending_decisions", []))
        last = decisions[-1] if decisions else {}
        reason = str(last.get("error_type") or "internal")
        merged = _merged_steps(state)
        all_actions = [a for m in merged for a in m.get("actions", [])]
        payload = build_degrade_payload(
            reason=reason,
            actions=all_actions,
            unresolved=list(state.get("unresolved", [])),
        )
        return {
            "degrade_payload": payload,
            "stopped_reason": "degraded",
            "final_answer": render_degrade_text(payload),
        }

    def finalize_degraded(state: AgentGraphState) -> dict:
        """降级终点：补 assistant 消息，让 messages 与 final_answer 一致。

        与 `finalize` 分开是刻意的：`finalize` 收的是模型自己说的最终答案，
        这里收的是**代码拼的**降级文本。两者混在一个节点里，轨迹上就分不清
        "这条答案是模型给的还是我们给的" —— 而那正是降级最重要的信息。
        """
        return {"messages": [{"role": "assistant", "content": state.get("final_answer", "")}]}

    def parse_retry(state: AgentGraphState) -> dict:
        """对应 W9 ③'：解析失败回灌重试；连续 2 次 → 终止。"""
        streak = state.get("parse_fail_streak", 0) + 1
        if streak >= PARSE_FAILURE_LIMIT:
            content = state["messages"][-1]["content"]
            return {
                "parse_fail_streak": streak,
                "stopped_reason": "parse_failure",
                "final_answer": (
                    f"[无法完成] 模型输出的动作连续 {streak} 次无法解析，已停止。"
                    f"最后一次输出片段：{content[:200]!r}"
                ),
            }
        return {
            "parse_fail_streak": streak,
            "messages": [
                {
                    "role": "user",
                    "content": "[ParseError] 你的输出里有 Action: 但无法解析出"
                    "「工具名 + JSON 参数」。请严格使用格式："
                    'Action: 工具名 {"参数": "值"}（JSON 必须闭合），'
                    "或直接用自然语言给出最终答案。",
                }
            ],
        }

    def finalize(state: AgentGraphState) -> dict:
        """终点：补最终 assistant 消息。ReactResult 由 harness 从 state 组装。"""
        return {"messages": [{"role": "assistant", "content": state.get("final_answer", "")}]}

    def finalize_budget(state: AgentGraphState) -> dict:
        """业务预算软闸：优雅退出（部分结果 + 原因），对应 W9 的 max_steps 分支。"""
        return {
            "stopped_reason": "max_steps",
            "final_answer": (
                f"[部分结果] 已达最大步数 {state.get('max_steps')}，任务未完成。\n"
                f"已完成部分：\n{_partial_progress(_as_step_records(state))}\n"
                "未完成原因：步数上限触发，模型在最后一步仍在请求工具。"
            ),
        }

    # ============================================================ 路由（纯函数）
    #
    # W11 周二：这三个函数从闭包里提出来成为模块级函数 `_route_after_*`。
    # 理由与 W10 周一「路由是纯函数、不起图就能单测」同源：它们只读 state，
    # 不碰 client、不碰 executor。放在闭包里，测试就只能通过跑整张图来验证
    # 一条边 —— 那是"用集成测试测一个 if"，慢且难定位。

    g = StateGraph(AgentGraphState)
    g.add_node("model_step", model_step)
    g.add_node("tools", tools_node)
    g.add_node("retry", retry_node)
    g.add_node("degrade", degrade_node)
    g.add_node("parse_retry", parse_retry)
    g.add_node("finalize", finalize)
    g.add_node("finalize_degraded", finalize_degraded)
    g.add_node("finalize_budget", finalize_budget)
    g.add_edge(START, "model_step")
    g.add_conditional_edges(
        "model_step",
        _route_after_model,
        {"tools": "tools", "parse_retry": "parse_retry", "finalize": "finalize"},
    )
    g.add_conditional_edges(
        "tools",
        _route_after_tools,
        {
            "model_step": "model_step",
            "retry": "retry",
            "degrade": "degrade",
            "finalize_budget": "finalize_budget",
            "finalize": "finalize",
        },
    )
    g.add_conditional_edges(
        "retry",
        _route_after_retry,
        {
            "retry": "retry",
            "degrade": "degrade",
            "model_step": "model_step",
            "finalize": "finalize",
        },
    )
    g.add_edge("degrade", "finalize_degraded")
    g.add_edge("parse_retry", "model_step")
    g.add_edge("finalize", END)
    g.add_edge("finalize_degraded", END)
    g.add_edge("finalize_budget", END)
    return g.compile()


def result_from_state(
    task_id: str, category: str, model: str, instruction: str, state: AgentGraphState
) -> dict:
    """从图终态组装与 W9 result_to_trajectory 同格式的轨迹 dict（格式对齐验收）。"""
    merged = _merged_steps(state)
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in state.get("usage_log", []):
        for k in usage_total:
            usage_total[k] += u.get(k, 0)
    return {
        "task_id": task_id,
        "category": category,
        "model": model,
        "instruction": instruction,
        "stopped_reason": state.get("stopped_reason", "completed"),
        "completed": state.get("stopped_reason", "completed") == "completed",
        "usage_total": usage_total,
        "steps": [
            {
                "step": m["step"],
                "mode": m["mode"],
                "thought": m.get("thought", ""),
                "model_ms": m.get("model_ms", 0),
                "tokens": m.get("tokens", {}),
                "actions": m.get("actions", []),
                "observation": m.get("observation", ""),
            }
            for m in merged
        ],
        "final_answer": state.get("final_answer", ""),
    }


class _FakeCall:
    """build_tool_messages 需要的 call 形状（name/arguments/id 三件套）。"""

    def __init__(self, call: dict) -> None:
        self.id = call.get("id", "call_0")

        class _Fn:
            def __init__(self, c: dict) -> None:
                self.name = c["name"]
                self.arguments = c["arguments"]

        self.function = _Fn(call)
