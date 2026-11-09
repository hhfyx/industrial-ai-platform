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
from typing import Annotated, Any, Final

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.core.llm.agent_registry import AGENT_TOOL_NAMES, AGENT_TOOLS_SCHEMA, execute_agent_tool
from app.core.llm.budget import (
    BudgetConfig,
    BudgetUsed,
    build_graceful_exit,
    check_budget,
    check_no_progress,
)
from app.core.llm.error_contract import tool_error
from app.core.llm.guardrails import (
    SEMANTIC_REPEAT_LIMIT,
    DedupState,
    GuardrailConfig,
    GuardStats,
    TaskPlan,
    check_dedup,
    decompose_task,
    dedup_nudge,
    self_check,
    validate_action,
)
from app.core.llm.react_loop import (
    _ACTION_HINT_RE,
    MODEL_STEP_TIMEOUT_S,
    PARSE_FAILURE_LIMIT,
    REPEAT_LIMIT,
    TOOL_STEP_TIMEOUT_S,
    UNREPAIRABLE_FAILURE_LIMIT,
    _action_record,
    _canonical_args,
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
    # ---- W11 周三：三道防跑偏的状态 ----
    plan: Any  # TaskPlan | None；None = 分解失败或未开启，校验自动退化
    plan_step_index: int  # 当前未完成步骤下标
    dedup_state: dict  # {counts, samples}：语义判重的累计状态
    guard_stats: dict  # 每道机制自己的命中数与 token（周日算价格的依据）
    # ---- W11 周四：预算与早停 ----
    budget: dict  # BudgetConfig.to_dict()：三道上限（随任务注入）
    budget_used: dict  # BudgetUsed.to_dict()：已用量，每步累加
    seen_result_keys: list  # 已出现过的工具结果语义键（判"有无新信息"）
    budget_exit: dict  # 触顶时的优雅退出载荷（degraded + reason + suggestion）
    degrade_payload: dict  # W11 周二：{answer, missing, degraded, reason}
    stopped_reason: str
    final_answer: str


# ============================================================ 小工具


def _result_semantic_key(tool: str, data: Any) -> str:
    """工具结果的语义键：判"这一步有没有带来新信息"。

    取**结果内容**的指纹而不是调用参数：同一段条文被两次不同查询命中，
    对"还要不要继续查"这个问题来说是同一份信息。这批语料只有 3 份文档，
    重复命中非常常见 —— 用参数判会把"又查了一遍"误当"有进展"，
    那早停条件②就永远不触发。

    截断到 2000 字再哈希：结果长尾（数值列表）会让同一段条文的指纹
    因毫秒级差异而不同，反而制造"新信息"的假象。
    """
    import hashlib

    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)[:2000]
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{tool}:{digest}"


def _bump_progress(state: AgentGraphState, progressed: bool) -> BudgetUsed:
    """按"这一步有没有新信息"更新连续无进展计数。

    `model_step` 已经把步数与 token 累加了；这里只改连续无进展计数。
    分开写是因为"花了钱"与"买到东西"是两件事 —— 前者在 model_step 记账，
    后者要等工具结果回来才知道。
    """
    used = BudgetUsed.from_dict(state.get("budget_used"))
    if progressed:
        used.no_progress_streak = 0
    else:
        used.no_progress_streak += 1
    return used


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
    # W11 周四：预算与早停在这里判（**下一次模型调用之前**）。
    # 顺序：先预算（钱花光了就别再想），再早停（钱还有但没进展）。
    verdict = _budget_verdict(state)
    if verdict.stop:
        return "finalize_budget"
    return "model_step"


def _budget_config(state: AgentGraphState) -> BudgetConfig:
    """从 state 取预算配置，**兼容只有 `max_steps` 的老调用方**。

    为什么必须兼容：W10 的 harness（`run_eval_agent_graph.py`）传的是
    `max_steps=8`，没有 budget 字段。如果这里直接用 `BudgetConfig()` 的默认
    10 步，那同一条任务在"加预算机制"之后会多跑 2 步 —— 成功率与 token
    都会变，而这变化**不是预算造成的，是我改了默认值**。
    那正是今天的坑（把机制的影响归因成别的东西）的镜像版本。

    规则：state 里有 budget 就用它（三道上限齐全）；
    没有就用默认值，但把 `max_steps` 覆盖成调用方传的值。
    """
    raw = state.get("budget")
    if raw:
        cfg = BudgetConfig(**raw)
        # 老调用方可能同时传了 max_steps：以显式 budget 为准，但步数取小者，
        # 防止"传了 budget 又传了更小的 max_steps"时反而放宽。
        legacy_steps = state.get("max_steps")
        if legacy_steps is not None and int(legacy_steps) < cfg.max_steps:
            cfg = BudgetConfig(
                max_tokens=cfg.max_tokens,
                max_steps=int(legacy_steps),
                max_wall_time_s=cfg.max_wall_time_s,
                no_progress_limit=cfg.no_progress_limit,
            )
        return cfg
    legacy_steps = state.get("max_steps")
    if legacy_steps is not None:
        return BudgetConfig(max_steps=int(legacy_steps))
    return BudgetConfig()


def _budget_verdict(state: AgentGraphState) -> Any:
    """汇总三道上限 + 一条早停，给路由用。"""
    cfg = _budget_config(state)
    used = BudgetUsed.from_dict(state.get("budget_used"))
    # 预判下一步开销：用上一步的实际 token 估，没有就用 0（首步）。
    steps = state.get("steps") or []
    last_tokens = 0
    if steps:
        last_tokens = int((steps[-1].get("tokens") or {}).get("total_tokens", 0) or 0)
    verdict = check_budget(used, cfg, next_step_tokens=last_tokens)
    if verdict.stop:
        return verdict
    return check_no_progress(used, cfg)


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


def build_agent_graph(
    client: Any,
    *,
    executor: Any | None = None,
    guardrails: GuardrailConfig | None = None,
):
    """构建评测图。client 走闭包（不在 state 里）；executor 默认 W9 注册表。

    `guardrails` 默认 None = 三道机制全关 = 与 v1.2.0 基线行为一致。
    这是刻意的：W11 的对照实验必须先有"没加机制"的基线可比，
    而三道机制都会改轨迹（多一次 plan 调用、多一条自检提示），
    开着跑出来的数字与 W10 不可比。周日那张"关掉哪一道、损失多少"
    靠的就是这个开关逐个试。
    """
    run_tool = executor if executor is not None else execute_agent_tool
    guards = guardrails or GuardrailConfig()

    async def plan_node(state: AgentGraphState) -> dict:
        """**执行前**的任务分解（W11 周三第一道）。

        必答问题决定了这里是节点而不是后置检查：事后校验只能解释失败，
        事前校验能阻止浪费。一个跑偏 5 步的任务，事后发现时钱已经花了 ——
        而偏离通常在第 1 步就发生（看到"把所有数据导出来"就想写全表查询）。

        分解失败**不阻断任务**：返回 plan=None，整个校验退化成 W10 行为。
        理由是这个调用是"可选的优化"，不是"正确性的前提"——
        为它阻断任务，等于把一次网关抖动升级成任务失败。
        """
        sink: dict[str, int] = {}
        plan = await decompose_task(
            client,
            model=state["model"],
            instruction=state["instruction"],
            tool_names=list(AGENT_TOOL_NAMES),
            usage_sink=sink,
        )
        stats = GuardStats(plan_tokens=sink.get("total_tokens", 0))
        stats.plan_steps = len(plan.steps) if plan else 0
        out: dict[str, Any] = {"plan": plan, "guard_stats": stats.to_dict()}
        if sink:
            out["usage_log"] = [sink]
        return out

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

        # ---- W11 周四：预算累加（每步都记，包括这一步可能是最后一步）----
        used = BudgetUsed.from_dict(state.get("budget_used"))
        used.add_step(
            tokens=usage.get("total_tokens", 0) if usage else 0,
            wall_time_s=model_ms / 1000.0,
            # 「这一步有没有进展」在 model_step 里恒为 True：
            # 模型这一步产出了新内容（工具调用或答案）就是进展。
            # 真正的"无进展"由 tools 节点按**结果内容**判（见 judge_progress）。
            progressed=True,
        )

        out: dict[str, Any] = {
            "step_no": step_no,
            "usage_log": [usage] if usage else [],
            "pending_mode": "final",
            "final_answer": content.strip(),
            "budget_used": used.to_dict(),
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
        # W11 周三：三道机制的 state 载体（关着时全是 None/空，行为不变）
        plan: TaskPlan | None = state.get("plan")
        plan_step_index = int(state.get("plan_step_index", 0) or 0)
        dedup_raw = state.get("dedup_state") or {}
        dedup = DedupState(
            counts=dict(dedup_raw.get("counts", {})), samples=dict(dedup_raw.get("samples", {}))
        )
        stock = state.get("guard_stats") or {}
        guard_stats = GuardStats(**{k: int(stock.get(k, 0) or 0) for k in GuardStats().to_dict()})
        nudge_messages: list[dict] = []
        # W11 周四：结果语义键（判"这一步有没有带来新信息"）
        seen_keys = set(state.get("seen_result_keys", []))
        progressed = False

        for call in calls:
            name, raw_args = call["name"], call["arguments"]
            key = f"{name}|{_canonical_args(raw_args)}"  # W9 口径：规范化后再比
            action_seen[key] = action_seen.get(key, 0) + 1
            occurrence = action_seen[key]

            # ---- 三道机制之一：语义判重（W11 周三，纯字符串，零 token）----
            # 在**旧判重之前**跑：旧判重按 (tool, 规范化参数) 比，语义判重
            # 还要把空格/大小写/同义写法/列表顺序抹平。先跑语义判重，
            # 才能让"换个说法重试同一动作"落到 2 次警告上，而不是漏到 3 次熔断。
            sem_occurrence = 0
            if guards.semantic_dedup:
                sem_occurrence, sem_key = check_dedup(dedup, name, raw_args)
                if sem_occurrence > SEMANTIC_REPEAT_LIMIT:
                    guard_stats.semantic_dup_hits += 1
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
                        "data": {"message": "语义等价的调用已重复超过上限，按判重规则终止。"},
                        "truncated": False,
                        "chars": 0,
                    }
                    tool_messages.extend(build_tool_messages([_FakeCall(call)], [wrapped]))
                    terminated = "repeat_loop"
                    break
                if sem_occurrence == SEMANTIC_REPEAT_LIMIT:
                    guard_stats.semantic_dup_hits += 1
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
                            "message": dedup_nudge(
                                sem_occurrence, sem_key, dedup.samples.get(sem_key, [])
                            )
                        },
                        "truncated": False,
                        "chars": 0,
                    }
                    tool_messages.extend(build_tool_messages([_FakeCall(call)], [wrapped]))
                    continue

            # ---- 三道机制之二：任务分解校验（执行**前**，W11 周三）----
            # 只打约束、**不阻断执行**。为什么不阻断：分解本身可能错，
            # 拿一个可能错的计划去硬拦动作，会把"计划错"变成"任务失败"。
            # 这一档的定位是"提醒 + 留痕"——轨迹里能看到偏离发生了几次，
            # 周日再决定要不要升级成硬拦。
            if guards.plan_check and plan is not None and not terminated:
                step_goal = (
                    plan.steps[plan_step_index].goal
                    if plan_step_index < len(plan.steps)
                    else plan.goal
                )
                dev = validate_action(
                    plan, step_index=plan_step_index, tool=name, arguments=raw_args
                )
                if dev is not None:
                    guard_stats.deviations += 1
                    nudge_messages.append(
                        {
                            "role": "user",
                            "content": f"{dev.constraint}\n（当前步骤目标：{step_goal}）",
                        }
                    )

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

            # ---- 三道机制之三：中间结果自检（W11 周三，纯规则，零 token）----
            # 放在**执行之后、回灌之前**：触发时把"换策略"提示追加到 observation，
            # 模型下一轮看到的就是"这条路不对"，而不是"再试一次"。
            # 注意这里**不筛 ok**：SQL 的"查无批次"是 ok=False + not_found，
            # 那正是最该触发换策略的形态之一。
            if guards.self_check:
                step_goal = (
                    plan.steps[plan_step_index].goal
                    if plan is not None and plan_step_index < len(plan.steps)
                    else state.get("instruction", "")
                )
                check = self_check(
                    tool=name,
                    data=raw_result.get("data"),
                    goal=step_goal,
                    ok=bool(raw_result.get("ok")),
                    error_type=raw_result.get("error_type"),
                )
                if check.verdict != "ok":
                    guard_stats.self_check_hits += 1
                    nudge_messages.append({"role": "user", "content": check.nudge})

            # ---- W11 周四：判"这一步有没有带来新信息"----
            # 用**结果内容的语义键**判，不用调用参数：
            # 同一个查询换个写法返回同样内容 → 不是新信息；
            # 不同查询返回同一段条文 → 也是没进展（这批语料只有 3 份文档，
            # 重复命中很正常）。所以键取工具名 + 结果内容的哈希，不取参数。
            if raw_result.get("ok"):
                result_key = _result_semantic_key(name, raw_result.get("data"))
                if result_key not in seen_keys:
                    seen_keys.add(result_key)
                    progressed = True

        out: dict[str, Any] = {
            "messages": tool_messages + nudge_messages,
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
            # 三道机制的累计状态：只在实际开启时才有意义（关着时保持初值）
            "dedup_state": {"counts": dedup.counts, "samples": dedup.samples},
            "guard_stats": guard_stats.to_dict(),
            "seen_result_keys": sorted(seen_keys),
            "budget_used": _bump_progress(state, progressed).to_dict(),
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

        ## W11 周五补的一条：**已完成的工作不该被降级抹掉**

        1/01 回归实测抓到的真问题：T7（出报告）已经成功调了 `generate_report`，
        但模型随后**自作主张**多调了一次 notify；端点不通 → 走到这里 →
        整个任务被标成 degraded → 判据（要求 completed + 报告生成）失败。

        那不是模型的错，是这个节点的错：它把"最后一个动作失败"当成了
        "整件事没做成"。**降级应当只降级真正缺的那部分**，不能回收已经
        落地的成果 —— 否则"做了 90% 然后多试一步"比"什么都没做"还糟。

        所以这里先判：如果失败的动作属于**可选/收尾**类（notify 就是），
        且此前的必需动作都成功了，那就保留 `completed`，把最后一次失败
        记进 observation 供审计，而不是降级。
        """
        decisions = list(state.get("pending_decisions", []))
        last = decisions[-1] if decisions else {}
        reason = str(last.get("error_type") or "internal")
        merged = _merged_steps(state)
        all_actions = [a for m in merged for a in m.get("actions", [])]

        # ---- 判：是不是"必需的事都做完了，只是收尾动作失败" ----
        if _only_optional_tail_failed(all_actions, last):
            done = sorted({a["tool"] for a in all_actions if a.get("ok")})
            return {
                "stopped_reason": "completed",
                "final_answer": state.get("final_answer") or _completed_with_note(done, last),
            }

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
        """预算触顶或早停时的**优雅退出**（W9 B7 的落地）。

        三件事一起给：已完成部分、未完成原因、建议下一步。缺一件都不叫优雅 ——
        只有原因没有已完成，用户不知道拿到了什么；没有建议，用户不知道该
        加预算还是拆任务。

        `degraded=True` 并计入降级率：预算触顶是**降级**不是失败
        （我们确实产出了一部分），周日那张表要能和"任务失败"分开看。
        """
        cfg = _budget_config(state)
        used = BudgetUsed.from_dict(state.get("budget_used"))
        verdict = _budget_verdict(state)
        reason = verdict.reason or "budget_steps"
        partial = _partial_progress(_as_step_records(state))
        payload = build_graceful_exit(used=used, config=cfg, reason=reason, partial=partial)
        return {
            "budget_exit": payload,
            "degrade_payload": payload,  # 与降级共用同一条轨迹出口
            "stopped_reason": reason,
            "final_answer": (
                f"[部分结果] {payload['missing'][0]}（{verdict.detail}）\n"
                f"已完成部分：\n{partial}\n"
                f"建议下一步：{payload['suggestion']}"
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
    # W11 周三：分解节点只在开启时才进图。**关着的时候图的结构与 v1.2.0
    # 完全一致** —— 多一个直通节点也会让 `get_graph()` 的拓扑变化，
    # 而"结构没变"是对照实验最容易被忽略的那条边。
    if guards.plan_check:
        g.add_node("plan", plan_node)
        g.add_edge(START, "plan")
        g.add_edge("plan", "model_step")
    else:
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
    """从图终态组装与 W9 result_to_trajectory 同格式的轨迹 dict（格式对齐验收）。

    W11 周三：开了任一道防跑偏机制时，额外带 `guard_stats` 与 `plan`。
    **关着的时候绝不带** —— 轨迹多一个键就会让 W10 的对比脚本失配，
    而 W10 复盘里最贵的那个坑正是"静默失配"。
    """
    merged = _merged_steps(state)
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in state.get("usage_log", []):
        for k in usage_total:
            usage_total[k] += u.get(k, 0)
    trace: dict[str, Any] = {
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
    # 只有真的开了机制才带这些键（关着 → 轨迹与 v1.2.0 逐键一致）
    plan = state.get("plan")
    stats = state.get("guard_stats") or {}
    if plan is not None:
        trace["plan"] = {
            "goal": getattr(plan, "goal", ""),
            "steps": [
                {
                    "step_id": s.step_id,
                    "goal": s.goal,
                    "expected_tools": list(s.expected_tools),
                }
                for s in getattr(plan, "steps", [])
            ],
            "out_of_scope": list(getattr(plan, "out_of_scope", [])),
        }
    if stats and any(
        stats.get(k)
        for k in ("plan_tokens", "plan_steps", "deviations", "self_check_hits", "semantic_dup_hits")
    ):
        trace["guard_stats"] = dict(stats)

    # W11 周四：预算信息。**总是带**（与三道机制不同）——
    # 预算判定已经是路由的一部分，任何一次运行都受它影响，
    # 轨迹里没有它就无法解释"为什么停在这里"。
    used = state.get("budget_used") or {}
    if used:
        trace["budget"] = {"config": _budget_config(state).to_dict(), "used": dict(used)}
    if state.get("budget_exit"):
        trace["budget_exit"] = dict(state["budget_exit"])
    return trace


def state_to_react_result(state: AgentGraphState) -> Any:
    """把图终态转成 `ReactResult`，让 `task.verify(result)` 原样工作。

    W11 周三从 `scripts/run_eval_agent_graph.py` 提上来：诱导任务脚本
    （`run_induced_tasks.py`）也要判据，两条路径共用一份转换。
    副本一旦分叉，"同一份判据"就不再是同一份 —— 那是 W10 冻结声明里
    最看重的东西。
    """
    from app.core.llm.react_loop import ReactResult, StepRecord

    stopped = state.get("stopped_reason", "completed")
    steps = [
        StepRecord(
            step=m["step"],
            mode=m["mode"],
            thought=m.get("thought", ""),
            actions=m.get("actions", []),
            model_ms=m.get("model_ms", 0),
            tokens=m.get("tokens", {}),
            observation=m.get("observation", ""),
        )
        for m in _merged_steps(state)
    ]
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in state.get("usage_log", []):
        for k in usage_total:
            usage_total[k] += u.get(k, 0)
    return ReactResult(
        final_answer=state.get("final_answer", ""),
        messages=list(state.get("messages", [])),
        steps=steps,
        stopped_reason=stopped,
        usage_total=usage_total,
    )


#: **收尾类**工具：它们失败不改变"任务是否完成"。
#:
#: `notify` 是唯一的例子，也是最典型的：通知失败意味着"有人没收到消息"，
#: 但数据查到了、报告生成了 —— 这两件事才是任务的主体。
#: 把整个任务降级，等于说"因为没人收到通知，所以这份报告不存在"。
#:
#: 加新工具进这张表要问一句：**它失败会不会让此前成果失去意义？**
#: 会 → 不能进（比如 generate_report 失败就等于报告没生成）。
OPTIONAL_TAIL_TOOLS: Final[frozenset[str]] = frozenset({"notify"})


def _only_optional_tail_failed(actions: list[dict[str, Any]], last: dict[str, Any]) -> bool:
    """判断"必需的事都做完了，只是收尾动作失败"。

    三条同时成立才算：
    1. 最后失败的动作属于收尾类（`OPTIONAL_TAIL_TOOLS`）；
    2. 此前**至少有一个**成功动作（不是从头到尾什么都没做成）；
    3. 没有其它非收尾类的失败（真正的必需动作都成功了）。

    第 3 条是关键：如果 `generate_report` 也失败了，那任务确实没完成，
    必须老实降级。
    """
    failed_tool = str(last.get("tool") or "")
    if failed_tool not in OPTIONAL_TAIL_TOOLS:
        return False
    succeeded = [a for a in actions if a.get("ok")]
    if not succeeded:
        return False
    for action in actions:
        if action.get("ok"):
            continue
        tool = str(action.get("tool") or "")
        if tool not in OPTIONAL_TAIL_TOOLS:
            return False  # 有必需动作失败 → 不能算"只是收尾失败"
    return True


def _completed_with_note(done: list[str], last: dict[str, Any]) -> str:
    """收尾失败但主体完成时的答案文本。

    必须**明说哪个动作没成**，不能悄悄当作全部成功 —— 通知没发出去
    是运营上要处理的事，藏起来比报错更糟。
    """
    tail = last.get("tool") or "收尾动作"
    etype = last.get("error_type") or "internal"
    return (
        "主体任务已完成：" + "、".join(done) + " 均成功。\n"
        f"但收尾动作 {tail} 未成功（{etype}）—— "
        "结论本身有效，这项收尾需要人工补做（例如手动通知）。"
    )


class _FakeCall:
    """build_tool_messages 需要的 call 形状（name/arguments/id 三件套）。"""

    def __init__(self, call: dict) -> None:
        self.id = call.get("id", "call_0")

        class _Fn:
            def __init__(self, c: dict) -> None:
                self.name = c["name"]
                self.arguments = c["arguments"]

        self.function = _Fn(call)
