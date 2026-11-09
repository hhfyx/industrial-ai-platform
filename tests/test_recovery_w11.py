"""W11 周二：恢复策略矩阵、retry 与 degrade 两个节点、故障注入与三次演练。

## 这个文件守什么

周一的契约测试守"错误分类对不对"，本文件守**"错了以后怎么办"**：

1. **矩阵定版** —— 每一类错误 × 动作 × 上限，不许有无上限的格子；
2. **两条边可走** —— 图里真的存在 `tools → retry` 与 `tools/retry → degrade`；
3. **降级说实话** —— `{answer, missing, degraded, reason}` 四个字段都在，
   且 `answer` 里不含编造数值（不再进 LLM）；
4. **三次演练** —— 超时退避后成功 / 空结果换关键词 / 内部异常两次降级，
   三条轨迹各自可见预期形状。

## 演练为什么用图而不只测矩阵

矩阵是纯函数，测它只能证明"判定对了"。而今天的交付是**图上的两条边**——
判定对了但边没连上，等于没做。所以演练全部走 `build_agent_graph()` +
脚本化假 client，跑完整图，再从 `result_from_state` 的轨迹里核对。
"""

from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.llm.agent_graph import build_agent_graph, result_from_state
from app.core.llm.error_contract import ERROR_TYPES, tool_error, tool_ok
from app.core.llm.recovery import (
    RECOVERY_MATRIX,
    build_degrade_payload,
    decide,
    render_degrade_text,
    spec_of,
)
from tests.fault_injection import FaultInjector, make_fault_envelope, normalize_mode

# ============================================================ 假 client（剧本回放）


def _msg(content: str = "", tool_calls: Any = None) -> Any:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _call(name: str, arguments: str, call_id: str = "call_1") -> Any:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


USAGE = SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120)


def _resp(*, content: str = "", tool_calls: Any = None) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=_msg(content=content, tool_calls=tool_calls))],
        usage=USAGE,
    )


class ScriptedClient:
    """按剧本回放；用尽后重复最后一条（避免无限循环时崩掉）。"""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if len(self.script) > 1:
            return self.script.pop(0)
        return self.script[0]

    @property
    def chat(self) -> Any:
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))


def _invoke(app: Any, *, max_steps: int = 8, thread: str = "w11-tue") -> dict[str, Any]:
    return asyncio.run(
        app.ainvoke(
            {
                "instruction": "演练",
                "model": "fake",
                "max_steps": max_steps,
                "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            },
            config={"configurable": {"thread_id": thread}},
        )
    )


def _tool_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [a for m in state.get("steps", []) for a in m.get("actions", [])]


# ============================================================ 1. 策略矩阵


def test_matrix_covers_every_error_type() -> None:
    """**每一类 error_type 都必须在矩阵里有一格，且每格都有上限。**

    封住"无限重试"的口子：`limit` 缺失或为负都算违规。`limit=0` 是合法的，
    意思是"一次都不该再试"（权限错、幂等冲突）。
    """
    assert set(RECOVERY_MATRIX) == set(ERROR_TYPES), "矩阵与契约的 error_type 不一致"
    for etype, spec in RECOVERY_MATRIX.items():
        assert spec.limit >= 0, f"{etype} 的 limit 不能为负"
        assert spec.action in ("auto_retry", "model_fix", "skip_and_tell", "degrade")
        if spec.action == "auto_retry":
            assert spec.limit >= 1, f"{etype} 说自动重试却没有额度"
            assert spec.by_code, f"{etype} 是代码重试，by_code 必须为 True"
        if spec.action == "model_fix":
            assert not spec.by_code, f"{etype} 需要模型改策略，代码不该替它改"
            assert spec.limit <= 1, f"{etype} 交给模型改参只该给 1 次机会"


def test_matrix_no_unbounded_retry() -> None:
    """矩阵里**任何一格**都不能表示"一直重试"。

    这条对着 W9 B7 的判词写：让它继续直到成功 = 会自己续费的故障。
    上限必须是有限整数，且 <= 2（一次失败 + 一次重试是本周的定版口径）。
    """
    for etype, spec in RECOVERY_MATRIX.items():
        assert isinstance(spec.limit, int)
        assert 0 <= spec.limit <= 2, f"{etype} 上限 {spec.limit} 超出本周口径（<=2）"


def test_matrix_actions_match_documented_table() -> None:
    """矩阵与文档里的那张表逐格一致（防止"改了代码忘改文档"）。"""
    expected = {
        "invalid_json": "model_fix",
        "invalid_arg": "model_fix",
        "unknown_tool": "model_fix",
        "not_found": "model_fix",
        "empty_result": "model_fix",
        "timeout": "auto_retry",
        "internal": "auto_retry",
        "not_permitted": "skip_and_tell",
        "conflict": "skip_and_tell",
    }
    actual = {k: v.action for k, v in RECOVERY_MATRIX.items()}
    assert actual == expected


# ============================================================ 2. decide()：判定与执行分离


def test_decide_retries_timeout_once_then_degrades() -> None:
    """超时：第 1 次 → auto_retry（带退避），第 2 次 → degrade（用尽）。"""
    counts: dict[str, int] = {}
    env = tool_error("query_metrics_sql", "timeout", "超时")
    first = decide(env, counts, rng=random.Random(7))
    assert first.action == "auto_retry"
    assert first.limit == 1
    assert first.delay_s is not None and first.delay_s > 0

    second = decide(env, counts, rng=random.Random(7))
    assert second.action == "degrade"
    assert second.attempt == 2
    assert second.delay_s is None


def test_decide_backoff_has_jitter() -> None:
    """退避必须带 jitter：同样的尝试次数，多次取样的延迟不该是同一个数。

    这不是"为了好看"——同步的固定延迟会让批量任务在同一时刻集体重试，
    把瞬时故障放大成雪崩（W2 D3 的结论）。测试用两个不同的 rng 种子取样。
    """
    delays = {
        round(decide(tool_error("t", "timeout", "x"), {}, rng=random.Random(s)).delay_s, 4)
        for s in range(8)
    }
    assert len(delays) > 1, "退避没有抖动：所有取样都相同"


def test_decide_param_error_returns_to_model() -> None:
    """参数错：交回模型（代码不知道该改成什么），上限 1。"""
    d = decide(tool_error("retrieve_spec_chunks", "invalid_arg", "参数错"), {})
    assert d.action == "model_fix"
    assert d.by_code is False
    assert d.limit == 1
    assert d.delay_s is None


def test_decide_permission_degrades_immediately() -> None:
    """权限错：`limit=0` 一用即尽 → 立刻降级，**一次都不重试**。

    为什么不是"跳过并告知后继续"：权限错意味着**这块数据永远拿不到**，
    而它往往是任务的必需部分。让模型继续，它只会拿同样的参数再撞一次，
    然后触发重复熔断 —— 那是一条更差的退出路径（`repeat_loop`），
    轨迹上看起来像"模型乱来"，实际是矩阵没给出收场方式。
    """
    counts: dict[str, int] = {}
    d = decide(tool_error("notify", "not_permitted", "403"), counts)
    assert d.action == "degrade"
    assert d.limit == 0
    assert d.attempt == 1
    assert d.delay_s is None
    assert spec_of("not_permitted").degrade_when_unresolved is True


def test_decide_conflict_skips_without_degrading() -> None:
    """幂等冲突：效果**已经发生**，任务其实完成了 → 跳过并告知，不标降级。

    这是 `degrade_when_unresolved` 与 `action` 分成两个字段的理由：
    权限错与冲突都是 limit=0，但一个该降级、一个不该。
    """
    d = decide(tool_error("notify", "conflict", "已投递过"), {})
    assert d.action == "skip_and_tell"
    assert d.limit == 0
    assert spec_of("conflict").degrade_when_unresolved is False


def test_decide_counts_per_tool_and_error_type() -> None:
    """计数按 (tool, error_type)：换措辞重试同一个坏工具拦得住，别的工具不牵连。"""
    counts: dict[str, int] = {}
    for _ in range(2):
        decide(tool_error("query_metrics_sql", "timeout", "x"), counts)
    assert decide(tool_error("query_metrics_sql", "timeout", "x"), counts).action == "degrade"
    assert decide(tool_error("notify", "timeout", "x"), counts).action == "auto_retry"


def test_decide_respects_envelope_retryable_false() -> None:
    """信封说不可重试时，矩阵不许擅自重试（两层结论必须一致）。

    `internal` 的矩阵动作是 `auto_retry`；信封把它标成 retryable=False
    （典型场景：重试预算已被 `apply_retry_budget` 抑制）。此时矩阵必须
    退化成 `degrade` —— 否则就是"代码判定不能重试，策略层却还想去重试"，
    两层打架的结果是重试预算在任何一层都不生效。
    """
    env = tool_error("query_metrics_sql", "internal", "内部错")
    env["retryable"] = False
    d = decide(env, {})
    assert d.action == "degrade"
    assert d.delay_s is None


# ============================================================ 3. 降级载荷：说实话


def test_degrade_payload_has_four_fields() -> None:
    """降级答案必须是 `{answer, missing, degraded, reason}` 四条。"""
    payload = build_degrade_payload(
        reason="timeout",
        actions=[
            {"tool": "retrieve_spec_chunks", "ok": True},
            {"tool": "query_metrics_sql", "ok": False, "error_type": "timeout"},
        ],
    )
    assert set(payload) == {"answer", "missing", "degraded", "reason"}
    assert payload["degraded"] is True
    assert payload["reason"] == "timeout"
    assert payload["missing"] == ["query_metrics_sql（timeout）"]
    assert "notify" not in payload["missing"]


def test_degrade_payload_never_fabricates_numbers() -> None:
    """降级的 answer **不出现任何具体数值** —— 它只描述"拿到了什么"。

    这条对着今天的坑写：把降级写成"随便给个通用回答"。更隐蔽的一种坏法
    是让 LLM 润色部分结论 —— 缺的那块会被它顺手补上，答案看起来完整了。
    所以降级文本里一个数字都不该有（数值只能来自成功工具的 data，
    而这里刻意不复述 data）。
    """
    payload = build_degrade_payload(
        reason="internal",
        actions=[{"tool": "retrieve_spec_chunks", "ok": True}],
    )
    text = render_degrade_text(payload)
    for digit in ("42.5", "36.1", "8.2", "11.4", "40.0"):
        assert digit not in text


def test_degrade_payload_no_success_says_so() -> None:
    """一次都没成功时，answer 必须明说"没有取得任何结论"。"""
    payload = build_degrade_payload(
        reason="timeout",
        actions=[{"tool": "query_metrics_sql", "ok": False, "error_type": "timeout"}],
    )
    assert "没有取得任何可用结论" in payload["answer"]


def test_degrade_payload_unresolved_is_authoritative() -> None:
    """`unresolved` 直接替换 missing（不求交集）。

    第一版写成求交集，结果 missing 恒为空 —— 降级退化成"只说失败不说缺什么"。
    这条测试就是那个 bug 的墓碑。
    """
    payload = build_degrade_payload(
        reason="not_permitted",
        actions=[{"tool": "query_metrics_sql", "ok": False, "error_type": "not_permitted"}],
        unresolved=["query_metrics_sql（not_permitted：未授权 batches 表）"],
    )
    assert payload["missing"] == ["query_metrics_sql（not_permitted：未授权 batches 表）"]


def test_degrade_text_flags_incompleteness() -> None:
    """渲染出来的文本必须有"未完成 / 不要当完整结论"的显式声明。"""
    payload = build_degrade_payload(reason="timeout", actions=[])
    text = render_degrade_text(payload)
    assert "降级回答" in text
    assert "未完成" in text
    assert "请勿把它当作完整结论" in text


# ============================================================ 4. 图上的两条边


def _state_with_failure(**state: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "instruction": "t",
        "model": "fake",
        "max_steps": 8,
        "step_no": 1,
        "messages": [],
        "steps": [],
    }
    base.update(state)
    return base


def test_route_after_tools_goes_to_retry() -> None:
    """tools 之后：有待重试 → retry 边。"""
    from app.core.llm.agent_graph import _route_after_tools as route

    state = _state_with_failure(
        pending_decisions=[{"action": "auto_retry"}], retry_queue=[{"call": {}, "delay_s": 0.01}]
    )
    assert route(state) == "retry"


def test_route_after_tools_goes_to_degrade() -> None:
    """tools 之后：预算用尽 → degrade 边（即使队列里还有别的重试）。"""
    from app.core.llm.agent_graph import _route_after_tools as route

    state = _state_with_failure(
        pending_decisions=[{"action": "auto_retry"}, {"action": "degrade"}],
        retry_queue=[{"call": {}, "delay_s": 0.01}],
    )
    assert route(state) == "degrade"


def test_route_after_retry_loops_or_degrades() -> None:
    """retry 之后：还能重试就再进 retry，用尽就 degrade，否则回模型。"""
    from app.core.llm.agent_graph import _route_after_retry as route

    assert (
        route(
            _state_with_failure(
                pending_decisions=[{"action": "auto_retry"}],
                retry_queue=[{"call": {}, "delay_s": 0.01}],
            )
        )
        == "retry"
    )
    assert route(_state_with_failure(pending_decisions=[{"action": "degrade"}])) == "degrade"
    assert route(_state_with_failure(pending_decisions=[])) == "model_step"


def test_graph_has_retry_and_degrade_nodes() -> None:
    """图里两个节点真的存在，且边连上了（编译后看 graph 的节点表）。"""
    app = build_agent_graph(ScriptedClient([_resp(content="ok")]))
    nodes = {n for n in app.get_graph().nodes if not n.startswith("__")}
    assert {"retry", "degrade", "finalize_degraded"} <= nodes


# ============================================================ 5. 故障注入骨架


def test_fault_injector_three_modes() -> None:
    """三种模式（timeout / empty / internal_error）都能注入，且信封同形。"""

    async def inner(name: str, raw: str) -> dict[str, Any]:
        return tool_ok(name, {"rows": [1]})

    async def scenario() -> None:
        for mode in ("timeout", "empty", "internal_error"):
            fi = FaultInjector(inner, failures={1: mode})
            r = await fi("query_metrics_sql", "{}")
            if mode == "empty":
                assert r["ok"] is True and r["outcome"] == "empty_result"
            else:
                assert r["ok"] is False
                assert r["error_type"] == ("internal" if mode == "internal_error" else "timeout")
            # 第 2 次起放行
            assert (await fi("query_metrics_sql", "{}"))["ok"] is True

    asyncio.run(scenario())


def test_fault_injector_counts_per_tool() -> None:
    """计数按工具各记一份：一次任务里 retrieve 的失败不该让 notify 第一次就失败。"""

    async def inner(name: str, raw: str) -> dict[str, Any]:
        return tool_ok(name, {})

    async def scenario() -> None:
        fi = FaultInjector(inner, failures={2: "timeout"})
        await fi("retrieve_spec_chunks", "{}")  # 第 1 次，放行
        assert (await fi("notify", "{}"))["ok"] is True  # notify 自己的第 1 次，放行
        assert (await fi("retrieve_spec_chunks", "{}"))["ok"] is False  # 第 2 次，注入
        assert fi.attempts("retrieve_spec_chunks") == 2

    asyncio.run(scenario())


def test_fault_injector_envelope_shapes_match_real_tools() -> None:
    """注入信封与真实工具的信封**同键** —— 否则演练验的是注入器自己。"""
    for mode in ("timeout", "internal_error", "not_permitted", "invalid_arg"):
        env = make_fault_envelope("query_metrics_sql", mode)
        assert {"ok", "tool", "error_type", "message", "hint", "retryable", "retry_after"} <= set(
            env
        )
    empty = make_fault_envelope("retrieve_spec_chunks", "empty")
    assert {"ok", "tool", "data", "outcome", "hint"} <= set(empty)


def test_fault_injector_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="未知注入模式"):
        make_fault_envelope("t", "nonexistent")


def test_normalize_mode_maps_contract_types() -> None:
    assert normalize_mode("internal") == "internal_error"
    assert normalize_mode("timeout") == "timeout"
    assert normalize_mode("empty_result") == "empty"


# ============================================================ 6. 三次演练（端到端图）


def test_drill_timeout_then_success() -> None:
    """**演练一**：超时 → 代码退避后成功，轨迹可见 2 次调用。

    剧本：模型第 1 步请求查库 → 注入 timeout → 图走 retry 节点（代码重试）
    → 第 2 次成功 → 模型拿到结果给最终答案。

    验收要点：
    - 轨迹里同一次工具调用出现 **2 条** action 记录；
    - 第 2 条带 `retried=True`（否则看不出是代码重试还是模型又调了一次）；
    - 第 1 条 `error_type=timeout`、第 2 条 `ok=True`；
    - 最终 `stopped_reason == completed`（没有降级）。
    """
    from app.core.llm.agent_registry import execute_agent_tool

    injector = FaultInjector(execute_agent_tool, failures={1: "timeout"})
    client = ScriptedClient(
        [
            _resp(tool_calls=[_call("query_metrics_sql", '{"table": "inspections", "limit": 1}')]),
            _resp(content="完成：拿到 1 行。"),
        ]
    )
    app = build_agent_graph(client, executor=injector)
    state = _invoke(app)

    tool_calls = [a for a in _tool_actions(state) if a["tool"] == "query_metrics_sql"]
    assert len(tool_calls) == 2, f"应当看到 2 次调用（首次 + 代码重试），实际 {len(tool_calls)}"
    assert tool_calls[0]["ok"] is False and tool_calls[0]["error_type"] == "timeout"
    assert tool_calls[1]["ok"] is True
    assert tool_calls[1].get("retried") is True
    assert state.get("stopped_reason", "completed") == "completed"

    trace = result_from_state("drill", "drill", "fake", "i", state)
    assert len([a for m in trace["steps"] for a in m["actions"]]) == 2


def test_drill_empty_result_falls_back_to_new_keyword() -> None:
    """**演练二**：空结果 → 模型换关键词成功。

    空结果是 `ok=true`，所以**不该**触发 retry 节点 —— 它走的是
    "模型自己换关键词"这条路（model_fix）。这条演练验的正是
    "矩阵里空结果交给模型"这一格：图不该替模型重试，但模型确实拿到了
    空结果 + hint，于是换了词。

    验收：两次调用、第一次 `outcome=empty_result`、第二次成功、
    参数**确实不同**（模型换了关键词）。
    """
    from app.core.llm.agent_registry import execute_agent_tool

    injector = FaultInjector(execute_agent_tool, failures={1: "empty"})
    client = ScriptedClient(
        [
            _resp(tool_calls=[_call("retrieve_spec_chunks", '{"query": "抗压强度 判定"}', "c1")]),
            _resp(tool_calls=[_call("retrieve_spec_chunks", '{"query": "抗压 限值"}', "c2")]),
            _resp(content="完成：按新关键词拿到了条款。"),
        ]
    )
    app = build_agent_graph(client, executor=injector)
    state = _invoke(app)

    calls = [a for a in _tool_actions(state) if a["tool"] == "retrieve_spec_chunks"]
    assert len(calls) == 2
    assert calls[0]["ok"] is True and calls[0]["kind"] == "ok"  # 空结果不是失败
    assert calls[1]["ok"] is True
    assert calls[0]["args"] != calls[1]["args"], "模型应当换了关键词，而不是原样重发"
    assert state.get("stopped_reason", "completed") == "completed"


def test_drill_internal_twice_degrades() -> None:
    """**演练三**：内部异常连续 2 次 → 降级并标 degraded。

    剧本：模型请求查库 → 注入 internal → retry 节点重试 → 又注入 internal
    → 预算用尽 → degrade 节点 → finalize_degraded。

    验收要点：
    - `stopped_reason == "degraded"`；
    - state 里有 `degrade_payload`，四个字段齐全且 `degraded=True`；
    - final_answer 是**代码拼的**降级文本（不是模型说的）；
    - 轨迹里 2 次调用都失败，第 2 次带 retried 标记。
    """
    from app.core.llm.agent_registry import execute_agent_tool

    injector = FaultInjector(
        execute_agent_tool, failures={1: "internal_error", 2: "internal_error"}
    )
    client = ScriptedClient(
        [_resp(tool_calls=[_call("query_metrics_sql", '{"table": "inspections"}')])]
    )
    app = build_agent_graph(client, executor=injector)
    state = _invoke(app)

    assert state["stopped_reason"] == "degraded"
    payload = state["degrade_payload"]
    assert set(payload) == {"answer", "missing", "degraded", "reason"}
    assert payload["degraded"] is True
    assert payload["reason"] == "internal"
    assert payload["missing"], "降级必须说明缺了什么"
    assert "[降级回答]" in state["final_answer"]
    assert "未完成" in state["final_answer"]

    calls = [a for a in _tool_actions(state) if a["tool"] == "query_metrics_sql"]
    assert len(calls) == 2
    assert all(c["ok"] is False for c in calls)
    assert calls[1].get("retried") is True


def test_drill_permission_degrades_without_retry() -> None:
    """补充演练：权限错**一次都不重试**，直接降级。

    与演练三对照：同样 2 次"卡住"，权限错只打 1 次包（重试无意义），
    内部异常打 2 次（重试可能好）。这个差异就是矩阵存在的意义。
    """
    from app.core.llm.agent_registry import execute_agent_tool

    injector = FaultInjector(execute_agent_tool, failures={1: "not_permitted", 2: "not_permitted"})
    client = ScriptedClient(
        [_resp(tool_calls=[_call("query_metrics_sql", '{"table": "batches"}')])]
    )
    app = build_agent_graph(client, executor=injector)
    state = _invoke(app)

    calls = [a for a in _tool_actions(state) if a["tool"] == "query_metrics_sql"]
    assert len(calls) == 1, "权限错不该重试"
    assert state["stopped_reason"] == "degraded"
    assert state["degrade_payload"]["reason"] == "not_permitted"
