"""W11 周四：预算与早停的机制测试。

## 分五组

1. **三道上限** —— token / 步数 / 墙钟，各自独立可触发，不能互相替代；
2. **三条早停** —— 路由可判，不靠模型自觉；
3. **优雅退出** —— 已完成 + 原因 + 建议，三件缺一不可；
4. **归因纪律** —— 预算触顶的失败必须能**单独计数**（今天的坑）；
5. **图接线** —— 预算真的进了路由（纯函数对了不等于接线对了，周三的教训）。

## 为什么第 4 组单列

任务原文的坑是："把预算设得太低显得省。成功率会掉，而你在表里会把它
归因成任务太难。"代码层能做的只有一件事：让 `stopped_reason` 把
预算触顶**显式标出来**，使那个归因无处藏身。所以这组测试守的是
"分类没被合并"。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.llm.budget import (
    BUDGET_REASONS,
    DEFAULT_BUDGET,
    BudgetConfig,
    BudgetUsed,
    build_graceful_exit,
    check_budget,
    check_no_progress,
    judge_progress,
)

# ============================================================ 1. 三道上限


def test_token_limit_triggers() -> None:
    """token 触顶：已用 + 预计 > 上限。"""
    used = BudgetUsed(tokens=14_000, steps=2)
    v = check_budget(used, BudgetConfig(max_tokens=15_000), next_step_tokens=2_000)
    assert v.stop and v.reason == "budget_tokens"


def test_token_limit_predicts_before_overspending() -> None:
    """**预判**：还没超就该停，不是超了才停。

    这是"预算"与"记账"的区别：花完再判意味着已经超了，
    而预判能在跨过上限之前停手。
    """
    used = BudgetUsed(tokens=13_000, steps=2)
    # 已用 13k 没超 15k，但下一步预计 3k 会超 → 该停
    assert check_budget(used, BudgetConfig(max_tokens=15_000), next_step_tokens=3_000).stop
    # 预计 1k 不超 → 继续
    assert not check_budget(used, BudgetConfig(max_tokens=15_000), next_step_tokens=1_000).stop


def test_step_limit_triggers() -> None:
    used = BudgetUsed(tokens=100, steps=10)
    v = check_budget(used, BudgetConfig(max_steps=10))
    assert v.stop and v.reason == "budget_steps"


def test_wall_time_limit_triggers() -> None:
    used = BudgetUsed(tokens=100, steps=2, wall_time_s=121.0)
    v = check_budget(used, BudgetConfig(max_wall_time_s=120.0))
    assert v.stop and v.reason == "budget_wall_time"


def test_three_limits_are_independent() -> None:
    """三道**不能互相替代**：token 不涨但步数涨、步数不多但很慢，都是真实形态。"""
    # token 很少但步数满了
    assert check_budget(BudgetUsed(tokens=10, steps=10), BudgetConfig(max_steps=10)).stop
    # token 多但步数没满、时间没超
    assert check_budget(BudgetUsed(tokens=20_000, steps=1), BudgetConfig(max_tokens=15_000)).stop
    # 只是慢
    assert check_budget(
        BudgetUsed(tokens=10, steps=1, wall_time_s=200), BudgetConfig(max_wall_time_s=120)
    ).stop


def test_budget_ok_when_under_all_limits() -> None:
    v = check_budget(BudgetUsed(tokens=1000, steps=2, wall_time_s=5), BudgetConfig())
    assert not v.stop and v.reason == ""


def test_budget_used_accumulates() -> None:
    used = BudgetUsed()
    used.add_step(tokens=1000, wall_time_s=5.0, progressed=True)
    used.add_step(tokens=2000, wall_time_s=7.0, progressed=True)
    assert used.tokens == 3000
    assert used.steps == 2
    assert used.wall_time_s == pytest.approx(12.0)


def test_budget_used_roundtrip() -> None:
    """序列化往返（进 state 必须可序列化 —— W10 周二的纪律）。"""
    used = BudgetUsed(tokens=1234, steps=3, wall_time_s=9.5, no_progress_streak=1)
    restored = BudgetUsed.from_dict(used.to_dict())
    assert restored == used
    assert BudgetUsed.from_dict(None) == BudgetUsed()


# ============================================================ 2. 三条早停


def test_no_progress_triggers_after_limit() -> None:
    """连续 2 步无新信息 → 早停。给一步"换个角度"的机会。"""
    cfg = BudgetConfig(no_progress_limit=2)
    assert not check_no_progress(BudgetUsed(no_progress_streak=1), cfg).stop
    v = check_no_progress(BudgetUsed(no_progress_streak=2), cfg)
    assert v.stop and v.reason == "no_progress"


def test_progress_resets_streak() -> None:
    """一旦有新信息，连续计数清零（不能把整场任务的累计当成连续）。"""
    used = BudgetUsed()
    used.add_step(tokens=100, wall_time_s=1, progressed=False)
    used.add_step(tokens=100, wall_time_s=1, progressed=False)
    assert used.no_progress_streak == 2
    used.add_step(tokens=100, wall_time_s=1, progressed=True)
    assert used.no_progress_streak == 0


def test_judge_progress_uses_semantic_keys() -> None:
    """新信息 = 出现**之前没有过**的语义键。

    同一个查询换个写法返回同样内容 → 不是新信息；
    不同查询返回同一段条文（这批语料只有 3 份文档）→ 也不是新信息。
    用字符串比会把"再查一遍"误当"有进展"，那早停永远不触发。
    """
    assert judge_progress({"a"}, {"a", "b"}) is True
    assert judge_progress({"a", "b"}, {"a"}) is False
    assert judge_progress({"a"}, {"a"}) is False
    assert judge_progress(set(), {"a"}) is True


def test_early_stop_is_decidable_in_route_state() -> None:
    """三条早停都必须能**从 state 判定**，不依赖模型说什么。

    这条守的是"不靠模型自觉"：`check_budget` / `check_no_progress` 的入参
    只有 `BudgetUsed`（纯数据），拿不到 messages 或模型输出 ——
    类型层面就堵住了"让模型决定何时停"这条路。
    """
    import inspect

    for fn in (check_budget, check_no_progress):
        params = set(inspect.signature(fn).parameters)
        assert params <= {"used", "config", "next_step_tokens"}, (
            f"{fn.__name__} 的入参不该包含模型输出：{params}"
        )


# ============================================================ 3. 优雅退出


def test_graceful_exit_has_three_parts() -> None:
    """已完成 + 原因 + 建议，三件缺一不可。"""
    payload = build_graceful_exit(
        used=BudgetUsed(tokens=15_000, steps=4),
        config=BudgetConfig(),
        reason="budget_tokens",
        partial="已查到 B20241207-01 的抗压强度 42.5 MPa",
    )
    assert payload["degraded"] is True
    assert payload["reason"] == "budget_tokens"
    assert "42.5" in payload["answer"]  # 已完成部分
    assert payload["missing"] and "token" in payload["missing"][0]  # 未完成原因
    assert payload["suggestion"]  # 建议下一步


def test_graceful_exit_suggestion_is_reason_specific() -> None:
    """不同触顶原因给**不同**建议 —— 一句通用建议等于没建议。"""
    base = {"used": BudgetUsed(tokens=1, steps=1), "config": BudgetConfig(), "partial": "x"}
    tok = build_graceful_exit(reason="budget_tokens", **base)["suggestion"]
    steps = build_graceful_exit(reason="budget_steps", **base)["suggestion"]
    wall = build_graceful_exit(reason="budget_wall_time", **base)["suggestion"]
    assert tok != steps != wall
    assert "token" in tok and "步" in steps and "耗时" in wall


def test_graceful_exit_without_partial_says_so() -> None:
    """没有任何产出时必须明说，不能用含糊话。"""
    payload = build_graceful_exit(
        used=BudgetUsed(), config=BudgetConfig(), reason="budget_steps", partial=""
    )
    assert "没有产出" in payload["answer"]


def test_graceful_exit_counts_as_degraded() -> None:
    """预算触顶计入降级率（`degraded=True`），不是失败。

    理由：我们确实产出了一部分。把它当失败，周日那张表就分不清
    "做不出来"与"没钱做完"。
    """
    payload = build_graceful_exit(
        used=BudgetUsed(steps=10), config=BudgetConfig(), reason="budget_steps", partial="部分"
    )
    assert payload["degraded"] is True


# ============================================================ 4. 归因纪律（今天的坑）


def test_budget_reasons_are_separable() -> None:
    """预算触顶必须是**独立可枚举**的 stopped_reason。

    今天的坑：预算设低 → 任务半路被截 → 判据失败 → 归因成"任务太难"。
    代码层唯一能做的是让这三类在轨迹里**长得不一样**，
    使那个归因无法偷懒。
    """
    assert {"budget_tokens", "budget_steps", "budget_wall_time"} == BUDGET_REASONS
    # 与"模型侧失败"不重叠
    assert BUDGET_REASONS.isdisjoint(
        {"completed", "unrepairable_failure", "repeat_loop", "parse_failure", "model_timeout"}
    )


def test_no_progress_is_not_a_budget_reason() -> None:
    """早停（无进展）与预算触顶**分开计**。

    两者的处置不同：预算触顶该加预算，无进展该换策略。
    混成一类，周日就分不清"该加钱"还是"该改流程"。
    """
    assert "no_progress" not in BUDGET_REASONS


# ============================================================ 5. 图接线


def _state(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "instruction": "t",
        "model": "fake",
        "max_steps": 8,
        "step_no": 1,
        "messages": [],
        "steps": [],
    }
    base.update(kw)
    return base


def test_route_stops_on_token_budget() -> None:
    """路由：token 触顶 → finalize_budget（不是继续调模型）。"""
    from app.core.llm.agent_graph import _route_after_tools

    state = _state(
        budget=BudgetConfig(max_tokens=1000).to_dict(),
        budget_used=BudgetUsed(tokens=1200, steps=1).to_dict(),
    )
    assert _route_after_tools(state) == "finalize_budget"


def test_route_stops_on_no_progress() -> None:
    from app.core.llm.agent_graph import _route_after_tools

    state = _state(
        budget=BudgetConfig(no_progress_limit=2).to_dict(),
        budget_used=BudgetUsed(tokens=100, steps=3, no_progress_streak=2).to_dict(),
    )
    assert _route_after_tools(state) == "finalize_budget"


def test_route_continues_when_under_budget() -> None:
    from app.core.llm.agent_graph import _route_after_tools

    state = _state(
        budget=BudgetConfig(max_tokens=15_000).to_dict(),
        budget_used=BudgetUsed(tokens=1000, steps=2).to_dict(),
    )
    assert _route_after_tools(state) == "model_step"


def test_legacy_max_steps_still_honoured() -> None:
    """老调用方只传 `max_steps=8`（W10 harness 的形态）时，预算要认它。

    为什么必须兼容：如果这里用默认的 10 步，那同一条任务在"加预算机制"
    之后会多跑 2 步 —— 成功率与 token 都会变，而这变化**不是预算造成的，
    是默认值变了**。那正是今天的坑的镜像版本。
    """
    from app.core.llm.agent_graph import _budget_config

    cfg = _budget_config(_state(max_steps=8))
    assert cfg.max_steps == 8
    # 传了 budget 又传更小的 max_steps → 取小者（不能反而放宽）
    cfg2 = _budget_config(_state(max_steps=5, budget=BudgetConfig(max_steps=10).to_dict()))
    assert cfg2.max_steps == 5


def test_default_budget_documented_as_measured() -> None:
    """`DEFAULT_BUDGET` 是**定档结果**，必须能追溯到曲线。

    这条测试守的是纪律：定档不许凭空改。它不校验具体数字对不对
    （那是曲线的事），只校验配置里存在三道上限且都是正数。
    """
    assert DEFAULT_BUDGET.max_tokens > 0
    assert DEFAULT_BUDGET.max_steps > 0
    assert DEFAULT_BUDGET.max_wall_time_s > 0
    assert DEFAULT_BUDGET.no_progress_limit >= 1


# ============================================================ 6. 端到端：预算真的会停


class _StepEatingClient:
    """每步固定花 tokens_per_step；中间步都带工具调用（避免被当成最终答案）。"""

    def __init__(self, tokens_per_step: int = 5000) -> None:
        self.tokens_per_step = tokens_per_step
        self.calls = 0

    async def _create(self, **kwargs: Any) -> Any:
        self.calls += 1
        usage = SimpleNamespace(
            prompt_tokens=self.tokens_per_step - 50,
            completion_tokens=50,
            total_tokens=self.tokens_per_step,
        )
        msg = SimpleNamespace(
            content=f"第 {self.calls} 步",
            tool_calls=[
                SimpleNamespace(
                    id=f"c{self.calls}",
                    function=SimpleNamespace(
                        name="query_metrics_sql",
                        arguments=f'{{"table": "inspections", "limit": {self.calls + 1}}}',
                    ),
                )
            ],
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)

    @property
    def chat(self) -> Any:
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))


def test_budget_actually_stops_the_loop() -> None:
    """端到端：小预算下任务必须被截停，且 `stopped_reason` 是 budget_*。

    这条是"纯函数对了不等于接线对了"的补丁（周三的教训）。
    """
    from app.core.llm.agent_graph import build_agent_graph, result_from_state

    app = build_agent_graph(_StepEatingClient(tokens_per_step=8000))
    state = asyncio.run(
        app.ainvoke(
            {
                "instruction": "无限取数测试",
                "model": "fake",
                "budget": BudgetConfig(max_tokens=10_000, max_steps=20).to_dict(),
                "messages": [{"role": "user", "content": "u"}],
            },
            config={"configurable": {"thread_id": "budget::stop"}},
        )
    )
    assert state["stopped_reason"] in BUDGET_REASONS, state["stopped_reason"]
    trace = result_from_state("t", "c", "fake", "i", state)
    assert "budget_exit" in trace
    payload = trace["budget_exit"]
    assert payload["degraded"] is True
    assert payload["suggestion"]


def test_budget_exit_marks_trajectory() -> None:
    """触顶时轨迹里带 `budget_exit`，且 `stopped_reason` 可区分于任务失败。"""
    from app.core.llm.agent_graph import build_agent_graph, result_from_state

    app = build_agent_graph(_StepEatingClient(tokens_per_step=9000))
    state = asyncio.run(
        app.ainvoke(
            {
                "instruction": "t",
                "model": "fake",
                "budget": BudgetConfig(max_tokens=10_000, max_steps=20).to_dict(),
                "messages": [{"role": "user", "content": "u"}],
            },
            config={"configurable": {"thread_id": "budget::trace"}},
        )
    )
    trace = result_from_state("t", "c", "fake", "i", state)
    assert trace["stopped_reason"].startswith("budget_")
    assert trace["budget"]["config"]["max_tokens"] == 10_000
    assert trace["budget"]["used"]["tokens"] > 0
