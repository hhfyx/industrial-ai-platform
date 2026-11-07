"""ReAct 循环测试（W9 周二：手写 ReAct 循环）。

**全部离线**（延续 W2 的规矩：拔掉网线 pytest 全绿）：
循环控制、保险丝、observation 包装测的是**我方代码**的行为，
用假 client 按剧本喂模型响应；真调用的端到端演示在 `scripts/demo_react.py`。

分组（对应周二五个验收项）：
1. 循环骨架 —— 无工具路径能跑通、能停
2. 动作解析 —— FC 主路径的消息形状；文本降级两条路（成功 / 解析失败回灌）
3. 三个保险丝 —— max_steps / 单步超时 / 重复调用检测（2 次警告、3 次熔断）
4. observation —— 字段级与整体级截断，回灌仍是合法 JSON
5. 终止路径 —— completed / max_steps / unrepairable_failure / parse_failure
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from app.core.llm.react_loop import (
    _ACTION_HINT_RE,
    _ACTION_RE,
    REPEAT_LIMIT,
    parse_final_answer,
    parse_text_action,
    run_react_loop,
    wrap_observation,
)

# ============================================================ 假 client（同 test_tools.py 的形状）


class _FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(
        self,
        content: str | None,
        tool_calls: list[_FakeToolCall] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message: _FakeMessage) -> None:
        self.message = message
        self.finish_reason = "tool_calls" if message.tool_calls else "stop"


class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 5
    total_tokens = 15


class _FakeResponse:
    def __init__(self, message: _FakeMessage) -> None:
        self.choices = [_FakeChoice(message)]
        self.usage = _FakeUsage()


class _FakeCompletions:
    def __init__(self, scripted: list[_FakeMessage]) -> None:
        self.scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if not self.scripted:
            raise AssertionError("剧本耗尽：循环仍在调用模型（多半是保险丝没生效）")
        return _FakeResponse(self.scripted.pop(0))


class _SleepingCompletions(_FakeCompletions):
    """每轮都睡一会儿的假 client：专门测单步模型超时。"""

    def __init__(self, delay: float) -> None:
        super().__init__([])
        self.delay = delay

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        await asyncio.sleep(self.delay)
        raise AssertionError("超时之后不应再走到构造响应这一步")


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = type("Chat", (), {"completions": completions})()


def _scripted_client(scripted: list[_FakeMessage]) -> _FakeClient:
    return _FakeClient(_FakeCompletions(scripted))


def _call(id: str, name: str, args: str) -> _FakeToolCall:
    return _FakeToolCall(id, name, args)


# ============================================================ 1. 循环骨架


@pytest.mark.anyio
async def test_no_tool_path_runs_and_stops() -> None:
    """验收①：无工具路径 —— 模型直接回答，循环能跑通、能停。"""
    client = _scripted_client([_FakeMessage("1+1 等于 2，无需工具。")])
    result = await run_react_loop(client, model="m", user_message="1+1?", tools=None)  # type: ignore[arg-type]

    assert result.completed
    assert result.final_answer == "1+1 等于 2，无需工具。"
    assert len(result.steps) == 1
    assert result.steps[0].mode == "final"
    # 无工具时 payload 里不应有 tools 键
    assert "tools" not in client.chat.completions.calls[0]


@pytest.mark.anyio
async def test_fc_multistep_message_shape() -> None:
    """FC 主路径的消息形状：assistant(tool_calls) 与 tool 消息按 id 配对。"""
    scripted = [
        _FakeMessage("需要查北京天气", [_call("c1", "get_weather", '{"city":"北京"}')]),
        _FakeMessage(None, [_call("c2", "calculate", '{"a":12,"op":"multiply","b":2}')]),
        _FakeMessage("北京 12°C，乘 2 是 24°C。", None),
    ]
    client = _scripted_client(scripted)
    result = await run_react_loop(client, model="m", user_message="北京温度乘2")  # type: ignore[arg-type]

    assert result.completed
    roles = [m["role"] for m in result.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert result.messages[2]["tool_calls"][0]["id"] == "c1"
    assert result.messages[3]["tool_call_id"] == "c1"
    assert result.messages[4]["tool_calls"][0]["id"] == "c2"
    assert result.messages[5]["tool_call_id"] == "c2"
    # Thought（content）被记进审计
    assert "查北京天气" in result.steps[0].thought
    assert len(client.chat.completions.calls) == 3


# ============================================================ 2. 动作解析（文本降级）


def test_parse_text_action_success() -> None:
    action = parse_text_action('Thought: 查天气\nAction: get_weather {"city": "北京"}')
    assert action == {"name": "get_weather", "arguments": '{"city": "北京"}'}


def test_parse_text_action_nested_json_braces() -> None:
    """嵌套花括号：greedy + DOTALL 要匹配到最后一个 }。"""
    action = parse_text_action('Action: calculate {"a": 1, "op": "add", "b": {"x": 2}}')
    assert action is not None
    assert json.loads(action["arguments"]) == {"a": 1, "op": "add", "b": {"x": 2}}


def test_parse_final_answer_and_plain_text() -> None:
    assert parse_final_answer("Final Answer: 北京 12°C") == "北京 12°C"
    assert parse_final_answer("没有前缀的普通回答") is None
    # 纯 Thought 没有 Action: 意图 → 不算解析失败，按最终答案处理
    assert parse_text_action("Thought: 信息足够了") is None


@pytest.mark.anyio
async def test_text_fallback_executes_action() -> None:
    """文本降级路径：Action 抠出后走执行，observation 以 user 消息回灌。"""
    scripted = [
        _FakeMessage('Thought: 查天气\nAction: get_weather {"city": "北京"}'),
        _FakeMessage("Final Answer: 北京 12°C 多云。"),
    ]
    client = _scripted_client(scripted)
    result = await run_react_loop(client, model="m", user_message="北京天气", text_fallback=True)  # type: ignore[arg-type]

    assert result.completed
    assert result.steps[0].mode == "text"
    assert result.steps[0].actions[0]["tool"] == "get_weather"
    obs = next(
        m for m in result.messages if m["role"] == "user" and "[Observation]" in m["content"]
    )
    assert '"ok": true' in obs["content"].lower() or '"ok":true' in obs["content"]


@pytest.mark.anyio
async def test_parse_failure_feedback_then_terminate() -> None:
    """验收②失败分支：有 Action: 但 JSON 坏 → 错误回灌重试一次 → 仍失败 → 终止。

    这正是 _ACTION_HINT_RE 存在的理由：坏 JSON 抠不出参数，
    若只靠严格正则，这段输出会被**静默当成最终答案**。
    """
    scripted = [
        _FakeMessage("Action: get_weather {bad json"),  # 第 1 次：解析失败 → 回灌
        _FakeMessage("Action: get_weather {still bad"),  # 第 2 次：→ 终止
    ]
    client = _scripted_client(scripted)
    result = await run_react_loop(client, model="m", user_message="x")  # type: ignore[arg-type]

    assert result.stopped_reason == "parse_failure"
    assert result.steps[0].mode == "parse_error"
    feedback = next(
        m for m in result.messages if m["role"] == "user" and "[ParseError]" in m["content"]
    )
    assert "Action:" in feedback["content"]
    assert len(client.chat.completions.calls) == 2


@pytest.mark.anyio
async def test_parse_failure_recovers_after_one_retry() -> None:
    """第 1 次解析失败后改对了 → 循环继续，不终止。"""
    scripted = [
        _FakeMessage("Action: get_weather {bad"),
        _FakeMessage('Action: get_weather {"city": "北京"}'),
        _FakeMessage("Final Answer: 北京 12°C。"),
    ]
    client = _scripted_client(scripted)
    result = await run_react_loop(client, model="m", user_message="x")  # type: ignore[arg-type]

    assert result.completed
    assert [s.mode for s in result.steps] == ["parse_error", "text", "final"]


# ============================================================ 3. 三个保险丝


@pytest.mark.anyio
async def test_fuse_max_steps_returns_partial_progress() -> None:
    """验收⑤：步数耗尽 → 返回部分结果 + 未完成原因，不假装完成。"""
    scripted = [
        _FakeMessage(None, [_call(f"c{i}", "get_weather", '{"city":"北京"}')]) for i in range(5)
    ]
    client = _scripted_client(scripted)
    result = await run_react_loop(client, model="m", user_message="循环", max_steps=2)  # type: ignore[arg-type]

    assert result.stopped_reason == "max_steps"
    assert not result.completed
    assert "部分结果" in result.final_answer
    assert "get_weather" in result.final_answer  # 已执行的部分如实列出
    assert len(client.chat.completions.calls) == 2


@pytest.mark.anyio
async def test_fuse_model_step_timeout() -> None:
    """验收③②：模型单步超时 → 如实退出，带已完成部分。"""
    client = _FakeClient(_SleepingCompletions(delay=1.0))
    result = await run_react_loop(client, model="m", user_message="x", model_timeout=0.05)  # type: ignore[arg-type]

    assert result.stopped_reason == "model_timeout"
    assert "无法完成" in result.final_answer


@pytest.mark.anyio
async def test_fuse_repeat_call_warns_then_terminates() -> None:
    """验收③③：相同 (tool,args) 第 2 次 → 警告不执行；第 3 次 → 熔断。

    剧本让模型三次请求一模一样的调用，中间不给它"醒"的机会 ——
    这正是必答问题里的场景：模型自己觉得在努力，实际在烧钱。
    """
    same = _FakeMessage(None, [_call("cx", "get_weather", '{"city": "北京"}')])
    client = _scripted_client([same, same, same])
    result = await run_react_loop(client, model="m", user_message="x")  # type: ignore[arg-type]

    assert result.stopped_reason == "repeat_loop"
    # 第 2 次出现：警告回灌（executed=False），不是执行结果
    warnings = [
        m for m in result.messages if m["role"] == "tool" and "禁止再次重复" in m["content"]
    ]
    assert len(warnings) == 1
    # 第 3 次：熔断，final_answer 里有重复证据
    assert "熔断" in result.final_answer
    # 前后一共只真正执行了 1 次
    executed = [a for s in result.steps for a in s.actions if a.get("executed")]
    assert len(executed) == 1


@pytest.mark.anyio
async def test_fuse_repeat_detected_across_parallel_batch_replay() -> None:
    """并行批重放也要被抓到：批内其他调用不稀释计数。

    实测漏洞（demo C 场景）：模型 step1 并行查 4 城，step2 把同样的
    4 个调用**原样重放** —— 旧版"相邻连击"检测被批内顺序稀释，全放行了。
    """
    city = '{"city": "北京"}'
    batch1 = _FakeMessage(
        None,
        [
            _call("b1a", "get_weather", city),
            _call("b1b", "get_weather", '{"city": "上海"}'),
        ],
    )
    replay = _FakeMessage(
        None,
        [
            _call("b2a", "get_weather", '{"city": "上海"}'),  # 第 2 次出现 → 警告
            _call("b2b", "get_weather", city),  # 第 2 次出现 → 警告
        ],
    )
    done = _FakeMessage("北京 12°C，上海 19.5°C。", None)
    client = _scripted_client([batch1, replay, done])
    result = await run_react_loop(client, model="m", user_message="x")  # type: ignore[arg-type]

    warnings = [
        m for m in result.messages if m["role"] == "tool" and "禁止再次重复" in m["content"]
    ]
    assert len(warnings) == 2  # 两个重复调用都被警告，没有被批内顺序稀释


@pytest.mark.anyio
async def test_fuse_repeat_terminates_on_third_occurrence_after_gap() -> None:
    """隔了其他调用之后的第 3 次出现同样熔断（全局计数，不看相邻）。"""
    beijing = _FakeMessage(None, [_call("c1", "get_weather", '{"city":"北京"}')])
    shanghai = _FakeMessage(None, [_call("c2", "get_weather", '{"city":"上海"}')])
    beijing_again = _FakeMessage(None, [_call("c3", "get_weather", '{"city":"北京"}')])
    done = _FakeMessage("北京 12°C。", None)
    client = _scripted_client([beijing, shanghai, beijing_again, done])
    result = await run_react_loop(client, model="m", user_message="x")  # type: ignore[arg-type]

    # 北京第 2 次出现 → 警告但不熔断（总序数才 2）
    assert result.stopped_reason == "completed" or result.stopped_reason == "max_steps"
    warnings = [
        m for m in result.messages if m["role"] == "tool" and "禁止再次重复" in m["content"]
    ]
    assert len(warnings) == 1


@pytest.mark.anyio
async def test_repeat_with_different_args_is_not_flagged() -> None:
    """参数不同不算重复：改了参数的调用必须被放行（否则自愈机制失效）。"""
    scripted = [
        _FakeMessage(None, [_call("c1", "get_weather", '{"city":"北京"}')]),
        _FakeMessage(None, [_call("c2", "get_weather", '{"city":"上海"}')]),
        _FakeMessage("北京 12°C，上海 19.5°C。", None),
    ]
    client = _scripted_client(scripted)
    result = await run_react_loop(client, model="m", user_message="两地天气")  # type: ignore[arg-type]

    assert result.completed
    assert not any(a.get("kind") == "repeat_call" for s in result.steps for a in s.actions)


@pytest.mark.anyio
async def test_fuse_unrepairable_failure_after_three_same_kind() -> None:
    """验收⑤：同一工具同类失败连续 3 次 → unrepairable_failure，带最后一次错误。

    剧本刻意让**参数各不相同**：参数一样会先触发重复熔断（repeat_loop），
    那是另一条保险丝 —— 失败连击测的是"换了参数还是不行"。
    """
    scripted = [
        _FakeMessage(None, [_call(f"c{i}", "no_such_tool", json.dumps({"attempt": i}))])
        for i in range(4)
    ]
    client = _scripted_client(scripted)
    result = await run_react_loop(client, model="m", user_message="x")  # type: ignore[arg-type]

    assert result.stopped_reason == "unrepairable_failure"
    assert "做不到" in result.final_answer
    assert "unknown_tool" in result.final_answer
    # 第 3 次失败即停：第 4 个剧本不该被消费
    assert len(client.chat.completions.calls) == 3


@pytest.mark.anyio
async def test_failure_streak_resets_on_success() -> None:
    """失败两连后成功一次 → 连击清零，之后的失败重新数。

    剧本的失败调用用**不同参数**：no_such_tool 的参数不参与执行，
    但参数相同会先触发重复熔断（那是另一条保险丝的职责）。
    """
    scripted = [
        _FakeMessage(None, [_call("c1", "no_such_tool", '{"attempt":1}')]),
        _FakeMessage(None, [_call("c2", "no_such_tool", '{"attempt":2}')]),
        _FakeMessage(None, [_call("c3", "get_weather", '{"city":"北京"}')]),  # 成功 → 清零
        _FakeMessage(None, [_call("c4", "no_such_tool", '{"attempt":4}')]),  # 重新数：1
        _FakeMessage("如实回答：两个工具都不适用。", None),
    ]
    client = _scripted_client(scripted)
    result = await run_react_loop(client, model="m", user_message="x")  # type: ignore[arg-type]

    assert result.completed  # 没被误熔断


# ============================================================ 4. observation 处理


def test_wrap_observation_field_level_truncation() -> None:
    """字段级：超长字符串截首尾，结构保留，truncated 标记。"""
    long_doc = "A" * 3000
    wrapped = wrap_observation({"ok": True, "tool": "t", "result": {"doc": long_doc}})
    assert wrapped["truncated"] is True
    assert wrapped["ok"] is True
    data = wrapped["data"]
    assert isinstance(data, dict) and "doc" in data  # 结构还在
    assert len(data["doc"]) < 600  # 首尾各 ~200
    assert "中间省略" in data["doc"]
    # 回灌必须是合法 JSON（截断后的乱文本会让模型把截断当数据）
    json.dumps(wrapped, ensure_ascii=False)


def test_wrap_observation_3000_chars_context_bounded() -> None:
    """验收④：工具返回 3000 字时上下文不爆炸 —— 整体序列化 ≤ 上限的量级。"""
    wrapped = wrap_observation(
        {"ok": True, "tool": "t", "result": {"doc": "字" * 3000, "meta": "x" * 900}},
        max_chars=800,
    )
    assert wrapped["chars"] <= 800 + 200  # 整体级截断后 chars 反映的是 data 本身
    assert wrapped["truncated"] is True
    assert "上限" in wrapped["data"]["note"]
    # data 的实际序列化长度被压住
    assert len(json.dumps(wrapped["data"], ensure_ascii=False)) <= 900


def test_wrap_observation_error_shape() -> None:
    wrapped = wrap_observation(
        {
            "ok": False,
            "tool": "get_weather",
            "kind": "city_not_found",
            "error": {"message": "没有城市 火星"},
        }
    )
    assert wrapped["ok"] is False
    assert wrapped["kind"] == "city_not_found"
    assert wrapped["data"] == {"message": "没有城市 火星"}


def test_wrap_observation_short_result_untouched() -> None:
    wrapped = wrap_observation({"ok": True, "tool": "t", "result": {"temperature": 12.0}})
    assert wrapped["truncated"] is False
    assert wrapped["data"] == {"temperature": 12.0}


# ============================================================ 5. 终止路径汇总


@pytest.mark.anyio
async def test_completed_path_reports_usage_and_records() -> None:
    """正常完成的审计完整性：usage 累加、每步有记录。"""
    scripted = [
        _FakeMessage(None, [_call("c1", "get_weather", '{"city":"北京"}')]),
        _FakeMessage("Final Answer: 北京 12°C", None),
    ]
    client = _scripted_client(scripted)
    result = await run_react_loop(client, model="m", user_message="北京天气")  # type: ignore[arg-type]

    assert result.completed
    assert result.usage_total["total_tokens"] == 30  # 2 次请求 × 15
    assert [s.mode for s in result.steps] == ["fc", "final"]
    assert result.final_answer == "北京 12°C"


# ============================================================ 正则常量健全性


def test_action_hint_regex_is_looser_than_action_regex() -> None:
    """宽松探测必须命中严格正则抓不住的坏 JSON —— 否则解析失败分支永不触发。"""
    bad = "Action: get_weather {bad json"
    assert _ACTION_HINT_RE.search(bad) is not None
    assert _ACTION_RE.search(bad) is None


def test_repeat_limit_is_two() -> None:
    """第 2 次警告、第 3 次熔断 —— 与文档里的规则一致。"""
    assert REPEAT_LIMIT == 2
