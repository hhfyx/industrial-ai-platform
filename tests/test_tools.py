"""Function Calling 工具层测试（W1 周二：schema / 注册表 / 安全执行）。

**全部离线**：这里测的是"模型给了不可信输入之后，我们这侧能不能守住"，
不需要模型参与。真调用（含并行、多轮）在 `scripts/demo_tools.py`。

分组：
1. schema 生成 —— 两份 schema 的形态，且与 Pydantic 模型**不会漂移**
2. 本地实现 —— 直接调用可用（不经过模型）
3. execute_tool 的四类失败 —— 每一类都必须是"返回数据"而不是"抛异常"
4. 消息构造 —— tool_call_id 配对（漏了它是本课最经典的错）
5. run_tool_loop 的循环控制 —— 用假 client，验证轮数上限与不调工具的分支
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.core.llm.tool_loop import MAX_TOOL_ROUNDS, run_tool_loop
from app.core.llm.tools import (
    TOOL_REGISTRY,
    TOOL_SPECS,
    TOOLS_SCHEMA,
    ToolExecutionError,
    build_tool_messages,
    calculate,
    execute_tool,
    execute_tool_calls,
    get_weather,
)

# ============================================================ 1. schema


def test_tools_schema_has_two_tools_with_when_to_use() -> None:
    """验收点：两份 schema，且 description 里写了「什么时候该用」。"""
    assert len(TOOLS_SCHEMA) == 2
    names = [item["function"]["name"] for item in TOOLS_SCHEMA]
    assert names == ["get_weather", "calculate"]

    for item in TOOLS_SCHEMA:
        desc = item["function"]["description"]
        # 只写"做什么"不够，必须写触发场景，否则模型不会主动选它
        assert "当" in desc, f"{item['function']['name']} 的 description 没写触发场景"
        assert len(desc) > 50, "description 太短，模型选工具的依据不足"


def test_schema_derived_from_pydantic_not_handwritten() -> None:
    """schema 的 parameters 必须来自 Pydantic，否则一定漂移。

    断言两件事：① enum 出现在 schema 里（Literal 才有的效果）；
    ② schema 里的 required 与模型一致。
    """
    by_name = {item["function"]["name"]: item["function"]["parameters"] for item in TOOLS_SCHEMA}

    weather = by_name["get_weather"]
    assert weather["properties"]["unit"]["enum"] == ["celsius", "fahrenheit"]
    assert weather["required"] == ["city"]

    calc = by_name["calculate"]
    assert calc["properties"]["op"]["enum"] == ["add", "subtract", "multiply", "divide"]
    assert calc["required"] == ["a", "op", "b"]
    assert calc["properties"]["a"]["type"] == "number"


def test_spec_count_matches_registry() -> None:
    """规格表与注册表必须一一对应：漏注册一个工具，模型看得见但执行不了。"""
    assert {spec.name for spec in TOOL_SPECS} == set(TOOL_REGISTRY)


# ============================================================ 2. 本地实现


def test_get_weather_celsius_and_fahrenheit() -> None:
    c = get_weather("北京")
    assert c["temperature"] == 12.0
    assert c["unit_label"] == "°C"
    assert c["source"] == "fake-dataset"  # 明示假数据，防止被当真

    f = get_weather("北京", "fahrenheit")
    # 12°C = 53.6°F
    assert f["temperature"] == pytest.approx(53.6)
    assert f["unit_label"] == "°F"


def test_get_weather_accepts_english_and_unknown_city_raises() -> None:
    assert get_weather("beijing")["temperature"] == 12.0
    with pytest.raises(ToolExecutionError) as exc:
        get_weather("火星")
    assert exc.value.kind == "city_not_found"


def test_calculate_four_operations() -> None:
    assert calculate(1, "add", 2)["result"] == 3
    assert calculate(1, "subtract", 2)["result"] == -1
    assert calculate(3, "multiply", 4)["result"] == 12
    assert calculate(7, "divide", 2)["result"] == 3.5


def test_calculate_divide_by_zero_and_overflow() -> None:
    with pytest.raises(ToolExecutionError) as exc:
        calculate(1, "divide", 0)
    assert exc.value.kind == "division_by_zero"

    # 溢出成 inf 会写出非法 JSON（Infinity），必须拦
    with pytest.raises(ToolExecutionError) as exc:
        calculate(1e308, "multiply", 10)
    assert exc.value.kind == "overflow"


# ============================================================ 3. 安全执行
#
# 核心断言：**任何输入都不许抛异常**。抛了就等于让一个幻觉出来的工具名
# 打挂整个请求，而模型连"我错了"的机会都没有。


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("get_weather", '{"city":"杭州"}'),  # 正常
        ("calculate", '{"a":"abc","op":"add","b":1}'),  # 类型错
        ("calculate", '{"a":1,"op":"add"}'),  # 缺必填
        ("no_such_tool", "{}"),  # 工具名不存在
        ("get_weather", "{bad json"),  # JSON 坏
        ("get_weather", "[1,2]"),  # 不是对象
        ("get_weather", ""),  # 空串
        ("calculate", '{"a":1,"op":"divide","b":0}'),  # 业务失败
        ("get_weather", '{"city":"火星"}'),  # 业务失败
        ("calculate", '{"a":1,"op":"power","b":2}'),  # 枚举外
    ],
)
def test_execute_tool_never_raises(name: str, raw: str) -> None:
    """十种输入，一律返回 dict。这是"闸门"该有的行为。"""
    result = execute_tool(name, raw)
    assert isinstance(result, dict)
    assert "ok" in result
    assert "tool" in result
    # 结果必须可 JSON 序列化 —— 它马上就要变成 tool 消息的 content
    json.dumps(result, ensure_ascii=False)


def test_execute_tool_success_shape() -> None:
    result = execute_tool("get_weather", '{"city":"杭州"}')
    assert result["ok"] is True
    assert result["tool"] == "get_weather"
    assert result["result"]["temperature"] == 17.0


def test_unknown_tool_lists_available() -> None:
    result = execute_tool("send_email", '{"to":"a@b.c"}')
    assert result["ok"] is False
    assert result["kind"] == "unknown_tool"
    # 光说"没有"没用，得告诉模型有哪些可用，它才能换
    assert set(result["error"]["available_tools"]) == {"get_weather", "calculate"}


def test_invalid_arguments_reports_field_and_got() -> None:
    result = execute_tool("calculate", '{"a":"abc","op":"add","b":1}')
    assert result["kind"] == "invalid_arguments"
    field = result["error"]["fields"][0]
    assert field["field"] == "a"
    assert field["got"] == "abc"


def test_string_numbers_are_coerced_not_rejected() -> None:
    """Pydantic lax 模式：`"12"` 能转成 12.0。这是刻意的 ——

    模型经常把数字写成字符串，为此报错会让模型白跑一轮。真正转不了的 `"abc"`
    才是错误（见上一个用例）。
    """
    result = execute_tool("calculate", '{"a":"12","op":"add","b":"8"}')
    assert result["ok"] is True
    assert result["result"]["result"] == 20


def test_execute_tool_calls_preserves_order() -> None:
    """顺序必须与输入一致 —— 回灌是按位置配对的。"""

    class _Fn:
        def __init__(self, name: str, arguments: str) -> None:
            self.name = name
            self.arguments = arguments

    class _Call:
        def __init__(self, id: str, name: str, arguments: str) -> None:
            self.id = id
            self.function = _Fn(name, arguments)

    calls = [
        _Call("id_bj", "get_weather", '{"city":"北京"}'),
        _Call("id_sh", "get_weather", '{"city":"上海"}'),
    ]
    results = execute_tool_calls(calls)
    assert [r["result"]["city"] for r in results] == ["北京", "上海"]


# ============================================================ 4. 消息构造


def test_build_tool_messages_pairs_by_id() -> None:
    """**本课最经典的错**：漏 tool_call_id，模型对不上号。"""

    class _Fn:
        def __init__(self, arguments: str) -> None:
            self.name = "get_weather"
            self.arguments = arguments

    class _Call:
        def __init__(self, id: str, arguments: str) -> None:
            self.id = id
            self.function = _Fn(arguments)

    calls = [_Call("call_1", '{"city":"北京"}'), _Call("call_2", '{"city":"上海"}')]
    results = [
        {"ok": True, "tool": "get_weather", "result": {"city": "北京"}},
        {"ok": True, "tool": "get_weather", "result": {"city": "上海"}},
    ]

    messages = build_tool_messages(calls, results)

    assert len(messages) == 2
    assert [m["role"] for m in messages] == ["tool", "tool"]
    assert [m["tool_call_id"] for m in messages] == ["call_1", "call_2"]
    # content 必须是**字符串**，不能是 dict（协议要求 string）
    assert all(isinstance(m["content"], str) for m in messages)
    # 中文不能被转成 unicode 转义，否则日志不可读
    assert "北京" in messages[0]["content"]


def test_build_tool_messages_rejects_length_mismatch() -> None:
    """数量不匹配要立刻炸：静默错配比崩溃危险得多。"""

    class _Fn:
        name = "get_weather"
        arguments = "{}"

    class _Call:
        id = "call_1"
        function = _Fn()

    with pytest.raises(ValueError, match="数量不匹配"):
        build_tool_messages([_Call()], [])


# ============================================================ 5. 循环控制
#
# 用一个假 client 走 run_tool_loop 的分支，不需要真实模型。


class _FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content: str | None, tool_calls: list[_FakeToolCall] | None) -> None:
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
        self.scripted = scripted
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse(self.scripted.pop(0))


class _FakeClient:
    def __init__(self, scripted: list[_FakeMessage]) -> None:
        self.chat = type("Chat", (), {"completions": _FakeCompletions(scripted)})()


@pytest.mark.anyio
async def test_loop_stops_immediately_when_model_does_not_call_tools() -> None:
    """模型直接回答：不调工具、只发一次请求。"""
    client = _FakeClient([_FakeMessage("1+1=2", None)])
    result = await run_tool_loop(client, model="m", user_message="1+1?")  # type: ignore[arg-type]

    assert result.tool_calls_count == 0
    assert result.rounds == 1
    assert result.stopped_reason == "completed"
    assert result.final_message == "1+1=2"
    assert len(client.chat.completions.calls) == 1


@pytest.mark.anyio
async def test_loop_executes_and_feeds_back_then_finishes() -> None:
    """单轮闭环的消息顺序：assistant(tool_calls) 在前，tool 结果在后。"""
    scripted = [
        _FakeMessage(None, [_FakeToolCall("c1", "get_weather", '{"city":"北京"}')]),
        _FakeMessage("北京 12°C", None),
    ]
    client = _FakeClient(scripted)
    result = await run_tool_loop(client, model="m", user_message="北京天气")  # type: ignore[arg-type]

    roles = [m["role"] for m in result.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert result.messages[2]["tool_calls"][0]["id"] == "c1"
    assert result.messages[3]["tool_call_id"] == "c1"
    assert result.tool_calls_count == 1
    assert result.rounds == 2


@pytest.mark.anyio
async def test_loop_respects_max_rounds() -> None:
    """模型一直要调工具时，必须在 max_rounds 处停下 —— 否则无限计费。"""
    scripted = [
        _FakeMessage(None, [_FakeToolCall(f"c{i}", "get_weather", '{"city":"北京"}')])
        for i in range(10)
    ]
    client = _FakeClient(scripted)
    result = await run_tool_loop(client, model="m", user_message="循环", max_rounds=2)  # type: ignore[arg-type]

    assert result.rounds == 2
    assert result.stopped_reason == "max_rounds"
    assert len(client.chat.completions.calls) == 2
    assert MAX_TOOL_ROUNDS >= 1


@pytest.mark.anyio
async def test_loop_reports_tool_error_back_to_model() -> None:
    """执行失败也是数据：错误必须出现在 tool 消息里，供模型下一轮自愈。"""
    scripted = [
        _FakeMessage(None, [_FakeToolCall("c1", "no_such_tool", "{}")]),
        _FakeMessage("抱歉，我换个方式回答", None),
    ]
    client = _FakeClient(scripted)
    result = await run_tool_loop(client, model="m", user_message="x")  # type: ignore[arg-type]

    tool_message = next(m for m in result.messages if m["role"] == "tool")
    payload = json.loads(tool_message["content"])
    assert payload["ok"] is False
    assert payload["kind"] == "unknown_tool"
