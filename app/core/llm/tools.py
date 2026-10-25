"""Function Calling 的工具层：schema、Pydantic 校验、注册表、安全执行。

**必答问题的答案就藏在这个文件的边界上。**

模型从来不会执行任何函数。它只产出一个 `tool_calls` 列表，每项是
「工具名 + 一段 JSON 字符串」，描述它**想**调什么。真正的执行一律发生在这里 ——
在**你这侧**。所以这个文件是"模型和真实世界之间唯一的闸门"，必须假设送进来的
一切都是不可信的：

| 模型可能干的事 | 如果不管会怎样 | 这里怎么处理 |
| --- | --- | --- |
| 点名一个不存在的工具 | `KeyError` 崩掉整个请求 | 返回结构化错误，让模型改 |
| 参数类型不对（`a="abc"`） | `TypeError` / 脏数据写库 | Pydantic 拦下，错误回灌 |
| 参数的 JSON 本身坏了 | `json.JSONDecodeError` 崩掉 | 同上 |
| 参数合法但业务失败（城市不存在、除以零） | 异常穿透到路由变成 500 | 由 handler 抛 `ToolExecutionError` |
| 一次返回多个 tool_call | 只处理第一个 | 循环处理并按 id 配对回灌 |

**错误必须是"数据"而不是"异常"**：把错误当作 `role="tool"` 的内容喂回去，模型
下一轮能自己改参数重试；直接抛出去则用户只看到一个 500，而模型连"我错了"都不知道。
这是 Agent 循环能自愈的前提。

**单一事实来源**：`TOOL_SPECS` 同时生成 OpenAI 的 `tools` schema 和 Pydantic 校验模型。
两边分开写迟早会漂移 —— 模型按 schema 生成了参数，校验模型却不认，那种 bug 最难查。
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Literal

from pydantic import BaseModel, Field, ValidationError

# ============================================================== 参数模型
#
# 复用 W1 的习惯：参数的**唯一权威定义**是 Pydantic 模型。
# OpenAI 的 JSON schema 由 `model_json_schema()` 自动生成，不手写 —— 手写就一定会漂移。


class GetWeatherArgs(BaseModel):
    """`get_weather` 的入参。

    `unit` 用 `Literal` 而不是 `str`：这样 schema 里会带 `enum`，模型看到枚举
    就不会瞎编 `"C"` / `"摄氏度"` 这类值。**约束写在类型里，比写在 description 里可靠**。
    """

    city: str = Field(
        min_length=1, max_length=32, description="城市名，中文或英文，例如 北京 / Shanghai"
    )
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="温度单位：celsius（摄氏度，默认）或 fahrenheit（华氏度）",
    )


class CalculateArgs(BaseModel):
    """`calculate` 的入参。

    `a` / `b` 声明成 `float`：Pydantic 的 lax 模式会把 `"12"` 这类字符串数字
    自动转成 `12.0` —— 这正是我们要的，因为模型经常把数字写成字符串。
    但 `"abc"` 转不了，会在校验阶段被拦下（这就是"参数类型错"的边界用例）。
    """

    a: float = Field(description="第一个操作数")
    op: Literal["add", "subtract", "multiply", "divide"] = Field(
        description="运算符：add（加）subtract（减）multiply（乘）divide（除）"
    )
    b: float = Field(description="第二个操作数（除数不能为 0）")


# ============================================================== 执行期异常
#
# 与"校验失败"分开：校验失败是**参数形状**问题，执行失败是**业务语义**问题。
# 两者回灌给模型的措辞不同，模型据此做的修正动作也不同。


class ToolExecutionError(Exception):
    """工具执行阶段的业务失败（城市不存在、除以 0 等）。

    由 handler 主动抛出，`execute_tool` 捕获后转成结构化的 `role="tool"` 内容。
    **不要**拿它当编程错误的容器 —— 真正的 bug 该崩就崩，吞掉只会把问题藏得更深。
    """

    def __init__(self, message: str, *, kind: str = "execution_error") -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind


# ============================================================== 工具实现
#
# 都是"假数据/本地实现"，不联网。get_weather 用一张固定的气温表。

# 假天气数据：城市 -> 摄氏温度。真实系统这里会是 HTTP 请求。
_FAKE_WEATHER_C: Final[dict[str, float]] = {
    "北京": 12.0,
    "上海": 19.5,
    "深圳": 26.0,
    "杭州": 17.0,
    "成都": 15.5,
    "beijing": 12.0,
    "shanghai": 19.5,
}


def get_weather(city: str, unit: str = "celsius") -> dict[str, Any]:
    """查询某城市的假天气。查不到抛 `ToolExecutionError`（交给模型去改城市名）。"""
    key = city.strip()
    celsius = _FAKE_WEATHER_C.get(key) or _FAKE_WEATHER_C.get(key.lower())
    if celsius is None:
        available = "、".join(sorted(_FAKE_WEATHER_C))
        raise ToolExecutionError(
            f"没有城市 {city!r} 的天气数据；当前支持：{available}",
            kind="city_not_found",
        )

    if unit == "fahrenheit":
        temperature = round(celsius * 9 / 5 + 32, 2)
        unit_label = "°F"
    else:
        temperature = round(celsius, 2)
        unit_label = "°C"

    return {
        "city": city,
        "temperature": temperature,
        "unit": unit,
        "unit_label": unit_label,
        "condition": "多云",
        "source": "fake-dataset",  # 明示这是假数据，别被当成真实天气
    }


_OPERATIONS: Final[dict[str, Callable[[float, float], float]]] = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
}


def calculate(a: float, op: str, b: float) -> dict[str, Any]:
    """四则运算。除以 0 抛 `ToolExecutionError`。

    为什么不拦 `inf`：`1e308 * 10` 会溢出成 `inf`，JSON 序列化会写出 `Infinity` ——
    那不是合法 JSON，下游解析器会崩。所以这里显式判 `isfinite`。
    """
    func = _OPERATIONS.get(op)
    if func is None:  # 理论上到不了这里（Literal 已经拦过），留作防御
        raise ToolExecutionError(f"不支持的运算符 {op!r}", kind="unsupported_operation")

    if op == "divide" and b == 0:
        raise ToolExecutionError("除数不能为 0", kind="division_by_zero")

    value = func(a, b)
    if not math.isfinite(value):
        raise ToolExecutionError(f"结果溢出（{a} {op} {b} 不是有限数）", kind="overflow")

    return {"a": a, "op": op, "b": b, "result": round(value, 10)}


# ============================================================== 工具描述
#
# **description 是模型选工具的唯一依据，也是本课最重要的手写内容。**
# 它必须回答"什么时候该用"，而不只是"这个工具做什么"。
# 只写"查询天气"的话，模型遇到"北京冷不冷"可能自己去猜；写清触发场景它才会调。

GET_WEATHER_DESCRIPTION: Final[str] = (
    "查询指定城市的实时天气和温度，数据是本地的假数据。"
    "当用户询问某个城市**现在/今天**的天气、气温、多少度、冷不冷、热不热、"
    "要不要带伞、要不要加衣服时，调用这个工具。"
    "支持中文和英文城市名。"
    "不要用它查历史天气或未来多天预报（本工具只返回单个当前值），"
    "也不要凭常识替用户猜温度——凡是要给出具体温度都必须调用本工具。"
)

CALCULATE_DESCRIPTION: Final[str] = (
    "对两个数字做一次精确的四则运算（加/减/乘/除）。"
    "当用户要求算数、需要确切的数值结果、或要对**前面工具返回的数值**做二次计算"
    "（例如把摄氏温度换算成华氏、求两地温差、算平均值）时，调用这个工具。"
    "每次调用只做一步运算；多步运算请连续调用多次。"
    "不要用它做开方、三角函数、百分比等本工具不支持的运算。"
    "**不要心算**：凡是要向用户给出确切数字，都应该走本工具，而不是自己在回答里估。"
)


# ============================================================== 规格 + 注册表


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """一个工具的完整描述：给模型看的 schema + 给执行用的校验模型和函数。

    把三样东西绑在一起（而不是建三个平行 dict），是为了让"加一个工具"只有
    **一个地方**要改。三张表最容易出的问题是"注册了但 schema 没写"、
    "schema 有但校验模型漏了" —— 合成一个结构后，漏项在类型层面就不成立了。
    """

    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[..., dict[str, Any]]

    def to_openai_schema(self) -> dict[str, Any]:
        """生成 OpenAI `tools` 参数里的一项。

        `parameters` 直接由 Pydantic 的 `model_json_schema()` 生成，**不手写**：
        手写必然和校验模型漂移，而漂移的后果是"模型按 schema 生成参数、
        校验模型拒绝"——错误信息会指向参数，实际错在 schema。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


TOOL_SPECS: Final[tuple[ToolSpec, ...]] = (
    ToolSpec(
        name="get_weather",
        description=GET_WEATHER_DESCRIPTION,
        args_model=GetWeatherArgs,
        func=get_weather,
    ),
    ToolSpec(
        name="calculate",
        description=CALCULATE_DESCRIPTION,
        args_model=CalculateArgs,
        func=calculate,
    ),
)

# 名字 -> 规格。执行时按名字查这里。
TOOL_REGISTRY: Final[dict[str, ToolSpec]] = {spec.name: spec for spec in TOOL_SPECS}

# 直接给 `client.chat.completions.create(tools=...)` 用的列表。
TOOLS_SCHEMA: Final[list[dict[str, Any]]] = [spec.to_openai_schema() for spec in TOOL_SPECS]


# ============================================================== 安全执行
#
# 这是全篇最关键的函数：把模型给的**不可信** input 变成可以喂回去的**确定性**结果。


def execute_tool(name: str, raw_arguments: str) -> dict[str, Any]:
    """执行一个 tool_call，**永不抛异常**，失败也返回结构化结果。

    返回形状（无论成败都是 dict，这样回灌时不用分支）：

        成功: {"ok": True,  "tool": name, "result": {...}}
        失败: {"ok": False, "tool": name, "error": {...}, "kind": "..."}

    四类失败分别处理，因为它们对应模型**不同**的修正动作：

    | 情况 | kind | 模型应该做什么 |
    | --- | --- | --- |
    | 工具名不存在 | `unknown_tool` | 换一个工具，或直接回答用户 |
    | 参数 JSON 坏了 | `invalid_arguments_json` | 重新生成合法 JSON |
    | 参数类型/必填不满足 | `invalid_arguments` | 按 `fields` 里的提示改参数 |
    | 业务失败 | 由 handler 定（如 `city_not_found`） | 换参数值再试，或告知用户 |

    `raw_arguments` 刻意收 **str** 而不是 dict：模型返回的就是一段字符串，
    先 `json.loads` 再校验这条路必须由执行侧走一遍 —— 提前 parse 等于替模型
    把错误吞了，而这类错误恰恰是模型下一轮要据此改正的信息。
    """
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return {
            "ok": False,
            "tool": name,
            "kind": "unknown_tool",
            "error": {
                "message": f"没有名为 {name!r} 的工具",
                "available_tools": sorted(TOOL_REGISTRY),
            },
        }

    # ---- ① JSON 解析 ----
    try:
        parsed = json.loads(raw_arguments) if raw_arguments.strip() else {}
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "tool": name,
            "kind": "invalid_arguments_json",
            "error": {
                "message": f"参数不是合法 JSON：{exc.msg}（位置 {exc.pos}）",
                "raw": raw_arguments[:200],
            },
        }

    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "tool": name,
            "kind": "invalid_arguments",
            "error": {"message": f"参数必须是 JSON 对象，收到 {type(parsed).__name__}"},
        }

    # ---- ② Pydantic 校验 ----
    # 这是"参数类型错"的关卡：模型给 `{"a": "abc"}` 会在这里被拦，
    # 而不是到 calculate 里变成 TypeError。
    try:
        args = spec.args_model.model_validate(parsed)
    except ValidationError as exc:
        # 只把 field / reason 回灌，不回灌 pydantic 的完整 str ——
        # 那一坨多行文本会吃掉大量 token，而且模型并不需要 loc 的完整路径。
        fields = [
            {
                "field": ".".join(str(p) for p in err["loc"]),
                "reason": err["msg"],
                "got": err.get("input"),
            }
            for err in exc.errors()
        ]
        return {
            "ok": False,
            "tool": name,
            "kind": "invalid_arguments",
            "error": {"message": "参数不符合工具签名", "fields": fields},
        }

    # ---- ③ 真正执行 ----
    try:
        result = spec.func(**args.model_dump())
    except ToolExecutionError as exc:
        return {
            "ok": False,
            "tool": name,
            "kind": exc.kind,
            "error": {"message": exc.message},
        }
    # 故意**不**捕 Exception：handler 里真正的 bug（拼写错、NoneType）必须崩，
    # 在这里 catch-all 会让 bug 伪装成"工具业务失败"，模型还会一本正经地
    # 向用户解释一个根本不存在的业务错误。

    return {"ok": True, "tool": name, "result": result}


def execute_tool_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
    """执行一批 tool_calls，返回与输入**同序**的结果列表。

    **顺序必须与 `tool_calls` 一致**：回灌时是按位置配对的（见 `build_tool_messages`），
    一旦乱序，`tool_call_id` 和内容就会错配 —— 模型会看到"北京的结果挂在上海的 id 上"，
    然后给出莫名其妙的答案。这种 bug 在日志里很难看出来，所以顺序在这里钉死。
    """
    return [execute_tool(call.function.name, call.function.arguments) for call in tool_calls]


def build_tool_messages(
    tool_calls: list[Any],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """把执行结果转成 `role="tool"` 消息。

    **`tool_call_id` 是这里唯一不能漏的字段**：模型靠它把"我要的那次调用"和
    "你给的结果"对上号。漏了（或写错）会得到 400：
    `Invalid parameter: messages with role 'tool' must be a response to a preceding message with 'tool_calls'`
    —— 或者更糟，模型把结果张冠李戴，还不报错。

    `content` 必须是**字符串**（JSON 文本），不能直接塞 dict：
    OpenAI 协议规定 tool 消息的 content 是 string。
    `ensure_ascii=False` 让中文在日志里可读，而不是一片 unicode 转义码。

    失败结果也照样回灌 —— 那是模型自愈的依据，不是要藏起来的内部错误。
    """
    if len(tool_calls) != len(results):
        raise ValueError(f"tool_calls 与 results 数量不匹配：{len(tool_calls)} vs {len(results)}")

    messages: list[dict[str, Any]] = []
    for call, result in zip(tool_calls, results, strict=True):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,  # ← 漏了这个，模型对不上号
                "content": json.dumps(result, ensure_ascii=False),
            }
        )
    return messages


__all__ = [
    "CALCULATE_DESCRIPTION",
    "GET_WEATHER_DESCRIPTION",
    "TOOLS_SCHEMA",
    "TOOL_REGISTRY",
    "TOOL_SPECS",
    "CalculateArgs",
    "GetWeatherArgs",
    "ToolExecutionError",
    "ToolSpec",
    "build_tool_messages",
    "calculate",
    "execute_tool",
    "execute_tool_calls",
    "get_weather",
]
