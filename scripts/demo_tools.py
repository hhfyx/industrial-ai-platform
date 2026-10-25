"""demo_tools：Function Calling 五格验收的可执行版本。

对应周二 20:00–22:00 的五个验收项，一次性全跑：

    A 工具 schema        两份 schema 的形态（给模型看的东西）
    B 本地实现 + 注册表  直接调用可用（不经过模型）
    C 单轮闭环           请求 → tool_calls → 执行 → 回灌 → 二次请求
    D 并行 tool_calls    一次返回多个，按 id 一一配对
    E 边界               参数类型错 / 工具名不存在 / 不调工具 / 连续 3 轮

运行：

    uv run python -X utf8 scripts/demo_tools.py

`-X utf8` 是必须的：Windows 控制台 GBK，中文输出会乱码或直接崩。
脚本自己也会切 stdout 到 UTF-8，双保险。
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

# 直接 `python scripts/demo_tools.py` 时 sys.path[0] 是 scripts/，import app 会失败。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.llm.client import get_llm_client  # noqa: E402
from app.core.llm.tool_loop import render_messages, run_tool_loop  # noqa: E402
from app.core.llm.tools import (  # noqa: E402
    TOOL_REGISTRY,
    TOOLS_SCHEMA,
    calculate,
    execute_tool,
    execute_tool_calls,
    get_weather,
)

TITLE = "=" * 72


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        with suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")


def section(title: str) -> None:
    print(f"\n{TITLE}\n{title}\n{TITLE}")


# ---------------------------------------------------------------- A. schema
def demo_schema() -> None:
    section("A. 工具 schema（两份）")
    for spec in TOOLS_SCHEMA:
        fn = spec["function"]
        print(f"\n--- {fn['name']} ---")
        print(f"  description : {fn['description'][:100]}...")
        params = fn["parameters"]
        print(f"  properties  : {list(params.get('properties', {}))}")
        print(f"  required    : {params.get('required', [])}")
        for key, prop in params.get("properties", {}).items():
            enum = f" enum={prop['enum']}" if "enum" in prop else ""
            print(f"    {key}: type={prop.get('type')}{enum}")

    print("\n提示：description 里必须写「什么时候该用」，模型靠它选工具。")
    print("      只写「查询天气」→ 模型遇到「北京冷不冷」可能自己猜；")
    print("      写上触发场景 → 它才会去调。")


# ----------------------------------------------------- B. 本地实现 + 注册表
def demo_registry() -> None:
    section("B. 本地实现 + 注册表（不经过模型）")
    print("TOOL_REGISTRY:", list(TOOL_REGISTRY))

    print(f"\n  get_weather('北京')            -> {get_weather('北京')}")
    print(f"  get_weather('上海','fahrenheit') -> {get_weather('上海', 'fahrenheit')}")
    print(f"  calculate(12,'multiply',3)     -> {calculate(12, 'multiply', 3)}")

    print("\n  --- execute_tool 的四类失败（都返回 dict，不抛异常）---")
    cases = [
        ("正常", "get_weather", '{"city":"杭州"}'),
        ("参数类型错", "calculate", '{"a":"abc","op":"add","b":1}'),
        ("工具名不存在", "no_such_tool", "{}"),
        ("参数 JSON 坏", "get_weather", "{bad"),
        ("业务失败-除零", "calculate", '{"a":1,"op":"divide","b":0}'),
        ("业务失败-无此城市", "get_weather", '{"city":"火星"}'),
    ]
    for label, name, raw in cases:
        result = execute_tool(name, raw)
        kind = result.get("kind") if not result["ok"] else "OK"
        print(f"    {label:<16} ok={result['ok']!s:<5} kind={kind}")


# ------------------------------------------------------------ C. 单轮闭环
async def demo_single(client: AsyncOpenAI, model: str, extra: dict[str, Any]) -> None:
    section("C. 单轮闭环（请求 → tool_calls → 执行 → 回灌 → 二次请求）")
    result = await run_tool_loop(
        client,
        model=model,
        user_message="北京现在天气怎么样？",
        extra_request=extra,
    )
    print(render_messages(result.messages))
    print(f"\n  rounds={result.rounds} tool_calls={result.tool_calls_count}")
    print(f"  usage={result.usage_total}")
    print(f"  最终答案: {result.final_message}")


# ------------------------------------------------------- D. 并行 tool_calls
async def demo_parallel(client: AsyncOpenAI, model: str, extra: dict[str, Any]) -> None:
    section("D. 并行 tool_calls（一次多个，按 id 一一配对）")
    result = await run_tool_loop(
        client,
        model=model,
        user_message="北京和上海今天天气分别怎么样？",
        extra_request=extra,
    )
    print(render_messages(result.messages))

    # 显式核对 id 配对：这是最容易写错、且多数情况下**不会报错**的地方
    assistant_calls = [
        c
        for m in result.messages
        if m.get("role") == "assistant"
        for c in (m.get("tool_calls") or [])
    ]
    tool_ids = [m["tool_call_id"] for m in result.messages if m.get("role") == "tool"]
    print(f"\n  模型请求 {len(assistant_calls)} 个调用，回灌 {len(tool_ids)} 条 tool 消息")
    print(f"  请求的 id : {[c['id'] for c in assistant_calls]}")
    print(f"  回灌的 id : {tool_ids}")
    print(f"  一一配对  : {[c['id'] for c in assistant_calls] == tool_ids}")
    print(f"  rounds={result.rounds} tool_calls={result.tool_calls_count}")
    print(f"  最终答案: {result.final_message}")


# ---------------------------------------------------------------- E. 边界
async def demo_boundaries(client: AsyncOpenAI, model: str, extra: dict[str, Any]) -> None:
    section("E. 边界")

    print("\n--- E1 模型直接回答，不调工具 ---")
    r = await run_tool_loop(
        client, model=model, user_message="1+1 等于几？直接回答，不用工具。", extra_request=extra
    )
    print(f"  tool_calls={r.tool_calls_count} rounds={r.rounds}")
    print(f"  答案: {r.final_message}")

    print("\n--- E2 参数类型错：把错误回灌，看模型能否自愈 ---")
    fake_call = _FakeCall(
        id="call_fake_type_err", name="calculate", arguments='{"a":"abc","op":"add","b":1}'
    )
    results = execute_tool_calls([fake_call])
    print(f"  执行结果: {json.dumps(results[0], ensure_ascii=False)}")
    print("  ↑ 这就是回灌给模型的内容；它据此知道该把 a 改成数字。")

    print("\n--- E3 工具名不存在 ---")
    fake_missing = _FakeCall(id="call_fake_missing", name="send_email", arguments='{"to":"a@b.c"}')
    print(f"  执行结果: {json.dumps(execute_tool_calls([fake_missing])[0], ensure_ascii=False)}")
    print("  ↑ 关键：没有 KeyError 崩掉，而是告诉模型「没有这个工具，可用的是…」")

    print("\n--- E4 模型连续调 3 轮（多步推理：两地温差）---")
    r = await run_tool_loop(
        client,
        model=model,
        user_message="北京和上海温差多少度？先查两地天气，再算出差值。",
        extra_request=extra,
        max_rounds=3,
    )
    print(render_messages(r.messages))
    print(f"\n  rounds={r.rounds} tool_calls={r.tool_calls_count} stopped={r.stopped_reason}")
    print(f"  最终答案: {r.final_message}")

    print("\n--- E5 轮数上限强制触发（max_rounds=1，问需要多步的问题）---")
    r = await run_tool_loop(
        client,
        model=model,
        user_message="北京和上海温差多少度？先查两地天气，再算出差值。",
        extra_request=extra,
        max_rounds=1,
    )
    print(f"  stopped_reason={r.stopped_reason} rounds={r.rounds}")
    print(f"  返回: {r.final_message}")
    print("  ↑ 没有上限的话，这类「继续调」会无限循环，每次都计费。")


class _FakeCall:
    """构造一个假的 tool_call，用来单独测执行层（不联网）。

    真实 SDK 对象结构是 `call.id` / `call.function.name` / `call.function.arguments`，
    这里只实现这三个属性，够 `execute_tool_calls` 用。
    """

    def __init__(self, *, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.function = type("Fn", (), {"name": name, "arguments": arguments})()


async def main() -> int:
    _force_utf8_stdout()

    settings = get_settings()
    llm = settings.llm
    # 本机网关默认开推理，会吃掉 max_tokens 导致 content=null（见 01_params.md §3.1）。
    # 这是网关特性，不是协议的一部分，所以只从这里注入，不写进 tool_loop。
    extra = {"reasoning_effort": "none"}

    print(f"model={llm.model}  base_url={llm.base_url}  key={llm.api_key_masked}")
    print(
        "\n注意：模型必须支持 role=tool 消息的回灌。实测本机网关的\n"
        "  deepseek-account/deepseek-flash 会在第二轮报 502 UNSUPPORTED_CONTENT\n"
        "  （连裸 httpx 也一样，是上游适配器限制）；换 buddy/glm-5.3 正常。\n"
        "  详见 learn/llm/02_function_calling.md"
    )

    demo_schema()
    demo_registry()

    client = get_llm_client(settings)
    try:
        await demo_single(client, llm.model, extra)
        await demo_parallel(client, llm.model, extra)
        await demo_boundaries(client, llm.model, extra)
    finally:
        await client.close()

    print(f"\n{TITLE}\n全部演示完成\n{TITLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
