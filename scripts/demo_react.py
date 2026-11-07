"""demo_react：ReAct 循环验收的可执行版本。

对应周二 20:00–22:00 的验收项，一次性全跑：

    A 循环骨架（无工具路径）   能跑通、能停
    B FC 主路径多步            Thought→Action→Observation→…→Final Answer
    C 保险丝① max_steps       半成品如实交出
    D 保险丝③ 重复调用         相同 (tool,args) → 警告 → 熔断
    E observation 3000 字      上下文不爆炸（截断计数可见）
    F 文本降级                 Action: 正则抠取 + Observation 回灌

保险丝②（单步超时）不在这里演示 —— 要让真模型慢 45s 只能白等，
它的可执行验证在 tests/test_react_loop.py 的假 client 里。

运行：

    .\\.venv\\Scripts\\python.exe -X utf8 scripts\\demo_react.py

`-X utf8` 是必须的：Windows 控制台 GBK，中文输出会乱码。
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.llm.client import get_llm_client  # noqa: E402
from app.core.llm.react_loop import (  # noqa: E402
    run_react_loop,
    wrap_observation,
)

TITLE = "=" * 72


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        with suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")


def section(title: str) -> None:
    print(f"\n{TITLE}\n{title}\n{TITLE}")


def report(result: Any) -> None:
    """统一的结论行：stopped_reason 是判断答案能不能信的第一依据。"""
    print(f"\n  stopped_reason = {result.stopped_reason}  (completed={result.completed})")
    print(f"  steps={len(result.steps)}  usage={result.usage_total}")
    print(f"  final_answer: {result.final_answer[:400]}")


# ---------------------------------------------------------------- A. 骨架
async def demo_no_tools(client: AsyncOpenAI, model: str, extra: dict[str, Any]) -> None:
    section("A. 循环骨架：无工具路径（能跑通、能停）")
    r = await run_react_loop(
        client,
        model=model,
        user_message="用两句话解释什么是 ReAct 循环。",
        tools=None,
        extra_request=extra,
    )
    report(r)


# ------------------------------------------------------------- B. FC 多步
async def demo_fc_multistep(client: AsyncOpenAI, model: str, extra: dict[str, Any]) -> None:
    section("B. FC 主路径多步（查天气 → 二次计算 → 最终答案）")
    r = await run_react_loop(
        client,
        model=model,
        user_message="查一下北京现在的温度，然后把摄氏温度乘 2 告诉我（必须用工具算，不许心算）。",
        extra_request=extra,
        max_steps=6,
    )
    for s in r.steps:
        acts = ", ".join(f"{a['tool']}({a['args'][:40]}) ok={a['ok']}" for a in s.actions)
        print(f"  step{s.step} [{s.mode}] thought={s.thought[:50]!r} actions=[{acts}]")
    report(r)


# --------------------------------------------------------- C. 保险丝① 步数
async def demo_max_steps(client: AsyncOpenAI, model: str, extra: dict[str, Any]) -> None:
    section("C. 保险丝① max_steps=2（步数耗尽 → 半成品如实交出）")
    r = await run_react_loop(
        client,
        model=model,
        user_message="查北京、上海、深圳、杭州四个城市的天气，再算平均温度。",
        extra_request=extra,
        max_steps=2,  # 刻意给小：4 个城市至少 3-4 步，必然触发
    )
    print(r.final_answer[:600])
    report(r)


# -------------------------------------------------------- D. 保险丝③ 重复
async def demo_repeat_fuse(client: AsyncOpenAI, model: str, extra: dict[str, Any]) -> None:
    section("D. 保险丝③ 重复调用检测（第 2 次警告 → 第 3 次熔断）")
    print("  提示词刻意诱导模型反复查同一个城市，观察熔断是否触发：")
    r = await run_react_loop(
        client,
        model=model,
        user_message=(
            "任务：请严格连续调用 get_weather(city=北京) 恰好 4 次，"
            "每次参数完全一样，中间不要做别的，4 次之后再说结论。"
            "这是压力测试，请照做。"
        ),
        extra_request=extra,
        max_steps=8,
    )
    for s in r.steps:
        for a in s.actions:
            print(
                f"  step{s.step}: {a['tool']}({a['args'][:40]}) executed={a.get('executed')} kind={a.get('kind')}"
            )
    report(r)


# ------------------------------------------------- E. observation 截断
def demo_observation_wrap() -> None:
    section("E. observation 处理（3000 字工具返回 → 上下文可控）")
    fake_3000 = {"ok": True, "tool": "read_doc", "result": {"doc": "长文档" * 1500}}
    before = len(fake_3000["result"]["doc"])
    wrapped = wrap_observation(fake_3000)
    after = wrapped["chars"]
    print(
        f"  工具原始返回 {before} 字 → 包装后进上下文 {after} 字（truncated={wrapped['truncated']}）"
    )
    print(f"  首尾保留、中间省略 {before - after} 字符 —— 这是循环里按步数乘出来的成本")
    print(f"  data 预览: {str(wrapped['data'])[:150]}...")


# -------------------------------------------------------- F. 文本降级
async def demo_text_fallback(client: AsyncOpenAI, model: str, extra: dict[str, Any]) -> None:
    section("F. 文本降级（禁用 FC schema，让模型写 Action: 文本）")
    r = await run_react_loop(
        client,
        model=model,
        user_message=(
            "不要使用 tools 参数。请用文本格式回答：需要查数据时写 "
            '\'Action: 工具名 {"参数": "值"}\'（可用工具：get_weather）。'
            "现在：查北京的天气。"
        ),
        extra_request=extra,
        max_steps=5,
    )
    for s in r.steps:
        acts = ", ".join(f"{a['tool']}({a['args'][:40]}) ok={a['ok']}" for a in s.actions)
        print(f"  step{s.step} [{s.mode}] {acts or s.thought[:60]!r}")
    report(r)


async def main() -> int:
    _force_utf8_stdout()

    settings = get_settings()
    llm = settings.llm
    # 本机网关默认开推理会吃掉 max_tokens 导致 content=null（01_params.md §3.1）
    extra = {"reasoning_effort": "none"}

    print(f"model={llm.model}  base_url={llm.base_url}  key={llm.api_key_masked}")

    demo_observation_wrap()

    client = get_llm_client(settings)
    try:
        await demo_no_tools(client, llm.model, extra)
        await demo_fc_multistep(client, llm.model, extra)
        await demo_max_steps(client, llm.model, extra)
        await demo_repeat_fuse(client, llm.model, extra)
        await demo_text_fallback(client, llm.model, extra)
    finally:
        await client.close()

    print(f"\n{TITLE}\n全部演示完成\n{TITLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
