"""demo_qa_tools：工具设计验收的可执行版本。

对应周三 20:00–22:00 的验收项：

    A 四要素描述检查       门禁式断言（离线部分）
    B 统一错误信封         七类 error_type 各一例（离线部分）
    C 描述改写 A/B 对比    同一任务，坏描述 vs 四要素描述，看模型选工具的变化（真调用）

C 是今晚的核心实验：任务故意设计成「只该调 query_metrics」——
「B20241129-03 批次的抗压强度是多少？」。
坏描述（「查询数据。」）下模型可能先去 search_knowledge_base 盲搜；
四要素描述写明「查规范用 search_knowledge_base、查批次数据用 query_metrics」后
应当一步选中。每个条件跑 3 次看稳定性。

运行：

    .\\.venv\\Scripts\\python.exe -X utf8 scripts\\demo_qa_tools.py
"""

from __future__ import annotations

import asyncio
import json
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
from app.core.llm.qa_tools import (  # noqa: E402
    ERROR_TYPES,
    QA_TOOLS_SCHEMA,
    VAGUE_DESCRIPTIONS,
    build_schema_with_descriptions,
    execute_qa_tool,
)

TITLE = "=" * 72


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        with suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")


def section(title: str) -> None:
    print(f"\n{TITLE}\n{title}\n{TITLE}")


# ---------------------------------------------------------------- A. 四要素门禁
def demo_four_elements() -> None:
    section("A. 四要素描述检查（离线门禁）")
    for item in QA_TOOLS_SCHEMA:
        fn = item["function"]
        desc = fn["description"]
        checks = {
            "做什么（首句功能）": True,
            "什么时候用": "什么时候用" in desc,
            "什么时候不用": "什么时候不用" in desc,
            "参数": "参数" in desc,
            "返回形状": "返回" in desc,
        }
        flags = " ".join(f"{k}={'√' if v else '×'}" for k, v in checks.items())
        print(f"  {fn['name']:<22} {flags}  len={len(desc)}")


# ---------------------------------------------------------------- B. 统一错误信封
def demo_error_envelope() -> None:
    section("B. 统一错误信封：七类 error_type（离线，错误是数据不是异常）")
    # conflict 需要先真实投递一次，制造重复
    execute_qa_tool(
        "notify", '{"channel": "team", "message": "首次投递", "idempotency_key": "demo-dup-key"}'
    )
    cases = [
        ("unknown_tool", "send_email", "{}"),
        ("invalid_json", "query_metrics", "{bad"),
        (
            "invalid_arg",
            "notify",
            '{"channel": "wechat", "message": "x", "idempotency_key": "abcdef"}',
        ),
        (
            "not_found",
            "query_metrics",
            '{"sql_template": "inspections_by_batch", "params": {"batch_id": "B19990101-99"}}',
        ),
        (
            "conflict",
            "notify",
            '{"channel": "team", "message": "重复投递", "idempotency_key": "demo-dup-key"}',
        ),
    ]
    seen: set[str] = set()
    for expected, name, raw in cases:
        r = execute_qa_tool(name, raw)
        et = r.get("error_type") or "(成功)"
        seen.add(r.get("error_type") or "")
        status = "√" if et == expected else "×"
        msg = r.get("error", {}).get("message", json.dumps(r.get("data", {}), ensure_ascii=False))
        print(f"  [{status}] 期望 {expected:<12} 实得 {et:<12} msg={str(msg)[:50]}")
    print(f"\n  覆盖的 error_type: {sorted(seen)} / 封闭枚举 {sorted(ERROR_TYPES)}")
    print("  （timeout / internal 在单测与假 client 里覆盖：真调用难以稳定复现）")


# ---------------------------------------------------------------- C. 描述 A/B 对比
async def run_once(
    client: AsyncOpenAI,
    model: str,
    extra: dict[str, Any],
    schema: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    """跑一次，只统计「模型选了哪些工具」——描述实验的因变量。"""
    from app.core.llm.qa_tools import execute_qa_tool
    from app.core.llm.react_loop import run_react_loop

    # tools 与 executor 必须成对出现：schema 是 QA 四件套，
    # 执行层也必须是 execute_qa_tool —— 周三实测抓到的 P0（见笔记）。
    r = await run_react_loop(
        client,
        model=model,
        user_message="B20241129-03 这个批次的抗压强度，规范要求应该是多少？",
        tools=schema,
        executor=execute_qa_tool,
        max_steps=6,
        extra_request=extra,
    )
    chosen = [a["tool"] for s in r.steps for a in s.actions if a.get("executed")]
    print(f"    [{label}] 选中: {chosen or '（未调工具）'} stopped={r.stopped_reason}")
    return {"chosen": chosen, "stopped": r.stopped_reason, "answer": r.final_answer[:120]}


async def demo_description_ab(client: AsyncOpenAI, model: str, extra: dict[str, Any]) -> None:
    section("C. 描述改写 A/B 对比")

    # 场景一（第一轮实测）：任务太直白，两种描述都选对，且 glm-pool 链式循环
    # 污染终态（全部 repeat_loop）→ 结论：无区分度，弃用。
    # 场景二：语义重叠任务 ——「限值」既是规范概念又被坏描述归进"查询数据"。
    # 实测（修好执行层后）：两种条件都 3/3 选对 → 任务太直白，仍无区分度。
    # 场景三（正式实验）：**反面任务** —— 问「某批次的规范限值」。
    # 这个任务的正确答案是 search_knowledge_base（限值是规范概念，不是数据），
    # 但任务文本里的"批次号"会诱导模型去 query_metrics —— 坏描述
    # （"查询数据。"）不阻止这种诱导，四要素描述（"不要用它查规范条文"）
    # 应该阻止。这是对"什么时候不用"的直接测试。
    task_model = "trae/qwen-3.5"
    print(f"  模型: {task_model}（排除 glm-pool 链式循环对实验的干扰）")
    print("  任务：「B20241129-03 这个批次的抗压强度，规范要求应该是多少？」")
    print(
        "  正确行为：调 search_knowledge_base（限值是规范概念）；误调 query_metrics = 被批次号带偏。\n"
    )

    results: dict[str, list[dict[str, Any]]] = {"bad": [], "good": []}
    for label, overrides in (
        ("bad", VAGUE_DESCRIPTIONS),
        ("good", None),
    ):
        schema = build_schema_with_descriptions(overrides)
        desc_preview = schema[1]["function"]["description"][:22]
        print(f"  --- 条件 {label}（query_metrics 描述：{desc_preview!r}...）---")
        for i in range(3):
            try:
                r = await run_once(client, task_model, extra, schema, f"{label}#{i + 1}")
                results[label].append(r)
            except Exception as exc:  # noqa: BLE001
                print(f"    [{label}#{i + 1}] 异常: {type(exc).__name__}: {str(exc)[:80]}")

    print("\n  --- 汇总 ---")
    for label in ("bad", "good"):
        runs = results[label]
        if not runs:
            print(f"  {label}: 全部异常，无数据")
            continue
        right = sum(1 for r in runs if r["chosen"] and "search_knowledge_base" in r["chosen"])
        wrong = sum(1 for r in runs if "query_metrics" in r["chosen"])
        print(f"  {label}: 调用 search_knowledge_base {right}/3；调用 query_metrics {wrong}/3")
        for r in runs:
            print(f"    chosen={r['chosen']} stopped={r['stopped']}")
            print(f"    answer: {r['answer'][:150]}")


async def main() -> int:
    _force_utf8_stdout()

    settings = get_settings()
    llm = settings.llm
    extra = {"reasoning_effort": "none"}

    print(f"model={llm.model}  base_url={llm.base_url}  key={llm.api_key_masked}")

    demo_four_elements()
    demo_error_envelope()

    client = get_llm_client(settings)
    try:
        await demo_description_ab(client, llm.model, extra)
    finally:
        await client.close()

    print(f"\n{TITLE}\n全部演示完成\n{TITLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
