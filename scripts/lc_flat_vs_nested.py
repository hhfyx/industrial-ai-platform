"""核实：扁平 schema 是否稳定成功？嵌套 schema 是否必然失败？

上一轮我用**1 条**输入发现扁平 schema 成功，这不足以支撑结论 ——
必须跑 10 条才算。本脚本把这个混淆因素彻底查清。

结论关系到 docs/why-i-use-langchain.md §2.3 的说法是否要改写：
- 若扁平 10/10 成功 → 说明"本网关不强制 schema"这个论断**过强**，
  真实情况是"只在复杂/嵌套 schema 上不强制"
- 若扁平也大量失败 → 原来的论断成立

运行：python -X utf8 scripts/lc_flat_vs_nested.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTPUT = Path(__file__).with_name("lc_flat_vs_nested.json")
SETS = Path(__file__).resolve().parents[1] / "eval" / "prompt-sets"


def load_items() -> list[dict]:
    rows = []
    for line in (SETS / "extract.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def make_flat(item: dict):
    """为每个 item 动态造一个**扁平** schema，字段就是它自己的字段名。

    这样与嵌套版对比时，唯一的差别就是"结构层次"，其他全相同。
    """
    from pydantic import create_model

    fields = {}
    for f in item["fields"]:
        fields[f["name"]] = (str | None, None)
    return create_model(f"Flat_{item['id'].replace('-', '_')}", **fields)


def make_nested():
    from pydantic import BaseModel, Field

    class Nested(BaseModel):
        fields: dict[str, str | None] = Field(default_factory=dict, description="字段名→值")
        缺失字段: list[str] = Field(default_factory=list)
        置信度: float = Field(ge=0.0, le=1.0)

    return Nested


def build_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "deepseek-account/deepseek-flash"),
        base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8326/v1"),
        api_key=os.environ.get("LLM_API_KEY", ""),
        temperature=0.0,
        max_tokens=8192,
    )


async def run_flat_items(items: list[dict]) -> list[dict]:
    """扁平 schema × 10 条。"""
    from langchain_core.prompts import ChatPromptTemplate

    llm = build_llm()
    rows = []
    for item in items:
        schema = make_flat(item)
        chain = ChatPromptTemplate.from_messages(
            [
                ("system", "你是严谨的信息抽取器。只依据正文，原文没有的值填 null。只输出 JSON。"),
                ("human", "字段：{f}\n正文：{t}"),
            ]
        ) | llm.with_structured_output(schema, method="json_schema", strict=True)
        rec = {"item_id": item["id"]}
        try:
            out = await chain.ainvoke(
                {
                    "f": "、".join(f["name"] for f in item["fields"]),
                    "t": item["text"],
                }
            )
            rec["ok"] = True
            rec["result"] = out.model_dump()
            print(f"  ✅ {item['id']}")
        except Exception as exc:  # noqa: BLE001
            rec["ok"] = False
            rec["error_type"] = type(exc).__name__
            rec["error_full"] = f"{type(exc).__name__}: {exc}"[:400]
            print(f"  ❌ {item['id']} [{rec['error_type']}] {str(exc)[:100]}")
        rows.append(rec)
    return rows


async def main() -> None:
    if not os.environ.get("LLM_API_KEY"):
        raise SystemExit("缺 LLM_API_KEY")

    items = load_items()
    print("=== 扁平 schema × 10 条（method=json_schema, strict=True）===")
    flat_rows = await run_flat_items(items)
    flat_ok = sum(1 for r in flat_rows if r["ok"])
    print(f"\n扁平 schema: {flat_ok}/{len(flat_rows)} 成功")

    # 与已存的嵌套结果对照
    nested_path = Path(__file__).with_name("lc_extract_result.json")
    nested_ok = None
    if nested_path.exists():
        prev = json.loads(nested_path.read_text(encoding="utf-8"))
        arm = prev["arms"].get("lc_json_schema_strict")
        if arm:
            nested_ok = arm["success"]
            print(f"嵌套 schema: {nested_ok}/{arm['n']} 成功（来自 lc_extract_result.json）")

    print("\n=== 结论 ===")
    if flat_ok == len(flat_rows) and (nested_ok is not None and nested_ok == 0):
        print("❗ 扁平 10/10、嵌套 0/10 → **『本网关不强制 schema』这个论断过强**")
        print("   真实情况：schema 在扁平结构上能生效，在嵌套结构上不生效。")
        print("   docs §2.3 必须修正，不能再说『连扁平也失败』。")
    elif flat_ok < len(flat_rows) // 2:
        print("扁平也大量失败 → 原论断『本网关不强制 schema』成立")
    else:
        print(f"扁平部分成功（{flat_ok}/{len(flat_rows)}）→ 结论需要更细的措辞")

    OUTPUT.write_text(
        json.dumps(
            {
                "flat": {"success": flat_ok, "n": len(flat_rows), "rows": flat_rows},
                "nested_from_prev": nested_ok,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[done] 写入 {OUTPUT.name}")


if __name__ == "__main__":
    asyncio.run(main())
