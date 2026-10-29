"""采集四种 LangChain 配置的**完整失败栈**，供 docs/why-i-use-langchain.md 引用。

为什么要单独一个脚本：`lc_extract.py` 为了跑 10 条 × 4 臂会把错误压成一行，
而"为什么不用它"这份文档需要**完整原文**（尤其 `tool_choice` 那句话，
它是整个论证最有力的证据）。

本脚本每种配置只跑**1 条**，控制调用量。

运行：python -X utf8 scripts/lc_failure_stacks.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTPUT = Path(__file__).with_name("lc_failure_stacks.json")

# 用一条最典型的输入（ex-10，字段最少，排除"输入太复杂"这个借口）
ITEM = {
    "id": "ex-10-stack",
    "text": "Windows 上必须用 curl.exe 而不是 PowerShell 的 curl，因为后者是 Invoke-WebRequest 的别名。",
    "fields": [
        {"name": "平台", "desc": "操作系统", "type": "string"},
        {"name": "正确命令", "desc": "应当使用的命令", "type": "string"},
    ],
}


def build_flat():
    """扁平 schema —— 排除"嵌套结构导致失败"这个混淆因素。"""
    from pydantic import BaseModel, Field

    class Flat(BaseModel):
        平台: str = Field(description="操作系统")
        正确命令: str = Field(description="应当使用的命令")

    return Flat


def build_nested():
    from pydantic import BaseModel, Field

    class ExtractV2(BaseModel):
        fields: dict[str, str | None] = Field(default_factory=dict, description="字段名→值")
        缺失字段: list[str] = Field(default_factory=list)
        置信度: float = Field(ge=0.0, le=1.0)

    return ExtractV2


async def probe(arm: str, method: str, strict: bool | None, schema) -> dict:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "deepseek-account/deepseek-flash"),
        base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8326/v1"),
        api_key=os.environ.get("LLM_API_KEY", ""),
        temperature=0.0,
        max_tokens=8192,
    )
    structured = llm.with_structured_output(schema, method=method, strict=strict)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是严谨的信息抽取器。只输出 JSON，键与字段名逐字一致。"),
            ("human", "字段：{f}\n正文：{t}"),
        ]
    )
    chain = prompt | structured

    rec = {"arm": arm, "method": method, "strict": strict, "schema": schema.__name__}
    try:
        out = await chain.ainvoke(
            {"f": "、".join(f["name"] for f in ITEM["fields"]), "t": ITEM["text"]}
        )
        rec["ok"] = True
        rec["result"] = out.model_dump()
        print(f"  ✅ {arm} 成功: {rec['result']}")
    except Exception as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["error_type"] = type(exc).__name__
        rec["error_full"] = f"{type(exc).__name__}: {exc}"
        # 只留最后 3 帧，且去掉本机绝对路径，便于写进文档
        tb = traceback.format_exc().splitlines()
        rec["traceback_tail"] = [
            ln.replace(str(Path.home()), "~").replace("D:\\git\\网页\\industrial-ai-platform\\", "")
            for ln in tb[-6:]
        ]
        print(f"  ❌ {arm} 失败 [{rec['error_type']}]")
        print(f"     {rec['error_full'][:300]}")
    return rec


async def main() -> None:
    if not os.environ.get("LLM_API_KEY"):
        raise SystemExit("缺 LLM_API_KEY")

    print("=== 采集完整失败栈（每种配置 1 条输入）===\n")
    nested = build_nested()
    flat = build_flat()

    plan = [
        ("lc_json_schema_strict", "json_schema", True, nested),
        ("lc_json_schema", "json_schema", None, nested),
        ("lc_function_calling", "function_calling", None, nested),
        ("lc_json_mode", "json_mode", None, nested),
        # 关键对照：扁平 schema，排除"嵌套结构"这个混淆因素
        ("lc_flat_json_schema", "json_schema", None, flat),
    ]

    results = []
    for arm, method, strict, schema in plan:
        print(f"-- {arm} (method={method}, strict={strict}, schema={schema.__name__}) --")
        results.append(await probe(arm, method, strict, schema))
        print()

    OUTPUT.write_text(
        json.dumps({"item": ITEM, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] 写入 {OUTPUT.name}")


if __name__ == "__main__":
    asyncio.run(main())
