"""用 LangChain 重写「信息抽取」这一个模板，与自研 `structured_call` 做横向对比。

**目的不是"哪个更好"，而是把抽象层次摆出来看**：
LangChain 替我们抽象了哪三层、付出什么代价、在本项目这套约束下值不值。

对照的两条实现：
  A 自研   app/core/llm/structured.py::structured_call
           → json_object + Pydantic 校验 + 报错原文回灌（最多 2 轮）
  B LC     ChatPromptTemplate | ChatOpenAI.with_structured_output(method=..., strict=...)
           → 交给框架，不自己写回灌

跑同一批 10 条（eval/prompt-sets/extract.jsonl），同一模型、同一温度。

**四种 LangChain 配置**，因为 `with_structured_output` 的 method 不同，
在本网关上的实际行为完全不同（这是本周最有价值的对比点）：

  lc_json_schema_strict  method=json_schema, strict=True   ← 理论上最强
  lc_json_schema         method=json_schema               ← 默认
  lc_function_calling    method=function_calling          ← 走 tool 调用
  lc_json_mode           method=json_mode                 ← 只保证是 JSON

运行：
    python -X utf8 scripts/lc_extract.py
    python -X utf8 scripts/lc_extract.py --only lc_json_schema_strict
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pydantic import BaseModel, Field  # noqa: E402

HERE = Path(__file__).resolve().parent
SETS = HERE.parent / "eval" / "prompt-sets"
OUTPUT = HERE / "lc_extract_result.json"


# ---------------------------------------------------------------- 输出模型


class ExtractV2(BaseModel):
    """与 app/prompts/extract/v2_schema.py 保持一致的形状。

    为什么这里单独定义而不 import：LangChain 需要把 schema 交给它自己的
    解析器，而项目里的模型带有 `description` 等中文说明 —— 那份定义是对的，
    这里复制一份是为了让对比脚本**不依赖项目内部的 prompt 目录**，
    从而能公平比较"两个方案各自需要多少胶水代码"。
    """

    fields: dict[str, str | None] = Field(
        default_factory=dict,
        description="字段名 → 抽取到的值；原文没有的字段值为 null",
    )
    缺失字段: list[str] = Field(default_factory=list, description="填了 null 的字段名")
    置信度: float = Field(ge=0.0, le=1.0, description="整体抽取完整度的把握")


# ---------------------------------------------------------------- 输入准备


def load_items() -> list[dict]:
    path = SETS / "extract.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def field_list(item: dict) -> str:
    """把字段清单渲染成文本（两边都要这一步，属于公平的一部分）。"""
    return "\n".join(
        f"- {f['name']}（{f.get('type', 'string')}）：{f.get('desc', f['name'])}"
        for f in item["fields"]
    )


# ---------------------------------------------------------------- 判定（与 ab_test 口径一致）


def _value_match(got: Any, want: Any) -> bool:
    import re

    if got is None or want is None:
        return got == want
    g = str(got).replace(" ", "").replace(",", "")
    w = str(want).replace(" ", "").replace(",", "")
    if g == w:
        return True
    gn = re.findall(r"\d+\.?\d*", g)
    wn = re.findall(r"\d+\.?\d*", w)
    if gn and wn and gn == wn:
        return True
    return g in w or w in g


def judge(values: dict[str, Any], item: dict) -> tuple[float, str]:
    gold = item["gold"]
    hits, misses = [], []
    for k, want in gold.items():
        got = values.get(k)
        if _value_match(got, want):
            hits.append(k)
        else:
            misses.append(f"{k}: 给 {got!r} 应 {want!r}")
    score = len(hits) / len(gold)
    detail = f"字段 {len(hits)}/{len(gold)}"
    if misses:
        detail += "；" + "；".join(misses[:2])
    return score, detail


# ---------------------------------------------------------------- LangChain 实现


def build_llm():
    """构造 ChatOpenAI，指向本机网关。

    **这里就能看出第一层代价**：LangChain 仍要你把 base_url/api_key/model
    塞进它自己的类，而不是复用项目里已有的 `get_llm_client()` ——
    于是同一个进程里出现两个 client（两套连接池、两套超时/重试设置）。
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "deepseek-account/deepseek-flash"),
        base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8326/v1"),
        api_key=os.environ.get("LLM_API_KEY", ""),
        temperature=0.0,
        max_tokens=8192,
    )


def build_chain(method: str, strict: bool | None):
    """PromptTemplate | LLM.with_structured_output → 一条 Runnable。"""
    from langchain_core.prompts import ChatPromptTemplate

    llm = build_llm()
    # method / strict 是 with_structured_output 的参数：
    #   json_schema      → 用 response_format={"type":"json_schema"}（需网关支持）
    #   function_calling → 用 tools + tool_choice 强制调用
    #   json_mode        → 用 response_format={"type":"json_object"}
    structured = llm.with_structured_output(ExtractV2, method=method, strict=strict)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是严谨的信息抽取器。只依据正文明确写出的信息，"
                "原文没有的字段值填 null，严禁猜测。"
                "键必须与给定字段名逐字一致。只输出 JSON。",
            ),
            (
                "human",
                "字段清单：\n{fields}\n\n正文：\n{text}",
            ),
        ]
    )
    return prompt | structured


async def run_arm(arm: str, method: str, strict: bool | None, items: list[dict]) -> list[dict]:
    chain = build_chain(method, strict)
    rows = []

    for item in items:
        t0 = time.perf_counter()
        rec: dict[str, Any] = {"arm": arm, "item_id": item["id"]}
        try:
            out = await chain.ainvoke({"fields": field_list(item), "text": item["text"]})
            elapsed = int((time.perf_counter() - t0) * 1000)
            values = out.model_dump()
            score, detail = judge(values.get("fields") or {}, item)
            rec.update(
                ok=True,
                score=round(score, 4),
                detail=detail,
                latency_ms=elapsed,
                raw=json.dumps(values, ensure_ascii=False)[:400],
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = int((time.perf_counter() - t0) * 1000)
            # **保存完整错误文本，不截断**：这份数据要拿去做"为什么不用它"的证据，
            # 截断到 200 字符会把关键原因（如 "only tool_choice auto and none
            # are supported"）切掉 —— 那是整个论证最有力的一句。
            rec.update(
                ok=False,
                score=0.0,
                detail=f"{type(exc).__name__}: {exc}",
                detail_short=f"{type(exc).__name__}: {exc}"[:200],
                error_type=type(exc).__name__,
                latency_ms=elapsed,
                raw="",
            )
        rows.append(rec)
        mark = "✅" if rec["ok"] else "❌"
        text = rec.get("detail_short") or rec["detail"][:70]
        print(f"  {mark} {item['id']:<8} score={rec['score']:.2f} {rec['latency_ms']}ms  {text}")
    return rows


# ---------------------------------------------------------------- 主流程


ARMS = {
    "lc_json_schema_strict": ("json_schema", True),
    "lc_json_schema": ("json_schema", None),
    "lc_function_calling": ("function_calling", None),
    "lc_json_mode": ("json_mode", None),
}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只跑某个臂")
    args = ap.parse_args()

    items = load_items()
    print(f"items={len(items)}  model={os.environ.get('LLM_MODEL')}")

    # **累积保存，不覆盖**：分臂用 --only 跑时，前面的结果不能被冲掉。
    # 第一版每次新建 store，结果连续跑 4 次 --only 只剩最后一臂 ——
    # 这是"数据收集脚本"最不该犯的错：跑完才发现前面的没了，只能重跑（花真钱）。
    if OUTPUT.exists():
        store = json.loads(OUTPUT.read_text(encoding="utf-8"))
    else:
        store = {"meta": {}, "arms": {}}
    store.setdefault("arms", {})
    store["meta"].update(
        {
            "model": os.environ.get("LLM_MODEL"),
            "n": len(items),
            "lc_version": _lc_version(),
        }
    )

    todo = [args.only] if args.only else list(ARMS)
    for arm in todo:
        method, strict = ARMS[arm]
        print(f"\n== {arm}  method={method} strict={strict} ==")
        rows = await run_arm(arm, method, strict, items)
        ok = [r for r in rows if r["ok"]]
        store["arms"][arm] = {
            "method": method,
            "strict": strict,
            "n": len(rows),
            "success": len(ok),
            "mean_score_of_ok": round(sum(r["score"] for r in ok) / len(ok), 4) if ok else 0.0,
            "mean_score_all": round(sum(r["score"] for r in rows) / len(rows), 4),
            "mean_latency_ms": round(sum(r["latency_ms"] for r in ok) / len(ok)) if ok else 0,
            "rows": rows,
        }
        a = store["arms"][arm]
        print(
            f"  → 成功 {a['success']}/{a['n']}  "
            f"均分(仅成功)={a['mean_score_of_ok']}  "
            f"均分(全量)={a['mean_score_all']}"
        )

    OUTPUT.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] 写入 {OUTPUT.name}")


def _lc_version() -> str:
    try:
        import langchain

        return langchain.__version__
    except Exception:  # noqa: BLE001
        return "?"


if __name__ == "__main__":
    asyncio.run(main())
