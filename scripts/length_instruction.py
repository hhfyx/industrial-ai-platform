"""验证「off + 显式长度指令」能否拿回长度合规 —— 决定 W4 摘要档位的最后一测。

前面的链条：
  1. 消融：off 快 14 倍便宜 17 倍，但长度合规仅 0.30
  2. 截断：本地截断能拉回 1.00，但要点率从 0.96 掉到 0.71（丢关键专有名词）
     → 截断不可行，因为摘要最该保全的就是那类词

所以剩下唯一的路：**在 prompt 里把长度要求写得更硬**，让 off 档自己守住。
对照三种指令强度：
  L1 现状（v1 模板：只说"不超过 N 字"）
  L2 强指令（重复 + 说明超长的后果）
  L3 强指令 + 自检（要求先数字数再收尾）
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

HERE = Path(__file__).resolve().parent
SETS = HERE.parent / "eval" / "prompt-sets"
OUTPUT = HERE / "length_instruction.json"

ENDPOINT = "http://127.0.0.1:8326/v1/chat/completions"
MODEL = "deepseek-account/deepseek-flash"


def load_items() -> list[dict]:
    out = []
    for line in (SETS / "summarize.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def prompt_l1(item: dict) -> str:
    """现状：v1 模板。"""
    from app.prompts.loader import load_prompt

    return load_prompt(
        "summarize",
        version="v1",
        text=item["text"],
        max_chars=item["max_chars"],
        must_keep=item["must_include"],
    )


def prompt_l2(item: dict) -> str:
    """强指令：把上限放到最前面 + 说明超长会失败 + 给硬性收尾。"""
    must = item["must_include"]
    return (
        f"你是摘要助手。**硬性要求：摘要总长不超过 {item['max_chars']} 个汉字。**\n"
        f"超过这个长度的输出会被系统直接判为失败，必须重来。\n\n"
        "规则：\n"
        "1. 数字与专有名词逐字照抄（千分位、单位、百分号都不能改）。\n"
        "2. 先想好要写哪几条，再动笔；写的过程中盯着字数。\n"
        "3. 写完后如果超了，删掉最不重要的内容，**不要删数字和专有名词**。\n\n"
        f"必须原样出现在摘要里的内容：{'、'.join(must)}\n\n"
        f"正文：\n{item['text']}\n\n"
        '只输出 JSON：{"摘要": "...", "字数": 0, "覆盖要点": ["..."]}'
    )


def prompt_l3(item: dict) -> str:
    """强指令 + 自检：要求先报预算再写。"""
    must = item["must_include"]
    return (
        f"你是摘要助手。目标长度：**{item['max_chars']} 字以内**（硬上限，超了算失败）。\n\n"
        f"第一步：在脑子里规划——{item['max_chars']} 字大约能写几个要点？\n"
        "第二步：按规划写，优先保证数字和专有名词。\n"
        "第三步：写完数字数，超了就删形容词和连接词，**绝不删数字、型号、专有名词**。\n\n"
        f"必须保留：{'、'.join(must)}\n\n"
        f"正文：\n{item['text']}\n\n"
        '只输出 JSON：{"摘要": "...", "字数": 0, "覆盖要点": ["..."]}'
    )


PROMPTS = {"L1_现状": prompt_l1, "L2_强指令": prompt_l2, "L3_强指令+自检": prompt_l3}


def call(prompt: str, api_key: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.0,
        "reasoning_effort": "off",
        "response_format": {"type": "json_object"},
    }
    t0 = time.perf_counter()
    try:
        r = httpx.post(
            ENDPOINT, headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=90
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "err": str(exc)[:100],
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }
    lat = int((time.perf_counter() - t0) * 1000)
    if r.status_code != 200:
        return {"ok": False, "err": f"HTTP {r.status_code}", "latency_ms": lat}
    d = r.json()
    msg = d["choices"][0].get("message") or {}
    c = msg.get("content")
    if c is None:
        return {"ok": False, "err": "content=None", "latency_ms": lat}
    u = d.get("usage") or {}
    return {
        "ok": True,
        "content": c,
        "latency_ms": lat,
        "completion_tokens": u.get("completion_tokens", 0),
    }


def parse_summary(c: str) -> str:
    try:
        return str(json.loads(c).get("摘要", ""))
    except Exception:  # noqa: BLE001
        return ""


def main() -> None:
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        raise SystemExit("缺 LLM_API_KEY")

    items = load_items()
    store: dict = {}
    if OUTPUT.exists():
        store = json.loads(OUTPUT.read_text(encoding="utf-8"))

    for name, fn in PROMPTS.items():
        print(f"\n== {name} ==")
        rows = []
        for it in items:
            r = call(fn(it), api_key)
            if r["ok"]:
                s = parse_summary(r["content"])
                lim = it["max_chars"]
                must = it["must_include"]
                hits = [m for m in must if m in s]
                r.update(
                    summary_len=len(s),
                    limit=lim,
                    compliant=len(s) <= lim,
                    point_rate=len(hits) / len(must),
                    lost=[m for m in must if m not in s],
                )
                flag = "" if r["compliant"] else f" 超{r['summary_len'] - lim}"
                print(
                    f"  {it['id']:<8} {r['summary_len']:>4}/{lim:<4}{flag:<8} "
                    f"要点 {len(hits)}/{len(must)}  {r['latency_ms']}ms"
                )
            else:
                r.update(compliant=False, point_rate=0.0, lost=[])
                print(f"  {it['id']:<8} ❌ {r['err']}")
            rows.append({**r, "item_id": it["id"]})

        ok = [r for r in rows if r["ok"]]
        store[name] = {
            "n": len(rows),
            "ok": len(ok),
            "len_compliance": round(sum(1 for r in ok if r["compliant"]) / len(ok), 3)
            if ok
            else 0.0,
            "point_rate": round(statistics.mean(r["point_rate"] for r in ok), 3) if ok else 0.0,
            "mean_latency_ms": round(statistics.mean(r["latency_ms"] for r in ok)) if ok else 0,
            "mean_completion_tokens": round(statistics.mean(r["completion_tokens"] for r in ok))
            if ok
            else 0,
            "lost_terms": [t for r in ok for t in r["lost"]],
            "rows": rows,
        }
        OUTPUT.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print(f"{'方案':<16}{'成功':>7}{'长度合规':>10}{'要点率':>9}{'平均延迟':>10}{'compl_tok':>11}")
    print("=" * 76)
    for name, v in store.items():
        print(
            f"{name:<16}{v['ok']:>4}/{v['n']:<2}{v['len_compliance']:>10.2f}"
            f"{v['point_rate']:>9.3f}{v['mean_latency_ms']:>9}ms"
            f"{v['mean_completion_tokens']:>11}"
        )
    print(f"\n[done] 写入 {OUTPUT.name}")


if __name__ == "__main__":
    main()
