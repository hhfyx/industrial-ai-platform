"""W4 开工前第一件事：摘要任务的 reasoning 档位该选哪个。

这是 W3 留下的**未决技术决策**：摘要 P95 55 秒，不适合线上。
本脚本把四个档位各跑同一批摘要，量出"延迟 vs 质量"的实际权衡。

**为什么必须先做这件事**：W4 要建 RAG 链路，生成层会继承这个延迟。
不先定档位，就会把"摘要慢"误判成"RAG 慢"，然后在错误的地方优化。

四个档位（实测网关上 deepseek-account/deepseek-flash 支持）：
  off  low  high(当前默认)  max

指标：平均延迟、P95、每条成功成本、must_include 要点命中率、成功率。
**成本口径按每条成功归一**（W3 的教训，见 docs/weekly/w3.md 3.1）。

运行：
    python -X utf8 scripts/reasoning_ablation.py
    python -X utf8 scripts/reasoning_ablation.py --levels off,low
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

HERE = Path(__file__).resolve().parent
SETS = HERE.parent / "eval" / "prompt-sets"
OUTPUT = HERE / "reasoning_ablation.json"

ENDPOINT = "http://127.0.0.1:8326/v1/chat/completions"
MODEL = "deepseek-account/deepseek-flash"
MAX_TOKENS = 8192
TIMEOUT = 120.0


def load_items() -> list[dict]:
    rows = []
    for line in (SETS / "summarize.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_prompt(item: dict) -> str:
    """直接用 v1 模板渲染（已定版，不改）。"""
    from app.prompts.loader import load_prompt

    return load_prompt(
        "summarize",
        version="v1",
        text=item["text"],
        max_chars=item.get("max_chars", 200),
        must_keep=item.get("must_include", []),
    )


def call(prompt: str, effort: str, api_key: str) -> dict[str, Any]:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "reasoning_effort": effort,
        "response_format": {"type": "json_object"},
    }
    t0 = time.perf_counter()
    try:
        r = httpx.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            timeout=TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "err": f"{type(exc).__name__}: {exc}"[:150],
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }
    lat = int((time.perf_counter() - t0) * 1000)

    if r.status_code != 200:
        return {"ok": False, "err": f"HTTP {r.status_code}: {r.text[:150]}", "latency_ms": lat}

    d = r.json()
    ch = d["choices"][0]
    msg = ch.get("message") or {}
    u = d.get("usage") or {}
    content = msg.get("content")
    if content is None:
        return {
            "ok": False,
            "err": "content=None",
            "latency_ms": lat,
            "finish": ch.get("finish_reason"),
            "reasoning_chars": len(msg.get("reasoning_content") or ""),
        }
    return {
        "ok": True,
        "content": content,
        "latency_ms": lat,
        "prompt_tokens": u.get("prompt_tokens", 0),
        "completion_tokens": u.get("completion_tokens", 0),
        "reasoning_chars": len(msg.get("reasoning_content") or ""),
    }


def score(content: str, item: dict) -> tuple[float, str]:
    """must_include 逐字命中（与 ab_test 的 summarize 判定一致）。"""
    import re

    try:
        d = json.loads(content.strip())
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            d = {}
    summary = str(d.get("摘要", ""))
    must = item["must_include"]
    hits = [x for x in must if x in summary]
    rate = len(hits) / len(must) if must else 0.0
    over = len(summary) > item.get("max_chars", 10**9)
    return rate, f"{len(hits)}/{len(must)}" + (" 超长" if over else "")


def percentile(vals: list[int], p: int) -> int:
    if not vals:
        return 0
    import math

    s = sorted(vals)
    k = max(1, math.ceil(len(s) * p / 100))
    return s[min(k, len(s)) - 1]


async def run_level(level: str, items: list[dict], api_key: str) -> dict[str, Any]:
    print(f"\n== reasoning_effort={level} ==")
    rows = []
    for item in items:
        prompt = build_prompt(item)
        r = call(prompt, level, api_key)
        if r["ok"]:
            rate, detail = score(r["content"], item)
            r["score"] = rate
            r["detail"] = detail
            print(
                f"  ✅ {item['id']:<8} {r['latency_ms']:>6}ms  "
                f"compl={r['completion_tokens']:>5}  "
                f"reasoning={r['reasoning_chars']:>5}字  要点 {detail}"
            )
        else:
            r["score"] = 0.0
            print(f"  ❌ {item['id']:<8} {r['latency_ms']:>6}ms  {r['err']}")
        rows.append({**r, "item_id": item["id"]})

    ok = [r for r in rows if r["ok"]]
    lat = [r["latency_ms"] for r in ok]
    n = len(rows)
    return {
        "level": level,
        "n": n,
        "ok": len(ok),
        "mean_score_of_ok": round(statistics.mean(r["score"] for r in ok), 4) if ok else 0.0,
        "mean_score_all": round(sum(r["score"] for r in rows) / n, 4),
        "mean_latency_ms": round(statistics.mean(lat)) if lat else 0,
        "p95_latency_ms": percentile(lat, 95),
        "mean_completion_tokens_of_ok": round(statistics.mean(r["completion_tokens"] for r in ok))
        if ok
        else 0,
        "mean_reasoning_chars_of_ok": round(statistics.mean(r["reasoning_chars"] for r in ok))
        if ok
        else 0,
        "rows": rows,
    }


async def main() -> None:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="off,low,high,max")
    args = ap.parse_args()

    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        raise SystemExit("缺 LLM_API_KEY")

    items = load_items()
    print(f"model={MODEL}  items={len(items)}  max_tokens={MAX_TOKENS}")
    print("这是一次**真实计费**的消融实验：4 档 × 10 条 = 40 次调用")

    # **累积保存，不覆盖已有档位**。
    # 第一版每次新建 store，结果跑 `--levels off,high` 再跑 `--levels low,max`
    # 就把前两档的数据冲掉了 —— 这是本周第二次犯同一个错（上次在 lc_extract.py），
    # 一次实验要花真钱，冲掉就得重跑。
    if OUTPUT.exists():
        store: dict[str, Any] = json.loads(OUTPUT.read_text(encoding="utf-8"))
    else:
        store = {"meta": {}, "levels": {}}
    store.setdefault("levels", {})
    store["meta"].update({"model": MODEL, "n": len(items), "max_tokens": MAX_TOKENS})

    for level in [x.strip() for x in args.levels.split(",") if x.strip()]:
        store["levels"][level] = await run_level(level, items, api_key)
        OUTPUT.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(
        f"{'档位':<8}{'成功':>6}{'要点(仅成功)':>12}{'要点(全量)':>12}"
        f"{'平均延迟':>10}{'P95':>9}{'completion':>11}{'思维链字数':>11}"
    )
    print("=" * 72)
    for level, v in store["levels"].items():
        print(
            f"{level:<8}{v['ok']:>4}/{v['n']:<2}"
            f"{v['mean_score_of_ok']:>12.3f}{v['mean_score_all']:>12.3f}"
            f"{v['mean_latency_ms']:>9}ms{v['p95_latency_ms']:>8}ms"
            f"{v['mean_completion_tokens_of_ok']:>11}{v['mean_reasoning_chars_of_ok']:>11}"
        )
    print(f"\n[done] 写入 {OUTPUT.name}")


if __name__ == "__main__":
    asyncio.run(main())
