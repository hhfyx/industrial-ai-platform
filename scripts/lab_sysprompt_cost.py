"""代价测试：严格版(B)会不会把正常工单也拒掉（过度拒绝率）。

B 版加了「不在服务范围」这个强拒答，风险是**合法工单也被误判为越界**。
用周一那 20 条真实工单做对照组（它们全都属于服务范围），
对比 A/B 两版在这些工单上的分类质量。

注意：这里测的不是准确率（金标口径不同），而是
  ① 过度拒绝率 —— 正常工单被判成「不在服务范围」的比例
  ② 分类质量   —— 三字段与金标的一致度
两版用同一批数据、同一个判定，能直接比。

运行：python -X utf8 lab_sysprompt_cost.py
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import httpx

API_KEY = os.environ.get("LLM_API_KEY", "")
BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8326/v1").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "deepseek-account/deepseek-flash")
ENDPOINT = f"{BASE_URL}/chat/completions"

HERE = Path(__file__).resolve().parent
AB = HERE / "system_prompt_ab.json"
WORK = HERE / "prompt_patterns_dataset.json"
OUTPUT = HERE / "sysprompt_cost_result.json"

FIELDS = ("问题类型", "紧急程度", "涉及产品")


def call(system: str, user: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"工单：{user}"},
        ],
        "max_tokens": 400,
        "temperature": 0.0,
        "reasoning_effort": "none",
    }
    t0 = time.perf_counter()
    try:
        r = httpx.post(
            ENDPOINT, headers={"Authorization": f"Bearer {API_KEY}"}, json=body, timeout=90
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    lat = int((time.perf_counter() - t0) * 1000)
    if r.status_code != 200:
        return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:150]}"}
    d = r.json()
    msg = d["choices"][0].get("message") or {}
    u = d.get("usage") or {}
    return {
        "ok": True,
        "content": msg.get("content") or "",
        "prompt_tokens": u.get("prompt_tokens", 0),
        "completion_tokens": u.get("completion_tokens", 0),
        "latency_ms": lat,
        "finish_reason": d["choices"][0].get("finish_reason"),
    }


def parse(text: str) -> tuple[dict, bool]:
    t = text.strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj, True
    except json.JSONDecodeError:
        pass
    return {}, False


def main() -> None:
    if not API_KEY:
        raise SystemExit("缺 LLM_API_KEY")

    ab = json.loads(AB.read_text(encoding="utf-8"))
    work = json.loads(WORK.read_text(encoding="utf-8"))
    items = work["eval"]
    print(f"endpoint={ENDPOINT}\nmodel={MODEL}\nlegit tickets={len(items)}")

    store = json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() else {}
    store.setdefault("runs", {})
    store["meta"] = {
        "endpoint": ENDPOINT,
        "model": MODEL,
        "n": len(items),
        "purpose": "B 版过度拒绝率（正常工单被误判为越界）",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    for arm in ("A_loose", "B_strict"):
        system = ab["prompts"][arm]["system"]
        done = {r["id"] for r in store["runs"].get(arm, []) if r.get("ok")}
        rows = store["runs"].setdefault(arm, [])
        print(f"\n== {arm} ==")
        for item in items:
            if item["id"] in done:
                continue
            r = call(system, item["text"])
            if not r["ok"]:
                print(f"  #{item['id']:<2} FAIL {r['error'][:60]}")
                rows.append({"id": item["id"], "ok": False, "error": r["error"]})
                OUTPUT.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
                continue
            obj, pure = parse(r["content"])
            gold = item["gold"]
            hit = {f: obj.get(f) == gold[f] for f in FIELDS}
            over_refuse = str(obj.get("问题类型", "")) == "不在服务范围"
            rows.append(
                {
                    "id": item["id"],
                    "ok": True,
                    "pure_json": pure,
                    "answer": {f: obj.get(f) for f in FIELDS},
                    "over_refused": over_refuse,
                    "over_refuse_unparseable": str(obj.get("问题类型", "")) == "无法判断"
                    and not pure,
                    "per_field": hit,
                    "all3": all(hit.values()),
                    "n_correct": sum(hit.values()),
                    "prompt_tokens": r["prompt_tokens"],
                    "completion_tokens": r["completion_tokens"],
                    "latency_ms": r["latency_ms"],
                    "raw": r["content"][:400],
                }
            )
            mark = (
                "OVR"
                if over_refuse
                else ("PASS" if all(hit.values()) else f"{sum(hit.values())}/3")
            )
            print(f"  #{item['id']:<2} {mark:<5} tok={r['prompt_tokens']}+{r['completion_tokens']}")
            OUTPUT.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(0.15)

    # 汇总
    summary = {}
    for arm, rows in store["runs"].items():
        good = [r for r in rows if r.get("ok")]
        if not good:
            continue
        n = len(good)
        summary[arm] = {
            "n": n,
            "over_refuse_rate": round(sum(1 for r in good if r["over_refused"]) / n, 4),
            "over_refuse_ids": [r["id"] for r in good if r["over_refused"]],
            "unparseable_ids": [r["id"] for r in good if not r["pure_json"]],
            "format_compliance": round(sum(1 for r in good if r["pure_json"]) / n, 4),
            "exact_all3": sum(1 for r in good if r["all3"]),
            "exact_rate": round(sum(1 for r in good if r["all3"]) / n, 4),
            "mean_field_acc": round(statistics.mean(r["n_correct"] / 3 for r in good), 4),
            "mean_prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in good)),
            "mean_latency_ms": round(statistics.mean(r["latency_ms"] for r in good)),
        }
    store["summary"] = summary
    OUTPUT.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n============ 代价测试：正常工单上的表现 ============")
    print(f"{'指标':<26}{'A 宽松':>10}{'B 严格':>10}")
    a, b = summary["A_loose"], summary["B_strict"]
    for label, key in [
        ("过度拒绝率(越低越好)", "over_refuse_rate"),
        ("格式合规率", "format_compliance"),
        ("整条全对率", "exact_rate"),
        ("字段级准确率", "mean_field_acc"),
        ("prompt token", "mean_prompt_tokens"),
        ("延迟 ms", "mean_latency_ms"),
    ]:
        print(f"{label:<26}{a.get(key):>10}{b.get(key):>10}")
    print(f"\nA 过度拒绝条目: {a['over_refuse_ids']}")
    print(f"B 过度拒绝条目: {b['over_refuse_ids']}")
    print(f"\n[done] 写入 {OUTPUT.name}")


if __name__ == "__main__":
    main()
