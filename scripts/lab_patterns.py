"""lab_patterns：Prompt 四范式的实测台（产出 docs/prompt-patterns.md 里的表）。

任务：从客服工单里抽取 {问题类型, 紧急程度, 涉及产品}（固定一个任务，跨范式可比）
数据集：scripts/prompt_patterns_dataset.json（20 条评测 + 6 条 few-shot 候选）

八个实验臂：
  A zero_shot            直接给答案，无示例            reasoning=off
  B few_order1           few-shot，3 个示例（原序）      reasoning=off
  C few_order2           few-shot，同样 3 示例（逆序）    reasoning=off  ← 隔离「顺序」
  D few_dist             few-shot，换 3 示例（标签分布不同）reasoning=off  ← 隔离「分布」
  E cot_prompt           Prompt 里要求先写理由再给答案    reasoning=off  ← "CoT 提示"
  F cot_native           直接给答案（无 CoT 提示）        reasoning=high ← 模型自带思维链
  G cot_prompt_native    CoT 提示 + 自带思维链            reasoning=high ← 叠加
  H self_consistency     E 的配置采样 5 次、多数投票       reasoning=off

E vs A：CoT 提示本身有没有用（同一个关闭推理的模型内对比）
F vs A：模型自带思维链有没有用
G vs F：在已经有思维链的模型上再加 CoT 提示 —— **会不会反而变差**
G vs F 就是必答题的直接证据。

运行：
    python -X utf8 lab_patterns.py            # 全量
    python -X utf8 lab_patterns.py --arms A,E  # 只跑指定臂

结果写 patterns_result.json；支持断点续跑（已完成的 (arm,item) 不再重跑）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path

import httpx

API_KEY = os.environ.get("LLM_API_KEY", "")
BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8326/v1").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "deepseek-account/deepseek-flash")
ENDPOINT = f"{BASE_URL}/chat/completions"

HERE = Path(__file__).resolve().parent
DATASET = HERE / "prompt_patterns_dataset.json"
OUTPUT = HERE / "patterns_result.json"

FIELDS = ("问题类型", "紧急程度", "涉及产品")

# ------------------------------------------------------------------ prompt

SYSTEM_ZERO = "你是客服工单分类助手。只输出 JSON，不要输出任何解释。"

SYSTEM_ANSWER_ONLY = (
    "你是客服工单分类助手。严格按照给定的取值集合判断，只输出 JSON，不要输出任何解释。"
)

SYSTEM_COT = """你是客服工单分类助手。请**先写理由，再给分类**。

理由部分要针对下面这几条判定规则逐条过一遍：
1. 涉及产品取「业务实际被阻断的那一个产品」；短信、邮件、企微只是通道，通道故障算到使用该通道的产品上。
2. 一条工单里出现多个问题时，取最先阻断业务的那一个。
3. 用户自称的「紧急」不算数，按 P0/P1/P2/P3 的定义判。
4. 纯咨询（问能力、问价格、问流程、提建议）一律归「咨询与售前」；紧急程度看是否真有时间压力。
5. 用户自己已经给出绕过方案时，不算 P0/P1。

输出格式：先用「理由：」开头写 2-4 句推理，再另起一行输出分类 JSON。"""


def label_block(taxonomy: dict) -> str:
    types = "、".join(taxonomy["问题类型"])
    prods = "、".join(taxonomy["涉及产品"])
    urg = "；".join(f"{k}={v}" for k, v in taxonomy["紧急程度"].items())
    return f"问题类型只能取：{types}\n涉及产品只能取：{prods}\n紧急程度只能取（含义）：{urg}"


def build_messages(arm: str, item_text: str, data: dict) -> list[dict]:
    tax = label_block(data["taxonomy"])
    devs = {d["id"]: d for d in data["dev_examples"]}

    def shots(ids: list[str]) -> str:
        out = []
        for i, did in enumerate(ids, 1):
            d = devs[did]
            out.append(
                f"示例{i}\n工单：{d['text']}\n分类：{json.dumps(d['answer'], ensure_ascii=False)}"
            )
        return "\n\n".join(out)

    sets = data["few_shot_sets"]

    if arm == "A":
        sys = f"{SYSTEM_ZERO}\n\n{tax}"
        return [
            {"role": "system", "content": sys},
            {
                "role": "user",
                "content": f'工单：{item_text}\n\n输出 JSON：{{"问题类型":"","紧急程度":"","涉及产品":""}}',
            },
        ]

    if arm == "E":
        sys = f"{SYSTEM_COT}\n\n{tax}"
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": f"工单：{item_text}"},
        ]

    if arm in ("B", "C", "D"):
        ids = list(sets["dist_X"])
        if arm == "C":
            ids = list(reversed(ids))  # 只改顺序
        elif arm == "D":
            ids = list(sets["dist_Y"])  # 只改标签分布
        sys = (
            f"{SYSTEM_ZERO}\n\n{tax}\n\n"
            f"下面是几个已标注的示例，请照着同样的格式作答：\n\n{shots(ids)}"
        )
        return [
            {"role": "system", "content": sys},
            {
                "role": "user",
                "content": f'工单：{item_text}\n\n输出 JSON：{{"问题类型":"","紧急程度":"","涉及产品":""}}',
            },
        ]

    # 直接给答案型（F/G 的对照基底，也用于 self-consistency 的同配置）
    sys = f"{SYSTEM_ANSWER_ONLY}\n\n{tax}"
    return [
        {"role": "system", "content": sys},
        {
            "role": "user",
            "content": f'工单：{item_text}\n\n输出 JSON：{{"问题类型":"","紧急程度":"","涉及产品":""}}',
        },
    ]


# ------------------------------------------------------------------ 各臂配置

ARMS: dict[str, dict] = {
    "A_zero_shot": {"arm": "A", "prompt": "zero", "reasoning": "none", "temp": 0.0},
    "B_few_order1": {"arm": "B", "prompt": "few", "reasoning": "none", "temp": 0.0},
    "C_few_order2": {"arm": "C", "prompt": "few", "reasoning": "none", "temp": 0.0},
    "D_few_dist": {"arm": "D", "prompt": "few", "reasoning": "none", "temp": 0.0},
    "E_cot_prompt": {"arm": "E", "prompt": "cot", "reasoning": "none", "temp": 0.0},
    "F_native_reasoning": {"arm": "F", "prompt": "direct", "reasoning": "high", "temp": 0.0},
    "G_cot_x_native": {"arm": "E", "prompt": "cot", "reasoning": "high", "temp": 0.0},
}

SC_ARM = "C_few_order2"  # self-consistency 用「本任务实测最优的单次配置」采样
SC_SAMPLES = 5
SC_TEMP = 0.7  # 全 0 温度时多次采样几乎重合，投票没意义

MAX_TOKENS = 3000


# ------------------------------------------------------------------ 调用


def call(messages: list[dict], reasoning: str, temperature: float) -> dict:
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
        "reasoning_effort": reasoning,
    }
    started = time.perf_counter()
    try:
        resp = httpx.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=body,
            timeout=180,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "latency_ms": 0}

    latency_ms = int((time.perf_counter() - started) * 1000)
    if resp.status_code != 200:
        return {
            "ok": False,
            "status": resp.status_code,
            "error": resp.text[:300],
            "latency_ms": latency_ms,
        }

    data = resp.json()
    msg = data["choices"][0].get("message") or {}
    usage = data.get("usage") or {}
    return {
        "ok": True,
        "status": 200,
        "content": msg.get("content") or "",
        "reasoning_content": msg.get("reasoning_content") or "",
        "finish_reason": data["choices"][0].get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
        "completion_tokens": usage.get("completion_tokens", 0) or 0,
        "latency_ms": latency_ms,
    }


# ------------------------------------------------------------------ 解析与打分

JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.S)


def parse_answer(text: str, taxonomy: dict) -> dict:
    """从模型输出里抠出三个字段。返回 {'answer':..,'parse':'ok|partial|fail','note':..}"""
    cands = JSON_BLOCK.findall(text)
    best: dict = {}
    for c in reversed(cands):  # 取最后一个完整 JSON
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if any(k in obj for k in FIELDS):
            best = obj
            break

    if not best:  # 退化：按行找 "问题类型: xxx"
        for f in FIELDS:
            m = re.search(rf"{f}\s*[:：]\s*([^\n,，;；}}]+)", text)
            if m:
                best[f] = m.group(1).strip().strip("\"' ")

    ans = {f: (str(best.get(f, "")).strip()) for f in FIELDS}
    if all(ans.values()):
        status = "ok"
    elif any(ans.values()):
        status = "partial"
    else:
        status = "fail"

    # 越界值检测（不在词表内 = 指令遵循失败）
    out_of_tax = []
    if ans["问题类型"] and ans["问题类型"] not in taxonomy["问题类型"]:
        out_of_tax.append("问题类型")
    if ans["涉及产品"] and ans["涉及产品"] not in taxonomy["涉及产品"]:
        out_of_tax.append("涉及产品")
    if ans["紧急程度"] and ans["紧急程度"] not in taxonomy["紧急程度"]:
        out_of_tax.append("紧急程度")

    return {"answer": ans, "parse": status, "out_of_taxonomy": out_of_tax}


def score(ans: dict, gold: dict) -> dict:
    hit = {f: ans.get(f) == gold[f] for f in FIELDS}
    return {
        "per_field": hit,
        "all3": all(hit.values()),
        "n_correct_fields": sum(hit.values()),
    }


def majority(votes: list[dict], gold: dict) -> dict:
    """按整条答案做多数投票；全部不同时退回「字段级多数」。"""
    from collections import Counter

    keys = [tuple(v[f] for f in FIELDS) for v in votes]
    keys = [k for k in keys if any(k)]
    if not keys:
        return dict.fromkeys(FIELDS, "")
    common = Counter(keys).most_common(1)[0]
    if common[1] > 1:
        return dict(zip(FIELDS, common[0], strict=True))
    field_vote = {}
    for f in FIELDS:
        vals = [v.get(f, "") for v in votes if v.get(f)]
        field_vote[f] = Counter(vals).most_common(1)[0][0] if vals else ""
    return field_vote


# ------------------------------------------------------------------ 主流程


def load_done(store: dict) -> set:
    done = set()
    for arm, rows in store.get("runs", {}).items():
        for row in rows:
            done.add((arm, row["item_id"]))
    return done


def run_arm(arm_name: str, data: dict, store: dict, done: set) -> None:
    cfg = ARMS[arm_name]
    rows = store.setdefault("runs", {}).setdefault(arm_name, [])
    print(
        f"\n== {arm_name} (prompt={cfg['prompt']}, reasoning={cfg['reasoning']}, T={cfg['temp']}) =="
    )

    for item in data["eval"]:
        if (arm_name, item["id"]) in done:
            continue
        msgs = build_messages(cfg["arm"], item["text"], data)
        r = call(msgs, cfg["reasoning"], cfg["temp"])
        if not r["ok"]:
            print(f"  #{item['id']:<2} FAILED {r.get('error')[:80]}")
            rows.append({"item_id": item["id"], "ok": False, "error": r.get("error")})
            save(store)
            time.sleep(1.0)
            continue

        parsed = parse_answer(r["content"], data["taxonomy"])
        sc = score(parsed["answer"], item["gold"])
        rows.append(
            {
                "item_id": item["id"],
                "ok": True,
                "answer": parsed["answer"],
                "parse": parsed["parse"],
                "out_of_taxonomy": parsed["out_of_taxonomy"],
                "score": sc,
                "finish_reason": r["finish_reason"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "reasoning_chars": len(r["reasoning_content"]),
                "content_chars": len(r["content"]),
                "latency_ms": r["latency_ms"],
                "raw": r["content"][:800],
            }
        )
        mark = "PASS" if sc["all3"] else f"{sc['n_correct_fields']}/3"
        print(
            f"  #{item['id']:<2} {mark:<5} tok={r['prompt_tokens']}+{r['completion_tokens']} lat={r['latency_ms']}ms"
        )
        save(store)
        time.sleep(0.15)


def run_self_consistency(data: dict, store: dict) -> None:
    global SC_ARM
    cfg = ARMS[SC_ARM]
    print(f"\n== I_self_consistency (base={SC_ARM}, {SC_SAMPLES} samples, T={SC_TEMP}) ==")
    rows = store.setdefault("runs", {}).setdefault("I_self_consistency", [])

    for item in data["eval"]:
        samples = []
        for _k in range(SC_SAMPLES):
            msgs = build_messages(cfg["arm"], item["text"], data)
            r = call(msgs, cfg["reasoning"], SC_TEMP)
            if not r["ok"]:
                continue
            p = parse_answer(r["content"], data["taxonomy"])
            samples.append(
                {
                    "answer": p["answer"],
                    "parse": p["parse"],
                    "prompt_tokens": r["prompt_tokens"],
                    "completion_tokens": r["completion_tokens"],
                    "latency_ms": r["latency_ms"],
                }
            )
            time.sleep(0.15)
        if not samples:
            rows.append({"item_id": item["id"], "ok": False, "error": "all samples failed"})
            save(store)
            continue

        voted = majority([s["answer"] for s in samples], item["gold"])
        sc = score(voted, item["gold"])
        rows.append(
            {
                "item_id": item["id"],
                "ok": True,
                "n_samples": len(samples),
                "votes": [s["answer"] for s in samples],
                "voted": voted,
                "score": sc,
                "prompt_tokens": sum(s["prompt_tokens"] for s in samples),
                "completion_tokens": sum(s["completion_tokens"] for s in samples),
                "latency_ms": sum(s["latency_ms"] for s in samples),
            }
        )
        nc = sc["n_correct_fields"]
        mark = "PASS" if sc["all3"] else f"{nc}/3"
        print(f"  #{item['id']:<2} {mark:<5} samples={len(samples)}")
        save(store)


def summarise(store: dict, data: dict) -> dict:
    golds = {i["id"]: i["gold"] for i in data["eval"]}
    out: dict = {}

    for arm_name, rows in store.get("runs", {}).items():
        good = [r for r in rows if r.get("ok")]
        if not good:
            continue
        n = len(good)
        all3 = sum(1 for r in good if r["score"]["all3"])
        per_field = {f: sum(1 for r in good if r["score"]["per_field"][f]) / n for f in FIELDS}
        mean_fields = statistics.mean(r["score"]["n_correct_fields"] / 3 for r in good)
        pt = [r["prompt_tokens"] for r in good]
        ct = [r["completion_tokens"] for r in good]
        out[arm_name] = {
            "n": n,
            "exact_all3": all3,
            "exact_all3_rate": round(all3 / n, 4),
            "mean_field_accuracy": round(mean_fields, 4),
            "per_field_accuracy": {k: round(v, 4) for k, v in per_field.items()},
            "parse_fail": sum(1 for r in good if r.get("parse") == "fail"),
            "parse_partial": sum(1 for r in good if r.get("parse") == "partial"),
            "out_of_taxonomy": sum(1 for r in good if r.get("out_of_taxonomy")),
            "finish_length": sum(1 for r in good if r.get("finish_reason") == "length"),
            "mean_prompt_tokens": round(statistics.mean(pt)),
            "mean_completion_tokens": round(statistics.mean(ct)),
            "mean_total_tokens": round(statistics.mean(pt) + statistics.mean(ct)),
            "total_tokens": sum(pt) + sum(ct),
            "mean_latency_ms": round(statistics.mean(r["latency_ms"] for r in good)),
            "total_latency_ms": sum(r["latency_ms"] for r in good),
        }

    # 相对基线 A 的差值
    base = out.get("A_zero_shot")
    if base:
        for _k, v in out.items():
            v["delta_exact_vs_A"] = round(v["exact_all3_rate"] - base["exact_all3_rate"], 4)
            v["token_multiplier_vs_A"] = round(
                v["mean_total_tokens"] / base["mean_total_tokens"], 2
            )

    # self-consistency：多数投票 vs 单采样（同配置、切到 T=0.7 后单次采样的成绩）
    sc = store.get("runs", {}).get("I_self_consistency")
    if sc:
        good = [r for r in sc if r.get("ok")]
        if good:
            n = len(good)
            singles = [score(v, golds[r["item_id"]])["all3"] for r in good for v in r["votes"] if v]
            all_single = [score(v, golds[r["item_id"]]) for r in good for v in r["votes"] if v]
            voted_all3 = sum(1 for r in good if r["score"]["all3"])
            out["_self_consistency"] = {
                "n_items": n,
                "single_sample_exact_rate": round(sum(singles) / len(singles), 4)
                if singles
                else None,
                "single_sample_mean_field_acc": round(
                    statistics.mean(s["n_correct_fields"] / 3 for s in all_single), 4
                )
                if all_single
                else None,
                "voted_exact_rate": round(voted_all3 / n, 4),
                "voted_mean_field_acc": round(
                    statistics.mean(r["score"]["n_correct_fields"] / 3 for r in good), 4
                ),
                "cost_multiplier": SC_SAMPLES,
                "mean_tokens_per_item": round(
                    statistics.mean(r["prompt_tokens"] + r["completion_tokens"] for r in good)
                ),
            }
    return out


def save(store: dict) -> None:
    OUTPUT.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="", help="逗号分隔，如 A_zero_shot,E_cot_prompt；空=全部")
    ap.add_argument("--skip-sc", action="store_true")
    args = ap.parse_args()

    if not API_KEY:
        raise SystemExit("缺 LLM_API_KEY：设环境变量，或用 .env 里的值 export")

    data = json.loads(DATASET.read_text(encoding="utf-8"))
    store = json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() else {}
    store.setdefault("meta", {})
    store["meta"].update(
        {
            "endpoint": ENDPOINT,
            "model": MODEL,
            "dataset": DATASET.name,
            "n_eval": len(data["eval"]),
            "fields": list(FIELDS),
            "max_tokens": MAX_TOKENS,
            "sc_samples": SC_SAMPLES,
            "sc_temp": SC_TEMP,
            "sc_base_arm": SC_ARM,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )

    done = load_done(store)
    todo = [a.strip() for a in args.arms.split(",") if a.strip()] or list(ARMS)

    print(f"endpoint={ENDPOINT}\nmodel={MODEL}\neval={len(data['eval'])} items\narms={todo}")

    for arm_name in todo:
        if arm_name not in ARMS:
            print(f"!! 未知臂 {arm_name}，跳过")
            continue
        run_arm(arm_name, data, store, done)

    if not args.skip_sc:
        run_self_consistency(data, store)

    store["summary"] = summarise(store, data)
    save(store)

    print("\n================ 汇总 ================")
    print(f"{'arm':<22}{'all3':>7}{'ΔvsA':>8}{'field':>8}{'tok':>8}{'latency':>10}")
    for k, v in store["summary"].items():
        if k.startswith("_"):
            continue
        print(
            f"{k:<22}{v['exact_all3']:>4}/{v['n']:<3}"
            f"{v.get('delta_exact_vs_A', 0):>+8.2f}"
            f"{v['mean_field_accuracy']:>8.3f}"
            f"{v['mean_total_tokens']:>8}"
            f"{v['mean_latency_ms']:>9}ms"
        )
    print(f"\n[done] 结果写入 {OUTPUT.name}")


if __name__ == "__main__":
    main()
