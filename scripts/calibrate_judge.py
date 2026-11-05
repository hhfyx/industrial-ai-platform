"""W7 周五：judge rubric 冻结 + 防偏差 + 人工校准。

## 本脚本做什么

    python -X utf8 scripts/calibrate_judge.py --calibrate          # 生成盲判材料
    python -X utf8 scripts/calibrate_judge.py --grade 人工判定.json # 算一致率
    python -X utf8 scripts/calibrate_judge.py --faithfulness       # Faithfulness 校准

## rubric 冻结（20:00-20:25）

rubric 文本写死在 JUDGE_RUBRIC，git commit 锁时间戳。改 rubric = 新 commit，
校准记录里必须引用对应 commit —— **防事后放宽**。

## 防偏差四条（20:25-20:50）

1. **位置偏差**：本评测是单答案判分（非 A/B 对比），无候选排序 → 该偏差不适用，
   已在 RUBRIC_APPLIES 里写明；Faithfulness 校准的多答案场景按随机序展示
2. **偏好泄露**：judge 看不到答案来自哪个系统/prompt 版本（提示词里无来源信息）
3. **长度偏好**：rubric 明确"更长/更详细不加分，缺一个数字照样扣"
4. **自我偏好**：生成用 buddy-pool/glm-5.3，judge 用 deepseek-account/deepseek-flash
   （不同家族，已实测网关两个模型都可用）；若换不了模型，退路是人工抽检 30%，
   本仓选前者

## 校准纪律（21:20-22:00）

**先人工判、再看 judge**（W3/W5 E5 第三次出现）：--calibrate 产出盲判表
（只含问题/标准答案/答案，无 judge 结论），人填完 --grade 才对出 judge 结果。
一致率 <85% 改 rubric 重跑；每次改动记 CHANGELOG。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pydantic import BaseModel, Field  # noqa: E402

CALIB_DIR = PROJECT_ROOT / "eval" / "calibration"
SEED = 20261204

#: 生成模型与 judge 模型（防自我偏好：不同家族）
GENERATOR_MODEL = "buddy-pool/glm-5.3"
JUDGE_MODEL = "deepseek-account/deepseek-flash"

# ---------------------------------------------------------------- rubric（冻结）
JUDGE_RUBRIC = """\
你是评测 judge。只判下面列出的东西，其他一律不判。

【判什么】
1. 关键事实是否与标准答案一致（结论层面，不是措辞层面）
2. 数字是否一致（0.575 与 0.58 是不一致；"约 0.575" 与 "0.575" 是一致）
3. 单位是否一致（ms 与 s 不是一致）
4. 否定词是否一致（"不解决" 与 "解决" 是相反的）

【不判什么】
- 措辞风格、是否口语化
- 长度：更长的答案**不加分**；标准答案有的关键点缺一个就扣
- 是否更详细、是否多说了标准答案之外的话（多说但不错 = correct，除非说错）
- 回答的格式（JSON/纯文本/分点）

【判定】
- correct：关键事实、数字、单位、否定词全部与标准答案一致
- partially_correct：主要事实对，但标准答案的要点缺 1 个，或一处细节存疑
- wrong：关键事实错、或数字/单位/否定词错、或答非所问

输出 JSON：{"verdict": "correct|partially_correct|wrong", "reason": "≤30字", "key_fact_ok": true|false}
key_fact_ok = 答案是否包含标准答案的那个**唯一关键事实**（数字/型号/结论）。
"""

RUBRIC_APPLIES = {
    "position_bias": "不适用：单答案判分，无候选排序。Faithfulness 校准的多答案场景按随机序（seed=20261204）展示",
    "preference_leak": "适用：judge 提示词不含任何来源/系统/prompt 版本信息",
    "length_preference": "适用：rubric 明确「更长不加分；缺一个关键点就扣」",
    "self_preference": "适用：生成=glm-5.3（buddy-pool），judge=deepseek-flash（不同家族）",
}


class JudgeOut(BaseModel):
    verdict: str = Field(description="correct / partially_correct / wrong")
    reason: str = Field(description="≤30 字")
    key_fact_ok: bool = Field(description="答案是否包含唯一关键事实")


def make_judge_prompt(item: dict, answer: str) -> str:
    """rubric + 题目材料。不含任何来源信息（防偏好泄露）。"""
    multi = "\n".join(f"- {a['quote']}" for a in item.get("anchors") or [])
    return (
        JUDGE_RUBRIC
        + f"\n【问题】{item['question']}\n"
        + f"【标准答案】{item['gold_answer']}\n"
        + (f"【原文锚点】\n{multi}\n" if multi else "")
        + f"【待判回答】{answer[:600]}"
    )


async def judge_with_model(item: dict, answer: str) -> dict:
    """用独立 judge 模型跑判定（不同家族防自我偏好）。"""
    from app.core.config import get_settings
    from app.core.llm.client import get_llm_client
    from app.core.llm.structured import DEFAULT_TEMPERATURE, _request, build_prompt

    # 复用 structured 的请求路径但换 client/model —— 保持同一 json_object 行为
    settings = get_settings()
    client = get_llm_client(settings)
    messages = [
        {"role": "user", "content": build_prompt(make_judge_prompt(item, answer), JudgeOut)}
    ]
    raw = await _request(
        client, model=JUDGE_MODEL, messages=messages, temperature=DEFAULT_TEMPERATURE
    )
    return JudgeOut.model_validate_json(raw).model_dump()


def build_calibration(n: int = 30) -> None:
    """产出盲判表：人先判，无 judge 结论。

    池 = 最新报告里有答案的记录。slice 只有 14 条可答，
    不足 30 时**先用 run_eval 补生成**（--questions 传 120 条的 fact+multi_hop
    子集，只为校准花这笔钱是值当的 —— 校准不足的 judge 给出的所有数字都不可信）。
    """
    reports = sorted(
        (PROJECT_ROOT / "eval" / "reports").glob("*.json"), key=lambda p: p.stat().st_mtime
    )
    recs: list[dict] = []
    for report in reversed(reports):
        payload = json.loads(report.read_text(encoding="utf-8"))
        g = payload.get("generation") or {}
        recs = [r for r in g.get("records", []) if r.get("answer")]
        if len(recs) >= n:
            break
    rng = random.Random(SEED)
    if len(recs) < n:
        print(f"⚠ 报告里只有 {len(recs)} 条答案，目标 {n} 条 —— 先跑全量补生成：")
        print("  python -X utf8 scripts/run_eval.py --questions eval/questions-120-calib.jsonl")
        picks = recs  # 有多少用多少，绝不用旧答案凑数
    else:
        picks = rng.sample(recs, n)

    CALIB_DIR.mkdir(exist_ok=True)
    blind = []
    for r in picks:
        item = _find_item(r["id"])
        blind.append(
            {
                "id": r["id"],
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "answer": r["answer"],
                "human_verdict": "",  # 人填：correct / partially_correct / wrong
                "human_key_fact_ok": "",  # 人填：true / false
            }
        )
    # 随机序展示（防位置偏差在多材料场景的残留）
    rng.shuffle(blind)
    for i, b in enumerate(blind, 1):
        b["no"] = i
    out = CALIB_DIR / "blind-30.json"
    out.write_text(json.dumps(blind, ensure_ascii=False, indent=1), encoding="utf-8")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
    ).stdout.strip()[:10]
    log = {
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit": commit,
        "rubric_note": "改 rubric 必须新 commit；本文件记录校准时点的 rubric 指纹",
        "rubric_sha256": __import__("hashlib").sha256(JUDGE_RUBRIC.encode()).hexdigest()[:16],
        "mitigations": RUBRIC_APPLIES,
        "generator_model": GENERATOR_MODEL,
        "judge_model": JUDGE_MODEL,
        "n": len(blind),
        "instruction": "先人工填 human_verdict / human_key_fact_ok（不看 judge 输出），填完跑 --grade",
    }
    (CALIB_DIR / "rubric-freeze.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"✓ 盲判表 {len(blind)} 条 → {out.relative_to(PROJECT_ROOT)}")
    print(
        f"  rubric 冻结记录 → {CALIB_DIR / 'rubric-freeze.json'}（rubric sha {log['rubric_sha256']}）"
    )
    print("  ⚠ 纪律：先人工判，填完 blind-30.json 再跑 --grade 对出 judge 结果")


def _find_item(item_id: str) -> dict:
    for f in ("eval/questions-slice20.jsonl", "eval/questions-120.jsonl"):
        for line in (PROJECT_ROOT / f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                it = json.loads(line)
                if it["id"] == item_id:
                    return it
    raise SystemExit(f"找不到 {item_id}")


async def run_judge_side(blind: list[dict]) -> list[dict]:
    """judge 独立判盲判表里的 30 条（此时人工已填完）。"""
    results = []
    for b in blind:
        try:
            out = await judge_with_model(_find_item(b["id"]), b["answer"])
            results.append({"no": b["no"], "id": b["id"], **out, "status": "ok"})
        except Exception as exc:
            results.append(
                {"no": b["no"], "id": b["id"], "status": "failed", "error": str(exc)[:100]}
            )
    return results


def grade(judged_path: Path | None = None) -> None:
    """人工 vs judge 一致率 + 混淆矩阵。"""
    blind_path = CALIB_DIR / "blind-30.json"
    blind = json.loads(blind_path.read_text(encoding="utf-8"))
    missing = [b["no"] for b in blind if not b["human_verdict"]]
    if missing:
        raise SystemExit(f"人工还没判完，缺 no={missing} —— 先判再看 judge（纪律）")

    if judged_path is None:
        judged = asyncio.run(run_judge_side(blind))
        (CALIB_DIR / "judge-30.json").write_text(
            json.dumps(judged, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    else:
        judged = json.loads(judged_path.read_text(encoding="utf-8"))
    jmap = {j["no"]: j for j in judged}

    agree = 0
    total = 0
    confusion: Counter = Counter()
    key_fact_agree = 0
    detail = []
    for b in blind:
        j = jmap.get(b["no"], {})
        if j.get("status") != "ok":
            detail.append(f"[no{b['no']} {b['id']}] judge 失败：{j.get('error', '')}")
            continue
        total += 1
        hv, jv = b["human_verdict"], j["verdict"]
        confusion[(hv, jv)] += 1
        if hv == jv:
            agree += 1
        if str(b["human_key_fact_ok"]).lower() == str(j.get("key_fact_ok", "")).lower():
            key_fact_agree += 1
        detail.append(f"[no{b['no']} {b['id']}] 人={hv} judge={jv} ({j['reason'][:30]})")

    rate = agree / total if total else 0
    print(f"一致率：{agree}/{total} = {rate:.1%}（目标 ≥85%）")
    print(f"key_fact 一致：{key_fact_agree}/{total}")
    print("混淆（人 → judge）：")
    for (hv, jv), n in sorted(confusion.items()):
        mark = " ← judge 偏松" if hv != jv and jv in ("correct",) else ""
        print(f"  {hv} → {jv}: {n}{mark}")
    print()
    print("\n".join(detail))
    (CALIB_DIR / "calibration-result.json").write_text(
        json.dumps(
            {
                "agree": agree,
                "total": total,
                "rate": round(rate, 4),
                "key_fact_agree": key_fact_agree,
                "confusion": {f"{k[0]}->{k[1]}": v for k, v in confusion.items()},
                "detail": detail,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\n落盘 → {CALIB_DIR / 'calibration-result.json'}")


def faithfulness(n: int = 15) -> None:
    """Faithfulness 校准：每句能否在上下文找到支撑；引用越界识别（W4 F3 复现）。"""
    from run_eval import retrieve_contexts  # 复用同一份检索实现（冻结口径）

    from app.core.llm.structured import structured_call

    class FaithOut(BaseModel):
        supported: bool = Field(description="答案的每一句都能在上下文找到支撑")
        invalid_citations: list[int] = Field(description="引用了不存在编号的列表")
        reason: str = Field(description="≤30 字")

    report = sorted(
        (PROJECT_ROOT / "eval" / "reports").glob("*.json"), key=lambda p: p.stat().st_mtime
    )[-1]
    payload = json.loads(report.read_text(encoding="utf-8"))
    recs = [
        r
        for r in payload["generation"]["records"]
        if r.get("answer") and r.get("citations") is not None
    ]
    rng = random.Random(SEED + 1)
    picks = rng.sample(recs, min(n, len(recs)))

    out_rows = []
    for r in picks:
        item = _find_item(r["id"])
        ctx_ids = sorted({c["id"] for c in retrieve_contexts(item, 5)})
        cited = r["citations"]
        # 程序化判定引用越界（W4 F3 的机器真值，judge 要能对上它）
        program_invalid = [c for c in cited if c not in ctx_ids]
        prompt = (
            "你是 judge。判断【回答】的每一句是否能在【上下文】里找到支撑；"
            "并列出回答里引用了但不存在的编号。\n"
            f"【上下文编号】{ctx_ids}\n"
            f"【回答】{r['answer'][:500]}\n"
            f"【回答里的引用编号】{cited}"
        )
        try:
            res = asyncio.run(structured_call(prompt, FaithOut))
            row = {
                "id": r["id"],
                "cited": cited,
                "ctx_ids": ctx_ids,
                "program_invalid_refs": program_invalid,
                "judge_invalid_refs": res.invalid_citations,
                "judge_supported": res.supported,
                "reason": res.reason,
                "f3_detected_agree": sorted(res.invalid_citations) == sorted(program_invalid),
            }
        except Exception as exc:
            row = {"id": r["id"], "status": "failed", "error": str(exc)[:100]}
        out_rows.append(row)

    ok = [r for r in out_rows if "status" not in r]
    agree = sum(1 for r in ok if r["f3_detected_agree"])
    print(
        f"Faithfulness 校准：{len(ok)}/{len(out_rows)} 判定成功，引用越界识别一致 {agree}/{len(ok)}"
    )
    for r in out_rows:
        if "status" in r:
            print(f"  [{r['id']}] 失败: {r['error']}")
        else:
            flag = "✓" if r["f3_detected_agree"] else "✗ 不一致"
            print(
                f"  [{r['id']}] 引用{r['cited']} 上下文{r['ctx_ids']} 程序判定越界={r['program_invalid_refs']} judge={r['judge_invalid_refs']} {flag}"
            )
    (CALIB_DIR / "faithfulness-15.json").write_text(
        json.dumps(out_rows, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"落盘 → {CALIB_DIR / 'faithfulness-15.json'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true", help="产出盲判表（人先判）")
    parser.add_argument("--grade", nargs="?", const=None, help="算一致率（可传 judge 结果文件）")
    parser.add_argument("--faithfulness", action="store_true")
    args = parser.parse_args()
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    if args.calibrate:
        build_calibration()
        return 0
    if args.faithfulness:
        faithfulness()
        return 0
    if args.grade is not None or "--grade" in sys.argv:
        grade(Path(args.grade) if args.grade else None)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
