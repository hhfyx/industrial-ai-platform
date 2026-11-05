"""W7 一条命令出全部指标：检索（自算）+ 生成 + judge。

    python -X utf8 scripts/tasks.py run-eval --questions eval/questions-slice20.jsonl

## 结构（复用优先，不重写）

- 检索指标：复用 `run_retrieval_eval` 的锚点反查 + summarize（W6 修好口径的那份），
  只补 NDCG@5 —— 锚点 1~2 个、二元相关度，实现 10 行，不引第三方
- 生成：`structured_call` + qa 模板（引用编号 + 不确定字段，自带 2 轮修复重试）
- judge：**不用 RAGAS/DeepEval** —— 它们的 faithfulness 用 LLM 拆句判支持，
  口径与本仓 metrics.md 的"复述问题 → 核对数字/单位/否定词清单"不一致，
  数字不可比。judge 也走 `structured_call`（独立 schema，与生成消息完全隔离）

## 必答题的答案（写死，全批一致）

- **拒答题没有生成/judge** → 在生成指标分母中**排除**并标注 N/A（合法排除）
- **生成失败 / judge 解析失败** → verdict 记 wrong、**留在分母**（合法记 0）
- 绝不"顺手跳过"——分母悄悄变小、指标虚高（metrics.md §七第 3 条）
- 每类异常的重试次数全部落盘；报告同时给 n_total 与 n_measured，
  两者差异必须能被上面的两条规则**完整解释**
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

REPORTS = PROJECT_ROOT / "eval" / "reports"

#: 指标口径版本 —— 改了 metrics.md 就改这里（可复现三要素之一）
METRICS_VERSION = "metrics@2026-11-30"

#: qa 模板固定版本（生成侧可复现要素）
PROMPT_VERSION = "v1"

VERDICTS = (
    "correct",
    "partially_correct",
    "wrong",
    "refused_correctly",
    "should_refuse_but_answered",
)


class JudgeVerdict(BaseModel):
    """judge 的输出 schema。三档 + 理由 ≤30 字。"""

    verdict: str = Field(description="必须取：correct / partially_correct / wrong")
    reason: str = Field(description="≤30 字，说清扣分点（数字错/单位错/否定词错/答非所问）")


class JudgeRefusal(BaseModel):
    """拒答题 judge 的输出 schema（独立于生成侧口径）。"""

    verdict: str = Field(description="必须取：correct / partially_correct / wrong")
    reason: str = Field(description="≤30 字")


def ndcg_at_k(anchor_ranks: list[int | None], k: int = 5) -> float:
    """二元相关度 NDCG@k（metrics.md §三：log2、rel∈{0,1}）。"""
    gold_n = len(anchor_ranks)
    if gold_n == 0:
        return 0.0
    hit_set = {r for r in anchor_ranks if r is not None and r <= k}
    dcg = sum(1.0 / math.log2(r + 1) for r in hit_set)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(gold_n, k) + 1))
    return dcg / ideal if ideal else 0.0


def run_retrieval(questions: Path, threshold: float) -> dict[str, Any]:
    """检索指标：复用 W6 定版脚手架，补 NDCG@5。"""
    out = PROJECT_ROOT / "eval" / "results" / f"retrieval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    cmd = [
        sys.executable,
        "-X",
        "utf8",
        "scripts/run_retrieval_eval.py",
        "--questions",
        str(questions),
        "--threshold",
        str(threshold),
        "--out",
        str(out),
    ]
    proc = subprocess.run(
        cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=900
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"检索评测失败（exit {proc.returncode}）\nstdout: {proc.stdout[-400:]}\nstderr: {proc.stderr[-400:]}"
        )
    payload = json.loads(out.read_text(encoding="utf-8"))

    by_type: dict[str, list[float]] = {}
    for row in payload["per_question"]:
        if row["type"] == "refusal":
            continue
        by_type.setdefault(row["type"], []).append(ndcg_at_k(row["anchor_ranks"], 5))
    all_v = [v for vs in by_type.values() for v in vs]
    payload["metrics"]["ndcg@5"] = {
        "overall": round(sum(all_v) / len(all_v), 4) if all_v else 0.0,
        **{t: round(sum(vs) / len(vs), 4) for t, vs in by_type.items() if vs},
    }
    return payload


def retrieve_contexts(item: dict, top_k: int) -> list[dict]:
    """按冻结口径检索 top_k 条（进程内缓存索引，多次调用不重建）。"""
    from retrieval_harness import load_raw_corpus  # noqa: F401 局部导入：检索 harness 在 scripts/ 下

    from app.rag.chunking import ChunkingConfig, build_strategy
    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.stores.memory import InMemoryVectorStore

    if not hasattr(retrieve_contexts, "_store"):
        chunks = build_strategy("recursive", cfg=ChunkingConfig(500, 50)).chunk(
            load_raw_corpus(), ChunkingConfig(500, 50)
        )
        embedder = CachedEmbedder(get_embedder("bge-small-zh"))
        store = InMemoryVectorStore()
        store.upsert(chunks, embedder.embed([c.text for c in chunks]))
        retrieve_contexts._store = store  # type: ignore[attr-defined]
        retrieve_contexts._embedder = embedder  # type: ignore[attr-defined]
    store = retrieve_contexts._store  # type: ignore[attr-defined]
    embedder = retrieve_contexts._embedder  # type: ignore[attr-defined]
    hits = store.search(embedder.embed_query([item["question"]])[0], top_k=top_k)
    return [{"id": i + 1, "text": h.chunk.text} for i, h in enumerate(hits)]


async def judge_answer(item: dict, answer: str) -> tuple[str, str]:
    """judge 一条答案。返回（verdict, reason）。解析失败抛异常，由调用方计数。"""
    from app.core.llm.structured import structured_call

    multi = "\n".join(f"- {a['quote']}" for a in item.get("anchors") or [])
    prompt = (
        "你是评测 judge。严格按步骤执行：\n"
        "第一步：用一句话复述这个问题要求回答的那个事实。\n"
        "第二步：对照标准答案与原文锚点，逐项核对待判回答里的数字、单位、否定词。\n"
        "第三步：结论。回答与事实一致=correct；主要事实对但有遗漏或多余=partially_correct；"
        "数字/单位/否定词错或答非所问=wrong。\n"
        f"【问题】{item['question']}\n"
        f"【标准答案】{item['gold_answer']}\n"
        + (f"【原文锚点】\n{multi}\n" if multi else "")
        + f"【待判回答】{answer[:500]}"
    )
    result = await structured_call(prompt, JudgeVerdict)
    if result.verdict not in ("correct", "partially_correct", "wrong"):
        raise ValueError(f"verdict 越界：{result.verdict}")
    return result.verdict, result.reason


async def run_generation_and_judge(items: list[dict], top_k: int) -> dict[str, Any]:
    """生成 + judge 逐条跑。失败计数不跳过（必答题的口径）。"""
    from app.core.llm.structured import structured_call
    from app.prompts import load_prompt
    from app.prompts.qa.schema import QAAnswer

    records: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for it in items:
        rec: dict[str, Any] = {
            "id": it["id"],
            "type": it["type"],
            "question": it["question"],
        }

        # ---- 拒答题：合法排除（无生成、无 judge）----
        if it["type"] == "refusal":
            rec["gen_status"] = "na_excluded"
            rec["judge_status"] = "na_excluded"
            rec["note"] = "拒答题由检索侧阈值三分类判定，不进生成指标分母"
            records.append(rec)
            continue

        # ---- 生成：检索 → qa 模板 → structured_call ----
        try:
            contexts = retrieve_contexts(it, top_k)
        except Exception as exc:
            rec["gen_status"] = "failed"
            rec["stage"] = "retrieve"
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
            rec["judge_status"] = "blocked"
            rec["verdict"] = "wrong"  # 合法记 0：留在分母
            records.append(rec)
            continue

        try:
            prompt = load_prompt(
                "qa",
                version=PROMPT_VERSION,
                question=it["question"],
                contexts=contexts,
                max_answer_chars=300,
                require_citation=True,
            )
        except Exception as exc:
            rec["gen_status"] = "failed"
            rec["stage"] = "prompt"
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
            rec["judge_status"] = "blocked"
            rec["verdict"] = "wrong"
            records.append(rec)
            continue

        try:
            qa = await structured_call(prompt, QAAnswer, max_repair=2)
        except Exception as exc:
            rec["gen_status"] = "failed"
            rec["stage"] = "generate"
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:150]}"
            rec["judge_status"] = "blocked"
            rec["verdict"] = "wrong"
            records.append(rec)
            continue

        rec["gen_status"] = "ok"
        rec["answer"] = qa.答案
        rec["citations"] = qa.引用编号
        rec["uncertain"] = qa.不确定

        # ---- judge：独立 structured_call，解析失败重试 2 轮后记 0 ----
        try:
            verdict, reason = await judge_answer(it, qa.答案)
            rec["judge_status"] = "ok"
            rec["verdict"] = verdict
            rec["judge_reason"] = reason
        except Exception as exc:
            rec["judge_status"] = "parse_failed"
            rec["verdict"] = "wrong"
            rec["judge_reason"] = f"judge 失败记 0：{type(exc).__name__}: {str(exc)[:100]}"
        records.append(rec)

    elapsed = round(time.perf_counter() - t0, 1)
    return {"records": records, "elapsed_sec": elapsed}


def summarize_generation(records: list[dict]) -> dict[str, Any]:
    """生成指标汇总。分母完整性在这里断言（无静默丢弃的机器验证）。"""
    n_total = len(records)
    excluded = [r for r in records if r["gen_status"] == "na_excluded"]
    measured = [r for r in records if r["gen_status"] != "na_excluded"]
    verdicts = Counter(r["verdict"] for r in measured)
    failures = {
        "retrieve": [r["id"] for r in records if r.get("stage") == "retrieve"],
        "prompt": [r["id"] for r in records if r.get("stage") == "prompt"],
        "generate": [r["id"] for r in records if r.get("stage") == "generate"],
        "judge_parse": [r["id"] for r in records if r["judge_status"] == "parse_failed"],
    }
    n_fail = sum(len(v) for v in failures.values())
    # 分母完整性：排除（拒答）+ 记 0（失败）+ 正常判定 == 总数
    assert len(excluded) + len(measured) == n_total, "分母出现静默丢失"
    correct = verdicts.get("correct", 0) + verdicts.get("partially_correct", 0) * 0.5
    return {
        "n_total": n_total,
        "n_excluded_na": len(excluded),
        "n_measured": len(measured),
        "verdict_counts": dict(verdicts),
        "answer_score": round(correct / len(measured), 4) if measured else None,
        "failures": failures,
        "n_failures": n_fail,
    }


def _render_md(payload: dict) -> str:
    m = payload["retrieval"]["metrics"]
    meta = payload["meta"]
    rf = m["refusal_three_way"]
    lines = [
        f"# 全链路评测报告（{meta['generated_at']}）",
        "",
        f"> commit `{meta['commit'][:10]}` ｜ 口径 `{meta['metrics_version']}` ｜ "
        f"prompt {meta['prompt_version']} ｜ 阈值 {meta['threshold']} ｜ k={meta['top_k']}",
        "",
        "## 检索指标",
        "",
    ]
    for k in (1, 5):
        r = m[f"recall@{k}"]
        lines.append(f"- Recall@{k}: **{r['rate'] * 100:.1f}%**（{r['hits']}/{r['total']}）")
    lines.append(f"- MRR: {m['mrr']}")
    n = m["ndcg@5"]
    lines.append(
        f"- NDCG@5: {n['overall']}（fact {n.get('fact', '—')} / multi_hop {n.get('multi_hop', '—')}）"
    )
    lines.append(
        f"- 拒答三分类：正确拒 **{rf['correct_refusals']}/{rf['refusal_total']}**，"
        f"误拒 {rf['false_refusals']}，该拒不拒（幻觉） {rf['false_accepts']}"
    )

    g = payload.get("generation")
    if g is None:
        lines += ["", "## 生成 + judge", "", "（--skip-generation：本次只跑检索）"]
    else:
        s = g["summary"]
        lines += [
            "",
            "## 生成 + judge",
            "",
            f"- 分母完整性：n_total={s['n_total']} = 排除(拒答) {s['n_excluded_na']} + 判定 {s['n_measured']} "
            f"（失败 {s['n_failures']} 条**记 0 留在分母**）",
            f"- verdict 分布：{json.dumps(s['verdict_counts'], ensure_ascii=False)}",
            f"- 答案得分：**{s['answer_score']}**（correct=1 / partially=0.5 / wrong=0）",
        ]
        if s["n_failures"]:
            lines.append(f"- ⚠ 失败明细：{json.dumps(s['failures'], ensure_ascii=False)}")
        else:
            lines.append("- 无失败、无静默丢弃")
        lines += [
            "",
            f"（生成+judge 耗时 {g['elapsed_sec']} s；逐条记录见同名 json 的 generation.records）",
        ]
    lines += [
        "",
        "## 复现",
        "",
        "```bash",
        f"git checkout {meta['commit'][:10]}",
        f"python -X utf8 scripts/run_eval.py --questions {meta['questions']}",
        "```",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="W7 一条命令出全部指标")
    parser.add_argument(
        "--questions", type=Path, default=PROJECT_ROOT / "eval" / "questions-slice20.jsonl"
    )
    parser.add_argument("--threshold", type=float, default=0.575)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--skip-generation", action="store_true", help="只跑检索（零成本调试）")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    started = time.strftime("%Y-%m-%d %H:%M:%S")

    retrieval = run_retrieval(args.questions.resolve(), args.threshold)
    m = retrieval["metrics"]
    metrics = {
        "recall@1": m["recall"]["recall@1"],
        "recall@5": m["recall"]["recall@5"],
        "mrr": m["mrr"],
        "ndcg@5": m["ndcg@5"],
        "refusal_three_way": {
            "correct_refusals": m["refusal"]["refused_correctly"],
            "refusal_total": m["refusal"]["refusal_total"],
            "false_refusals": len(m["refusal"]["false_refusal"]),
            "false_accepts": len(m["refusal"]["false_accept"]),
        },
    }
    print("检索指标：")
    print(
        f"  Recall@1 {metrics['recall@1']['rate'] * 100:.1f}%  Recall@5 {metrics['recall@5']['rate'] * 100:.1f}%"
    )
    print(f"  MRR {metrics['mrr']}  NDCG@5 {metrics['ndcg@5']['overall']}")
    print(f"  拒答三分类: {json.dumps(metrics['refusal_three_way'], ensure_ascii=False)}")

    gen_payload: dict[str, Any] | None = None
    if not args.skip_generation:
        items = [
            json.loads(line)
            for line in args.questions.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(f"\n生成 + judge（{len(items)} 条，拒答题排除）…")
        gen_payload = asyncio.run(run_generation_and_judge(items, args.top_k))
        summary = summarize_generation(gen_payload["records"])
        gen_payload["summary"] = summary
        print(
            f"  分母: n_total={summary['n_total']} = 排除 {summary['n_excluded_na']} + 判定 {summary['n_measured']}"
        )
        print(f"  verdict: {json.dumps(summary['verdict_counts'], ensure_ascii=False)}")
        print(
            f"  答案得分: {summary['answer_score']}  失败: {summary['n_failures']} 条（全部记 0 留分母）"
        )

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
    ).stdout.strip()
    n_q = sum(1 for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip())
    payload = {
        "meta": {
            "generated_at": started,
            "questions": str(args.questions.relative_to(PROJECT_ROOT)),
            "n_questions": n_q,
            "commit": commit,
            "metrics_version": METRICS_VERSION,
            "prompt_version": PROMPT_VERSION,
            "threshold": args.threshold,
            "top_k": args.top_k,
        },
        "retrieval": {"metrics": metrics, "source_meta": retrieval["meta"]},
        "generation": gen_payload,
    }

    if not args.no_write:
        REPORTS.mkdir(parents=True, exist_ok=True)
        stem = time.strftime("%Y%m%d-%H%M%S")
        (REPORTS / f"{stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (REPORTS / f"{stem}.md").write_text(_render_md(payload), encoding="utf-8")
        print(f"\n落盘：eval/reports/{stem}.json + .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
