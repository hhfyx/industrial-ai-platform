"""阈值精标：用 50 条集的命中/未命中分数分布重新定阈值。

## 与周四那次的区别（为什么叫"精标"）

周四用的是 **20 条有答案 + 5 条无答案**，结论是"两组重叠、无可用阈值"。
今天是 50 条：**30 事实 + 10 多跳（可答 40）+ 10 拒答**。

样本从 25 条到 50 条，拒答从 5 条到 10 条。这让"错误数"的粒度从 20pp
降到 10pp（拒答侧），可以做一次真正的取舍。

## 判据：不是"错误最少"，是"可接受代价下的拒答率"

周四踩过的坑：单看"合计错"会选出牺牲大量真答案的阈值。
所以这里报**四个数**，让人自己权衡：

    refused_correctly   该拒且拒了
    false_accept        该拒却答了（模型的幻觉风险）
    false_refusal       该答却拒了（用户的服务失败）
    recall_miss         没检索到（**不是阈值问题**，单独排除）

**`recall_miss` 必须排除**：那 20 多条题无论阈值多少都答不对，
把它们算进"误拒"会让所有阈值看起来一样差。

## 两种"最优"的定义

| 定义 | 选什么 | 什么时候用 |
| --- | --- | --- |
| 错误最少 | `false_accept + false_refusal` 最小 | 只想平衡两类错误 |
| 代价受限 | `false_refusal <= 上限` 里 `false_accept` 最小 | 线上不能拒真答案 |

两个都算出来并列出。**最终采用哪个是产品决定，不是数据决定** ——
但数据必须把两个候选都摆出来，否则那个决定是盲的。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_retrieval_eval as rre  # noqa: E402
import strategy_compare as sc  # noqa: E402

DEFAULT_OUT: Path = PROJECT_ROOT / "scripts" / "threshold_fine.json"

# 候选阈值：0.30~0.75，步长 0.025。
# **比周四更密**：样本翻倍后，相邻阈值之间的差别才有意义。
CANDIDATES: tuple[float, ...] = tuple(round(0.30 + 0.025 * i, 3) for i in range(19))

# 代价上限：允许误拒多少条真答案。
# 1 条 = 2.5pp（40 条可答题）。取 2 条（5%），超过它线上体感明显。
DEFAULT_MAX_FALSE_REFUSAL: int = 2


def collect_scores(
    *,
    strategy: str,
    chunk_chars: int,
    embedder: Any,  # noqa: ANN401
    docs: list[Any],
    items: list[Any],
    top_k: int,
) -> list[dict[str, Any]]:
    """跑一遍，收集**每条题的 top1 分数 + 是否命中**。

    与 `strategy_compare.run_one` 的区别：那边只汇总指标，
    这里要逐条的原始分数 —— 阈值扫描必须在细粒度上重算，
    而不是拿汇总数反推。
    """
    from app.rag.stores.memory import InMemoryVectorStore

    chunks = sc.build_chunks(
        docs=docs, strategy=strategy, chunk_chars=chunk_chars, embedder=embedder
    )
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))

    probe = [
        rre.GoldItem(
            id=item.id,
            type=item.type,
            question=item.question,
            anchors=item.anchors,
            gold_answer=item.gold_answer,
            notes=item.notes,
        )
        for item in items
    ]
    rre.resolve_all(probe, chunks)

    rows: list[dict[str, Any]] = []
    for item in probe:
        vector = embedder.embed_query([item.question])[0]
        hits = store.search(vector, top_k=top_k)
        outcome = rre.evaluate_query(item=item, hits=hits, k=top_k)
        rows.append(
            {
                "id": item.id,
                "type": item.type,
                "is_refusal": item.is_refusal,
                "hit": outcome.hit,
                "top_score": round(outcome.top_score, 4) if outcome.top_score is not None else None,
            }
        )
    return rows


def sweep_thresholds(
    rows: list[dict[str, Any]], *, candidates: tuple[float, ...]
) -> list[dict[str, Any]]:
    """对每个候选阈值算四个数。**`recall_miss` 单列，不混进误拒。**"""
    answerable = [row for row in rows if not row["is_refusal"]]
    refusals = [row for row in rows if row["is_refusal"]]

    table: list[dict[str, Any]] = []
    for threshold in candidates:
        # 可答题：命中 **且** 分数达阈值才算"答对"
        answered = [
            row for row in answerable if row["hit"] and (row["top_score"] or 0) >= threshold
        ]
        # 误拒 = 检索到了、但分数被阈值挡掉。
        # **检索没到的那批不算误拒** —— 那是召回问题，阈值改不了它。
        false_refusal = [
            row for row in answerable if row["hit"] and (row["top_score"] or 0) < threshold
        ]
        # 该拒且拒了
        refused_correctly = [row for row in refusals if (row["top_score"] or 0) < threshold]
        false_accept = [row for row in refusals if (row["top_score"] or 0) >= threshold]
        recall_miss = [row for row in answerable if not row["hit"]]

        table.append(
            {
                "threshold": threshold,
                "answered": len(answered),
                "refused_correctly": len(refused_correctly),
                "refusal_total": len(refusals),
                "refusal_accuracy": round(len(refused_correctly) / len(refusals), 4)
                if refusals
                else 0.0,
                "false_accept": len(false_accept),
                "false_refusal": len(false_refusal),
                "recall_miss": len(recall_miss),
                "errors": len(false_accept) + len(false_refusal),
                # 端到端答对 = 命中且过阈值 + 正确拒答
                "end_to_end_correct": len(answered) + len(refused_correctly),
                "end_to_end_total": len(rows),
            }
        )
    return table


def pick(table: list[dict[str, Any]], *, max_false_refusal: int) -> dict[str, Any]:
    """两种"最优"。**都返回，让人自己选。**"""
    fewest_errors = min(table, key=lambda row: (row["errors"], -row["threshold"]))
    # 代价受限：误拒不超过上限的里面，误放最少（同分取阈值高者，更保守）
    capped = [row for row in table if row["false_refusal"] <= max_false_refusal]
    cost_capped = (
        min(capped, key=lambda row: (row["false_accept"], -row["threshold"])) if capped else None
    )
    best_e2e = max(table, key=lambda row: (row["end_to_end_correct"], row["threshold"]))
    return {
        "fewest_errors": fewest_errors,
        "cost_capped": cost_capped,
        "best_end_to_end": best_e2e,
        "max_false_refusal": max_false_refusal,
    }


def score_ranges(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """两组分数的分布（回答"有没有分界点"）。"""
    answerable = [row["top_score"] for row in rows if not row["is_refusal"] and row["top_score"]]
    refusals = [row["top_score"] for row in rows if row["is_refusal"] and row["top_score"]]
    if not answerable or not refusals:
        return {"separable": False, "reason": "样本不足"}
    pos_min, pos_max = min(answerable), max(answerable)
    neg_min, neg_max = min(refusals), max(refusals)
    return {
        "separable": neg_max < pos_min,
        "answerable_min": round(pos_min, 4),
        "answerable_max": round(pos_max, 4),
        "refusal_min": round(neg_min, 4),
        "refusal_max": round(neg_max, 4),
        "gap": round(pos_min - neg_max, 4) if neg_max < pos_min else None,
        "overlap": (round(max(pos_min, neg_min), 4), round(min(pos_max, neg_max), 4))
        if neg_max >= pos_min
        else None,
    }


def render(
    table: list[dict[str, Any]],
    picks: dict[str, Any],
    ranges: dict[str, Any],
    *,
    strategy: str,
    chunk_chars: int,
    baseline: float,
) -> str:
    lines: list[str] = []
    lines.append(
        f"阈值精标（strategy={strategy} chunk={chunk_chars}，"
        f"40 可答 + 10 拒答；基线阈值 {baseline}）"
    )
    lines.append("")
    lines.append("  分数分布")
    if ranges.get("separable"):
        lines.append(
            f"    ✓ 完全可分：拒答最高 {ranges['refusal_max']} < "
            f"可答最低 {ranges['answerable_min']}，空隙 {ranges['gap']}"
        )
    else:
        lines.append(
            f"    可答 [{ranges['answerable_min']}, {ranges['answerable_max']}]  "
            f"拒答 [{ranges['refusal_min']}, {ranges['refusal_max']}]"
        )
        if ranges.get("overlap"):
            lines.append(f"    ✗ 重叠区间 {ranges['overlap']}")

    lines.append("")
    lines.append("  阈值扫描（误拒只算「检索到了但被阈值挡掉」的；召回没到的不算）")
    lines.append(
        f"    {'阈值':>6} {'正确拒答':>9} {'拒答率':>7} {'误放':>5} {'误拒':>5} "
        f"{'召回未命中':>10} {'端到端对':>9}"
    )
    for row in table:
        mark = "  ← 基线" if abs(row["threshold"] - baseline) < 1e-9 else ""
        lines.append(
            f"    {row['threshold']:>6.3f} "
            f"{row['refused_correctly']:>4}/{row['refusal_total']:<4} "
            f"{row['refusal_accuracy'] * 100:>6.1f}% {row['false_accept']:>5} "
            f"{row['false_refusal']:>5} {row['recall_miss']:>10} "
            f"{row['end_to_end_correct']:>4}/{row['end_to_end_total']}{mark}"
        )

    lines.append("")
    lines.append("  两种「最优」")
    fewest = picks["fewest_errors"]
    lines.append(
        f"    ① 错误最少：t={fewest['threshold']}  "
        f"误放 {fewest['false_accept']} + 误拒 {fewest['false_refusal']} = {fewest['errors']}"
    )
    capped = picks["cost_capped"]
    if capped:
        lines.append(
            f"    ② 误拒 ≤{picks['max_false_refusal']} 时误放最少：t={capped['threshold']}  "
            f"误放 {capped['false_accept']}，误拒 {capped['false_refusal']}，"
            f"拒答 {capped['refused_correctly']}/{capped['refusal_total']}"
        )
    else:
        lines.append(f"    ② 没有任何阈值的误拒 ≤{picks['max_false_refusal']}")
    best_e2e = picks["best_end_to_end"]
    lines.append(
        f"    ③ 端到端答对最多：t={best_e2e['threshold']}  "
        f"{best_e2e['end_to_end_correct']}/{best_e2e['end_to_end_total']}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阈值精标（50 条集）")
    parser.add_argument("--strategy", default=None, help="默认读 RagConfig")
    parser.add_argument("--chunk-chars", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-false-refusal", type=int, default=DEFAULT_MAX_FALSE_REFUSAL)
    parser.add_argument("--embedder", choices=["bge", "hashing"], default="bge")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.settings import get_rag_config

    cfg = get_rag_config()
    strategy = args.strategy or cfg.strategy
    chunk_chars = args.chunk_chars if args.chunk_chars is not None else cfg.chunk_chars

    if args.embedder == "hashing":
        print("⚠ hashing 下分数无可分性", file=sys.stderr)
        embedder: Any = get_embedder("hashing")
    else:
        embedder = CachedEmbedder(get_embedder("bge-small-zh"))

    docs = sc.load_raw_corpus()
    items = rre.load_questions()
    print(f"语料 {len(docs)} 段，题目 {len(items)} 条", file=sys.stderr)

    rows = collect_scores(
        strategy=strategy,
        chunk_chars=chunk_chars,
        embedder=embedder,
        docs=docs,
        items=items,
        top_k=args.top_k,
    )
    table = sweep_thresholds(rows, candidates=CANDIDATES)
    picks = pick(table, max_false_refusal=args.max_false_refusal)
    ranges = score_ranges(rows)

    print()
    print(
        render(
            table,
            picks,
            ranges,
            strategy=strategy,
            chunk_chars=chunk_chars,
            baseline=cfg.score_threshold,
        )
    )

    payload = {
        "meta": {
            "strategy": strategy,
            "chunk_chars": chunk_chars,
            "top_k": args.top_k,
            "embedder": args.embedder,
            "answerable": sum(1 for row in rows if not row["is_refusal"]),
            "refusals": sum(1 for row in rows if row["is_refusal"]),
            "baseline_threshold": cfg.score_threshold,
            "max_false_refusal": args.max_false_refusal,
        },
        "scores": rows,
        "sweep": table,
        "picks": picks,
        "ranges": ranges,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘：{args.out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
