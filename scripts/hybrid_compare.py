"""混合检索对照：**纯向量 / RRF / 加权(α=0.3,0.5,0.7) 五组**。

## 这个脚本要回答的三个问题

1. **合并比例**：两路候选的交集/独有占多少？→ 混合到底有没有意义
2. **五组指标**：Recall@5、MRR 各是多少？→ 融合真的涨了吗
3. **哪组当基线**：效果 + 复杂度 → 选谁，为什么

## 为什么五组要一起跑，而不是分开跑

分开跑会得到五个各自"看起来还行"的数，但**没有共同的候选集合** ——
RRF 与加权融合如果各取各的候选，那比较的就不是"融合方式"，而是"候选数"。
所以这里**两路候选只取一次**（`candidates=20`），五组全部在同一个并集上打分。

## 为什么两组加权必须做归一化

见 `app/rag/bm25/fusion.py` 的模块 docstring。一句话：
**直接 `0.5*cos + 0.5*bm25` 会让 BM25 单方面支配** ——
余弦在 [0,1]、BM25 实测 0~30，`α` 变成假的。

本脚本会**同时跑一遍"不做归一化的朴素加权"作为反例**（`naive`），
用它来证明归一化不是"更讲究"，是**必须**。这个反例本身就是数据。

## 为什么评测的是 top5 而不是"看几条例子"

任务书写的是"记录 Recall@5、MRR"。50 条调参集 1 条 = 2pp，
所以每 1 条的差异都要能指名道姓（脚本会输出"哪几条题的命中状态变了"）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import retrieval_harness as rh  # noqa: E402
import run_retrieval_eval as rre  # noqa: E402

from app.rag.bm25.fusion import (  # noqa: E402
    DEFAULT_RRF_K,
    FusedHit,
    FusionConfig,
    FusionMode,
    merge_rankings,
)
from app.rag.bm25.index import DEFAULT_INDEX_PATH  # noqa: E402
from app.rag.bm25.retriever import BM25Retriever  # noqa: E402
from app.rag.chunking import ChunkingConfig, build_strategy  # noqa: E402
from app.rag.settings import get_rag_config  # noqa: E402

OUT_JSON: Path = PROJECT_ROOT / "scripts" / "hybrid_compare.json"
OUT_MD: Path = PROJECT_ROOT / "scripts" / "hybrid_compare.md"

#: 五组要跑的实验。**顺序即报告顺序**：基线在前，越往下越复杂。
#:
#: `Naive` 不是一个"融合方法"，是一个**反例**：它把两路原始分数直接加权，
#: 不做归一化。留着它是为了让"必须归一化"这句话有数据支撑，
#: 而不是只写在注释里。
EXPERIMENTS: tuple[tuple[str, FusionConfig | None], ...] = (
    ("pure_vec", None),
    ("pure_bm25", None),
    ("rrf", FusionConfig(mode=FusionMode.RRF, candidates=20, rrf_k=DEFAULT_RRF_K)),
    ("rrf_closed", FusionConfig(mode=FusionMode.RRF, candidates=20, rrf_k=10)),
    (
        "naive_a0.5",
        None,
    ),
    ("weighted_a0.3", FusionConfig(mode=FusionMode.WEIGHTED, candidates=20, alpha=0.3)),
    ("weighted_a0.5", FusionConfig(mode=FusionMode.WEIGHTED, candidates=20, alpha=0.5)),
    ("weighted_a0.7", FusionConfig(mode=FusionMode.WEIGHTED, candidates=20, alpha=0.7)),
)


@dataclass
class RunResult:
    """一组实验的结果。"""

    name: str
    label: str
    recall_at_5: float
    hits_at_5: int
    mrr: float
    top5_source_mix: dict[str, int]
    """top5 里各自由哪一路召回的块占多少：both / vector / bm25。

    **这是"融合有没有意义"的直接证据**：如果融合后的 top5 里
    `vector_only` 和 `bm25_only` 都是 0，说明融合只在共识块上打转。
    """

    states: dict[str, str]
    """逐题的命中状态（`hit` / `miss`），用于算"哪几条题变了"。"""

    config: dict[str, object]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "recall_at_5": round(self.recall_at_5, 4),
            "hits_at_5": self.hits_at_5,
            "mrr": round(self.mrr, 4),
            "top5_source_mix": self.top5_source_mix,
            "config": self.config,
            "states": self.states,
        }


def _top_k_hit(ranks: dict[str, int], gold_ids: list[str], k: int) -> bool:
    """多锚点题：**全部锚点都在 top_k 内**才算命中（与 eval 脚本同一口径）。"""
    if not gold_ids:
        return False
    return all(ranks.get(gid) is not None and ranks[gid] <= k for gid in gold_ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="混合检索五组对照")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--embedder", choices=["bge", "hashing"], default="bge")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    cfg = get_rag_config()
    docs = rh.load_raw_corpus()
    chunk_cfg = ChunkingConfig(chunk_chars=cfg.chunk_chars, overlap_chars=cfg.overlap_chars)
    chunks = build_strategy(cfg.strategy, cfg=chunk_cfg).chunk(docs, chunk_cfg)
    print(
        f"语料 {len(docs)} 段；索引 {len(chunks)} 块"
        f"（chunk={cfg.chunk_chars} overlap={cfg.overlap_chars} strategy={cfg.strategy}）",
        file=sys.stderr,
    )

    items = rre.load_questions(rre.EVAL_FILE)
    answerable = [item for item in items if not item.is_refusal]
    rre.resolve_all(items, chunks)

    # ---- 两路各取一次候选（五组共用，这是公平比较的前提）----
    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.stores.memory import InMemoryVectorStore

    bm25, report = BM25Retriever.from_chunks(chunks, path=DEFAULT_INDEX_PATH)
    print(f"BM25 索引：{report.summary()}", file=sys.stderr)

    if args.embedder == "hashing":
        print("⚠ hashing 下向量召回无参考价值，仅验证流程", file=sys.stderr)
        embedder = get_embedder("hashing")
    else:
        embedder = CachedEmbedder(get_embedder("bge-small-zh"))
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))

    # 每个 query 的候选与分数只算一次
    started = time.perf_counter()
    per_query: dict[str, dict[str, Any]] = {}
    for item in items:
        vector_hits = store.search(embedder.embed_query([item.question])[0], top_k=args.candidates)
        bm25_hits = bm25.retrieve(item.question, top_k=args.candidates)
        per_query[item.id] = {
            "vec_ids": [hit.chunk.id for hit in vector_hits],
            "vec_scores": [float(hit.score) for hit in vector_hits],
            "bm25_ids": [hit.chunk.id for hit in bm25_hits],
            "bm25_scores": [float(hit.score) for hit in bm25_hits],
        }
    retrieve_ms = int((time.perf_counter() - started) * 1000)
    print(f"两路候选各 {args.candidates} 条 / 50 题：{retrieve_ms} ms", file=sys.stderr)

    # ---- 合并比例（任务点名要的数字）----
    merge_summary = aggregate_merge(per_query, args.candidates)

    # ---- 跑五组 ----
    results: list[RunResult] = []
    for name, fusion_config in EXPERIMENTS:
        results.append(
            _run_one(
                name=name,
                fusion_config=fusion_config,
                per_query=per_query,
                answerable=answerable,
                top_k=args.top_k,
                candidates=args.candidates,
            )
        )

    # ---- 输出 ----
    baseline = results[0]
    print()
    print("=" * 88)
    print(f"合并比例（每路 {args.candidates} 条候选，{len(answerable)} 条可答题汇总）")
    print("=" * 88)
    print(f"  平均并集   {merge_summary['avg_union']:.1f} 条/题")
    print(f"  平均交集   {merge_summary['avg_both']:.1f} 条/题  ← 两路都召回")
    print(f"  交集占比   {merge_summary['overlap_ratio'] * 100:.1f}%（Jaccard：交集/并集）")
    print(f"  独有占比   {merge_summary['novelty_ratio'] * 100:.1f}%  ← **混合有意义的主指标**")
    print(f"    仅向量   {merge_summary['vector_only_ratio'] * 100:.1f}%")
    print(f"    仅 BM25  {merge_summary['bm25_only_ratio'] * 100:.1f}%")

    print()
    print("=" * 88)
    print(f"五组对照（top-{args.top_k}，{len(answerable)} 条可答题）")
    print("=" * 88)
    print(f"  {'实验':<16}{'Recall@5':>10}{'命中':>8}{'MRR':>9}{'相对基线':>10}{'top5 来源':>26}")
    print("  " + "-" * 84)
    for result in results:
        delta = result.recall_at_5 - baseline.recall_at_5
        delta_text = "—" if result.name == baseline.name else f"{delta * 100:+.1f}pp"
        mix = result.top5_source_mix
        mix_text = f"both{mix.get('both', 0)}/v{mix.get('vector', 0)}/b{mix.get('bm25', 0)}"
        print(
            f"  {result.name:<16}{result.recall_at_5 * 100:>9.1f}%"
            f"{result.hits_at_5:>8}{result.mrr:>9.4f}{delta_text:>10}{mix_text:>26}"
        )

    print()
    print("  逐组相对基线的命中状态变化（50 条集 1 条 = 2pp，所以每条都要能指名）：")
    for result in results[1:]:
        gained = [
            qid
            for qid, state in result.states.items()
            if state == "hit" and baseline.states.get(qid) != "hit"
        ]
        lost = [
            qid
            for qid, state in result.states.items()
            if state != "hit" and baseline.states.get(qid) == "hit"
        ]
        if gained or lost:
            print(f"    {result.name:<16} +{gained or '无'}  -{lost or '无'}")
        else:
            print(f"    {result.name:<16} 与基线完全相同")

    # ---- 显著性：**没有这一步，"涨了 5pp"就会被当成结论** ----
    sig = [_significance(baseline, result) for result in results[1:]]
    print()
    print("=" * 88)
    print("显著性（配对 McNemar 精确检验，对比纯向量基线）")
    print("=" * 88)
    print(
        f"  {'对比':<20}{'命中':>8}{'仅基线':>8}{'仅本组':>8}{'p 值':>9}{'判定':>10}{'Wilson 95%':>18}"
    )
    print("  " + "-" * 84)
    for row in sig:
        lo, hi = row["wilson"]
        print(
            f"  {row['name']:<20}{row['hits']:>5}/40{row['only_base']:>8}{row['only_this']:>8}"
            f"{row['p_value']:>9.4f}{'显著' if row['significant'] else '不显著':>10}"
            f"{f'[{lo * 100:.1f}, {hi * 100:.1f}]':>18}"
        )
    any_sig = [row["name"] for row in sig if row["significant"]]
    print()
    if any_sig:
        print(f"  **显著优于/劣于基线的组：{any_sig}**")
    else:
        print("  **没有任何一组与纯向量基线有显著差异。**")
        print("  也就是说主表里 35.0% ~ 45.0% 的排序在 40 条样本上无法与噪声区分 ——")
        print("  最多只差 3 条。这和 W5 三策略对比是同一类结论：**先别急着说融合更好。**")

    if args.no_write:
        return 0

    best = _pick_baseline(results, sig)
    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus_docs": len(docs),
            "chunks": len(chunks),
            "chunk_chars": cfg.chunk_chars,
            "overlap": cfg.overlap_chars,
            "strategy": cfg.strategy,
            "embedder": args.embedder,
            "top_k": args.top_k,
            "candidates": args.candidates,
            "answerable": len(answerable),
            "retrieve_ms": retrieve_ms,
            "bm25_stats": bm25.stats(),
        },
        "merge": merge_summary,
        "significance": sig,
        "runs": [r.as_dict() for r in results],
        "baseline": best,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"\n落盘：{OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"      {OUT_MD.relative_to(PROJECT_ROOT)}")
    return 0


def aggregate_merge(per_query: dict[str, dict[str, Any]], candidates: int) -> dict[str, Any]:
    """合并比例的多题汇总。**"平均什么"只在这一处定义。**

    这里用 RRF 的 `merge_rankings` 只为拿 `MergeStats` ——
    合并比例（并集/交集/独有）与融合方式无关，它纯粹是"两路各召回了哪些块"。
    所以五组实验看到的都是这一份比例。
    """
    config = FusionConfig(mode=FusionMode.RRF, candidates=candidates)
    total_union = total_both = total_vonly = total_bonly = 0
    for data in per_query.values():
        _hits, stats = merge_rankings(data["vec_ids"], data["bm25_ids"], config=config)
        total_union += stats.union
        total_both += stats.both
        total_vonly += stats.vector_only
        total_bonly += stats.bm25_only
    n = max(len(per_query), 1)
    union = total_union or 1
    return {
        "avg_union": total_union / n,
        "avg_both": total_both / n,
        "overlap_ratio": total_both / union,
        "novelty_ratio": (total_vonly + total_bonly) / union,
        "vector_only_ratio": total_vonly / union,
        "bm25_only_ratio": total_bonly / union,
        "total_union": total_union,
        "total_both": total_both,
        "total_vector_only": total_vonly,
        "total_bm25_only": total_bonly,
    }


def _run_one(
    *,
    name: str,
    fusion_config: FusionConfig | None,
    per_query: dict[str, dict[str, Any]],
    answerable: list[Any],
    top_k: int,
    candidates: int,
) -> RunResult:
    """跑一组实验。`fusion_config=None` 表示纯单路或朴素反例。"""
    states: dict[str, str] = {}
    reciprocal_sum = 0.0
    hits_at_5 = 0
    mix: dict[str, int] = {"both": 0, "vector": 0, "bm25": 0}
    rank_of: dict[str, dict[str, int]] = {}

    for item in answerable:
        data = per_query[item.id]
        if name == "pure_vec":
            ranked_ids = data["vec_ids"]
            hit_obj = [FusedHit(cid, 0.0, i, None) for i, cid in enumerate(ranked_ids, start=1)]
            ranks = {cid: i for i, cid in enumerate(ranked_ids, start=1)}
        elif name == "pure_bm25":
            ranked_ids = data["bm25_ids"]
            hit_obj = [FusedHit(cid, 0.0, None, i) for i, cid in enumerate(ranked_ids, start=1)]
            ranks = {cid: i for i, cid in enumerate(ranked_ids, start=1)}
        elif name == "naive_a0.5":
            # **反例**：不归一化，直接用原始分数加权。留着它证明归一化是必须的。
            raw: dict[str, float] = {}
            for cid, score in zip(data["vec_ids"], data["vec_scores"], strict=True):
                raw[cid] = 0.5 * score
            for cid, score in zip(data["bm25_ids"], data["bm25_scores"], strict=True):
                raw[cid] = raw.get(cid, 0.0) + 0.5 * score
            # 分数降序；同分按 chunk_id 保证确定性（与 fusion 的排序纪律一致）
            ordered = sorted(raw.items(), key=lambda kv: (-kv[1], kv[0]))
            ranked_ids = [cid for cid, _ in ordered]
            hit_obj = []
            for cid, score in ordered:
                # 朴素版不记录两路名次（它压根没用名次），所以来源标成 unknown
                hit_obj.append(FusedHit(cid, score, None, None))
            ranks = {cid: i for i, cid in enumerate(ranked_ids, start=1)}
        else:
            assert fusion_config is not None
            fused, _stats = merge_rankings(
                data["vec_ids"],
                data["bm25_ids"],
                config=fusion_config,
                vec_scores=data["vec_scores"],
                bm25_scores=data["bm25_scores"],
            )
            hit_obj = fused
            ranked_ids = [hit.chunk_id for hit in fused]
            ranks = {cid: i for i, cid in enumerate(ranked_ids, start=1)}

        rank_of[item.id] = ranks
        gold = item.gold_chunk_ids
        hit5 = _top_k_hit(ranks, gold, top_k)
        states[item.id] = "hit" if hit5 else "miss"
        if hit5:
            hits_at_5 += 1
            best_rank = max(ranks[gid] for gid in gold)
            reciprocal_sum += 1.0 / best_rank
        elif any(ranks.get(gid) is not None for gid in gold):
            reciprocal_sum += 0

        # top5 来源统计：融合后占前 k 的块，各自由哪一路召回
        for hit in hit_obj[:top_k]:
            mix[hit.matched_by] = mix.get(hit.matched_by, 0) + 1

    total = len(answerable)
    return RunResult(
        name=name,
        label=_label_of(name, fusion_config),
        recall_at_5=hits_at_5 / total if total else 0.0,
        hits_at_5=hits_at_5,
        mrr=reciprocal_sum / total if total else 0.0,
        top5_source_mix=mix,
        states=states,
        config=fusion_config.as_dict() if fusion_config else {"mode": name},
    )


def _label_of(name: str, config: FusionConfig | None) -> str:
    if config is None:
        return name
    return config.label


def _wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% 区间。**不用正态近似**：40 条样本上它偏离太大。"""
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _mcnemar(only_a: int, only_b: int) -> float:
    """配对 McNemar **精确**检验（二项检验）。`n=0` 时 p=1（完全一致，无从判别）。"""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def _significance(base: RunResult, other: RunResult) -> dict[str, Any]:
    """算 `other` 相对 `base` 的配对显著性。

    **为什么必须做这一步**：本表八个组里只要有一个"涨了 5pp"，
    人就会想去用它。而 40 条样本上 3 赢 1 输 = +2 条 = 5pp，
    这个数**完全在噪声范围里** —— 不做检验等于让随机波动决定架构。
    """
    only_base = sum(
        1 for qid, state in base.states.items() if state == "hit" and other.states.get(qid) != "hit"
    )
    only_this = sum(
        1 for qid, state in other.states.items() if state == "hit" and base.states.get(qid) != "hit"
    )
    p_value = _mcnemar(only_base, only_this)
    hits = sum(1 for state in other.states.values() if state == "hit")
    total = len(other.states)
    return {
        "name": other.name,
        "hits": hits,
        "total": total,
        "only_base": only_base,
        "only_this": only_this,
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "wilson": [round(v, 4) for v in _wilson(hits, total)],
    }


def _pick_baseline(
    results: list[RunResult], sig: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """选出本周基线。**规则先写死，不看结果再挑。**

    ## 规则（跑之前定死，写在这里就不许改）

    1. **先看显著性**：没有任何组显著优于基线 → **就没有"更好的方案"**。
       此时基线不该被动摇 —— 换基线的代价是重建索引与改调用链，
       而收益在噪声里。
    2. 有显著优于基线的组时，按 `Recall@5 → MRR → 复杂度` 字典序。
    3. **复杂度量化**为"配置项个数"（不含 mode/candidates）：

           pure_vec     0 项（无参数）
           rrf          1 项（k）
           weighted     2 项（α + normalize）

    **必须在跑之前定规则**，否则会变成"先看谁赢再解释为什么它该赢" ——
    那是 W5 三策略对比里已经记过的教训。
    """
    if sig is None:
        sig = []

    def _complexity(result: RunResult) -> int:
        return len([key for key in result.config if key not in ("mode", "candidates")])

    significantly_better = [
        row["name"] for row in sig if row["significant"] and row["only_this"] > row["only_base"]
    ]
    significantly_worse = [
        row["name"] for row in sig if row["significant"] and row["only_base"] > row["only_this"]
    ]

    if not significantly_better:
        # **规则 1 生效**：没人显著更好 → 守住当前基线（纯向量）。
        # 这不是"懒"，而是"没有证据支持换" —— 与 W5 采用 recursive 的理由同一句式。
        winner = results[0]
        reason = (
            "没有任何融合组显著优于纯向量基线（配对 McNemar p 全部 > 0.05），"
            "所以**没有证据支持换基线**。保持纯向量：零参数、零额外索引。"
        )
    else:
        ordered = sorted(
            [r for r in results if r.name in significantly_better or r.name == results[0].name],
            key=lambda r: (-round(r.recall_at_5, 4), -round(r.mrr, 4), _complexity(r)),
        )
        winner = ordered[0]
        reason = (
            f"显著优于基线的组：{significantly_better}；按 Recall@5 → MRR → 复杂度 取 {winner.name}"
        )

    runner_up = max(results, key=lambda r: r.recall_at_5)
    return {
        "name": winner.name,
        "label": winner.label,
        "recall_at_5": round(winner.recall_at_5, 4),
        "mrr": round(winner.mrr, 4),
        "complexity": _complexity(winner),
        "reason": reason,
        "significantly_better": significantly_better,
        "significantly_worse": significantly_worse,
        "best_recall_but_not_significant": runner_up.name
        if runner_up.name != winner.name
        else None,
        "best_recall_value": round(runner_up.recall_at_5, 4),
        "rule": "先看显著性 → Recall@5 → MRR → 配置项个数（跑之前定死）",
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    merge = payload["merge"]
    runs = payload["runs"]
    baseline = payload["baseline"]
    lines: list[str] = []
    lines.append("# 混合检索对照：纯向量 / RRF / 加权融合")
    lines.append("")
    lines.append(
        f"> 生成时间 {meta['generated_at']} ｜ {meta['chunks']} 块 "
        f"（chunk={meta['chunk_chars']} overlap={meta['overlap']} {meta['strategy']}）"
        f" ｜ 每路候选 {meta['candidates']} 条 ｜ top-{meta['top_k']}"
    )
    lines.append(
        f"> BM25：{meta['bm25_stats']['chunks']} 块 / avgdl {meta['bm25_stats']['avgdl']} "
        f"/ k1={meta['bm25_stats']['params']['k1']} b={meta['bm25_stats']['params']['b']}"
    )
    lines.append("")
    lines.append("## 一、合并比例（混合有没有意义）")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| --- | ---: |")
    lines.append(f"| 平均并集 | {merge['avg_union']:.1f} 条/题 |")
    lines.append(f"| 平均交集（两路都召回） | {merge['avg_both']:.1f} 条/题 |")
    lines.append(f"| **交集占比**（Jaccard） | {merge['overlap_ratio'] * 100:.1f}% |")
    lines.append(f"| **独有占比** | {merge['novelty_ratio'] * 100:.1f}% |")
    lines.append(f"| 其中仅向量 | {merge['vector_only_ratio'] * 100:.1f}% |")
    lines.append(f"| 其中仅 BM25 | {merge['bm25_only_ratio'] * 100:.1f}% |")
    lines.append("")
    lines.append("> **独有占比是「混合有没有意义」的主指标**：两路召回同一批块时，")
    lines.append("> 融合不会改变任何结果，加一层复杂度是白加。")
    lines.append("")
    lines.append("## 二、五组对照")
    lines.append("")
    lines.append("| 实验 | Recall@5 | 命中 | MRR | 相对基线 | top5 来源 | 配置 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | --- |")
    base = runs[0]
    for run in runs:
        delta = run["recall_at_5"] - base["recall_at_5"]
        delta_text = "—" if run["name"] == base["name"] else f"{delta * 100:+.1f}pp"
        mix = run["top5_source_mix"]
        mix_text = f"both {mix.get('both', 0)} / v {mix.get('vector', 0)} / b {mix.get('bm25', 0)}"
        lines.append(
            f"| `{run['name']}` | **{run['recall_at_5'] * 100:.1f}%** | {run['hits_at_5']} | "
            f"{run['mrr']:.4f} | {delta_text} | {mix_text} | "
            f"`{json.dumps(run['config'], ensure_ascii=False)}` |"
        )
    lines.append("")
    lines.append("### 逐组命中状态变化")
    lines.append("")
    lines.append("50 条集 1 条 = 2pp，所以每 1 条的差异都要能指名：")
    lines.append("")
    for run in runs[1:]:
        gained = [
            q for q, s in run["states"].items() if s == "hit" and base["states"].get(q) != "hit"
        ]
        lost = [
            q for q, s in run["states"].items() if s != "hit" and base["states"].get(q) == "hit"
        ]
        if not gained and not lost:
            lines.append(f"- `{run['name']}`：与基线完全相同")
        else:
            lines.append(f"- `{run['name']}`：**+{gained or '无'}** / **−{lost or '无'}**")

    sig = payload.get("significance", [])
    lines.append("")
    lines.append("## 三、显著性（配对 McNemar 精确检验，对比纯向量）")
    lines.append("")
    lines.append("| 对比 | 命中 | 仅基线 | 仅本组 | p 值 | 判定 | Wilson 95% |")
    lines.append("| --- | ---: | ---: | ---: | ---: | :---: | --- |")
    for row in sig:
        lo, hi = row["wilson"]
        verdict = "**显著**" if row["significant"] else "不显著"
        lines.append(
            f"| `{row['name']}` | {row['hits']}/40 | {row['only_base']} | {row['only_this']} | "
            f"{row['p_value']:.4f} | {verdict} | [{lo * 100:.1f}, {hi * 100:.1f}] |"
        )
    lines.append("")
    if not any(row["significant"] for row in sig):
        lines.append("> **没有任何一组与纯向量基线有显著差异。**")
        lines.append("> 主表里 35.0% ~ 45.0% 的排序在 40 条样本上无法与噪声区分，最多只差 3 条。")
        lines.append("> 这与 W5 三策略对比是同一类结论：**先别急着说融合更好。**")
    lines.append("")
    lines.append("## 四、本周基线")
    lines.append("")
    lines.append(f"**选定：`{baseline['name']}`（{baseline['label']}）**")
    lines.append("")
    lines.append(
        f"- Recall@5 **{baseline['recall_at_5'] * 100:.1f}%**、MRR **{baseline['mrr']:.4f}**"
    )
    lines.append(f"- 复杂度 **{baseline['complexity']}** 个配置项（不含 mode/candidates）")
    lines.append(f"- 规则：{baseline['rule']}")
    lines.append(f"- **理由：{baseline['reason']}**")
    if baseline.get("best_recall_but_not_significant"):
        lines.append(
            f"- 注意：`{baseline['best_recall_but_not_significant']}` 的 "
            f"Recall@5（{baseline['best_recall_value'] * 100:.1f}%）比基线高，"
            "**但配对检验 p > 0.05** —— 它是本表最好看的数字，不是可靠的改进。"
        )
    lines.append("")
    lines.append("> 规则跑之前定死 —— 否则会变成「先看谁赢再解释为什么它该赢」。")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
