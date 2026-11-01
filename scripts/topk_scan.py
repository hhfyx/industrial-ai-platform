"""TopK 扫描：k ∈ {3,5,8,10} 的召回、噪音与上下文成本。

## 必答题：k=10 相比 k=5，Recall@5 会下降吗

不会 —— 而且这不是"经验上不会"，是**结构上不可能**：
`store.search(vector, top_k=10)` 的前 5 条，与 `top_k=5` 返回的 5 条
是同一批（同样的向量、同样的库、同样的排序）。

所以本文用 `Recall@k` 这个口径时有个陷阱：**k 越大，Recall@k 一定不会下降**
（后面几条只是"可能多捞到一条"）。拿它当"k 该取几"的依据，
结论永远是"越大越好" —— 那是错的。

真正的决策变量是**代价**：

    k 变大 → 送进 LLM 的 token 变多（线性）→ 噪音变多（近似线性）
           → 召回只可能涨一点点（边际递减）

所以本脚本同时记三样：`Recall@k`、**上下文 token**、**噪音**。
选 k 就是在这三者之间找拐点，不是看召回。

## 噪音怎么量（这里必须说清口径，否则数字会被误读）

金标是**每条查询一个 anchor**，所以严格说 relevance 只有 0 或 1 ——
`precision@k` 的上界被钉死在 `1/k`，报出来只是 `k` 的函数，没有信息量。

所以这里报两个**真的有信息**的量：

1. `noise_in_top_k = k - relevant`：直接对应"上下文里有几条不是答案"。
   这是 `k` 的线性函数，但它的**绝对值**才是 LLM 实际看到的东西。
2. `distinct_sources`：top_k 里出现了几个不同的文档。
   答案通常在一份文档里，**top_k 却捞了 5 份文档的碎片**，说明上下文被稀释了 ——
   这个量和 k 不是简单线性关系，它反映"检索是在聚焦还是在散射"。

另外记 `wasted_tokens_before_hit`：答案排在第 r 位时，前 r-1 条白占的 token。
这是"k 大反而更贵"的最直接证据之一。

运行：

    uv run python -X utf8 scripts/topk_scan.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from retrieval_harness import (  # noqa: E402
    PROJECT_ROOT,
    chunk_corpus,
    load_corpus,
    load_gold,
)

DEFAULT_OUT: Path = PROJECT_ROOT / "scripts" / "topk_scan.json"

KS: tuple[int, ...] = (3, 5, 8, 10)

# 固定 chunk/overlap：今天只动 k，别的变量必须钉死，否则数字无法归因。
FIXED_CHUNK_CHARS: int = 500
FIXED_OVERLAP: int = 50


def build_index(*, embedder: Any, tokenizer: Any) -> tuple[Any, list[Any], list[int]]:  # noqa: ANN401
    """建一次索引，返回 (store, chunks, tokens)。**所有 k 共用这一份**。

    这是本脚本最重要的一个实现选择：如果每个 k 各自重建索引，
    `distinct_sources` 之类的量就可能因切分抖动而变化，而那会让人误以为
    "k 影响了文档多样性"。**只有一个变量在动**，是这张表能读的前提。
    """
    from app.rag.chunking import ChunkingConfig
    from app.rag.stores.memory import InMemoryVectorStore

    corpus = load_corpus()
    for warning in corpus.warnings:
        print(f"  ⚠ {warning}", file=sys.stderr)
    cfg = ChunkingConfig(chunk_chars=FIXED_CHUNK_CHARS, overlap_chars=FIXED_OVERLAP)
    chunks = chunk_corpus(corpus, cfg)
    vectors = embedder.embed([chunk.text for chunk in chunks])
    store = InMemoryVectorStore()
    store.upsert(chunks, vectors)
    tokens = tokenizer.count([chunk.text for chunk in chunks])
    return store, chunks, tokens


def scan_one(
    *,
    k: int,
    store: Any,  # noqa: ANN401
    tokens_by_store_id: dict[str, int],
    gold: list[Any],
    embedder: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """在固定索引上取 top-k，算召回/噪音/成本。

    **查询向量只算一次、所有 k 共用** —— 否则 k=10 与 k=5 的差异里会混进
    查询编码的抖动。同一向量 + 同一库 + 同一排序 → 前 5 条必然相同。
    """
    per_query: list[dict[str, Any]] = []
    total_tokens = 0
    total_sources = 0
    total_noise = 0
    wasted_before_hit = 0

    for item in gold:
        query_vector = embedder.embed_query([item.question])[0]
        hits = store.search(query_vector, top_k=k)

        rank: int | None = None
        for position, hit in enumerate(hits, start=1):
            if item.anchor in hit.chunk.text:
                rank = position
                break

        hit_tokens = [tokens_by_store_id.get(hit.chunk.store_id, 0) for hit in hits]
        context_tokens = sum(hit_tokens)
        sources = {hit.chunk.source for hit in hits}
        # 噪音 = top_k 里不是答案的条数。金标每条只有一个 anchor，
        # 所以"命中"最多贡献 1 条相关 —— 这个口径的上界是结构决定的，见模块 docstring。
        noise = len(hits) - (1 if rank else 0)
        wasted = sum(hit_tokens[: rank - 1]) if rank else 0

        total_tokens += context_tokens
        total_sources += len(sources)
        total_noise += noise
        wasted_before_hit += wasted

        per_query.append(
            {
                "id": item.id,
                "hit_rank": rank,
                "context_tokens": context_tokens,
                "noise": noise,
                "distinct_sources": len(sources),
                "wasted_tokens_before_hit": wasted,
            }
        )

    n = len(gold)
    hits = sum(1 for row in per_query if row["hit_rank"])
    return {
        "k": k,
        "queries": n,
        "hits": hits,
        "recall_at_k": round(hits / n, 4) if n else 0.0,
        "mrr": round(sum(1 / row["hit_rank"] for row in per_query if row["hit_rank"]) / n, 4)
        if n
        else 0.0,
        "mean_context_tokens": round(total_tokens / n, 1) if n else 0.0,
        "mean_noise": round(total_noise / n, 2) if n else 0.0,
        "mean_distinct_sources": round(total_sources / n, 2) if n else 0.0,
        "mean_wasted_tokens_before_hit": round(wasted_before_hit / n, 1) if n else 0.0,
        "per_query": per_query,
    }


def check_prefix_invariant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """验证必答题的结构前提：**大 k 的命中位置必须与小 k 逐条相同**。

    这是 k=10 相对 k=5 "Recall@5 不会下降" 的机制证明，不是经验观察。
    如果这条不成立，说明库或排序不稳定（比如同分时随机排），
    那所有"k 变大召回变好"的结论都得重来。
    """
    by_k = {row["k"]: {item["id"]: item for item in row["per_query"]} for row in rows}
    ordered = sorted(by_k)
    mismatches: list[str] = []
    for smaller, larger in zip(ordered, ordered[1:], strict=False):
        for query_id, small_row in by_k[smaller].items():
            large_row = by_k[larger][query_id]
            small_rank = small_row["hit_rank"]
            large_rank = large_row["hit_rank"]
            # 小 k 命中的，大 k 必须命中在同一个位置；
            # 小 k 未命中的，大 k 可能在新位置上命中（这正是 k 变大的唯一收益）。
            if small_rank is not None and small_rank != large_rank:
                mismatches.append(
                    f"k={smaller} 命中 top{small_rank}，但 k={larger} 变成 top{large_rank}（{query_id}）"
                )
    return {
        "holds": not mismatches,
        "checked_pairs": len(ordered) - 1,
        "mismatches": mismatches,
    }


def render(rows: list[dict[str, Any]]) -> str:
    lines = [
        f"TopK 扫描（chunk_chars={FIXED_CHUNK_CHARS} / overlap={FIXED_OVERLAP}，"
        f"{rows[0]['queries']} 条 gold）"
    ]
    lines.append(
        f"  {'k':>3} {'Recall@k':>9} {'MRR':>7} {'上下文token':>11} "
        f"{'噪音条数':>9} {'文档数':>7} {'命中前的废token':>14}"
    )
    lines.append("  " + "-" * 78)
    for row in rows:
        lines.append(
            f"  {row['k']:>3} {row['recall_at_k'] * 100:>8.1f}% {row['mrr']:>7.3f} "
            f"{row['mean_context_tokens']:>11} {row['mean_noise']:>9} "
            f"{row['mean_distinct_sources']:>7} {row['mean_wasted_tokens_before_hit']:>14}"
        )
    return "\n".join(lines)


def render_marginal(rows: list[dict[str, Any]]) -> str:
    """边际分析：**每多送 100 个 token，多换来多少召回**。

    这张表直接回答"k 该取几"，因为它把收益与成本放在同一个数上。
    """
    lines = ["", "  边际收益（相对上一个 k）："]
    lines.append(
        f"    {'区间':>10} {'Δ召回':>8} {'Δtoken':>8} {'Δ噪音':>7} {'每100token换来的召回':>22}"
    )
    for previous, current in zip(rows, rows[1:], strict=False):
        delta_recall = (current["recall_at_k"] - previous["recall_at_k"]) * 100
        delta_tokens = current["mean_context_tokens"] - previous["mean_context_tokens"]
        delta_noise = current["mean_noise"] - previous["mean_noise"]
        if delta_tokens > 0:
            per_100 = round(delta_recall / (delta_tokens / 100), 2)
            per_100_text = f"{per_100:+.2f} pp"
        else:
            per_100_text = "—"
        lines.append(
            f"    k={previous['k']}->{current['k']:<3} {delta_recall:>+7.1f}pp "
            f"{delta_tokens:>+8.1f} {delta_noise:>+7.1f} {per_100_text:>22}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TopK 扫描：召回 / 噪音 / 上下文成本")
    parser.add_argument("--embedder", choices=["bge", "hashing"], default="bge")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.embedding.tokenizer import BGETokenizer

    if args.embedder == "hashing":
        print("⚠ hashing 下召回率无参考价值，仅验证流程", file=sys.stderr)
        embedder: Any = get_embedder("hashing")
    else:
        embedder = CachedEmbedder(get_embedder("bge-small-zh"))

    tokenizer = BGETokenizer()
    gold = load_gold()

    started = time.perf_counter()
    store, chunks, tokens = build_index(embedder=embedder, tokenizer=tokenizer)
    build_ms = (time.perf_counter() - started) * 1000
    tokens_by_store_id = {
        chunk.store_id: token for chunk, token in zip(chunks, tokens, strict=True)
    }
    print(
        f"索引：{len(chunks)} 块（{build_ms:.0f} ms），gold {len(gold)} 条",
        file=sys.stderr,
    )

    rows = [
        scan_one(
            k=k,
            store=store,
            tokens_by_store_id=tokens_by_store_id,
            gold=gold,
            embedder=embedder,
        )
        for k in KS
    ]

    print()
    print(render(rows))
    print(render_marginal(rows))

    invariant = check_prefix_invariant(rows)
    print()
    print("  必答题的结构前提（大 k 的命中位置必须与小 k 一致）：")
    if invariant["holds"]:
        print("    ✓ 成立 —— 所有 k 的前缀命中位置逐条相同")
        print("      所以 k=10 的 Recall@5 与 k=5 的 Recall@5 **必然相等**，不可能是巧合")
    else:
        print("    ✗ 不成立：")
        for mismatch in invariant["mismatches"]:
            print(f"      {mismatch}")

    payload = {
        "meta": {
            "chunk_chars": FIXED_CHUNK_CHARS,
            "overlap": FIXED_OVERLAP,
            "embedder": args.embedder,
            "gold": len(gold),
            "chunks": len(chunks),
            "index_build_ms": round(build_ms, 1),
            "score_direction": "score 是**相似度**（越大越像），不是 distance",
        },
        "rows": rows,
        "prefix_invariant": invariant,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘：{args.out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
