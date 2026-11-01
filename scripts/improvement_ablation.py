"""小改进验证：**每类错例做一个最小改动，各跑一遍 50 条记录增益**。

## 为什么要这个脚本（而不是改完代码直接看一个总数）

三类错例指向三种完全不同的改法：

| 错例类 | 改进 | 预期救哪类 |
| --- | --- | --- |
| ③ 多跳需跨块 | **相邻块扩展**：命中一块后把它的邻居一起带进上下文 | 多跳缺一处 |
| ② 术语未命中 | **相邻块扩展**（同一机制的另一面：答案常和命中块相邻） | 排位靠后 |
| ① 切分断语义 | 已确认只有 2 条，不值得单独改 | —— |

只做一个改动、看总召回涨多少，会把"救了 3 条、弄坏 2 条"读成"净赚 1 条"。
所以这里**逐条记录谁被救、谁被弄坏**，并单独报增益/损失。

## 相邻块扩展怎么做（以及它的边界）

命中块 `c` 之后，把 `index-1` 与 `index+1` 的块**追加到候选列表尾部**。
它不改变 top_k 的排序，只扩充"能看到的范围"。

**为什么这能救多跳**：多跳的两个锚点常在同一文档的相邻位置（
`docs/eval-50.md` 里 10 条多跳有 6 条同源）。命中第一处后，
第二处就在隔壁，但 top_k 里挤不进来。

**代价必须一起量**：
- 上下文 token 增加（每个命中块最多多带 2 块）
- 噪音增加（邻居可能不相关）

所以这个改动**不是免费的**，必须同时报 token 增量。
如果多跳救回 2 条（+5pp）但上下文涨 40%，那要单独判断值不值。
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

import retrieval_harness as rh  # noqa: E402
import run_retrieval_eval as rre  # noqa: E402

DEFAULT_OUT: Path = PROJECT_ROOT / "scripts" / "improvement_ablation.json"

TOP_K: int = 5


def build_index(*, embedder: Any, cfg: Any, strategy: str) -> tuple[Any, list[Any]]:  # noqa: ANN401
    """建索引。`strategy` 与 `cfg` 分开传 —— `strategy` 是装配参数，
    不在 `ChunkingConfig` 里（见 `chunking/factory.py` 的说明）。"""
    from app.rag.chunking import build_strategy
    from app.rag.stores.memory import InMemoryVectorStore

    docs = rh.load_raw_corpus()
    chunks = build_strategy(strategy, cfg=cfg).chunk(docs, cfg)
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))
    return store, chunks


def expand_neighbors(
    hits: list[Any],
    by_source_index: dict[tuple[str, int], Any],
    *,
    budget: int = 2,
    max_total: int | None = None,
) -> list[Any]:
    """把命中块的**邻居**追加到候选尾部。

    ## 三个参数都在控制"偷偷加大 k"这个风险

    - `budget`：每个命中块带几个邻居（1 或 2）
    - `max_total`：候选总数上限。**没有它，5 个命中块各带 2 个邻居 = 15 块**，
      那已经不是"扩充上下文"，而是"把 k 从 5 改成 15" ——
      而 k 变大必然涨召回（`topk_scan.check_prefix_invariant` 证明过），
      这个改动的"收益"就变成假的。

    所以做三档 A/B：

        off              基线（5 块）
        budget=1, cap=8  克制版（最多 8 块）
        budget=2, cap=99 激进版（实测 14.2 块，**本质是加大 k**）

    只有"克制版"的收益才算这个改动的真实价值。

    ## 只追加，不重排

    如果插到前面就改变了 `hit@k` 的口径。追加到尾部意味着：
    top_k 排序不变 → 原命中不会降级，只有原本未命中的题**可能**被救。
    """
    extras: list[Any] = []
    seen = {id(hit.chunk) for hit in hits}
    limit = max_total if max_total is not None else len(hits) + len(hits) * budget
    for hit in hits:
        chunk = hit.chunk
        # budget=1 时优先取**下一个**邻居（语义上"接着往下读"更自然），
        # budget=2 时两侧都取。
        offsets = (1,) if budget == 1 else (-1, 1)
        for delta in offsets:
            if len(hits) + len(extras) >= limit:
                break
            neighbor = by_source_index.get((chunk.source, chunk.index + delta))
            if neighbor is not None and id(neighbor) not in seen:
                seen.add(id(neighbor))
                from app.rag.stores.base import SearchHit

                # 邻居的分数用**原块减去一个惩罚**：它是顺手带进来的，
                # 不该与真检索到的块同分。惩罚值 0.01 仅用于排序稳定。
                extras.append(SearchHit(chunk=neighbor, score=max(0.0, hit.score - 0.01)))
    return extras


def evaluate(
    *,
    store: Any,  # noqa: ANN401
    chunks: list[Any],
    items: list[Any],
    embedder: Any,  # noqa: ANN401
    expand: tuple[int, int | None] | None,
    tokenizer: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """跑一遍 50 条。`expand=(budget, max_total)` 时启用相邻块扩展。"""
    by_source_index = {(chunk.source, chunk.index): chunk for chunk in chunks}
    token_cache: dict[str, int] = {}

    def tokens_of(text: str) -> int:
        if text not in token_cache:
            token_cache[text] = tokenizer.count([text])[0]
        return token_cache[text]

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
    total_tokens = 0
    for item in probe:
        vector = embedder.embed_query([item.question])[0]
        hits = store.search(vector, top_k=TOP_K)
        if expand is not None and hits:
            budget, max_total = expand
            hits = [
                *hits,
                *expand_neighbors(hits, by_source_index, budget=budget, max_total=max_total),
            ]
        outcome = rre.evaluate_query(item=item, hits=hits, k=len(hits))
        context_tokens = sum(tokens_of(hit.chunk.text) for hit in hits)
        total_tokens += context_tokens
        rows.append(
            {
                "id": item.id,
                "type": item.type,
                "is_refusal": item.is_refusal,
                "hit": outcome.hit,
                "hit_rank": outcome.hit_rank,
                "anchor_ranks": list(outcome.anchor_ranks),
                "top_score": round(outcome.top_score, 4) if outcome.top_score else None,
                "context_tokens": context_tokens,
                "candidates": len(hits),
            }
        )

    answerable = [row for row in rows if not row["is_refusal"]]
    hits = sum(1 for row in answerable if row["hit"])
    multi = [row for row in answerable if row["type"] == "multi_hop"]
    fact = [row for row in answerable if row["type"] == "fact"]
    return {
        "expand": expand,
        "rows": rows,
        "recall@5": round(hits / len(answerable), 4) if answerable else 0.0,
        "hits": hits,
        "answerable": len(answerable),
        "multi_hop_recall": round(sum(1 for r in multi if r["hit"]) / len(multi), 4)
        if multi
        else 0.0,
        "fact_recall": round(sum(1 for r in fact if r["hit"]) / len(fact), 4) if fact else 0.0,
        "mean_context_tokens": round(total_tokens / len(rows), 1) if rows else 0.0,
        "mean_candidates": round(sum(r["candidates"] for r in rows) / len(rows), 1)
        if rows
        else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="小改进 A/B：相邻块扩展")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.embedding.tokenizer import BGETokenizer
    from app.rag.settings import get_rag_config

    cfg = get_rag_config()
    embedder = CachedEmbedder(get_embedder("bge-small-zh"))
    tokenizer = BGETokenizer()
    store, chunks = build_index(embedder=embedder, cfg=cfg.chunking_config(), strategy=cfg.strategy)
    items = rre.load_questions()
    print(f"索引 {len(chunks)} 块，题目 {len(items)} 条", file=sys.stderr)

    # 三档 A/B：off / 克制版（1 邻居、总上限 8）/ 激进版（2 邻居、无上限）
    # **只有克制版的收益才算这个改动的真实价值** —— 激进版本质是把 k 从 5 加大到 14。
    variants: list[tuple[str, tuple[int, int | None] | None]] = [
        ("off（基线）", None),
        ("克制版 budget=1 cap=8", (1, 8)),
        ("激进版 budget=2 无上限", (2, None)),
    ]

    results: dict[str, dict[str, Any]] = {}
    for label, expand in variants:
        results[label] = evaluate(
            store=store,
            chunks=chunks,
            items=items,
            embedder=embedder,
            expand=expand,
            tokenizer=tokenizer,
        )

    base = results["off（基线）"]
    base_hits = {row["id"]: row["hit"] for row in base["rows"]}

    print()
    print("▓ 相邻块扩展 A/B（top_k=5，只追加不重排）")
    print()
    print(f"  {'变体':<22} {'Recall@5':>9} {'事实':>7} {'多跳':>7} {'token/题':>9} {'候选块':>7}")
    print("  " + "-" * 68)
    for label in results:
        row = results[label]
        print(
            f"  {label:<22} {row['recall@5'] * 100:>8.1f}% "
            f"{row['fact_recall'] * 100:>6.1f}% {row['multi_hop_recall'] * 100:>6.1f}% "
            f"{row['mean_context_tokens']:>9.1f} {row['mean_candidates']:>7.1f}"
        )

    print()
    print("  逐档增益（相对 off）")
    ablations: dict[str, Any] = {}
    for label in results:
        if label == "off（基线）":
            continue
        row = results[label]
        exp_hits = {r["id"]: r["hit"] for r in row["rows"]}
        gained = [qid for qid in base_hits if exp_hits[qid] and not base_hits[qid]]
        lost = [qid for qid in base_hits if base_hits[qid] and not exp_hits[qid]]
        delta_pp = (row["recall@5"] - base["recall@5"]) * 100
        token_delta_pct = (
            (row["mean_context_tokens"] - base["mean_context_tokens"])
            / base["mean_context_tokens"]
            * 100
        )
        per_pp = (
            (row["mean_context_tokens"] - base["mean_context_tokens"]) / delta_pp
            if delta_pp > 0
            else None
        )
        print(
            f"    {label:<22} {delta_pp:>+6.1f}pp  token {token_delta_pct:>+6.1f}%  "
            f"救回 {len(gained)} 弄坏 {len(lost)}  "
            f"{f'每 pp 花 {per_pp:.0f} token' if per_pp else '—'}"
        )
        if gained:
            print(f"      救回：{gained}")
        if lost:
            print(f"      弄坏：{lost}")
        ablations[label] = {
            "delta_pp": round(delta_pp, 2),
            "token_delta_pct": round(token_delta_pct, 1),
            "gained": gained,
            "lost": lost,
            "tokens_per_pp": round(per_pp, 1) if per_pp else None,
            "mean_candidates": row["mean_candidates"],
        }

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chunk_chars": cfg.chunk_chars,
            "overlap": cfg.overlap_chars,
            "strategy": cfg.strategy,
            "top_k": TOP_K,
            "chunks": len(chunks),
            "note": (
                "只追加邻居到候选尾部，不改变 top_k 排序。"
                "激进版候选 14.2 块，本质是把 k 从 5 加大 —— 它的收益不算本改动的价值"
            ),
        },
        "variants": {
            key: {k: v for k, v in row.items() if k != "rows"} for key, row in results.items()
        },
        "ablations": ablations,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘：{args.out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
