"""导出全部错例的完整上下文，供人工归因。

## 为什么需要这个脚本（而不是直接看 json）

`eval/results/*.json` 里只有 `id / hit_rank / top_score`。
人工归因要看的是**检索返回了什么、gold 是什么、差在哪** ——
这三样分散在三个地方：

    gold 文本      → eval/questions-50.jsonl 的 anchors[].quote
    返回的块       → 要现场重跑检索才拿得到
    gold 在 top_k 里排第几（或完全不在）→ 已算好但没导出

所以这个脚本把三者拼成一份**给人读的 Markdown**：
每条错例一节，列出问题、gold 原文、top_k 返回的块（带分数与来源）、
以及"gold 实际落在哪个块里、那个块排第几"。

## 四类归因的判据（写在这里，因为人工归因最容易标准漂移）

| 类别 | 判据 | 怎么从数据看出来 |
| --- | --- | --- |
| ① 切分断语义 | gold 那句话被切开，没有任何一块完整包含它 | 锚点解析时会报"找不到"；或 gold 块很短/以半句结尾 |
| ② 术语/同义词未命中 | gold 块在库里、但 top_k 里全是别的主题 | 分数接近（0.5~0.65），说明"语义上不远但排不上" |
| ③ 多跳需跨块 | 单锚点命中但另一锚点没命中（或都没命中） | `anchor_ranks` 里有 `None` |
| ④ 问题本身模糊 | gold 有多处可答，或问题与 gold 的对应关系弱 | 看 gold_answer 与 question，人判 |

**这四类的边界不总是清楚**（比如"切分断语义"可能同时是"问题模糊"）。
所以每条错例允许**一个主类 + 一个副类**，并写一句判断依据 ——
那句话才是归因的价值所在，不是分类本身。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import retrieval_harness as rh  # noqa: E402
import run_retrieval_eval as rre  # noqa: E402

DEFAULT_OUT: Path = PROJECT_ROOT / "eval" / "results" / "error-cases.md"


def build_index(*, embedder: Any, chunk_chars: int, overlap: int, strategy: str) -> Any:  # noqa: ANN401
    from app.rag.chunking import ChunkingConfig, build_strategy
    from app.rag.stores.memory import InMemoryVectorStore

    docs = rh.load_raw_corpus()
    cfg = ChunkingConfig(chunk_chars=chunk_chars, overlap_chars=overlap)
    chunks = build_strategy(strategy, cfg=cfg).chunk(docs, cfg)
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))
    return store, chunks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出错例上下文供人工归因")
    parser.add_argument("--chunk-chars", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.settings import get_rag_config

    cfg = get_rag_config()
    chunk_chars = args.chunk_chars if args.chunk_chars is not None else cfg.chunk_chars
    overlap = args.overlap if args.overlap is not None else cfg.overlap_chars
    strategy = args.strategy or cfg.strategy

    embedder = CachedEmbedder(get_embedder("bge-small-zh"))
    store, chunks = build_index(
        embedder=embedder, chunk_chars=chunk_chars, overlap=overlap, strategy=strategy
    )
    items = rre.load_questions()
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

    # 建 chunk_id → chunk 的索引，用于标注"gold 块在库里排第几"
    lines: list[str] = []
    lines.append("# 错例归因（人工填 `归因` 与 `依据` 两列）")
    lines.append("")
    lines.append(
        f"> 配置：strategy={strategy} chunk={chunk_chars} overlap={overlap} top_k={args.top_k}  "
        f"｜ 索引 {len(chunks)} 块 ｜ 题目 {len(probe)} 条"
    )
    lines.append("")
    lines.append("四类归因：① 切分断语义 ② 术语/同义词未命中 ③ 多跳需跨块 ④ 问题本身模糊")
    lines.append("")

    error_rows: list[dict[str, Any]] = []
    for item in probe:
        vector = embedder.embed_query([item.question])[0]
        hits = store.search(vector, top_k=args.top_k)
        outcome = rre.evaluate_query(item=item, hits=hits, k=args.top_k)

        # 拒答题单独处理：它"没命中"是正常的，判据是 top1 是否低于阈值
        if item.is_refusal:
            if outcome.top_score is not None and outcome.top_score < cfg.score_threshold:
                continue  # 正确拒了，不算错例
            error_rows.append(
                {
                    "item": item,
                    "outcome": outcome,
                    "hits": hits,
                    "kind": "refusal_false_accept",
                }
            )
            continue

        if outcome.hit:
            continue
        error_rows.append({"item": item, "outcome": outcome, "hits": hits, "kind": "recall_miss"})

    lines.append(f"## 共 {len(error_rows)} 条错例")
    lines.append("")
    lines.append("| id | 类型 | 错法 | 主归因 | 依据 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in error_rows:
        item = row["item"]
        label = "该拒却答" if row["kind"] == "refusal_false_accept" else "未命中"
        ranks = "/".join("未" if r is None else f"t{r}" for r in row["outcome"].anchor_ranks)
        lines.append(f"| {item.id} | {item.type} | {label}（锚点 {ranks}） | _待填_ | _待填_ |")
    lines.append("")

    # 逐条详情
    for row in error_rows:
        item = row["item"]
        outcome = row["outcome"]
        hits = row["hits"]
        lines.append("---")
        lines.append("")
        lines.append(f"### {item.id}（{item.type}）")
        lines.append("")
        lines.append(f"**问题**：{item.question}")
        lines.append("")
        lines.append(f"**gold_answer**：{item.gold_answer}")
        lines.append("")
        for index, anchor in enumerate(item.anchors, start=1):
            lines.append(f"**锚点 {index}**（{anchor.get('source')}）：`{anchor.get('quote')}`")
            lines.append("")
        lines.append(f"**top{args.top_k} 返回**：")
        lines.append("")
        for rank, hit in enumerate(hits, start=1):
            marker = ""
            if hit.chunk.id in item.gold_chunk_ids:
                marker = "  ← **这是 gold**"
            lines.append(
                f"{rank}. `score={hit.score:.4f}` [{hit.chunk.source}] "
                f"{hit.chunk.text[:100].replace(chr(10), ' ')}…{marker}"
            )
        lines.append("")
        # gold 块在 top_k 之外时的排名
        if item.gold_chunk_ids:
            deeper = store.search(embedder.embed_query([item.question])[0], top_k=50)
            positions = {}
            for rank, hit in enumerate(deeper, start=1):
                if hit.chunk.id in item.gold_chunk_ids:
                    positions[hit.chunk.id] = rank
            if positions:
                lines.append(
                    f"**gold 在 top50 里的位置**：{positions}"
                    f"（top{args.top_k} 内没进 → 召回问题，不是排序问题）"
                )
            else:
                lines.append("**gold 在 top50 里都找不到** → 可能是检索完全没定位到")
            lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"错例 {len(error_rows)} 条，已导出 {args.out.relative_to(PROJECT_ROOT)}")
    from collections import Counter

    print("按类型：", dict(Counter(row["item"].type for row in error_rows)))
    print("按错法：", dict(Counter(row["kind"] for row in error_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
