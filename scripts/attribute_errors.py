"""逐条分析错例，按四类归因自动给出**候选判断**，人再复核。

## 自动归因能到哪一步、不能到哪一步

自动能判的（有明确信号）：

- **③ 多跳需跨块**：`anchor_ranks` 里有 `None` → 确定是多跳缺一块
- **① 切分断语义**：gold 块以半句结尾 / 开头是半句 → 强信号
- **② 术语未命中**：gold 块在 top50 内但排位靠后，且分数接近 →
  "能定位但排不上" 是排序/语义问题

自动**不能**判的：

- **④ 问题本身模糊** —— 这要读语义，机器给不出

所以脚本输出"候选主类 + 支持它的原始数字"，人在 `error-cases.md` 里填最终判断。
**不让脚本直接写结论**，因为归因表的价值在判断依据，不在分类标签。
"""

from __future__ import annotations

import json
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

DEFAULT_OUT: Path = PROJECT_ROOT / "scripts" / "error_attribution.json"

# 半句结尾的信号：块末不是这些标点，且长度接近预算
SENTENCE_END = "。！？；：…”』）)】` \n"


def looks_truncated(text: str) -> bool:
    """块尾是不是落在半句/半个词上。

    这是"切分断语义"的**直接证据**：如果 gold 块的结尾没有句末标点，
    那它多半是从一句话中间切出来的。
    """
    stripped = text.rstrip()
    if not stripped:
        return False
    return stripped[-1] not in SENTENCE_END


def classify(
    *,
    item: Any,  # noqa: ANN401
    outcome: Any,  # noqa: ANN401
    gold_chunks: list[Any],
    top50_rank: int | None,
    top1_score: float | None,
    threshold: float,
) -> dict[str, Any]:
    """给一条错例打**候选中**的归因，并附判断依据。"""
    signals: list[str] = []
    candidate: str | None = None

    # ---- 拒答题误放：单独一类
    if item.is_refusal:
        return {
            "candidate": "refusal_false_accept",
            "signals": [
                f"该拒却答：top1={top1_score} >= 阈值 {threshold}",
                "拒答判据是分数，而分数不含「能不能回答」这个信息",
            ],
            "auto_confidence": "high",
        }

    # ---- ③ 多跳：锚点有 None
    if item.type == "multi_hop":
        missing = [i for i, rank in enumerate(outcome.anchor_ranks) if rank is None]
        if missing:
            signals.append(f"多跳的 {len(missing)} 个锚点未命中（锚点位置 {outcome.anchor_ranks}）")
            if len(missing) == 1:
                signals.append("只缺一处 → 需要「跨块合并」或两轮检索才能补齐")
            else:
                signals.append("两处都缺 → 不只是跨块问题，可能整体没定位到")
            return {
                "candidate": "③ 多跳需跨块",
                "signals": signals,
                "auto_confidence": "high",
            }

    # ---- **表格块要单独看**：它整块不切是契约，不是"断语义"
    #
    # 第一版把所有"以非句末标点结尾"的 gold 块都判成 ①，结果 7 条里 5 条
    # 是表格（结尾是 `|`）。**表格不以句号结尾是正常的** —— 它不是句子。
    # 把它们算成"切分断语义"会导向错误的改进方向（去调切分器），
    # 而真正该做的是"长块降权"或"表格单独路由"。
    table_chunks = [chunk for chunk in gold_chunks if chunk.metadata.get("block") == "table"]
    if table_chunks:
        signals.append(
            f"gold 块是**表格**（{len(table_chunks)} 个，长度 "
            f"{[len(c.text) for c in table_chunks]}）—— 整块不切是契约"
        )
        if top50_rank is None:
            signals.append("表格块在 top50 里找不到 → 表格通常很长，被更短更聚焦的正文块挤掉")
            signals.append("修法不是改切分，而是「长块降权」或「表格单独路由」")
            candidate = "② 术语/同义词未命中"
        else:
            signals.append(f"表格块在 top50 第 {top50_rank} 位 → 排序问题")
            candidate = "② 术语/同义词未命中"
        return {"candidate": candidate, "signals": signals, "auto_confidence": "medium"}

    # ---- ①②：gold 块在库里的状态
    if top50_rank is None:
        signals.append("gold 块在 top50 里都没有 → 检索完全没定位到")
        # 看 gold 块本身是否被切断
        truncated = [looks_truncated(chunk.text) for chunk in gold_chunks]
        if any(truncated):
            signals.append(f"gold 块以非句末标点结尾（{truncated}）→ 强信号：切分断了语义")
            candidate = "① 切分断语义"
        else:
            signals.append("gold 块结尾正常 → 更可能是术语/表述差异")
            candidate = "② 术语/同义词未命中"
        return {"candidate": candidate, "signals": signals, "auto_confidence": "medium"}

    if top50_rank > 20:
        signals.append(f"gold 块在 top50 的第 {top50_rank} 位 → 能定位但排得很后")
        candidate = "② 术语/同义词未命中"
        signals.append("分数相近但排位靠后 → 语义区分度不足，不是切分问题")
        return {"candidate": candidate, "signals": signals, "auto_confidence": "medium"}

    signals.append(f"gold 块在 top50 的第 {top50_rank} 位（top5 外）→ 排序问题而非定位问题")
    candidate = "② 术语/同义词未命中"
    return {"candidate": candidate, "signals": signals, "auto_confidence": "low"}


def main() -> int:
    from app.rag.chunking import ChunkingConfig, build_strategy
    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.settings import get_rag_config
    from app.rag.stores.memory import InMemoryVectorStore

    cfg = get_rag_config()
    embedder = CachedEmbedder(get_embedder("bge-small-zh"))
    docs = rh.load_raw_corpus()
    chunk_cfg = ChunkingConfig(chunk_chars=cfg.chunk_chars, overlap_chars=cfg.overlap_chars)
    chunks = build_strategy(cfg.strategy, cfg=chunk_cfg).chunk(docs, chunk_cfg)
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))

    by_id = {chunk.id: chunk for chunk in chunks}
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

    rows: list[dict[str, Any]] = []
    for item in probe:
        vector = embedder.embed_query([item.question])[0]
        hits = store.search(vector, top_k=5)
        outcome = rre.evaluate_query(item=item, hits=hits, k=5)

        if item.is_refusal:
            if outcome.top_score is not None and outcome.top_score < cfg.score_threshold:
                continue
        elif outcome.hit:
            continue

        # gold 在 top50 里的最好位置
        deeper = store.search(vector, top_k=50)
        rank50: int | None = None
        for rank, hit in enumerate(deeper, start=1):
            if hit.chunk.id in item.gold_chunk_ids:
                rank50 = rank
                break

        gold_chunks = [by_id[cid] for cid in item.gold_chunk_ids if cid in by_id]
        verdict = classify(
            item=item,
            outcome=outcome,
            gold_chunks=gold_chunks,
            top50_rank=rank50,
            top1_score=outcome.top_score,
            threshold=cfg.score_threshold,
        )
        rows.append(
            {
                "id": item.id,
                "type": item.type,
                "question": item.question,
                "gold_answer": item.gold_answer,
                "anchor_ranks": list(outcome.anchor_ranks),
                "top1_score": round(outcome.top_score, 4) if outcome.top_score else None,
                "gold_rank_in_top50": rank50,
                "gold_chunk_lengths": [len(chunk.text) for chunk in gold_chunks],
                "gold_chunk_tails": [chunk.text.rstrip()[-12:] for chunk in gold_chunks],
                "top5": [
                    {
                        "score": round(hit.score, 4),
                        "source": hit.chunk.source,
                        "head": hit.chunk.text[:60].replace("\n", " "),
                    }
                    for hit in hits
                ],
                **verdict,
            }
        )

    from collections import Counter

    print(f"错例 {len(rows)} 条")
    print("自动候选分布：", dict(Counter(row["candidate"] for row in rows)))
    print()
    for row in rows:
        print(
            f"── {row['id']}（{row['type']}） 得分 {row['top1_score']}  "
            f"gold 在 top50 第 {row['gold_rank_in_top50']}"
        )
        print(f"   候选：{row['candidate']}（置信 {row['auto_confidence']}）")
        for signal in row["signals"]:
            print(f"   · {signal}")

    DEFAULT_OUT.write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n落盘：{DEFAULT_OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
