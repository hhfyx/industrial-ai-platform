"""Reranker 小样本验证：**精排前后 top5 变化**。

## 这个脚本要回答的问题

任务书要求：10 条 query 上，精排前后 top5 是否变化，人工判"变好/变坏/没变"。

但"变好/变坏"需要客观判据，不能只靠感觉。这里用**三条可量化的判据**：

| 判据 | 怎么算 | 含义 |
| --- | --- | --- |
| **gold 名次变化** | 精排后 gold 在第几 vs 精排前 | 负数 = 往前挪 = 变好 |
| **gold 是否进 top5** | 精确命中判据 | 最硬的指标 |
| **MRR@10 变化** | 倒数排名 | 反映"挪了多少位"，不只是"进没进" |

人工判断的价值在于**看具体案例**：为什么某条变差了？
脚本会把"top5 里被换掉的是哪块、换进来的是哪块"都打出来。

## 为什么只跑 10 条

任务书定的。而且精排很慢（CPU 上 Top20 要 13 秒/条），跑 10 条已经要 2 分钟。
**50 条全跑要 10 分钟以上** —— 那属于"值得单独跑一次"的量级，不该塞进日常验证。

## 精排的候选来自哪里

**必须来自上游检索**，不能直接拿全库。理由：

1. 全库 1363 块 cross-encode 要 8 分钟/query，不可用（这正是两段式的理由）
2. 线上链路就是"检索 topN → 精排"，跳过检索测的不是线上行为

默认 `--candidates 20`（与 W6 周二定下的混合候选窗口一致）。
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

from app.rag.bm25.rerank import DEFAULT_MODEL_DIR, Reranker  # noqa: E402
from app.rag.chunking import ChunkingConfig, build_strategy  # noqa: E402
from app.rag.settings import get_rag_config  # noqa: E402

OUT_JSON: Path = PROJECT_ROOT / "scripts" / "rerank_eval.json"
OUT_MD: Path = PROJECT_ROOT / "scripts" / "rerank_eval.md"

#: 10 条 query，**按"预期精排该有改善"挑选**，不是随机抽。
#:
#: 依据：W5 归因表里被判为"术语未命中"的那 15 条（现在已知那个标签不准确，
#: 但它们的共性是**向量排序把 gold 排在中间位置**）。
#: 精排如果有效，应当把 gold 往前挪。
SAMPLE_IDS: tuple[str, ...] = (
    "f-30",  # 错误码，金标是含 INVALID_INPUT 的表格块
    "f-18",  # bge 512 token
    "f-20",  # 块的确定性 id
    "f-05",  # TTFT 口径
    "f-07",  # 六轮基准首字等待
    "f-25",  # CoT 提示代价
    "f-19",  # warmup / embedding 服务
    "f-22",  # 半句切分，块尾停位
    "f-08",  # 回灌 token 成本
    "f-10",  # 三条人工判据自动化
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reranker 小样本验证")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    cfg = get_rag_config()
    docs = rh.load_raw_corpus()
    chunk_cfg = ChunkingConfig(chunk_chars=cfg.chunk_chars, overlap_chars=cfg.overlap_chars)
    chunks = build_strategy(cfg.strategy, cfg=chunk_cfg).chunk(docs, chunk_cfg)
    print(
        f"语料 {len(docs)} 段；索引 {len(chunks)} 块 ｜ 候选 {args.candidates} ｜ top-{args.top_k}",
        file=sys.stderr,
    )

    items = {item.id: item for item in rre.load_questions(rre.EVAL_FILE)}
    missing = [qid for qid in SAMPLE_IDS if qid not in items]
    if missing:
        print(f"评测集里找不到：{missing}", file=sys.stderr)
        return 2
    probe_items = [items[qid] for qid in SAMPLE_IDS]
    rre.resolve_all(items.values(), chunks)
    chunks_by_id = {chunk.id: chunk for chunk in chunks}

    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.stores.memory import InMemoryVectorStore

    embedder = CachedEmbedder(get_embedder("bge-small-zh"))
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))

    reranker = Reranker(model_path=args.model_dir)
    started = time.perf_counter()
    reranker.load()
    load_ms = int((time.perf_counter() - started) * 1000)
    print(f"模型加载 {load_ms} ms", file=sys.stderr)

    rows: list[dict[str, Any]] = []
    for item in probe_items:
        search_started = time.perf_counter()
        hits = store.search(embedder.embed_query([item.question])[0], top_k=args.candidates)
        candidates = [(hit.chunk.id, hit.chunk.text) for hit in hits]
        before_ids = [cid for cid, _text in candidates]
        retrieval_ms = int((time.perf_counter() - search_started) * 1000)

        rerank_started = time.perf_counter()
        scores, report = reranker.rerank(item.question, candidates, top_k=args.candidates)
        rerank_ms = int((time.perf_counter() - rerank_started) * 1000)
        after_ids = [score.chunk_id for score in scores]

        gold = list(item.gold_chunk_ids)

        def _rank_of(ids: list[str], gold_ids: list[str]) -> int | None:
            """多锚点题：全部锚点都在才记名次（取最靠后的那个，与 eval 同口径）。"""
            positions = {cid: i for i, cid in enumerate(ids, start=1)}
            ranks = [positions.get(gid) for gid in gold_ids]
            if all(r is not None for r in ranks):
                return max(r for r in ranks if r is not None)  # type: ignore[type-var]
            return None

        rank_before = _rank_of(before_ids, gold)
        rank_after = _rank_of(after_ids, gold)

        def _mrr(rank: int | None) -> float:
            return 1.0 / rank if rank else 0.0

        # top5 里被换掉的块
        top_before = before_ids[: args.top_k]
        top_after = after_ids[: args.top_k]
        replaced_in = [cid for cid in top_before if cid not in top_after]
        pushed_in = [cid for cid in top_after if cid not in top_before]

        verdict = _judge_verdict(rank_before, rank_after, args.top_k)
        rows.append(
            {
                "id": item.id,
                "question": item.question,
                "gold_answer": item.gold_answer,
                "rank_before": rank_before,
                "rank_after": rank_after,
                "verdict": verdict,
                "mrr_before": round(_mrr(rank_before), 4),
                "mrr_after": round(_mrr(rank_after), 4),
                "pushed_in": [
                    {
                        "chunk_id": cid,
                        "source": chunks_by_id[cid].source,
                        "head": chunks_by_id[cid].text[:60].replace("\n", " "),
                    }
                    for cid in pushed_in
                    if cid in chunks_by_id
                ],
                "replaced_in": [
                    {
                        "chunk_id": cid,
                        "source": chunks_by_id[cid].source,
                        "head": chunks_by_id[cid].text[:60].replace("\n", " "),
                    }
                    for cid in replaced_in
                    if cid in chunks_by_id
                ],
                "rerank_ms": rerank_ms,
                "retrieval_ms": retrieval_ms,
                "truncated_pairs": report.truncated_pairs,
            }
        )

    # ---- 汇总 ----
    better = [r["id"] for r in rows if r["verdict"] == "better"]
    worse = [r["id"] for r in rows if r["verdict"] == "worse"]
    same = [r["id"] for r in rows if r["verdict"] == "same"]
    changed = [r["id"] for r in rows if r["rank_before"] != r["rank_after"]]

    mrr_before = sum(r["mrr_before"] for r in rows) / len(rows)
    mrr_after = sum(r["mrr_after"] for r in rows) / len(rows)

    print()
    print("=" * 84)
    print(f"精排前后（{len(rows)} 条 query，候选 {args.candidates}，top-{args.top_k}）")
    print("=" * 84)
    print(f"  {'id':<8}{'精排前':>8}{'精排后':>8}{'判定':>8}{'MRR前':>9}{'MRR后':>9}  变化")
    print("  " + "-" * 76)
    for row in rows:
        before = f"#{row['rank_before']}" if row["rank_before"] else "未进"
        after = f"#{row['rank_after']}" if row["rank_after"] else "未进"
        print(
            f"  {row['id']:<8}{before:>8}{after:>8}{row['verdict']:>8}"
            f"{row['mrr_before']:>9.4f}{row['mrr_after']:>9.4f}"
        )
    print()
    print(f"  变好 {len(better)}：{better}")
    print(f"  变坏 {len(worse)}：{worse}")
    print(f"  没变 {len(same)}：{same}")
    print(f"  名次有变化 {len(changed)} 条：{changed}")
    print(f"  MRR：{mrr_before:.4f} → {mrr_after:.4f}（{mrr_after - mrr_before:+.4f}）")
    hit_before = sum(1 for r in rows if r["rank_before"] and r["rank_before"] <= args.top_k)
    hit_after = sum(1 for r in rows if r["rank_after"] and r["rank_after"] <= args.top_k)
    print(f"  精确命中 top{args.top_k}：{hit_before}/{len(rows)} → {hit_after}/{len(rows)}")

    if args.no_write:
        return 0

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus_docs": len(docs),
            "chunks": len(chunks),
            "chunk_chars": cfg.chunk_chars,
            "overlap": cfg.overlap_chars,
            "strategy": cfg.strategy,
            "model": str(args.model_dir),
            "device": "cpu",
            "load_ms": load_ms,
            "candidates": args.candidates,
            "top_k": args.top_k,
        },
        "summary": {
            "queries": len(rows),
            "better": better,
            "worse": worse,
            "same": same,
            "changed_rank": changed,
            "mrr_before": round(mrr_before, 4),
            "mrr_after": round(mrr_after, 4),
            "hit_before": hit_before,
            "hit_after": hit_after,
        },
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"\n落盘：{OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"      {OUT_MD.relative_to(PROJECT_ROOT)}")
    return 0


def _judge_verdict(rank_before: int | None, rank_after: int | None, top_k: int) -> str:
    """判定"变好/变坏/没变"。**判据优先级**：

    1. **是否进 top_k** 优先：进 → 没进 = 变好；没进 → 进 = 变好
    2. 都在 top_k 内或都在外时，比名次：往前 = 变好，往后 = 变坏
    3. 名次不变（含两者都是 None）= 没变

    为什么"进没进 top5"优先于"挪了几位"：
    **top5 是喂给模型的上下文窗口，进没进决定答案能不能被答出来**；
    而从第 20 挪到第 15 对最终答案没有任何影响。
    """
    in_before = rank_before is not None and rank_before <= top_k
    in_after = rank_after is not None and rank_after <= top_k
    if in_before and not in_after:
        return "worse"
    if not in_before and in_after:
        return "better"
    if rank_before is None and rank_after is None:
        return "same"
    if rank_before is None or rank_after is None:
        # 一条在候选内、一条完全没有 → 后者已由上面处理，这里只剩异常
        return "same"
    if rank_after < rank_before:
        return "better"
    if rank_after > rank_before:
        return "worse"
    return "same"


def _render_markdown(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    summary = payload["summary"]
    lines: list[str] = []
    lines.append("# Reranker 小样本验证（精排前后 top5）")
    lines.append("")
    lines.append(
        f"> {meta['generated_at']} ｜ {meta['chunks']} 块 ｜ 候选 {meta['candidates']} "
        f"｜ top-{meta['top_k']} ｜ device={meta['device']} ｜ 加载 {meta['load_ms']} ms"
    )
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| --- | ---: |")
    lines.append(f"| 变好 | **{len(summary['better'])}** 条 {summary['better']} |")
    lines.append(f"| 变坏 | **{len(summary['worse'])}** 条 {summary['worse']} |")
    lines.append(f"| 没变 | {len(summary['same'])} 条 {summary['same']} |")
    lines.append(f"| 名次有变化 | {len(summary['changed_rank'])} 条 {summary['changed_rank']} |")
    lines.append(f"| MRR | {summary['mrr_before']:.4f} → **{summary['mrr_after']:.4f}** |")
    lines.append(
        f"| 精确命中 top{meta['top_k']} | {summary['hit_before']}/{summary['queries']} → "
        f"**{summary['hit_after']}/{summary['queries']}** |"
    )
    lines.append("")
    lines.append("## 逐条")
    lines.append("")
    lines.append("| id | 精排前 | 精排后 | 判定 | MRR 前 | MRR 后 |")
    lines.append("| --- | ---: | ---: | :---: | ---: | ---: |")
    for row in payload["rows"]:
        before = f"#{row['rank_before']}" if row["rank_before"] else "未进"
        after = f"#{row['rank_after']}" if row["rank_after"] else "未进"
        lines.append(
            f"| {row['id']} | {before} | {after} | {row['verdict']} | "
            f"{row['mrr_before']:.4f} | {row['mrr_after']:.4f} |"
        )
    lines.append("")
    lines.append("## 换进来的块（人工复核用）")
    lines.append("")
    for row in payload["rows"]:
        if not row["pushed_in"] and not row["replaced_in"]:
            continue
        lines.append(f"### {row['id']}（{row['verdict']}）")
        lines.append("")
        lines.append(f"**问题**：{row['question']}")
        lines.append("")
        lines.append(f"**gold**：{row['gold_answer']}")
        lines.append("")
        if row["pushed_in"]:
            lines.append("**精排后进入 top5**：")
            for item in row["pushed_in"]:
                lines.append(f"- [{item['source']}] {item['head']}")
        if row["replaced_in"]:
            lines.append("")
            lines.append("**被换出 top5**：")
            for item in row["replaced_in"]:
                lines.append(f"- [{item['source']}] {item['head']}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
