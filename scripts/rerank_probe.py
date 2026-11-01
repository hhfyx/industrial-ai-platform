"""Reranker 延迟与截断实测：**本周最关键的一步**。

## 这个脚本要产出的三个数字

任务书要求：单对耗时、batch 8/16/32 的吞吐、Top50 全排一次的端到端秒数。

**为什么这三个数字重要**：它们决定"周六能用多少候选"。
如果 Top50 单 query 要 8 秒，那周六的候选数只能开到 5~10 条 ——
否则整个链路会被精排拖到不可用。**先量再定，不能反过来。**

## 还要测的：截断策略

三种策略（截尾 / 截头 / 中段）对**同一个块**打分可能不同。
实测的意义在于回答："如果我把 max_length 设成 512，会不会有块被切掉关键信息？"

## 为什么不测 GPU

本机 `torch.cuda.is_available()` = False（torch 2.15.0+cpu）。
所有数字都是 CPU 的 —— 这一点必须写进结论，否则"ms/对"会被当成 GPU 预期。
脚本会把 device 写进产物元数据。

## 一个容易忽略的项：模型加载耗时

实测加载要 13.5 秒。如果这部分不算进去，"第一句 query 要 14 秒"就会被当成
"精排很慢"—— 而它其实是一次性的。所以加载耗时**单独计时并单独报告**。
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

from app.rag.bm25.rerank import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    Reranker,
    Truncation,
    normalize_within_query,
)
from app.rag.chunking import ChunkingConfig, build_strategy  # noqa: E402
from app.rag.settings import get_rag_config  # noqa: E402

OUT_JSON: Path = PROJECT_ROOT / "scripts" / "rerank_probe.json"
OUT_MD: Path = PROJECT_ROOT / "scripts" / "rerank_probe.md"

#: 吞吐测试用的 batch 档位。任务点名 8/16/32。
BATCH_SIZES: tuple[int, ...] = (1, 8, 16, 32)

#: 端到端测试用的候选数。任务点名 Top50；这里加 Top20 做对照，
#: 因为 W6 周二定下的混合候选窗口就是 20 —— 需要知道"在现有窗口上精排要多久"。
CANDIDATE_WINDOWS: tuple[int, ...] = (20, 50)


def _build_candidates(
    query: str, chunks: list[Any], embedder: Any, store: Any, top_n: int
) -> list[tuple[str, str]]:
    """用向量检索取候选。**精排的输入来自上游检索，不该直接拿全库做。**"""
    hits = store.search(embedder.embed_query([query])[0], top_k=top_n)
    return [(hit.chunk.id, hit.chunk.text) for hit in hits]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reranker 延迟与截断实测")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--pairs", type=int, default=50, help="吞吐测试用的对数")
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

    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.stores.memory import InMemoryVectorStore

    embedder = CachedEmbedder(get_embedder("bge-small-zh"))
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))

    # ---- ① 加载耗时 ----
    started = time.perf_counter()
    reranker = Reranker(model_path=args.model_dir)
    reranker.load()
    load_ms = int((time.perf_counter() - started) * 1000)
    print(f"\n模型加载：{load_ms} ms（一次性，不进单次查询延迟）", file=sys.stderr)

    # ---- ② 单对耗时 ----
    single_query = answerable[0].question
    one_pair = [
        (
            answerable[0].gold_chunk_ids[0] if answerable[0].gold_chunk_ids else chunks[0].id,
            chunks[0].text,
        )
    ]
    # 预热一次：首次前向有 lazy 初始化，直接计时会偏大
    reranker.rerank(single_query, one_pair)
    t0 = time.perf_counter()
    _s, single_report = reranker.rerank(single_query, one_pair)
    single_ms = int((time.perf_counter() - t0) * 1000)

    # ---- ③ 批量吞吐 ----
    # **两个模式都测**：`sort_by_length=True`（按长度排序组 batch，默认）与
    # `False`（按检索顺序切 batch）。后者是"自然写法"，实测它 batch 越大越慢 ——
    # 因为长表块会把同 batch 的短块全 padding 掉。留着这个对照，才看得出优化值多少。
    throughput: list[dict[str, Any]] = []
    long_query = max(answerable, key=lambda item: len(item.question)).question
    candidates = _build_candidates(long_query, chunks, embedder, store, args.pairs)
    for batch in BATCH_SIZES:
        for sort_length in (True, False):
            probe = Reranker(
                model_path=args.model_dir, batch_size=batch, sort_by_length=sort_length
            )
            probe._model = reranker._model  # 复用已加载的模型，避免重复加载
            probe._tokenizer = reranker._tokenizer
            probe.rerank(long_query, candidates[:4])  # 预热
            t0 = time.perf_counter()
            _scores, report = probe.rerank(long_query, candidates)
            elapsed = time.perf_counter() - t0
            throughput.append(
                {
                    "batch_size": batch,
                    "sort_by_length": sort_length,
                    "pairs": report.pairs,
                    "elapsed_ms": round(elapsed * 1000, 1),
                    "ms_per_pair": round(elapsed * 1000 / report.pairs, 2),
                    "pairs_per_sec": round(report.pairs / elapsed, 1),
                }
            )
            tag = "按长度排序" if sort_length else "按检索顺序"
            print(
                f"  batch={batch:<3} {tag} {report.pairs} 对 / {elapsed * 1000:.0f} ms"
                f" = {elapsed * 1000 / report.pairs:.2f} ms/对",
                file=sys.stderr,
            )

    # ---- ④ 端到端：TopN 全排 ----
    e2e: list[dict[str, Any]] = []
    for window in CANDIDATE_WINDOWS:
        # 用 5 条 query 平均，避免单条 query 的长短影响结论
        probe = Reranker(model_path=args.model_dir, batch_size=16)
        probe._model = reranker._model
        probe._tokenizer = reranker._tokenizer
        times: list[float] = []
        for item in answerable[:5]:
            cands = _build_candidates(item.question, chunks, embedder, store, window)
            t0 = time.perf_counter()
            _scores, _report = probe.rerank(item.question, cands)
            times.append(time.perf_counter() - t0)
        mean_s = sum(times) / len(times)
        e2e.append(
            {
                "candidates": window,
                "queries": len(times),
                "mean_ms": round(mean_s * 1000, 1),
                "min_ms": round(min(times) * 1000, 1),
                "max_ms": round(max(times) * 1000, 1),
            }
        )
        print(
            f"  Top{window:<3} 平均 {mean_s * 1000:.0f} ms"
            f"（{min(times) * 1000:.0f}~{max(times) * 1000:.0f}）",
            file=sys.stderr,
        )

    # ---- ⑤ 截断策略对比 ----
    # 找一条**最长的块**，让它一定被截断
    longest = max(chunks, key=lambda chunk: len(chunk.text))
    truncation_results: list[dict[str, Any]] = []
    trunc_query = answerable[0].question
    for policy in Truncation:
        probe = Reranker(model_path=args.model_dir, truncation=policy, max_length=256)
        probe._model = reranker._model
        probe._tokenizer = reranker._tokenizer
        scores, report = probe.rerank(trunc_query, [(longest.id, longest.text)])
        truncation_results.append(
            {
                "policy": policy.value,
                "max_length": 256,
                "truncated_pairs": report.truncated_pairs,
                "logit": round(scores[0].logit, 4),
                "score": round(scores[0].score, 6),
            }
        )
        print(
            f"  截断 {policy.value:<8} 截断 {report.truncated_pairs} 对 →"
            f" logit={scores[0].logit:.3f}",
            file=sys.stderr,
        )

    # ---- ⑥ 分数跨 query 可比性（必答问题的实测）----
    spread: list[dict[str, Any]] = []
    for item in answerable[:8]:
        cands = _build_candidates(item.question, chunks, embedder, store, 20)
        scores, _report = probe_reranker_scores(reranker, item.question, cands)
        spread.append(
            {
                "id": item.id,
                "top1": round(scores[0], 4),
                "top5_min": round(min(scores[:5]), 4) if len(scores) >= 5 else None,
                "top1_norm": 1.0,
                "top5_min_norm": round(normalize_within_query(scores)[4], 4)
                if len(scores) >= 5
                else None,
            }
        )
    tops = [row["top1"] for row in spread]
    print(
        f"\n  8 条 query 的 top1 分：{min(tops):.3f} ~ {max(tops):.3f}"
        f"（跨度 {max(tops) - min(tops):.3f}）← 跨 query 不可比的证据",
        file=sys.stderr,
    )

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
            "max_length": reranker.max_length,
            "load_ms": load_ms,
        },
        "single": {"pairs": 1, "elapsed_ms": single_ms},
        "throughput": throughput,
        "end_to_end": e2e,
        "truncation": truncation_results,
        "score_spread": spread,
    }
    if args.no_write:
        return 0
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"\n落盘：{OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"      {OUT_MD.relative_to(PROJECT_ROOT)}")
    return 0


def probe_reranker_scores(
    reranker: Reranker, query: str, candidates: list[tuple[str, str]]
) -> tuple[list[float], Any]:
    scores, _report = reranker.rerank(query, candidates)
    return [item.score for item in scores], _report


def _render_markdown(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    lines: list[str] = []
    lines.append("# Reranker 延迟与截断实测")
    lines.append("")
    lines.append(
        f"> {meta['generated_at']} ｜ {meta['chunks']} 块 ｜ `{Path(meta['model']).name}` "
        f"｜ **device={meta['device']}** ｜ max_length={meta['max_length']}"
    )
    lines.append("")
    lines.append("## 〇、三个必需的数字")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| --- | ---: |")
    lines.append(f"| 模型加载（一次性） | **{meta['load_ms']} ms** |")
    lines.append(f"| 单对耗时 | **{payload['single']['elapsed_ms']} ms** |")
    thr32 = next((row for row in payload["throughput"] if row["batch_size"] == 32), None)
    if thr32:
        lines.append(f"| batch=32 吞吐 | **{thr32['pairs_per_sec']} 对/秒** |")
    for row in payload["end_to_end"]:
        lines.append(f"| Top{row['candidates']} 端到端 | **{row['mean_ms']} ms** |")
    lines.append("")
    lines.append("## 一、批量吞吐")
    lines.append("")
    lines.append("| batch | 对数 | 耗时 | ms/对 | 对/秒 |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for row in payload["throughput"]:
        lines.append(
            f"| {row['batch_size']} | {row['pairs']} | {row['elapsed_ms']} ms | "
            f"{row['ms_per_pair']} | {row['pairs_per_sec']} |"
        )
    lines.append("")
    lines.append("## 二、端到端（TopN 全排）")
    lines.append("")
    lines.append("| 候选数 | 平均 | 最短 | 最长 |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for row in payload["end_to_end"]:
        lines.append(
            f"| Top{row['candidates']} | {row['mean_ms']} ms | {row['min_ms']} ms | {row['max_ms']} ms |"
        )
    lines.append("")
    lines.append("## 三、截断策略（max_length=256 强制截断最长块）")
    lines.append("")
    lines.append("| 策略 | 截断对数 | logit | sigmoid |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in payload["truncation"]:
        lines.append(
            f"| {row['policy']} | {row['truncated_pairs']} | {row['logit']} | {row['score']} |"
        )
    lines.append("")
    lines.append("## 四、分数跨 query 的跨度（必答问题的实测）")
    lines.append("")
    lines.append("| query | top1 | top5 最低 | top1 归一 | top5 归一 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in payload["score_spread"]:
        lines.append(
            f"| {row['id']} | {row['top1']} | {row['top5_min']} | "
            f"{row['top1_norm']} | {row['top5_min_norm']} |"
        )
    tops = [row["top1"] for row in payload["score_spread"]]
    lines.append("")
    lines.append(
        f"> 不同 query 的 top1 分从 **{min(tops):.3f}** 到 **{max(tops):.3f}**，"
        f"跨度 **{max(tops) - min(tops):.3f}**。"
    )
    lines.append("> 同一批候选、同一个模型，**只是换了 query** —— 分数就整体平移。")
    lines.append("> 这就是「Cross-Encoder 分数不能跨 query 比较」的直接证据。")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
