"""两段式工程化：**召回上限、N 的权衡、混合 vs 纯向量 → 精排**。

## 这个脚本要回答本任务书里最关键的那句话

> 如果 Recall@5 只有 0.7，但 Recall@50 = 0.95，问题在**精排**（候选里有，排不上来）。
> 如果 Recall@50 只有 0.8，问题在**召回层**（混合/切分/embedding），
> 此时调精排参数是**徒劳**的。

这就是"**召回上限**"的概念：`Recall@N` 是精排能达到的天花板 ——
gold 不在 N 条候选里，精排再准也救不回来。

**所以这个脚本先算 Recall@N，再算精排后的 Recall@5。** 顺序不能反：
先知道天花板，才知道精排还有多少空间。

## 三项产出

| 产出 | 怎么算 | 用来看什么 |
| --- | --- | --- |
| **Recall@N** 表 | 纯检索，不精排 | **精排的天花板**（本周最容易漏的一行） |
| N ∈ {20,30,50} 的 N→K 权衡 | 检索 N → 精排取 5 | N 该取多少（效果 vs 延迟） |
| 两组路线对比 | 纯向量→精排 / 混合→精排 | 混合值不值得放进第一段 |

## 为什么必须先算 Recall@N 再谈精排

W6 周三已经踩过一次：10 条小样本里 8 条 gold 根本不在候选 top20 里，
精排无从发挥。如果只看"精排前后 Recall@5 没变"就下结论"精排没用"，
那会把**召回层的问题误判成精排的问题** —— 而这两者该修的地方完全不同。

## 延迟

Top50 精排要 16.4 秒（CPU，见 `scripts/rerank_probe.json`）。
本脚本会对每条 query 计时，并把"端到端 p50/p95"写进产物 ——
**N 的选择必须同时看效果和这个数**，否则会选出跑不动的配置。

## 用法

    python -X utf8 scripts/rerank_pipeline.py                 # 全量（慢，约 40~60 分钟）
    python -X utf8 scripts/rerank_pipeline.py --limit 10      # 只跑 10 条（约 8 分钟）
    python -X utf8 scripts/rerank_pipeline.py --recall-only   # 只算 Recall@N（秒级）
"""

from __future__ import annotations

import argparse
import json
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

from app.rag.bm25.fusion import DEFAULT_RRF_K, FusionConfig, FusionMode, merge_rankings  # noqa: E402
from app.rag.bm25.index import DEFAULT_INDEX_PATH  # noqa: E402
from app.rag.bm25.rerank import DEFAULT_MODEL_DIR, Reranker  # noqa: E402
from app.rag.bm25.retriever import BM25Retriever  # noqa: E402
from app.rag.chunking import ChunkingConfig, build_strategy  # noqa: E402
from app.rag.settings import get_rag_config  # noqa: E402

OUT_JSON: Path = PROJECT_ROOT / "scripts" / "rerank_pipeline.json"
OUT_MD: Path = PROJECT_ROOT / "scripts" / "rerank_pipeline.md"

#: `--recall-only` 的产物**另存**，绝不覆盖上面那两份完整产物。
#:
#: 见 `main()` 里 `--recall-only` 分支的说明：第一版写同一个文件，
#: 导致跑一次 `make recall-ceiling` 就把 `n_sweep`/`route_compare` 覆盖成空。
OUT_JSON_RECALL_ONLY: Path = PROJECT_ROOT / "scripts" / "recall_ceiling.json"
OUT_MD_RECALL_ONLY: Path = PROJECT_ROOT / "scripts" / "recall_ceiling.md"

#: 要评估的候选窗口。任务书点名 20/30/50。
CANDIDATES: tuple[int, ...] = (20, 30, 50)

#: 精排后取几条（固定 5，任务书要求）。
TOP_K: int = 5

#: Recall@N 表要看的 N。**必须包含 5 和最大候选 50** ——
#: 前者是"当前线上水平"，后者是"精排天花板"。
RECALL_N: tuple[int, ...] = (5, 10, 20, 30, 50, 100)


@dataclass
class Route:
    """一条检索路线：怎么产生精排的候选。"""

    name: str
    """`pure_vec` 或 `hybrid_rrf`。"""

    description: str


ROUTES: tuple[Route, ...] = (
    Route("pure_vec", "纯向量 topN → 精排"),
    Route("hybrid_rrf", "向量+BM25 → RRF → topN → 精排"),
)


def _gold_rank(ids: list[str], gold_ids: list[str], limit: int | None = None) -> int | None:
    """gold 在 `ids` 里的名次。**多锚点题要求全部锚点都在**（与 eval 同一口径）。

    `limit` 给定时，超出 limit 的名次按"未命中"处理 —— 这正是 Recall@N 的定义。
    """
    if not gold_ids:
        return None
    positions = {cid: i for i, cid in enumerate(ids, start=1)}
    ranks = [positions.get(gid) for gid in gold_ids]
    if not all(r is not None for r in ranks):
        return None
    worst = max(r for r in ranks if r is not None)  # type: ignore[type-var]
    if limit is not None and worst > limit:
        return None
    return worst


def _recall_at_n(
    questions: list[Any], retrieve: Any, ns: tuple[int, ...]
) -> dict[int, dict[str, Any]]:
    """**精排的天花板**：gold 有没有进精排的输入集。

    这是本周最容易漏的一行 —— 因为它看起来"只是又一个 recall 数"，
    实际上它决定了精排还有多少空间可用。

    ## ⚠ 每个 N 必须**独立检索**，不能"拉 max_n 再截断"

    第一版为了省事，只 `retrieve(max(ns))` 一次，然后在结果上截不同长度。
    对纯向量这样是等价的（top-k 是稳定的前缀），但**对混合 RRF 不等价**：

    RRF 的融合池大小会改变名次本身 —— 池取 100 和池取 20，
    同一块在融合后的排位可能不同（因为它跟多少条候选竞争名次倒数）。

    实测差异：融合池 100 截前 20 → Recall@20 = **67.5%**；
    融合池 20 取前 20 → **62.5%**。差 2 条。

    而这个数要与 `_run_sweep` 里"检索 N 条再精排"的口径可比 ——
    那里是独立 `retrieve(n)`。所以这里也必须独立检索，
    **否则"天花板"与"精排后"两个数来自不同口径，比出来的差值没有意义。**
    """
    hits: dict[int, int] = dict.fromkeys(ns, 0)
    for n in ns:
        for item in questions:
            ids = retrieve(item.question, n)
            if _gold_rank(ids, item.gold_chunk_ids, limit=n) is not None:
                hits[n] += 1
    return {
        n: {
            "hits": hits[n],
            "total": len(questions),
            "rate": round(hits[n] / len(questions), 4) if questions else 0.0,
        }
        for n in ns
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="两段式工程化：召回上限 + N 权衡")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    parser.add_argument("--recall-only", action="store_true", help="只算 Recall@N，不精排")
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
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    if args.limit:
        answerable = answerable[: args.limit]
        print(f"⚠ 只跑前 {args.limit} 条（结果不是全量，别写进结论）", file=sys.stderr)

    # ---- 两条路线的候选来源 ----
    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.stores.memory import InMemoryVectorStore

    embedder = CachedEmbedder(get_embedder("bge-small-zh"))
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))
    bm25, bm25_report = BM25Retriever.from_chunks(chunks, path=DEFAULT_INDEX_PATH)
    print(f"BM25：{bm25_report.summary()}", file=sys.stderr)

    fusion_config = FusionConfig(
        mode=FusionMode.RRF, candidates=max(CANDIDATES), rrf_k=DEFAULT_RRF_K
    )

    def retrieve_pure(query: str, n: int) -> list[str]:
        hits = store.search(embedder.embed_query([query])[0], top_k=n)
        return [hit.chunk.id for hit in hits]

    def retrieve_hybrid(query: str, n: int) -> list[str]:
        vec_hits = store.search(embedder.embed_query([query])[0], top_k=n)
        bm25_hits = bm25.retrieve(query, top_k=n)
        fused, _stats = merge_rankings(
            [h.chunk.id for h in vec_hits],
            [h.chunk.id for h in bm25_hits],
            config=fusion_config,
        )
        return [hit.chunk_id for hit in fused[:n]]

    retrievers = {"pure_vec": retrieve_pure, "hybrid_rrf": retrieve_hybrid}

    # ---- ① Recall@N：精排天花板（不需要精排，秒级）----
    print()
    print("=" * 78)
    print("① 召回上限 Recall@N —— 精排的天花板（本周最容易漏的一行）")
    print("=" * 78)
    ceilings: dict[str, dict[int, dict[str, Any]]] = {}
    for route in ROUTES:
        table = _recall_at_n(answerable, retrievers[route.name], RECALL_N)
        ceilings[route.name] = table
        cells = "  ".join(f"@{n}={v['rate'] * 100:.0f}%" for n, v in table.items())
        print(f"  {route.name:<12} {cells}")

    if args.recall_only:
        payload = _build_payload(
            docs=docs,
            chunks=chunks,
            cfg=cfg,
            answerable=answerable,
            ceilings=ceilings,
            n_sweep={},
            route_compare={},
            bm25_stats=bm25.stats(),
            model_dir=args.model_dir,
            limited=bool(args.limit),
        )
        # ## `--recall-only` 必须写**另一个文件**
        #
        # 第一版写同一个 `rerank_pipeline.json`，于是跑一次 `make recall-ceiling`
        # （它带 `--recall-only`）就把完整产物里的 `n_sweep` 与 `route_compare`
        # 两节**覆盖成空**。
        #
        # 这个 bug 的隐蔽处：文件还在、还是合法 JSON、其余字段都对 ——
        # 只有那两节悄悄没了。而 `docs/rerank-pipeline.md` 引用的正是那两节。
        #
        # 实测踩到过（W6 周日在回归时发现产物被改写）。
        # 修法：轻量模式写独立文件，绝不覆盖完整产物。
        return _finish(
            payload, args.no_write, out_json=OUT_JSON_RECALL_ONLY, out_md=OUT_MD_RECALL_ONLY
        )

    # ---- ② N ∈ {20,30,50} × K=5：效果 vs 延迟 ----
    reranker = Reranker(model_path=args.model_dir)
    load_started = time.perf_counter()
    reranker.load()
    load_ms = int((time.perf_counter() - load_started) * 1000)
    print(f"\n模型加载 {load_ms} ms", file=sys.stderr)

    n_sweep: dict[str, dict[str, Any]] = {}
    print()
    print("=" * 78)
    print(f"② N → K 权衡（K 固定 {TOP_K}，纯向量路线）")
    print("=" * 78)
    print(f"  {'N':>4}{'天花板':>9}{'精排后':>9}{'提升':>8}{'p50 延迟':>10}{'超出 2s':>9}")
    for n in CANDIDATES:
        result = _run_sweep(
            answerable=answerable,
            retrieve=lambda q, n=n: retrieve_pure(q, n),
            reranker=reranker,
            n=n,
            chunks_by_id=chunks_by_id,
        )
        n_sweep[str(n)] = result
        ceil = ceilings["pure_vec"][n]["rate"] if n in ceilings["pure_vec"] else None
        gained = (result["rerank_recall"] - result["ceiling"]) * 100 if ceil is not None else 0.0
        over = sum(1 for ms in result["latency_ms"] if ms > 2000)
        print(
            f"  {n:>4}{result['ceiling'] * 100:>8.1f}%{result['rerank_recall'] * 100:>8.1f}%"
            f"{gained:>+7.1f}pp{result['p50_ms']:>9}ms{over:>6}/{len(result['latency_ms'])}"
        )

    # ---- ③ 两组路线：纯向量→精排 vs 混合→精排 ----
    compare_n = 20
    route_compare: dict[str, Any] = {}
    print()
    print("=" * 78)
    print(f"③ 纯向量→精排 vs 混合→精排（N={compare_n}, K={TOP_K}）")
    print("=" * 78)
    for route in ROUTES:
        result = _run_sweep(
            answerable=answerable,
            retrieve=lambda q, r=route.name: retrievers[r](q, compare_n),
            reranker=reranker,
            n=compare_n,
            chunks_by_id=chunks_by_id,
        )
        route_compare[route.name] = result | {"description": route.description}
        print(
            f"  {route.name:<12} 天花板 {result['ceiling'] * 100:.1f}%"
            f" → 精排后 {result['rerank_recall'] * 100:.1f}%"
            f"  p50 {result['p50_ms']}ms"
        )

    payload = _build_payload(
        docs=docs,
        chunks=chunks,
        cfg=cfg,
        answerable=answerable,
        ceilings=ceilings,
        n_sweep=n_sweep,
        route_compare=route_compare,
        bm25_stats=bm25.stats(),
        model_dir=args.model_dir,
        limited=bool(args.limit),
    )
    payload["meta"]["load_ms"] = load_ms
    payload["meta"]["compare_n"] = compare_n
    return _finish(payload, args.no_write)


def _run_sweep(
    *,
    answerable: list[Any],
    retrieve: Any,
    reranker: Reranker,
    n: int,
    chunks_by_id: dict[str, Any],
) -> dict[str, Any]:
    """跑一轮：检索 N → 精排 → 取 K，逐条计时。"""
    rerank_hits = 0
    ceiling_hits = 0
    latencies: list[int] = []
    per_question: list[dict[str, Any]] = []

    for item in answerable:
        started = time.perf_counter()
        ids = retrieve(item.question)
        candidates = [(cid, chunks_by_id[cid].text) for cid in ids if cid in chunks_by_id]
        scores, report = reranker.rerank(item.question, candidates, top_k=TOP_K)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        latencies.append(elapsed_ms)

        ceiling_rank = _gold_rank(ids, item.gold_chunk_ids, limit=n)
        reranked_ids = [score.chunk_id for score in scores]
        final_rank = _gold_rank(reranked_ids, item.gold_chunk_ids, limit=TOP_K)
        if ceiling_rank is not None:
            ceiling_hits += 1
        if final_rank is not None:
            rerank_hits += 1
        per_question.append(
            {
                "id": item.id,
                "ceiling_rank": ceiling_rank,
                "final_rank": final_rank,
                "rerank_ms": elapsed_ms,
                "truncated": report.truncated_pairs,
                # 精排把 gold 挪了几位（只在两者都命中时有意义）
                "delta": (final_rank - ceiling_rank)
                if ceiling_rank is not None and final_rank is not None
                else None,
            }
        )

    total = len(answerable)
    latencies_sorted = sorted(latencies)
    return {
        "n": n,
        "top_k": TOP_K,
        "ceiling": round(ceiling_hits / total, 4) if total else 0.0,
        "ceiling_hits": ceiling_hits,
        "rerank_recall": round(rerank_hits / total, 4) if total else 0.0,
        "rerank_hits": rerank_hits,
        "p50_ms": latencies_sorted[len(latencies_sorted) // 2] if latencies else 0,
        "p95_ms": latencies_sorted[int(len(latencies_sorted) * 0.95)] if latencies else 0,
        "latency_ms": latencies,
        "per_question": per_question,
    }


def _build_payload(
    *,
    docs: list[Any],
    chunks: list[Any],
    cfg: Any,
    answerable: list[Any],
    ceilings: dict[str, dict[int, dict[str, Any]]],
    n_sweep: dict[str, Any],
    route_compare: dict[str, Any],
    bm25_stats: dict[str, Any],
    model_dir: Path,
    limited: bool,
) -> dict[str, Any]:
    return {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus_docs": len(docs),
            "chunks": len(chunks),
            "chunk_chars": cfg.chunk_chars,
            "overlap": cfg.overlap_chars,
            "strategy": cfg.strategy,
            "questions": len(answerable),
            "model": str(model_dir),
            "device": "cpu",
            "limited_run": limited,
        },
        "ceilings": {
            route: {str(n): v for n, v in table.items()} for route, table in ceilings.items()
        },
        "n_sweep": n_sweep,
        "route_compare": route_compare,
        "bm25_stats": bm25_stats,
    }


def _finish(
    payload: dict[str, Any],
    no_write: bool,
    *,
    out_json: Path | None = None,
    out_md: Path | None = None,
) -> int:
    """打印结论并落盘。`out_*` 为 `None` 时用完整产物的默认路径。

    **`--recall-only` 必须传自定义路径** —— 否则会覆盖完整产物的
    `n_sweep` / `route_compare` 两节（见过这个 bug，见 main() 里的说明）。
    """
    ceilings = payload["ceilings"]
    print()
    print("=" * 78)
    print("结论：该修哪一层？")
    print("=" * 78)

    # **先看天花板够不够高，再谈精排有没有空间** —— 顺序不能反。
    #
    # 第一版把判据写成 `if r50 - r5 > 0.15: 问题在精排`，于是 62.5% 的天花板
    # 也被判成"问题在精排"。但 62.5% 意味着**即使精排完美，也只能做到 62.5%**
    # —— 那是召回层的硬上限，精排再有空间也只在 40%→62.5% 之间挪。
    #
    # 正确的判据分两步：
    #   ① 天花板本身够不够高？（<85% → 召回层是主要瓶颈，调精排是徒劳）
    #   ② 天花板与当前 Recall@5 的差距够不够大？（差距大才有精排的空间）
    for route, table in ceilings.items():
        r5 = table.get("5", {}).get("rate", 0.0)
        r50 = table.get("50", {}).get("rate", 0.0)
        headroom = r50 - r5
        print(f"\n  [{route}]")
        print(f"    Recall@5  = {r5 * 100:.1f}%")
        print(f"    Recall@50 = {r50 * 100:.1f}%   ← 精排天花板（完美精排的上限）")
        print(f"    精排可用空间 = {headroom * 100:+.1f}pp")
        if r50 < 0.85:
            print(f"    → **天花板只有 {r50 * 100:.1f}%，主要瓶颈在召回层**")
            print("      （混合 / 切分 / embedding）。此时调精排参数是徒劳的 ——")
            print(
                f"      完美精排也只能到 {r50 * 100:.1f}%，剩下的 {100 - r50 * 100:.1f}pp 要靠召回。"
            )
            if headroom > 0.15:
                print(f"      但精排仍有 {headroom * 100:.1f}pp 空间可拿（把已有的排上来）。")
        elif headroom > 0.15:
            print("    → 天花板够高、且与 Recall@5 差距大 → **问题在精排/排序层**，值得调")
        else:
            print("    → 天花板够高、差距不大 → 精排已接近饱和，需看逐条归因")
    if no_write:
        return 0
    target_json = out_json or OUT_JSON
    target_md = out_md or OUT_MD
    target_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    target_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"\n落盘：{target_json.relative_to(PROJECT_ROOT)}")
    print(f"      {target_md.relative_to(PROJECT_ROOT)}")
    return 0


def _render_markdown(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    lines: list[str] = []
    lines.append("# 两段式工程化：召回上限与 N 的权衡")
    lines.append("")
    lines.append(
        f"> {meta['generated_at']} ｜ {meta['chunks']} 块 ｜ {meta['questions']} 条可答题 "
        f"｜ **device={meta['device']}**"
    )
    if meta.get("limited_run"):
        lines.append("> ⚠ **这是一次限量运行，数字不是全量结论。**")
    lines.append("")
    lines.append("## 一、召回上限 Recall@N —— 精排的天花板")
    lines.append("")
    header_ns = sorted(int(n) for n in next(iter(payload["ceilings"].values())))
    lines.append("| 路线 | " + " | ".join(f"Recall@{n}" for n in header_ns) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in header_ns) + " |")
    for route, table in payload["ceilings"].items():
        cells = []
        for n in header_ns:
            row = table.get(str(n), {})
            cells.append(f"{row.get('rate', 0) * 100:.1f}%")
        lines.append(f"| `{route}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("> **这一行是精排的天花板**：gold 不在 N 条候选里，精排再准也救不回来。")
    lines.append("> 所以看精排效果之前必须先看它 —— 否则会把召回问题误判成精排问题。")
    lines.append("")
    if payload["n_sweep"]:
        lines.append("## 二、N → K 权衡（K 固定 5，纯向量路线）")
        lines.append("")
        lines.append("| N | 天花板 | 精排后 | 提升 | p50 | p95 | 超 2s 条数 |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for n, row in sorted(payload["n_sweep"].items(), key=lambda kv: int(kv[0])):
            gain = (row["rerank_recall"] - row["ceiling"]) * 100
            over = sum(1 for ms in row["latency_ms"] if ms > 2000)
            lines.append(
                f"| {n} | {row['ceiling'] * 100:.1f}% | **{row['rerank_recall'] * 100:.1f}%** | "
                f"{gain:+.1f}pp | {row['p50_ms']} ms | {row['p95_ms']} ms | "
                f"{over}/{len(row['latency_ms'])} |"
            )
        lines.append("")
    if payload["route_compare"]:
        lines.append("## 三、纯向量→精排 vs 混合→精排")
        lines.append("")
        lines.append("| 路线 | 说明 | 天花板 | 精排后 | p50 |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for route, row in payload["route_compare"].items():
            lines.append(
                f"| `{route}` | {row['description']} | {row['ceiling'] * 100:.1f}% | "
                f"**{row['rerank_recall'] * 100:.1f}%** | {row['p50_ms']} ms |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
