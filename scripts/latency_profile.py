"""延迟打点：**四段占比 + 优化前后对比**（W5 F7 那道题的现实版）。

## 为什么要单独一个脚本

W5 的 F7 问的是"首字延迟这个指标的口径怎么定"，答案是"从发起请求到第一个非空 content"。
今天是那道题的**现实版**：把检索链路拆成段，看钱花在哪。

W6 加了三层（BM25 / 精排 / 压缩）之后，不拆段的后果是：

    检索耗时 17.9 秒   ← 这个数字没法归因

拆段之后：

    embed   10 ms
    search   1 ms
    bm25     6 ms
    rerank  17676 ms   ← 99% 在这里

**只有拆开，才知道该优化哪一段。** 这也是 W6 周四"精排不进链路"那个决定的依据。

## 三个产出

| 产出 | 看什么 |
| --- | --- |
| **四段占比** | 每段占全链路的百分比 |
| **P50 / P95** | 20 条 query 连跑（任务书要求）；P95 比 P50 更能反映"卡顿感" |
| **优化前后对比** | 缓存命中 / 候选缩小 / 跳过精排 的效果 |

## 为什么要有 P95

P50 是"一半的请求比这快"，P95 是"95% 的请求比这快"。
用户感受到的"卡"来自后者 —— 平均快没有意义，如果每 20 次就有一次卡 30 秒。

## 优化项

| 优化 | 做法 | 预期 |
| --- | --- | --- |
| **embedding 缓存** | 同一 query 第二次直接命中 | 省 `embed_ms` |
| **精排结果缓存** | key = query + 候选集合哈希 | 省 `rerank_ms`（大头） |
| **候选缩小** | N=20 → N=10 | 精排时间约减半，但要记指标损失 |
| **跳过精排** | 直接返回混合结果 | 省 100% 精排，但 Recall@5 掉 15pp |

**最后一项不能只看延迟** —— 它是拿 15pp 召回换 17 秒。
脚本会同时输出"每省 1 秒付出多少 pp"，那个数才是决策依据。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
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

OUT_JSON: Path = PROJECT_ROOT / "scripts" / "latency_profile.json"
OUT_MD: Path = PROJECT_ROOT / "scripts" / "latency_profile.md"

#: 任务书要求"20 条 query 连跑"。
DEFAULT_QUERIES: int = 20

TOP_K: int = 5


@dataclass
class StageTiming:
    """一条 query 的分段耗时。**与 `QATiming` 的字段名保持一致**，
    这样打点结果能直接填进那个结构。"""

    embed_ms: int = 0
    search_ms: int = 0
    bm25_ms: int = 0
    rerank_ms: int = 0
    compress_ms: int = 0

    @property
    def chain_ms(self) -> int:
        return self.embed_ms + self.search_ms + self.bm25_ms + self.rerank_ms + self.compress_ms

    def as_dict(self) -> dict[str, int]:
        return {
            "embed_ms": self.embed_ms,
            "search_ms": self.search_ms,
            "bm25_ms": self.bm25_ms,
            "rerank_ms": self.rerank_ms,
            "compress_ms": self.compress_ms,
            "chain_ms": self.chain_ms,
        }


@dataclass
class RunResult:
    """一轮（某个配置）的完整记录。"""

    name: str
    note: str
    timings: list[StageTiming] = field(default_factory=list)
    recall_at_5: float = 0.0
    hits: int = 0

    def p(self, percentile: float, field_name: str) -> int:
        """取某一段的百分位。**用 `statistics` 而不是手写索引** ——
        手写 `int(n*0.95)` 在 n 小的时候会越界或取到最大值。"""
        values = [getattr(t, field_name) for t in self.timings]
        if not values:
            return 0
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(len(ordered) * percentile))
        return ordered[index]

    @property
    def p50_chain(self) -> int:
        return self.p(0.50, "chain_ms")

    @property
    def p95_chain(self) -> int:
        return self.p(0.95, "chain_ms")

    def mean_stage(self, field_name: str) -> float:
        if not self.timings:
            return 0.0
        return statistics.mean(getattr(t, field_name) for t in self.timings)

    def shares(self) -> dict[str, float]:
        """各段占全链路的比例。**这是"钱花在哪"的直接答案。**"""
        total = sum(
            self.mean_stage(f)
            for f in ("embed_ms", "search_ms", "bm25_ms", "rerank_ms", "compress_ms")
        )
        if total <= 0:
            return {}
        return {
            f: round(self.mean_stage(f) / total, 4)
            for f in ("embed_ms", "search_ms", "bm25_ms", "rerank_ms", "compress_ms")
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "note": self.note,
            "queries": len(self.timings),
            "recall_at_5": round(self.recall_at_5, 4),
            "hits": self.hits,
            "p50_chain_ms": self.p50_chain,
            "p95_chain_ms": self.p95_chain,
            "mean_stage_ms": {
                f: round(self.mean_stage(f), 1)
                for f in ("embed_ms", "search_ms", "bm25_ms", "rerank_ms", "compress_ms")
            },
            "shares": self.shares(),
        }


def _hit_at(ids: list[str], gold: list[str], k: int) -> bool:
    if not gold:
        return False
    pos = {cid: i for i, cid in enumerate(ids, start=1)}
    ranks = [pos.get(gid) for gid in gold]
    return all(r is not None and r <= k for r in ranks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="延迟打点与优化对比")
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    cfg = get_rag_config()
    docs = rh.load_raw_corpus()
    chunk_cfg = ChunkingConfig(chunk_chars=cfg.chunk_chars, overlap_chars=cfg.overlap_chars)
    chunks = build_strategy(cfg.strategy, cfg=chunk_cfg).chunk(docs, chunk_cfg)
    print(f"语料 {len(docs)} 段；索引 {len(chunks)} 块", file=sys.stderr)

    items = rre.load_questions(rre.EVAL_FILE)
    answerable = [item for item in items if not item.is_refusal][: args.queries]
    rre.resolve_all(items, chunks)
    chunks_by_id = {chunk.id: chunk for chunk in chunks}

    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.stores.memory import InMemoryVectorStore

    embedder = CachedEmbedder(get_embedder("bge-small-zh"))
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))
    bm25, _ = BM25Retriever.from_chunks(chunks, path=DEFAULT_INDEX_PATH)
    reranker = Reranker(model_path=args.model_dir)
    load_started = time.perf_counter()
    reranker.load()
    print(f"精排模型加载 {int((time.perf_counter() - load_started) * 1000)} ms", file=sys.stderr)

    fusion_cfg = FusionConfig(mode=FusionMode.RRF, candidates=20, rrf_k=DEFAULT_RRF_K)

    def _ms(started: float) -> int:
        return max(int((time.perf_counter() - started) * 1000), 1)

    def _run(name: str, note: str, *, candidates: int, do_rerank: bool) -> RunResult:
        result = RunResult(name=name, note=note)
        # W13 周一：评测脚本也开 trace（rag.bm25 / rag.fusion / rag.rerank /
        # rag.compress 四层的 span 只在这里出现——在线链路默认关）。
        from app.core.tracing import span as trace_span
        from app.core.tracing import trace_scope

        for item in answerable:
            with trace_scope(
                root_name="rag.recall_chain",
                root_attrs={"profile": name, "query_head": item.question[:60]},
            ):
                timing = StageTiming()
                started = time.perf_counter()
                with trace_span("rag.embed", embedder="bge-small-zh"):
                    vector = embedder.embed_query([item.question])[0]
                timing.embed_ms = _ms(started)

                started = time.perf_counter()
                with trace_span("rag.vector", top_k=candidates):
                    vec_hits = store.search(vector, top_k=candidates)
                timing.search_ms = _ms(started)

                started = time.perf_counter()
                with trace_span("rag.bm25", top_k=candidates):
                    bm25_hits = bm25.retrieve(item.question, top_k=candidates)
                timing.bm25_ms = _ms(started)

                with trace_span("rag.fusion", mode=fusion_cfg.label()):
                    fused, _ = merge_rankings(
                        [hit.chunk.id for hit in vec_hits],
                        [hit.chunk.id for hit in bm25_hits],
                        config=fusion_cfg,
                    )
                ids = [hit.chunk_id for hit in fused]

                if do_rerank:
                    started = time.perf_counter()
                    with trace_span("rag.rerank", candidates=candidates):
                        candidates_list = [
                            (cid, chunks_by_id[cid].text) for cid in ids if cid in chunks_by_id
                        ]
                        scores, _report = reranker.rerank(
                            item.question, candidates_list, top_k=candidates
                        )
                    timing.rerank_ms = _ms(started)
                    ids = [score.chunk_id for score in scores]

            hit = _hit_at(ids, item.gold_chunk_ids, TOP_K)
            result.hits += int(hit)
            result.timings.append(timing)

        result.recall_at_5 = result.hits / len(answerable) if answerable else 0.0
        return result

    # ---------- 三档配置 ----------
    print("\n跑配置 A（候选 20 + 精排，当前链路）…", file=sys.stderr)
    run_full = _run("A_候选20_精排", "当前链路", candidates=20, do_rerank=True)

    print("跑配置 B（候选 10 + 精排，缩小 N）…", file=sys.stderr)
    run_small = _run("B_候选10_精排", "缩小候选 N", candidates=10, do_rerank=True)

    print("跑配置 C（候选 20 不精排）…", file=sys.stderr)
    run_norank = _run("C_候选20_无精排", "跳过精排", candidates=20, do_rerank=False)

    runs = [run_full, run_small, run_norank]

    # ---------- 输出 ----------
    print()
    print("=" * 100)
    print(f"延迟打点（{len(answerable)} 条 query，k={TOP_K}）")
    print("=" * 100)
    print(f"  {'配置':<20}{'P50':>9}{'P95':>9}{'R@5':>8}{'命中':>7}  说明")
    print("  " + "-" * 96)
    for run in runs:
        print(
            f"  {run.name:<20}{run.p50_chain:>8}ms{run.p95_chain:>8}ms"
            f"{run.recall_at_5 * 100:>7.1f}%{run.hits:>5}/{len(answerable):<3}  {run.note}"
        )

    print()
    print("  四段占比（配置 A：当前链路）：")
    shares = run_full.shares()
    for stage, label in (
        ("embed_ms", "query 编码"),
        ("search_ms", "向量检索"),
        ("bm25_ms", "BM25"),
        ("rerank_ms", "精排"),
        ("compress_ms", "压缩"),
    ):
        share = shares.get(stage, 0.0)
        mean = run_full.mean_stage(stage)
        bar = "█" * int(share * 40)
        print(f"    {label:<10}{mean:>8.1f}ms{share * 100:>7.1f}%  {bar}")

    print()
    print("  优化前后对比：")
    base = run_full
    for run in runs[1:]:
        dp50 = run.p50_chain - base.p50_chain
        d95 = run.p95_chain - base.p95_chain
        drecall = (run.recall_at_5 - base.recall_at_5) * 100
        saved_s = -dp50 / 1000 if dp50 < 0 else 0
        cost_pp = -drecall if drecall < 0 else 0
        per = f"{cost_pp / saved_s:.1f} pp/秒" if saved_s > 0 and cost_pp > 0 else "无损失"
        print(
            f"    {run.name:<20} P50 {dp50:+6}ms  P95 {d95:+6}ms  "
            f"R@5 {drecall:+5.1f}pp   每省 1 秒付出 {per}"
        )

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "queries": len(answerable),
            "chunks": len(chunks),
            "top_k": TOP_K,
            "device": "cpu",
        },
        "runs": [run.as_dict() for run in runs],
    }
    if args.no_write:
        return 0
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"\n落盘：{OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"      {OUT_MD.relative_to(PROJECT_ROOT)}")
    return 0


def _render_markdown(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    runs = payload["runs"]
    lines: list[str] = []
    lines.append("# 延迟打点与优化对比")
    lines.append("")
    lines.append(
        f"> {meta['generated_at']} ｜ {meta['queries']} 条 query "
        f"｜ {meta['chunks']} 块 ｜ k={meta['top_k']} ｜ device={meta['device']}"
    )
    lines.append("")
    lines.append("## 一、三档配置")
    lines.append("")
    lines.append("| 配置 | P50 | P95 | Recall@5 | 命中 | 说明 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for run in runs:
        lines.append(
            f"| `{run['name']}` | {run['p50_chain_ms']} ms | **{run['p95_chain_ms']} ms** | "
            f"{run['recall_at_5'] * 100:.1f}% | {run['hits']}/{run['queries']} | {run['note']} |"
        )
    lines.append("")
    lines.append("## 二、四段占比（配置 A）")
    lines.append("")
    lines.append("| 段 | 均值 | 占比 |")
    lines.append("| --- | ---: | ---: |")
    labels = {
        "embed_ms": "query 编码",
        "search_ms": "向量检索",
        "bm25_ms": "BM25",
        "rerank_ms": "精排",
        "compress_ms": "压缩",
    }
    shares = runs[0]["shares"]
    for key, label in labels.items():
        lines.append(
            f"| {label} | {runs[0]['mean_stage_ms'][key]} ms | {shares.get(key, 0) * 100:.1f}% |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
