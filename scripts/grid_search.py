"""参数网格：**先粗后细**，7 次索引重建而不是 12 次。

## 为什么是 4+3 而不是 4×3

任务里定的纪律：

    粗筛：overlap 固定 50，只动 chunk ∈ {300,500,800,1000}   → 4 次
    细调：在最优 chunk 上，overlap ∈ {0,50,100}              → 3 次

全国网格 4×3=12 次，大部分信息是重复的：**overlap 对命中率的影响
远小于 chunk_chars**（前者只改 50 字的边界，后者改整块的粒度）。
先在 overlap=50 上找对数量级，再在那一档上细调 —— 这是 W6/W7 反复要用的纪律。

`overlap=50` 在两轮里都会被跑到，第二次直接复用缓存的向量，
所以真实成本是 4 + 2 次重建（不是 4+3）。

## 每组记录什么

任务要求的四项 + 两项必要的对照：

| 字段 | 为什么记 |
| --- | --- |
| `chunks` | 块数 —— 重建成本的代理 |
| `index_ms` | 索引耗时（切分 + 编码 + 入库，分开记） |
| `tokens_avg` | 平均 chunk token（bge tokenizer 数的，不是折算的） |
| `hit_at_5` / `mrr` | 命中率 —— 主指标 |
| `adjacent_pair_rate` | **重复内容率** —— overlap 唯一直接产生的坏处 |

## 为什么用 `InMemoryVectorStore` 而不是 Chroma

这个网格要跑 7 次重建 × 每份 400-700 块。Chroma 每次要建 collection、
写 sqlite、建 HNSW 索引 —— 那部分耗时会被算进 `index_ms`，
而它**与 chunk/overlap 无关**（是库的开销），混进来只会让"哪一组更快"变得不可解释。

内存库的代价：**索引耗时不含持久化**，所以 `index_ms` 只代表"切分 + 编码"。
这正是要测的两项，写清楚就好。持久化的开销另有 `chroma_acceptance.py` 在测。

## 缓存怎么用

`CachedEmbedder` 包住 bge（见 `app/rag/embedding/cache.py`）。
每组跑之前打印当轮命中率，跑完打印累计——**"缓存到底有没有生效"必须是数字，
不能是感觉**。20:25 那步的验收项"提速百分比"就是从这里来的：

    第一次重建（冷缓存）的 index_ms  vs  后续重建（热缓存）的 index_ms

运行：

    uv run python -X utf8 scripts/grid_search.py            # 粗筛 + 细调，全跑
    uv run python -X utf8 scripts/grid_search.py --stage coarse
    uv run python -X utf8 scripts/grid_search.py --stage fine
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
    evaluate_query,
    load_corpus,
    load_gold,
    summarize_outcomes,
)

DEFAULT_OUT: Path = PROJECT_ROOT / "scripts" / "grid_search.json"

COARSE_CHUNKS: tuple[int, ...] = (300, 500, 800, 1000)
COARSE_OVERLAP: int = 50
FINE_OVERLAPS: tuple[int, ...] = (0, 50, 100)


# top_k 用全项目唯一默认值，不在这里另定一个（见 `app/rag/retrieval.py`）。
def _default_top_k() -> int:
    from app.rag.retrieval import default_top_k

    return default_top_k()


def build_embedder(*, use_bge: bool) -> Any:  # noqa: ANN401
    """装配 embedder。**bge 才有语义，hashing 只用来跑通流程**。

    默认必须是 bge：`HashingEmbedder` 只有字面相似（见它的模块说明），
    用它做"命中率涨了多少"的判断，测的是字面重叠而不是语义检索 ——
    那在真实使用里几乎没有价值（用户不会用原文的词提问）。
    """
    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder

    if not use_bge:
        print("⚠ 用 hashing embedder：命中率数字**无参考价值**，只验证流程能跑通", file=sys.stderr)
        return get_embedder("hashing")
    inner = get_embedder("bge-small-zh")
    return CachedEmbedder(inner)


def run_one(
    *,
    corpus: Any,  # noqa: ANN401
    gold: list[Any],
    embedder: Any,  # noqa: ANN401
    tokenizer: Any,  # noqa: ANN401
    chunk_chars: int,
    overlap: int,
    top_k: int,
) -> dict[str, Any]:
    """跑一组参数：切分 → 向量化 → 入库 → 20 条查询 → 指标。"""
    from app.rag.chunking import ChunkingConfig
    from app.rag.stores.memory import InMemoryVectorStore

    cfg = ChunkingConfig(chunk_chars=chunk_chars, overlap_chars=overlap)

    # ① 切分
    started = time.perf_counter()
    chunks = chunk_corpus(corpus, cfg)
    split_ms = (time.perf_counter() - started) * 1000

    # ② 向量化（缓存层在这里起作用）
    stats_before = embedder.stats() if hasattr(embedder, "stats") else {}
    started = time.perf_counter()
    vectors = embedder.embed([chunk.text for chunk in chunks])
    embed_ms = (time.perf_counter() - started) * 1000
    stats_after = embedder.stats() if hasattr(embedder, "stats") else {}

    # ③ 入库
    started = time.perf_counter()
    store = InMemoryVectorStore()
    store.upsert(chunks, vectors)
    store_ms = (time.perf_counter() - started) * 1000

    # ④ 查询（查询向量**不缓存**，见 `embedding/cache.py` 的说明；
    #    但 20 条查询里同一个问题只算一次，所以这里是 20 次 embed_query）
    outcomes = []
    query_ms_total = 0.0
    for item in gold:
        started = time.perf_counter()
        query_vector = embedder.embed_query([item.question])[0]
        hits = store.search(query_vector, top_k=top_k)
        query_ms_total += (time.perf_counter() - started) * 1000
        outcomes.append(evaluate_query(hits, item))

    tokens = tokenizer.count([chunk.text for chunk in chunks])
    summary = summarize_outcomes(outcomes)

    hits_total_before = int(stats_before.get("hits", 0))
    hits_total_after = int(stats_after.get("hits", 0))
    miss_total = int(stats_after.get("misses", 0))

    return {
        "chunk_chars": chunk_chars,
        "overlap": overlap,
        "chunks": len(chunks),
        "chars_total": sum(len(chunk.text) for chunk in chunks),
        "tokens_total": sum(tokens),
        "tokens_avg": round(sum(tokens) / len(tokens), 1) if tokens else 0.0,
        "tokens_max": max(tokens) if tokens else 0,
        "tokens_over_512": sum(1 for value in tokens if value > 512),
        "split_ms": round(split_ms, 1),
        "embed_ms": round(embed_ms, 1),
        "store_ms": round(store_ms, 1),
        # `index_ms` = 切分 + 编码 + 入库。三项分开记是为了能回答"慢在哪一步"
        # （与 `indexing.IndexReport` 同口径）。
        "index_ms": round(split_ms + embed_ms + store_ms, 1),
        "query_ms_total": round(query_ms_total, 1),
        "cache_hits_this_run": hits_total_after - hits_total_before,
        "cache_misses_this_run": miss_total,
        "cache_hit_rate_this_run": round(
            (hits_total_after - hits_total_before)
            / max(1, (hits_total_after - hits_total_before) + miss_total),
            4,
        )
        if hasattr(embedder, "stats")
        else None,
        "hit_at_5": summary["hit_at_k"],
        "hits": summary["hits"],
        "mrr": summary["mrr"],
        "adjacent_pair_rate": summary["adjacent_pair_rate"],
        "per_query": summary["per_query"],
    }


def render(rows: list[dict[str, Any]], *, title: str, top_k: int) -> str:
    """人读表格。**列的顺序就是判断顺序**：先看命中率，再看成本。"""
    lines = [f"▓ {title}（top_k={top_k}，{len(rows[0]['per_query'])} 条 gold）"]
    lines.append(
        f"  {'chunk':>6} {'overlap':>8} {'块数':>6} {'平均tok':>8} {'越界':>5} "
        f"{'索引ms':>8} {'编码ms':>8} {'hit@5':>7} {'MRR':>7} {'重复率':>7}"
    )
    lines.append("  " + "-" * 92)
    for row in rows:
        lines.append(
            f"  {row['chunk_chars']:>6} {row['overlap']:>8} {row['chunks']:>6} "
            f"{row['tokens_avg']:>8} {row['tokens_over_512']:>5} "
            f"{row['index_ms']:>8} {row['embed_ms']:>8} "
            f"{row['hit_at_5'] * 100:>6.1f}% {row['mrr']:>7.3f} "
            f"{row['adjacent_pair_rate'] * 100:>6.1f}%"
        )
    return "\n".join(lines)


def bench_cache(
    *,
    corpus: Any,  # noqa: ANN401
    embedder: Any,  # noqa: ANN401
    chunk_chars: int,
    overlap: int,
) -> dict[str, Any]:
    """**缓存验证**：同一组参数连跑两次，对比"冷"与"热"的编码耗时。

    这是 21:25 那一步的验收项（"第二次重建应明显变快，记录时间差"）。

    ## 为什么必须"同一组参数跑两次"

    网格里相邻两组的块**内容完全不同**（300 与 500 切出来的文本没有一句是一样的），
    所以热缓存对第二组几乎没有帮助 —— 拿"第 2 组比第 1 组快"当缓存生效的证据，
    测到的是噪声。只有**同参数重跑**才是干净的对照：切分结果逐字节相同，
    `embed()` 的输入也就逐个相同，差异只可能来自缓存。

    ## 为什么还要单独测一次"清空后重建"

    "第二次快"有两种可能的成因：缓存真的命中了，或者只是 Python 进程热起来了
    （模型已加载、内存已分配、CPU 频率上来了）。这两者对后续工作的含义完全不同：
    前者说明"改参数重跑不用付编码费"，后者说明"只是这一次运气好"。

    所以这里测三次：

        ① 清掉这几条 key 后重建   → 冷缓存、热进程
        ② 立刻再重建             → 热缓存、热进程

    ①→② 的差值才是**缓存本身的贡献**。模型加载（17 s）早就在第一次 `embed`
    里付掉了，不进这个对比 —— 那部分成本与缓存无关，混进来会把提速比夸大到失真。
    """
    from app.rag.chunking import ChunkingConfig

    cfg = ChunkingConfig(chunk_chars=chunk_chars, overlap_chars=overlap)
    texts = [chunk.text for chunk in chunk_corpus(corpus, cfg)]

    cache = getattr(embedder, "cache", None)
    if cache is None:
        return {"supported": False, "reason": "embedder 没有缓存层（hashing 或未包裹）"}

    from app.rag.embedding.cache import cache_key  # noqa: PLC0415

    # ① 冷：把这批文本对应的 key 全删掉，再重建。
    keys = [
        cache_key(model=embedder.name, dimension=embedder.dimension, text=text) for text in texts
    ]
    deleted = 0
    with cache._lock:  # noqa: SLF001 - 实验脚本需要精确控制缓存状态，这是刻意的
        for key in keys:
            cur = cache._conn.execute("DELETE FROM vectors WHERE key = ?", (key,))
            deleted += cur.rowcount
        cache._conn.commit()  # noqa: SLF001
    cache.hits = 0
    cache.misses = 0

    started = time.perf_counter()
    embedder.embed(texts)
    cold_ms = (time.perf_counter() - started) * 1000
    cold_stats = {"hits": cache.hits, "misses": cache.misses}

    # ② 热：同样的输入，这次应该全部命中。
    cache.hits = 0
    cache.misses = 0
    started = time.perf_counter()
    embedder.embed(texts)
    hot_ms = (time.perf_counter() - started) * 1000
    hot_stats = {"hits": cache.hits, "misses": cache.misses}

    # ③ 外部再跑一次（模拟"另一次命令行重建"）：开一个新连接读同一个库。
    #    这一条才是"隔天重跑"的真实场景 —— 进程换了一个，库还在。
    from app.rag.embedding.cache import EmbeddingCache

    reopened = EmbeddingCache(cache.path, dimension=embedder.dimension)
    cold2 = reopened.get_many(keys)
    fresh_hits = sum(1 for vector in cold2 if vector is not None)
    reopened.close()

    speedup = round((cold_ms - hot_ms) / cold_ms, 4) if cold_ms > 0 else 0.0
    result = {
        "supported": True,
        "chunk_chars": chunk_chars,
        "overlap": overlap,
        "chunks": len(texts),
        "keys_deleted": deleted,
        "cold_ms": round(cold_ms, 1),
        "hot_ms": round(hot_ms, 1),
        "cold_stats": cold_stats,
        "hot_stats": hot_stats,
        "saved_ms": round(cold_ms - hot_ms, 1),
        # **这就是任务要的「提速百分比」**：相对冷缓存节省的比例。
        "speedup_ratio": speedup,
        "speedup_percent": round(speedup * 100, 1),
        "reopen_hit": fresh_hits,
        "reopen_miss": len(keys) - fresh_hits,
        "note": "冷/热都排除了模型加载（模型已在内存里），差值只归因于缓存本身",
    }
    print("\n▓ 缓存验证：同参数连跑两次")
    print(f"  块数 {len(texts)}，删除旧 key {deleted} 条")
    print(
        f"  ① 冷缓存：{result['cold_ms']} ms（命中 {cold_stats['hits']} / "
        f"未命中 {cold_stats['misses']}）"
    )
    print(
        f"  ② 热缓存：{result['hot_ms']} ms（命中 {hot_stats['hits']} / "
        f"未命中 {hot_stats['misses']}）"
    )
    print(f"  提速：{result['saved_ms']} ms，即 **{result['speedup_percent']}%**")
    print(f"  新连接复查：{fresh_hits}/{len(keys)} 条命中（模拟换一个进程重跑）")
    # 结论只在这里下判断，不在别处重复：命不中要看 key 设计。
    if hot_stats["misses"] > 0:
        print(
            f"  ⚠ 热缓存仍有 {hot_stats['misses']} 条未命中 —— 查 `cache_key` 的输入"
            "（模型名 / 维度 / 原文），常见成因是文本里有随调用变化的字段"
        )
    if fresh_hits != len(keys):
        print(
            f"  ⚠ 新连接只命中 {fresh_hits}/{len(keys)} —— 库路径漂移或写入未提交，"
            "查 `default_cache_path` 与 `put_many` 的 commit"
        )
    return result


def pick_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """选粗筛最优。**多指标按这个优先级**，因为单看命中率会被块数骗。

    1. `hit@5` 高者优
    2. 命中率相同 → `tokens_over_512` 少者优（越界 = 检索时被截断，等于白切）
    3. 还相同 → `chunks` 少者优（重建更便宜）
    4. 再相同 → `MRR` 高者优（排得更前）

    这条链写出来是因为"最优"必须有定义。只写"选最好的"，两个人跑同样的数据
    会选出不同的组，然后各自写的文档互相矛盾。
    """
    return min(
        rows,
        key=lambda row: (
            -row["hit_at_5"],
            row["tokens_over_512"],
            row["chunks"],
            -row["mrr"],
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="chunk × overlap 参数网格搜索")
    parser.add_argument("--stage", choices=["coarse", "fine", "cache", "all"], default="all")
    parser.add_argument("--embedder", choices=["bge", "hashing"], default="bge")
    parser.add_argument("--top-k", type=int, default=None, help="默认取全项目默认值")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--best-chunk",
        type=int,
        default=None,
        help="细调用哪个 chunk_chars（不给就先跑粗筛再自动选）",
    )
    args = parser.parse_args(argv)

    top_k = args.top_k if args.top_k is not None else _default_top_k()

    from app.rag.embedding.tokenizer import BGETokenizer

    corpus = load_corpus()
    for warning in corpus.warnings:
        print(f"  ⚠ {warning}", file=sys.stderr)
    gold = load_gold()
    embedder = build_embedder(use_bge=args.embedder == "bge")
    tokenizer = BGETokenizer()

    print(
        f"语料：{len(corpus.prose)} 份文档的正文（{corpus.total_chars} 字）"
        f" + {len(corpus.fixed)} 个表格/代码块；gold {len(gold)} 条；top_k={top_k}",
        file=sys.stderr,
    )

    payload: dict[str, Any] = {
        "meta": {
            "embedder": args.embedder,
            "top_k": top_k,
            "gold": len(gold),
            "corpus_prose_docs": len(corpus.prose),
            "corpus_fixed_docs": len(corpus.fixed),
            "corpus_chars": corpus.total_chars,
            "strategy": "recursive",
        },
        "coarse": [],
        "fine": [],
    }

    # ---------------------------------------------------------------- 粗筛
    if args.stage in ("coarse", "all"):
        print("\n【粗筛】overlap 固定 50，只动 chunk_chars", file=sys.stderr)
        for chunk_chars in COARSE_CHUNKS:
            row = run_one(
                corpus=corpus,
                gold=gold,
                embedder=embedder,
                tokenizer=tokenizer,
                chunk_chars=chunk_chars,
                overlap=COARSE_OVERLAP,
                top_k=top_k,
            )
            payload["coarse"].append(row)
            print(
                f"  ✓ chunk={chunk_chars} 块={row['chunks']} "
                f"hit@5={row['hit_at_5'] * 100:.1f}% 索引={row['index_ms']}ms",
                file=sys.stderr,
            )
        print()
        print(render(payload["coarse"], title="粗筛：chunk_chars 扫描（overlap=50）", top_k=top_k))
        best = pick_best(payload["coarse"])
        payload["best_chunk"] = best["chunk_chars"]
        print(
            f"\n  粗筛最优：chunk_chars={best['chunk_chars']}"
            f"（hit@5={best['hit_at_5'] * 100:.1f}%，越界={best['tokens_over_512']}，"
            f"块数={best['chunks']}）"
        )
    else:
        if args.best_chunk is None:
            print("细调需要 --best-chunk（或先跑粗筛）", file=sys.stderr)
            return 2
        payload["best_chunk"] = args.best_chunk

    best_chunk = args.best_chunk or payload.get("best_chunk")
    if best_chunk is None:
        print("无法确定细调用的 chunk_chars", file=sys.stderr)
        return 2
    if args.stage in ("fine", "all"):
        print(f"\n【细调】chunk_chars={best_chunk}，只动 overlap", file=sys.stderr)
        for overlap in FINE_OVERLAPS:
            row = run_one(
                corpus=corpus,
                gold=gold,
                embedder=embedder,
                tokenizer=tokenizer,
                chunk_chars=best_chunk,
                overlap=overlap,
                top_k=top_k,
            )
            payload["fine"].append(row)
            print(
                f"  ✓ overlap={overlap} 块={row['chunks']} "
                f"hit@5={row['hit_at_5'] * 100:.1f}% 重复率={row['adjacent_pair_rate'] * 100:.1f}%",
                file=sys.stderr,
            )
        print()
        print(
            render(
                payload["fine"],
                title=f"细调：overlap 扫描（chunk_chars={best_chunk}）",
                top_k=top_k,
            )
        )

    # ---------------------------------------------------------------- 缓存验证
    if args.stage in ("cache", "all") and hasattr(embedder, "stats"):
        payload["cache_bench"] = bench_cache(
            corpus=corpus,
            embedder=embedder,
            chunk_chars=best_chunk if best_chunk else 500,
            overlap=COARSE_OVERLAP,
        )

    # ---------------------------------------------------------------- 缓存统计
    if hasattr(embedder, "stats"):
        stats = embedder.stats()
        payload["cache"] = {**payload.get("cache", {}), **stats}
        print("\n▓ 缓存")
        print(f"  库：{stats['path']}")
        print(
            f"  条目 {stats['entries']}（{stats['bytes'] / 1024:.0f} KB）"
            f"  命中 {stats['hits']} / 未命中 {stats['misses']}"
            f"  总命中率 {stats['hit_rate'] * 100:.1f}%"
        )

    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘：{args.out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
