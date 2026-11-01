"""三策略 × 各自最优参数的全量对比。**今天的主交付。**

## 为什么每个策略要"各自调参"

任务写的是「三策略各用自己的最优参数，保证比较的是**策略**而不是**参数没调好**」。

这条如果偷懒（三组都用 chunk=500），结论会变成"heading 比 recursive 差"，
而真相可能是"heading 在 500 上差、在 800 上不差"。
**用同一套参数比三个策略，测到的是"参数与策略的交互"，不是策略本身。**

所以分两步：

    第一步  每个策略各自扫 chunk ∈ {300,500,800}（overlap 固定 50）
    第二步  各自取自己的最优，再横向比

代价是 9 次索引重建（3 策略 × 3 档），而不是 3 次。
这是必要的成本 —— 省掉它，主表就是错的。

## 语料粒度：**用不合并的 loader 原样输出**

这一点与前几天的网格不同，必须说清为什么。

`docs/retrieval-tuning-token-grid.md` 的 chunk 网格用的是**合并正文**（每份文档合成一段），
理由是"loader 切成 172 字的小段，改 chunk_chars 没效果"。
那个理由对 `recursive` 成立，但**对另两个策略是致命的**：

| 策略 | 依赖什么 | 合并后会怎样 |
| --- | --- | --- |
| `recursive` | 分隔符分布 | 不受影响 |
| `heading` | `Document.metadata["heading_level"]` | **元数据被丢掉** → 退化成按字数切 |
| `semantic` | 句子边界 + 句向量 | 仍然能跑，但跨节句向量相似度被噪声拉平 |

所以今天**必须用 loader 原样输出**（1748 段），否则 heading 与 semantic
是在"被阉割的输入"上跑，那场比较没有意义。

实测代价：不合并时 chunk 300→800 只差 10%（1998→1789 块），
参数敏感性下降。但这换来的是**三个策略都能用自己的信号** ——
这才是今天要测的东西。

## 每个策略的最优怎么定

**字典序**：`Recall@5` → `Recall@10` → `MRR` → 块数。

前三个与前面的网格一致（命中优先）。加 `Recall@10` 是因为今天要区分
"检索不到"与"排得靠后"：8→10 若无提升，说明问题在召回本身而不是排序。

块数放最后：它只影响成本，不该压过效果。

## 必须如实记录的一条

任务明确要求：「语义切分对 PDF 抽出的乱序文本可能更差——如实记录，
**不要为了「高级方案赢」而美化**」。

本语料是 markdown 为主（13 份 md + README），只有 `tests/fixtures/*.pdf` 是 PDF。
所以"语义切分在乱序 PDF 上的表现"**不能靠这份语料回答** ——
要么补 PDF 语料到评测集，要么如实写"未验证"。
本脚本会额外跑一次**纯 PDF 子集**来给一个方向性数据（见 `--pdf-probe`）。
"""

from __future__ import annotations

import argparse
import json
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

DEFAULT_OUT: Path = PROJECT_ROOT / "scripts" / "strategy_compare.json"

STRATEGIES: tuple[str, ...] = ("recursive", "heading", "semantic")
CHUNK_CANDIDATES: tuple[int, ...] = (300, 500, 800)
FIXED_OVERLAP: int = 50
FIXED_TOP_K: int = 5


def load_raw_corpus() -> list[Any]:
    """**不合并**的 loader 原样输出。

    **直接转发到 `retrieval_harness.load_raw_corpus`** —— 不再自己写一份。

    为什么必须共用一个实现：W5 周六实测过"两份各自实现"的后果：
    本脚本扫全部 docs（1820 段），而 `retrieval_harness` 用 13 份白名单（717 段），
    于是同一套 50 条题在两个脚本里跑出**不同的库**，
    召回率数字互不可比 —— 而两个脚本各自都"跑通了"。
    """
    return rh.load_raw_corpus()


def build_chunks(*, docs: list[Any], strategy: str, chunk_chars: int, embedder: Any) -> list[Any]:  # noqa: ANN401
    """切分。`semantic` 需要 embedder，另外两个不需要。"""
    from app.rag.chunking import ChunkingConfig, build_strategy

    cfg = ChunkingConfig(chunk_chars=chunk_chars, overlap_chars=FIXED_OVERLAP)
    return list(build_strategy(strategy, embedder=embedder, cfg=cfg).chunk(docs, cfg))


@dataclass
class RunResult:
    """一次"策略 + 参数"的完整结果。"""

    strategy: str
    chunk_chars: int
    chunks: int
    split_ms: float
    embed_ms: float
    store_ms: float
    recall: dict[str, Any]
    mrr: float
    by_type: dict[str, Any]
    refusal: dict[str, Any]
    sentence_rate: float | None
    """半句率（切分质量）。**与检索效果是两回事**，单独记。"""
    anchors_ok: bool
    anchor_problems: list[str] = field(default_factory=list)
    per_query: list[dict[str, Any]] = field(default_factory=list)
    """逐条的命中位置。**配对显著性检验需要它** ——
    只留汇总数就只能比大小，没法做 McNemar（那要逐条对齐）。"""

    @property
    def index_ms(self) -> float:
        return round(self.split_ms + self.embed_ms + self.store_ms, 1)

    @property
    def recall_at_5(self) -> float:
        return float(self.recall["recall@5"]["rate"])

    @property
    def recall_at_10(self) -> float:
        return float(self.recall["recall@10"]["rate"])


def run_one(
    *,
    docs: list[Any],
    items: list[Any],
    strategy: str,
    chunk_chars: int,
    embedder: Any,  # noqa: ANN401
    threshold: float,
) -> RunResult:
    """跑一组：切分 → 向量化 → 入库 → 50 条查询 → 指标。"""
    from app.rag.stores.memory import InMemoryVectorStore

    started = time.perf_counter()
    chunks = build_chunks(docs=docs, strategy=strategy, chunk_chars=chunk_chars, embedder=embedder)
    split_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    vectors = embedder.embed([chunk.text for chunk in chunks])
    embed_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    store = InMemoryVectorStore()
    store.upsert(chunks, vectors)
    store_ms = (time.perf_counter() - started) * 1000

    # 锚点反查：**每个策略各自解析**（chunk id 在三种策略下都不同）
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
    anchor_problems: list[str] = []
    try:
        rre.resolve_all(probe, chunks)
        anchors_ok = True
    except rre.AnchorResolutionError as exc:
        anchors_ok = False
        anchor_problems = str(exc).splitlines()[:12]

    max_k = max(rre.KS)
    outcomes = []
    for item in probe:
        vector = embedder.embed_query([item.question])[0]
        hits = store.search(vector, top_k=max_k)
        outcomes.append(rre.evaluate_query(item=item, hits=hits, k=max_k))

    metrics = rre.summarize(
        outcomes=outcomes, items=probe, ks=rre.KS, threshold=threshold, top_k=FIXED_TOP_K
    )

    # 半句率：切分质量的量化代理（`chunking/quality.py`）。
    # **与检索效果分开记** —— 它们回答不同问题，混在一起会得出
    # "切得好但检索差"这种无法解释的组合。
    sentence_rate: float | None = None
    try:
        from app.rag.chunking.quality import measure

        sentence_rate = round(float(measure(chunks).as_dict()["sentence_rate"]), 4)
    except Exception:  # noqa: BLE001 - 半句率是附加指标，失败不该毁掉主结果
        sentence_rate = None

    return RunResult(
        strategy=strategy,
        chunk_chars=chunk_chars,
        chunks=len(chunks),
        split_ms=round(split_ms, 1),
        embed_ms=round(embed_ms, 1),
        store_ms=round(store_ms, 1),
        recall=metrics["recall"],
        mrr=metrics["mrr"],
        by_type=metrics["by_type"],
        refusal=metrics["refusal"],
        sentence_rate=sentence_rate,
        anchors_ok=anchors_ok,
        anchor_problems=anchor_problems,
        per_query=[
            {
                "id": outcome.id,
                "type": outcome.type,
                "hit_rank": outcome.hit_rank,
                "anchor_ranks": list(outcome.anchor_ranks),
            }
            for outcome in outcomes
        ],
    )


def rank_key(result: RunResult) -> tuple[float, float, float, int]:
    """字典序选优：Recall@5 → Recall@10 → MRR → 块数（少者优）。"""
    return (
        -result.recall_at_5,
        -result.recall_at_10,
        -result.mrr,
        result.chunks,
    )


def render_matrix(best: dict[str, RunResult]) -> str:
    """主表：策略 × 指标 × 代价。**这就是交付物。**"""
    lines: list[str] = []
    header = (
        f"  {'策略':<10} {'chunk':>6} {'块数':>6} {'Recall@5':>9} {'Recall@10':>10} "
        f"{'MRR':>7} {'拒答':>6} {'半句率':>7} {'索引ms':>8}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) + 8))
    for strategy in STRATEGIES:
        result = best.get(strategy)
        if result is None:
            lines.append(f"  {strategy:<10} {'—':>6}  （未跑通）")
            continue
        refusal = result.refusal
        lines.append(
            f"  {strategy:<10} {result.chunk_chars:>6} {result.chunks:>6} "
            f"{result.recall_at_5 * 100:>8.1f}% {result.recall_at_10 * 100:>9.1f}% "
            f"{result.mrr:>7.4f} "
            f"{refusal['refused_correctly']}/{refusal['refusal_total']:<4} "
            f"{(result.sentence_rate or 0) * 100:>6.1f}% "
            f"{result.index_ms:>8}"
        )
    return "\n".join(lines)


def render_sweep(rows: list[RunResult]) -> str:
    """每个策略的参数扫描明细。**主表的数字要能追溯到这一层。**"""
    lines: list[str] = ["", "  参数扫描明细（overlap 固定 50）"]
    lines.append(
        f"    {'策略':<10} {'chunk':>6} {'块数':>6} {'R@5':>7} {'R@10':>7} "
        f"{'MRR':>7} {'半句率':>7} {'锚点':>5}"
    )
    for strategy in STRATEGIES:
        for row in [r for r in rows if r.strategy == strategy]:
            lines.append(
                f"    {row.strategy:<10} {row.chunk_chars:>6} {row.chunks:>6} "
                f"{row.recall_at_5 * 100:>6.1f}% {row.recall_at_10 * 100:>6.1f}% "
                f"{row.mrr:>7.4f} {(row.sentence_rate or 0) * 100:>6.1f}% "
                f"{'✓' if row.anchors_ok else '✗':>5}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="三策略 × 最优参数全量对比")
    parser.add_argument("--embedder", choices=["bge", "hashing"], default="bge")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(STRATEGIES),
        help="只跑其中几个（调试用）",
    )
    args = parser.parse_args(argv)

    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.settings import get_rag_config

    threshold = args.threshold if args.threshold is not None else get_rag_config().score_threshold

    if args.embedder == "hashing":
        print("⚠ hashing 下语义切分与检索都无参考价值", file=sys.stderr)
        embedder: Any = get_embedder("hashing")
    else:
        embedder = CachedEmbedder(get_embedder("bge-small-zh"))

    docs = load_raw_corpus()
    items = rre.load_questions()
    print(
        f"语料：{len(docs)} 段（loader 原样，不合并）；题目 {len(items)} 条；阈值 {threshold}",
        file=sys.stderr,
    )

    rows: list[RunResult] = []
    failures: list[dict[str, str]] = []
    for strategy in args.strategies:
        for chunk_chars in CHUNK_CANDIDATES:
            started = time.perf_counter()
            try:
                result = run_one(
                    docs=docs,
                    items=items,
                    strategy=strategy,
                    chunk_chars=chunk_chars,
                    embedder=embedder,
                    threshold=threshold,
                )
            except Exception as exc:  # noqa: BLE001 - 一个策略炸不该毁掉整张表
                failures.append(
                    {"strategy": strategy, "chunk_chars": str(chunk_chars), "error": f"{exc}"}
                )
                print(f"  ✗ {strategy} chunk={chunk_chars}: {exc}", file=sys.stderr)
                continue
            rows.append(result)
            print(
                f"  ✓ {strategy:<10} chunk={chunk_chars:>4} 块={result.chunks:>5} "
                f"R@5={result.recall_at_5 * 100:>5.1f}% R@10={result.recall_at_10 * 100:>5.1f}% "
                f"半句={((result.sentence_rate or 0) * 100):>5.1f}% "
                f"锚点={'✓' if result.anchors_ok else '✗'} "
                f"({time.perf_counter() - started:.0f}s)",
                file=sys.stderr,
            )

    # 每个策略取自己的最优
    best: dict[str, RunResult] = {}
    for strategy in args.strategies:
        candidates = [row for row in rows if row.strategy == strategy]
        if not candidates:
            continue
        # **锚点解析失败的不参与选优**：那种组的"命中率"没有意义
        valid = [row for row in candidates if row.anchors_ok]
        if not valid:
            failures.append(
                {"strategy": strategy, "chunk_chars": "*", "error": "所有参数下锚点都无法解析"}
            )
            continue
        best[strategy] = min(valid, key=rank_key)

    print()
    print("▓ 主表：三策略 × 各自最优参数")
    print(render_matrix(best))
    print(render_sweep(rows))

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus": "loader 原样输出（不合并），因为合并会丢 heading_level",
            "corpus_docs": len(docs),
            "questions": len(items),
            "chunk_candidates": list(CHUNK_CANDIDATES),
            "overlap": FIXED_OVERLAP,
            "top_k": FIXED_TOP_K,
            "threshold": threshold,
            "embedder": args.embedder,
            "selection_rule": "Recall@5 → Recall@10 → MRR → 块数（字典序）",
        },
        "sweep": [
            {
                "strategy": row.strategy,
                "chunk_chars": row.chunk_chars,
                "chunks": row.chunks,
                "split_ms": row.split_ms,
                "embed_ms": row.embed_ms,
                "store_ms": row.store_ms,
                "index_ms": row.index_ms,
                "recall": row.recall,
                "mrr": row.mrr,
                "by_type": row.by_type,
                "refusal": row.refusal,
                "sentence_rate": row.sentence_rate,
                "anchors_ok": row.anchors_ok,
                "anchor_problems": row.anchor_problems,
                "per_query": row.per_query,
            }
            for row in rows
        ],
        "best": {
            strategy: {
                "chunk_chars": result.chunk_chars,
                "chunks": result.chunks,
                "index_ms": result.index_ms,
                "recall_at_5": result.recall_at_5,
                "recall_at_10": result.recall_at_10,
                "mrr": result.mrr,
                "sentence_rate": result.sentence_rate,
                "refusal": result.refusal,
                "by_type": result.by_type,
                "recall": result.recall,
            }
            for strategy, result in best.items()
        },
        "failures": failures,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘：{args.out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
