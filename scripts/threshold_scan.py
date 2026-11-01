"""阈值粗标：命中 vs 未命中的分数分布，找分界点。

## 先说方向：这个代码库里是 `score`，不是 `distance`

任务的写法是「`distance > threshold` → 拒答」—— 那是**距离**口径（越小越像）。
本项目在 W4 就把口径统一成了**相似度**（`stores/base.SearchHit.score`，
越大越像，范围 [-1, 1]），`ChromaStore` 内部做 `1 - distance`。

所以这里的判据是 **`score < threshold` → 拒答**，方向相反。这不是笔误，
而是换库时"两种口径混在一个类型里"那类 bug 的源头（见 `stores/base.py` 的说明）。
**改这个方向不会报错，只会让拒答逻辑整体反转** —— 越不像的越会被答。

## 两组样本，缺一不可

| 组 | 条数 | 来源 | 它回答什么 |
| --- | ---: | --- | --- |
| 应命中 | 20 | `eval/retrieval-gold.jsonl` | 有答案时分数落在哪 |
| 应拒答 | 5 | `eval/refusal-gold.jsonl` | 没答案时分数落在哪 |

只测前者会得到"分数都很高，阈值随便取"的错误结论 ——
**阈值的好坏完全由后者的分数决定**，因为它是用来挡"库里没有"的。

## 为什么画分布图而不只报一个数

单个阈值（比如"0.4"）无法表达三件事：

1. **两组的重叠区间有多宽** —— 重叠越宽，任何阈值都必然同时误拒与误放
2. **取不同阈值时各错多少** —— 这是 ROC 的形状
3. **有没有一条干净的分界线** —— 有则阈值好取，没有则说明模型/语料有问题

W4 的实测（`embedding/base.py`）已经给过一次警告：**分数不能区分命中与未命中**，
未命中的两条最高分（0.5087 / 0.6514）比命中者的最低分（0.466）还高。
今天要把这个警告放大到 25 条样本上看清楚。

运行：

    uv run python -X utf8 scripts/threshold_scan.py
"""

from __future__ import annotations

import argparse
import json
import sys
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

DEFAULT_OUT: Path = PROJECT_ROOT / "scripts" / "threshold_scan.json"
REFUSAL_GOLD: Path = PROJECT_ROOT / "eval" / "refusal-gold.jsonl"

# 与前面保持一致：今天只动阈值，切分与 k 都钉死。
FIXED_CHUNK_CHARS: int = 500
FIXED_OVERLAP: int = 50
FIXED_TOP_K: int = 5

# 候选阈值。**刻意密集在 0.3~0.6**：那是昨天探针给的相关/无关间隔区，
# 而粗标就是在这个区间里看哪一档的分错最少。
CANDIDATES: tuple[float, ...] = (
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
)

# 直方图分桶（相似度）。**在 0.4~0.6 之间必须细分** —— 决策全落在那里。
BUCKETS: tuple[tuple[float, float], ...] = (
    (0.0, 0.10),
    (0.10, 0.20),
    (0.20, 0.30),
    (0.30, 0.40),
    (0.40, 0.45),
    (0.45, 0.50),
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 1.00),
)


def load_refusal_gold(path: Path = REFUSAL_GOLD) -> list[dict[str, str]]:
    """读"知识库里没有"的问题集。

    与 `retrieval-gold.jsonl` 的区别：**它没有 anchor**，因为正确答案就是"没有"。
    这也是它唯一需要的字段 —— 加 anchor 反而会诱导去写一条其实不存在的依据。
    """
    if not path.exists():
        raise FileNotFoundError(f"拒答样本集不存在：{path}")
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        rows.append({"id": raw["id"], "question": raw["question"], "why": raw.get("why", "")})
    if not rows:
        raise ValueError(f"{path} 是空的")
    return rows


def build_index(*, embedder: Any) -> Any:  # noqa: ANN401
    from app.rag.chunking import ChunkingConfig
    from app.rag.stores.memory import InMemoryVectorStore

    corpus = load_corpus()
    for warning in corpus.warnings:
        print(f"  ⚠ {warning}", file=sys.stderr)
    cfg = ChunkingConfig(chunk_chars=FIXED_CHUNK_CHARS, overlap_chars=FIXED_OVERLAP)
    chunks = chunk_corpus(corpus, cfg)
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))
    return store, chunks


def score_one(*, question: str, store: Any, embedder: Any) -> tuple[float, str]:  # noqa: ANN401
    """返回 (top1 相似度, top1 所属文档)。**只看 top1** —— 阈值作用在最高分上。

    为什么不看均值或中位数：拒答判据是 `max(scores) < threshold`（见
    `generation.base.generate_answer` 的门 2）。用别的统计量会和实际逻辑不一致，
    那条不一致会让"离线调好的阈值"上线后行为不同。
    """
    vector = embedder.embed_query([question])[0]
    hits = store.search(vector, top_k=FIXED_TOP_K)
    if not hits:
        return 0.0, ""
    return hits[0].score, hits[0].chunk.source


def histogram(positive: list[float], negative: list[float]) -> list[dict[str, Any]]:
    """两组各自的分桶。**必须叠在一起看** —— 分开看会漏掉重叠区间。"""
    rows: list[dict[str, Any]] = []
    for low, high in BUCKETS:
        pos = sum(1 for value in positive if low <= value < high)
        neg = sum(1 for value in negative if low <= value < high)
        rows.append(
            {
                "bucket": f"{low:.2f}-{high:.2f}",
                "low": low,
                "high": high,
                "hit": pos,
                "miss": neg,
                # 用字符画让形状直接可见：█ 是有答案，░ 是无答案
                "bar_hit": "█" * pos,
                "bar_miss": "░" * neg,
            }
        )
    return rows


def sweep(positive: list[float], negative: list[float]) -> list[dict[str, Any]]:
    """对每个候选阈值算"误拒/误放"。

    口径（**必须写清，否则"准确率"这个词会被各人各用**）：

    - **误拒**（false refusal）：有答案但分数 < 阈值 → 该答却拒了
    - **误放**（false accept）：没答案但分数 >= 阈值 → 该拒却答了

    两者代价不对称：误拒是"用户看到的服务失败"，误放是"模型拿着不相关材料硬答"
    （W4 的 F3 引用越界就是这么来的）。所以本表**不合成一个准确率**，
    让人各自按自己的立场读 —— 合成一个数会把这个权衡藏起来。
    """
    rows: list[dict[str, Any]] = []
    for threshold in CANDIDATES:
        false_refusal = sum(1 for value in positive if value < threshold)
        false_accept = sum(1 for value in negative if value >= threshold)
        rows.append(
            {
                "threshold": threshold,
                "false_refusal": false_refusal,
                "false_refusal_rate": round(false_refusal / len(positive), 4) if positive else 0.0,
                "false_accept": false_accept,
                "false_accept_rate": round(false_accept / len(negative), 4) if negative else 0.0,
                "errors": false_refusal + false_accept,
            }
        )
    return rows


def find_boundary(positive: list[float], negative: list[float]) -> dict[str, Any]:
    """找分界点。

    两种"分界"必须分开讲，否则会得出一个假结论：

    1. **完全可分**：`max(negative) < min(positive)` —— 存在一段空隙，
       阈值落在空隙里就零错误。这是最好的情况。
    2. **有重叠**：两组交错。此时"最优阈值"只能是**错误数最少**的那个，
       而且**必然有错**。把重叠时的最优点说成"调好了"，是在骗自己。
    """
    if not positive or not negative:
        return {"separable": False, "reason": "样本不足"}
    pos_min, pos_max = min(positive), max(positive)
    neg_min, neg_max = min(negative), max(negative)
    separable = neg_max < pos_min
    return {
        "separable": separable,
        "positive_min": round(pos_min, 4),
        "positive_max": round(pos_max, 4),
        "negative_min": round(neg_min, 4),
        "negative_max": round(neg_max, 4),
        # 重叠区间：负样本最高的那批 vs 正样本最低的那批。
        "overlap_low": round(neg_min if separable else max(neg_min, pos_min), 4),
        "overlap_high": round(neg_max if separable else min(neg_max, pos_max), 4),
        "gap": round(pos_min - neg_max, 4) if separable else None,
        "current_default": 0.40,
        "note": (
            "两组完全可分，阈值落在空隙里即可零错误"
            if separable
            else "两组有重叠，任何阈值都必然有错；表里最优只是错误数最少"
        ),
    }


def render(
    *,
    positive: list[tuple[str, float, str]],
    negative: list[tuple[str, float, str]],
    hist: list[dict[str, Any]],
    table: list[dict[str, Any]],
    boundary: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append(
        f"阈值粗标（chunk={FIXED_CHUNK_CHARS}/overlap={FIXED_OVERLAP}/top_k={FIXED_TOP_K}）"
    )
    lines.append(
        f"  样本：应命中 {len(positive)} 条，应拒答 {len(negative)} 条"
        "（score 是**相似度**，越大越像）"
    )

    lines.append("")
    lines.append("  分数分布（█ = 有答案，░ = 无答案）")
    for row in hist:
        if not (row["hit"] or row["miss"]):
            continue
        lines.append(
            f"    {row['bucket']}  █{row['hit']:<2} {row['bar_hit']:<4} "
            f"░{row['miss']:<2} {row['bar_miss']}"
        )

    lines.append("")
    lines.append("  逐条明细")
    lines.append(f"    {'应命中':<10} {'score':>7}  来源")
    for query_id, value, source in sorted(positive, key=lambda row: row[1]):
        lines.append(f"    {query_id:<10} {value:>7.4f}  {source}")
    lines.append(f"    {'应拒答':<10} {'score':>7}  top1 来源（应当无关）")
    for query_id, value, source in sorted(negative, key=lambda row: -row[1]):
        lines.append(f"    {query_id:<10} {value:>7.4f}  {source}")

    lines.append("")
    lines.append("  阈值扫描（误拒 = 该答却拒；误放 = 该拒却答）")
    lines.append(
        f"    {'阈值':>6} {'误拒':>6} {'误拒率':>8} {'误放':>6} {'误放率':>8} {'合计错':>7}"
    )
    for row in table:
        mark = "  ← 当前默认" if abs(row["threshold"] - 0.40) < 1e-9 else ""
        lines.append(
            f"    {row['threshold']:>6.2f} {row['false_refusal']:>6} "
            f"{row['false_refusal_rate'] * 100:>7.1f}% {row['false_accept']:>6} "
            f"{row['false_accept_rate'] * 100:>7.1f}% {row['errors']:>7}{mark}"
        )

    lines.append("")
    lines.append("  分界点")
    if boundary.get("separable"):
        lines.append(
            f"    ✓ 完全可分：无答案最高 {boundary['negative_max']} < "
            f"有答案最低 {boundary['positive_min']}，空隙 {boundary['gap']}"
        )
    else:
        lines.append(
            f"    ✗ 有重叠：有答案最低 {boundary.get('positive_min')}，"
            f"无答案最高 {boundary.get('negative_max')}"
        )
        lines.append(
            f"      重叠区间 [{boundary.get('overlap_low')}, {boundary.get('overlap_high')}]"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阈值粗标：命中 vs 未命中的分数分布")
    parser.add_argument("--embedder", choices=["bge", "hashing"], default="bge")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder

    if args.embedder == "hashing":
        print("⚠ hashing 下分数分布没有可分性（见 embedding/base.py），仅验证流程", file=sys.stderr)
        embedder: Any = get_embedder("hashing")
    else:
        embedder = CachedEmbedder(get_embedder("bge-small-zh"))

    store, chunks = build_index(embedder=embedder)
    gold = load_gold()
    refusal = load_refusal_gold()
    print(f"索引 {len(chunks)} 块，应命中 {len(gold)}，应拒答 {len(refusal)}", file=sys.stderr)

    positive = [
        (item.id, *score_one(question=item.question, store=store, embedder=embedder))
        for item in gold
    ]
    negative = [
        (row["id"], *score_one(question=row["question"], store=store, embedder=embedder))
        for row in refusal
    ]

    pos_scores = [value for _id, value, _source in positive]
    neg_scores = [value for _id, value, _source in negative]
    hist = histogram(pos_scores, neg_scores)
    table = sweep(pos_scores, neg_scores)
    boundary = find_boundary(pos_scores, neg_scores)

    print()
    print(render(positive=positive, negative=negative, hist=hist, table=table, boundary=boundary))

    best = min(table, key=lambda row: (row["errors"], -row["threshold"]))
    print()
    print(
        f"  错误最少的阈值：{best['threshold']}（误拒 {best['false_refusal']} + "
        f"误放 {best['false_accept']} = {best['errors']} 错 / {len(pos_scores) + len(neg_scores)} 条）"
    )
    print("  ⚠ 20+5 条样本：1 条 = 4pp。这个最优点**仅供周五 50 条集参考**，不当定论。")

    payload = {
        "meta": {
            "chunk_chars": FIXED_CHUNK_CHARS,
            "overlap": FIXED_OVERLAP,
            "top_k": FIXED_TOP_K,
            "embedder": args.embedder,
            "positive": len(pos_scores),
            "negative": len(neg_scores),
            "score_direction": "similarity（越大越像）；判据是 score < threshold 才拒答",
            "chunks": len(chunks),
        },
        "positive": [{"id": i, "score": round(v, 4), "top1_source": s} for i, v, s in positive],
        "negative": [{"id": i, "score": round(v, 4), "top1_source": s} for i, v, s in negative],
        "histogram": hist,
        "sweep": table,
        "boundary": boundary,
        "best": best,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘：{args.out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
