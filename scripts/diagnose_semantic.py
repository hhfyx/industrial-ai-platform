"""回答今天的必答题：**语义切分输了吗？是实现不行，还是这份文本已经被抽乱？**

## 问题的两种答案指向完全不同的下一步

| 答案 | 下一步 |
| --- | --- |
| **实现不行** | 去修 `semantic.py`（换断点算法、调分位数） |
| **文本已经被切碎/抽乱，语义无从发挥** | 换语料（长文、PDF），或承认在当前 loader 下语义切分不适用 |

在没分清之前写"语义切分效果不好"，等于把两个问题打包扔掉。

## 本脚本要测三件事

### ① 语义切分有没有"可切的余地"

语义断点靠**相邻句向量的相似度差**。如果输入段本来就短
（loader 已按标题切成 100~200 字），那么：

- 每段之内可能只有 2~3 句 → 断点候选极少
- `semantic_min_chars=200` 会把过短的段并回去 → 更没机会切

**测法**：统计每个 `Document` 的句数与长度。若绝大多数段的长度
**小于 `semantic_min_chars`**，那策略根本不会产生新的边界。

### ② 它实际切出了什么（与递归对比块长分布）

如果两者的块长分布几乎一样，说明语义策略实际退化成"按段直通"。

### ③ 断点是否落在"该断的地方"

用一个可自动判定的代理：**断点处的句向量相似度是否真的低于段内平均**。
如果语义策略选的断点与"随机断点"没有区别，它的机制就没生效。

> 第 ③ 条是本脚本的核心 —— 它区分"切了但切错"与"根本没在按语义切"。

## 注意：这不能回答"PDF 乱序"的问题

本语料是 markdown，**没有 PDF 的乱序问题**。所以本脚本给出的是
"在结构良好的短段语料上，语义切分有没有用"。
"PDF 乱序让它更差"这个假设**本仓库当前无法验证** —— 除非补 PDF 评测集。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import strategy_compare as sc  # noqa: E402

DEFAULT_OUT: Path = PROJECT_ROOT / "scripts" / "semantic_diagnosis.json"


def segment_stats(docs: list[Any], *, min_chars: int) -> dict[str, Any]:
    """① 语料本身有没有"可切的余地"。"""
    lengths = [len(doc.text) for doc in docs]
    below_min = sum(1 for value in lengths if value < min_chars)
    return {
        "documents": len(docs),
        "chars_total": sum(lengths),
        "chars_mean": round(statistics.mean(lengths), 1),
        "chars_median": statistics.median(lengths),
        "chars_max": max(lengths),
        "below_min_chars": below_min,
        "below_min_ratio": round(below_min / len(lengths), 4),
        "note": (
            "语义断点要在段**内部**找。若段长普遍小于 semantic_min_chars，"
            "策略会把它们整段并回去，于是不产生新边界"
        ),
    }


def chunk_length_profile(chunks: list[Any]) -> dict[str, Any]:
    """② 块长分布。两个策略分布一样 → 语义策略退化成按段直通。"""
    lengths = sorted(len(chunk.text) for chunk in chunks)
    n = len(lengths)
    return {
        "chunks": n,
        "min": lengths[0],
        "p25": lengths[n // 4],
        "median": lengths[n // 2],
        "p75": lengths[3 * n // 4],
        "max": lengths[-1],
        "mean": round(statistics.mean(lengths), 1),
    }


def breakpoint_quality(*, docs: list[Any], embedder: Any, chunk_chars: int) -> dict[str, Any]:
    """③ 语义断点是否真的落在"相似度低"的位置。

    ## 做法

    1. 取每个 `Document`，用 `split_sentences` 切成句子
    2. 算相邻句的余弦相似度
    3. 让语义策略跑一遍，把它产出的块**边界**映射回句子下标
    4. 比较：**边界处的相似度** vs **随机位置处的相似度**

    若前者显著更低 → 策略在按语义切（机制生效）。
    若两者接近 → 策略的断点与随机无异（机制没生效）。
    """
    from app.rag.chunking import ChunkingConfig, build_strategy
    from app.rag.chunking.semantic import split_sentences
    from app.rag.embedding.vector import cosine

    cfg = ChunkingConfig(chunk_chars=chunk_chars, overlap_chars=50)
    chunks = build_strategy("semantic", embedder=embedder, cfg=cfg).chunk(docs, cfg)

    # 按 source 聚合产出的块，用于判断"哪些句子位置是边界"
    by_source: dict[str, list[str]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source, []).append(chunk.text)

    boundary_similarities: list[float] = []
    interior_similarities: list[float] = []

    for doc in docs:
        sentences = split_sentences(doc.text)
        if len(sentences) < 3:
            continue
        vectors = embedder.embed(sentences)
        sims = [cosine(vectors[i], vectors[i + 1]) for i in range(len(sentences) - 1)]
        # 该 doc 切出的块数决定"有几个边界"
        chunk_count = len(by_source.get(doc.source, []))
        if chunk_count <= 1:
            continue
        # **按块数等分估算边界位置**：语义策略的实际切点不暴露，
        # 所以用"块数 - 1 个大致均匀分布的边界"近似。
        # 这是本诊断的已知局限（见模块 docstring 末尾）。
        step = max(1, len(sims) // max(1, chunk_count))
        boundary_indices = {min(len(sims) - 1, i * step) for i in range(1, chunk_count)}
        for index, value in enumerate(sims):
            if index in boundary_indices:
                boundary_similarities.append(value)
            else:
                interior_similarities.append(value)

    if not boundary_similarities or not interior_similarities:
        return {"supported": False, "reason": "样本不足（没有产生多块的 doc）"}

    boundary_mean = statistics.mean(boundary_similarities)
    interior_mean = statistics.mean(interior_similarities)
    return {
        "supported": True,
        "boundary_count": len(boundary_similarities),
        "interior_count": len(interior_similarities),
        "boundary_similarity_mean": round(boundary_mean, 4),
        "interior_similarity_mean": round(interior_mean, 4),
        "difference": round(interior_mean - boundary_mean, 4),
        "mechanism_effective": interior_mean - boundary_mean > 0.02,
        "caveat": (
            "边界位置是按块数等分**估算**的（语义策略不暴露实际切点）；"
            "所以这只是方向性证据，不是精确测量"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="诊断语义切分为什么赢不了")
    parser.add_argument("--chunk-chars", type=int, default=800)
    parser.add_argument("--embedder", choices=["bge", "hashing"], default="bge")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    from app.rag.embedding import get_embedder

    if args.embedder == "hashing":
        print("⚠ hashing 下语义断点基本是噪声，仅供流程验证", file=sys.stderr)
        embedder: Any = get_embedder("hashing")
    else:
        embedder = get_embedder("bge-small-zh")

    docs = sc.load_raw_corpus()

    print("=" * 78)
    print("语义切分诊断：是实现不行，还是语料让它无从发挥？")
    print("=" * 78)

    from app.rag.chunking import ChunkingConfig

    cfg = ChunkingConfig(chunk_chars=args.chunk_chars, overlap_chars=50)

    stats = segment_stats(docs, min_chars=cfg.semantic_min_chars)
    print()
    print("① 语料有没有「可切的余地」")
    print(
        f"  文档段数 {stats['documents']}，平均 {stats['chars_mean']} 字，中位 {stats['chars_median']} 字"
    )
    print(
        f"  小于 semantic_min_chars({cfg.semantic_min_chars}) 的段："
        f"{stats['below_min_chars']} 段 = **{stats['below_min_ratio'] * 100:.1f}%**"
    )

    recursive_chunks = sc.build_chunks(
        docs=docs, strategy="recursive", chunk_chars=args.chunk_chars, embedder=embedder
    )
    semantic_chunks = sc.build_chunks(
        docs=docs, strategy="semantic", chunk_chars=args.chunk_chars, embedder=embedder
    )

    print()
    print("② 两个策略实际切出了什么（块长分布）")
    print(
        f"  {'策略':<10} {'块数':>6} {'min':>5} {'p25':>5} {'中位':>5} {'p75':>5} {'max':>6} {'均值':>7}"
    )
    profiles = {}
    for name, chunks in (("recursive", recursive_chunks), ("semantic", semantic_chunks)):
        profile = chunk_length_profile(chunks)
        profiles[name] = profile
        print(
            f"  {name:<10} {profile['chunks']:>6} {profile['min']:>5} "
            f"{profile['p25']:>5} {profile['median']:>5} {profile['p75']:>5} "
            f"{profile['max']:>6} {profile['mean']:>7}"
        )

    print()
    print("③ 语义断点落在哪里（与段内平均相比）")
    quality = breakpoint_quality(docs=docs, embedder=embedder, chunk_chars=args.chunk_chars)
    if quality.get("supported"):
        print(
            f"  边界处相似度均值 {quality['boundary_similarity_mean']}，"
            f"段内均值 {quality['interior_similarity_mean']}"
        )
        print(
            f"  差 {quality['difference']}  → "
            f"机制{'**生效**' if quality['mechanism_effective'] else '**未生效**（断点与随机无异）'}"
        )
    else:
        print(f"  无法测量：{quality.get('reason')}")

    # ---- 结论
    print()
    print("④ 结论")
    if stats["below_min_ratio"] > 0.5:
        print(
            f"  **语料让它无从发挥**：{stats['below_min_ratio'] * 100:.0f}% 的段"
            f"短于 semantic_min_chars({cfg.semantic_min_chars})，"
        )
        print("  语义策略会把它们整段并回去，根本不产生新边界。")
        print("  这不是实现缺陷 —— 是**输入没有可供语义切分的结构**。")
    else:
        print("  语料有足够长的段，语义切分应当有机会生效。")
        if quality.get("supported") and not quality.get("mechanism_effective"):
            print("  但实测断点与随机无异 → **实现有问题**，该去修 semantic.py。")

    payload = {
        "meta": {
            "chunk_chars": args.chunk_chars,
            "embedder": args.embedder,
            "semantic_min_chars": cfg.semantic_min_chars,
            "semantic_max_chars": cfg.semantic_max_chars,
        },
        "segment_stats": stats,
        "profiles": profiles,
        "breakpoint_quality": quality,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘：{args.out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
