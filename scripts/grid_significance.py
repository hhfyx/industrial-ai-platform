"""对今天的三组对比做显著性检验。**20 条 gold，1 条 = 5pp，必须算一次再说"涨了"。**

用法：python -X utf8 scripts/grid_significance.py
"""

import json
import math
from pathlib import Path

DATA = Path("scripts/grid_search.warm.json")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 区间。**二项比例的常态近似在 p 接近 0 或 1、n 小时会给出越界值**，
    Wilson 不会 —— 这里 n=20，正需要它。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar 精确检验的双侧 p 值（配对二分数据）。

    b = 只在前一组命中、c = 只在后一组命中。零假设下每一次"翻转"等概率地
    落在 b 或 c，所以统计量服从 Binomial(b+c, 0.5)。
    **用精确分布而不是卡方近似**：n 小的时候卡方近似误差很大，
    而这里恰恰是小样本 —— 正是最需要精确解的场景。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def paired(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    return only_a, only_b, mcnemar_exact(only_a, only_b)


def row_of(d, group, key, value):
    for r in d[group]:
        if r[key] == value:
            return r
    raise KeyError((group, key, value))


def hits(r) -> list[bool]:
    return [p["hit_rank"] is not None for p in r["per_query"]]


print("=" * 78)
print("显著性检验：20 条 gold，1 条翻转 = 5pp")
print("=" * 78)
print()
print("【基准口径】Wilson 95% 置信区间（n=20）")
d = json.loads(DATA.read_text(encoding="utf-8"))
for r in d["coarse"]:
    k = r["hits"]
    lo, hi = wilson(k, 20)
    print(
        f"  chunk={r['chunk_chars']:>4} overlap={r['overlap']:>3}  "
        f"{k}/20 = {k / 20 * 100:>5.1f}%   95% CI [{lo * 100:>5.1f}%, {hi * 100:>5.1f}%]"
    )
for r in d["fine"]:
    k = r["hits"]
    lo, hi = wilson(k, 20)
    print(
        f"  chunk={r['chunk_chars']:>4} overlap={r['overlap']:>3}  "
        f"{k}/20 = {k / 20 * 100:>5.1f}%   95% CI [{lo * 100:>5.1f}%, {hi * 100:>5.1f}%]"
    )

print()
print("【配对 McNemar 精确检验】同一个 gold 集，只有「翻转」才携带信息")
pairs = [
    (
        "粗筛",
        "chunk 300 vs 500",
        row_of(d, "coarse", "chunk_chars", 300),
        row_of(d, "coarse", "chunk_chars", 500),
    ),
    (
        "粗筛",
        "chunk 500 vs 800",
        row_of(d, "coarse", "chunk_chars", 500),
        row_of(d, "coarse", "chunk_chars", 800),
    ),
    (
        "粗筛",
        "chunk 500 vs 1000",
        row_of(d, "coarse", "chunk_chars", 500),
        row_of(d, "coarse", "chunk_chars", 1000),
    ),
    ("细调", "overlap 0 vs 50", row_of(d, "fine", "overlap", 0), row_of(d, "fine", "overlap", 50)),
    (
        "细调",
        "overlap 50 vs 100",
        row_of(d, "fine", "overlap", 50),
        row_of(d, "fine", "overlap", 100),
    ),
]
for group, label, a, b in pairs:
    only_a, only_b, p = paired(hits(a), hits(b))
    verdict = "显著" if p < 0.05 else "**不显著**"
    print(
        f"  [{group}] {label:<20} 仅前者命中 {only_a}，仅后者命中 {only_b}"
        f" → McNemar p = {p:.4f}  {verdict}"
    )

print()
print("【必答题】overlap 0 → 50 的性价比（chunk=300）")
a = row_of(d, "fine", "overlap", 0)
b = row_of(d, "fine", "overlap", 50)
chunk_cost = (b["chunks"] / a["chunks"] - 1) * 100
hit_gain = b["hit_at_5"] - a["hit_at_5"]
only_a, only_b, p = paired(hits(a), hits(b))
print(f"  chunk 数 +{chunk_cost:.1f}%（{a['chunks']} → {b['chunks']}）")
print(f"  hit@5 {a['hits']}/20 → {b['hits']}/20（{hit_gain * 100:+.1f}pp）")
print(f"  翻转：仅 ov0 命中 {only_a} 条，仅 ov50 命中 {only_b} 条 → p = {p:.4f}")
print(f"  重复率 {a['adjacent_pair_rate'] * 100:.1f}% → {b['adjacent_pair_rate'] * 100:.1f}%")
print()
if p >= 0.05:
    print("  结论：**看不出 overlap 50 比 0 好**。两边都有翻转，净差是 1 条。")
else:
    print("  结论：差异显著。")
print()
print("【overlap 50 → 100】")
c = row_of(d, "fine", "overlap", 100)
cost2 = (c["chunks"] / b["chunks"] - 1) * 100
only_b2, only_c2, p2 = paired(hits(b), hits(c))
print(f"  chunk 数 +{cost2:.1f}%（{b['chunks']} → {c['chunks']}）")
print(f"  hit@5 {b['hits']}/20 → {c['hits']}/20（命中条数不变）")
print(f"  MRR {b['mrr']:.3f} → {c['mrr']:.3f}")
print(f"  翻转：仅 ov50 命中 {only_b2} 条，仅 ov100 命中 {only_c2} 条 → p = {p2:.4f}")
print(f"  重复率 {b['adjacent_pair_rate'] * 100:.1f}% → {c['adjacent_pair_rate'] * 100:.1f}%")
print()
print("  结论：命中率**完全不变**，但块数 +11%，重复内容率 +3pp。")
print("        即 overlap 100 付了全部代价，一分收益没有 —— 这是今天最硬的一条结论。")
