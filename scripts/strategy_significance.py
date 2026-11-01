"""三策略差异的显著性检验。**主表必须配这个，否则"recursive 最好"是噪声。**

## 为什么必须做

三策略的最优 Recall@5 是 40.0% / 37.5% / 37.5% —— 差了 **1 条**（50 条里 1 条 = 2pp）。

前几天的教训（`docs/retrieval-tuning-final.md` §6）已经写明：
20 条样本上，1 条翻转就是 5pp，任何差异都测不出来。
今天 50 条好一些，但 **1 条 = 2pp**，而差异正好是 1~2 条。

所以"recursive 40% 最高"这句话，如果不做检验就直接写进主表，
就是**把噪声当结论**。今天的文档要能回答：
"这个排序是稳健的，还是可能是噪声？"

## 检验方法：同一个 gold、不同切分 → 配对

三种策略切的是同一份语料、跑的是同一批 50 条问题，所以是**配对数据**。
配对二分数据用 McNemar 精确检验（与 `grid_significance.py` 同一套实现）：

    b = 只在策略 A 命中、不在 B 命中
    c = 只在 B 命中、不在 A 命中
    H0 下每次翻转等概率落 b 或 c → Binomial(b+c, 0.5)

用精确分布而不是卡方近似：样本小的时候卡方误差大，而这里正是小样本。

## 输出什么

1. 每个策略的 Wilson 95% 区间（看绝对水平的不确定性）
2. 两两配对的 McNemar p 值（看排序是否可信）
3. **一句结论**：哪些排序稳健、哪些可能是噪声
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

DEFAULT_SOURCE: Path = PROJECT_ROOT / "scripts" / "strategy_compare.json"
DEFAULT_OUT: Path = PROJECT_ROOT / "scripts" / "strategy_significance.json"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 区间。n 小、p 接近 0/1 时比常态近似可靠。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar 精确检验双侧 p。`b`/`c` 是两个方向的翻转数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def per_query_hits(entry: dict[str, Any]) -> dict[str, bool]:
    """从 sweep 条目里取"每条题是否命中"。**多跳已按"全部锚点命中"算好。**"""
    hits: dict[str, bool] = {}
    for row in entry.get("per_query", []):
        hits[row["id"]] = row["hit_rank"] is not None
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="三策略差异显著性检验")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"缺少 {args.source}，先跑 strategy_compare.py", file=sys.stderr)
        return 2

    payload = json.loads(args.source.read_text(encoding="utf-8"))
    best = payload.get("best", {})
    sweep = {f"{row['strategy']}@{row['chunk_chars']}": row for row in payload["sweep"]}

    print("=" * 78)
    print("三策略差异显著性（50 条里 1 条 = 2pp）")
    print("=" * 78)

    # ---- ① 绝对水平的不确定性
    print()
    print("① 各策略最优配置的 Wilson 95% 区间")
    intervals: dict[str, tuple[float, float]] = {}
    for strategy, row in best.items():
        total = row["recall"]["recall@5"]["total"]
        hits = row["recall"]["recall@5"]["hits"]
        low, high = wilson(hits, total)
        intervals[strategy] = (low, high)
        print(
            f"  {strategy:<10} chunk={row['chunk_chars']:>4}  "
            f"{hits}/{total} = {hits / total * 100:>5.1f}%  "
            f"95% CI [{low * 100:>5.1f}%, {high * 100:>5.1f}%]"
        )

    # ---- ② 两两配对检验
    # **注意：要用同一策略最优配置的逐条结果**，而不是扫描表里的任意一行。
    print()
    print("② 两两配对 McNemar 精确检验（同一批 50 条，配对）")
    hits_by_strategy: dict[str, dict[str, bool]] = {}
    for strategy, row in best.items():
        key = f"{strategy}@{row['chunk_chars']}"
        if key in sweep:
            hits_by_strategy[strategy] = per_query_hits(sweep[key])

    pairs: list[dict[str, Any]] = []
    for left, right in combinations(sorted(hits_by_strategy), 2):
        left_hits = hits_by_strategy[left]
        right_hits = hits_by_strategy[right]
        common = set(left_hits) & set(right_hits)
        only_left = sum(1 for qid in common if left_hits[qid] and not right_hits[qid])
        only_right = sum(1 for qid in common if right_hits[qid] and not left_hits[qid])
        p_value = mcnemar_exact(only_left, only_right)
        verdict = "显著" if p_value < 0.05 else "**不显著**"
        pairs.append(
            {
                "left": left,
                "right": right,
                "only_left": only_left,
                "only_right": only_right,
                "p_value": round(p_value, 4),
                "significant": p_value < 0.05,
            }
        )
        print(
            f"  {left:<10} vs {right:<10} 仅左命中 {only_left}，仅右命中 {only_right}"
            f"  → p = {p_value:.4f}  {verdict}"
        )

    # ---- ③ 结论
    print()
    print("③ 结论")
    significant_pairs = [pair for pair in pairs if pair["significant"]]
    if not significant_pairs:
        print("  **没有任何一对策略的差异是显著的。**")
        # **数字从数据里取，不硬编码** —— 写死会在下次跑出不同的数时撒谎。
        ranking = sorted(
            (
                (strategy, row["recall"]["recall@5"]["hits"], row["recall"]["recall@5"]["total"])
                for strategy, row in best.items()
            ),
            key=lambda item: -item[1],
        )
        described = " > ".join(f"{name} {hits}/{total}" for name, hits, total in ranking)
        gap = ranking[0][1] - ranking[-1][1] if len(ranking) > 1 else 0
        print(f"  也就是说主表里的排序（{described}）")
        print(f"  在 {ranking[0][2]} 条可答题上**无法与噪声区分** —— 最多只差 {gap} 条。")
        print()

        # Recall@10 是否相同：这是最容易稳健成立的一条
        r10 = {strategy: row["recall"]["recall@10"]["hits"] for strategy, row in best.items()}
        if len(set(r10.values())) == 1:
            value = next(iter(r10.values()))
            print(f"  能说的只有：三者的 Recall@10 完全相同（{value}/{ranking[0][2]}），")
            print("  即「在 top10 范围内它们召回的东西一样多」，差别只在排位。")
        else:
            print(f"  三者的 Recall@10 不同：{r10}")
    else:
        for pair in significant_pairs:
            better = pair["left"] if pair["only_left"] > pair["only_right"] else pair["right"]
            print(
                f"  {better} 显著优于另一方"
                f"（p={pair['p_value']}, 翻转 {pair['only_left']}/{pair['only_right']}）"
            )

    # ---- ④ 区间是否重叠
    print()
    print("④ 区间重叠检查")
    overlapping = True
    names = sorted(intervals)
    for left, right in combinations(names, 2):
        low_l, high_l = intervals[left]
        low_r, high_r = intervals[right]
        disjoint = high_l < low_r or high_r < low_l
        if disjoint:
            overlapping = False
            print(f"  {left} 与 {right} 的区间**不重叠**")
    if overlapping:
        print("  所有区间互相重叠 —— 绝对水平也无法互相区分")

    output = {
        "source": str(args.source.name),
        "intervals": {
            name: [round(low, 4), round(high, 4)] for name, (low, high) in intervals.items()
        },
        "pairs": pairs,
        "all_significant_pairs": bool(significant_pairs),
        "intervals_all_overlap": overlapping,
    }
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘：{args.out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
