"""cost.py：每千次 / 每万次成本 + **含重试的期望成本**，输出 Markdown 表。

**这个脚本和 `cost_report.py` 的分工：**
- `cost_report.py` 负责**测**：跑真实调用，拿到不同文本长度下的 token 用量。
- `cost.py`（本文件）负责**算**：拿已知的 token 数，算成钱，并给出可直接贴进文档的表。

分开的理由是**测算不该耦合**。你想改单价、改重试率、改日调用量时，
不该为了看新数字再去花真钱跑一遍模型。

### 为什么必须算"含重试的期望成本"

新手算成本只看"单次价格 × 次数"。真实世界的账是这样的：

    一次调用成功的概率 = p
    第一次失败、第二次成功 = (1-p) * p
    ...
    期望调用次数 = 1 / p

`p=0.9` 时期望次数是 1.11 次 —— 成本多 11%。**而且失败的请求也是要付费的**
（服务端已经处理了，429 之前的 token 通常照扣），所以不能只按"成功次数"算。

更糟的是：**重试还带来了尾延迟**。重试 3 次意味着可能出现
`timeout × 3` 的响应时间，用户等 90 秒。所以重试策略必须同时看成本与延迟。

### 输入输出分别计价

绝大多数供应商的输出单价是输入的 3–5 倍（昨天的 `cost_report` 实测也印证：
输出 token 数少但金额占比高）。所以表里必须**分开列**，
否则"优化 prompt 长度"和"限制输出长度"哪个更划算就看不出来。

运行：

    uv run python -X utf8 scripts/cost.py
    uv run python -X utf8 scripts/cost.py --input-price 1 --output-price 4 --success-rate 0.95
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 默认 token 用量：来自 scripts/cost_report.py 与 measure_tokens.py 的实测
# （buddy/glm-5.3，三种正文长度）。用实测值当默认，脚本才有实际意义；
# 换成别的模型时用 --input-tokens / --output-tokens 覆盖。
PROFILES: list[dict[str, Any]] = [
    # 名字, 输入 token, 输出 token, 说明
    {"name": "短文本", "input": 171, "output": 27, "note": "22 字正文 / 2 字段"},
    {"name": "标准合同", "input": 227, "output": 89, "note": "109 字正文 / 4 字段"},
    {"name": "长合同", "input": 1918, "output": 78, "note": "2636 字正文 / 6 字段"},
    {"name": "推理模型·长合同", "input": 1918, "output": 400, "note": "思维链计入 completion"},
]


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        with suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")


def expected_attempts(success_rate: float) -> float:
    """一次成功所需的期望尝试次数 = 1/p（几何分布的期望）。

    p=1 → 1.0；p=0.9 → 1.111；p=0.5 → 2.0。
    注意这是**没有**最大次数上限时的期望；有上限时会被截断，
    所以下面 `expected_cost_with_retry` 用的是"截断后的期望"。
    """
    if not 0 < success_rate <= 1:
        raise ValueError("success_rate 必须在 (0, 1] 区间")
    return 1.0 / success_rate


def expected_attempts_capped(success_rate: float, max_attempts: int) -> float:
    """**带最大次数上限**的期望尝试次数（这才是真实生效的公式）。

    推导：第 k 次尝试发生的概率 = 前 k-1 次全失败 = (1-p)^(k-1)。
    期望次数 = Σ_{k=1..N} (1-p)^(k-1)

    为什么不直接用 1/p：1/p 是无限次重试的期望。我们的 `max_attempts=3`，
    已经放弃的情况下不该再计入"还会继续试"的成本 —— 那会高估。
    """
    if not 0 < success_rate <= 1:
        raise ValueError("success_rate 必须在 (0, 1] 区间")
    if max_attempts < 1:
        raise ValueError("max_attempts 至少为 1")
    q = 1.0 - success_rate
    return sum(q**k for k in range(max_attempts))


def cost_per_call(
    *, input_tokens: int, output_tokens: int, input_price: float, output_price: float
) -> dict[str, float]:
    """单次调用的成本。单价单位：元 / 百万 token。"""
    input_cost = input_tokens * input_price / 1_000_000
    output_cost = output_tokens * output_price / 1_000_000
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def build_rows(
    *,
    input_price: float,
    output_price: float,
    success_rate: float,
    max_attempts: int,
    round_trip_ms: int,
) -> list[dict[str, Any]]:
    """算出完整的成本表。

    每行包含：单次成本、期望尝试次数、含重试期望成本、
    每千次/每万次成本、以及**含重试的最坏延迟**。
    """
    attempts = expected_attempts_capped(success_rate, max_attempts)
    rows: list[dict[str, Any]] = []

    for profile in PROFILES:
        base = cost_per_call(
            input_tokens=profile["input"],
            output_tokens=profile["output"],
            input_price=input_price,
            output_price=output_price,
        )
        # 每次尝试都要付出完整的 token 成本（失败的请求同样计费）
        expected_total = base["total_cost"] * attempts
        rows.append(
            {
                "name": profile["name"],
                "note": profile["note"],
                "input_tokens": profile["input"],
                "output_tokens": profile["output"],
                "single_cost": base["total_cost"],
                "input_share": (
                    base["input_cost"] / base["total_cost"] if base["total_cost"] else 0.0
                ),
                "expected_attempts": attempts,
                # 相对无重试的溢价
                "retry_overhead_pct": (attempts - 1) * 100,
                "expected_cost": expected_total,
                "per_1k": expected_total * 1_000,
                "per_10k": expected_total * 10_000,
                "per_100k": expected_total * 100_000,
                # 最坏情况：每次都超时重试
                "worst_latency_ms": round_trip_ms * max_attempts,
            }
        )
    return rows


def render_markdown(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    """输出可直接贴进文档的 Markdown 表。"""
    lines: list[str] = []
    lines.append("### 单次成本（输入/输出分开计价）\n")
    lines.append(
        f"> 单价：输入 {meta['input_price']} 元 / 百万 token，"
        f"输出 {meta['output_price']} 元 / 百万 token"
        f"（**示例锚点，非具体供应商报价**）"
    )
    lines.append(
        f"> 假设成功率 p={meta['success_rate']}，最多重试 {meta['max_attempts']} 次，"
        f"单次往返 {meta['round_trip_ms']}ms\n"
    )
    lines.append("| 场景 | 输入 tok | 输出 tok | 单次成本 | 输入占比 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['name']}<br/><sub>{row['note']}</sub> "
            f"| {row['input_tokens']} | {row['output_tokens']} "
            f"| {row['single_cost']:.6f} 元 | {row['input_share']:.0%} |"
        )

    lines.append("\n### 含重试的期望成本\n")
    lines.append(
        f"期望尝试次数 = Σ(1-p)^k，k=0..{meta['max_attempts'] - 1} = "
        f"**{rows[0]['expected_attempts']:.4f}** 次"
    )
    lines.append(
        f"→ 相比不重试溢价 **{rows[0]['retry_overhead_pct']:.1f}%**（失败的请求同样计费）\n"
    )
    lines.append("| 场景 | 期望成本/次 | 每 1,000 次 | 每 10,000 次 | 每 100,000 次 | 最坏延迟 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['expected_cost']:.6f} 元 "
            f"| {row['per_1k']:.2f} 元 | {row['per_10k']:.2f} 元 "
            f"| {row['per_100k']:.0f} 元 | {row['worst_latency_ms']} ms |"
        )
    return "\n".join(lines)


def main() -> int:
    _force_utf8_stdout()

    parser = argparse.ArgumentParser(description="LLM 成本测算（含重试期望）")
    parser.add_argument("--input-price", type=float, default=2.0, help="输入：元/百万 token")
    parser.add_argument("--output-price", type=float, default=8.0, help="输出：元/百万 token")
    parser.add_argument("--success-rate", type=float, default=0.95, help="单次调用成功率 p")
    parser.add_argument("--max-attempts", type=int, default=3, help="最大尝试次数")
    parser.add_argument("--round-trip-ms", type=int, default=3000, help="单次往返耗时(ms)")
    parser.add_argument("--out", default=str(Path(__file__).with_name("cost_table.md")))
    args = parser.parse_args()

    meta = {
        "input_price": args.input_price,
        "output_price": args.output_price,
        "success_rate": args.success_rate,
        "max_attempts": args.max_attempts,
        "round_trip_ms": args.round_trip_ms,
    }
    rows = build_rows(
        input_price=args.input_price,
        output_price=args.output_price,
        success_rate=args.success_rate,
        max_attempts=args.max_attempts,
        round_trip_ms=args.round_trip_ms,
    )
    markdown = render_markdown(rows, meta)

    print(markdown)

    # 附加：不同成功率对成本的影响（让"要不要重试"这个决策有数字依据）
    print("\n### 成功率对期望成本的影响（标准合同）\n")
    print("| 成功率 p | 期望尝试次数 | 每万次成本 | 相对 p=1.0 溢价 |")
    print("| ---: | ---: | ---: | ---: |")
    baseline = next(r for r in rows if r["name"] == "标准合同")
    base_per_10k = baseline["single_cost"] * 10_000
    for p in (1.0, 0.99, 0.95, 0.9, 0.8, 0.5):
        att = expected_attempts_capped(p, args.max_attempts)
        cost = base_per_10k * att
        print(f"| {p:.2f} | {att:.4f} | {cost:.2f} 元 | {(att - 1) * 100:.1f}% |")

    out_path = Path(args.out)
    out_path.write_text(markdown + "\n", encoding="utf-8")
    # 同时留一份结构化数据，便于后续被别的脚本消费
    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[done] Markdown -> {out_path.name}，数据 -> {json_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
