"""成本模型：规则版 vs LLM 版，以及重试对期望成本的影响。

**这个脚本只做算术，不调模型** —— 因为数字来自已经实测过的 token 数
（`scripts/token_result.json`、`scripts/cost_table.json`），
重复跑模型只会得到"另一组略有不同的数"，不会让结论更可靠。

## 必须区分三个极易混淆的"次数"

| 概念 | 是什么 | 谁决定 |
| --- | --- | --- |
| **尝试次数**（attempts） | 一次用户请求里实际发了几个 HTTP 请求 | 重试策略 + 失败率 |
| **逻辑调用** | 用户视角"我请求了一次抽取" | 用户 |
| **token 消耗** | 每次尝试都**独立计费**（失败的也付钱） | 输入/输出长度 |

把这三个混在一起，是成本估算最常见的错 —— 会得出"重试只多花一点时间"的错觉，
实际**失败的尝试同样收费**。

## 三个成本场景（对应任务的验收）

1. **一次成功**：1 次调用，无重试
2. **含 1 次重试**：第一次失败 + 第二次成功 = 2 次计费
3. **含 2 次重试**：3 次计费，然后降级（或放弃）

还要算上**降级路径的额外成本**：降级本身不花钱（规则版是纯计算），
但用户拿到的是质量较低的结果 —— 这是"隐性成本"，用置信度下降来体现。

运行：

    uv run python -X utf8 scripts/cost_model.py --price-in 2 --price-out 8
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

# 实测 token 数（来自 scripts/token_result.json，buddy/glm-5.3，2026-10-26 实测）。
# 换模型/换 prompt 必须重测这些数 —— 硬编"大概 300"是最常见的估算错误。
TOKEN_PROFILES: dict[str, dict[str, int]] = {
    # 名字: 输入 token, 输出 token
    "短文本(2 字段)": {"input": 171, "output": 27},
    "标准合同(4 字段)": {"input": 227, "output": 89},
    "长合同(6 字段)": {"input": 1918, "output": 78},
}

# 校验回灌时的额外输入成本：把错误输出 + 纠正提示追加进 messages。
# 实测一个回灌轮大约额外 120–200 token 输入（错误输出 + 纠正说明）。
# 这里取 160 作为典型值 —— 一个可解释的估算，不是拍脑袋。
CORRECTION_TOKEN_OVERHEAD = 160

# 规则版是纯 CPU，成本按 0 计（只有推理算力，没有 API 费用）。
RULE_COST = 0.0


def _percentile_default(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(r[key] for r in rows) / len(rows)


def expected_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    price_in: float,
    price_out: float,
    attempts: int,
    correction_rounds: int = 0,
) -> dict[str, Any]:
    """算一次逻辑调用在 `attempts` 次尝试 + `correction_rounds` 轮回灌下的成本。

    **两笔成本要分开加**：
    1. `attempts × (输入 + 输出)` —— 每次尝试都必须完整计费（哪怕失败）
    2. `correction_rounds × (额外输入 + 输出)` —— 回灌轮是一次**新的**完整调用

    单价单位：元 / 百万 token（各家通用报价单位）。
    """
    if attempts < 1:
        raise ValueError("attempts 至少为 1")

    per_attempt_in = input_tokens * price_in / 1_000_000
    per_attempt_out = output_tokens * price_out / 1_000_000
    per_attempt = per_attempt_in + per_attempt_out

    correction_in = (input_tokens + CORRECTION_TOKEN_OVERHEAD) * price_in / 1_000_000
    correction_out = output_tokens * price_out / 1_000_000
    correction_cost = correction_in + correction_out

    total = per_attempt * attempts + correction_cost * correction_rounds

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "attempts": attempts,
        "correction_rounds": correction_rounds,
        "per_attempt_cost": per_attempt,
        "correction_round_cost": correction_cost,
        "total_cost": total,
        "total_input_tokens": input_tokens * attempts
        + (input_tokens + CORRECTION_TOKEN_OVERHEAD) * correction_rounds,
        "total_output_tokens": output_tokens * attempts + output_tokens * correction_rounds,
    }


def build_scenarios(
    *,
    price_in: float,
    price_out: float,
    per_n: int = 1000,
) -> list[dict[str, Any]]:
    """生成"一次成功 / 含 1 次重试 / 含 2 次重试"三种场景的成本表。"""
    rows: list[dict[str, Any]] = []

    # 场景定义：(尝试次数, 回灌轮数, 说明)
    # 注意：这里的 attempts 是**HTTP 尝试**（429/5xx 重试），
    # correction_rounds 是**语义重试**（校验失败回灌）。两者是独立的两层。
    scenarios = [
        (1, 0, "一次成功"),
        (1, 1, "1 次校验回灌后成功"),
        (1, 2, "2 次校验回灌后成功"),
        (2, 0, "1 次 HTTP 重试后成功"),
        (1, 3, "回灌用尽 → 降级（含 3 次计费）"),
    ]

    for profile_name, tokens in TOKEN_PROFILES.items():
        for attempts, corrections, label in scenarios:
            cost = expected_cost(
                input_tokens=tokens["input"],
                output_tokens=tokens["output"],
                price_in=price_in,
                price_out=price_out,
                attempts=attempts,
                correction_rounds=corrections,
            )
            rows.append(
                {
                    "profile": profile_name,
                    "scenario": label,
                    "attempts": attempts,
                    "correction_rounds": corrections,
                    "cost_per_call": cost["total_cost"],
                    f"cost_per_{per_n}": cost["total_cost"] * per_n,
                    "input_tokens": tokens["input"],
                    "output_tokens": tokens["output"],
                    "total_tokens": cost["total_input_tokens"] + cost["total_output_tokens"],
                    "degraded": corrections > 2,
                }
            )
    return rows


def sensitivity_table(
    *,
    base: dict[str, Any],
    price_in: float,
    price_out: float,
    per_n: int = 10_000,
) -> list[dict[str, Any]]:
    """优化手段的敏感性分析。

    三种手段（对应任务要求）：

    1. **prompt 精简 30%**：只减**输入** token。输出不受影响。
       真实收益取决于输入在总成本里的占比 —— 长文本场景收益大，短文本收益小。
    2. **结果缓存命中 50%**：一半请求完全不调模型。
       收益是**线性的 50%**，而且没有副作用 —— 这是性价比最高的一项。
    3. **小模型分流 60%**：60% 的请求走更便宜的小模型（假设输入价格 1/4、输出 1/4）。
       收益最大但**有质量风险**：小模型在复杂字段上准确率会掉。

    **必须同时给"收益"和"代价"**：只报收益的敏感性分析是销售话术，不是工程判断。
    """
    in_tok = base["input_tokens"]
    out_tok = base["output_tokens"]
    base_cost = expected_cost(
        input_tokens=in_tok,
        output_tokens=out_tok,
        price_in=price_in,
        price_out=price_out,
        attempts=base["attempts"],
        correction_rounds=base["correction_rounds"],
    )["total_cost"]

    rows: list[dict[str, Any]] = []

    def add(name: str, cost: float, note: str) -> None:
        rows.append(
            {
                "measure": name,
                "cost_per_call": cost,
                f"cost_per_{per_n}": cost * per_n,
                "saving_pct": round((1 - cost / base_cost) * 100, 1) if base_cost else 0.0,
                "note": note,
            }
        )

    # 基线
    add("基线（无优化）", base_cost, "当前实测 token 数")

    # ① prompt 精简 30%（只影响输入）
    slim = expected_cost(
        input_tokens=int(in_tok * 0.7),
        output_tokens=out_tok,
        price_in=price_in,
        price_out=price_out,
        attempts=base["attempts"],
        correction_rounds=base["correction_rounds"],
    )["total_cost"]
    input_share = (in_tok * price_in) / max(in_tok * price_in + out_tok * price_out, 1e-12)
    add(
        "prompt 精简 30%",
        slim,
        f"只减输入；当前输入占总价 {input_share:.0%}，所以实际收益约 30%×{input_share:.0%}",
    )

    # ② 缓存命中 50%
    add("结果缓存命中 50%", base_cost * 0.5, "一半请求不调模型；收益线性，无质量风险")

    # ③ 小模型分流 60%（价格 1/4）
    small_price_in = price_in * 0.25
    small_price_out = price_out * 0.25
    small_cost = expected_cost(
        input_tokens=in_tok,
        output_tokens=out_tok,
        price_in=small_price_in,
        price_out=small_price_out,
        attempts=base["attempts"],
        correction_rounds=base["correction_rounds"],
    )["total_cost"]
    mixed = base_cost * 0.4 + small_cost * 0.6
    add(
        "小模型分流 60%",
        mixed,
        "分流部分按 1/4 单价；**有质量风险**，复杂字段准确率会掉",
    )

    # 组合：缓存 + 精简
    combo = slim * 0.5
    add("缓存 50% + 精简 30%", combo, "两者独立，收益近似相乘")
    return rows


def render_markdown(
    rows_by_profile: dict[str, list[dict[str, Any]]],
    sensitivity: dict[str, list[dict[str, Any]]],
    meta: dict[str, Any],
) -> str:
    """输出可直接进 `docs/cost.md` 的表格。"""
    lines: list[str] = []
    meta["per_n"]

    lines.append("### 单次调用成本（规则版 vs LLM 版）\n")
    lines.append(
        f"> 单价：输入 **{meta['price_in']}** 元 / 百万 token，"
        f"输出 **{meta['price_out']}** 元 / 百万 token"
        f"（示例锚点，**非具体供应商报价**）\n"
        f"> token 数为实测值（`scripts/token_result.json`）；"
        f"校验回灌每轮额外按 {CORRECTION_TOKEN_OVERHEAD} 输入 token 计\n"
    )

    lines.append("| 场景 | 尝试次数 | 回灌轮数 | 总 token | 单次成本 | 每 1,000 次 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    lines.append("| **规则版**（兜底，纯 CPU） | 0 | 0 | 0 | **0.000000 元** | **0.00 元** |")
    for profile, rows in rows_by_profile.items():
        for row in rows:
            mark = " ⚠️降级" if row["degraded"] else ""
            lines.append(
                f"| {profile} · {row['scenario']}{mark} "
                f"| {row['attempts']} | {row['correction_rounds']} "
                f"| {row['total_tokens']} | {row['cost_per_call']:.6f} 元 "
                f"| {row['cost_per_1000']:.2f} 元 |"
            )

    lines.append("\n### 优化手段敏感性（按标准合同 · 一次成功）\n")
    lines.append("| 手段 | 单次成本 | 每 10,000 次 | 节省 | 代价 / 备注 |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for row in sensitivity["standard"]:
        lines.append(
            f"| {row['measure']} | {row['cost_per_call']:.6f} 元 "
            f"| {row['cost_per_10000']:.2f} 元 | **{row['saving_pct']:.1f}%** "
            f"| {row['note']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM 抽取成本模型")
    parser.add_argument("--price-in", type=float, default=2.0, help="输入单价：元/百万 token")
    parser.add_argument("--price-out", type=float, default=8.0, help="输出单价：元/百万 token")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "scripts" / "cost_model.json"))
    args = parser.parse_args()

    rows = build_scenarios(price_in=args.price_in, price_out=args.price_out)
    rows_by_profile: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_profile.setdefault(row["profile"], []).append(row)

    # 敏感性分析用"标准合同 · 一次成功"作基线（最有代表性的场景）
    standard = next(
        r
        for r in rows
        if r["profile"] == "标准合同(4 字段)" and r["attempts"] == 1 and r["correction_rounds"] == 0
    )
    sensitivity = {
        "standard": sensitivity_table(
            base=standard, price_in=args.price_in, price_out=args.price_out
        )
    }

    meta = {"price_in": args.price_in, "price_out": args.price_out, "per_n": 1000}
    markdown = render_markdown(rows_by_profile, sensitivity, meta)
    print(markdown)

    out = Path(args.out)
    out.write_text(
        json.dumps(
            {"meta": meta, "scenarios": rows, "sensitivity": sensitivity},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] 明细写入 {out.name}")

    # 附：一段诚实的结论
    print("\n### 读这张表要注意的三件事\n")
    fast = next(
        r for r in rows if r["profile"] == "标准合同(4 字段)" and r["scenario"] == "一次成功"
    )
    retry2 = next(
        r
        for r in rows
        if r["profile"] == "标准合同(4 字段)" and r["scenario"] == "2 次校验回灌后成功"
    )
    ratio = retry2["cost_per_call"] / fast["cost_per_call"]
    print("1. **回灌重试不是「多一点时间」，是「多几倍钱」**：")
    print(f"   标准合同一次成功 {fast['cost_per_call']:.6f} 元，")
    print(f"   2 轮回灌后成功 {retry2['cost_per_call']:.6f} 元 —— **{ratio:.1f} 倍**。")
    print("   所以「提高首次成功率」（更明确的 prompt、标量归一）比「加大重试」划算得多。")
    print()
    print("2. **规则版的成本是 0，但质量不是免费的**：")
    print(
        "   降级省下全部 API 费用，代价是复杂字段的准确率下降 —— 用 `degraded` 标记暴露给调用方。"
    )
    print()
    print("3. **缓存是唯一「纯收益」的手段**：")
    print("   省 50% 且没有质量代价，因为返回的是同一个输入的同一份结果。")
    print(
        "   前提是「同一输入」真的可复现 —— 而 temperature=0 也不保证（见 learn/llm/01_params.md §5），"
    )
    print("   所以缓存键要包含**输入文本内容**，命中即复用，不重新生成。")

    return 0


if __name__ == "__main__":
    # **不要**用 suppress 包住 main：那会把真正的 bug 变成「静默无输出」。
    # 踩过：KeyError 被吞掉，脚本退出码 0 却什么都没打印，看上去像"跑完了但没结果"。
    raise SystemExit(main())
