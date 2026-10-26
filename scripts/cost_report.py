"""cost_report：真实 token 用量 + 每千次调用成本测算。

**为什么这件事不能"拍脑袋估"**：
决策者问"接 LLM 要花多少钱"时，答案必须是"按这个 prompt、这个正文长度，
实测 N 次，平均 X token/次，按单价 Y 算每千次 Z 元"。
凭感觉说"很便宜"或"大概几分钱"，等于没有回答。

本脚本做三件事：

1. 用**真实的** `LlmExtractor` 跑三种典型正文长度（短/中/长），记录 usage
2. 算出 p50 / mean / max 的 prompt+completion token
3. 按**可配置的单价**折算每千次 / 每月成本，并列出临界点
   （单价变了就改 `--input-price`，不用改代码）

单价默认取一个常见的商业模型量级（输入 2 元/百万 token、输出 8 元/百万 token），
**只是示例锚点**，不代表任何具体供应商。真实报价请查你的供应商价目表。

运行：

    uv run python -X utf8 scripts/cost_report.py
    uv run python -X utf8 scripts/cost_report.py --runs 5 --input-price 1 --output-price 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.llm.client import get_llm_client  # noqa: E402
from app.services.llm_extractor import LlmExtractor  # noqa: E402

# ---------------------------------------------------------------- 样本正文
# 三种长度对应真实业务的三种形态：短消息、标准合同、长合同全文。
# **长度是成本的主导变量**，所以必须分层测，只测一种没有代表性。
SHORT_TEXT = "合同编号：HT-2026-001，甲方：张三"
MEDIUM_TEXT = (
    "合同编号：HT-2026-001，甲方：杭州星辰科技有限公司，乙方：上海云脉信息技术有限公司，"
    "签订日期：2026-10-20，合同金额：¥128,000.00，履行期限：2026-11-01 至 2027-10-31。"
)
LONG_TEXT = MEDIUM_TEXT + "正文补充条款：" + ("乙方应按约定交付并保证质量，甲方按期付款。" * 120)

SAMPLES = [
    ("短文本", SHORT_TEXT, ["合同编号", "甲方"]),
    ("标准合同", MEDIUM_TEXT, ["合同编号", "甲方", "签订日期", "合同金额"]),
    ("长合同", LONG_TEXT, ["合同编号", "甲方", "乙方", "签订日期", "合同金额", "履行期限"]),
]


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        with suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")


def _percentile(values: list[int], p: float) -> int:
    """线性插值分位数。样本量小（默认 3），p50 = 中位数足够用了。"""
    if not values:
        return 0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    low, high = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (pos - low))


async def measure(
    extractor: LlmExtractor,
    *,
    runs: int,
) -> list[dict[str, Any]]:
    """对每种样本跑 `runs` 次，收集 usage。

    直接复用 `LlmExtractor` 而不是自己拼请求：**测的必须是线上那条路径**，
    自己另写一份 prompt 测出来的 token 数没有意义。
    """
    rows: list[dict[str, Any]] = []
    for label, text, fields in SAMPLES:
        for idx in range(runs):
            output = await extractor.extract(text=text, fields=fields)
            # LlmExtractor 不暴露 usage（契约响应体里也没有 usage 这个东西），
            # 所以这里用"正文+prompt 的字符数"反推，并用 completion 的字段数估算。
            # ⚠️ 这是一个**近似**，下面会用真实 usage 校准。
            rows.append(
                {
                    "sample": label,
                    "run": idx + 1,
                    "text_chars": len(text),
                    "field_count": len(fields),
                    "prompt_chars": len(text) + 200,
                    "completion_chars": sum(
                        len(v) for v in output.data.values() if isinstance(v, str)
                    )
                    + 40,
                    "filled": sum(1 for v in output.data.values() if v is not None),
                    "elapsed_ms": output.elapsed_ms,
                    "model": output.model,
                }
            )
    return rows


async def measure_real_usage(client: AsyncOpenAI, model: str, extra: dict[str, Any]) -> list[dict]:
    """用真实 usage 校准 —— 直接打一次同构请求并读 `usage`。

    `LlmExtractor` 的返回值是契约形状（不含 usage），所以校准只能在这一层单独做。
    这是**有意的分层**：usage 是运维指标，不该进对外契约（§2.1 只有 4 个字段）。
    """
    from app.services.llm_extractor import SYSTEM_PROMPT

    calibrated = []
    for label, text, fields in SAMPLES:
        # 复用 LlmExtractor 自己的 prompt 构造器，**不另抄一份**。
        # 抄一份的话，哪天改了提示词，成本测算就和线上脱节了 ——
        # 而这种漂移不会报错，只会让所有成本数字悄悄失真。
        user_content = LlmExtractor._build_user_prompt(text, fields)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=1024,
            **extra,
        )
        usage = response.usage
        calibrated.append(
            {
                "sample": label,
                "text_chars": len(text),
                "field_count": len(fields),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        )
    return calibrated


def cost_table(
    rows: list[dict[str, Any]],
    *,
    input_price: float,
    output_price: float,
    per: int,
) -> list[dict[str, Any]]:
    """折算每 `per` 次调用的成本。

    单价单位：元 / 百万 token。这是各家供应商的通用报价单位。
    """
    out = []
    for row in rows:
        one_input = row["prompt_tokens"] * input_price / 1_000_000
        one_output = row["completion_tokens"] * output_price / 1_000_000
        one_total = one_input + one_output
        out.append(
            {
                **row,
                "cost_per_call_cny": round(one_total, 6),
                f"cost_per_{per}_cny": round(one_total * per, 4),
            }
        )
    return out


async def main() -> int:
    _force_utf8_stdout()

    parser = argparse.ArgumentParser(description="LLM 抽取成本测算")
    parser.add_argument("--runs", type=int, default=2, help="每种样本重复次数（估延迟用）")
    parser.add_argument("--input-price", type=float, default=2.0, help="输入单价：元/百万 token")
    parser.add_argument("--output-price", type=float, default=8.0, help="输出单价：元/百万 token")
    parser.add_argument("--per", type=int, default=1000, help="折算多少次调用")
    parser.add_argument(
        "--out", type=str, default=str(Path(__file__).with_name("cost_result.json"))
    )
    args = parser.parse_args()

    settings = get_settings()
    llm = settings.llm
    client = get_llm_client(settings)
    extractor = LlmExtractor(
        client,
        model=llm.model,
        extra_request={"reasoning_effort": "none"},
    )

    print(f"model={llm.model}  base_url={llm.base_url}")
    print(
        f"单价假设：输入 {args.input_price} 元/百万 token，输出 {args.output_price} 元/百万 token"
    )
    print("（示例锚点，非任何具体供应商报价；请按实际价目表替换）\n")

    # ① 真实 usage
    print("== 真实 token 用量（每样本一次）==")
    usage_rows = await measure_real_usage(client, llm.model, {"reasoning_effort": "none"})
    for row in usage_rows:
        print(
            f"  {row['sample']:<8} 正文 {row['text_chars']:>6} 字 / {row['field_count']} 字段"
            f" -> prompt {row['prompt_tokens']:>5} + completion {row['completion_tokens']:>4}"
            f" = {row['total_tokens']:>5} tokens"
        )

    # ② 走真实链路测延迟与稳定性
    print(f"\n== 真实链路（LlmExtractor，每样本 {args.runs} 次）==")
    latency_rows = await measure(extractor, runs=args.runs)
    for label, _text, _fields in SAMPLES:
        subset = [r for r in latency_rows if r["sample"] == label]
        lat = [r["elapsed_ms"] for r in subset]
        filled = [r["filled"] for r in subset]
        print(
            f"  {label:<8} 延迟 p50={_percentile(lat, 0.5)}ms max={max(lat)}ms"
            f" | 命中字段 {statistics.mean(filled):.1f}/{subset[0]['field_count']}"
        )

    # ③ 成本表
    table = cost_table(
        usage_rows,
        input_price=args.input_price,
        output_price=args.output_price,
        per=args.per,
    )
    key = f"cost_per_{args.per}_cny"
    print(f"\n== 成本表（每 {args.per} 次调用）==")
    print(f"  {'样本':<10} {'tokens/次':>10} {key:>16}")
    for row in table:
        print(f"  {row['sample']:<10} {row['total_tokens']:>10} {row[key]:>16}")

    total_tokens = [r["total_tokens"] for r in table]
    mean_tokens = statistics.mean(total_tokens)
    mean_cost = statistics.mean(r[key] for r in table)
    print(f"\n  平均 {mean_tokens:.0f} token/次 -> 每 {args.per} 次约 {mean_cost:.2f} 元")

    # ④ 量级参考：不同日调用量下的月成本
    print("\n== 月成本量级（按均值，30 天）==")
    for daily in (1_000, 10_000, 100_000, 1_000_000):
        monthly = mean_cost / args.per * daily * 30
        print(f"  {daily:>9,} 次/天 -> 约 {monthly:>10,.0f} 元/月")

    print("\n== 成本结构（关键结论）==")
    for row in table:
        input_share = row["prompt_tokens"] * args.input_price
        output_share = row["completion_tokens"] * args.output_price
        total = input_share + output_share or 1
        print(
            f"  {row['sample']:<10} 输入占 {input_share / total:>5.1%}"
            f" / 输出占 {output_share / total:>5.1%}"
            f"（prompt 占 token 数 {row['prompt_tokens'] / max(row['total_tokens'], 1):>5.1%}）"
        )
    print("\n  → 输出单价通常是输入的 3–5 倍，所以『少让模型说话』比『少喂正文』更省。")

    result = {
        "meta": {
            "model": llm.model,
            "base_url": llm.base_url,
            "runs_per_sample": args.runs,
            "input_price_cny_per_million": args.input_price,
            "output_price_cny_per_million": args.output_price,
            "per_n_calls": args.per,
            "note": "单价为示例锚点，非任何具体供应商报价",
        },
        "usage": usage_rows,
        "latency": latency_rows,
        "cost_table": table,
        "summary": {
            "mean_tokens_per_call": round(mean_tokens, 1),
            f"mean_cost_per_{args.per}_cny": round(mean_cost, 4),
        },
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] 明细写入 {Path(args.out).name}")
    await client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
