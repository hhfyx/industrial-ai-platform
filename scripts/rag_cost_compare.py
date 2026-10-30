"""长上下文 vs RAG：把三个"量级"算成同一把尺子上可比的数字。

**这个脚本只做算术，不调模型。** 理由和 `cost_model.py` 一样：
数字的可靠性来自"输入是实测的"，不来自"又跑了一次"。

## 它回答的问题

「把整本书塞进 prompt」能不能替代 RAG？这个问题的三个理由里，
有两个是**可以量化的**（成本、注意力衰减），一个是**结构性的**
（无法更新与溯源）。脚本负责前两个，第三个只能靠论证 —— 见
`docs/design-decisions.md` 第 0 条。

## 输入从哪来（每个数字都能追溯）

| 符号 | 来源 |
| --- | --- |
| `chars_per_token` = 0.676 | `scripts/token_result.json` 中文实测（`learn/llm/03_cost_and_reliability.md`） |
| 50 页 / 5437 字 | `tests/fixtures/sample50.pdf` 实测（本脚本会重新量一遍并打印） |
| 单价 | 命令行参数，**故意不是任何供应商的报价** |

## 算的是什么

1. **线性成本**：同一份文档用"全书塞入"vs"RAG 取 4 块"，每次提问的输入 token 差几倍
2. **注意力摊薄**：给定一个窗口，塞 100% 满时"每个 token 分到多少注意力预算"
3. **更新代价**：文档改一页，两种方案分别要重算什么

运行：

    uv run python -X utf8 scripts/rag_cost_compare.py --price-in 2 --price-out 8
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

# 中文实测 token 密度（scripts/token_result.json，buddy/glm-5.3）
CHARS_PER_TOKEN: float = 0.676

# 与 app/rag/generation/base.py 的 DEFAULT_CONTEXT_CHARS 对齐。
# 不 import 那个常量是刻意的：这份脚本要能在"app 包坏了"时仍然跑出数字，
# 否则"验证工具依赖被验证对象"会让排查变成鸡生蛋。
RAG_CONTEXT_CHARS: int = 6000

# 典型问答的答案长度（输出 token）。RAG 与长上下文都一样 ——
# 差别只在输入侧，所以输出侧固定，不制造虚假差异。
OUTPUT_TOKENS: int = 200

# **真实文档的页密度假设**（不是实测，明确标注）。
# 为什么需要它：仓库里的 `sample50.pdf` 是脚本生成的合成样本，每页只有
# 108 字（`make_fixtures.py` 为了让 review 可读而刻意做短）。用一份"每页 100 字"
# 的样本去比"全书塞入 vs RAG"，结论会退化成"1x 没差别"—— 那是被样本骗了。
# 真实 A4 中文合同/报告的页密度通常在 800–1600 字，这里取 1200 作为中间值。
# **它只影响"1x 时差几倍"这一格，不影响线性放大的斜率。**
ASSUMED_CHARS_PER_PAGE: int = 1200


def measure_fixture() -> dict[str, Any]:
    """实测 50 页 PDF 的字符数。**不硬编数字** —— fixture 改了这里会跟着变。"""
    from app.rag.loaders import load_document

    path = PROJECT_ROOT / "tests" / "fixtures" / "sample50.pdf"
    result = load_document(path)
    chars = result.total_chars
    return {
        "file": path.name,
        "pages": result.page_count,
        "chars": chars,
        "chars_per_page": round(chars / result.page_count, 1),
    }


def tokens_of(chars: int) -> int:
    """字符 → token（向上取整）。用实测密度而不是"4 字 1 token"的英文经验值。"""
    return int(chars / CHARS_PER_TOKEN) + 1


def compare(
    *,
    chars: int,
    pages: int,
    price_in: float,
    price_out: float,
    context_chars: int = RAG_CONTEXT_CHARS,
    output_tokens: int = OUTPUT_TOKENS,
) -> dict[str, Any]:
    """核心对比：同一份文档，"全书塞入"vs"RAG 取 4 块"。"""
    full_tokens = tokens_of(chars)
    rag_tokens = tokens_of(min(context_chars, chars))

    full_cost = (full_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out
    rag_cost = (rag_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out

    return {
        "document": {"pages": pages, "chars": chars},
        "full_context": {"input_tokens": full_tokens, "cost_per_query": round(full_cost, 6)},
        "rag": {"input_tokens": rag_tokens, "cost_per_query": round(rag_cost, 6)},
        "ratio_input_tokens": round(full_tokens / rag_tokens, 2),
        "ratio_cost": round(full_cost / rag_cost, 2),
        # 每 1,000 次提问的差价 —— 这是"倍数"落到账本上的样子
        "extra_cost_per_1000_queries": round((full_cost - rag_cost) * 1000, 2),
        "output_tokens_assumed": output_tokens,
    }


def scaling(
    base_chars: int,
    *,
    base_pages: int,
    multipliers: tuple[int, ...] = (1, 10, 100, 1000),
) -> list[dict[str, Any]]:
    """线性放大：1x / 10x / 100x / 1000x 文档量下的输入 token。

    **这一列是"成本随长度线性涨"的证据**：RAG 的输入 token 在 `context_chars`
    处饱和（永远只放那么多），而全书塞入随文档量线性增长 —— 两条曲线必然交叉，
    交叉点之后就是"不可用"。找出"几倍于 50 页时会超过典型窗口"这件事，
    比一句"会很长"有用得多。
    """
    rows: list[dict[str, Any]] = []
    for mult in multipliers:
        chars = base_chars * mult
        full_tokens = tokens_of(chars)
        rows.append(
            {
                "multiplier": mult,
                "pages": base_pages * mult,
                "chars": chars,
                "full_context_input_tokens": full_tokens,
                "rag_input_tokens": tokens_of(min(RAG_CONTEXT_CHARS, chars)),
                "fits_128k": full_tokens <= 128_000,
                "fits_1m": full_tokens <= 1_000_000,
            }
        )
    return rows


def attention_budget(context_tokens: int, window_tokens: int = 128_000) -> dict[str, Any]:
    """注意力摊薄的量化版本。

    **这不是"注意力机制的真实复杂度"**，是一个刻意简化的代理指标：
    把窗口占用率当作"每个 token 平均能分到的注意力份额"的倒数。
    它能回答的问题只有一个 —— **"塞满 90% 和塞满 5%，中间部分的 token
    被读到的机会差几倍"**。用"lost in the middle"的实验结论对照它：
    U 形曲线意味着中间那一段最容易被跳过，而"塞得越满，中间越长"。

    基线取 3%（≈ 4 块 × 500 字 ≈ 1400 token / 128k）—— 那是 RAG 的典型占用率，
    不是随便挑的数字。**基线一变，倍数就变**，所以它必须和 RAG 的默认参数挂钩。
    """
    occupancy = min(context_tokens / window_tokens, 1.0)
    return {
        "context_tokens": context_tokens,
        "window_tokens": window_tokens,
        "occupancy": round(occupancy, 4),
        "dilution_vs_3pct": round(max(occupancy, 1e-9) / 0.03, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="长上下文 vs RAG 的量化对比")
    parser.add_argument("--price-in", type=float, default=2.0, help="输入单价（元/百万 token）")
    parser.add_argument("--price-out", type=float, default=8.0, help="输出单价（元/百万 token）")
    parser.add_argument(
        "--chars-per-page",
        type=int,
        default=ASSUMED_CHARS_PER_PAGE,
        help="真实文档每页字符数（假设值，见 ASSUMED_CHARS_PER_PAGE 的说明）",
    )
    parser.add_argument("--json", type=Path, default=None, help="把结果写到 json")
    args = parser.parse_args()

    fixture = measure_fixture()
    real_chars = args.chars_per_page * fixture["pages"]
    payload: dict[str, Any] = {
        "meta": {
            "price_in_per_mtok": args.price_in,
            "price_out_per_mtok": args.price_out,
            "chars_per_token": CHARS_PER_TOKEN,
            "assumed_chars_per_page": args.chars_per_page,
            "rag_context_chars": RAG_CONTEXT_CHARS,
            "note": (
                "单价是参数、不是任何供应商的报价；字符/token 密度来自 "
                "scripts/token_result.json 实测；每页字数为假设值，见脚本常量说明"
            ),
        },
        "fixture": fixture,
        # 场景 A：仓库里的真实样本（每页 108 字）。会显示"50 页时还看不出差距"——
        # **这个诚实的结果必须留在报告里**，否则 10x 那个数字就成了挑选过的证据。
        "scenario_fixture": compare(
            chars=fixture["chars"],
            pages=fixture["pages"],
            price_in=args.price_in,
            price_out=args.price_out,
        ),
        # 场景 B：按真实 A4 密度换算的同一份 50 页文档。
        "scenario_realistic": compare(
            chars=real_chars,
            pages=fixture["pages"],
            price_in=args.price_in,
            price_out=args.price_out,
        ),
        "scaling": scaling(fixture["chars"], base_pages=fixture["pages"]),
        "scaling_realistic": scaling(
            args.chars_per_page * 1, base_pages=1, multipliers=(50, 500, 5000)
        ),
        "attention_realistic": attention_budget(tokens_of(real_chars)),
        "attention_500p_realistic": attention_budget(tokens_of(real_chars * 10)),
    }

    print(
        f"样本：{fixture['file']}，{fixture['pages']} 页，{fixture['chars']} 字"
        f"（{fixture['chars_per_page']} 字/页 —— 合成样本，每页偏短）"
    )
    print(f"密度：{CHARS_PER_TOKEN} tok/字\n")

    for title, key in (
        ("场景 A · 仓库合成样本（每页 108 字，**实测**）", "scenario_fixture"),
        (
            f"场景 B · 同样 50 页按真实密度（每页 {args.chars_per_page} 字，**假设**）",
            "scenario_realistic",
        ),
    ):
        block = payload[key]
        print(f"① {title}")
        print(
            f"   全书塞入 : {block['full_context']['input_tokens']:>7} token"
            f"  {block['full_context']['cost_per_query']:.6f} 元"
        )
        print(
            f"   RAG 取 4 : {block['rag']['input_tokens']:>7} token"
            f"  {block['rag']['cost_per_query']:.6f} 元"
        )
        print(
            f"   → 输入 token 差 {block['ratio_input_tokens']}x；"
            f"每 1000 次多花 {block['extra_cost_per_1000_queries']} 元\n"
        )

    print("② 线性放大（RAG 输入在上下文预算处饱和，全书塞入不会）")
    print(f"   {'页数':>7} {'全书输入tok':>12} {'RAG输入tok':>11} {'128k窗口':>9} {'1M窗口':>8}")
    for row in payload["scaling_realistic"]:
        print(
            f"   {row['pages']:>7} {row['full_context_input_tokens']:>12} "
            f"{row['rag_input_tokens']:>11} "
            f"{'✓' if row['fits_128k'] else '✗':>9} "
            f"{'✓' if row['fits_1m'] else '✗':>8}"
        )

    print("\n③ 注意力摊薄（代理指标，见函数 docstring；基于场景 B）")
    for label, key in (("50 页", "attention_realistic"), ("500 页", "attention_500p_realistic")):
        info = payload[key]
        print(
            f"   {label}: 占窗口 {info['occupancy'] * 100:.2f}%，"
            f"相对 3% 占用稀释 {info['dilution_vs_3pct']}x"
        )

    if args.json is not None:
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n已写入 {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
