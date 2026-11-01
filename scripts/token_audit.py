"""Token 审计：**有几句真话只能靠 bge 自己的 tokenizer 说出来**。

## 这个脚本要回答的唯一问题

> 现在这套「500 字 + overlap 50」的切块，喂进 bge 之后**有多少块超过 512 token**？

超过的部分不会报错 —— bge 的 `max_seq_length=512`，超了**静默截断**
（见 `app/rag/embedding/bge.py` 模块说明第一节）。被截断的块在检索里表现为
"这段的前 757 字能搜到，后面搜不到"，而**日志里没有任何异常**。

所以这条要有个数字，而不是一句"应该没事"。

## 为什么不能用 `字数 × 0.676` 估算

`docs/embedding.md` 第 105 行写着"按中文 0.676 tok/字算，500 字 ≈ 350 token"。
那个系数是从**另一个模型网关的 usage** 反推的（`scripts/token_result.json`），
不是 bge 的分词结果。而 bge 的 vocab 是**字级**的：实测 `付款计划` → 4 个 token，
即 1.0 tok/字，比 0.676 **高 48%**。

系数偏小的方向是确定的：它让"350 token"比真实值低，也就是让 500 字**看起来更安全**。
这正是这条要查的地方 —— 用错单位做的安全判断，只有在真数一遍之后才知道差多少。

## 输出什么

1. **越界占比**（`> 512` 的块 / 全部块）—— 这是任务要求的验收项，一个百分比。
2. **分布直方图** —— 光有占比看不出"是差一点还是差一倍"。直方图按 token 分桶，
   并把 `512` 这条线显式标出来。
3. **按 block 类型拆开** —— 这是本脚本最有用的一个视图：
   表格块整块不切（契约要求），所以越界几乎必然发生在表格上；
   把它们混在一起统计，会得到一个"平均块 124 token，溢出率 0.06%"的漂亮数字，
   而那个数字对"表格会丢内容"这件事毫无信息量。
4. **字数 → token 的真实系数** —— 用来替换文档里那个外来的 0.676。

## 为什么默认扫「4 个 chunk_chars」而不是只扫 500

任务要求的是"每个 chunk 的 token"（当前配置）。但既然 tokenizer 已经加载了，
把 300/500/800/1000 一起数一遍的增量成本几乎为零 —— 而它能直接回答
"**改到 800/1000 会不会开始大面积越界**"，这正是 20:25 那个粗筛网格要用的先验。

运行：

    uv run python -X utf8 scripts/token_audit.py
    uv run python -X utf8 scripts/token_audit.py --chunk-chars 500 --overlap 50
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from retrieval_harness import chunk_corpus, load_corpus  # noqa: E402

DEFAULT_OUT: Path = PROJECT_ROOT / "scripts" / "token_audit.json"

# 直方图分桶。**在 512 附近必须细分**：整段的价值就在那条线上，
# 用一个宽的 400-600 桶会把"499 还是 601"这个差别抹掉。
BUCKETS: tuple[tuple[int, int], ...] = (
    (0, 32),
    (32, 64),
    (64, 128),
    (128, 256),
    (256, 384),
    (384, 448),
    (448, 480),
    (480, 512),
    (512, 640),
    (640, 1024),
    (1024, 10**9),
)


def _bucket_label(low: int, high: int) -> str:
    """桶标签。最后一桶用 `1024+` 而不是 `1024-inf`，读起来像人话。"""
    return f"{low}+" if high >= 10**9 else f"{low}-{high}"


def histogram(values: list[int]) -> list[dict[str, Any]]:
    """按 `BUCKETS` 分桶，返回带计数与占比的列表（**含空桶**，否则形状会骗人）。"""
    total = len(values)
    rows: list[dict[str, Any]] = []
    for low, high in BUCKETS:
        count = sum(1 for value in values if low <= value < high)
        rows.append(
            {
                "bucket": _bucket_label(low, high),
                "low": low,
                "high": None if high >= 10**9 else high,
                "count": count,
                "share": round(count / total, 4) if total else 0.0,
                # 这一个字段是给"用眼睛看"准备的：一份 40 行的表比 11 个百分比快得多。
                "bar": "█" * max(1, round(count / total * 60)) if count else "",
            }
        )
    return rows


def audit_one(
    *,
    corpus: Any,  # noqa: ANN401
    chunk_chars: int,
    overlap: int,
    tokenizer: Any,  # noqa: ANN401
    max_tokens: int,
) -> dict[str, Any]:
    """对一组 (chunk_chars, overlap) 做完整审计。"""
    from app.rag.chunking import ChunkingConfig

    cfg = ChunkingConfig(chunk_chars=chunk_chars, overlap_chars=overlap)
    chunks = chunk_corpus(corpus, cfg)
    texts = [chunk.text for chunk in chunks]
    tokens = tokenizer.count(texts)

    over = [i for i, value in enumerate(tokens) if value > max_tokens]
    by_kind: dict[str, dict[str, Any]] = {}
    kind_counter = Counter()
    for chunk, value in zip(chunks, tokens, strict=True):
        kind = str(chunk.metadata.get("block", "text"))
        kind_counter[kind] += 1
        bucket = by_kind.setdefault(
            kind,
            {"chunks": 0, "over": 0, "tokens": [], "chars": [], "examples": []},
        )
        bucket["chunks"] += 1
        bucket["tokens"].append(value)
        bucket["chars"].append(len(chunk.text))
        if value > max_tokens:
            bucket["over"] += 1
            if len(bucket["examples"]) < 5:
                bucket["examples"].append(
                    {
                        "source": chunk.source,
                        "index": chunk.index,
                        "chars": len(chunk.text),
                        "tokens": value,
                        "head": chunk.text[:40].replace("\n", " "),
                    }
                )

    for bucket in by_kind.values():
        values = bucket.pop("tokens")
        chars = bucket.pop("chars")
        bucket["tokens_avg"] = round(sum(values) / len(values), 1) if values else 0.0
        bucket["tokens_max"] = max(values) if values else 0
        bucket["chars_max"] = max(chars) if chars else 0
        bucket["over_rate"] = (
            round(bucket["over"] / bucket["chunks"], 4) if bucket["chunks"] else 0.0
        )

    total = len(tokens)
    return {
        "chunk_chars": chunk_chars,
        "overlap": overlap,
        "chunks": total,
        "chars_total": sum(len(chunk.text) for chunk in chunks),
        "tokens_total": sum(tokens),
        "tokens_avg": round(sum(tokens) / total, 1) if total else 0.0,
        "tokens_max": max(tokens) if total else 0,
        "tokens_p50": _percentile(tokens, 50),
        "tokens_p95": _percentile(tokens, 95),
        "max_tokens": max_tokens,
        "over_count": len(over),
        # **这就是任务要的「超 512 的占比」。**
        "over_rate": round(len(over) / total, 6) if total else 0.0,
        # 系数：用它替换 docs/embedding.md 里那个外来的 0.676。
        "tokens_per_char": round(sum(tokens) / sum(len(t) for t in texts), 4) if texts else 0.0,
        "histogram": histogram(tokens),
        "by_kind": by_kind,
        "kind_counts": dict(kind_counter),
        "over_examples": [
            {
                "source": chunks[i].source,
                "index": chunks[i].index,
                "kind": str(chunks[i].metadata.get("block", "text")),
                "chars": len(chunks[i].text),
                "tokens": tokens[i],
                "head": chunks[i].text[:60].replace("\n", " "),
            }
            for i in over[:10]
        ],
    }


def _percentile(values: list[int], percent: int) -> int:
    """最近秩法分位数。**不插值**：token 是整数，"P95 = 431.6" 没有意义。"""
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(percent / 100 * len(ordered)) - 1))
    return ordered[index]


def render(report: dict[str, Any]) -> str:
    """把报告渲染成人读的文本。**表格优先**，因为要对照着写文档。"""
    lines: list[str] = []
    lines.append(f"Token 审计（bge tokenizer，max_seq_length={report[0]['max_tokens']}）")
    lines.append("=" * 78)
    header = f"{'chunk':>6} {'overlap':>8} {'块数':>7} {'平均tok':>8} {'P95':>6} {'最大':>6} {'>512':>6} {'占比':>8}"
    lines.append(header)
    lines.append("-" * 78)
    for row in report:
        lines.append(
            f"{row['chunk_chars']:>6} {row['overlap']:>8} {row['chunks']:>7} "
            f"{row['tokens_avg']:>8} {row['tokens_p95']:>6} {row['tokens_max']:>6} "
            f"{row['over_count']:>6} {row['over_rate'] * 100:>7.2f}%"
        )
    return "\n".join(lines)


def render_detail(row: dict[str, Any]) -> str:
    """单个配置的详细视图：直方图 + 按类型的拆分。"""
    lines: list[str] = []
    lines.append("")
    lines.append(
        f"▓ 明细：chunk_chars={row['chunk_chars']} / overlap={row['overlap']}"
        f"（{row['chunks']} 块，{row['tokens_total']} token，"
        f"{row['tokens_per_char']} tok/字）"
    )
    lines.append("")
    lines.append("  token 分布直方图（512 那条线是截断边界）")
    for bucket in row["histogram"]:
        mark = " ← 越界" if bucket["low"] >= row["max_tokens"] else ""
        lines.append(
            f"    {bucket['bucket']:>10} {bucket['count']:>5} "
            f"{bucket['share'] * 100:>6.2f}%  {bucket['bar']}{mark}"
        )
    lines.append("")
    lines.append("  按 block 类型拆分（这一格才是真相）")
    lines.append(
        f"    {'kind':<10} {'块数':>6} {'平均tok':>8} {'最大tok':>8} {'最大字数':>9} {'越界':>5} {'越界率':>7}"
    )
    for kind, bucket in sorted(row["by_kind"].items(), key=lambda kv: -kv[1]["chunks"]):
        lines.append(
            f"    {kind:<10} {bucket['chunks']:>6} {bucket['tokens_avg']:>8} "
            f"{bucket['tokens_max']:>8} {bucket['chars_max']:>9} "
            f"{bucket['over']:>5} {bucket['over_rate'] * 100:>6.2f}%"
        )
    if row["over_examples"]:
        lines.append("")
        lines.append("  越界样本（前 10 条）")
        for example in row["over_examples"]:
            lines.append(
                f"    {example['tokens']:>4} tok / {example['chars']:>5} 字 "
                f"[{example['kind']}] {example['source']}#{example['index']} "
                f"{example['head']}…"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用 bge 的 tokenizer 审计切块的 token 越界情况")
    parser.add_argument("--chunk-chars", type=int, default=500, help="主配置的块长（明细用）")
    parser.add_argument("--overlap", type=int, default=50, help="主配置的重叠（明细用）")
    parser.add_argument(
        "--sweep",
        type=int,
        nargs="+",
        default=[300, 500, 800, 1000],
        help="额外扫这些 chunk_chars（与粗筛网格对齐，用来提前看 800/1000 会不会越界）",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    from app.rag.embedding.tokenizer import BGETokenizer

    tokenizer = BGETokenizer()
    max_tokens = tokenizer.max_seq_length
    corpus = load_corpus()
    if corpus.warnings:
        for warning in corpus.warnings:
            print(f"  ⚠ {warning}", file=sys.stderr)

    targets: list[tuple[int, int]] = [(args.chunk_chars, args.overlap)]
    for chunk_chars in args.sweep:
        if (chunk_chars, args.overlap) not in targets:
            targets.append((chunk_chars, args.overlap))

    rows: list[dict[str, Any]] = []
    for chunk_chars, overlap in targets:
        row = audit_one(
            corpus=corpus,
            chunk_chars=chunk_chars,
            overlap=overlap,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        )
        rows.append(row)
        print(
            f"  ✓ chunk_chars={chunk_chars} overlap={overlap}：{row['chunks']} 块", file=sys.stderr
        )

    main_row = rows[0]
    print(render(rows))
    print(render_detail(main_row))

    payload = {
        "meta": {
            "tokenizer": "BAAI/bge-small-zh-v1.5（AutoTokenizer）",
            "max_seq_length": max_tokens,
            "add_special_tokens": True,
            "corpus_sources": len(corpus.prose),
            "corpus_docs": len(corpus.all_docs),
            "corpus_chars": corpus.total_chars,
            "note": "token 数为**未截断**的真实长度；> max_seq_length 的块在 bge 里会被静默截断。",
        },
        "primary": {
            "chunk_chars": main_row["chunk_chars"],
            "overlap": main_row["overlap"],
            "over_rate": main_row["over_rate"],
            "over_count": main_row["over_count"],
            "chunks": main_row["chunks"],
        },
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘：{args.out.relative_to(PROJECT_ROOT)}")

    rate = main_row["over_rate"]
    verdict = "安全" if rate <= 0.05 else "**不安全**"
    print(
        f"\n结论（chunk_chars={main_row['chunk_chars']} / overlap={main_row['overlap']}）："
        f"越界 {main_row['over_count']}/{main_row['chunks']} = {rate * 100:.2f}% → {verdict}"
    )
    print(
        f"真实 token 密度 {main_row['tokens_per_char']} tok/字；"
        "`docs/embedding.md` 里用的 0.676 是另一个模型网关的数字，不是 bge 的。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
