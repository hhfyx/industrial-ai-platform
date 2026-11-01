"""拒答演示：**用真实数据证明"阈值挡不住的那批，形态判据能不能挡住"**。

## 为什么要有这个脚本（而不是只有测试）

单元测试能证明 `decide()` 的每个分支都对，但证明不了
**上线之后真的少误答了几条**。而后者才是周二那个结论的兑现：

    threshold_scan 实测：5 条"库里没有"的问题，默认阈值 0.40 只挡住 1 条。
    其中 rf-02 拿 0.6577，比 17/20 条真答案还高。

只调阈值是死路（同一个表里：阈值升到 0.50 才挡住 3 条，
而 0.60 就开始误拒真答案）。所以这里要量的是：

    单纯阈值  vs  阈值 + 证据形态     两者的「误拒 / 误放」各多少

## 两种错误的代价不对称

| 错误 | 表现 | 代价 |
| --- | --- | --- |
| 误拒（false refusal） | 有答案却拒了 | 用户看到的直接服务失败 |
| 误放（false accept） | 没答案却答了 | 模型拿着不相关材料硬答，**答案看起来有依据** |

任务点名要防的是后者（W4 的 F3 引用越界）。所以本脚本报的是
**两个数分开**，不合成一个"准确率" —— 合成会把权衡藏起来。

## 演示的输出长什么样

每条问题一行：`id | top1 分数 | 文档数 | 分差 | 判定 | 理由`。
拒答的条目额外打印**建议动作** —— 那是用户唯一能看到的东西，
也是"拒答可演示"这条验收的实际内容。

运行：

    uv run python -X utf8 scripts/refusal_demo.py
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

import threshold_scan as ts  # noqa: E402
from retrieval_harness import PROJECT_ROOT, load_gold  # noqa: E402

# 从 `threshold_scan` 复用的常量与入口：**刻意共享而不是各抄一份** ——
# 两个脚本如果各自定一套 chunk/overlap/top_k，"为什么两边的分数分布对不上"
# 就会变成一个查不出原因的谜。
THRESHOLD_OUT: Path = ts.DEFAULT_OUT
FIXED_CHUNK_CHARS = ts.FIXED_CHUNK_CHARS
FIXED_OVERLAP = ts.FIXED_OVERLAP
FIXED_TOP_K = ts.FIXED_TOP_K
build_index = ts.build_index
load_refusal_gold = ts.load_refusal_gold

DEFAULT_OUT: Path = PROJECT_ROOT / "scripts" / "refusal_demo.json"


def _hits_for(*, question: str, store: Any, embedder: Any) -> tuple[list[Any], list[float]]:  # noqa: ANN401
    """拿完整的 top_k 命中（不是只拿 top1）—— 形态判据需要全部。

    `score_one` 只返回 top1，而"几个文档""分差多少"都要看整个列表。
    这是本脚本与 `threshold_scan` 的唯一区别：**它必须拿到全部命中**。
    """
    vector = embedder.embed_query([question])[0]
    hits = store.search(vector, top_k=FIXED_TOP_K)
    return [hit.chunk for hit in hits], [hit.score for hit in hits]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拒答策略演示（阈值 vs 阈值+形态）")
    parser.add_argument("--embedder", choices=["bge", "hashing"], default="bge")
    parser.add_argument("--threshold", type=float, default=None, help="默认读 RagConfig")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.refusal import decide
    from app.rag.settings import get_rag_config

    threshold = args.threshold if args.threshold is not None else get_rag_config().score_threshold

    if args.embedder == "hashing":
        print("⚠ hashing 下分数无可分性，仅验证流程", file=sys.stderr)
        embedder: Any = get_embedder("hashing")
    else:
        embedder = CachedEmbedder(get_embedder("bge-small-zh"))

    store, chunks = build_index(embedder=embedder)
    answerable = load_gold()
    unanswerable = load_refusal_gold()
    print(
        f"索引 {len(chunks)} 块；应答 {len(answerable)} 条，应拒 {len(unanswerable)} 条；"
        f"阈值 {threshold}",
        file=sys.stderr,
    )

    rows: list[dict[str, Any]] = []

    def run(items: list[Any], *, expect: str) -> None:
        for item in items:
            question = item.question if hasattr(item, "question") else item["question"]
            query_id = item.id if hasattr(item, "id") else item["id"]
            hit_chunks, scores = _hits_for(question=question, store=store, embedder=embedder)
            verdict = decide(hit_chunks, scores, threshold=threshold, top_k=FIXED_TOP_K)
            rows.append(
                {
                    "id": query_id,
                    "expect": expect,
                    "question": question,
                    "kind": verdict.kind.value,
                    "refused": verdict.refused,
                    "top_score": round(max(scores), 4) if scores else None,
                    "distinct_sources": verdict.diagnostics.get("distinct_sources"),
                    "margin": verdict.diagnostics.get("score_margin"),
                    "advice": list(verdict.advice),
                    "text": verdict.text,
                }
            )

    run(answerable, expect="answer")
    run(unanswerable, expect="refuse")

    # ---- 三条流水线：纯阈值 / 阈值+形态（已撤销，留作对照）/ 只看形态
    def verdict_of_pure_threshold(row: dict[str, Any]) -> bool:
        """只比 top1 与阈值 —— 这就是现在的实现（形态判据已撤销）。返回"是否拒答"。"""
        return row["top_score"] is None or row["top_score"] < threshold

    def verdict_of_shape_as_well(row: dict[str, Any]) -> bool:
        """**已被撤销的方案**：阈值 + 形态（文档太散 / 分差太小 也算拒答）。

        留着算一遍是为了让"为什么撤销"有数字，而不是一句"试过没用"。
        它不是可选项 —— 这一列的意义就是展示它的代价。
        """
        if verdict_of_pure_threshold(row):
            return True
        spread = row.get("distinct_sources")
        margin = row.get("margin")
        if isinstance(spread, int) and spread > 3:
            return True
        return bool(isinstance(margin, float) and margin < 0.02)

    pipelines = {
        "纯阈值（现方案）": verdict_of_pure_threshold,
        "阈值+形态（已撤销）": verdict_of_shape_as_well,
    }

    print()
    print(
        f"拒答演示（chunk={FIXED_CHUNK_CHARS}/overlap={FIXED_OVERLAP}/top_k={FIXED_TOP_K}，"
        f"阈值 {threshold}）"
    )
    print()
    print("  应答的 20 条：被拒了就是误拒")
    print(f"    {'id':<8} {'score':>7} {'文档数':>6} {'分差':>8}  判定")
    for row in [r for r in rows if r["expect"] == "answer"]:
        mark = "  ← 误拒" if row["refused"] else ""
        margin = f"{row['margin']:.4f}" if isinstance(row["margin"], float) else "—"
        print(
            f"    {row['id']:<8} {row['top_score']:>7.4f} {str(row['distinct_sources']):>6} "
            f"{margin:>8}  {row['kind']}{mark}"
        )

    print()
    print("  应拒的 5 条：答了就是误放")
    print(f"    {'id':<8} {'score':>7} {'文档数':>6} {'分差':>8}  判定")
    for row in [r for r in rows if r["expect"] == "refuse"]:
        mark = "  ← 误放！" if not row["refused"] else ""
        margin = f"{row['margin']:.4f}" if isinstance(row["margin"], float) else "—"
        print(
            f"    {row['id']:<8} {row['top_score']:>7.4f} {str(row['distinct_sources']):>6} "
            f"{margin:>8}  {row['kind']}{mark}"
        )

    print()
    print("  三条流水线对比")
    print(f"    {'流水线':<20} {'误拒':>6} {'误拒率':>8} {'误放':>6} {'误放率':>8}")
    summary: dict[str, Any] = {}
    for name, decide_refuse in pipelines.items():
        false_refusal = sum(1 for row in rows if row["expect"] == "answer" and decide_refuse(row))
        false_accept = sum(
            1 for row in rows if row["expect"] == "refuse" and not decide_refuse(row)
        )
        n_ans = sum(1 for row in rows if row["expect"] == "answer")
        n_ref = sum(1 for row in rows if row["expect"] == "refuse")
        summary[name] = {
            "false_refusal": false_refusal,
            "false_accept": false_accept,
            "false_refusal_rate": round(false_refusal / n_ans, 4) if n_ans else 0.0,
            "false_accept_rate": round(false_accept / n_ref, 4) if n_ref else 0.0,
        }
        print(
            f"    {name:<20} {false_refusal:>6} "
            f"{summary[name]['false_refusal_rate'] * 100:>7.1f}% {false_accept:>6} "
            f"{summary[name]['false_accept_rate'] * 100:>7.1f}%"
        )

    # ---- 演示拒答文案（"拒答可演示"这条验收的实物）
    print()
    print("  拒答文案示例（用户实际看到的内容）")
    shown = 0
    for row in rows:
        if row["expect"] != "refuse" or not row["refused"] or shown >= 2:
            continue
        shown += 1
        print(f"    ── {row['id']}（{row['kind']}，top1={row['top_score']}）")
        for line in row["text"].split("\n"):
            print(f"       {line}")
    if shown == 0:
        print("    （没有可展示的拒答案例 —— 说明形态判据在这批数据上没拦住任何一条）")

    payload = {
        "meta": {
            "chunk_chars": FIXED_CHUNK_CHARS,
            "overlap": FIXED_OVERLAP,
            "top_k": FIXED_TOP_K,
            "threshold": threshold,
            "embedder": args.embedder,
            "chunks": len(chunks),
            "rows": len(rows),
        },
        "rows": rows,
        "pipelines": summary,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 与阈值扫描的产物互相引用，避免两份数据各自被当成"唯一真相"
    if THRESHOLD_OUT.exists():
        print(f"\n  阈值分布详见：{THRESHOLD_OUT.relative_to(PROJECT_ROOT)}")
    print(f"落盘：{args.out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
