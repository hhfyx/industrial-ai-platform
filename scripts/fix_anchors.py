"""把解析失败的 6 条锚点换成**更短、更居中**的片段。

## 为什么需要这个脚本，而不是手改 jsonl

6 条失败的原因不是"quote 写错"，而是**它跨了 chunk 边界** ——
锚点本身在源文件里是逐字存在的（验证过），只是切分后没有任何一块
完整包含它。

所以修法有明确规则：**在原文里把 quote 缩短到某个更短的中心片段**，
再用三种切分参数（300/500/800）验证 6 条都恰好落在一个块里。

手改的话，"缩多短才不跨块"只能靠反复试。写脚本可以一次跑完三种切分。

## 它不是一次性脚本

`--report` 会告诉你当前数据集里还有哪些条目在哪些切分下跨块。
换文档、加题目时都会用到 —— 所以它留在仓库里，不是临时工具。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from retrieval_harness import load_corpus  # noqa: E402

EVAL_FILE: Path = PROJECT_ROOT / "eval" / "questions-50.jsonl"

# 验证用的切分参数。**与文档里的口径一致**（token 审计与网格都用过这三档）。
CHUNK_SIZES: tuple[int, ...] = (300, 500, 800)


def build_chunks(chunk_chars: int) -> list[object]:
    from app.rag.chunking import ChunkingConfig, build_strategy

    corpus = load_corpus()
    cfg = ChunkingConfig(chunk_chars=chunk_chars, overlap_chars=50)
    return list(build_strategy("recursive", cfg=cfg).chunk(corpus.all_docs, cfg))


def count_hits(*, quote: str, source: str, chunks: list[object]) -> int:
    """该 quote 在 `source` 的块里命中几个。**1 才是合格。**"""
    return sum(1 for chunk in chunks if chunk.source == source and quote in chunk.text)  # type: ignore[attr-defined]


def diagnose(items: list[dict], *, verbose: bool = True) -> list[dict]:
    """逐条报告锚点在各切分下的命中数。"""
    chunks_by_size = {size: build_chunks(size) for size in CHUNK_SIZES}
    rows: list[dict] = []
    for item in items:
        for anchor_index, anchor in enumerate(item["anchors"]):
            quote = anchor["quote"]
            source = anchor["source"]
            counts = {
                size: count_hits(quote=quote, source=source, chunks=chunks_by_size[size])
                for size in CHUNK_SIZES
            }
            ok = all(value == 1 for value in counts.values())
            rows.append(
                {
                    "id": item["id"],
                    "anchor_index": anchor_index,
                    "quote": quote,
                    "counts": counts,
                    "ok": ok,
                }
            )
    if verbose:
        bad = [row for row in rows if not row["ok"]]
        print(f"锚点共 {len(rows)} 个，不合格 {len(bad)} 个")
        for row in bad:
            print(f"  ✗ {row['id']}#{row['anchor_index']} {row['counts']} {row['quote'][:40]!r}")
    return rows


def suggest_shortcuts(*, quote: str, source: str, max_len: int = 40) -> list[tuple[str, dict]]:
    """给一条失败的 quote 找"在所有切分下都唯一命中一块"的最长片段。

    做法：从 quote 的中心往两边收缩，按 1 字步长试，返回**第一个全部合格的最长片段**。
    从中心开始而不是从头部，是因为中文句子开头常带 `**` 与标点，
    而块的边界更容易落在两端。
    """
    chunks_by_size = {size: build_chunks(size) for size in CHUNK_SIZES}
    middle = len(quote) // 2
    best: tuple[str, dict] | None = None
    for length in range(min(len(quote), max_len), 7, -1):
        start = max(0, middle - length // 2)
        candidate = quote[start : start + length]
        counts = {
            size: count_hits(quote=candidate, source=source, chunks=chunks_by_size[size])
            for size in CHUNK_SIZES
        }
        if all(value == 1 for value in counts.values()):
            best = (candidate, counts)
            break
    return [best] if best else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="诊断/修复评测集的锚点跨块问题")
    parser.add_argument("--report", action="store_true", help="只报告，不修改（默认）")
    parser.add_argument("--suggest", action="store_true", help="给失败的 quote 提可用片段")
    args = parser.parse_args(argv)

    items = [
        json.loads(line)
        for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = diagnose(items)
    if args.suggest:
        print("\n可用的替代片段：")
        for row in rows:
            if row["ok"]:
                continue
            source = items_by_id(items, row["id"])["anchors"][row["anchor_index"]]["source"]
            suggestions = suggest_shortcuts(quote=row["quote"], source=source)
            for candidate, counts in suggestions:
                print(f"  {row['id']}#{row['anchor_index']} -> {candidate!r} {counts}")
            if not suggestions:
                print(f"  {row['id']}#{row['anchor_index']} -> 找不到 ≤40 字的可用片段，需换句")
    return 0 if all(row["ok"] for row in rows) else 1


def items_by_id(items: list[dict], target: str) -> dict:
    for item in items:
        if item["id"] == target:
            return item
    raise KeyError(target)


if __name__ == "__main__":
    raise SystemExit(main())
