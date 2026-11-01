"""为"跨块/重复"的锚点找可用片段。

## 为什么会有"命中 2 块"这种情况

`quote` 在源文件里只出现 **1 次**（已验证），但切分时 `overlap=50` 会把
上一块的尾部复制到下一块开头 —— 于是同一段文字**同时存在于相邻两块里**。

这不是数据错误，是重叠切分的必然结果。而它对评测的影响是实打实的：
**锚点不再唯一，gold 就有两个候选**，"命中哪一块"不可复现。

## 两种修法，优先第一种

1. **缩短 quote**：缩到不跨越重叠区的长度（通常 12~20 字就够）。
   这是首选 —— 它保留同一句话，只是不再贪长。
2. **换一句**：如果缩短后仍不唯一（说明该短语在文档里重复），换一处。

本脚本从 quote 中心向外收缩，逐步试到 8 字，返回第一个在
**300/500/800 三种切分下都恰好命中 1 块**的片段。
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
CHUNK_SIZES: tuple[int, ...] = (300, 500, 800)


def build_chunks(chunk_chars: int) -> list[object]:
    from app.rag.chunking import ChunkingConfig, build_strategy

    corpus = load_corpus()
    cfg = ChunkingConfig(chunk_chars=chunk_chars, overlap_chars=50)
    return list(build_strategy("recursive", cfg=cfg).chunk(corpus.all_docs, cfg))


def counts_for(
    *, quote: str, source: str, chunks_by_size: dict[int, list[object]]
) -> dict[str, int]:
    result: dict[str, int] = {}
    for size, chunks in chunks_by_size.items():
        result[str(size)] = sum(
            1
            for chunk in chunks
            if chunk.source == source and quote in chunk.text  # type: ignore[attr-defined]
        )
    return result


def find_usable(
    *, quote: str, source: str, chunks_by_size: dict[int, list[object]]
) -> tuple[str, dict[str, int]] | None:
    """从中心收缩，返回第一个在三种切分下都唯一命中的片段。"""
    middle = len(quote) // 2
    # 从全长往下试；**先试中心**，因为边界最可能落在两端
    for length in range(len(quote), 5, -1):
        start = max(0, middle - length // 2)
        candidate = quote[start : start + length].strip()
        # 首尾若是标点/反引号，去掉再试（它们不影响语义但会降低可读性）
        candidate = candidate.strip("*`，。、；：\"'（）()")
        if len(candidate) < 6:
            continue
        counts = counts_for(quote=candidate, source=source, chunks_by_size=chunks_by_size)
        if all(value == 1 for value in counts.values()):
            return candidate, counts
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="为不唯一的锚点找可用片段")
    parser.add_argument("--apply", action="store_true", help="直接写回数据集")
    args = parser.parse_args(argv)

    items = [
        json.loads(line)
        for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunks_by_size = {size: build_chunks(size) for size in CHUNK_SIZES}

    changed = 0
    unresolved: list[str] = []
    for item in items:
        for index, anchor in enumerate(item["anchors"]):
            counts = counts_for(
                quote=anchor["quote"], source=anchor["source"], chunks_by_size=chunks_by_size
            )
            if all(value == 1 for value in counts.values()):
                continue

            found = find_usable(
                quote=anchor["quote"], source=anchor["source"], chunks_by_size=chunks_by_size
            )
            if found is None:
                unresolved.append(f"{item['id']}#{index}")
                print(f"  ✗ {item['id']}#{index} 无法缩短到唯一，需换句：{anchor['quote'][:40]!r}")
                continue

            candidate, new_counts = found
            print(f"  ✓ {item['id']}#{index}")
            print(f"      旧 ({counts}) {anchor['quote'][:40]!r}")
            print(f"      新 ({new_counts}) {candidate!r}")
            anchor["quote"] = candidate
            anchor["head12"] = candidate[:12]
            changed += 1

    if args.apply and changed:
        EVAL_FILE.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n",
            encoding="utf-8",
        )
        print(f"\n已写回（{changed} 处）")
    elif changed:
        print("\n（未写回，加 --apply 生效）")

    if unresolved:
        print(f"\n仍需人工处理：{unresolved}")
    return 0 if not unresolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
