"""把"正好落在块尾"的锚点**向外扩展**，让它离开重叠区。

## 病症与病根

`shorten_anchors.py` 缩短无用，说明问题不在长度而在**位置**：

    实测 f-19：
      chunk#91  quote at 419..459   ← 恰好结束在块尾（len=459）
      chunk#92  quote at   8..48    ← overlap 把它复制到下一块开头

`overlap=50` 会把上一块的尾部 50 字复制到下一块开头（`delimiters._merge` 的实现）。
所以**任何贴着块尾结束的 quote 都必然命中 2 块**，缩短也没有用 ——
因为它的右边界始终贴着块尾。

## 修法：向前、向后各延伸一点

把 quote 的**右边界推进到块内**（向后取更多原文），让它在两块里都不再完整出现：

- 在 chunk#91 里：右边界越过块尾 → 不再命中
- 在 chunk#92 里：左边界前移 → 不再命中（因为下一块开头只有尾部 50 字）

所以只要向后取到"超过 overlap 长度"，就能唯一化。本脚本从左边界固定、
右边界逐步增加，找到第一个唯一命中的延伸版。

**优先向前延伸（加左边界）**：它更能保证语义完整，且不依赖 overlap 的具体值。
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
    return {
        str(size): sum(
            1
            for chunk in chunks
            if chunk.source == source and quote in chunk.text  # type: ignore[attr-defined]
        )
        for size, chunks in chunks_by_size.items()
    }


def read_source(source: str) -> str:
    for candidate in (PROJECT_ROOT / "docs" / source, PROJECT_ROOT / source):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(source)


def expand_right(
    *,
    quote: str,
    source: str,
    chunks_by_size: dict[int, list[object]],
    max_extra: int = 120,
) -> tuple[str, dict[str, int]] | None:
    """固定左边界，向右延伸，直到唯一命中。"""
    raw = read_source(source)
    start = raw.find(quote)
    if start < 0:
        return None
    for extra in range(1, max_extra + 1):
        candidate = raw[start : start + len(quote) + extra].strip()
        counts = counts_for(quote=candidate, source=source, chunks_by_size=chunks_by_size)
        if all(value == 1 for value in counts.values()):
            return candidate, counts
    return None


def expand_left(
    *,
    quote: str,
    source: str,
    chunks_by_size: dict[int, list[object]],
    max_extra: int = 120,
) -> tuple[str, dict[str, int]] | None:
    """固定右边界，向左延伸 —— **优先这个**，语义更完整。"""
    raw = read_source(source)
    end = raw.find(quote)
    if end < 0:
        return None
    end += len(quote)
    for extra in range(1, max_extra + 1):
        start = max(0, end - len(quote) - extra)
        candidate = raw[start:end].strip()
        counts = counts_for(quote=candidate, source=source, chunks_by_size=chunks_by_size)
        if all(value == 1 for value in counts.values()):
            return candidate, counts
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="向外扩展落在块尾的锚点")
    parser.add_argument("--apply", action="store_true")
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

            # 先向左（语义完整），再向右
            found = expand_left(
                quote=anchor["quote"], source=anchor["source"], chunks_by_size=chunks_by_size
            ) or expand_right(
                quote=anchor["quote"], source=anchor["source"], chunks_by_size=chunks_by_size
            )
            if found is None:
                unresolved.append(f"{item['id']}#{index}")
                print(f"  ✗ {item['id']}#{index} 扩展后仍不唯一：{anchor['quote'][:40]!r}")
                continue

            candidate, new_counts = found
            print(f"  ✓ {item['id']}#{index}")
            print(f"      旧 ({counts}) {anchor['quote'][:44]!r}")
            print(f"      新 ({new_counts}) {candidate[:44]!r}")
            anchor["quote"] = candidate
            anchor["head12"] = candidate[:12]
            changed += 1

    if args.apply and changed:
        EVAL_FILE.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n",
            encoding="utf-8",
        )
        print(f"\n已写回（{changed} 处）")
    if unresolved:
        print(f"\n仍需人工：{unresolved}")
    return 0 if not unresolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
