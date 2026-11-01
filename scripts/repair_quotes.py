"""把评测集里"看起来一样、字节不同"的锚点修回原文。

## 为什么必须自动修，而不是手改

子代理标注时把 `"` 写成了 `“”`（直引号 → 中文弯引号）。
两者**视觉上几乎相同**，但字节不同，于是 `quote in text` 为 False。

这类错在 50 条里出现了 8 处，而人工看 jsonl 是看不出来的 ——
只有"拿去和原文比对"才会暴露。所以修法必须是程序化的：

    对每条 quote，在源文件里找**忽略标点差异**后的匹配位置，
    然后**把源文件里的原文片段替换回去**。

这样修出来的 quote 一定是逐字原文，而不是"我以为的原文"。

## 归一化只用于**定位**，不用于存储

定位时把两侧都做"引号/破折号/全半角统一"，找到位置后取**原文的切片**。
如果反过来（把归一化后的字符串存进去），下一次比对又会失败 ——
因为归一化是有损的。

## 找不到就报错，不改

"找不到"说明这条不是引号问题（可能真的跨块或抄错了），
那种情况必须人看。**自动修只处理"只差标点"这一类**。
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

# 标点归一化表。**只覆盖"视觉相同"的那几类** ——
# 把整个 Unicode 标点都映射进来会让定位变得过于宽松，可能匹配到错误位置。
_PUNCT_MAP = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
        "，": ",",
        "。": ".",
        "：": ":",
        "；": ";",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
    }
)


def normalize(text: str) -> str:
    """把"视觉相同但字节不同"的标点统一。**只用于定位。**"""
    return text.translate(_PUNCT_MAP)


def find_in_source(*, raw: str, quote: str) -> str | None:
    """在原文里定位 quote，返回**原文切片**（不是归一化后的串）。

    先试精确匹配（最快、最常见）；失败再用归一化定位。
    """
    if quote in raw:
        return quote

    # 归一化定位：在归一化后的原文里找归一化后的 quote，
    # 再用**同一个下标区间**去原文切 —— 因为归一化是 1:1 字符替换，
    # 长度不变，所以下标可以直接映射。
    normalized_raw = normalize(raw)
    normalized_quote = normalize(quote)
    position = normalized_raw.find(normalized_quote)
    if position < 0:
        return None
    return raw[position : position + len(quote)]


def build_chunks(chunk_chars: int) -> list[object]:
    from app.rag.chunking import ChunkingConfig, build_strategy

    corpus = load_corpus()
    cfg = ChunkingConfig(chunk_chars=chunk_chars, overlap_chars=50)
    return list(build_strategy("recursive", cfg=cfg).chunk(corpus.all_docs, cfg))


def count_hits(*, quote: str, source: str, chunks: list[object]) -> int:
    return sum(
        1
        for chunk in chunks
        if chunk.source == source and quote in chunk.text  # type: ignore[attr-defined]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把锚点 quote 修回逐字原文")
    parser.add_argument("--dry-run", action="store_true", help="只报告，不写入")
    args = parser.parse_args(argv)

    items = [
        json.loads(line)
        for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    fixed: list[tuple[str, str, str]] = []
    unfixable: list[tuple[str, str]] = []

    for item in items:
        for index, anchor in enumerate(item["anchors"]):
            source = anchor["source"]
            quote = anchor["quote"]
            path = PROJECT_ROOT / "docs" / source
            if not path.exists():
                path = PROJECT_ROOT / source
            raw = path.read_text(encoding="utf-8")

            corrected = find_in_source(raw=raw, quote=quote)
            if corrected is None:
                unfixable.append((f"{item['id']}#{index}", quote[:40]))
                continue
            if corrected != quote:
                anchor["quote"] = corrected
                # head12 也要跟着改，否则两处对不上
                anchor["head12"] = corrected[:12]
                fixed.append((f"{item['id']}#{index}", quote[:30], corrected[:30]))

    print(f"标点修正 {len(fixed)} 处，无法定位 {len(unfixable)} 处")
    for qid, old, new in fixed:
        print(f"  ✓ {qid}\n      旧: {old!r}\n      新: {new!r}")
    for qid, quote in unfixable:
        print(f"  ✗ {qid} 在源文件里找不到（不是标点问题）：{quote!r}")

    if not args.dry_run and fixed:
        # 逐行重写，保持一行一条的格式
        EVAL_FILE.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n",
            encoding="utf-8",
        )
        print(f"\n已写回 {EVAL_FILE.relative_to(PROJECT_ROOT)}")

    # 修完后在三种切分下复查
    if not args.dry_run and fixed:
        print("\n三种切分下复查：")
        chunks_by_size = {size: build_chunks(size) for size in CHUNK_SIZES}
        bad = 0
        for item in items:
            for index, anchor in enumerate(item["anchors"]):
                counts = {
                    size: count_hits(
                        quote=anchor["quote"], source=anchor["source"], chunks=chunks_by_size[size]
                    )
                    for size in CHUNK_SIZES
                }
                if any(value != 1 for value in counts.values()):
                    bad += 1
                    print(f"  ✗ {item['id']}#{index} {counts} {anchor['quote'][:36]!r}")
        print(f"仍不合格 {bad} 个")

    return 0 if not unfixable else 1


if __name__ == "__main__":
    raise SystemExit(main())
