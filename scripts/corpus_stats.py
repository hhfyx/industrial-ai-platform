"""打印当前语料的权威规模（篇数 / 段数 / 块数）。

## 用法

```bash
python -X utf8 scripts/corpus_stats.py         # 给人看
python -X utf8 scripts/corpus_stats.py --json  # 给 CI / 测试
```

## 这个脚本解决什么

2026-11-04 核实发现同一件事在文档里有**四种数字**：

    README L341      1365 块
    README L376/415  1364 块
    README L433      1160 块
    retrieval-baseline.json  1592 块

而实测是 **814 块**（26 篇 / 463 段 / chunk_chars=500, overlap=50）。

四个数字都不是编的 —— 它们对应**四个不同时间点**的语料与参数。
问题在于**没有一个地方能回答"现在是多少"**。

这个脚本就是那个地方。文档里的数字从它的输出复制，
`tests/test_corpus_stats.py` 会校验两边一致。

## 为什么块数会变（不要把变化当 bug）

- 语料篇数变化（新增文档、或删掉污染检索的分析类文档）
- 分块规则变化（如 2026-11-04 的短块合并让 Markdown 报告 105 块 → 60 块）
- 分块参数变化（300/500/1000 三档差异很大：1068/814/637）

**所以任何写进文档的块数都必须带坐标** (篇数, chunk_chars, overlap)，
否则它就是一个无法核对的孤立数字 —— 这正是原来那四个数字的问题。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.corpus_stats import (  # noqa: E402
    EVAL_CHUNK_CHARS,
    EVAL_OVERLAP_CHARS,
    REFERENCE_CONFIGS,
    measure,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="打印语料的权威规模")
    parser.add_argument("--json", action="store_true", help="输出 JSON（给 CI 用）")
    parser.add_argument(
        "--cached",
        action="store_true",
        help="用上次的缓存，不重新切分（快，但可能过时）",
    )
    args = parser.parse_args(argv)

    stats = measure(rebuild=not args.cached)

    if args.json:
        payload = {
            "documents": stats.documents,
            "segments": stats.segments,
            "chunks": stats.chunks,
            "chunk_chars": stats.chunk_chars,
            "overlap_chars": stats.overlap_chars,
            "by_chunk_chars": stats.by_chunk_chars,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("=" * 66)
    print("语料规模（实测，非手写）")
    print("=" * 66)
    print(f"  文档篇数    : {stats.documents}")
    print(f"  段数        : {stats.segments}（loader 产出，切块前）")
    print(f"  块数        : {stats.chunks}  ← 评测口径")
    print(f"  分块参数    : chunk_chars={stats.chunk_chars}, overlap={stats.overlap_chars}")
    print()
    print("  三档对照：")
    for cc, ov in REFERENCE_CONFIGS:
        n = stats.by_chunk_chars.get(cc, "?")
        mark = " ← 评测口径" if cc == EVAL_CHUNK_CHARS else ""
        print(f"    chunk_chars={cc:4d} overlap={ov:3d} -> {n} 块{mark}")
    print()
    print("=" * 66)
    print("可直接粘进文档的一行：")
    print(f"  {stats.as_badge()}")
    print("=" * 66)
    print()
    print("⚠ 写进文档时必须带 (篇数, chunk_chars, overlap) 三个坐标。")
    print("  只写『XXX 块』的数字无法核对 —— 见本脚本 docstring。")
    if stats.overlap_chars != EVAL_OVERLAP_CHARS:
        print(f"  ⚠ 注意：EVAL_OVERLAP_CHARS={EVAL_OVERLAP_CHARS} 与实测用的不一致！")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
