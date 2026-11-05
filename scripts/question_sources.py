"""W7 事实型黄金集出题助手：从干净文档抽段，供人工出题参考。

不做任何自动出题 —— 题目必须人写（这是评测集，不是生成物）。
本脚本只做三件事：
1. 按"从未出题的 13 份"清单抽出候选段（过滤掉目录/表格头/过短段）
2. 每段给出 doc + 段落序号 + 前 120 字预览，人从中挑能出"答案唯一"事实题的
3. --check 模式：把已写好的 questions-120.jsonl 逐条回原文核对 quote 唯一性
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from retrieval_harness import load_raw_corpus  # noqa: E402

#: 红线一：从未出过题的文档（调参集 50+20+5+5 条全部出自其余 13 份）。
#: 2026-12-01 用当前索引核实：这 13 份共 491 段，与调参集零交集。
FREE_DOCS: tuple[str, ...] = (
    "prompt-ab.md",
    "deploy-notes.md",
    "why-i-use-langchain.md",
    "prompt-templates.md",
    "system-prompt.md",
    "chunking-boundaries.md",
    "ab-scaffold.md",
    "streamlit-ui.md",
    "structured-output.md",
    "container-networking.md",
    "chunking-predictions.md",
    "release-notes-v0.3.0.md",
    "release-notes-v0.3.1.md",
)

MIN_LEN = 40  # 太短的段（目录行、分隔行）出不了事实题


def preview(docs_by_source: dict, per_doc: int = 40) -> None:
    for source in FREE_DOCS:
        segs = docs_by_source.get(source, [])
        usable = [d for d in segs if len(d.text) >= MIN_LEN and not d.text.startswith("|-")]
        print(f"=== {source}（{len(segs)} 段，可用 {len(usable)}）===")
        for i, d in enumerate(usable[:per_doc]):
            text = d.text.replace("\n", " ")[:120]
            print(f"  [{i:>2}] {text}")
        print()


def check(path: Path) -> int:
    """逐条核对 quote 是否逐字出现在对应文档的 loader 清洗后文本里。"""
    docs = load_raw_corpus()
    full_text: dict[str, str] = {}
    for d in docs:
        full_text.setdefault(d.source, "")
        full_text[d.source] += d.text + "\n"

    bad = 0
    items = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    for it in items:
        for a in it["anchors"]:
            source, quote = a["source"], a["quote"]
            n = full_text.get(source, "").count(quote)
            if n == 0:
                print(f"✗ [{it['id']}] quote 在 {source} 中找不到（可能不是逐字原文）")
                bad += 1
            elif n > 1:
                print(f"✗ [{it['id']}] quote 在 {source} 中出现 {n} 次（必须唯一）")
                bad += 1
    print(f"\n{len(items)} 条，{bad} 个 quote 问题")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=Path, help="核对已有 jsonl 的 quote 唯一性")
    args = parser.parse_args()
    if args.check:
        return check(args.check)

    docs = load_raw_corpus()
    by_source: dict[str, list] = {}
    for d in docs:
        by_source.setdefault(d.source, []).append(d)
    counts = Counter(d.source for d in docs)
    missing = [s for s in FREE_DOCS if s not in counts]
    if missing:
        print(f"⚠ 以下 FREE 文档在当前索引中不存在：{missing}")
    preview(by_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
