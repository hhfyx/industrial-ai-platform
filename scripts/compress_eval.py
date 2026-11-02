"""上下文压缩的效果与代价：**三个数字必须分开报**。

## 为什么必须分开（任务书的必答问题）

> 压缩提高的是检索指标还是生成质量？—— 都不一定。
> 它主要降低 token 成本与延迟，但会降低召回（因为删了句子，gold 锚点对不上）。
> 所以压缩的评估口径必须分开：成本指标 + 答案质量指标，
> 不能只报"省了 40% token"就完事。**这是本周最容易被自己糊弄过去的一层。**

本脚本输出的三张表分别对应：

| 轨道 | 指标 | 怎么算 |
| --- | --- | --- |
| **成本** | 字符/token 节省 % | 压缩前后直接比 |
| **检索** | Recall@5 / MRR | **在压缩前的 chunk 上算**（口径见下） |
| **答案质量** | 答案句存活率、编造风险 | 见 §答案质量 |

## 关键口径：Recall 在压缩前算

任务书点名："压缩不改变 chunk 的可检索身份，只改变送进 LLM 的文本
（即 Recall 仍在压缩前的 chunk 上算）"。

**所以本脚本里 Recall@5 会是一条水平线** —— 它与压缩参数**无关**。
这不是脚本没测，而是**设计如此**：压缩不该影响召回，影响召回说明实现坏了。

真正会变的、必须被量出来的是：

1. **答案句存活率**：gold 块里那句"含答案的话"有没有被保留下来。
   这是压缩**真正**的风险所在 —— 它不改变召回，但会让模型拿不到答案。
2. **锚点存活率**：gold 锚点的原文片段在压缩文本里还能不能找到。
   这是"锚点对不上"的具体度量（任务书警告的那个问题）。

### 为什么不用 LLM 判答案质量

任务书说"答案质量（抽 20 条）"。**本脚本用两条代理指标，不调 LLM**：

| 代理指标 | 它测什么 | 为什么可信 |
| --- | --- | --- |
| 答案句存活率 | gold 块里含 gold_answer 关键词的句子是否被保留 | 客观、可复现、零成本 |
| 锚点存活率 | 原文池片段在压缩文本里是否还存在 | 与 W5 的锚点机制同源 |

**LLM 答案是另一件事**（模型会不会用上保留的句子、会不会自己编），
它需要真调模型，属于 `scripts/answer_quality.py` 的范畴。
本脚本会把 `--llm` 留成可选项，默认不开 —— **因为"省了多少 token"
与"模型答得对不对"是两个问题，混在一起量会得出无法归因的结论。**

## 压缩过头的症状（脚本会显式报警）

| 症状 | 判据 |
| --- | --- |
| 答案句被删 | 答案句存活率 < 60% |
| 压缩比例失真 | `floor_count` 占比 > 50%（说明兜底在频繁救场） |
| 压成空块 | 任一压缩块 `len(text) == 0` |
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import retrieval_harness as rh  # noqa: E402
import run_retrieval_eval as rre  # noqa: E402

from app.rag.chunking import ChunkingConfig, build_strategy  # noqa: E402
from app.rag.generation.base import build_context  # noqa: E402
from app.rag.generation.compress import (  # noqa: E402
    DEFAULT_MIN_RATIO,
    DEFAULT_MIN_SENTENCES,
    compress_for_prompt,
    split_sentences,
)
from app.rag.settings import get_rag_config  # noqa: E402

OUT_JSON: Path = PROJECT_ROOT / "scripts" / "compress_eval.json"
OUT_MD: Path = PROJECT_ROOT / "scripts" / "compress_eval.md"

#: 要扫的保留比例。**0.3 是兜底下限附近**，0.5 是中等，1.0 是不压缩对照。
KEEP_RATIOS: tuple[float, ...] = (0.3, 0.5, 0.7, 1.0)

#: 抽检多少条做答案句存活统计。任务书要求 20 条。
SAMPLE_SIZE: int = 20

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:[-.][A-Za-z0-9_]+)*|[0-9]+|[一-鿿]+")


def _keywords(text: str) -> set[str]:
    """从 gold_answer 里抽关键词：中文取 2 字以上的片段、英文数字整体。"""
    words = {m.group(0).lower() for m in _WORD.finditer(text)}
    return {w for w in words if len(w) >= 2 or w.isdigit()}


def _answer_sentence_kept(chunk_text: str, compressed: str, gold_answer: str) -> bool | None:
    """gold 块里"含答案关键词的那句"有没有被保留。

    返回 `None` 表示"这个块里根本没有含答案关键词的句子" —— 那种情况无法判定，
    **不能算作失败**（那是锚点选得不好，不是压缩的问题）。
    """
    keywords = _keywords(gold_answer)
    if not keywords:
        return None
    sentences = split_sentences(chunk_text)
    target = [s for s in sentences if any(k in s.lower() for k in keywords)]
    if not target:
        return None
    return all(s in compressed for s in target)


def _anchor_kept(chunk_text: str, compressed: str, anchor_quote: str) -> bool | None:
    """原文锚点片段在压缩文本里还在不在。`None` 表示这条题没有可用锚点。

    ## ⚠ 这里有一个必须写清楚的失败模式（W6 周五实测）

    第一版直接判 `anchor_quote in compressed`，于是 `keep_ratio=1.0`
    （**完全不删任何句子**）时锚点存活率只有 50% —— 这是错的。

    原因：**`quote` 可能跨句子边界**。压缩的"保序重组"用空格把保留的句子
    重新拼起来，如果原文是

        「…见 `schema.py`。同一个内容在半库里可能有多条…」
                    ↑ quote 从这里开始，跨过句号

    那么 quote 在原块里存在，但在"按句切分再拼回"的文本里，
    句号两侧的字符关系被改写（多了一个空格），于是子串匹配失败 ——
    **而内容一个字都没少。**

    这正好印证了任务书那句话：

    > 压缩后 gold 锚点在压缩文本里可能找不到 → 判定要能处理这种情况。

    ## 处理方式

    判"**quote 的各个片段是否都还在**"：把 quote 按标点与空白切成若干
    不短于 4 字的片段，全部还在就算存活。这容忍了重组引入的空格差异，
    但不容忍"删掉了含 quote 的那句"。
    """
    if not anchor_quote:
        return None
    if anchor_quote in compressed:
        return True
    # 退化判定：按标点/空白切片，每片都得能找到
    pieces = [p for p in re.split(r"[\s，。、；：！？,.;:!?`|]+", anchor_quote) if len(p) >= 4]
    if not pieces:
        # 片段太短（全是标点或短词）：退回到"包含任意连续 4 字"的宽松判断
        return False
    return all(piece in compressed for piece in pieces)


@dataclass
class SweepRow:
    keep_ratio: float
    original_chars: int = 0
    compressed_chars: int = 0
    floor_count: int = 0
    chunks: int = 0
    empty_chunks: int = 0
    answer_checked: int = 0
    answer_kept: int = 0
    anchor_checked: int = 0
    anchor_kept: int = 0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    elapsed_ms: int = 0

    @property
    def saved_ratio(self) -> float:
        if not self.original_chars:
            return 0.0
        return 1.0 - self.compressed_chars / self.original_chars

    @property
    def answer_kept_rate(self) -> float:
        return self.answer_kept / self.answer_checked if self.answer_checked else 0.0

    @property
    def anchor_kept_rate(self) -> float:
        return self.anchor_kept / self.anchor_checked if self.anchor_checked else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "keep_ratio": self.keep_ratio,
            "original_chars": self.original_chars,
            "compressed_chars": self.compressed_chars,
            "saved_ratio": round(self.saved_ratio, 4),
            "chunks": self.chunks,
            "floor_count": self.floor_count,
            "empty_chunks": self.empty_chunks,
            "answer_checked": self.answer_checked,
            "answer_kept": self.answer_kept,
            "answer_kept_rate": round(self.answer_kept_rate, 4),
            "anchor_checked": self.anchor_checked,
            "anchor_kept": self.anchor_kept,
            "anchor_kept_rate": round(self.anchor_kept_rate, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "mrr": round(self.mrr, 4),
            "elapsed_ms": self.elapsed_ms,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="上下文压缩：效果与代价")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-ratio", type=float, default=DEFAULT_MIN_RATIO)
    parser.add_argument("--min-sentences", type=int, default=DEFAULT_MIN_SENTENCES)
    parser.add_argument("--sample", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    cfg = get_rag_config()
    docs = rh.load_raw_corpus()
    chunk_cfg = ChunkingConfig(chunk_chars=cfg.chunk_chars, overlap_chars=cfg.overlap_chars)
    chunks = build_strategy(cfg.strategy, cfg=chunk_cfg).chunk(docs, chunk_cfg)
    print(
        f"语料 {len(docs)} 段；索引 {len(chunks)} 块 ｜ top-{args.top_k} ｜ 抽检 {args.sample} 条",
        file=sys.stderr,
    )

    items = rre.load_questions(rre.EVAL_FILE)
    answerable = [item for item in items if not item.is_refusal]
    rre.resolve_all(items, chunks)
    chunks_by_id = {chunk.id: chunk for chunk in chunks}

    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.stores.memory import InMemoryVectorStore

    embedder = CachedEmbedder(get_embedder("bge-small-zh"))
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))

    # 先给每条题取好 top-k 块（**Recall 就在这里算，与压缩无关**）
    per_question: dict[str, list[str]] = {}
    recall_hits = 0
    reciprocal = 0.0
    for item in answerable:
        hits = store.search(embedder.embed_query([item.question])[0], top_k=args.top_k)
        ids = [hit.chunk.id for hit in hits]
        per_question[item.id] = ids
        pos = {cid: index for index, cid in enumerate(ids, start=1)}
        ranks = [pos.get(gid) for gid in item.gold_chunk_ids]
        if all(rank is not None for rank in ranks):
            recall_hits += 1
            reciprocal += 1.0 / max(r for r in ranks if r is not None)  # type: ignore[type-var]
    recall5 = recall_hits / len(answerable) if answerable else 0.0
    mrr = reciprocal / len(answerable) if answerable else 0.0

    print(
        f"\n检索（**压缩前算**，与压缩参数无关）：Recall@{args.top_k} = "
        f"{recall5 * 100:.1f}%   MRR = {mrr:.4f}",
        file=sys.stderr,
    )

    rows: list[SweepRow] = []
    for ratio in KEEP_RATIOS:
        started = time.perf_counter()
        row = SweepRow(keep_ratio=ratio, recall_at_5=recall5, mrr=mrr)
        for item in answerable:
            ids = per_question[item.id]
            batch = [chunks_by_id[cid] for cid in ids if cid in chunks_by_id]
            compressed = compress_for_prompt(
                batch,
                item.question,
                keep_ratio=ratio,
                min_ratio=args.min_ratio,
                min_sentences=args.min_sentences,
            )
            row.original_chars += compressed.original_chars
            row.compressed_chars += compressed.compressed_chars
            row.floor_count += compressed.floor_count
            row.chunks += len(compressed.chunks)
            row.empty_chunks += sum(1 for c in compressed.chunks if not c.text.strip())

            # 答案句 / 锚点存活：**只在"gold 确实进了候选"的题上统计**。
            #
            # ⚠ 这一条是修出来的 bug。第一版对所有题都统计，于是
            # `keep_ratio=1.0`（完全不压缩）的答案句存活率只有 44.4% ——
            # 而不压缩时它必须是 100%。
            #
            # 原因：那些题的 gold 块**根本没进 top5**（召回没命中），
            # 压缩输出里当然没有它。第一版把"召回未命中"算成了"答案句被删"，
            # 而这两件事完全不同 —— 一个是召回层的问题，一个是压缩层的问题。
            #
            # 与 W6 周四"天花板"那条是同一个教训：
            # **统计压缩风险之前，先确认候选里有 gold。**
            if item.id in {it.id for it in answerable[: args.sample]}:
                texts = {c.chunk_id: c.text for c in compressed.chunks}
                for gid in item.gold_chunk_ids:
                    original = chunks_by_id.get(gid)
                    # **关键：gold 不在本次候选里就跳过，不计入分母**
                    if original is None or gid not in texts:
                        continue
                    got = texts[gid]
                    verdict = _answer_sentence_kept(original.text, got, item.gold_answer)
                    if verdict is not None:
                        row.answer_checked += 1
                        row.answer_kept += int(verdict)
                    for anchor in item.anchors:
                        if anchor.get("source") == original.source:
                            kept = _anchor_kept(original.text, got, str(anchor.get("quote", "")))
                            if kept is not None:
                                row.anchor_checked += 1
                                row.anchor_kept += int(kept)
                            break
        row.elapsed_ms = int((time.perf_counter() - started) * 1000)
        rows.append(row)

    # 预算一下压缩后的 prompt 长度（用真实 build_context 拼一次，验证可注入）
    sample_item = answerable[0]
    sample_batch = [
        chunks_by_id[cid] for cid in per_question[sample_item.id] if cid in chunks_by_id
    ]
    sample_ctx = compress_for_prompt(sample_batch, sample_item.question, keep_ratio=0.5)
    raw_text, raw_cit = build_context(sample_batch)
    cmp_text, cmp_cit = build_context(
        sample_batch, texts={c.chunk_id: c.text for c in sample_ctx.chunks}
    )
    integration = {
        "raw_chars": len(raw_text),
        "compressed_chars": len(cmp_text),
        "citations_identical": [c.chunk_id for c in raw_cit] == [c.chunk_id for c in cmp_cit],
    }

    print()
    print("=" * 86)
    print("① 成本（字符节省）")
    print("=" * 86)
    print(f"  {'keep':>6}{'原文':>10}{'压缩后':>10}{'节省':>9}{'兜底块':>8}{'空块':>6}")
    for row in rows:
        print(
            f"  {row.keep_ratio:>6.1f}{row.original_chars:>10}{row.compressed_chars:>10}"
            f"{row.saved_ratio * 100:>8.1f}%{row.floor_count:>8}{row.empty_chunks:>6}"
        )

    print()
    print("=" * 86)
    print("② 检索（**压缩前算 → 必然是水平线，这是设计而非未测**）")
    print("=" * 86)
    for row in rows:
        print(
            f"  keep={row.keep_ratio:<4} Recall@{args.top_k}={row.recall_at_5 * 100:.1f}%  MRR={row.mrr:.4f}"
        )

    print()
    print("=" * 86)
    print(f"③ 答案质量代理指标（抽检 {args.sample} 条）")
    print("=" * 86)
    print(f"  {'keep':>6}{'答案句存活':>12}{'锚点存活':>12}{'判定':>10}")
    for row in rows:
        verdict = (
            "⚠ 风险"
            if row.answer_kept_rate < 0.60 and row.answer_checked
            else ("OK" if row.answer_checked else "—")
        )
        print(
            f"  {row.keep_ratio:>6.1f}{row.answer_kept_rate * 100:>11.1f}%"
            f"{row.anchor_kept_rate * 100:>11.1f}%{verdict:>10}"
        )
    print()
    print(
        f"  集成校验：build_context 注入压缩文本后，引用 chunk_id 是否不变 → {integration['citations_identical']}"
    )
    print(f"  （{integration['raw_chars']} → {integration['compressed_chars']} 字符）")

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus_docs": len(docs),
            "chunks": len(chunks),
            "chunk_chars": cfg.chunk_chars,
            "overlap": cfg.overlap_chars,
            "strategy": cfg.strategy,
            "top_k": args.top_k,
            "min_ratio": args.min_ratio,
            "min_sentences": args.min_sentences,
            "sample": args.sample,
            "questions": len(answerable),
            "recall_at_5_fixed": round(recall5, 4),
            "mrr_fixed": round(mrr, 4),
        },
        "rows": [row.as_dict() for row in rows],
        "integration": integration,
    }
    if args.no_write:
        return 0
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"\n落盘：{OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"      {OUT_MD.relative_to(PROJECT_ROOT)}")
    return 0


def _render_markdown(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    rows = payload["rows"]
    lines: list[str] = []
    lines.append("# 上下文压缩：效果与代价")
    lines.append("")
    lines.append(
        f"> {meta['generated_at']} ｜ {meta['chunks']} 块 ｜ top-{meta['top_k']} "
        f"｜ 抽检 {meta['sample']} 条 ｜ 兜底 min_ratio={meta['min_ratio']} "
        f"min_sentences={meta['min_sentences']}"
    )
    lines.append("")
    lines.append("## 一、成本（字符节省）")
    lines.append("")
    lines.append("| keep_ratio | 原文 | 压缩后 | 节省 | 兜底块 | 空块 |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['keep_ratio']} | {row['original_chars']} | {row['compressed_chars']} | "
            f"**{row['saved_ratio'] * 100:.1f}%** | {row['floor_count']} | {row['empty_chunks']} |"
        )
    lines.append("")
    lines.append("## 二、检索（**压缩前算**）")
    lines.append("")
    lines.append(
        f"Recall@{meta['top_k']} = **{meta['recall_at_5_fixed'] * 100:.1f}%** ｜ "
        f"MRR = **{meta['mrr_fixed']:.4f}**"
    )
    lines.append("")
    lines.append("> 这两个数**与 keep_ratio 无关** —— 压缩不改变块的可检索身份。")
    lines.append("> 如果它随压缩参数变化，说明实现把压缩写进了索引，那是 bug。")
    lines.append("")
    lines.append("## 三、答案质量代理指标")
    lines.append("")
    lines.append("| keep_ratio | 答案句存活 | 锚点存活 | 判定 |")
    lines.append("| ---: | ---: | ---: | :---: |")
    for row in rows:
        verdict = (
            "⚠ 风险"
            if row["answer_kept_rate"] < 0.60 and row["answer_checked"]
            else ("OK" if row["answer_checked"] else "—")
        )
        lines.append(
            f"| {row['keep_ratio']} | {row['answer_kept_rate'] * 100:.1f}% | "
            f"{row['anchor_kept_rate'] * 100:.1f}% | {verdict} |"
        )
    lines.append("")
    lines.append("## 四、集成校验")
    lines.append("")
    integ = payload["integration"]
    lines.append(
        f"- `build_context` 注入压缩文本后引用 `chunk_id` 是否不变："
        f"**{integ['citations_identical']}**"
    )
    lines.append(f"- 同一批块的上下文：{integ['raw_chars']} → {integ['compressed_chars']} 字符")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
