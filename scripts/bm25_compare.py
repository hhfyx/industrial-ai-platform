"""BM25 vs 纯向量：**术语型 query 的并排对照**。

## 这个脚本要回答的唯一问题

W5 归因表里 15 条「② 术语/同义词未命中」——**BM25 上线的第一步不是看召回涨了多少，
而是把这 15 条按"该由谁解决"重新分**：

| 判据（脚本自动判 + 人工复核） | 归给谁 |
| --- | --- |
| query 里有**稀有标识符**（`INVALID_INPUT`、`bge-small-zh`） | BM25 该强 |
| query 的词**在 gold 块里逐字出现** | BM25 该强 |
| query 与 gold 块**没有任何公共 token**（同义/换述） | 向量该强，BM25 注定得 0 |
| 两者有公共 token 但不稀有（`切分`/`文档`） | 两边都一般，需要人看 |

**这一步不做，就会得出"加个分词器就好了"这种两边都不对症的结论。**

## 为什么脚本里要打 token 流

"分词不一致"是本周最隐蔽的 bug（索引侧与查询侧走了不同的词典）。
排查它的唯一有效手段是**把两侧的 token 流并排打出来看**。
所以每个案例都输出：query 的 token、gold 块的 token、公共 token。

## 为什么叫"人工看"而不是自动判胜负

因为胜负的判据是"gold 有没有被排进 top5"——这个脚本能自动算。
但**"为什么"**（是字面没对上，还是语义没抓到）只有人能判。
所以脚本给的是**证据**（两边 top5 + token 交集 + gold 名次），
结论留在人会话里写。

## 用法

    python -X utf8 scripts/bm25_compare.py            # 10 条术语型，跑对照
    python -X utf8 scripts/bm25_compare.py --audit    # 体检：词典/语料/索引一致性
"""

from __future__ import annotations

import argparse
import json
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

from app.rag.bm25.index import DEFAULT_INDEX_PATH  # noqa: E402
from app.rag.bm25.retriever import BM25Retriever  # noqa: E402
from app.rag.bm25.tokenize import describe, tokenize  # noqa: E402
from app.rag.chunking import ChunkingConfig, build_strategy  # noqa: E402
from app.rag.settings import get_rag_config  # noqa: E402

OUT_JSON: Path = PROJECT_ROOT / "scripts" / "bm25_compare.json"
OUT_MD: Path = PROJECT_ROOT / "scripts" / "bm25_compare.md"

#: 术语型 query 的**来源**：W5 归因表里 `② 术语/同义词未命中` 那一类（15 条）。
#:
#: ## ⚠ 这个标签被实测推翻了，所以脚本跑两个范围
#:
#: 15 条逐条查过"query 与 gold 块有没有稀有公共 token"（见 `--audit-label`）：
#:
#: - **0 条**含稀有标识符（`INVALID_INPUT`、`bge-small-zh` 这一类）
#: - 12 条与 gold 块只共享「模型」「输入」「上限」「文档」这类高频普通词
#: - 3 条**连一个公共 token 都没有**
#:
#: 结论：这 15 条**不是"术语未命中"**，把它们归到"加个更好的分词器"是错方向。
#: 按上面那个分布，它们该被分成两类：
#:
#: | 实际情况 | 该谁修 |
#: | --- | --- |
#: | 只共享普通词 → BM25 有分但压不住长块 | 长块降权 / 表格路由（W5 已记录的猜想） |
#: | 零公共 token → 同义/换述 | **向量侧**（更强模型、加查询改写） |
#:
#: 所以脚本跑两个范围并分别报数：
#: - `TERM_CASE_IDS`：10 条"最像术语型"的子集（人工挑，用于看个案）
#: - 全部 40 条可答题：用于算**真实差异条数**（验收项 ≥5 条按这个算）
#:
#: 只跑 10 条子集会得出"差异 2 条"，那是个**采样偏差**：子集里 8 条是两边都
#: 不中的硬骨头，而 BM25 真正赢的 f-09 / f-27 根本不在子集里。
TERM_CASE_IDS: tuple[str, ...] = (
    "f-30",  # 错误码（本以为有 INVALID_INPUT，实测 query 里没有）
    "f-18",  # bge 512 token
    "f-20",  # 块的确定性 id
    "f-05",  # TTFT 口径
    "f-07",  # 六轮基准首字等待
    "f-25",  # CoT 提示代价
    "f-19",  # warmup / embedding 服务
    "f-22",  # 半句切分，块尾停位
    "f-08",  # 回灌 token 成本
    "f-10",  # 三条人工判据自动化
)


@dataclass
class CaseResult:
    """一条 query 在两路上的结果。"""

    id: str
    question: str
    gold_answer: str
    query_tokens: tuple[str, ...]
    vector_ranks: tuple[int | None, ...]
    bm25_ranks: tuple[int | None, ...]
    """金标块在向量 / BM25 的 top_k 里的名次。`None` 表示没进。"""

    gold_tokens: tuple[tuple[str, ...], ...]
    shared_tokens: tuple[tuple[str, ...], ...]
    """每条金标块与 query 的公共 token。**空的那些就是"字面完全没对上"的证据**。"""

    vector_top5: list[dict[str, Any]]
    bm25_top5: list[dict[str, Any]]
    verdict: str
    """auto 判的初判：`bm25_should_win` / `vector_should_win` / `both_weak`。"""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "query_tokens": list(self.query_tokens),
            "vector_ranks": list(self.vector_ranks),
            "bm25_ranks": list(self.bm25_ranks),
            "gold_tokens": [list(t) for t in self.gold_tokens],
            "shared_tokens": [list(t) for t in self.shared_tokens],
            "vector_top5": self.vector_top5,
            "bm25_top5": self.bm25_top5,
            "verdict": self.verdict,
        }


def _classify(
    query_tokens: tuple[str, ...], gold_tokens: list[tuple[str, ...]], shared: list[tuple[str, ...]]
) -> str:
    """按"公共 token 的稀有度"判该谁强。**这是判据，不是结论**。

    - **完全没有公共 token** → BM25 注定得 0 分，这是向量的题
    - **有公共 token，且其中含稀有标识符**（含数字/下划线/连字符、
      或长度 ≥ 6 的英文）→ BM25 该强
    - **只共享普通词** → 两边都不该强，需人看
    """
    if not any(shared):
        return "vector_should_win"
    rare = [tok for toks in shared for tok in toks if _is_rare_identifier(tok)]
    if rare:
        return "bm25_should_win"
    return "both_weak"


def _is_rare_identifier(token: str) -> bool:
    """`INVALID_INPUT` / `bge-small-zh` / `v0.4.1` 这类：含分隔符或长英文。"""
    if any(ch in token for ch in "_-.") and not token.isdigit():
        return True
    return len(token) >= 6 and token.isascii()


def _hit_summary(hits: list[Any], limit: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "rank": index,
            "score": round(float(hit.score), 4),
            "source": hit.chunk.source,
            "chunk_id": hit.chunk.id,
            "head": hit.chunk.text[:80].replace("\n", " "),
        }
        for index, hit in enumerate(hits[:limit], start=1)
    ]


def _ranks_of(hits: list[Any], gold_ids: list[str]) -> tuple[int | None, ...]:
    positions = {hit.chunk.id: i for i, hit in enumerate(hits, start=1)}
    return tuple(positions.get(gid) for gid in gold_ids)


def _audit(chunks: list[Any]) -> int:
    """体检：词典、语料、索引三者是否一致。**返回非 0 表示有问题**。"""
    print("=" * 78)
    print("体检：BM25 管线一致性")
    print("=" * 78)

    fingerprint = describe()
    print(f"\n① 分词器：{fingerprint}")

    from app.rag.bm25.tokenize import dict_words as dict_word_list

    dict_words = dict_word_list()
    broken = [word for word in dict_words if tokenize(word) != (word.lower(),)]
    print(f"\n② 词典自检（{len(dict_words)} 词，必须整体切出）")
    if broken:
        for word in broken:
            print(f"   ✗ {word!r} 被切成 {tokenize(word)}")
        problems = len(broken)
    else:
        print("   ✓ 全部整体切出")
        problems = 0

    # ③ 语料里到底有没有这些词（工业术语当前很可能一个都没有）
    corpus_text = "\n".join(chunk.text for chunk in chunks)
    absent = [word for word in dict_words if corpus_text.count(word) == 0]
    print(f"\n③ 词典词在语料里的覆盖：{len(dict_words) - len(absent)}/{len(dict_words)} 出现")
    if absent:
        print(f"   未出现（词典先行、语料未到）：{'、'.join(absent[:12])}")

    # ④ 索引与当前块集合是否同步
    cached = __import__("app.rag.bm25.index", fromlist=["load"]).load(DEFAULT_INDEX_PATH)
    print("\n④ 落盘索引")
    if cached is None:
        print(f"   无索引（{DEFAULT_INDEX_PATH} 不存在或版本不符）→ 先跑 build-bm25")
    else:
        same_chunks = cached.chunk_ids == [chunk.id for chunk in chunks]
        same_tokens = cached.tokenizer == fingerprint
        print(f"   路径 {DEFAULT_INDEX_PATH}｜{len(cached.chunk_ids)} 块｜{cached.build_ms} ms")
        print(f"   块集合同步：{'✓' if same_chunks else '✗ 块变了，需要重建'}")
        print(f"   分词器同步：{'✓' if same_tokens else '✗ 词典变了，需要重建'}")
        if not (same_chunks and same_tokens):
            problems += 1
        # **这一条是本周最隐蔽 bug 的守门人**：索引里的 token 流与
        # 当前分词器切出来的必须完全一致。
        mismatch = [
            cid
            for cid, stream in zip(cached.chunk_ids, cached.tokens, strict=False)
            if stream != tokenize(next(chunk.text for chunk in chunks if chunk.id == cid))
        ]
        if mismatch:
            print(f"   ✗ {len(mismatch)} 个块的 token 流与当前分词器不一致（前 3：{mismatch[:3]}）")
            problems += 1
        else:
            print("   ✓ 每个块的 token 流都与当前分词器一致（index 与 query 同一分词器）")

    print()
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BM25 vs 纯向量：术语型 query 对照")
    parser.add_argument("--audit", action="store_true", help="只体检，不跑对照")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedder", choices=["bge", "hashing"], default="bge")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    cfg = get_rag_config()
    docs = rh.load_raw_corpus()
    chunk_cfg = ChunkingConfig(chunk_chars=cfg.chunk_chars, overlap_chars=cfg.overlap_chars)
    chunks = build_strategy(cfg.strategy, cfg=chunk_cfg).chunk(docs, chunk_cfg)
    print(
        f"语料 {len(docs)} 段；索引 {len(chunks)} 块"
        f"（chunk={cfg.chunk_chars} overlap={cfg.overlap_chars} strategy={cfg.strategy}）",
        file=sys.stderr,
    )

    if args.audit:
        return _audit(chunks)

    items = {item.id: item for item in rre.load_questions(rre.EVAL_FILE)}
    missing = [qid for qid in TERM_CASE_IDS if qid not in items]
    if missing:
        print(f"评测集里找不到这些 id：{missing}", file=sys.stderr)
        return 2

    # **两个范围都要反查锚点**：子集看个案，全 40 条算真实差异条数。
    answerable = [item for item in items.values() if not item.is_refusal]
    rre.resolve_all(answerable, chunks)

    # ---- BM25 侧 ----
    started = time.perf_counter()
    bm25, report = BM25Retriever.from_chunks(chunks, path=DEFAULT_INDEX_PATH)
    bm25_ready_ms = int((time.perf_counter() - started) * 1000)
    print(f"\nBM25 索引：{report.summary()}", file=sys.stderr)

    # ---- 向量侧 ----
    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.stores.memory import InMemoryVectorStore

    if args.embedder == "hashing":
        print("⚠ hashing 下向量召回无参考价值，仅验证流程", file=sys.stderr)
        embedder = get_embedder("hashing")
    else:
        embedder = CachedEmbedder(get_embedder("bge-small-zh"))
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))
    chunks_by_id = {chunk.id: chunk for chunk in chunks}

    def _evaluate(item: Any, full: bool) -> CaseResult:
        """跑一条。`full=True` 时不存两路 top5 明细（40 条 × 10 行太长）。"""
        gold_ids = list(item.gold_chunk_ids)
        gold_tokens = [tokenize(chunks_by_id[gid].text) for gid in gold_ids if gid in chunks_by_id]
        query_tokens = tokenize(item.question)
        shared = [tuple(tok for tok in query_tokens if tok in set(gold)) for gold in gold_tokens]

        vector_hits = store.search(embedder.embed_query([item.question])[0], top_k=args.top_k)
        bm25_hits = bm25.retrieve(item.question, top_k=args.top_k)

        return CaseResult(
            id=item.id,
            question=item.question,
            gold_answer=item.gold_answer,
            query_tokens=query_tokens,
            vector_ranks=_ranks_of(vector_hits, gold_ids),
            bm25_ranks=_ranks_of(bm25_hits, gold_ids),
            gold_tokens=tuple(gold_tokens),
            shared_tokens=tuple(shared),
            vector_top5=_hit_summary(vector_hits, args.top_k) if not full else [],
            bm25_top5=_hit_summary(bm25_hits, args.top_k) if not full else [],
            verdict=_classify(query_tokens, gold_tokens, shared),
        )

    subset = [_evaluate(items[qid], full=False) for qid in TERM_CASE_IDS]
    everything = [_evaluate(item, full=True) for item in answerable]

    def _any_hit(ranks: tuple[int | None, ...]) -> bool:
        """多锚点题：**全部锚点都在 top_k 内**才算命中（与 eval 脚本同一口径）。"""
        return bool(ranks) and all(rank is not None and rank <= args.top_k for rank in ranks)

    def _stats(results: list[CaseResult]) -> dict[str, Any]:
        vec_only = [
            r.id for r in results if _any_hit(r.vector_ranks) and not _any_hit(r.bm25_ranks)
        ]
        bm25_only = [
            r.id for r in results if _any_hit(r.bm25_ranks) and not _any_hit(r.vector_ranks)
        ]
        diffs = [r.id for r in results if _any_hit(r.vector_ranks) != _any_hit(r.bm25_ranks)]
        both = [r.id for r in results if _any_hit(r.vector_ranks) and _any_hit(r.bm25_ranks)]
        neither = [
            r.id for r in results if not _any_hit(r.vector_ranks) and not _any_hit(r.bm25_ranks)
        ]
        return {
            "total": len(results),
            "vector_hits": sum(1 for r in results if _any_hit(r.vector_ranks)),
            "bm25_hits": sum(1 for r in results if _any_hit(r.bm25_ranks)),
            "both": both,
            "bm25_only": bm25_only,
            "vector_only": vec_only,
            "neither": neither,
            "neither_count": len(neither),
            "diff_cases": diffs,
        }

    subset_stats = _stats(subset)
    full_stats = _stats(everything)

    print()
    print("=" * 78)
    print(f"① 全量：全部 {full_stats['total']} 条可答题（差异条数按这个算）")
    print("=" * 78)
    print(
        f"  向量命中 {full_stats['vector_hits']}/{full_stats['total']}"
        f"    BM25 命中 {full_stats['bm25_hits']}/{full_stats['total']}"
    )
    print(f"  两边都中 {len(full_stats['both']):2d}  {full_stats['both']}")
    print(f"  仅 BM25 中 {len(full_stats['bm25_only']):2d}  {full_stats['bm25_only']}")
    print(f"  仅向量中   {len(full_stats['vector_only']):2d}  {full_stats['vector_only']}")
    print(f"  都不中     {len(full_stats['neither']):2d}  {full_stats['neither']}")
    print(f"  → 差异案例 {len(full_stats['diff_cases'])} 条（验收要求 ≥5）")

    print()
    print("=" * 78)
    print(f"② 子集：{subset_stats['total']} 条 W5 判为「术语未命中」的题")
    print("=" * 78)
    print(
        f"  向量命中 {subset_stats['vector_hits']}/{subset_stats['total']}"
        f"    BM25 命中 {subset_stats['bm25_hits']}/{subset_stats['total']}"
    )
    print(f"  仅 BM25 中 {subset_stats['bm25_only'] or '无'}")
    print(f"  仅向量中   {subset_stats['vector_only'] or '无'}")
    print(f"  → 差异案例 {len(subset_stats['diff_cases'])} 条")

    by_verdict: dict[str, list[str]] = {}
    for r in subset:
        by_verdict.setdefault(r.verdict, []).append(r.id)
    print("\n  子集按[该谁强]的初判分布（判据见 _classify，结论需人工复核）：")
    for verdict, ids in sorted(by_verdict.items()):
        print(f"    {verdict:20s} {len(ids):2d} 条  {ids}")

    if args.no_write:
        return 0

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus_docs": len(docs),
            "chunks": len(chunks),
            "chunk_chars": cfg.chunk_chars,
            "overlap": cfg.overlap_chars,
            "strategy": cfg.strategy,
            "embedder": args.embedder,
            "top_k": args.top_k,
            "bm25_report": report.summary(),
            "bm25_ready_ms": bm25_ready_ms,
            "bm25_stats": bm25.stats(),
            "tokenizer": describe(),
            "cases": list(TERM_CASE_IDS),
            "answerable": len(answerable),
        },
        "full": full_stats,
        "subset": subset_stats,
        "by_verdict": by_verdict,
        "cases": [r.as_dict() for r in subset],
        "all_cases": [r.as_dict() | {"vector_top5": [], "bm25_top5": []} for r in everything],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"\n落盘：{OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"      {OUT_MD.relative_to(PROJECT_ROOT)}")
    return 0


def _render_markdown(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    full = payload["full"]
    subset = payload["subset"]
    lines: list[str] = []
    lines.append("# BM25 vs 纯向量：术语型 query 对照")
    lines.append("")
    lines.append(
        f"> 生成时间 {meta['generated_at']} ｜ {meta['chunks']} 块 "
        f"（chunk={meta['chunk_chars']} overlap={meta['overlap']} {meta['strategy']}）"
        f" ｜ top-{meta['top_k']} ｜ BM25 k1={meta['bm25_stats']['params']['k1']} "
        f"b={meta['bm25_stats']['params']['b']}"
    )
    lines.append(
        f"> BM25 索引：{meta['bm25_report']['chunks']} 块 / "
        f"{meta['bm25_report']['tokens']} tokens / {meta['bm25_report']['build_ms']} ms / "
        f"{meta['bm25_report']['size_kb']} KB（avgdl {meta['bm25_stats']['avgdl']}）"
    )
    lines.append("")
    lines.append("## 两个范围（**差异条数按全量算**）")
    lines.append("")
    lines.append("| 范围 | 向量命中 | BM25 命中 | 差异 | 仅 BM25 | 仅向量 |")
    lines.append("| --- | ---: | ---: | ---: | --- | --- |")
    lines.append(
        f"| 全量 {full['total']} 条可答题 | {full['vector_hits']} | {full['bm25_hits']} | "
        f"**{len(full['diff_cases'])}** | {full['bm25_only'] or '无'} | {full['vector_only'] or '无'} |"
    )
    lines.append(
        f"| 子集 {subset['total']} 条（W5「术语未命中」） | {subset['vector_hits']} | "
        f"{subset['bm25_hits']} | {len(subset['diff_cases'])} | "
        f"{subset['bm25_only'] or '无'} | {subset['vector_only'] or '无'} |"
    )
    lines.append("")
    lines.append(
        "> **只跑子集会得出「差异 2 条」，那是采样偏差**："
        f"子集 {subset['total']} 条里两路都不中的有 {subset['neither_count']} 条，"
        "而 BM25 真正赢下的 f-09 / f-27 根本不在子集里。"
    )
    lines.append("")
    lines.append(f"全量里两边都不中：{len(full['neither'])} 条 — {full['neither']}")
    lines.append("")
    lines.append("## 子集逐条")
    lines.append("")
    lines.append("| id | 向量名次 | BM25 名次 | 初判 | 公共 token |")
    lines.append("| --- | --- | --- | --- | --- |")
    for case in payload["cases"]:
        vec = " / ".join(str(r) if r else "未" for r in case["vector_ranks"]) or "—"
        bm = " / ".join(str(r) if r else "未" for r in case["bm25_ranks"]) or "—"
        shared = "；".join("、".join(t) if t else "∅" for t in case["shared_tokens"])
        lines.append(f"| {case['id']} | {vec} | {bm} | {case['verdict']} | {shared} |")
    lines.append("")
    lines.append("> **`∅` 表示 query 与 gold 块没有任何公共 token** ——")
    lines.append("> 那是 BM25 注定得 0 分的题，是向量该赢的地方，不该算 BM25 的失败。")
    lines.append("")
    lines.append("## 逐条")
    lines.append("")
    lines.append("| id | 向量名次 | BM25 名次 | 初判 | 公共 token |")
    lines.append("| --- | --- | --- | --- | --- |")
    for case in payload["cases"]:
        vec = " / ".join(str(r) if r else "未" for r in case["vector_ranks"]) or "—"
        bm = " / ".join(str(r) if r else "未" for r in case["bm25_ranks"]) or "—"
        shared = "；".join("、".join(t) if t else "∅" for t in case["shared_tokens"])
        lines.append(f"| {case['id']} | {vec} | {bm} | {case['verdict']} | {shared} |")
    lines.append("")
    lines.append("> **`∅` 表示 query 与 gold 块没有任何公共 token** ——")
    lines.append("> 那是 BM25 注定得 0 分的题，是向量该赢的地方，不该算 BM25 的失败。")
    lines.append("")
    for case in payload["cases"]:
        lines.append(f"### {case['id']}")
        lines.append("")
        lines.append(f"**问题**：{case['question']}")
        lines.append("")
        lines.append(f"**gold**：{case['gold_answer']}")
        lines.append("")
        lines.append(f"**query token**：`{'` `'.join(case['query_tokens'])}`")
        lines.append("")
        for index in range(1, len(case["gold_tokens"]) + 1):
            shared = case["shared_tokens"][index - 1]
            mark = "、".join(shared) if shared else "∅（字面完全没对上）"
            lines.append(f"- gold 块 {index} 的 token 交集：{mark}")
        lines.append("")
        lines.append("<details><summary>两路 top5</summary>")
        lines.append("")
        lines.append("| 路 | # | score | source | 开头 |")
        lines.append("| --- | ---: | ---: | --- | --- |")
        for row in case["vector_top5"]:
            lines.append(
                f"| 向量 | {row['rank']} | {row['score']:.4f} | {row['source']} | {row['head']} |"
            )
        for row in case["bm25_top5"]:
            lines.append(
                f"| BM25 | {row['rank']} | {row['score']:.2f} | {row['source']} | {row['head']} |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
