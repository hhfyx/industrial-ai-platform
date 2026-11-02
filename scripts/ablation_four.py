"""四配置消融（W6 主交付）：**严格累加，一次只加一件事**。

## 四组定义（任务书原文）

| 组 | 配置 | 相对上一组只加 |
| --- | --- | --- |
| ① 朴素向量 | bge 向量 → top5 | — |
| ② +混合检索 | 向量 + BM25 → RRF → top5 | RRF 融合 |
| ③ +精排 | ①②的候选 → Cross-Encoder → top5 | 精排 |
| ④ +压缩 | ③的 top5 → 句子级压缩 → 送 LLM | 压缩 |

**每组固定切分/参数/k=5。** 唯一的变量是"这一层"。

## 严格累加的两个陷阱（都在这份脚本里处理了）

### 陷阱 1：④ 组的 Recall 必须在**压缩前**算

任务书的原话：

> 组 4 同上；按"压缩不改变 chunk 身份"的口径算 Recall，额外记 token 节省

所以 ④ 组的 Recall@5 / MRR / Recall@10 **与 ③ 组完全相同** ——
这不是"压缩没用"，是**压缩按设计不影响检索**。压缩带来的变化只出现在
"上下文 token"这一列。

如果把 ④ 组的 Recall 在压缩文本上算，会出现"Recall 掉下来"的假信号，
而真正的原因是压缩删了句子 —— 那是把压缩的风险伪装成检索的退化。

### 陷阱 2：② 组的候选数必须与 ③ 组一致

② 组取 top5、③ 组取 top20 再精排到 5 —— 那样 ③ 的收益里混了"候选更多"。
所以 ② 和 ③ 都用**同一个候选窗口 N**，② 只是不做精排直接截 top5。

## 指标口径

| 指标 | 怎么算 | 谁受影响 |
| --- | --- | --- |
| Recall@5 / @10 / MRR | 检索命中 | ①②③ 会变，④ 不变 |
| Recall@50（上限） | 候选窗口内的命中 | 只对 ③ 有意义（①② 没有 50 条候选） |
| 拒答准确率 | top1 分数 vs 阈值 | ①②③ 会变（分数变了），④ 不变 |
| 索引耗时 | 建索引一次 | ② 加 BM25 索引，③ 加模型加载 |
| 查询延迟 | 单 query 端到端 | 每层都加 |
| 平均上下文 token | top5 拼起来 | ④ 唯一会变 |
| 实现复杂度 | 新增代码行数 + 配置项 | 手工填（脚本给行数） |

## 答案质量抽检（今天唯一花钱处）

**本脚本不调 LLM。** 任务书要求"20 条 × 4 组 = 80 次生成"，那需要网关可用。
本脚本产出的是**检索层全部指标 + 压缩成本**，并把答案质量留成一个
显式的 `answer_quality: "未测"` 字段 —— **不编造，不估算**。

补测入口：`python -X utf8 scripts/tasks.py answer-quality`（需网关）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import retrieval_harness as rh  # noqa: E402
import run_retrieval_eval as rre  # noqa: E402

from app.rag.bm25.fusion import DEFAULT_RRF_K, FusionConfig, FusionMode, merge_rankings  # noqa: E402
from app.rag.bm25.index import DEFAULT_INDEX_PATH  # noqa: E402
from app.rag.bm25.rerank import DEFAULT_MODEL_DIR, Reranker  # noqa: E402
from app.rag.bm25.retriever import BM25Retriever  # noqa: E402
from app.rag.chunking import ChunkingConfig, build_strategy  # noqa: E402
from app.rag.generation.compress import compress_for_prompt  # noqa: E402
from app.rag.settings import get_rag_config  # noqa: E402

OUT_JSON: Path = PROJECT_ROOT / "scripts" / "ablation_four.json"
OUT_MD: Path = PROJECT_ROOT / "scripts" / "ablation_four.md"

TOP_K: int = 5
"""四组统一 k=5（任务书要求"每组固定 k=5"）。"""

CANDIDATES: int = 20
"""候选窗口。②③ 共用它，保证"只加一件事"。

取 20 的依据见 `docs/rerank-pipeline.md` §3.2：N=20 是效果/延迟最划算的一档。
"""

COMPRESS_KEEP_RATIO: float = 0.5
"""④ 组的压缩比例。取 0.5 是因为它落在"成本-质量曲线"的中段
（省 37.5%、答案句存活 37.5%，见 `docs/context-compression.md` §3.3）。

**但它同时在"风险区"** —— 这就是为什么 ④ 组的结论会是"不进链路"。
"""

#: 索引耗时的来源。**不现场测**（那要重建索引，前置时间太长），
#: 而是引用各层自己的产物。留 `None` 表示"该组没有额外索引"。
INDEX_MS_REFERENCE: dict[str, str] = {
    "vec": "向量索引：bge embed 1592 块，见 scripts/strategy_compare.json（index_ms）",
    "bm25": "BM25 索引：见 scripts/bm25_index_report.json（build_ms）",
    "rerank": "精排模型加载：见 scripts/rerank_probe.json（load_ms）",
    "compress": "无索引（纯计算，无持久化）",
}


@dataclass
class GroupResult:
    """一组实验的完整记录。"""

    key: str
    name: str
    added: str
    """相对上一组"只加"的那件事。这是消融表的灵魂列。"""

    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    recall_at_ceiling: float | None = None
    """候选窗口内的召回（= 精排的天花板）。只有 ③④ 记它有意义。"""

    refusal_accuracy: float = 0.0
    refusal_correct: int = 0
    refusal_total: int = 0
    false_refusal: int = 0
    false_accept: int = 0

    query_latency_p50_ms: int = 0
    query_latency_p95_ms: int = 0
    avg_context_chars: float = 0.0
    avg_context_tokens: float = 0.0
    """token 估算：中文约 0.67 token/字（项目既有口径）。"""

    token_saved_ratio: float = 0.0
    """只对 ④ 有意义。"""

    per_question: dict[str, str] = field(default_factory=dict)
    """逐题命中状态，用于算"哪几条题在这一层被救回/丢掉"。"""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "added": self.added,
            "recall_at_5": round(self.recall_at_5, 4),
            "recall_at_10": round(self.recall_at_10, 4),
            "mrr": round(self.mrr, 4),
            "recall_at_ceiling": round(self.recall_at_ceiling, 4)
            if self.recall_at_ceiling is not None
            else None,
            "refusal_accuracy": round(self.refusal_accuracy, 4),
            "refusal_correct": self.refusal_correct,
            "refusal_total": self.refusal_total,
            "false_refusal": self.false_refusal,
            "false_accept": self.false_accept,
            "query_latency_p50_ms": self.query_latency_p50_ms,
            "query_latency_p95_ms": self.query_latency_p95_ms,
            "avg_context_chars": round(self.avg_context_chars, 1),
            "avg_context_tokens": round(self.avg_context_tokens, 1),
            "token_saved_ratio": round(self.token_saved_ratio, 4),
            "per_question": self.per_question,
        }


def _hit_at(ids: list[str], gold: list[str], k: int) -> bool:
    """多锚点：全部锚点在 top-k 内才算命中（与 eval 同一口径）。"""
    if not gold:
        return False
    pos = {cid: index for index, cid in enumerate(ids, start=1)}
    ranks = [pos.get(gid) for gid in gold]
    return all(rank is not None and rank <= k for rank in ranks)


def _mrr_of(ids: list[str], gold: list[str], limit: int) -> float:
    if not gold:
        return 0.0
    pos = {cid: index for index, cid in enumerate(ids, start=1)}
    ranks = [pos.get(gid) for gid in gold]
    if not all(r is not None and r <= limit for r in ranks):
        return 0.0
    return 1.0 / max(r for r in ranks if r is not None)  # type: ignore[type-var]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="四配置消融")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--candidates", type=int, default=CANDIDATES)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    cfg = get_rag_config()
    threshold = cfg.score_threshold

    # ---------- 建索引 ----------
    docs = rh.load_raw_corpus()
    chunk_cfg = ChunkingConfig(chunk_chars=cfg.chunk_chars, overlap_chars=cfg.overlap_chars)
    chunks = build_strategy(cfg.strategy, cfg=chunk_cfg).chunk(docs, chunk_cfg)
    print(
        f"语料 {len(docs)} 段；索引 {len(chunks)} 块"
        f"（chunk={cfg.chunk_chars} overlap={cfg.overlap_chars} strategy={cfg.strategy}）",
        file=sys.stderr,
    )

    items = rre.load_questions(rre.EVAL_FILE)
    answerable = [item for item in items if not item.is_refusal]
    refusals = [item for item in items if item.is_refusal]
    rre.resolve_all(items, chunks)
    chunks_by_id = {chunk.id: chunk for chunk in chunks}

    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.stores.memory import InMemoryVectorStore

    embed_started = time.perf_counter()
    embedder = CachedEmbedder(get_embedder("bge-small-zh"))
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))
    vec_index_ms = int((time.perf_counter() - embed_started) * 1000)
    print(f"向量索引 {vec_index_ms} ms", file=sys.stderr)

    bm25_started = time.perf_counter()
    bm25, bm25_report = BM25Retriever.from_chunks(chunks, path=DEFAULT_INDEX_PATH)
    bm25_index_ms = int((time.perf_counter() - bm25_started) * 1000)
    print(f"BM25 索引 {bm25_index_ms} ms（{bm25_report.summary()}）", file=sys.stderr)

    reranker = Reranker(model_path=args.model_dir)
    load_started = time.perf_counter()
    reranker.load()
    rerank_load_ms = int((time.perf_counter() - load_started) * 1000)
    print(f"精排模型加载 {rerank_load_ms} ms", file=sys.stderr)

    fusion_cfg = FusionConfig(mode=FusionMode.RRF, candidates=args.candidates, rrf_k=DEFAULT_RRF_K)

    # ---------- 逐 query 产出四层的候选 ----------
    # **一次算好，四组共用** —— 保证"只加一件事"，也避免重复 embed。
    per_query: dict[str, dict[str, Any]] = {}
    print("\n逐 query 计算四层候选…", file=sys.stderr)
    for item in items:
        started = time.perf_counter()
        vec_hits = store.search(embedder.embed_query([item.question])[0], top_k=args.candidates)
        vec_ms = int((time.perf_counter() - started) * 1000)
        vec_ids = [hit.chunk.id for hit in vec_hits]
        vec_scores = [float(hit.score) for hit in vec_hits]

        bm25_started = time.perf_counter()
        bm25_hits = bm25.retrieve(item.question, top_k=args.candidates)
        bm25_ms = int((time.perf_counter() - bm25_started) * 1000)

        fused, _ = merge_rankings(
            vec_ids,
            [hit.chunk.id for hit in bm25_hits],
            config=fusion_cfg,
        )
        hybrid_ids = [hit.chunk_id for hit in fused]

        # ③ 精排：对**混合**的候选做（这是"累加"的定义）
        rerank_started = time.perf_counter()
        candidates = [(cid, chunks_by_id[cid].text) for cid in hybrid_ids if cid in chunks_by_id]
        reranked, _report = reranker.rerank(item.question, candidates, top_k=args.candidates)
        rerank_ms = int((time.perf_counter() - rerank_started) * 1000)
        rerank_ids = [score.chunk_id for score in reranked]

        per_query[item.id] = {
            "vec_ids": vec_ids,
            "vec_scores": vec_scores,
            "hybrid_ids": hybrid_ids,
            "rerank_ids": rerank_ids,
            "vec_ms": vec_ms,
            "bm25_ms": bm25_ms,
            "rerank_ms": rerank_ms,
        }

    # ---------- 四组汇总 ----------
    def _score_group(
        key: str,
        name: str,
        added: str,
        *,
        id_pick: Any,
        latency_of: Any,
        compress: bool = False,
        with_ceiling: bool = False,
    ) -> GroupResult:
        result = GroupResult(key=key, name=name, added=added)
        latencies: list[int] = []
        ctx_chars: list[int] = []
        ctx_orig: list[int] = []
        reportable_hits = 0
        ceiling_hits = 0

        for item in answerable:
            data = per_query[item.id]
            ids = id_pick(data)
            latencies.append(latency_of(data))

            # ④ 压缩：**只改送进 prompt 的文本，不改 ids**
            if compress:
                top = [chunks_by_id[cid] for cid in ids[:TOP_K] if cid in chunks_by_id]
                ctx = compress_for_prompt(top, item.question, keep_ratio=COMPRESS_KEEP_RATIO)
                ctx_chars.append(ctx.compressed_chars)
                ctx_orig.append(ctx.original_chars)
            else:
                top = [chunks_by_id[cid] for cid in ids[:TOP_K] if cid in chunks_by_id]
                # 用与压缩侧一致的"块正文长度和"口径，而不是 build_context 的
                # 带编号/来源开销 —— 两者比的是**正文**，不是格式
                total = sum(len(chunk.text) for chunk in top)
                ctx_chars.append(total)
                ctx_orig.append(total)

            hit5 = _hit_at(ids, item.gold_chunk_ids, TOP_K)
            result.per_question[item.id] = "hit" if hit5 else "miss"
            reportable_hits += int(hit5)
            if with_ceiling and _hit_at(ids, item.gold_chunk_ids, args.candidates):
                ceiling_hits += 1

        total_q = len(answerable)
        result.recall_at_5 = reportable_hits / total_q if total_q else 0.0
        result.recall_at_10 = (
            sum(
                1
                for item in answerable
                if _hit_at(id_pick(per_query[item.id]), item.gold_chunk_ids, 10)
            )
            / total_q
            if total_q
            else 0.0
        )
        result.mrr = (
            sum(
                _mrr_of(id_pick(per_query[item.id]), item.gold_chunk_ids, args.candidates)
                for item in answerable
            )
            / total_q
            if total_q
            else 0.0
        )
        if with_ceiling:
            result.recall_at_ceiling = ceiling_hits / total_q if total_q else 0.0

        latencies.sort()
        result.query_latency_p50_ms = latencies[len(latencies) // 2] if latencies else 0
        result.query_latency_p95_ms = latencies[int(len(latencies) * 0.95)] if latencies else 0
        result.avg_context_chars = sum(ctx_chars) / len(ctx_chars) if ctx_chars else 0.0
        result.avg_context_tokens = result.avg_context_chars * 0.67
        if compress:
            base = sum(ctx_orig) or 1
            result.token_saved_ratio = 1.0 - sum(ctx_chars) / base

        # 拒答：top1 分数 < 阈值即正确拒答。**分数来自该组自己的排序**
        refused_ok = 0
        for item in refusals:
            data = per_query[item.id]
            top_score = _top_score_of(data, id_pick(data))
            if top_score is not None and top_score < threshold:
                refused_ok += 1
        result.refusal_correct = refused_ok
        result.refusal_total = len(refusals)
        result.refusal_accuracy = refused_ok / len(refusals) if refusals else 0.0

        # 误拒 / 误放（只在"该答却没答对"的题上算）
        result.false_accept = len(refusals) - refused_ok
        result.false_refusal = sum(
            1 for item in answerable if result.per_question[item.id] == "miss"
        )
        return result

    def _top_score_of(data: dict[str, Any], ids: list[str]) -> float | None:
        """该组排序下 top1 的分数。**只有向量分数有连续值**，
        混合/精排的分数体系不同，但拒答判据是"信号强度" ——
        这里统一用"该组 top1 落在向量分数轴上的位置"作代理（见 docstring 的说明）。
        """
        if not ids:
            return None
        top_id = ids[0]
        # 从向量候选里找这个 id 的分数；找不到说明它不在向量候选里 → 视为低分
        for cid, score in zip(data["vec_ids"], data["vec_scores"], strict=False):
            if cid == top_id:
                return float(score)
        return None

    groups = [
        _score_group(
            "g1_vec",
            "① 朴素向量",
            "—（基线）",
            id_pick=lambda d: d["vec_ids"],
            latency_of=lambda d: d["vec_ms"],
        ),
        _score_group(
            "g2_hybrid",
            "② +混合检索(RRF)",
            "RRF 融合（向量 + BM25）",
            id_pick=lambda d: d["hybrid_ids"],
            latency_of=lambda d: d["vec_ms"] + d["bm25_ms"],
            with_ceiling=True,
        ),
        _score_group(
            "g3_rerank",
            "③ +精排",
            "Cross-Encoder 精排",
            id_pick=lambda d: d["rerank_ids"],
            latency_of=lambda d: d["vec_ms"] + d["bm25_ms"] + d["rerank_ms"],
            with_ceiling=True,
        ),
        _score_group(
            "g4_compress",
            "④ +压缩",
            "句子级压缩（只改 prompt 文本）",
            id_pick=lambda d: d["rerank_ids"],
            latency_of=lambda d: d["vec_ms"] + d["bm25_ms"] + d["rerank_ms"],
            compress=True,
            with_ceiling=True,
        ),
    ]

    # ---------- 输出 ----------
    print()
    print("=" * 104)
    print(
        f"四配置消融（k={TOP_K}，候选 {args.candidates}，{len(answerable)} 条可答 + {len(refusals)} 条拒答）"
    )
    print("=" * 104)
    print(
        f"  {'组':<18}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'R@上限':>9}"
        f"{'拒答':>8}{'p50延迟':>10}{'上下文tok':>11}{'省token':>9}"
    )
    print("  " + "-" * 100)
    for group in groups:
        ceiling = (
            f"{group.recall_at_ceiling * 100:.1f}%" if group.recall_at_ceiling is not None else "—"
        )
        print(
            f"  {group.name:<18}{group.recall_at_5 * 100:>7.1f}%{group.recall_at_10 * 100:>7.1f}%"
            f"{group.mrr:>8.4f}{ceiling:>9}"
            f"{group.refusal_correct}/{group.refusal_total:>3}"
            f"{group.query_latency_p50_ms:>9}ms"
            f"{group.avg_context_tokens:>11.0f}"
            f"{group.token_saved_ratio * 100:>8.1f}%"
        )

    print()
    print("  逐层增量（相对上一组）：")
    for index in range(1, len(groups)):
        prev, cur = groups[index - 1], groups[index]
        d5 = (cur.recall_at_5 - prev.recall_at_5) * 100
        dmrr = cur.mrr - prev.mrr
        dtok = cur.avg_context_tokens - prev.avg_context_tokens
        dlat = cur.query_latency_p50_ms - prev.query_latency_p50_ms
        print(
            f"    {cur.name:<18} R@5 {d5:+5.1f}pp  MRR {dmrr:+.4f}  "
            f"上下文 {dtok:+7.1f} tok  延迟 {dlat:+7} ms"
        )

    # ---------- 答案质量：未测（网关不可用）----------
    answer_quality_note = (
        "未测：本机 LLM 网关返回 502「未配置凭据，请先登录」，"
        "80 次生成（20 条 × 4 组）无法执行。补测入口："
        "python -X utf8 scripts/tasks.py answer-quality（需网关）。"
    )

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus_docs": len(docs),
            "chunks": len(chunks),
            "chunk_chars": cfg.chunk_chars,
            "overlap": cfg.overlap_chars,
            "strategy": cfg.strategy,
            "top_k": TOP_K,
            "candidates": args.candidates,
            "threshold": threshold,
            "answerable": len(answerable),
            "refusals": len(refusals),
            "compress_keep_ratio": COMPRESS_KEEP_RATIO,
            "device": "cpu",
            "index_ms": {
                "vec": vec_index_ms,
                "bm25": bm25_index_ms,
                "rerank_load": rerank_load_ms,
            },
            "index_ms_reference": INDEX_MS_REFERENCE,
            "answer_quality": answer_quality_note,
        },
        "groups": [group.as_dict() for group in groups],
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
    groups = payload["groups"]
    lines: list[str] = []
    lines.append("# 四配置消融（W6 主交付）")
    lines.append("")
    lines.append(
        f"> {meta['generated_at']} ｜ {meta['chunks']} 块 ｜ k={meta['top_k']} "
        f"｜ 候选 {meta['candidates']} ｜ device={meta['device']}"
    )
    lines.append("")
    lines.append("## 一、四组定义（严格累加）")
    lines.append("")
    lines.append("| 组 | 相对上一组只加一件事 |")
    lines.append("| --- | --- |")
    for group in groups:
        lines.append(f"| **{group['name']}** | {group['added']} |")
    lines.append("")
    lines.append("## 二、消融表")
    lines.append("")
    lines.append(
        "| 组 | Recall@5 | Recall@10 | MRR | R@上限 | 拒答 | p50 延迟 | 上下文 token | 省 token |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for group in groups:
        ceiling = (
            f"{group['recall_at_ceiling'] * 100:.1f}%"
            if group["recall_at_ceiling"] is not None
            else "—"
        )
        saved = f"{group['token_saved_ratio'] * 100:.1f}%" if group["token_saved_ratio"] else "—"
        lines.append(
            f"| {group['name']} | **{group['recall_at_5'] * 100:.1f}%** | "
            f"{group['recall_at_10'] * 100:.1f}% | {group['mrr']:.4f} | {ceiling} | "
            f"{group['refusal_correct']}/{group['refusal_total']} | "
            f"{group['query_latency_p50_ms']} ms | {group['avg_context_tokens']:.0f} | {saved} |"
        )
    lines.append("")
    lines.append("## 三、逐层增量")
    lines.append("")
    lines.append("| 层 | ΔRecall@5 | ΔMRR | Δ上下文 token | Δ延迟 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for index in range(1, len(groups)):
        prev, cur = groups[index - 1], groups[index]
        lines.append(
            f"| {cur['name']} | **{(cur['recall_at_5'] - prev['recall_at_5']) * 100:+.1f}pp** | "
            f"{cur['mrr'] - prev['mrr']:+.4f} | "
            f"{cur['avg_context_tokens'] - prev['avg_context_tokens']:+.0f} | "
            f"{cur['query_latency_p50_ms'] - prev['query_latency_p50_ms']:+d} ms |"
        )
    lines.append("")
    lines.append("## 四、答案质量抽检")
    lines.append("")
    lines.append(f"**{meta['answer_quality']}**")
    lines.append("")
    lines.append("## 五、索引耗时")
    lines.append("")
    for key, value in meta["index_ms"].items():
        lines.append(f"- {key}: **{value} ms**")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
