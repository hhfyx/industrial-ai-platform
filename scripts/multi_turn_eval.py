"""多轮改写对照实验（W8 周一 21:40 格）：改写前 vs 改写后的检索命中。

## 一条命令

    python -X utf8 scripts/multi_turn_eval.py            # 真 LLM 改写（花钱，10 条 × 1 次调用）
    python -X utf8 scripts/multi_turn_eval.py --dry-run  # 跳过改写，只跑两臂检索（零成本冒烟）

## 实验设计

10 条多轮对话（`eval/multi-turn-dialogs.json`），每条两臂：

| 臂 | 检索输入 | 测什么 |
| --- | --- | --- |
| A（改写前） | follow_up 原文（"它的公差""上面那个型号"） | 基线：代词直接进检索的下场 |
| B（改写后） | rewrite(follow_up, history).standalone_question | 改写层的收益 |

命中判定**完全复用** `retrieval_harness.evaluate_query` 的锚点口径
（逐字原文片段是否落在 top_k 块里）—— 与 50 条检索评测同一把尺子，
数字才可互相比。

## 计分口径（歧义样本单独一行，不计入分母）

| 样本类型 | 计分 |
| --- | --- |
| `ambiguity=none` 且改写未降级 | 进主分母：hit@5(A) vs hit@5(B) |
| `ambiguity=ambiguous` | **单独一行**：只记录改写层是否正确标记 needs_clarification，不比较检索命中 |
| 改写降级（degraded=true） | 进主分母但 B 臂等于原问（等价于没改写），行内标 degraded |

歧义样本混进主分母会污染结论：它的"正确行为"是反问而不是命中，
拿原问去检索命中与否没有意义 —— 计入只会把噪声引进分母。

## 落盘（W7 纪律）

- 每次改写调用经 `set_rewrite_sink` 落进 `eval/results/rewrite-<ts>.jsonl`
  （含降级、歧义标记——"改了什么/为什么没改"必须可回放）。
- 汇总表 `eval/results/multi-turn-<ts>.json` + `.md`。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from retrieval_harness import (  # noqa: E402
    GoldItem,
    evaluate_query,
    full_corpus_sources,
    load_raw_corpus,
)

from app.rag.embedding import get_embedder  # noqa: E402
from app.rag.query_rewrite import RewriteOutcome, rewrite, set_rewrite_sink  # noqa: E402
from app.rag.retrieval import DEFAULT_TOP_K  # noqa: E402
from app.rag.stores import InMemoryVectorStore  # noqa: E402

DIALOGS_PATH: Path = PROJECT_ROOT / "eval" / "multi-turn-dialogs.json"
RESULTS_DIR: Path = PROJECT_ROOT / "eval" / "results"
KS: tuple[int, ...] = (5,)
TOP_K: int = DEFAULT_TOP_K


# --------------------------------------------------------------------------- 语料与索引
# 与 run_retrieval_eval 同构：loader 原样粒度（线上同构）+ hashing（离线可复现）。


def build_index() -> tuple[Any, Any]:
    """建一个 offline 可复现的检索器（hashing embedder + memory store）。

    用 hashing 而不是 bge：对照实验比的是**两臂相对差**，不是绝对召回率。
    hashing 离线、确定性、秒级建库 —— 让实验可以反复跑而不烧 API/模型时间。
    绝对召回率是 questions-50 那条线的事（bge + 真库），不在这里重复。

    块的来源与 `run_retrieval_eval` 同构：loader 原样 Document → recursive 切分。
    `Indexer` 不在这里用 —— 它的职责是"Document → 切分 → 编码 → 入库"整链，
    而本实验要自己控制切分输入（原样粒度），直接 embed + upsert 已切好的块。
    """
    from app.rag.chunking import ChunkingConfig, build_strategy

    docs = load_raw_corpus()
    cfg = ChunkingConfig()
    chunks = build_strategy("recursive", cfg=cfg).chunk(docs, cfg)

    embedder = get_embedder("hashing")
    store = InMemoryVectorStore()
    vectors = embedder.embed([c.text for c in chunks])
    store.upsert(chunks, vectors)
    return embedder, store


# --------------------------------------------------------------------------- 改写记录落盘


class RewriteRecorder:
    """把每一次 rewrite() 的记录追加进 jsonl（W7 纪律：可回放）。"""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh = path.open("w", encoding="utf-8")

    def __call__(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# --------------------------------------------------------------------------- 两臂执行


@dataclass
class ArmResult:
    """一臂的检索结果。"""

    query: str
    hit_rank: int | None
    top_scores: list[float] = field(default_factory=list)

    @property
    def hit(self) -> bool:
        return self.hit_rank is not None


def run_arm(
    embedder: Any,
    store: Any,
    query: str,
    anchor: dict[str, str],
) -> ArmResult:
    """对一条 query 跑检索并按锚点判命中（复用 harness 口径）。"""
    hits = store.search(embedder.embed([query])[0], top_k=TOP_K)
    gold = GoldItem(id="x", question=query, anchor=anchor["quote"], source=anchor["source"])
    outcome = evaluate_query(hits, gold)
    return ArmResult(query=query, hit_rank=outcome.hit_rank, top_scores=outcome.top_scores)


async def rewrite_follow_up(dialog: dict[str, Any]) -> tuple[RewriteOutcome, float]:
    """改写一条追问，返回（产物, 耗时 ms）。"""
    started = time.perf_counter()
    history = [{"role": m["role"], "content": m["content"]} for m in dialog["turns"]]
    outcome = await rewrite(dialog["follow_up"], history)
    return outcome, (time.perf_counter() - started) * 1000


# --------------------------------------------------------------------------- 主流程


def load_dialogs(path: Path = DIALOGS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"对话集不存在：{path}")
    dialogs = json.loads(path.read_text(encoding="utf-8"))
    seen: set[str] = set()
    for d in dialogs:
        if d["id"] in seen:
            raise ValueError(f"对话 id 重复：{d['id']}")
        seen.add(d["id"])
        for key in ("turns", "follow_up", "ambiguity", "note"):
            if key not in d:
                raise ValueError(f"{d['id']} 缺字段 {key!r}")
        if d["ambiguity"] not in ("none", "ambiguous"):
            raise ValueError(f"{d['id']} ambiguity 非法：{d['ambiguity']!r}")
        if d["ambiguity"] == "none" and not d.get("anchors"):
            raise ValueError(f"{d['id']} 非歧义样本必须有 anchors（否则无法判命中）")
    return dialogs


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


async def run(*, dry_run: bool) -> int:
    dialogs = load_dialogs()
    print(f"语料：loader 原样粒度（{len(full_corpus_sources())} 份）· hashing · top_k={TOP_K}")
    embedder, store = build_index()
    print(f"索引完成：{store.count()} 块\n")

    stamp = _ts()
    recorder = RewriteRecorder(RESULTS_DIR / f"rewrite-{stamp}.jsonl")
    set_rewrite_sink(recorder)

    rows: list[dict[str, Any]] = []
    try:
        for dialog in dialogs:
            did = dialog["id"]
            ambiguous = dialog["ambiguity"] == "ambiguous"
            anchor = dialog["anchors"][0] if dialog["anchors"] else None

            # ---- A 臂：改写前（follow_up 原文直接检索）----
            arm_a = run_arm(embedder, store, dialog["follow_up"], anchor) if anchor else None

            # ---- B 臂：改写后 ----
            if dry_run:
                # 零成本冒烟：不调模型，用人工参考改写充当 B 臂
                outcome = RewriteOutcome(
                    standalone_question=dialog.get("rewritten_reference") or dialog["follow_up"],
                    rewrote=bool(dialog.get("rewritten_reference")),
                )
                rewrite_ms = 0.0
                degraded = False
            else:
                outcome, rewrite_ms = await rewrite_follow_up(dialog)
                degraded = outcome.degraded

            arm_b = (
                run_arm(embedder, store, outcome.standalone_question, anchor) if anchor else None
            )

            if ambiguous:
                # 歧义样本：单独统计改写层的行为，不比检索命中。
                # dry-run 不调模型 → 改写层没参与，标记判定记 skipped 而不是 ✗
                # （✗ 会让人误以为改写层没通过歧义边界）。
                if dry_run:
                    rows.append(
                        {
                            "id": did,
                            "type": "ambiguous",
                            "follow_up": dialog["follow_up"],
                            "needs_clarification": None,
                            "missing": [],
                            "expected_needs_clarification": True,
                            "correct_flag": None,
                            "degraded": False,
                            "skipped": True,
                            "note": dialog["note"],
                        }
                    )
                    print(f"[{did}] 歧义样本 · dry-run 跳过改写判定")
                    continue
                expected_flag = dialog.get("rewritten_reference") is None
                rows.append(
                    {
                        "id": did,
                        "type": "ambiguous",
                        "follow_up": dialog["follow_up"],
                        "rewritten": outcome.standalone_question if outcome.rewrote else None,
                        "needs_clarification": outcome.needs_clarification,
                        "missing": outcome.missing,
                        "expected_needs_clarification": expected_flag,
                        "correct_flag": outcome.needs_clarification == expected_flag,
                        "degraded": degraded,
                        "note": dialog["note"],
                    }
                )
                flag = (
                    "✓ 正确标记歧义"
                    if outcome.needs_clarification == expected_flag
                    else "✗ 未按预期标记"
                )
                print(f"[{did}] 歧义样本 · {flag}")
                continue

            assert arm_a is not None and arm_b is not None
            delta = (arm_b.hit_rank or 99) - (arm_a.hit_rank or 99)
            rows.append(
                {
                    "id": did,
                    "type": "scored",
                    "follow_up": dialog["follow_up"],
                    "standalone": outcome.standalone_question,
                    "rewrote": outcome.rewrote,
                    "degraded": degraded,
                    "rewrite_ms": round(rewrite_ms),
                    "a_query": arm_a.query,
                    "a_hit_rank": arm_a.hit_rank,
                    "a_scores": arm_a.top_scores,
                    "b_query": arm_b.query,
                    "b_hit_rank": arm_b.hit_rank,
                    "b_scores": arm_b.top_scores,
                    "delta_rank": delta,
                    "note": dialog["note"],
                }
            )
            a = f"hit@{arm_a.hit_rank}" if arm_a.hit else "miss"
            b = f"hit@{arm_b.hit_rank}" if arm_b.hit else "miss"
            arrow = "→" if delta <= 0 else "↑" if arm_b.hit and not arm_a.hit else "↓"
            degraded_mark = "（降级）" if degraded else ""
            print(f"[{did}] A {a} → B {b} {arrow}{degraded_mark}  「{dialog['follow_up'][:18]}…」")
    finally:
        set_rewrite_sink(None)
        recorder.close()

    # ---- 汇总 ----
    scored = [r for r in rows if r["type"] == "scored" and not r["degraded"]]
    hits_a = sum(1 for r in scored if r["a_hit_rank"] is not None)
    hits_b = sum(1 for r in scored if r["b_hit_rank"] is not None)
    mrr_a = sum(1 / r["a_hit_rank"] for r in scored if r["a_hit_rank"]) / max(len(scored), 1)
    mrr_b = sum(1 / r["b_hit_rank"] for r in scored if r["b_hit_rank"]) / max(len(scored), 1)
    summary = {
        "scored": len(scored),
        "hit_a": hits_a,
        "hit_b": hits_b,
        "hit5_a": round(hits_a / max(len(scored), 1), 4),
        "hit5_b": round(hits_b / max(len(scored), 1), 4),
        "mrr_a": round(mrr_a, 4),
        "mrr_b": round(mrr_b, 4),
        "lift": round(hits_b - hits_a, 4),
        "degraded_rows": sum(1 for r in rows if r["type"] == "scored" and r["degraded"]),
        "ambiguous_rows": sum(1 for r in rows if r["type"] == "ambiguous"),
        "ambiguous_correct": sum(1 for r in rows if r["type"] == "ambiguous" and r["correct_flag"]),
        "dry_run": dry_run,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_json = RESULTS_DIR / f"multi-turn-{stamp}.json"
    out_json.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_md = RESULTS_DIR / f"multi-turn-{stamp}.md"
    lines = [
        "# 多轮改写对照实验",
        "",
        f"- 时间：{stamp} · dry_run={dry_run} · 语料=hashing/memory（对照实验专用）",
        f"- 主分母：{summary['scored']} 条（歧义 {summary['ambiguous_rows']} 条单独统计，降级 {summary['degraded_rows']} 条标 degraded）",
        "",
        "| 臂 | hit@5 | MRR |",
        "| --- | ---: | ---: |",
        f"| A（改写前） | {summary['hit5_a']:.0%} | {summary['mrr_a']:.4f} |",
        f"| B（改写后） | {summary['hit5_b']:.0%} | {summary['mrr_b']:.4f} |",
        "",
        f"**提升：{summary['lift']:+d} 条**（样本只有 {summary['scored']} 条，"
        "1 条翻转就是 10pp —— 只看方向，不做显著性声明）",
        "",
        "## 歧义样本（不计入分母）",
        "",
        "| id | 是否按预期标记 needs_clarification | missing |",
        "| --- | --- | --- |",
        *[
            f"| {r['id']} | {'✓' if r['correct_flag'] else '✗'} | {'; '.join(r['missing']) or '—'} |"
            for r in rows
            if r["type"] == "ambiguous"
        ],
        "",
        "## 逐条明细",
        "",
        "| id | A | B | 改写后的问题 |",
        "| --- | --- | --- | --- |",
        *[
            f"| {r['id']} | {'hit@' + str(r['a_hit_rank']) if r['a_hit_rank'] else 'miss'}"
            f" | {'hit@' + str(r['b_hit_rank']) if r['b_hit_rank'] else 'miss'}"
            f" | {r.get('standalone') or r.get('rewritten') or '—'} |"
            for r in rows
            if r["type"] == "scored"
        ],
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"\nhit@5：A {summary['hit5_a']:.0%} → B {summary['hit5_b']:.0%}"
        f"（{summary['lift']:+d} 条）· MRR {summary['mrr_a']} → {summary['mrr_b']}"
    )
    print(
        f"歧义：{summary['ambiguous_correct']}/{summary['ambiguous_rows']} 正确标记"
        f" · 降级：{summary['degraded_rows']} 条"
    )
    print(f"落盘：{out_json.name} · {out_md.name} · rewrite-{stamp}.jsonl")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="跳过改写调用（零成本冒烟）")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(run(dry_run=args.dry_run))
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
