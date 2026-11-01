"""答案质量抽检：从 50 条里抽 20 条走**完整链路**（检索 + 生成），人工判对错。

## 为什么这一步不能省（也不能用自动指标替代）

前面所有指标（Recall@5 / MRR / 拒答准确率）测的都是**检索层**：
"该找的那块有没有进 top_k"。

但用户看到的不是块，是**答案**。而检索对了、答案错了的情况真实存在：

- 5 块里有一块是对的，模型却引用了别的那块
- 材料里只有半个答案，模型把另外半个编了出来
- 检索没命中，模型仍然"依据材料"给了一个自信的错答案

这些**只有跑完整链路才看得见**。所以任务是：抽 20 条，跑，人读，判对错。

## 为什么是 20 条而不是 50

成本。这是本周唯一花 LLM 钱的地方，预算 ¥2-5。
20 条 × 约 2.5k prompt token + 300 completion token ≈ 56k token，
按标准合同成本 0.019 元/次（`docs/cost.md`）算约 ¥0.4 —— 在预算内。

20 条里 1 条 = 5pp，所以**只报"20 条里对了几条"，不报百分比以外的推断**。

## 抽样规则：分层，不是随机

纯随机抽 20 条可能全抽到事实型、或全抽到"检索已命中"的题，
那样测的就不是"端到端质量"。所以分层抽：

| 组 | 抽多少 | 为什么 |
| --- | ---: | --- |
| 事实型·检索命中 | 8 | 看"检索对了，答案是否也对" |
| 事实型·检索未命中 | 4 | 看"检索错了，模型会不会编" |
| 多跳 | 4 | 多跳最可能"答一半" |
| 拒答 | 4 | 看拒答是否生效、有没有硬答 |

**固定种子**（`--seed`），保证任何人重跑抽到的是同一批 —— 否则
"我抽的那 20 条"与"你抽的那 20 条"不可比，而人工判断本来就有主观性。

## 判定口径（写死，否则每次判的标准会漂）

| 判定 | 含义 |
| --- | --- |
| `correct` | 结论与 `gold_answer` 一致，且引用的编号确实支持它 |
| `partially_correct` | 主要事实对，但有遗漏或多说了没依据的话 |
| `wrong` | 结论错，或引用了不存在的编号（`invalid_refs` 非空） |
| `refused_correctly` | 该拒且拒了 |
| `should_refuse_but_answered` | 该拒却答了 —— **这是最危险的一类** |

> `invalid_refs` 非空时**一律记 wrong**，不管结论对不对。
> 理由：引用越界是 W4 的 F3，它让"看起来有依据"变成不可核对 ——
> 那比答错更伤信任。

## 当前状态：**未执行**（网关 502）

实测本机网关：

```
openai.InternalServerError: Error code: 502
{'error': {'message': '未配置凭据，请先登录', ...}}
```

这与 `docs/release-notes-v0.3.1.md` 记录的是同一个问题
（网关鉴权绑宿主登录态，容器内与重新登录后都可能复现）。

所以 `eval/results/answer-quality.json` **没有产出**。
本文档与主表里**不会出现"20 条正确率 X%"这个数字** ——
没有跑就不能写。

**怎么补**：网关恢复后跑

```bash
python -X utf8 scripts/tasks.py answer-quality
```

脚本会把 20 条的 prompt / 回答 / 引用 / 判定表全部落盘，
人工只需填 `verdict` 一列。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_retrieval_eval as rre  # noqa: E402
import strategy_compare as sc  # noqa: E402

DEFAULT_OUT: Path = PROJECT_ROOT / "eval" / "results" / "answer-quality.json"

# 分层抽样配额。**加起来必须等于 20。**
STRATA: dict[str, int] = {
    "fact_hit": 8,
    "fact_miss": 4,
    "multi_hop": 4,
    "refusal": 4,
}

DEFAULT_SEED: int = 20261121

VERDICTS: tuple[str, ...] = (
    "correct",
    "partially_correct",
    "wrong",
    "refused_correctly",
    "should_refuse_but_answered",
)


def recall_level_hits(
    *, strategy: str, chunk_chars: int, embedder: Any, docs: list[Any], items: list[Any], top_k: int
) -> dict[str, bool]:
    """先只做检索，得到"每条题在 top_k 内是否命中"。**抽层要用它。**"""
    from app.rag.stores.memory import InMemoryVectorStore

    chunks = sc.build_chunks(
        docs=docs, strategy=strategy, chunk_chars=chunk_chars, embedder=embedder
    )
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))

    probe = [
        rre.GoldItem(
            id=item.id,
            type=item.type,
            question=item.question,
            anchors=item.anchors,
            gold_answer=item.gold_answer,
            notes=item.notes,
        )
        for item in items
    ]
    rre.resolve_all(probe, chunks)

    hits: dict[str, bool] = {}
    for item in probe:
        vector = embedder.embed_query([item.question])[0]
        found = store.search(vector, top_k=top_k)
        hits[item.id] = rre.evaluate_query(item=item, hits=found, k=top_k).hit
    return hits


def sample_ids(items: list[rre.GoldItem], hits: dict[str, bool], *, seed: int) -> list[str]:
    """分层抽样。**固定种子**，保证可复现。"""
    buckets: dict[str, list[str]] = {name: [] for name in STRATA}
    for item in items:
        if item.is_refusal:
            buckets["refusal"].append(item.id)
        elif item.type == "multi_hop":
            buckets["multi_hop"].append(item.id)
        elif hits.get(item.id):
            buckets["fact_hit"].append(item.id)
        else:
            buckets["fact_miss"].append(item.id)

    rng = random.Random(seed)
    chosen: list[str] = []
    for name, quota in STRATA.items():
        pool = sorted(buckets[name])  # 先排序再打乱，保证同种子同结果
        rng.shuffle(pool)
        if len(pool) < quota:
            print(
                f"  ⚠ 分层 {name} 只有 {len(pool)} 条，少于配额 {quota}",
                file=sys.stderr,
            )
        chosen.extend(pool[:quota])
    return chosen


async def run_one(
    *,
    item: rre.GoldItem,
    strategy: str,
    chunk_chars: int,
    embedder: Any,  # noqa: ANN401
    docs: list[Any],
    client: Any,  # noqa: ANN401
    model: str,
    max_context_chars: int,
    threshold: float,
) -> dict[str, Any]:
    """跑一条完整链路：检索 → 拼上下文 → 调模型 → 收集引用。"""
    from app.rag.generation import Answer, build_context, generate_answer

    chunks = sc.build_chunks(
        docs=docs, strategy=strategy, chunk_chars=chunk_chars, embedder=embedder
    )
    from app.rag.stores.memory import InMemoryVectorStore

    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))
    probe = rre.GoldItem(
        id=item.id,
        type=item.type,
        question=item.question,
        anchors=item.anchors,
        gold_answer=item.gold_answer,
    )
    rre.resolve_all([probe], chunks)

    vector = embedder.embed_query([item.question])[0]
    found = store.search(vector, top_k=5)
    scores = [hit.score for hit in found]
    context, available = build_context([hit.chunk for hit in found], max_chars=max_context_chars)

    started = time.perf_counter()
    try:
        answer: Answer = await generate_answer(
            item.question,
            [hit.chunk for hit in found],
            client=client,
            model=model,
            max_context_chars=max_context_chars,
            min_score=threshold,
            scores=scores,
        )
    except Exception as exc:  # noqa: BLE001 - 单条失败不该毁掉整批
        return {
            "id": item.id,
            "type": item.type,
            "question": item.question,
            "gold_answer": item.gold_answer,
            "error": f"{type(exc).__name__}: {exc}",
            "verdict": None,
        }
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return {
        "id": item.id,
        "type": item.type,
        "question": item.question,
        "gold_answer": item.gold_answer,
        "answer": answer.text,
        "refused": answer.refused,
        "citations": [c.index for c in answer.citations],
        "invalid_refs": list(answer.invalid_refs),
        "top_score": round(max(scores), 4) if scores else None,
        "context_chunks": len(available),
        "context_chars": len(context),
        "llm_ms": elapsed_ms,
        "snippet": [
            {
                "index": index,
                "source": chunk.source,
                "head": chunk.text[:80].replace("\n", " "),
            }
            for index, chunk in enumerate([hit.chunk for hit in found][: len(available)], start=1)
        ],
        # **人填这一列**。留 None 表示还没判，不是"判对了"
        "verdict": None,
        "verdict_note": "",
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """统计已判定的行。**未判定的单独计数，不混进分母。**"""
    judged = [row for row in rows if row.get("verdict") in VERDICTS]
    pending = [row for row in rows if row.get("verdict") not in VERDICTS]
    counts = dict.fromkeys(VERDICTS, 0)
    for row in judged:
        counts[row["verdict"]] += 1

    correct_like = counts["correct"] + counts["refused_correctly"]
    return {
        "total": len(rows),
        "judged": len(judged),
        "pending": len(pending),
        "counts": counts,
        "correct": correct_like,
        "accuracy": round(correct_like / len(judged), 4) if judged else None,
        "invalid_refs_rows": [row["id"] for row in rows if row.get("invalid_refs")],
        "errors": [row["id"] for row in rows if row.get("error")],
        "note": (
            "accuracy 的分母是**已判定**行数，不是 20；未判定的 pending 单列。没有判定就不报准确率"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="答案质量抽检（20 条，走完整链路）")
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--chunk-chars", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（调试）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    from app.core.config import get_settings
    from app.core.llm.client import get_llm_client
    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.settings import get_rag_config

    cfg = get_rag_config()
    strategy = args.strategy or cfg.strategy
    chunk_chars = args.chunk_chars if args.chunk_chars is not None else cfg.chunk_chars
    threshold = args.threshold if args.threshold is not None else cfg.score_threshold

    embedder = CachedEmbedder(get_embedder("bge-small-zh"))
    docs = sc.load_raw_corpus()
    items = rre.load_questions()

    hits = recall_level_hits(
        strategy=strategy,
        chunk_chars=chunk_chars,
        embedder=embedder,
        docs=docs,
        items=items,
        top_k=args.top_k,
    )
    chosen = sample_ids(items, hits, seed=args.seed)
    if args.limit:
        chosen = chosen[: args.limit]
    by_id = {item.id: item for item in items}
    print(f"抽中 {len(chosen)} 条：{chosen}", file=sys.stderr)

    settings = get_settings()
    client = get_llm_client(settings)

    async def run_all() -> list[dict[str, Any]]:
        results = []
        for index, qid in enumerate(chosen, start=1):
            print(f"  [{index}/{len(chosen)}] {qid} …", file=sys.stderr)
            results.append(
                await run_one(
                    item=by_id[qid],
                    strategy=strategy,
                    chunk_chars=chunk_chars,
                    embedder=embedder,
                    docs=docs,
                    client=client,
                    model=settings.llm.model,
                    max_context_chars=cfg.max_context_chars,
                    threshold=threshold,
                )
            )
        return results

    rows = asyncio.run(run_all())
    summary = summarize(rows)

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "strategy": strategy,
            "chunk_chars": chunk_chars,
            "top_k": args.top_k,
            "threshold": threshold,
            "model": settings.llm.model,
            "seed": args.seed,
            "strata": STRATA,
            "verdicts": list(VERDICTS),
            "judgement_rule": (
                "invalid_refs 非空一律记 wrong；correct 要求结论与 gold_answer 一致且引用支持它"
            ),
        },
        "summary": summary,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"已跑 {len(rows)} 条，落盘 {args.out.relative_to(PROJECT_ROOT)}")
    if summary["errors"]:
        print(f"  ⚠ {len(summary['errors'])} 条调用失败：{summary['errors']}")
    print(f"  待人工判定：{summary['pending']} 条（填 rows[].verdict 后重跑本脚本汇总）")
    if summary["accuracy"] is not None:
        print(
            f"  已判定准确率：{summary['accuracy'] * 100:.1f}%（{summary['correct']}/{summary['judged']}）"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
