"""过滤场景评测：**加过滤前后，命中率到底变没变**。

## 为什么必须单独做这一步

上一轮的 `tests/test_filters.py` 证明了"过滤器能缩小结果集"、
"写错条件会抛异常"。但那个任务真正要的验收是：

> 测三种场景：**加过滤前后命中率**、以及「过滤条件写错时会不会静默返回空」

第一条从未测过。而它恰恰是最容易被想当然的一条 ——
"过滤当然是提升命中率"听起来天经地义，但**它只在过滤条件正确时成立**，
而且提升幅度取决于"该文档在整个库里的占比"。

## 三种过滤场景各自测什么

| 场景 | 条件 | 它该提升什么 |
| --- | --- | --- |
| 单文档 | `{"source": "sse.md"}` | 排除其他 12 份文档的干扰 |
| 章节 | `{"section_path": ...}` | **只在 heading 策略下可用**（recursive 没有这个字段） |
| 表格 | `{"block": "table"}` | 让表格类问题只命中表格块 |

## 怎么算"命中率变了"

对每条 gold，跑两次：不加过滤、加过滤。然后比：

    hit_without → hit_with        （记住"未命中→命中"和"命中→未命中"两个方向）

**只看总数会漏掉反向翻转。** 加过滤也可能**弄丢**原本能命中的问题 ——
比如答案所在块的 `source` 与 gold 写的文档名不一致（命中被过滤掉），
或者答案那个块被标成了 table 而问的是正文。

## 一个必须先解决的陷阱：gold 的 source 与块的 source 不是同一个字符串

`eval/retrieval-gold.jsonl` 里写 `docs/sse.md`，而 `Chunk.source`
只有文件名 `sse.md`（loader 用 `file_path.name`）。
直接拿 gold 的 source 去 `where` 会**抛 `UnknownFilterValue`** ——
这正好是上一轮做的校验在起作用（否则会静默返回空，然后得出
"过滤让命中率归零"的错误结论）。

所以这里做显式映射，并把映射本身当作一条被测项：
**如果哪天 gold 改成写文件名，这个映射要跟着改**，测试会提醒。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import retrieval_harness as rh  # noqa: E402
from retrieval_harness import PROJECT_ROOT, load_corpus, load_gold  # noqa: E402

DEFAULT_OUT: Path = PROJECT_ROOT / "scripts" / "filter_eval.json"

FIXED_CHUNK_CHARS: int = 500
FIXED_OVERLAP: int = 50
FIXED_TOP_K: int = 5


def gold_source_to_chunk_source(gold_source: str) -> str:
    """`docs/sse.md` → `sse.md`。

    `Chunk.source` 来自 `Document.source`，而 loader 填的是 `file_path.name`
    （见 `loaders/markdown.py`）——**不含目录**。

    为什么不改成统一存相对路径：那会让所有已入库的块 `store_id` 变化
    （`doc_id` 由 source 派生），等于全量重建索引。代价远大于这里一行转换。
    """
    return Path(gold_source).name


def load_table_gold(path: Path = PROJECT_ROOT / "eval" / "table-gold.jsonl") -> list[rh.GoldItem]:
    """读"答案在表格里"的问题集。**anchor 必须落在唯一一个 table 块上**。

    这套金标是专门为"仅表格"场景造的：主 gold 集刻意避开了表格行
    （subagent 的选题口径），所以用它测 table 过滤会得到 0 条样本 ——
    而"0 条样本"和"过滤没用"是两件完全不同的事。
    """
    if not path.exists():
        raise FileNotFoundError(f"表格金标不存在：{path}")
    items: list[rh.GoldItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        items.append(
            rh.GoldItem(
                id=raw["id"],
                question=raw["question"],
                anchor=raw["anchor"],
                source=raw["source"],
            )
        )
    return items


def build_index(*, embedder: Any, strategy: str) -> tuple[Any, list[Any]]:  # noqa: ANN401
    from app.rag.chunking import ChunkingConfig, build_strategy
    from app.rag.stores.memory import InMemoryVectorStore

    corpus = load_corpus()
    for warning in corpus.warnings:
        print(f"  ⚠ {warning}", file=sys.stderr)
    cfg = ChunkingConfig(chunk_chars=FIXED_CHUNK_CHARS, overlap_chars=FIXED_OVERLAP)
    chunks = build_strategy(strategy, cfg=cfg).chunk(corpus.all_docs, cfg)
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))
    return store, chunks


def rank_of(*, hits: list[Any], anchor: str) -> int | None:
    """命中位置（1 起）；未命中为 None。"""
    for position, hit in enumerate(hits, start=1):
        if anchor in hit.chunk.text:
            return position
    return None


def eval_scenario(
    *,
    name: str,
    items: list[Any],
    store: Any,  # noqa: ANN401
    embedder: Any,  # noqa: ANN401
    where_of: Any,  # noqa: ANN401
    schema: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """对一个过滤场景跑"过滤前 vs 过滤后"。

    `where_of(item)` 返回该条 gold 的过滤条件；返回 `None` 表示跳过
    （比如问的是表格但该文档没有表格可过滤）。
    """
    from app.rag.filters import guard_where

    rows: list[dict[str, Any]] = []
    for item in items:
        vector = embedder.embed_query([item.question])[0]
        raw_where = where_of(item)
        if raw_where is None:
            continue
        # **校验放在这里**：条件写错应当抛，而不是变成一条"未命中"。
        checked = guard_where(schema, raw_where)

        before = store.search(vector, top_k=FIXED_TOP_K)
        after = store.search(vector, top_k=FIXED_TOP_K, where=checked)

        rank_before = rank_of(hits=before, anchor=item.anchor)
        rank_after = rank_of(hits=after, anchor=item.anchor)
        rows.append(
            {
                "id": item.id,
                "where": checked,
                "rank_before": rank_before,
                "rank_after": rank_after,
                "hit_before": rank_before is not None,
                "hit_after": rank_after is not None,
                "hits_before": len(before),
                "hits_after": len(after),
            }
        )

    gained = [row["id"] for row in rows if row["hit_after"] and not row["hit_before"]]
    lost = [row["id"] for row in rows if row["hit_before"] and not row["hit_after"]]
    hit_before = sum(1 for row in rows if row["hit_before"])
    hit_after = sum(1 for row in rows if row["hit_after"])
    return {
        "scenario": name,
        "queries": len(rows),
        "hit_before": hit_before,
        "hit_after": hit_after,
        "hit_rate_before": round(hit_before / len(rows), 4) if rows else 0.0,
        "hit_rate_after": round(hit_after / len(rows), 4) if rows else 0.0,
        "delta_pp": round((hit_after - hit_before) / len(rows) * 100, 1) if rows else 0.0,
        "gained": gained,
        "lost": lost,
        "rows": rows,
    }


def probe_misconfigurations(*, store: Any, embedder: Any, schema: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """「过滤条件写错时会不会静默返回空」——逐个试探。

    每条都记录：**不过校验时直接调 store 会返回什么**，以及**过校验会怎样**。
    前者是历史行为（静默 0 条），后者是现在的行为（抛异常）。
    """
    from app.rag.filters import FilterError, guard_where

    vector = embedder.embed_query(["付款"])[0]
    cases = [
        ("键名拼错", {"blockk": "table"}),
        ("文档名拼错", {"source": "不存在的文档.md"}),
        ("文档名带目录", {"source": "docs/sse.md"}),
        ("recursive 下用 section_path", {"section_path": "架构"}),
        ("合法条件（对照）", {"source": "sse.md"}),
    ]
    results: list[dict[str, Any]] = []
    for label, where in cases:
        # ① 绕过校验，直接查库 —— 这是"以前"的行为
        raw_count = len(store.search(vector, top_k=5, where=where))
        # ② 过校验 —— 这是"现在"的行为
        try:
            checked = guard_where(schema, where)
            outcome = "pass"
            message = ""
            guarded_count = len(store.search(vector, top_k=5, where=checked))
        except FilterError as exc:
            outcome = "raise"
            message = type(exc).__name__
            guarded_count = None
        results.append(
            {
                "case": label,
                "where": where,
                "raw_count": raw_count,
                "guarded_outcome": outcome,
                "guarded_error": message,
                "guarded_count": guarded_count,
            }
        )
    return results


def render(scenarios: list[dict[str, Any]], probes: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append(
        f"过滤场景评测（chunk={FIXED_CHUNK_CHARS}/overlap={FIXED_OVERLAP}/top_k={FIXED_TOP_K}）"
    )
    lines.append("")
    lines.append("  ① 加过滤前后的命中率")
    lines.append(
        f"    {'场景':<12} {'条目':>4} {'过滤前':>8} {'过滤后':>8} {'Δ':>8}  {'新增命中':>8} {'丢失命中':>8}"
    )
    lines.append("    " + "-" * 74)
    for row in scenarios:
        lines.append(
            f"    {row['scenario']:<12} {row['queries']:>4} "
            f"{row['hit_rate_before'] * 100:>7.1f}% {row['hit_rate_after'] * 100:>7.1f}% "
            f"{row['delta_pp']:>+7.1f}pp {str(len(row['gained'])):>8} {str(len(row['lost'])):>8}"
        )

    for row in scenarios:
        if not row["rows"]:
            continue
        lines.append("")
        lines.append(f"    ── {row['scenario']} 逐条")
        for item in row["rows"]:
            before = f"top{item['rank_before']}" if item["rank_before"] else "未命中"
            after = f"top{item['rank_after']}" if item["rank_after"] else "未命中"
            flag = ""
            if item["hit_after"] and not item["hit_before"]:
                flag = "  ← 过滤救回"
            elif item["hit_before"] and not item["hit_after"]:
                flag = "  ← 过滤弄丢"
            lines.append(
                f"       {item['id']:<8} {before:>7} → {after:>7}   "
                f"({item['hits_before']}→{item['hits_after']} 条){flag}"
            )

    lines.append("")
    lines.append("  ② 过滤条件写错时会不会静默返回空")
    lines.append(f"    {'场景':<22} {'原样查库':>9} {'过校验':>8} {'校验后查库':>11}")
    lines.append("    " + "-" * 56)
    for probe in probes:
        guarded = (
            f"{probe['guarded_count']} 条"
            if probe["guarded_count"] is not None
            else f"抛 {probe['guarded_error']}"
        )
        lines.append(
            f"    {probe['case']:<22} {probe['raw_count']:>7} 条 "
            f"{probe['guarded_outcome']:>8} {guarded:>11}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="过滤场景评测：加过滤前后的命中率")
    parser.add_argument("--embedder", choices=["bge", "hashing"], default="bge")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    from app.rag.embedding import get_embedder
    from app.rag.embedding.cache import CachedEmbedder
    from app.rag.filters import schema_from_store

    if args.embedder == "hashing":
        print("⚠ hashing 下命中率无参考价值", file=sys.stderr)
        embedder: Any = get_embedder("hashing")
    else:
        embedder = CachedEmbedder(get_embedder("bge-small-zh"))

    store, chunks = build_index(embedder=embedder, strategy=rh.INDEX_STRATEGY)
    schema = schema_from_store(store)
    gold = load_gold()
    print(f"索引 {len(chunks)} 块，gold {len(gold)} 条", file=sys.stderr)
    print(f"可过滤维度：{schema.describe()}", file=sys.stderr)

    # ---- 场景一：单文档过滤（每条 gold 限定在它自己的文档里）
    single_doc = eval_scenario(
        name="单文档",
        items=gold,
        store=store,
        embedder=embedder,
        where_of=lambda item: {"source": gold_source_to_chunk_source(item.source)},
        schema=schema,
    )

    # ---- 场景二：表格过滤
    # 用**专属的表格金标**，因为主 gold 集刻意避开了表格行 ——
    # 用它测 table 过滤会得到 0 条样本，而"没有样本"与"过滤没用"不是一回事。
    table_items = load_table_gold()
    table = eval_scenario(
        name="仅表格",
        items=table_items,
        store=store,
        embedder=embedder,
        where_of=lambda _item: {"block": "table"},
        schema=schema,
    )

    # ---- 场景三：章节过滤
    # **必须用 heading 索引**：section_path 只有 heading 策略会写
    # （见 filters.py 的说明）。这里单独建一份索引，并与同一策略的
    # "不过滤"基线比较 —— 否则"Δ"里会混进策略差异，不是过滤的效果。
    print("\n按章节：用 heading 策略另建一份索引…", file=sys.stderr)
    section_store, section_chunks = build_index(embedder=embedder, strategy="heading")
    section_schema = schema_from_store(section_store)
    print(f"  heading 索引 {len(section_chunks)} 块", file=sys.stderr)

    def section_where(item: rh.GoldItem) -> dict[str, Any] | None:
        """把 gold 的 anchor 落到它所在的 section_path 上。

        这就是"按某章过滤"的真实形态：用户知道答案在哪一章。
        用 anchor 反查而不是手工写章节名，是为了让 20 条 gold 全都能参与 ——
        手写会因为路径写错而被校验拦住，那就变成在测我的手误。
        """
        paths = {
            str(chunk.metadata.get("section_path"))
            for chunk in section_chunks
            if item.anchor in chunk.text and chunk.metadata.get("section_path")
        }
        if len(paths) != 1:
            # anchor 跨多个节（或不在任何节里）→ 这条不适合"按章过滤"，跳过并记录
            return None
        return {"section_path": next(iter(paths))}

    section = eval_scenario(
        name="按章节",
        items=gold,
        store=section_store,
        embedder=embedder,
        where_of=section_where,
        schema=section_schema,
    )
    section["skipped"] = len(gold) - section["queries"]
    section["note"] = (
        "跳过的是 anchor 跨节或不在任何节的 gold；基线也是 heading 索引，所以 Δ 只反映过滤的影响"
    )

    probes = probe_misconfigurations(store=store, embedder=embedder, schema=schema)

    scenarios = [single_doc, table, section]
    print()
    print(render(scenarios, probes))

    payload = {
        "meta": {
            "chunk_chars": FIXED_CHUNK_CHARS,
            "overlap": FIXED_OVERLAP,
            "top_k": FIXED_TOP_K,
            "strategy": rh.INDEX_STRATEGY,
            "embedder": args.embedder,
            "chunks": len(chunks),
            "gold": len(gold),
            "schema": schema.describe(),
        },
        "scenarios": scenarios,
        "misconfiguration_probe": probes,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘：{args.out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
