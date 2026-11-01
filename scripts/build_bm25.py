"""重建 BM25 索引并落盘。**`make build-bm25` 的唯一实现**。

## 为什么单独一个脚本，而不是塞进 `bm25_compare.py`

三个不同的人/场景要用它：

| 谁 | 什么时候 | 要什么 |
| --- | --- | --- |
| 人 | 改了词典 / 改了切分参数后 | 一条命令重建 |
| CI | 每次提交 | 能快速判断"索引该不该重建" |
| 对照脚本 | 跑实验前 | 索引已就绪就不用等 |

塞进对照脚本会让"我只想重建索引"也跑一遍检索 —— 那是 10 秒白等。
而且**重建与对照是两个动作**，混在一起时"对照跑得慢"会被误判成"BM25 慢"。

## 记录什么（任务点名要求）

    chunk 数 / token 数 / 重建耗时 / 落盘体积 / avgdl / k1 / b / 分词器指纹

**耗时分两段记**：`tokenize_ms`（分词，重活）与 `total_ms`（含 pickle 落盘）。
分开记的理由：改词典后重建，分词会变慢（词变了），但落盘不变；
合并成一个数就看不出是哪一段变了。而"重建耗时"这个代价列将来要和
"每 +1pp 召回"算性价比，必须能解释它为什么变。

## 幂等 / 复用

默认**指纹相同就复用**（分词器 + 块 id 集合 + 参数三项都比）。
`--force` 强制重建。指纹不同的判定见 `app/rag/bm25/index.build()`。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import retrieval_harness as rh  # noqa: E402

from app.rag.bm25.index import DEFAULT_B, DEFAULT_INDEX_PATH, DEFAULT_K1, BM25Params  # noqa: E402
from app.rag.bm25.retriever import BM25Retriever  # noqa: E402
from app.rag.bm25.tokenize import describe  # noqa: E402
from app.rag.chunking import ChunkingConfig, build_strategy  # noqa: E402
from app.rag.settings import get_rag_config  # noqa: E402

OUT_PATH: Path = PROJECT_ROOT / "scripts" / "bm25_index_report.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="重建 BM25 索引并落盘")
    parser.add_argument("--out", type=Path, default=DEFAULT_INDEX_PATH, help="索引落盘位置")
    parser.add_argument("--k1", type=float, default=DEFAULT_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_B)
    parser.add_argument("--force", action="store_true", help="无视指纹强制重建")
    parser.add_argument("--no-write", action="store_true", help="只打印，不落报告")
    args = parser.parse_args(argv)

    cfg = get_rag_config()
    started = time.perf_counter()
    docs = rh.load_raw_corpus()
    chunk_cfg = ChunkingConfig(chunk_chars=cfg.chunk_chars, overlap_chars=cfg.overlap_chars)
    chunks = build_strategy(cfg.strategy, cfg=chunk_cfg).chunk(docs, chunk_cfg)
    corpus_ms = int((time.perf_counter() - started) * 1000)
    print(
        f"语料 {len(docs)} 段 → {len(chunks)} 块"
        f"（chunk={cfg.chunk_chars} overlap={cfg.overlap_chars} strategy={cfg.strategy}，"
        f"切分 {corpus_ms} ms）",
        file=sys.stderr,
    )

    retriever, report = BM25Retriever.from_chunks(
        chunks,
        params=BM25Params(k1=args.k1, b=args.b),
        path=args.out,
        force=args.force,
    )
    stats = retriever.stats()

    print()
    print("=" * 70)
    print("BM25 索引已就绪" if not report.reused else "BM25 索引已存在，按指纹复用（未重建）")
    print("=" * 70)
    print(f"  落盘路径   {report.path}")
    print(f"  块 / token {report.chunks} / {report.tokens}")
    print(f"  avgdl      {stats['avgdl']} token/块")
    print(f"  k1 / b     {stats['params']['k1']} / {stats['params']['b']}")
    print(f"  重建耗时   {report.build_ms} ms（分词 + pickle 落盘）")
    print(f"  落盘体积   {report.size_kb} KB")
    print(
        f"  分词器     {report.tokenizer['tokenizer']}，词典 {report.tokenizer['dict_words']} 词，"
        f"停用词 {report.tokenizer['stopwords']}"
    )

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
            "corpus_ms": corpus_ms,
        },
        "report": report.summary(),
        "stats": stats,
        "tokenizer": describe(),
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘：{OUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
