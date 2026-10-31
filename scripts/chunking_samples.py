"""三策略切分样本导出器：**同一份文档，三种策略各跑一遍，导出到 `data/samples/chunking/`**。

## 这个脚本存在的目的

W5 的第一件事是"用数据替换直觉"（`overlap ∈ {0,50,100}` × `top_k ∈ {3,5,8}`）。
但在跑那张矩阵之前，必须先回答一个更基础的问题：

> **这三种策略本身，哪一种切得更好？**

如果策略选错了，矩阵里最优的那个参数组合也只是"错误策略上的较优参数"。

而"切得好不好"**没法靠单块指标回答**（块长分布、块数都不反映质量），
只能人工逐段读 —— 所以这个脚本的产物是**给人读的文件**，不是分数。

## 导出格式为什么是 Markdown 而不是 JSON

读的人要看的是"这一段是不是被拦腰切了"，那需要**连续的上下文**：
JSON 里一块一条，读到第 3 块已经忘了第 2 块结尾是什么。

所以每份样本导出成一个 `.md`：块之间用 `─── [n] chars=… ───` 分隔，
`chars` 直接标出来（判断"这一块是不是异常短/长"用得上）。
块首若有标题路径（heading 策略补的那行），原样显示 —— 那正是要读的东西。

## `--dump-meta`

`metadata` 不导出到正文（会淹掉可读性），但排查"这一块是哪一页"时要看。
`--dump-meta` 会在每块标题行后面附一行紧凑的字段摘要。

## 为什么样本要"各跑一遍"而不是"同一份跑三遍"

任务写的是"**同一份 20 页 PDF + 一份 md 各跑一遍**" —— 即两份文档 × 三种策略 = 六份样本。
这份脚本按这个口径产出 `{文档名}.{策略名}.md`。
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

# 输出目录：`data/` 已在 .gitignore 里（见那个文件的说明）。
# **样本本身不该入库**：它们是"某次运行 + 某个 embedder + 某套参数"的产物，
# 参数一改就全部失效。入库的是脚本和读法（docs/chunking-quality.md）。
DEFAULT_OUT = PROJECT_ROOT / "data" / "samples" / "chunking"

SEPARATOR = "\u2500\u2500\u2500"  # ───


def _embedder(name: str) -> Any:  # noqa: ANN401
    from app.rag.embedding import get_embedder

    return get_embedder(name)


def export_one(
    *,
    source: Path,
    strategy_name: str,
    cfg: Any,  # noqa: ANN401
    out_dir: Path,
    embedder_name: str,
    dump_meta: bool = False,
) -> dict[str, Any]:
    """跑一种策略，导出 md，返回该样本的统计（给汇总表用）。"""
    from app.rag.chunking import build_strategy
    from app.rag.loaders import load_document

    result = load_document(source)
    embedder = _embedder(embedder_name) if strategy_name == "semantic" else None
    strategy = build_strategy(strategy_name, embedder=embedder, cfg=cfg)
    chunks = strategy.chunk(result.documents, cfg)

    lengths = [len(chunk.text) for chunk in chunks]
    text = _render(source, strategy_name, chunks, cfg, embedder_name, dump_meta)

    target = out_dir / f"{source.stem}.{strategy_name}.md"
    target.write_text(text, encoding="utf-8")

    return {
        "document": source.name,
        "strategy": strategy_name,
        "embedder": embedder_name if strategy_name == "semantic" else None,
        "chunks": len(chunks),
        "chars_total": sum(lengths),
        "chars_min": min(lengths) if lengths else 0,
        "chars_max": max(lengths) if lengths else 0,
        "chars_avg": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        # 冗余率 = (各块字数之和 - 原文总字数) / 原文总字数。
        # **语义策略这里理论上近似 0**（它不加重叠），而递归字符约 9.4% ——
        # 这两个数字的差异本身就是策略差异的一部分，值得单独记。
        "source_chars": result.total_chars,
        "redundancy": round((sum(lengths) - result.total_chars) / result.total_chars, 4)
        if result.total_chars
        else 0.0,
        "load_warnings": list(result.warnings),
        "file": str(target.relative_to(PROJECT_ROOT))
        if target.is_relative_to(PROJECT_ROOT)
        else str(target),
        # 降级标记：PDF 走 heading 策略时它会是 "page"。**不记这个字段，
        # "标题策略在 PDF 上表现如何"这条结论就分不清是切标题还是分页。**
        "heading_fallback": _count_fallback(chunks),
        "splitter_labels": _count_splitter(chunks),
    }


def _count_fallback(chunks: list[Any]) -> str | None:
    values = {c.metadata.get("heading_fallback") for c in chunks}
    values.discard(None)
    values.discard("")
    if not values:
        return None
    return ",".join(sorted(str(v) for v in values))


def _count_splitter(chunks: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        key = str(chunk.metadata.get("splitter", "?"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _render(
    source: Path,
    strategy: str,
    chunks: list[Any],
    cfg: Any,  # noqa: ANN401
    embedder_name: str,
    dump_meta: bool,
) -> str:
    """把块渲染成**给人读**的 Markdown。

    顶部写清"这份样本是用什么参数跑出来的"—— 没有这段话，
    一周后打开文件无法判断它对应哪版参数（而 `data/` 里的文件会堆很多份）。
    """
    lines: list[str] = [
        f"# 切分样本：{source.name} · {strategy}",
        "",
        f"- 来源：`{source.relative_to(PROJECT_ROOT) if source.is_relative_to(PROJECT_ROOT) else source}`",
        f"- 策略：`{strategy}`",
        f"- 参数：chunk_chars={cfg.chunk_chars} overlap={cfg.overlap_chars} "
        f"heading_max_level={cfg.heading_max_level} "
        f"semantic_pct={cfg.semantic_breakpoint_percentile} "
        f"semantic_min={cfg.semantic_min_chars} semantic_max={cfg.semantic_max_chars}",
        f"- embedder：`{embedder_name if strategy == 'semantic' else '（本策略不需要）'}`",
        f"- 块数：{len(chunks)}",
        "",
        "> 这是**给人读**的样本，不是分数。读法见 `docs/chunking-quality.md` 的三条判据。",
        "",
        "---",
        "",
    ]

    for position, chunk in enumerate(chunks, start=1):
        header = f"{SEPARATOR} [{position}] chars={len(chunk.text)} {SEPARATOR}"
        lines.append(header)
        if dump_meta:
            meta = {k: v for k, v in chunk.metadata.items() if k not in {"block"}}
            lines.append(f"`pages={chunk.pages}` `{json.dumps(meta, ensure_ascii=False)}`")
        lines.append("")
        lines.append(chunk.text)
        lines.append("")

    return "\n".join(lines)


def summarize(rows: list[dict[str, Any]], out_path: Path) -> None:
    """打印并落盘一张对比表。**这张表是周六做结论的原料，不是结论本身。**"""
    print(
        f"\n{'文档':<22}{'策略':<11}{'块数':>5}{'最小':>5}{'最大':>6}"
        f"{'平均':>7}{'冗余':>8}  降级/来源"
    )
    for row in rows:
        fallback = row["heading_fallback"] or "-"
        print(
            f"{row['document']:<22}{row['strategy']:<11}{row['chunks']:>5}"
            f"{row['chars_min']:>5}{row['chars_max']:>6}{row['chars_avg']:>7}"
            f"{row['redundancy']:>7.1%}  {fallback}"
        )

    out_path.write_text(
        json.dumps(
            {
                "note": (
                    "块数/长度/冗余只是『可量化的那一面』；切分质量要人工按 "
                    "docs/chunking-quality.md 的三条判据读 data/samples/chunking/ 下的样本。"
                ),
                "samples": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n汇总已写入 {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出三策略切分样本（给人读）")
    parser.add_argument(
        "--docs",
        nargs="+",
        type=Path,
        default=[
            PROJECT_ROOT / "tests" / "fixtures" / "sample20.pdf",
            PROJECT_ROOT / "docs" / "w4-checklist.md",
        ],
        help="要导出的文档（默认：20 页 PDF + 一份多级标题的 md）",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["recursive", "heading", "semantic"],
        choices=["recursive", "heading", "semantic"],
    )
    parser.add_argument(
        "--embedder",
        default="hashing",
        help="semantic 策略用的 embedder（hashing 下结论无参考价值，仅用于跑通）",
    )
    parser.add_argument("--chunk-chars", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dump-meta", action="store_true", help="每块附一行 metadata 摘要")
    args = parser.parse_args(argv)

    from app.rag.chunking import ChunkingConfig, build_strategy, validate_config

    cfg = ChunkingConfig(chunk_chars=args.chunk_chars, overlap_chars=args.overlap)
    validate_config(cfg)
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for source in args.docs:
        if not source.exists():
            print(f"跳过（不存在）：{source}", file=sys.stderr)
            continue
        for strategy_name in args.strategies:
            # 先构造一次，把"缺 embedder / 配置非法"这类装配期错误提前炸在这里，
            # 而不是在 export_one 里跑到一半才炸
            build_strategy(
                strategy_name,
                embedder=_embedder(args.embedder) if strategy_name == "semantic" else None,
                cfg=cfg,
            )
            row = export_one(
                source=source,
                strategy_name=strategy_name,
                cfg=cfg,
                out_dir=args.out,
                embedder_name=args.embedder,
                dump_meta=args.dump_meta,
            )
            rows.append(row)
            print(f"✓ {row['file']}（{row['chunks']} 块）")

    summarize(rows, args.out / "summary.json")
    print(f"\n样本目录：{args.out}")
    print("下一步：人工按 docs/chunking-quality.md 的三条判据逐段读，把打分写回那张表。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
