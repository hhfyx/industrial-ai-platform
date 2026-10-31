"""RAG 问答 CLI：`python -m app.cli.ask "问题" [--docs ...]`。

**CLI 在这一格的作用不是"给用户用"，而是"给联调提供一个不依赖浏览器的入口"。**
答案不对时，先在这里跑一遍能立刻二分出是"没检索到"、"检索到了但分数低"、
还是"模型没用好"—— 而如果是前端问题，这里就完全正常。

## 三个输出层次，对应三种排查需求

    （默认）      人能读的答案 + 引用列表
    --debug       加上检索到的每一块（分数、来源、页码、正文前 60 字）
    --json        结构化输出，供脚本消费（含 scores / refused / invalid_refs）

`--debug` 是刻意做厚的：**"为什么答错了"这个问题的 90% 答案在检索结果里**，
不在模型输出里。让人一眼看到"命中的是哪 5 条"，比让他去读日志快得多。

## 拒答是正常输出，不是错误

检索无结果或最高分低于阈值时，输出固定话术并以**退出码 0** 结束。
退出码非 0 表示"程序出错"（没配 key、文档读不了）——
把"拒答"和"出错"混成同一个退出码，会让调用方脚本把"知识库没有"当成"服务挂了"。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.llm.client import get_llm_client  # noqa: E402
from app.rag.embedding import get_embedder  # noqa: E402
from app.rag.pipeline import RagPipeline  # noqa: E402


def _force_utf8_stdout() -> None:
    """Windows 控制台默认 GBK，中文会崩。

    必须在任何 print 之前调用 —— 所以放在 `main()` 的第一行，
    而不是模块导入时（导入即产生副作用是另一种坏习惯）。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with suppress(ValueError, OSError):
                reconfigure(encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli.ask",
        description="对知识库提问，答案带 [n] 引用",
    )
    parser.add_argument("question", nargs="+", help="要问的问题（可不加引号，多个词会拼起来）")
    parser.add_argument(
        "--docs",
        nargs="*",
        default=[],
        help="要索引的文档路径（.pdf/.docx/.md）。传了就先索引再提问",
    )
    parser.add_argument("--top-k", type=int, default=5, help="检索取几条（默认 5）")
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="低于该相似度视为未找到（默认取 embedder 的推荐值；传 0 关闭闸门）",
    )
    parser.add_argument(
        "--embedder",
        choices=("hashing", "bge-small-zh"),
        default="hashing",
        help="向量化实现：hashing（离线，无语义）/ bge-small-zh（真语义，需模型）",
    )
    parser.add_argument(
        "--store",
        choices=("memory", "chroma"),
        default="memory",
        help="向量库：memory（不持久化）/ chroma（data/chroma）",
    )
    parser.add_argument("--debug", action="store_true", help="打印检索到的每一块与分数")
    parser.add_argument("--json", action="store_true", help="结构化输出")
    return parser


def _index_docs(pipeline: RagPipeline, paths: list[str], *, debug: bool) -> int:
    """索引文档。返回失败份数（不是异常 —— 一份坏文件不该让整批白跑）。"""
    failed = 0
    for path in paths:
        try:
            report = pipeline.index_file(path)
        except Exception as exc:  # noqa: BLE001 - 单文件失败不该毁掉整批
            print(f"✗ 索引失败 {path}：{type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1
            continue
        status = "✓" if report.ok else "✗"
        print(
            f"{status} 索引 {Path(path).name}："
            f"{report.documents} 段 → {report.chunks} 块，库内共 {report.store_total} 条"
            f"（{report.elapsed_ms} ms）",
            file=sys.stderr,
        )
        for warning in report.warnings:
            print(f"  ⚠ {warning}", file=sys.stderr)
        if not report.ok:
            failed += 1
        elif debug:
            print(f"  splitter 报告：{report.to_dict()}", file=sys.stderr)
    return failed


async def _run(args: argparse.Namespace) -> int:
    question = " ".join(args.question).strip()
    if not question:
        print("问题不能为空。", file=sys.stderr)
        return 2

    settings = get_settings()
    embedder = get_embedder(args.embedder)
    pipeline = RagPipeline(
        store=_make_store(args.store),
        embedder=embedder,
        top_k=args.top_k,
    )
    # **阈值从 embedder 取**（见 `Embedder.recommended_min_score`）：
    # 不同模型的相似度分布完全不同，全局常量必然在某个模型上误判。
    threshold = embedder.recommended_min_score if args.min_score is None else args.min_score

    if args.docs:
        failed = _index_docs(pipeline, args.docs, debug=args.debug)
        if failed == len(args.docs):
            # 全部失败才算真失败；部分失败继续，库里仍有可用内容
            print("所有文档都索引失败了，无法提问。", file=sys.stderr)
            return 2
        if pipeline.store.count() == 0:
            print("索引后库仍为空，无法提问。", file=sys.stderr)
            return 2

    # 先检索一次：`--debug` 要打印它，`answer()` 内部还会再检索一次。
    # 这看起来是重复工作，但**只有这样才能在没调模型之前就展示检索结果** ——
    # 而"先看检索"正是排查 RAG 的第一动作。多花一次 embed 调用（约 10 ms）值。
    retrieved = pipeline.retrieve(question)

    if args.debug:
        print(
            f"\n检索：{len(retrieved.hits)} 条，{retrieved.elapsed_ms} ms，"
            f"阈值 {threshold}（{args.embedder}）",
            file=sys.stderr,
        )
        for rank, hit in enumerate(retrieved.hits, start=1):
            print(
                f"  [{rank}] score={hit.score:+.4f}  {hit.chunk.location}  block={hit.chunk.block}",
                file=sys.stderr,
            )
            print(f"       {hit.chunk.text[:60]!r}", file=sys.stderr)
        print(file=sys.stderr)

    if retrieved.is_empty or (retrieved.scores() and max(retrieved.scores()) < threshold):
        # 走 pipeline 让它产出统一的拒答对象（保持 refused/REFUSAL_TEXT 一致）
        answer = await pipeline.answer(
            question,
            client=_NullClient(),
            model=settings.llm.model,
            top_k=args.top_k,
            min_score=threshold,
        )
    else:
        client = get_llm_client(settings)
        extra: dict[str, Any] = {}
        # 本机网关默认开推理模式，思维链会吃光 max_tokens 导致 content=null
        # （实测见 learn/llm/01_params.md §3.1）。设成 0 可关掉这个补偿。
        if "glm" in settings.llm.model.lower():
            extra["reasoning_effort"] = "none"
        answer = await pipeline.answer(
            question,
            client=client,
            model=settings.llm.model,
            top_k=args.top_k,
            min_score=threshold,
            extra_request=extra,
        )

    if args.json:
        print(
            json.dumps(
                {
                    "question": question,
                    "answer": answer.text,
                    "refused": answer.refused,
                    "citations": [
                        {
                            "index": c.index,
                            "source": c.source,
                            "page": c.page,
                            "label": c.label,
                            "chunk_id": c.chunk_id,
                        }
                        for c in answer.citations
                    ],
                    "invalid_refs": answer.invalid_refs,
                    "top_score": answer.top_score,
                    "context_chars": answer.context_chars,
                    "used_chunks": answer.used_chunks,
                    "retrieved": [
                        {
                            "score": hit.score,
                            "source": hit.chunk.source,
                            "page": hit.chunk.page,
                            "text": hit.chunk.text,
                        }
                        for hit in retrieved.hits
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(answer.text)
    citations = answer.citation_block()
    if citations:
        print("\n引用：")
        print(citations)
    if answer.invalid_refs:
        # **必须显式告警**：模型引用了不存在的编号，说明它在编。
        # 静默丢弃会让"看起来正常但有编造成分"的输出流向用户 —— 那是最危险的。
        print(
            f"\n⚠ 模型引用了不存在的编号 {answer.invalid_refs}，这些引用已忽略。"
            "这次回答可能有编造成分，请核对原文。",
            file=sys.stderr,
        )
    if answer.refused:
        print("\n（拒答：检索无结果或分数低于阈值，未调用模型）", file=sys.stderr)
    return 0


class _NullClient:
    """拒答路径用一个永远不该被调用的 client。

    **为什么不留 `client=None`**：`generate_answer` 在拒答时不会碰 client，
    但如果哪天有人把闸门挪到调用之后，`None.chat` 会抛 `AttributeError` ——
    那个错误信息会让人以为是 client 装配问题，而不是"闸门失效了"。
    用这个替身在误调用时给出明确信息。
    """

    @property
    def chat(self) -> Any:  # noqa: ANN401
        raise AssertionError(
            "拒答路径不应调用模型。若走到这里，说明拒答闸门失效了 —— "
            "检查 generate_answer 里空结果/低分检查的位置。"
        )


def _make_store(kind: str) -> Any:  # noqa: ANN401
    """按名装配 store。chroma 走持久化目录（`data/chroma`）。"""
    if kind == "chroma":
        from app.rag.stores.chroma import ChromaStore

        return ChromaStore()
    from app.rag.stores import InMemoryVectorStore

    return InMemoryVectorStore()


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    args = _build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI 的顶层兜底
        print(f"出错了：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
