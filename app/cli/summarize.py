"""文档摘要 CLI：`python -m app.cli.summarize <file> [--stream]`。

**CLI 在这一格的作用不是"给用户用"，而是"给联调提供一个不依赖浏览器的入口"。**
前端出问题时，先用 CLI 确认服务端链路正常 —— 这样能立刻二分出
"是后端慢"还是"前端渲染慢"。这个习惯直接对应今天必须回答的那个问题。

三种输出模式：

    --stream      逐字打印最终摘要，**带分段时间进度**（默认）
    --no-stream   一次性输出，用于对比 TTFT
    --json        结构化输出，供脚本消费

进度显示的位置很关键：50 页文档的 map 阶段要跑 30 秒以上，
**如果这期间屏幕一动不动，用户会以为卡死了**。所以进度写在 stderr
（而不是 stdout）：这样 `--stream` 的输出可以被管道消费而不被进度污染。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.llm.client import get_llm_client
from app.rag.loaders import SUPPORTED_EXTENSIONS, load_document
from app.rag.summarize import summarize_document, summarize_streaming


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        with suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")
    # stderr 也要切：进度是中文，GBK 控制台会崩
    reconfigure = getattr(sys.stderr, "reconfigure", None)
    if reconfigure is not None:
        with suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")


def _progress(done: int, total: int, label: str) -> None:
    """进度写 stderr，带时间戳与百分比。

    **必须 flush**：stderr 在重定向时是块缓冲的，
    不 flush 的话进度会在程序结束时一次性涌出 —— 那就失去了"进度"的意义。
    """
    pct = done / total * 100 if total else 0
    bar_len = 20
    filled = int(bar_len * done / total) if total else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stderr.write(f"\r  [{bar}] {done}/{total} ({pct:5.1f}%) {label:<16}")
    sys.stderr.flush()


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    model = settings.llm.model

    # ---- ① 加载（这一步是纯本地，不花钱）----
    t_load = time.perf_counter()
    result = load_document(args.file)
    load_ms = int((time.perf_counter() - t_load) * 1000)

    if args.json:
        pass  # JSON 模式不打印人类文案
    else:
        print(f"文件   : {args.file}")
        print(f"格式   : {Path(args.file).suffix}")
        print(f"页/段  : {result.page_count}（{len(result.documents)} 个文本块）")
        print(f"字符数 : {result.total_chars}")
        print(f"加载耗时: {load_ms} ms")
        for warning in result.warnings:
            print(f"⚠  {warning}", file=sys.stderr)
        print()

    if not result.documents:
        print("文档没有可摘要的文本。", file=sys.stderr)
        return 2

    client = get_llm_client(settings)
    extra: dict[str, Any] = {}
    if "glm" in model.lower():
        extra["reasoning_effort"] = "none"

    try:
        if args.no_stream:
            return await _run_buffered(args, result, client, model, extra)
        return await _run_streaming(args, result, client, model, extra)
    finally:
        await client.close()


async def _run_streaming(
    args: argparse.Namespace,
    result: Any,
    client: Any,
    model: str,
    extra: dict[str, Any],
) -> int:
    """流式模式：进度 + 逐字输出。"""
    started = time.perf_counter()
    first_delta_ms: int | None = None
    delta_count = 0
    meta: dict[str, Any] = {}

    if not args.json:
        sys.stderr.write("\n  分段摘要中…\n")

    async for kind, payload in summarize_streaming(
        result.documents,
        client=client,
        model=model,
        chunk_tokens=args.chunk_tokens,
        extra_request=extra,
    ):
        if kind == "progress":
            if not args.json:
                _progress(
                    int(payload["done"]),  # type: ignore[arg-type]
                    int(payload["total"]),  # type: ignore[arg-type]
                    str(payload["label"]),
                )
            continue

        if kind == "delta":
            if first_delta_ms is None:
                first_delta_ms = int((time.perf_counter() - started) * 1000)
            delta_count += 1
            text = str(payload["text"])
            if args.json:
                # JSON 模式不逐字输出（那样不是合法 JSON），攒着
                continue
            sys.stdout.write(text)
            sys.stdout.flush()  # **必须**：否则终端缓冲会把逐字效果攒成一次性输出
            continue

        if kind == "done":
            meta = payload
            break

    total_ms = int((time.perf_counter() - started) * 1000)

    if args.json:
        out = {
            "file": str(args.file),
            "pages": result.page_count,
            "chars": result.total_chars,
            "summary": meta.get("summary", ""),
            "chunks": meta.get("chunks", 0),
            "api_calls": meta.get("api_calls", 0),
            "ttft_ms": first_delta_ms,
            "total_ms": total_ms,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    sys.stderr.write("\n\n")
    print()
    print("─" * 50)
    print(f"首字延迟(TTFT): {first_delta_ms} ms   ← 从开始到第一个摘要字")
    print(f"总耗时        : {total_ms} ms")
    print(f"分块数        : {meta.get('chunks', 0)}")
    print(f"模型调用次数  : {meta.get('api_calls', 0)}")
    print(f"输出块数      : {delta_count}")
    return 0


async def _run_buffered(
    args: argparse.Namespace,
    result: Any,
    client: Any,
    model: str,
    extra: dict[str, Any],
) -> int:
    """非流式：一次性输出，用于和流式对比 TTFT。"""
    started = time.perf_counter()
    on_progress = None if args.json else (lambda d, t, lbl: _progress(d, t, lbl))
    if not args.json:
        sys.stderr.write("\n  分段摘要中…\n")

    res = await summarize_document(
        result.documents,
        client=client,
        model=model,
        chunk_tokens=args.chunk_tokens,
        extra_request=extra,
        on_progress=on_progress,
    )
    total_ms = int((time.perf_counter() - started) * 1000)

    if args.json:
        print(
            json.dumps(
                {
                    "file": str(args.file),
                    "pages": result.page_count,
                    "chars": result.total_chars,
                    "summary": res.summary,
                    "chunks": res.chunk_count,
                    "api_calls": res.api_calls,
                    "hierarchical": res.hierarchical,
                    "reduce_rounds": res.reduce_rounds,
                    "ttft_ms": total_ms,  # 非流式：首字出现 = 全部完成
                    "total_ms": total_ms,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    sys.stderr.write("\n\n")
    print(res.summary)
    print()
    print("─" * 50)
    print(f"首字延迟(TTFT): {total_ms} ms   ← 非流式：等于总耗时")
    print(f"总耗时        : {total_ms} ms")
    print(f"分块数        : {res.chunk_count}（层级汇总: {res.hierarchical}）")
    print(f"模型调用次数  : {res.api_calls}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli.summarize",
        description="文档摘要：map-reduce 处理超长文档，支持流式输出",
    )
    parser.add_argument(
        "file", type=Path, help=f"输入文件（支持 {'、'.join(SUPPORTED_EXTENSIONS)}）"
    )
    parser.add_argument(
        "--stream", dest="stream", action="store_true", default=True, help="逐字输出（默认）"
    )
    parser.add_argument(
        "--no-stream",
        dest="no_stream",
        action="store_true",
        help="一次性输出，用于对比 TTFT",
    )
    parser.add_argument(
        "--chunk-tokens", type=int, default=3000, help="每段目标 token 数（默认 3000）"
    )
    parser.add_argument("--json", action="store_true", help="输出结构化 JSON（便于脚本消费）")
    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    # --no-stream 与 --stream 互斥，后者是默认值
    if args.no_stream:
        args.stream = False

    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI 的顶层边界，给出人能看懂的提示
        print(f"\n错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
