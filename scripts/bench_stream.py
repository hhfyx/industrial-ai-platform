"""bench_stream：流式 vs 非流式的实测对比（TTFT / 总耗时 / 分块节奏）。

**测什么，为什么这样测：**

非流式和流式的**总耗时几乎一样** —— 它们调的是同一个模型、生成同样的 token。
差别只在"第一个字什么时候到"。所以这里同时测两个数：

    TTFT  = 用户看到第一个字的等待时间（体感）
    total = 整个响应用完的时间（成本/吞吐）

如果只测 total，会得出"流式没用"的错误结论；如果只测 TTFT，会忽略
"流式并没有让总时间变短，只是把等待摊平了"。

运行：

    uv run python -X utf8 scripts/bench_stream.py --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.llm.client import get_llm_client  # noqa: E402
from app.core.llm.streaming import StreamStats, stream_chat  # noqa: E402

PROMPT = "用三句话介绍杭州，每句不超过 30 字。"
MAX_TOKENS = 200


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        with suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")


async def _retry(coro_factory, *, attempts: int = 3, label: str = ""):  # noqa: ANN001, ANN003, ANN201
    """重试包装：本机网关会间歇性返回 502（上游偶发），基准测试不该被它打断。

    只重试**瞬时**错误（网络 / 5xx）。如果连 3 次都失败，说明是配置或模型问题，
    直接抛出比继续跑出半张表更有用 —— 半张表会被误当成结论。
    """
    last: BaseException | None = None
    for idx in range(attempts):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001 - 只用于基准脚本，失败就重试并最终抛出
            last = exc
            body = getattr(exc, "body", None)
            print(f"    ! {label} 第 {idx + 1} 次失败：{type(exc).__name__} {body or ''}")
            await asyncio.sleep(1.5 * (idx + 1))
    assert last is not None
    raise last


async def non_stream(client: AsyncOpenAI, model: str, extra: dict[str, Any]) -> dict[str, Any]:
    """非流式：一次性拿到全部内容。TTFT = 总耗时（用户什么都看不到，直到结束）。"""
    started = time.perf_counter()
    resp = await _retry(
        lambda: client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            **extra,
        ),
        label="非流式",
    )
    total_ms = int((time.perf_counter() - started) * 1000)
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    return {
        "kind": "non-stream",
        "ttft_ms": total_ms,  # 真相：非流式的"首个字"就是全部结束那一刻
        "total_ms": total_ms,
        "chars": len(text),
        "chunks": 1,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        },
        "finish_reason": resp.choices[0].finish_reason,
    }


async def streaming(
    client: AsyncOpenAI, model: str, extra: dict[str, Any], *, trace: bool
) -> dict[str, Any]:
    """流式：逐块到达。记录每块的到达时刻，用来看"打字机节奏"。"""
    stats = StreamStats()
    arrival: list[int] = []
    started = time.perf_counter()
    parts: list[str] = []
    async for delta in stream_chat(
        client,
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=MAX_TOKENS,
        extra_request=extra,
        stats=stats,
    ):
        arrival.append(int((time.perf_counter() - started) * 1000))
        parts.append(delta)
        if trace and len(arrival) <= 3:
            print(f"      第 {len(arrival)} 块 @ +{arrival[-1]}ms: {delta!r}")

    gaps = [b - a for a, b in zip(arrival, arrival[1:], strict=False)]
    return {
        "kind": "stream",
        "ttft_ms": stats.ttft_ms,
        "total_ms": stats.total_ms,
        "chars": stats.text_chars,
        "chunks": stats.chunk_count,
        "delta_count": len(arrival),
        "arrival_ms": arrival[:10],  # 只留前 10 个，够看节奏
        "mean_gap_ms": round(statistics.mean(gaps), 1) if gaps else None,
        "usage": stats.usage,
        "finish_reason": stats.finish_reason,
    }


async def main() -> int:
    _force_utf8_stdout()

    parser = argparse.ArgumentParser(description="流式 vs 非流式实测")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", default=str(Path(__file__).with_name("stream_result.json")))
    args = parser.parse_args()

    settings = get_settings()
    llm = settings.llm
    client = get_llm_client(settings)
    extra = {"reasoning_effort": "none"}  # 本机网关特性，见 learn/llm/01_params.md §3.1

    print(f"model={llm.model}  base_url={llm.base_url}  runs={args.runs}")
    print(f"prompt={PROMPT!r}\n")

    non_stream_rows: list[dict[str, Any]] = []
    stream_rows: list[dict[str, Any]] = []

    for i in range(args.runs):
        print(f"--- 第 {i + 1} 轮 ---")
        print("  非流式：")
        ns = await non_stream(client, llm.model, extra)
        print(
            f"    total={ns['total_ms']}ms chars={ns['chars']} tokens={ns['usage']['total_tokens']}"
        )
        non_stream_rows.append(ns)

        print("  流式（前 3 块逐个显示）：")
        sm = await streaming(client, llm.model, extra, trace=True)
        print(
            f"    TTFT={sm['ttft_ms']}ms total={sm['total_ms']}ms "
            f"chunks={sm['chunks']} delta={sm['delta_count']} 平均间隔={sm['mean_gap_ms']}ms"
        )
        stream_rows.append(sm)
        print()

    ns_ttft = [r["ttft_ms"] for r in non_stream_rows]
    sm_ttft = [r["ttft_ms"] for r in stream_rows if r["ttft_ms"]]
    ns_total = [r["total_ms"] for r in non_stream_rows]
    sm_total = [r["total_ms"] for r in stream_rows]

    print("=" * 60)
    print("对比结论")
    print("=" * 60)
    print(f"  非流式 TTFT : {statistics.mean(ns_ttft):>7.0f} ms  (= 总耗时，用户全程白屏)")
    print(f"  流式   TTFT : {statistics.mean(sm_ttft):>7.0f} ms  (首个字到达)")
    if sm_ttft and ns_ttft:
        ratio = statistics.mean(ns_ttft) / statistics.mean(sm_ttft)
        print(
            f"  → 体感提升约 {ratio:.1f}×（TTFT 缩短 {statistics.mean(ns_ttft) - statistics.mean(sm_ttft):.0f}ms）"
        )
    print(f"  非流式 total: {statistics.mean(ns_total):>7.0f} ms")
    print(f"  流式   total: {statistics.mean(sm_total):>7.0f} ms")
    diff = statistics.mean(sm_total) - statistics.mean(ns_total)
    print(f"  → 总耗时差异 {diff:+.0f}ms（流式不缩短总时间，只是把等待摊平）")
    print(
        f"  平均块间隔  : {statistics.mean([r['mean_gap_ms'] for r in stream_rows if r['mean_gap_ms']]):.1f} ms"
    )

    result = {
        "meta": {"model": llm.model, "base_url": llm.base_url, "runs": args.runs, "prompt": PROMPT},
        "non_stream": non_stream_rows,
        "stream": stream_rows,
        "summary": {
            "non_stream_ttft_ms": round(statistics.mean(ns_ttft)),
            "stream_ttft_ms": round(statistics.mean(sm_ttft)) if sm_ttft else None,
            "non_stream_total_ms": round(statistics.mean(ns_total)),
            "stream_total_ms": round(statistics.mean(sm_total)),
        },
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] 明细写入 {Path(args.out).name}")
    await client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
