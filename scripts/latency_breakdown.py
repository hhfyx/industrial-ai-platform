"""latency_breakdown：**首 token 慢，是模型慢还是我的链路慢？**

这是今天必须回答的问题。做法是**在每一层打点**，把总时间拆成可归因的几段。

## 打点位置（每一段都能单独归因）

```
    ① load      pypdf 读文件     → 纯本地 CPU，与网络无关
    ② chunk     切分              → 纯本地，微秒级
    ③ map_call  分段摘要（N 次）  → 每次都是一次完整 LLM 往返
    ④ reduce_ttft  汇总的首字节   → **这才是用户感知的"首字"**
    ⑤ stream     汇总的剩余 token → 生成速度（tokens/s）
```

**关键洞察：用户说的"首 token 慢"，实际可能是 ③ 太多次。**
50 页文档按 500 token 切是 9 段 → 9 次串行 map 调用 ≈ 30 秒，
那期间用户一个字节都看不到。**这不叫"模型慢"，这叫"我的链路慢"** ——
模型每次可能只花 2-3 秒，是我们串行调了 9 次。

## 对比方法（任务提示的"SSH 到 API 的首字节时间"）

对一个**单次**请求做两个测量：
- `raw_ttft`：直接打上游 `/chat/completions`，测到第一个 SSE 字节
- `end_to_end_ttft`：走我们自己的服务（含 FastAPI、中间件、SDK、重试层）

两者之差 = **我方链路引入的额外延迟**。这个差值就是优化的靶子：
如果是 50ms，链路没问题，该优化模型/prompt；如果是 2s，那是我方的问题。

运行：

    uv run python -X utf8 scripts/latency_breakdown.py
    uv run python -X utf8 scripts/latency_breakdown.py --file tests/fixtures/sample50.pdf
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.llm.client import get_llm_client  # noqa: E402
from app.core.llm.streaming import StreamStats, stream_chat  # noqa: E402
from app.rag.loaders import load_document  # noqa: E402
from app.rag.summarize import segment_documents, summarize_document  # noqa: E402

SIMPLE_PROMPT = "用两句话介绍杭州。"


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        with suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")


def pct(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (pos - low))


async def measure_upstream_raw_ttft(runs: int) -> dict[str, Any]:
    """**直接打上游**，绕过我们的服务：这是"模型侧"的基准。

    用裸 httpx 而不是 SDK，连 SDK 的解析开销都排除掉 ——
    这样测出来的才是"网络 + 模型"的纯延迟。
    """
    settings = get_settings()
    llm = settings.llm
    url = f"{llm.base_url}/chat/completions"
    payload = {
        "model": llm.model,
        "messages": [{"role": "user", "content": SIMPLE_PROMPT}],
        "stream": True,
        "max_tokens": 100,
        "temperature": 0.0,
        "reasoning_effort": "none",
    }

    ttfts: list[int] = []
    totals: list[int] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        for _ in range(runs):
            started = time.perf_counter()
            first: int | None = None
            async with client.stream(
                "POST",
                url,
                headers={"Authorization": f"Bearer {llm.api_key.get_secret_value()}"},
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    # 只认第一个**带 content 的** data 帧（与我们的 TTFT 口径一致）
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw.strip() == "[DONE]":
                        break
                    try:
                        obj = json.loads(raw)
                        choices = obj.get("choices") or []
                        if (
                            first is None
                            and choices
                            and (choices[0].get("delta") or {}).get("content")
                        ):
                            first = int((time.perf_counter() - started) * 1000)
                    except json.JSONDecodeError:
                        continue
            totals.append(int((time.perf_counter() - started) * 1000))
            if first is not None:
                ttfts.append(first)

    return {
        "raw_ttft_ms": ttfts,
        "raw_total_ms": totals,
        "raw_ttft_p50": pct(ttfts, 0.5) if ttfts else None,
    }


async def measure_sdk_ttft(runs: int) -> dict[str, Any]:
    """走我们的 SDK 封装（含重试层、StreamStats），但与服务器无关。

    与 `raw` 的差 = **SDK + 我们封装层**的开销。
    """
    settings = get_settings()
    llm = settings.llm
    client = get_llm_client(settings)
    ttfts: list[int] = []
    totals: list[int] = []
    try:
        for _ in range(runs):
            stats = StreamStats()
            async for _delta in stream_chat(
                client,
                model=llm.model,
                messages=[{"role": "user", "content": SIMPLE_PROMPT}],
                max_tokens=100,
                extra_request={"reasoning_effort": "none"},
                stats=stats,
            ):
                pass
            if stats.ttft_ms is not None:
                ttfts.append(stats.ttft_ms)
            totals.append(stats.total_ms)
    finally:
        await client.close()

    return {
        "sdk_ttft_ms": ttfts,
        "sdk_total_ms": totals,
        "sdk_ttft_p50": pct(ttfts, 0.5) if ttfts else None,
    }


async def measure_end_to_end_ttft(runs: int, port: int) -> dict[str, Any] | None:
    """走完整的 HTTP 服务（FastAPI + 中间件 + 我们的 SSE 端点）。

    与 `sdk` 的差 = **FastAPI 层 + 网络栈 + SSE 编码**的开销。
    服务没起来就返回 None（不阻塞整份报告）。
    """
    url = f"http://127.0.0.1:{port}/api/v1/chat/stream"
    ttfts: list[int] = []
    totals: list[int] = []

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=5.0)) as client:
            # 先探活
            probe = await client.get(f"http://127.0.0.1:{port}/health")
            if probe.status_code != 200:
                return None

            for _ in range(runs):
                started = time.perf_counter()
                first: int | None = None
                async with client.stream(
                    "POST",
                    url,
                    json={"messages": [{"role": "user", "content": SIMPLE_PROMPT}]},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if first is None and line.startswith("data: ") and '"delta"' in line:
                            first = int((time.perf_counter() - started) * 1000)
                            break
                totals.append(int((time.perf_counter() - started) * 1000))
                if first is not None:
                    ttfts.append(first)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return None

    return {
        "e2e_ttft_ms": ttfts,
        "e2e_total_ms": totals,
        "e2e_ttft_p50": pct(ttfts, 0.5) if ttfts else None,
    }


async def measure_document_pipeline(path: Path) -> dict[str, Any]:
    """长文档全链路：**看清"首字慢"到底被哪一段吃掉了**。"""
    settings = get_settings()
    llm = settings.llm

    t0 = time.perf_counter()
    result = load_document(path)
    load_ms = int((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    chunks = segment_documents(result.documents, chunk_tokens=500)
    chunk_ms = int((time.perf_counter() - t1) * 1000)

    client = get_llm_client(settings)
    map_times: list[int] = []

    def on_progress(done: int, total: int, label: str) -> None:  # noqa: ARG001
        map_times.append(int((time.perf_counter() - t2) * 1000))

    t2 = time.perf_counter()
    res = await summarize_document(
        result.documents,
        client=client,
        model=llm.model,
        chunk_tokens=500,
        extra_request={"reasoning_effort": "none"},
        on_progress=on_progress,
    )
    pipeline_ms = int((time.perf_counter() - t2) * 1000)
    await client.close()

    return {
        "file": path.name,
        "pages": result.page_count,
        "chars": result.total_chars,
        "chunks": len(chunks),
        "load_ms": load_ms,
        "chunk_ms": chunk_ms,
        "pipeline_ms": pipeline_ms,
        "api_calls": res.api_calls,
        "avg_per_call_ms": round(pipeline_ms / max(res.api_calls, 1)),
        "map_progress_ms": map_times,
        # 关键派生指标
        "first_visible_content_ms": pipeline_ms,  # 非流式 map-reduce：全部完成后才出字
        "per_call_share": round(pipeline_ms / max(res.api_calls, 1) / max(pipeline_ms, 1), 3),
    }


async def main() -> int:
    _force_utf8_stdout()

    parser = argparse.ArgumentParser(description="首 token 延迟归因")
    parser.add_argument("--runs", type=int, default=3, help="每个基准重复次数")
    parser.add_argument("--port", type=int, default=8000, help="本地服务端口（测端到端）")
    parser.add_argument("--file", default=str(PROJECT_ROOT / "tests/fixtures/sample50.pdf"))
    parser.add_argument(
        "--skip-doc", action="store_true", help="跳过长文档流水线（它要跑 30 秒以上）"
    )
    args = parser.parse_args()

    print("=" * 74)
    print("首 token 延迟归因：模型慢 还是 链路慢？")
    print("=" * 74)
    print(f"prompt: {SIMPLE_PROMPT!r}   runs={args.runs}\n")

    print("① 上游直连（裸 httpx，绕过一切我方代码）…")
    raw = await measure_upstream_raw_ttft(args.runs)
    for i, v in enumerate(raw["raw_ttft_ms"], 1):
        print(f"     第 {i} 次: TTFT {v} ms   total {raw['raw_total_ms'][i - 1]} ms")
    print(f"     p50 TTFT = {raw['raw_ttft_p50']} ms  ← **这是模型+网络的真实下限**\n")

    print("② 经 SDK 封装（+ 重试层 + StreamStats）…")
    sdk = await measure_sdk_ttft(args.runs)
    for i, v in enumerate(sdk["sdk_ttft_ms"], 1):
        print(f"     第 {i} 次: TTFT {v} ms   total {sdk['sdk_total_ms'][i - 1]} ms")
    print(f"     p50 TTFT = {sdk['sdk_ttft_p50']} ms")
    if raw["raw_ttft_p50"] and sdk["sdk_ttft_p50"]:
        delta = sdk["sdk_ttft_p50"] - raw["raw_ttft_p50"]
        print(f"     → SDK 层开销 = {delta:+d} ms\n")

    print(f"③ 端到端（FastAPI 服务 :{args.port}）…")
    e2e = await measure_end_to_end_ttft(args.runs, args.port)
    if e2e is None:
        print("     (服务未启动，跳过。启动：uvicorn app.main:app --port 8000)\n")
    else:
        for i, v in enumerate(e2e["e2e_ttft_ms"], 1):
            print(f"     第 {i} 次: TTFT {v} ms")
        print(f"     p50 TTFT = {e2e['e2e_ttft_p50']} ms")
        if sdk["sdk_ttft_p50"] and e2e["e2e_ttft_p50"]:
            delta = e2e["e2e_ttft_p50"] - sdk["sdk_ttft_p50"]
            print(f"     → FastAPI/HTTP 层开销 = {delta:+d} ms\n")

    doc: dict[str, Any] | None = None
    if not args.skip_doc:
        path = Path(args.file)
        if path.exists():
            print(f"④ 长文档流水线（{path.name}，chunk=500 tokens）…")
            doc = await measure_document_pipeline(path)
            print(f"     加载      : {doc['load_ms']:>6} ms")
            print(f"     切分      : {doc['chunk_ms']:>6} ms")
            print(f"     分段数    : {doc['chunks']:>6}")
            print(f"     模型调用  : {doc['api_calls']:>6} 次")
            print(f"     单次平均  : {doc['avg_per_call_ms']:>6} ms")
            print(f"     流水线总  : {doc['pipeline_ms']:>6} ms")
            print(
                f"     **用户首次看到字** : {doc['first_visible_content_ms']} ms（非流式：要等全部完成）"
            )
            print()
        else:
            print(f"④ 文件不存在，跳过：{path}\n")

    # ---- 结论 ----
    print("=" * 74)
    print("结论")
    print("=" * 74)
    if raw["raw_ttft_p50"]:
        print(f"  **模型+网络的下限**       : {raw['raw_ttft_p50']} ms（这个无法绕过）")
    if sdk["sdk_ttft_p50"] and raw["raw_ttft_p50"]:
        overhead = sdk["sdk_ttft_p50"] - raw["raw_ttft_p50"]
        verdict = "可忽略" if abs(overhead) < 150 else ("需要优化" if overhead > 0 else "反而更快")
        print(f"  **我方封装层开销**         : {overhead:+d} ms → {verdict}")
    if e2e and sdk["sdk_ttft_p50"] and e2e["e2e_ttft_p50"]:
        overhead = e2e["e2e_ttft_p50"] - sdk["sdk_ttft_p50"]
        verdict = "可忽略" if abs(overhead) < 150 else ("需要优化" if overhead > 0 else "反而更快")
        print(f"  **FastAPI/HTTP 层开销**    : {overhead:+d} ms → {verdict}")

    if doc:
        n = doc["api_calls"]
        per = doc["avg_per_call_ms"]
        print()
        print("  **长文档「首字慢」的真正原因**：")
        print(f"     不是单次调用慢（平均 {per} ms/次），而是**串行调了 {n} 次**。")
        print(
            f"     用户要等 {doc['pipeline_ms']} ms 才看到第一个字 —— 那是 {n} × {per} ms 的累加。"
        )
        print()
        print("  → 优化方向（按收益排序）：")
        print("     1. **把 map 改成并发**（本项目的 summarize_streaming 已实现，限流 4 路）")
        print("        理论收益：9 段从 30s 降到约 10s（4 路并发）")
        print("     2. **降低段数**：调大 chunk_tokens（段少 = 调用少 = 首字快，但单段更贵）")
        print("     3. **流式输出每一段**：边 map 边把该段摘要推给前端，")
        print("        用户能看到「正在总结第 3/9 段」，体感完全不同")
        print("     4. 最后才考虑换更快的模型 —— 那是收益最低的一项")

    out = {
        "meta": {"prompt": SIMPLE_PROMPT, "runs": args.runs, "model": get_settings().llm.model},
        "upstream_raw": raw,
        "through_sdk": sdk,
        "end_to_end": e2e,
        "document_pipeline": doc,
    }
    out_path = Path(__file__).with_name("latency_result.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] 明细写入 {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
