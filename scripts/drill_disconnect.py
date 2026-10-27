"""drill_disconnect：验证"客户端断开后服务端停止生成"。

**为什么这个演练必须做，而不能只看代码：**

`request.is_disconnected()` 的行为在有/无真实连接时完全不同。
单元测试里用假 Request 只能测"调用到了这个函数"，
测不到"它在真断开时返回 True、以及我们的 break 真的生效"。

这个脚本做一件事：**连上 → 读几块 → 主动掐断 → 观察服务端是否停止生成**。
判断"服务端停了"的证据是服务端日志里的 `chat/stream 客户端断开，停止生成`。

### 一个重要细节：为什么用原始 socket 而不是 httpx

httpx 的 `stream()` 上下文管理器在 `break` 出 `async for` 后会有一套自己的清理，
可能"优雅地"等响应结束 —— 那正好掩盖了我们要测的行为。
**用 asyncio 原始 socket 才真正等于"用户把浏览器关掉"**：直接 RST/FIN，不留任何余地。

运行（服务端需要单独起）：

    # 终端 1
    uv run uvicorn app.main:app --port 8000 --log-level info

    # 终端 2
    uv run python -X utf8 scripts/drill_disconnect.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from contextlib import suppress


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        with suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")


async def drill(host: str, port: int, *, read_chunks: int, hold_ms: int) -> int:
    """连上流、读若干块、然后**粗暴断开**。"""
    print(f"连接 {host}:{port} …")

    reader, writer = await asyncio.open_connection(host, port)
    body = json.dumps(
        {"messages": [{"role": "user", "content": "写一篇 300 字的短文介绍杭州。"}]},
        ensure_ascii=False,
    ).encode("utf-8")

    request = (
        b"POST /api/v1/chat/stream HTTP/1.1\r\n"
        + f"Host: {host}:{port}\r\n".encode()
        + b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n".encode()
        + b"Accept: text/event-stream\r\n"
        + b"Connection: close\r\n\r\n"
    )
    writer.write(request + body)
    await writer.drain()

    # 读响应头，确认流真的建立了
    status_line = await reader.readline()
    headers: list[str] = []
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        headers.append(line.decode("latin-1").strip())

    print(f"  响应：{status_line.decode('latin-1').strip()}")
    for h in headers:
        if h.lower().startswith(("cache-control", "x-accel-buffering", "content-type")):
            print(f"  {h}")

    # 读若干块数据，证明流真的在推
    received = 0
    started = time.perf_counter()
    while received < read_chunks:
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=15)
        except TimeoutError:
            print("  读超时，可能流已结束")
            break
        if not line:
            print("  连接已结束（没读到预期块数）")
            break
        if line.startswith(b"data:"):
            received += 1
            elapsed = int((time.perf_counter() - started) * 1000)
            preview = line.decode("utf-8", "replace").strip()[:70]
            print(f"  第 {received} 块 @ +{elapsed}ms: {preview}")

    print(f"\n  已读 {received} 块，现在**主动断开**（模拟用户关掉页面）")
    if hold_ms > 0:
        # 先停一会儿再断 —— 让服务端有明确的时间窗口观察到"客户端还在"
        await asyncio.sleep(hold_ms / 1000)

    # 关键：直接关 socket。不给优雅结束的机会。
    writer.close()
    with suppress(Exception):
        await writer.wait_closed()
    print("  连接已关闭。")
    print("\n  下一步：看服务端日志。应当出现：")
    print("    chat/stream 客户端断开，停止生成（已发送 N 块，M 字）")
    print("  以及**不应**出现后续的生成活动。")
    return 0


def main() -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="客户端断开演练")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--read-chunks", type=int, default=5, help="读多少块后断开")
    parser.add_argument("--hold-ms", type=int, default=2000, help="断开前再等多久")
    args = parser.parse_args()

    return asyncio.run(
        drill(args.host, args.port, read_chunks=args.read_chunks, hold_ms=args.hold_ms)
    )


if __name__ == "__main__":
    raise SystemExit(main())
