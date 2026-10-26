"""裸 SSE 最小实现：只用 asyncio + 原始 socket，**不套 FastAPI**。

这一格的目的不是"再写一个 demo"，而是把 SSE 从框架里拆出来看清三件事：

    ① SSE 就是"HTTP 响应体不能结束，一直往里写" —— 它不是协议，是约定
    ② 所谓 flush，在裸 socket 上就是 `writer.write()` + `await writer.drain()`
    ③ `data: ...\\n\\n` 的**双换行**是唯一的分帧手段，少一个换行浏览器就不认

**为什么不套 FastAPI**：`StreamingResponse` 会替你做分块编码、缓冲、disconnect 检测，
你写对了不知道为什么对，写错了也看不到字节。裸 socket 上你能直接 `curl` 看到
每一行的到达时刻 —— 那才是"理解 yield + flush 时机"的意思。

运行：

    uv run python -X utf8 scripts/raw_sse.py --delay 1.0 --port 8765
    curl.exe -N http://127.0.0.1:8765/events      # 另开一个终端看逐条到达

观察点（`--delay 1.0` 时）：

- 每条 `data:` 之间**间隔 1 秒** —— 证明服务端每条都 flush 了，没有被攒着
- `curl -N` 的 `-N` 是**必须**的：不加的话 curl 自带缓冲，数字会一次性全出来，
  你会误以为服务端没 flush
- 事件之间是空行（双换行）。把 `\\n\\n` 改成 `\\n`，浏览器 EventSource 会一直不触发
- 结束时发 `event: end` 再关连接；SSE 没有 `[DONE]` 这种标准终止符 ——
  `[DONE]` 是 OpenAI 自己的约定，不是 SSE 规范的一部分

**刻意不做的**：`Transfer-Encoding: chunked` 的手写分块。
用 `Connection: close` + 无 Content-Length，HTTP/1.1 下靠关闭连接做结束，
是最简单合法的写法；分块编码由服务器自动处理更省事（FastAPI 就是这么干的）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from contextlib import suppress

# 一个最小的合法 HTTP/1.1 响应头。
# 关键三行：
#   Content-Type: text/event-stream    ← 浏览器凭这个决定走 EventSource
#   Cache-Control: no-cache            ← 中间代理不许缓存整个响应
#   Connection: close                  ← 结束靠关连接（我们没有 Content-Length）
HEADERS = (
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/event-stream; charset=utf-8\r\n"
    "Cache-Control: no-cache\r\n"
    "Connection: close\r\n"
    # 关掉 nginx 一类反代的缓冲，否则它会攒够一个 buffer 才吐
    "X-Accel-Buffering: no\r\n"
    "\r\n"
)


def sse_frame(data: str, *, event: str | None = None, event_id: str | None = None) -> bytes:
    """拼一个 SSE 帧。**双换行结尾是硬性要求。**

    SSE 的分帧规则（规范原文的意思）：

    - 每一行是 `字段: 值`
    - **一个空行 = 一个事件结束**
    - 所以帧必然是 `...\\n\\n`。少一个 `\\n`，客户端会把下一条也当成同一个事件的一部分，
      或者干脆不派发 —— 表现为"浏览器完全不触发 onmessage"，而服务端日志一切正常。
      这是 SSE 最难查的错，因为**没有任何报错**。
    """
    parts: list[str] = []
    if event_id is not None:
        # 断线重连时浏览器会带 `Last-Event-ID` 头，用它续传
        parts.append(f"id: {event_id}")
    if event is not None:
        # 不带 event 字段时是默认事件（onmessage）；带了就是命名事件（addEventListener）
        parts.append(f"event: {event}")
    # data 里如果有换行，**每个换行都要单独一行 data:**，否则会被当成新字段
    for line in data.split("\n"):
        parts.append(f"data: {line}")
    return ("\n".join(parts) + "\n\n").encode("utf-8")


async def handle(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    delay: float,
    ping_interval: float,
) -> None:
    """处理一个连接：读完请求行/头，然后持续写 SSE 事件。"""
    peer = writer.get_extra_info("peername")
    try:
        # 必须把请求头读完，否则客户端的 curl 会一直等着发完请求
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
    except (ConnectionResetError, asyncio.IncompleteReadError):
        return

    print(f"[conn] {peer} 已连上")
    writer.write(HEADERS.encode())
    await writer.drain()  # ← 第一个 flush：头必须马上发出去，否则浏览器不认为连接建立

    try:
        for idx in range(1, 6):
            payload = {
                "seq": idx,
                "ts": round(time.time(), 3),
                "message": f"第 {idx} 条事件",
            }
            frame = sse_frame(json.dumps(payload, ensure_ascii=False), event_id=str(idx))
            writer.write(frame)
            # ← 每一次 drain 就是一次 flush。**不 drain 的话数据留在缓冲区里，
            #   客户端看到的是"一次性全到"，而不是逐条到达。**
            await writer.drain()
            print(f"  -> 已发送 seq={idx}")

            # 心跳：长时间没有内容时，代理/负载均衡会认为连接空闲而掐断。
            # 注释行（以 `:` 开头）是合法的 SSE 帧，客户端会忽略但能保活。
            if delay > ping_interval:
                await asyncio.sleep(ping_interval)
                writer.write(b": ping\n\n")
                await writer.drain()
                print("  -> 已发送 ping 心跳")

            await asyncio.sleep(max(delay - (ping_interval if delay > ping_interval else 0), 0))

        # 结束事件。SSE 规范没有"终止符"，靠关连接或自定义事件表达
        writer.write(sse_frame("bye", event="end"))
        await writer.drain()
        print(f"[conn] {peer} 正常结束")
    except (ConnectionResetError, BrokenPipeError):
        # 用户在流中途断开（点"停止生成"）—— 这是**正常**路径，不是错误
        print(f"[conn] {peer} 客户端提前断开（这是正常路径）")
    finally:
        with suppress(Exception):
            writer.close()
            await writer.wait_closed()


async def serve(port: int, delay: float, ping_interval: float) -> None:
    server = await asyncio.start_server(
        lambda r, w: handle(r, w, delay=delay, ping_interval=ping_interval),
        host="127.0.0.1",
        port=port,
    )
    print(f"裸 SSE 服务已启动：http://127.0.0.1:{port}/events")
    print(f"  每条间隔 {delay}s，心跳间隔 {ping_interval}s")
    print(f"  另开终端：curl.exe -N http://127.0.0.1:{port}/events")
    print("  对照实验：把 -N 去掉，看输出是否变成一次性全出（那是 curl 在缓冲）")
    async with server:
        await server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="裸 SSE 最小实现（asyncio + socket）")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--delay", type=float, default=1.0, help="每条事件间隔秒数")
    parser.add_argument("--ping", type=float, default=0.5, help="心跳间隔秒数")
    args = parser.parse_args()

    try:
        asyncio.run(serve(args.port, args.delay, args.ping))
    except KeyboardInterrupt:
        print("\n[exit] 已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
