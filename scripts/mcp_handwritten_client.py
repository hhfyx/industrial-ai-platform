"""W13 周四 · 手写 MCP client：逐帧打印原始 JSON-RPC。

## 为什么不用 SDK 的 ClientSession 高层封装

任务书原话："打印每一条原始消息（这是理解协议最快的方式）"。
ClientSession 会把 initialize/list/call 包成方法调用，帧被藏起来；
这里用**最底层**的 stdio 读写：自己拼 JSON-RPC 帧、自己按行读响应、
自己打印每一帧的方向与内容。SDK 的 ClientSession 只用来在最后
做一次对照（证明手写帧与 SDK 行为一致）。

## 帧序列（与 docs/mcp-server.md 开篇的抓帧一一对应）

    → initialize      （request，带 protocolVersion/clientInfo/capabilities）
    ← initialize 结果  （server 回 capabilities + serverInfo）
    → notifications/initialized （notification，无 id —— JSON-RPC 2.0 的约定）
    → tools/list
    ← 工具清单（name/description/inputSchema）
    → tools/call hello {name: "MCP"}
    ← tool result（content 数组，type=text）
    → tools/call ping {}      （无参工具）
    → tools/call add {a,b}    （带参工具 + 一次错误参数）
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

SERVER_CMD = [sys.executable, "-X", "utf8", "scripts/mcp_min_server.py"]

_next_id = 0


def _frame(method: str, params: dict | None = None, *, id_: int | None = None) -> dict:
    """拼一帧 JSON-RPC 2.0。id=None 即 notification（协议约定 notification 不带 id）。"""
    f: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        f["params"] = params
    if id_ is not None:
        f["id"] = id_
    return f


def _pp(direction: str, obj: dict) -> None:
    arrow = "→ SEND" if direction == "out" else "← RECV"
    print(f"\n{arrow} " + "-" * 60)
    print(json.dumps(obj, ensure_ascii=False, indent=2))


async def _read_response(
    proc: asyncio.subprocess.Process, want_id: int, timeout: float = 15.0
) -> dict:
    """按行读 server 的 stdout，直到拿到 id 匹配的 response。

    MCP stdio 帧分隔：JSON-RPC over line-delimited JSON（每帧一行，
    \n 结尾）。中间可能混入 notification（无 id）——跳过它们继续读。
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remain = deadline - asyncio.get_event_loop().time()
        if remain <= 0:
            raise TimeoutError(f"等 id={want_id} 的响应超时")
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=remain)
        if not line:
            raise RuntimeError("server stdout 已关闭（进程可能崩了）")
        text = line.decode("utf-8").strip()
        if not text:
            continue
        msg = json.loads(text)
        _pp("in", msg)
        if msg.get("id") == want_id and ("result" in msg or "error" in msg):
            return msg
        # notification 或其它 id 的消息：继续读


async def main() -> None:
    global _next_id
    print(f"启动 server: {' '.join(SERVER_CMD)}")
    proc = await asyncio.create_subprocess_exec(
        *SERVER_CMD,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,  # stderr 独立——不污染协议帧
    )

    # ---- ① initialize（握手请求）
    _next_id += 1
    init_req = _frame(
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {"roots": {"listChanged": True}},
            "clientInfo": {"name": "handwritten-client", "version": "0.1.0"},
        },
        id_=_next_id,
    )
    _pp("out", init_req)
    proc.stdin.write((json.dumps(init_req) + "\n").encode())
    await proc.stdin.drain()
    init_resp = await _read_response(proc, _next_id)
    server_info = init_resp["result"]["serverInfo"]
    print(
        f"\n### server: {server_info['name']} v{server_info.get('version')}"
        f" | 协议 {init_resp['result']['protocolVersion']}"
    )

    # ---- ② notifications/initialized（notification：无 id，无响应）
    initd = _frame("notifications/initialized")
    _pp("out", initd)
    proc.stdin.write((json.dumps(initd) + "\n").encode())
    await proc.stdin.drain()
    print("\n### notification 已发（无 id = 无响应，协议层不回帧）")

    # ---- ③ tools/list
    _next_id += 1
    list_req = _frame("tools/list", {}, id_=_next_id)
    _pp("out", list_req)
    proc.stdin.write((json.dumps(list_req) + "\n").encode())
    await proc.stdin.drain()
    list_resp = await _read_response(proc, _next_id)
    for t in list_resp["result"]["tools"]:
        schema = t.get("inputSchema", {})
        props = list(schema.get("properties", {}).keys())
        print(
            f"\n### tool: {t['name']}  参数: {props or '（无参）'}  — {t.get('description', '')[:50]}"
        )

    # ---- ④ tools/call hello
    _next_id += 1
    call_req = _frame(
        "tools/call",
        {"name": "hello", "arguments": {"name": "MCP"}},
        id_=_next_id,
    )
    _pp("out", call_req)
    proc.stdin.write((json.dumps(call_req) + "\n").encode())
    await proc.stdin.drain()
    call_resp = await _read_response(proc, _next_id)
    text = call_resp["result"]["content"][0]["text"]
    print(f"\n### hello 结果: {text}")

    # ---- ⑤ tools/call ping（无参）
    _next_id += 1
    ping_req = _frame("tools/call", {"name": "ping", "arguments": {}}, id_=_next_id)
    proc.stdin.write((json.dumps(ping_req) + "\n").encode())
    await proc.stdin.drain()
    ping_resp = await _read_response(proc, _next_id)
    print(f"\n### ping 结果: {ping_resp['result']['content'][0]['text']}")

    # ---- ⑥ tools/call add（带参 + 错误参数对照）
    _next_id += 1
    add_req = _frame("tools/call", {"name": "add", "arguments": {"a": 2, "b": 40}}, id_=_next_id)
    proc.stdin.write((json.dumps(add_req) + "\n").encode())
    await proc.stdin.drain()
    add_resp = await _read_response(proc, _next_id)
    print(f"\n### add(2,40) 结果: {add_resp['result']['content'][0]['text']}")

    _next_id += 1
    bad_req = _frame("tools/call", {"name": "add", "arguments": {"a": "x", "b": 1}}, id_=_next_id)
    _pp("out", bad_req)
    proc.stdin.write((json.dumps(bad_req) + "\n").encode())
    await proc.stdin.drain()
    bad_resp = await _read_response(proc, _next_id)
    if "error" in bad_resp:
        print(f"\n### 参数校验 → JSON-RPC error: code={bad_resp['error']['code']}")
    else:
        print(f"\n### 参数校验 → 不是 error 帧，result.isError={bad_resp['result'].get('isError')}")

    proc.stdin.close()
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except TimeoutError:
        proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
