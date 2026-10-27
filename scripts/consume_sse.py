"""consume_sse：httpx 侧的 SSE 消费端，逐行解析并逐字打印。

**这个脚本的价值是验证"服务端写对了"之外的下一环：客户端读得对不对。**

服务端跑通不代表客户端能用 —— 常见的客户端侧问题：
- 把整个响应 `await resp.aread()` 读完才处理 → 完全失去流式意义（而且永远等不到结束）
- 用 `resp.text` 而不是 `resp.aiter_lines()` → 同上
- 忘记 `-N` 那类缓冲（httpx 这边要用 `client.stream(...)` 的上下文管理器）
- 按 `\\n\\n` 自己切分而不是按行读 → 遇到多行 data 就错

正确姿势就是任务里写的那行：

    async with client.stream("POST", url, json=...) as resp:
        async for line in resp.aiter_lines():
            ...

`aiter_lines()` 会在数据到达时逐行产出，**不会**等整个响应体结束。

三种运行模式：

    # 1) 打本地服务的 /api/v1/chat/stream
    uv run python -X utf8 scripts/consume_sse.py

    # 2) 打任意 SSE 端点（如 OpenAI 兼容网关）
    uv run python -X utf8 scripts/consume_sse.py --url http://127.0.0.1:8326/v1/chat/completions --raw-openai

    # 3) 只验证协议解析（不发请求，用内置样本）
    uv run python -X utf8 scripts/consume_sse.py --selftest
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

LOCAL_URL = "http://127.0.0.1:8000/api/v1/chat/stream"


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        with suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")


def parse_sse_line(line: str) -> tuple[str, str] | None:
    """解析一行 SSE，返回 `(字段, 值)`。

    规则很简单，但**细节都在边角上**：

    - 空行 = 事件结束（返回 None，由调用方决定"这一帧处理完了"）
    - `:` 开头 = 注释（心跳），**没有字段值**，直接返回 ("comment", "")
    - `field: value` —— 冒号后**一个空格**是规范写法，但多余空格也应容忍
    - `field:value`（无空格）也合法

    返回 None 表示"这是一个空行（帧边界）"。
    """
    if line == "":
        return None
    if line.startswith(":"):
        return ("comment", line[1:].strip())
    if ":" not in line:
        # 没有冒号的行按规范是"字段名 + 空值"，实际中很少见，容忍成 data
        return ("data", line)
    field, _, value = line.partition(":")
    # 规范说"去掉前导一个空格"，这里用 lstrip 更宽容，避免 "data:  x" 丢空格
    return (field.strip(), value[1:] if value.startswith(" ") else value)


def handle_event(fields: dict[str, str]) -> dict[str, Any] | None:
    """把累积到的一个事件处理成结构化结果。

    一个事件可能有多行 `data:`，按规范要**用换行拼接**（不是直接相加）。
    """
    if not fields.get("data"):
        return None
    raw = "\n".join(fields["data"].split("\x00"))  # 占位：单行时原样返回
    return {"raw": raw, "event": fields.get("event")}


async def consume(url: str, payload: dict[str, Any], *, raw_openai: bool = False) -> int:
    """核心：用 `client.stream` + `aiter_lines` 逐行消费。

    返回 0 表示正常（收到 [DONE]），1 表示异常。
    """
    started = time.perf_counter()
    first_token_ms: int | None = None
    collected: list[str] = []
    got_done = False
    ping_count = 0
    chunk_count = 0

    print(f"POST {url}")
    print(f"  请求体：{json.dumps(payload, ensure_ascii=False)[:120]}...")
    print("-" * 60)

    # 关键：`client.stream(...)` 是**上下文管理器**，不能 `await client.post(...)`。
    # 后者会把整个响应体读完才返回，流式就废了。
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        try:
            async with client.stream("POST", url, json=payload) as response:
                print(
                    f"  HTTP {response.status_code}  "
                    f"content-type={response.headers.get('content-type')}"
                )
                # 这两行的存在性本身就是要验证的（缺了会被 Nginx 缓冲）
                for header in ("cache-control", "x-accel-buffering"):
                    print(f"  {header}: {response.headers.get(header, '<缺失>')}")
                print("-" * 60)

                if response.status_code != 200:
                    body = await response.aread()
                    print(f"  非 200，正文：{body.decode('utf-8', 'replace')[:300]}")
                    return 1

                current: dict[str, str] = {}
                async for line in response.aiter_lines():
                    parsed = parse_sse_line(line)
                    if parsed is None:
                        # 空行 = 一个事件结束，处理累积的数据
                        if current.get("data"):
                            chunk_count += 1
                            text = _extract_text(current["data"], raw_openai=raw_openai)
                            if text and first_token_ms is None:
                                first_token_ms = int((time.perf_counter() - started) * 1000)
                            if text:
                                collected.append(text)
                                # end="" + flush：真正的"逐字打印"，否则 stdout 缓冲会攒着
                                sys.stdout.write(text)
                                sys.stdout.flush()
                        current = {}
                        continue

                    field, value = parsed
                    if field == "comment":
                        ping_count += 1
                        continue
                    if field == "data":
                        if value.strip() == "[DONE]":
                            got_done = True
                            break
                        # 多行 data 用 \n 拼接
                        current["data"] = (
                            current["data"] + "\n" + value if "data" in current else value
                        )
                    else:
                        current[field] = value

        except httpx.RemoteProtocolError as exc:
            print(f"\n  连接中断：{type(exc).__name__}: {exc}")
            print("  （对 SSE 而言，服务端正常关流也可能触发，不一定是错误）")

    total_ms = int((time.perf_counter() - started) * 1000)
    print("\n" + "-" * 60)
    print(f"  首块延迟(TTFT): {first_token_ms} ms")
    print(f"  总耗时        : {total_ms} ms")
    print(f"  数据块数      : {chunk_count}")
    print(f"  心跳帧数      : {ping_count}")
    print(f"  收到 [DONE]   : {got_done}")
    print(f"  拼接后文本    : {''.join(collected)[:160]!r}")

    # 判定：有内容 + 收到 [DONE] 算成功。
    # **不强制要求 [DONE]**：有些端点用关连接结束，那也是合法 SSE。
    return 0 if collected else 1


def _extract_text(raw: str, *, raw_openai: bool) -> str:
    """从一帧 data 里取出要打印的文本。

    两种格式：
    - 本服务的 `/chat/stream`：`{"delta": "..."}`
    - OpenAI 兼容：`{"choices":[{"delta":{"content":"..."}}]}`
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not isinstance(obj, dict):
        return ""

    if "delta" in obj and isinstance(obj["delta"], str):
        return obj["delta"]

    if raw_openai or "choices" in obj:
        choices = obj.get("choices") or []
        if not choices:
            return ""
        delta = (choices[0] or {}).get("delta") or {}
        return delta.get("content") or ""

    return ""


def selftest() -> int:
    """不发网络的解析自检：验证 `parse_sse_line` 的边角。"""
    cases = [
        ("data: hello", ("data", "hello")),
        ("data:hello", ("data", "hello")),  # 无空格也合法
        ("data:  two-spaces", ("data", " two-spaces")),  # 只吃掉一个前导空格
        (": ping", ("comment", "ping")),
        ("event: delta", ("event", "delta")),
        ("id: 7", ("id", "7")),
        ("", None),  # 空行 = 帧边界
    ]
    ok = True
    for line, expected in cases:
        got = parse_sse_line(line)
        status = "OK " if got == expected else "FAIL"
        if got != expected:
            ok = False
        print(f"  [{status}] {line!r:24} -> {got!r}")
    print(f"\n  解析自检：{'全部通过' if ok else '有失败'}")
    return 0 if ok else 1


async def main() -> int:
    _force_utf8_stdout()

    parser = argparse.ArgumentParser(description="SSE 消费端（httpx 侧）")
    parser.add_argument("--url", default=LOCAL_URL, help=f"默认 {LOCAL_URL}")
    parser.add_argument(
        "--message", default="用三句话介绍杭州，每句不超过 30 字。", help="发给服务端的用户消息"
    )
    parser.add_argument(
        "--raw-openai", action="store_true", help="按 OpenAI 兼容格式解析 choices[].delta.content"
    )
    parser.add_argument("--selftest", action="store_true", help="只跑协议解析自检，不发请求")
    parser.add_argument("--api-key", default="", help="raw-openai 模式的 Bearer token")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    if args.raw_openai:
        payload: dict[str, Any] = {
            "model": args.model if hasattr(args, "model") else "buddy/glm-5.3",
            "messages": [{"role": "user", "content": args.message}],
            "stream": True,
            "max_tokens": 200,
            "stream_options": {"include_usage": True},
        }
        return await consume(args.url, payload, raw_openai=True)

    payload = {"messages": [{"role": "user", "content": args.message}]}
    return await consume(args.url, payload)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
