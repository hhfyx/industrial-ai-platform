"""W13 周四 · 最小 MCP Server：把"任务书最小可用"与"周五要暴露的真工具"放在同一文件。

## 运行方式（stdio）

    python -X utf8 scripts/mcp_min_server.py
    # 进程不打印任何东西就是对的：stdio 传输下 stdout 是协议通道，
    # 手写日志会污染 JSON-RPC 帧（排错时第一个要排除的低级坑）。

## 与任务书的偏差（如实记录）

任务书写 `from mcp.server.fastmcp import FastMCP`——那是 **SDK 1.x** 的
入口。本仓库装的是 **mcp 2.3.0**：FastMCP 更名 `MCPServer`
（`mcp.server.mcpserver`），装饰器与运行方式不变。这个偏差本身就是
今天"概念澄清"价值的实证：**协议是稳定的（JSON-RPC 2.0 的
initialize/tools/list/tools/call），SDK 的 Python 包装层在快速演进**——
抓帧笔记（docs/mcp-server.md）记录的是协议帧，不是某个 SDK 版本的
调用姿势，所以换了 SDK 大版本笔记依然成立。

## 三个工具的刻意设计

- `hello`：任务书指定的最小工具，验证"能起、能 list、能 call"；
- `ping`：无参工具——验证 schema 里 `inputSchema.properties` 为空时
  帧长什么样（周 5 暴露 query_metrics 时会用到无参工具）；
- `add`：两个带类型参数的工具——验证参数校验（传字符串给 int 参数
  时协议层返回的是 JSON-RPC error 还是空结果，这是周五工具契约的
  前置知识）。

真正的平台工具（query_metrics / retrieve_spec / generate_report）周五接，
不放今天：今天的验收是"协议理解"，不是"工具数量"。
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

mcp = MCPServer(
    name="industrial-ai-mcp",
    version="0.1.0",
    instructions=(
        "工业 AI 平台的 MCP 最小 Server（W13 周四）。"
        "当前只有三个演示工具；周五将暴露 query_metrics / retrieve_spec / generate_report。"
    ),
)


@mcp.tool()
def hello(name: str) -> str:
    """向指定的人打招呼。参数 name 是要问候的名字。"""
    return f"你好，{name}！来自 industrial-ai-mcp 的问候。"


@mcp.tool()
def ping() -> str:
    """存活探针：无参数，返回固定字符串。用于验证空参数工具的协议帧。"""
    return "pong"


@mcp.tool()
def add(a: int, b: int) -> int:
    """两数相加。用于验证带类型参数的校验行为。"""
    return a + b


if __name__ == "__main__":
    mcp.run(transport="stdio")
