"""W13 周五 · MCP 文件系统 Server 的协议层入口（stdio）。

## 薄包装的"薄"是怎么来的（今天调出来的一个 SDK 事实）

第一版在这里手工拼 `content/structuredContent/isError`——实测发现
**SDK 2.x 的 tool manager 会把工具返回值再包一层**（我的 dict 变成了
content 字符串里的 JSON 字符串，结构化字段全部失踪）。SDK 的正确契约：

- 工具函数**直接返回 data dict** → 框架自动包 content + structuredContent；
- 业务失败**抛 `ToolError`**（`mcp.server.mcpserver.exceptions`）→
  框架转成 `isError=true` 帧，content 是异常消息——周四帧 ⑦ 的结论
  （"业务失败走 isError，让模型读到原因"）在 2.x 的落地方式。

所以本文件真的薄：信封拆包 + 异常翻译，无手工帧构造。

## 运行

    python -X utf8 scripts/mcp_fs_server.py
    # 白名单根 = ./mcp_fs_root（MCP_FS_ROOT 改）；写能力默认禁用，
    # 根目录放 .mcp-writable 标记才启用。
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from app.core.mcp_fs_tools import (
    get_summarize_prompt,
    list_dir,
    read_file,
    read_resource,
    search_files,
    write_file,
)

mcp = MCPServer(
    name="industrial-ai-fs",
    version="0.1.0",
    instructions=(
        "工业 AI 平台的文件系统 MCP Server。只暴露白名单根目录内的文本文件；"
        "路径参数一律是相对白名单根的相对路径；写能力默认禁用"
        "（根目录有 .mcp-writable 标记才启用）。"
    ),
)


def _unwrap(envelope: dict) -> dict:
    """W11 信封 → 2.x 契约：成功返回 data（框架包装）；失败抛 ToolError。"""
    if envelope.get("ok"):
        return envelope["data"]
    raise ToolError(f"[{envelope['error_type']}] {envelope['message']} 建议：{envelope['hint']}")


@mcp.tool()
def fs_list_dir(path: str = ".") -> dict:
    """列出白名单目录的条目（分页）。

    什么时候用：需要知道白名单里有哪些文件时。
    什么时候不用：已经知道确切文件路径时直接读。
    参数 path：相对白名单根的路径，"." 表示根目录。
    返回：{entries: [{path, type, size}], total, truncated}。
    """
    return _unwrap(list_dir(path))


@mcp.tool()
def fs_read_file(path: str, max_bytes: int = 262144) -> dict:
    """读白名单内的文本文件（大小受限、可截断）。

    什么时候用：需要文件内容时（路径从 list 或 search 获得）。
    什么时候不用：路径是目录时改用 list；文件太大时配合 max_bytes 分段。
    参数 path：相对路径；max_bytes：最多读多少字节（默认 256KB）。
    返回：{content, size, truncated}——truncated=true 表示只读了前段。
    """
    return _unwrap(read_file(path, max_bytes))


@mcp.tool()
def fs_search_files(pattern: str) -> dict:
    """按文件名通配符搜索白名单内文件。

    什么时候用：知道文件名模式不知道确切路径时。
    什么时候不用：能直接说出路径时（少一跳）。
    参数 pattern：如 *.md、report*。
    返回：{matches: [相对路径], total, truncated}。
    """
    return _unwrap(search_files(pattern))


@mcp.tool()
def fs_write_file(path: str, content: str) -> dict:
    """写文件（**默认禁用**：根目录无 .mcp-writable 标记时返回 permission 错误）。

    什么时候用：部署方明确启用写能力、且需要落盘结果时。
    什么时候不用：只读场景；内容超过 200k 字符时。
    返回：{bytes_written, created}。
    """
    return _unwrap(write_file(path, content))


# ---------------------------------------------------------------- resources


@mcp.resource("fs://{rel_path}")
def fs_resource(rel_path: str) -> str:
    """把白名单内文件暴露为 fs:// 资源。read 走同一套路径校验（不是绕过安全层的路）。"""
    envelope = read_resource(f"fs://{rel_path}")
    if envelope.get("ok"):
        return envelope["data"]["content"]
    raise ToolError(f"[{envelope['error_type']}] {envelope['message']}")


@mcp.resource("config://fs-allowlist")
def fs_allowlist_info() -> str:
    """静态资源：白名单根与安全策略摘要（client 连上即可发现边界）。"""
    root = Path(os.environ.get("MCP_FS_ROOT", "mcp_fs_root")).resolve()
    writable = (root / ".mcp-writable").exists()
    return (
        f"白名单根：{root}\n"
        f"写能力：{'启用（检测到 .mcp-writable）' if writable else '禁用（只读）'}\n"
        f"扩展名白名单：.md .txt .py .json .yml .yaml .toml .csv .log\n"
        f"单文件读取上限：256 KB；写入上限：200k 字符\n"
        f"绝不提供：shell 执行、网络访问、删除操作"
    )


# ---------------------------------------------------------------- prompts


@mcp.prompt("summarize_file")
def summarize_file_prompt(path: str) -> str:
    """生成"总结某文件"的提示模板（内容经 fs_read_file 同闸读取）。"""
    envelope = read_file(path)
    if envelope.get("ok"):
        content = envelope["data"]["content"]
    else:
        content = f"（读取失败：[{envelope['error_type']}] {envelope['message']}）"
    return get_summarize_prompt(path, content)["messages"][0]["content"]["text"]


if __name__ == "__main__":
    mcp.run(transport="stdio")
