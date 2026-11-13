"""W13 周五 · MCP 文件系统 Server：4 工具 + 资源 + 提示模板。

## 与周四最小 Server 的关系

周四解决"协议帧长什么样"（hello/ping/add 是协议探针）；
今天是**第一个真实能力域**：文件系统——也是"同一批能力以 MCP 形态
发布"的第一步（W9 的 Agent 工具走 FC 形态，这里走 MCP 形态）。

## 工具行为层（本文件）与协议层（mcp_fs_server.py）分离

工具实现是纯函数（`{ok, data}` / `{ok=False, error_type, message, hint}`
信封，W11 错误结构），不 import mcp——测试不依赖 SDK、错误结构可以
与 W9/FC 形态逐字段对齐。`scripts/mcp_fs_server.py` 只做薄包装：
把信封塞进 MCP 的 tool result。

## 资源与提示

- resources：把白名单内的文件暴露为 `fs://<相对路径>`（只列白名单内，
  read 走同一套 check_path——资源不是第二条绕过安全层的路）；
- prompts：`summarize_file` 模板（参数化文件路径），生成"总结某文件"
  的固定套路 prompt。
"""

from __future__ import annotations

from typing import Any, Final

from app.core.mcp_fs_security import (
    MAX_FILE_BYTES,
    PathDenied,
    _allowed_root,
    check_path,
    check_size_for_read,
    check_size_for_write,
)

LIST_PAGE_SIZE: Final[int] = 50  # 资源/目录列表分页：防大目录把响应撑爆


def _envelope_ok(data: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "data": data}


def _envelope_err(error_type: str, message: str, hint: str) -> dict[str, Any]:
    return {"ok": False, "error_type": error_type, "message": message, "hint": hint}


# ============================================================ 工具 ①：list_dir


def list_dir(path: str = ".") -> dict[str, Any]:
    """列目录（白名单内）。返回分页的条目清单（子目录与允许的文件）。"""
    try:
        resolved = check_path(path if path != "." else ".")
    except PathDenied as e:
        return e.envelope()

    if not resolved.exists():
        return _envelope_err("not_found", f"目录不存在：{path}", "先用 list_dir 列根目录看看有什么")
    if not resolved.is_dir():
        return _envelope_err("invalid_arg", f"{path} 是文件不是目录", "read_file 用于读文件")

    from app.core.mcp_fs_security import ALLOWED_EXTENSIONS

    entries: list[dict[str, Any]] = []
    try:
        for child in sorted(resolved.iterdir()):
            rel = child.relative_to(_allowed_root()).as_posix()
            if child.is_dir():
                entries.append({"path": rel, "type": "dir", "size": None})
            elif child.suffix.lower() in ALLOWED_EXTENSIONS or child.name.startswith("."):
                entries.append({"path": rel, "type": "file", "size": child.stat().st_size})
    except OSError as exc:
        return _envelope_err("permission", f"无法读取目录：{exc}", "检查目录权限")

    total = len(entries)
    page = entries[:LIST_PAGE_SIZE]
    return _envelope_ok(
        {
            "path": path,
            "total": total,
            "truncated": total > LIST_PAGE_SIZE,
            "entries": page,
        }
    )


# ============================================================ 工具 ②：read_file


def read_file(path: str, max_bytes: int = MAX_FILE_BYTES) -> dict[str, Any]:
    """读文件（白名单内、大小受限、可截断）。截断时如实带 truncated 标记。"""
    try:
        if max_bytes < 1 or max_bytes > MAX_FILE_BYTES:
            return _envelope_err(
                "invalid_arg",
                f"max_bytes 必须在 1..{MAX_FILE_BYTES}",
                f"默认 {MAX_FILE_BYTES}；需要更大请分段读",
            )
        resolved = check_path(path)
        if not resolved.exists():
            return _envelope_err("not_found", f"文件不存在：{path}", "用 list_dir 确认文件名")
        if resolved.is_dir():
            return _envelope_err("invalid_arg", f"{path} 是目录", "list_dir 用于列目录")
        check_size_for_read(resolved)

        raw = resolved.read_bytes()[:max_bytes]
        truncated = resolved.stat().st_size > len(raw)
        text = raw.decode("utf-8", errors="replace")
        return _envelope_ok(
            {
                "path": path,
                "size": resolved.stat().st_size,
                "truncated": truncated,
                "content": text,
            }
        )
    except PathDenied as e:
        return e.envelope()


# ============================================================ 工具 ③：search_files


def search_files(pattern: str) -> dict[str, Any]:
    """按文件名通配符搜白名单内文件（不动内容，防用搜索绕过大小闸）。"""
    if not pattern or "*" not in pattern and "?" not in pattern and "." not in pattern:
        return _envelope_err(
            "invalid_arg",
            f"pattern {pattern!r} 不含通配符或扩展名",
            "示例：*.md、report*、*.json",
        )
    try:
        root = _allowed_root()
        matches = [
            p.relative_to(root).as_posix()
            for p in root.rglob(pattern)
            if p.is_file() and not p.name.startswith(".mcp-")
        ]
        total = len(matches)
        return _envelope_ok(
            {
                "pattern": pattern,
                "total": total,
                "truncated": total > LIST_PAGE_SIZE,
                "matches": matches[:LIST_PAGE_SIZE],
            }
        )
    except PathDenied as e:
        return e.envelope()
    except OSError as exc:
        return _envelope_err("internal", f"搜索失败：{exc}", "缩小 pattern 范围重试")


# ============================================================ 工具 ④：write_file（默认禁用）


def write_file(path: str, content: str) -> dict[str, Any]:
    """写文件。**默认禁用**：根目录无 .mcp-writable 标记时一律拒绝。"""
    try:
        check_size_for_write(content)
        resolved = check_path(path, for_write=True)
        if resolved.is_dir():
            return _envelope_err("invalid_arg", f"{path} 是目录", "换个文件名")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        existed = resolved.exists()
        resolved.write_text(content, encoding="utf-8")
        return _envelope_ok(
            {
                "path": path,
                "bytes_written": len(content.encode("utf-8")),
                "created": not existed,
            }
        )
    except PathDenied as e:
        return e.envelope()
    except OSError as exc:
        return _envelope_err("permission", f"写入失败：{exc}", "检查目标目录权限")


# ============================================================ 资源与提示


def list_resources(page: int = 1) -> dict[str, Any]:
    """资源清单：白名单内的全部允许文件，分页（防大目录撑爆响应）。"""
    root = _allowed_root()
    uris: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {
            ".md",
            ".txt",
            ".py",
            ".json",
            ".yml",
            ".yaml",
            ".toml",
            ".csv",
            ".log",
        }:
            continue
        uris.append(f"fs://{p.relative_to(root).as_posix()}")
    total = len(uris)
    start = (page - 1) * LIST_PAGE_SIZE
    return {
        "total": total,
        "page": page,
        "truncated": total > start + LIST_PAGE_SIZE,
        "resources": [
            {"uri": u, "mimeType": "text/plain"} for u in uris[start : start + LIST_PAGE_SIZE]
        ],
    }


def read_resource(uri: str) -> dict[str, Any]:
    """资源读取：走同一套 check_path（资源不是绕过安全层的路）。"""
    prefix = "fs://"
    if not uri.startswith(prefix):
        return _envelope_err(
            "invalid_arg", f"资源 uri 必须以 {prefix} 开头", "示例：fs://notes/todo.md"
        )
    return read_file(uri[len(prefix) :])


SUMMARIZE_PROMPT_TEMPLATE: Final[str] = (
    "请阅读以下文件内容并总结要点。\n\n"
    "文件：{path}\n"
    "要求：\n"
    "1. 用不超过 5 条要点概括核心内容；\n"
    "2. 保留关键数字与专有名词；\n"
    "3. 如果文件为空或内容与预期不符，如实说明，不要编造。\n\n"
    "文件内容：\n{content}"
)


def get_summarize_prompt(path: str, content: str) -> dict[str, Any]:
    """提示模板"总结某文件"：参数化路径与内容。"""
    return {
        "description": "总结白名单内某个文本文件的要点",
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": SUMMARIZE_PROMPT_TEMPLATE.format(path=path, content=content),
                },
            }
        ],
    }


__all__ = [
    "LIST_PAGE_SIZE",
    "get_summarize_prompt",
    "list_dir",
    "list_resources",
    "read_file",
    "read_resource",
    "search_files",
    "write_file",
]
