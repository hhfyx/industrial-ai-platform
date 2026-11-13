"""W13 周五 · MCP 文件系统 Server 的路径安全层（本周最重要的 20 行代码）。

## 为什么路径校验必须 resolve() 之后再做（必答问题）

`..` 与符号链接在**字符串层面看不出来**：

    "mcp_fs_root/notes/../../../../etc/passwd"   字符串上以白名单开头 ✅
    resolve() 之后 → "C:/etc/passwd"            真实位置暴露 ❌
不 resolve 的白名单等于没有白名单——而它会通过你看得懂的代码评审
（`path.startswith(ALLOWED)` 谁都觉得没问题）。所以本模块的全部校验
都发生在 `Path.resolve()`（真实路径、解析符号链接）之后。

## 五道闸（顺序即逻辑）

| # | 闸 | 拦什么 |
| --- | --- | --- |
| 1 | 空路径 / NUL 字节 | 明显的垃圾输入 |
| 2 | **绝对路径与盘符**（`C:` 盘符、`/etc`、`D:/`） | MCP client 传来的参数约定是**相对白名单根**的路径；绝对路径一律拒绝——放行它就要回答"凭什么 server 知道宿主机的盘符布局" |
| 3 | resolve() 后仍在 ALLOWED_ROOT 内 | `..` 逃逸与符号链接逃逸（唯一能防这两样的就是 resolve） |
| 4 | 扩展名白名单 | 只放行文本类（.md/.txt/.py/.json/.yml/...）；二进制（exe/图片）不进这个 server 的用途 |
| 5 | 文件大小上限 | read_file 读前检查 size（防"无限读"）；write_file 写前检查 content 长度 |

另有一道**写闸**（只对 write_file）：ALLOWED_ROOT 下必须存在
`.mcp-writable` 标记文件——写能力默认禁用，整棵树只读；想开写
必须显式在根目录放标记（部署时的显式决定，不是配置文件里的默认 true）。

## 与 W11 错误结构的对齐

全部拒绝都返回 W11 的封闭枚举（invalid_arg / not_found / permission /
too_large），模型拿到 error_type 就知道修正动作——与 qa_tools 的
"错误分类学是模型的修正动作菜单"同一纪律。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

# ------------------------------------------------------------------ 策略常量

DEFAULT_ALLOWED_ROOT: Final[str] = "mcp_fs_root"
WRITABLE_MARKER: Final[str] = ".mcp-writable"

ALLOWED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".md", ".txt", ".py", ".json", ".yml", ".yaml", ".toml", ".csv", ".log"}
)

MAX_FILE_BYTES: Final[int] = 256 * 1024  # read/write 单文件上限 256 KB
MAX_WRITE_CHARS: Final[int] = 200_000  # write_file 内容上限（字符）

#: error_type 封闭枚举（W11 对齐；本模块用到的子集 + too_large 扩展）
E_INVALID_ARG = "invalid_arg"
E_NOT_FOUND = "not_found"
E_PERMISSION = "permission"
E_TOO_LARGE = "too_large"


class PathDenied(Exception):
    """路径被安全层拒绝。error_type 是封闭枚举，message 给模型读。"""

    def __init__(self, error_type: str, message: str, hint: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.hint = hint

    def envelope(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error_type": self.error_type,
            "message": self.message,
            "hint": self.hint,
        }


# ------------------------------------------------------------------ 五道闸


def _allowed_root() -> Path:
    root = Path(os.environ.get("MCP_FS_ROOT", DEFAULT_ALLOWED_ROOT))
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _has_writable_marker(root: Path) -> bool:
    return (root / WRITABLE_MARKER).exists()


def check_path(
    raw: str,
    *,
    for_write: bool = False,
    root: Path | None = None,
) -> Path:
    """五道闸顺序校验，全过返回 resolve 后的真实路径；否则抛 PathDenied。

    这是全模块唯一入口——tools 与 resources 的 read 都走这里，
    不存在"另一条没走闸的读取路径"。
    """
    root = (root or _allowed_root()).resolve()

    # 闸 1：空 / NUL
    if not raw or not raw.strip():
        raise PathDenied(
            E_INVALID_ARG, "路径为空", "请提供白名单根目录下的相对路径，如 notes/todo.md"
        )
    if "\x00" in raw:
        raise PathDenied(E_INVALID_ARG, "路径含 NUL 字节", "路径必须是普通文本")

    p = Path(raw)

    # 闸 2：绝对路径与盘符（在字符串层就能判，不用等 resolve）。
    # `/etc/passwd` 在 Windows 上 Path.is_absolute()=False（无盘符），
    # 但它明确表达"从文件系统根开始"——同样拒绝，宁可错杀。
    if (
        p.is_absolute()
        or p.drive
        or raw.startswith(("/", "\\"))
        or (len(raw) >= 2 and raw[1] == ":")
    ):
        raise PathDenied(
            E_INVALID_ARG,
            f"拒绝绝对路径/盘符 {raw!r}：本 server 只接受白名单根目录下的相对路径",
            "改成相对路径，如 docs/readme.md",
        )

    # 闸 3：resolve 之后必须仍在根内（.. 逃逸与符号链接逃逸的唯一防线）
    resolved = (root / p).resolve()
    if not resolved.is_relative_to(root):
        raise PathDenied(
            E_PERMISSION,
            f"路径 {raw!r} 解析后 ({resolved}) 越出白名单根 {root}",
            "路径不能包含逃逸白名单的 .. 或符号链接",
        )

    # 闸 4：扩展名白名单（文件存在时按真实后缀判；不存在时按请求后缀判）
    ext = resolved.suffix.lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise PathDenied(
            E_PERMISSION,
            f"扩展名 {ext!r} 不在白名单（允许：{', '.join(sorted(ALLOWED_EXTENSIONS))}）",
            "本 server 只暴露文本类文件",
        )

    # 写闸：可写标记（写能力默认禁用）
    if for_write and not _has_writable_marker(root):
        raise PathDenied(
            E_PERMISSION,
            "写能力未启用：白名单根目录没有 .mcp-writable 标记文件",
            "只读模式；需要写时由部署方在根目录显式放置标记文件",
        )

    return resolved


def check_size_for_read(path: Path) -> None:
    """闸 5a：读前大小检查（防无限读）。"""
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise PathDenied(
            E_TOO_LARGE,
            f"文件 {path.name} 有 {size} 字节，超过读取上限 {MAX_FILE_BYTES}",
            f"用 read_file 的 max_bytes 分段读，或让部署方把上限调大（当前 {MAX_FILE_BYTES // 1024} KB）",
        )


def check_size_for_write(content: str) -> None:
    """闸 5b：写前内容检查。"""
    if len(content) > MAX_WRITE_CHARS:
        raise PathDenied(
            E_TOO_LARGE,
            f"写入内容 {len(content)} 字符超过上限 {MAX_WRITE_CHARS}",
            "分段写或精简内容",
        )


__all__ = [
    "ALLOWED_EXTENSIONS",
    "E_INVALID_ARG",
    "E_NOT_FOUND",
    "E_PERMISSION",
    "E_TOO_LARGE",
    "MAX_FILE_BYTES",
    "PathDenied",
    "WRITABLE_MARKER",
    "check_path",
    "check_size_for_read",
    "check_size_for_write",
]
