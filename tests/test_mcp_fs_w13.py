"""W13 周五：文件系统 MCP Server 的安全与功能测试。

攻击测试 5 条（任务书 13:30–14:30，必须全过）：
① ../ 逃逸 ② 绝对路径/盘符 ③ 符号链接逃逸 ④ 超大文件 ⑤ 越权写。
每条断言被拒且 error_type 可读（W11 封闭枚举）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import mcp_fs_tools as fs
from app.core.mcp_fs_security import (
    MAX_FILE_BYTES,
    PathDenied,
    check_path,
)


@pytest.fixture()
def fs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "mcp_fs_root"
    root.mkdir()
    (root / "notes").mkdir()
    (root / "notes" / "todo.md").write_text("# 待办\n- 写周记", encoding="utf-8")
    (root / "readme.md").write_text("readme", encoding="utf-8")
    monkeypatch.setenv("MCP_FS_ROOT", str(root))
    return root


# ============================================================ 攻击测试（5 条全拒）


def test_attack_1_dotdot_escape(fs_root: Path) -> None:
    """① ../../etc/passwd：resolve 后越出白名单 → 拒绝。"""
    r = fs.read_file("../../etc/passwd")
    assert r["ok"] is False
    assert r["error_type"] == "permission"
    assert "越出白名单" in r["message"]


def test_attack_2_absolute_path_and_drive(fs_root: Path) -> None:
    """② 绝对路径与盘符：C:\\Windows\\win.ini、D:\\x → 字符串层直接拒。"""
    for bad in ("C:\\Windows\\win.ini", "D:/git/x.md", "/etc/passwd"):
        r = fs.read_file(bad)
        assert r["ok"] is False, f"{bad} 应被拒"
        assert r["error_type"] == "invalid_arg", f"{bad}: {r}"
        assert "相对路径" in r["message"]


def test_attack_3_symlink_escape(fs_root: Path, tmp_path: Path) -> None:
    """③ 符号链接指向外部：link 名在白名单内、目标在外——resolve 后拒绝。"""
    secret = tmp_path / "secret.md"
    secret.write_text("top secret", encoding="utf-8")
    link = fs_root / "notes" / "evil.md"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("Windows 无符号链接权限")
    r = fs.read_file("notes/evil.md")
    assert r["ok"] is False
    assert r["error_type"] == "permission"
    assert "越出白名单" in r["message"]


def test_attack_4_oversized_read(fs_root: Path) -> None:
    """④ 超大文件：读前大小检查 → too_large（不是读进内存才发现）。"""
    big = fs_root / "big.md"
    big.write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    r = fs.read_file("big.md")
    assert r["ok"] is False
    assert r["error_type"] == "too_large"
    assert "超过读取上限" in r["message"]


def test_attack_5_write_without_marker(fs_root: Path) -> None:
    """⑤ 写：默认禁用（无标记）→ permission；且系统目录/逃逸路径写也全拒。"""
    r = fs.write_file("notes/new.md", "hello")
    assert r["ok"] is False
    assert r["error_type"] == "permission"
    assert ".mcp-writable" in r["message"]

    # 越权写目标即使给了标记也走五道闸
    (fs_root / ".mcp-writable").write_text("", encoding="utf-8")
    r2 = fs.write_file("../../Windows/temp.md", "x")
    assert r2["ok"] is False and r2["error_type"] == "permission"
    r3 = fs.write_file("C:\\Windows\\Temp\\x.md", "x")
    assert r3["ok"] is False and r3["error_type"] == "invalid_arg"


def test_write_allowed_with_marker(fs_root: Path) -> None:
    """写闸的正向用例：有标记 → 白名单内可写；可写不豁免五道闸。"""
    (fs_root / ".mcp-writable").write_text("", encoding="utf-8")
    r = fs.write_file("notes/done.md", "写完了")
    assert r["ok"] is True
    assert r["data"]["created"] is True
    assert (fs_root / "notes" / "done.md").read_text(encoding="utf-8") == "写完了"

    # 可写标记不豁免扩展名白名单
    r2 = fs.write_file("notes/evil.exe", "MZ")
    assert r2["ok"] is False and r2["error_type"] == "permission"


# ============================================================ 路径闸单元


def test_check_path_nul_byte(fs_root: Path) -> None:
    with pytest.raises(PathDenied) as ei:
        check_path("notes\x00.md")
    assert ei.value.error_type == "invalid_arg"


def test_check_path_empty(fs_root: Path) -> None:
    with pytest.raises(PathDenied) as ei:
        check_path("  ")
    assert ei.value.error_type == "invalid_arg"


def test_resolve_before_check_is_the_whole_point(fs_root: Path) -> None:
    """必答问题的回归钉：字符串前缀检查会放行、resolve 检查必拒。

    经典逃逸形态：`notes/todo.md/../../../outside.md` 在字符串上以
    白名单相对路径开头（假象），resolve 后落到白名单外（真相）。
    """
    evil = "notes/todo.md/../../../../outside.md"
    assert evil.startswith("notes/")  # 字符串层面"白名单内"——假象
    naive = fs_root / evil  # 字符串拼接不 resolve 的检查会看它
    assert str(naive).startswith(str(fs_root))  # 前缀检查：放行（错误的实现会过）
    with pytest.raises(PathDenied):
        check_path(evil)  # resolve 层面：路径越出白名单——正确实现必拒


# ============================================================ 功能测试


def test_list_dir_and_read(fs_root: Path) -> None:
    r = fs.list_dir("notes")
    assert r["ok"] is True
    assert any(e["path"] == "notes/todo.md" for e in r["data"]["entries"])

    r2 = fs.read_file("notes/todo.md")
    assert r2["ok"] is True
    assert "待办" in r2["data"]["content"]
    assert r2["data"]["truncated"] is False


def test_read_truncates_with_flag(fs_root: Path) -> None:
    content = "字" * 2000
    (fs_root / "long.md").write_text(content, encoding="utf-8")
    r = fs.read_file("long.md", max_bytes=100)
    assert r["ok"] is True
    assert r["data"]["truncated"] is True
    assert len(r["data"]["content"]) <= 100


def test_read_not_found_vs_dir(fs_root: Path) -> None:
    assert fs.read_file("nope.md")["error_type"] == "not_found"
    assert fs.read_file("notes")["error_type"] == "invalid_arg"


def test_search_files(fs_root: Path) -> None:
    r = fs.search_files("*.md")
    assert r["ok"] is True
    assert "notes/todo.md" in r["data"]["matches"]


def test_list_dir_not_found(fs_root: Path) -> None:
    r = fs.list_dir("ghost")
    assert r["ok"] is False and r["error_type"] == "not_found"


# ============================================================ 资源与提示


def test_resources_list_and_read(fs_root: Path) -> None:
    lst = fs.list_resources()
    assert lst["total"] >= 2
    assert any(x["uri"] == "fs://notes/todo.md" for x in lst["resources"])

    r = fs.read_resource("fs://notes/todo.md")
    assert r["ok"] is True and "待办" in r["data"]["content"]

    # 资源不是绕过安全层的路：../ 逃逸同样被拒
    bad = fs.read_resource("fs://../../etc/passwd")
    assert bad["ok"] is False and bad["error_type"] == "permission"


def test_summarize_prompt_template(fs_root: Path) -> None:
    p = fs.get_summarize_prompt("notes/todo.md", "# 待办\n- 写周记")
    msg = p["messages"][0]["content"]["text"]
    assert "notes/todo.md" in msg
    assert "待办" in msg
    assert "如实说明" in msg
