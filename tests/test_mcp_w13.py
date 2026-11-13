"""W13 周四：最小 MCP Server 与手写 client 的测试。

不测 SDK 内部（那是 SDK 的测试），测我们自己的两件事：
① server 暴露的工具行为正确（直接调函数层）；
② 手写 client 的帧构造器遵守 JSON-RPC 2.0（notification 无 id、
   request 带 id、error 帧判定）。
端到端帧序列由 scripts/mcp_handwritten_client.py 实测（见
docs/mcp-server.md §一），不放进 pytest（起子进程慢且脆）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.mcp_handwritten_client import _frame
from scripts.mcp_min_server import add, hello, ping

# ============================================================ 工具行为


def test_hello_greets() -> None:
    assert "MCP" in hello("MCP")


def test_ping_pong() -> None:
    assert ping() == "pong"


def test_add() -> None:
    assert add(2, 40) == 42


# ============================================================ 帧构造器


def test_request_frame_has_id() -> None:
    f = _frame("tools/list", {}, id_=7)
    assert f["jsonrpc"] == "2.0"
    assert f["method"] == "tools/list"
    assert f["id"] == 7


def test_notification_frame_has_no_id() -> None:
    """JSON-RPC 2.0：notification 无 id 字段（连 id=None 都不能有）。"""
    f = _frame("notifications/initialized")
    assert "id" not in f, "notification 带 id 字段会被当成 request 等响应"


def test_initialize_frame_shape() -> None:
    f = _frame(
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "t", "version": "0"},
        },
        id_=1,
    )
    assert f["params"]["protocolVersion"] == "2025-06-18"
    assert f["params"]["clientInfo"]["name"] == "t"
