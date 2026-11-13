"""W13 周五 · 端到端验证：手写 host 驱动 MCP fs server 完成"读 md 并总结要点"。

任务书 15:30-16:30 的验收：接 Claude Desktop 或自写 host，让 Agent 完成
"读某个 md 并总结要点"，观察它是否只调了该调的。

本脚本是**自写 host** 的最小形态（不用真 LLM，用确定性决策脚本模拟
"模型只调该调的"）：决策序列写死为 list -> read -> prompt get，
这正是 Host 的职责边界——**决策在 Host，执行在 Server**（周四的概念
在端到端里的落地）。每一步打印"Host 决策了什么、Server 执行了什么"，
并断言全程只产生预期的 7 次调用。

跑法：
    python -X utf8 scripts/mcp_fs_e2e_host.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SERVER_CMD = [sys.executable, "-X", "utf8", "scripts/mcp_fs_server.py"]


def _server_env() -> dict[str, str]:
    """子进程环境：PYTHONPATH 指到项目根（server 要 import app.*）。"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["PYTHONIOENCODING"] = "utf-8"
    return env


_next_id = 0
_calls: list[str] = []


def _frame(method: str, params: dict | None = None, *, id_: int | None = None) -> dict:
    global _next_id
    f: dict = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        f["params"] = params
    if id_ is not None:
        _next_id += 1
        f["id"] = _next_id
    return f


async def _send(proc, frame: dict) -> dict | None:
    """发一帧；notification 无响应。打印 Host 决策与 Server 执行的分工。"""
    has_id = "id" in frame
    label = f"{frame['method']}"
    if frame.get("params", {}).get("name"):
        label += f" {frame['params']['name']}"
    _calls.append(label)
    print(f"\n[Host 决策] {label}{'（notification）' if not has_id else ''}")
    proc.stdin.write((json.dumps(frame, ensure_ascii=False) + "\n").encode())
    await proc.stdin.drain()
    if not has_id:
        return None
    while True:
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=20)
        msg = json.loads(line.decode("utf-8"))
        if msg.get("id") == frame["id"] and ("result" in msg or "error" in msg):
            if "error" in msg:
                print(f"[Server 异常] {msg['error']}")
                return msg
            result = msg["result"]
            if result.get("isError"):
                print(f"[Server 拒绝] {result['content'][0]['text'][:120]}")
            else:
                print("[Server 执行] 成功")
            return msg


async def main() -> int:
    # 被总结的目标文件（准备在白名单内）
    root = Path("mcp_fs_root")
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "weekly.md").write_text(
        "# W13 周记\n\n- 完成三级 trace 模型与全链路埋点\n"
        "- 对账抓出 extract 漏埋点，成本低估 45%\n"
        "- 告警阈值全部来自 W12 拐点数据\n"
        "- 演练：P95 与错误率告警触发、恢复通知送达\n",
        encoding="utf-8",
    )

    proc = await asyncio.create_subprocess_exec(
        *SERVER_CMD,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        # stderr 接管到 DEVNULL 而不是 PIPE：mcp 框架会往 stderr 写日志，
        # PIPE 满了会阻塞 server 进程（演练第一次跑就卡在这里）。
        stderr=asyncio.subprocess.DEVNULL,
        env=_server_env(),
    )
    ok = True
    try:
        # ---- 握手
        r = await _send(
            proc,
            _frame(
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "e2e-host", "version": "0.1.0"},
                },
                id_=1,
            ),
        )
        assert r and "result" in r
        # ---- initialized
        await _send(proc, _frame("notifications/initialized"))

        # ---- Host 决策 1：先发现能力（tools/list）
        r = await _send(proc, _frame("tools/list", {}, id_=2))
        tools = [t["name"] for t in r["result"]["tools"]]
        print(f"  → 发现 {len(tools)} 个工具: {tools}")

        # ---- Host 决策 2：列目录找目标（fs_list_dir）
        r = await _send(
            proc,
            _frame("tools/call", {"name": "fs_list_dir", "arguments": {"path": "reports"}}, id_=3),
        )
        # 2.x 契约：工具返回 dict → 框架包进 structuredContent
        entries = r["result"].get("structuredContent") or json.loads(
            r["result"]["content"][0]["text"]
        )
        assert any(e["path"] == "reports/weekly.md" for e in entries["entries"])

        # ---- Host 决策 3：读文件（fs_read_file）
        r = await _send(
            proc,
            _frame(
                "tools/call",
                {"name": "fs_read_file", "arguments": {"path": "reports/weekly.md"}},
                id_=4,
            ),
        )
        data = r["result"].get("structuredContent") or json.loads(r["result"]["content"][0]["text"])
        print(f"  → 读到 {data['size']} 字节，truncated={data['truncated']}")

        # ---- Host 决策 4：拿"总结"模板（prompts/get）——prompt 原语不是 tool
        r = await _send(
            proc,
            _frame(
                "prompts/get",
                {"name": "summarize_file", "arguments": {"path": "reports/weekly.md"}},
                id_=5,
            ),
        )
        prompt_text = r["result"]["messages"][0]["content"]["text"]
        assert "W13 周记" in prompt_text
        print(f"  → 模板已参数化（{len(prompt_text)} 字符），Host 可把它交给模型")

        # ---- 安全观察：Host 试图越权读，看 Server 拒绝（决策错时 Server 兜底）
        r = await _send(
            proc,
            _frame(
                "tools/call",
                {"name": "fs_read_file", "arguments": {"path": "../../../etc/passwd"}},
                id_=6,
            ),
        )
        assert r["result"]["isError"] is True
        print("  → 越权读取被 Server 拒绝（isError=true）——决策在 Host，安全在 Server")

        # ---- 断言：只调了该调的
        expected = [
            "initialize",
            "notifications/initialized",
            "tools/list",
            "tools/call fs_list_dir",
            "tools/call fs_read_file",
            "prompts/get summarize_file",
            "tools/call fs_read_file",
        ]
        assert _calls == expected, f"调用序列不符：{_calls}"
        print(f"\n=== 端到端通过：全程 {len(_calls)} 次调用，无多余调用 ===")
        print("=== 分工确认：Host 决策（找文件→读→取模板），Server 执行+安全兜底 ===")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\n端到端失败: {type(exc).__name__}: {exc}")
    finally:
        proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            proc.kill()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
