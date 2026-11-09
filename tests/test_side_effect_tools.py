"""副作用工具测试（W9 周五：报告落盘 + 通知 HTTP）。

**全部离线**：通知用本地 mock HTTP 端（进程内线程起 http.server），
不依赖外部服务。报告用临时目录。

分组（对应今晚验收）：
1. 报告工具 —— 落盘、返回路径与字数、覆盖标记、内容纪律分支
2. 通知工具 —— 成功投递、失败重试 1 次、仍失败结构化错误
3. 幂等验证 —— 同 key 两次调用：账本拦截 + mock 端只处理一次
4. 副作用分离 —— 先落记录再执行：账本在投递前就有 in_flight 记录
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from pydantic import ValidationError

# 顶部绑定第一代模块（test_llm_client 会清 sys.modules，教训见周四笔记）
import app.core.llm.side_effect_tools as se
from app.core.llm.side_effect_tools import (
    GenerateReportFileArgs,
    generate_report_file,
    notify_http,
    render_report_markdown,
    write_mock_receiver,
)
from app.core.llm.tools import ToolExecutionError

# ============================================================ mock 接收端（进程内）


class _MockReceiver:
    """进程内 HTTP 接收端：记录 payload，按 idempotency_key 去重。

    与 scripts/mock_notify_receiver.py 同一逻辑（那边是独立进程版，
    这里是测试线程版），双重实现刻意保持一致——端到端验证两边都能跑。
    """

    def __init__(self) -> None:
        self.received: list[dict] = []
        self.seen: set[str] = set()
        self.fail_next: int = 0  # >0 时接下来 N 次请求返回 500（测重试）
        self._server = HTTPServer(("127.0.0.1", 0), self._make_handler())
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if outer.fail_next > 0:
                    outer.fail_next -= 1
                    self.send_response(500)
                    self.end_headers()
                    return
                key = payload.get("idempotency_key", "")
                duplicate = key in outer.seen
                outer.seen.add(key)
                outer.received.append({**payload, "duplicate_at_receiver": duplicate})
                body = json.dumps({"accepted": True, "duplicate": duplicate}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        return Handler

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/notify"

    def stop(self) -> None:
        self._server.shutdown()


@pytest.fixture()
def receiver() -> _MockReceiver:
    r = _MockReceiver()
    yield r
    r.stop()


@pytest.fixture()
def reports_dir(tmp_path: Path) -> Path:
    """把报告目录与账本都指到临时目录，测试互不污染。"""
    d = tmp_path / "reports"
    original_dir = se.REPORTS_DIR
    original_ledger = se.NOTIFY_LEDGER_PATH
    se.REPORTS_DIR = d
    se.NOTIFY_LEDGER_PATH = d / "notify_ledger.jsonl"
    yield d
    se.REPORTS_DIR = original_dir
    se.NOTIFY_LEDGER_PATH = original_ledger


# ============================================================ 1. 报告工具


def test_report_writes_file_with_path_and_chars(reports_dir: Path) -> None:
    """验收①：能出文件，返回路径与字数。"""
    r = generate_report_file(
        "qa-B20241129-03",
        "批次 B20241129-03 质检报告",
        [
            {"heading": "检测结果", "content": "抗压强度 36.1 MPa（来自 query_metrics 返回）"},
            {"heading": "结论", "content": "低于限值 40.0 MPa，判定不合格"},
        ],
    )
    assert Path(r["path"]).exists()
    assert r["chars"] > 50 and r["sections"] == 2 and r["overwritten"] is False
    content = Path(r["path"]).read_text(encoding="utf-8")
    assert content.startswith("# 批次 B20241129-03 质检报告")
    assert "36.1 MPa" in content


def test_report_overwrite_flag(reports_dir: Path) -> None:
    """同 task_id 重跑覆盖（报告是当前结论的载体，不是追加日志）。"""
    args: dict = {
        "task_id": "qa-demo",
        "title": "t",
        "sections": [{"heading": "h", "content": "c"}],
    }
    first = generate_report_file(**args)
    second = generate_report_file(**args)
    assert first["overwritten"] is False and second["overwritten"] is True
    assert first["path"] == second["path"]


def test_report_rejects_empty_section(reports_dir: Path) -> None:
    with pytest.raises(ToolExecutionError, match="为空"):
        generate_report_file("qa-x", "t", [{"heading": "", "content": "c"}])


def test_report_args_validation() -> None:
    """缺参/类型错在 Pydantic 层拦截（task_id 太短、sections 空列表）。"""
    with pytest.raises(ValidationError):
        GenerateReportFileArgs.model_validate({"task_id": "ab", "title": "t", "sections": []})
    with pytest.raises(ValidationError):
        GenerateReportFileArgs.model_validate(
            {"title": "t", "sections": [{"heading": "h", "content": "c"}]}
        )


def test_render_markdown_shared_with_fake_tool() -> None:
    """渲染排版与周三哑工具一致（同一函数抽出复用，两个实现不漂移）。"""
    md = render_report_markdown("标题", [{"heading": "一", "content": "正文"}])
    assert md == "# 标题\n\n## 1. 一\n\n正文"


# ============================================================ 2. 通知工具（成功 / 重试 / 失败）


def test_notify_delivered_first_attempt(receiver: _MockReceiver, reports_dir: Path) -> None:
    r = notify_http(
        "supervisor", "批次 B20241129-03 不合格", "batch-0103-fail", endpoint=receiver.url()
    )
    assert r["delivered"] is True and r["attempt"] == 1 and r["duplicate"] is False
    assert len(receiver.received) == 1
    assert receiver.received[0]["idempotency_key"] == "batch-0103-fail"


def test_notify_retries_once_then_succeeds(receiver: _MockReceiver, reports_dir: Path) -> None:
    """失败重试 1 次：第一次 500，第二次成功，attempt=2。"""
    receiver.fail_next = 1
    r = notify_http("team", "重试后成功", "retry-succeed-key", endpoint=receiver.url())
    assert r["delivered"] is True and r["attempt"] == 2


def test_notify_structured_error_after_retries(receiver: _MockReceiver, reports_dir: Path) -> None:
    """重试 1 次仍失败 → ToolExecutionError（internal），不抛裸异常。

    W11 周一改判据：mock 端返回的是 500（服务端自己出错），不属于权限/超时，
    归 internal —— **但 retryable=True**（服务端错误退避后可能好）。
    这条测试守的是"错误类型没被误分类成 not_permitted"。
    """
    receiver.fail_next = 5  # 比重试次数多，必然失败
    with pytest.raises(ToolExecutionError, match="仍失败") as exc_info:
        notify_http("team", "必然失败", "fail-always-key", endpoint=receiver.url())
    assert exc_info.value.kind == "internal"


def test_notify_invalid_channel(reports_dir: Path) -> None:
    with pytest.raises(ToolExecutionError, match="未知通知渠道"):
        notify_http("wechat", "x", "invalid-channel-key", endpoint="http://127.0.0.1:1/none")


# ============================================================ 3. 幂等验证


def test_idempotent_same_key_blocked_by_ledger(receiver: _MockReceiver, reports_dir: Path) -> None:
    """验收③：同 key 调两次 → 第二次账本拦截（conflict），接收端只收到一次。"""
    first = notify_http("supervisor", "第一次", "idem-double-key", endpoint=receiver.url())
    assert first["delivered"] is True
    with pytest.raises(ToolExecutionError, match="已成功投递过") as exc_info:
        notify_http("supervisor", "第二次（同 key）", "idem-double-key", endpoint=receiver.url())
    assert exc_info.value.kind == "conflict"
    # mock 端确实只处理了一次
    assert len(receiver.received) == 1


def test_idempotent_receiver_side_dedup(receiver: _MockReceiver, reports_dir: Path) -> None:
    """第二层防线：绕过账本直接打接收端（模拟账本丢失），接收端仍去重。

    mock 端记录**所有**请求（审计需要），但第二次标记 duplicate_at_receiver=True
    —— "收到两次"与"处理两次"是两回事，幂等关心的是后者。
    """
    payload = json.dumps(
        {"channel": "team", "message": "m", "idempotency_key": "receiver-dedup-key"}
    ).encode()
    import urllib.request

    # 断网演练（死代理环境）下 urlopen 会把发往 127.0.0.1 的请求扔进代理——
    # 与被测代码 _post_json 修的是同一个问题（回环不走代理）。
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(2):
        req = urllib.request.Request(
            receiver.url(), data=payload, headers={"Content-Type": "application/json"}
        )
        opener.open(req, timeout=5)
    assert len(receiver.received) == 2  # 两次请求都被记录（审计）
    assert receiver.received[0]["duplicate_at_receiver"] is False  # 第一次不是重复
    assert receiver.received[1]["duplicate_at_receiver"] is True  # 第二次被标记为重复


def test_ledger_survives_rewrite(reports_dir: Path) -> None:
    """账本是文件（JSON 行）：重新读还能恢复（进程重启后重试不重复投递）。"""
    with pytest.raises(ToolExecutionError):
        notify_http(
            "team", "记账验证", "ledger-persist-key", endpoint="http://127.0.0.1:1/none"
        )  # 端点不通，但账本已写
    ledger = se._load_ledger()
    assert "ledger-persist-key" in ledger
    assert ledger["ledger-persist-key"]["status"] == "in_flight"


# ============================================================ 4. 副作用分离


def test_side_effect_order_record_before_send(receiver: _MockReceiver, reports_dir: Path) -> None:
    """验收④的核心：先落记录、再执行 —— 投递进行中账本就有 in_flight。"""
    # 让第一次尝试慢一点不方便 mock；改用失败路径：投递失败后账本里仍有记录
    receiver.fail_next = 5
    with pytest.raises(ToolExecutionError):
        notify_http("team", "副作用顺序验证", "order-check-key", endpoint=receiver.url())
    ledger = se._load_ledger()
    assert "order-check-key" in ledger  # 失败了但意图已留痕
    assert ledger["order-check-key"]["status"] == "in_flight"


def test_retry_does_not_duplicate_effect(receiver: _MockReceiver, reports_dir: Path) -> None:
    """重试不重复：第一次成功后，任何形式的重试都不会再发。"""
    notify_http("team", "成功一次", "no-dup-key", endpoint=receiver.url())
    with pytest.raises(ToolExecutionError, match="已成功投递过"):
        notify_http("team", "重试（同 key）", "no-dup-key", endpoint=receiver.url())
    assert len(receiver.received) == 1  # 效果只发生一次


def test_mock_receiver_script_written(tmp_path: Path) -> None:
    """独立进程版 mock 端脚本能生成且语法正确。"""
    target = write_mock_receiver(tmp_path / "mock_notify_receiver.py")
    import ast

    ast.parse(target.read_text(encoding="utf-8"))


def test_envelope_paths(reports_dir: Path, receiver: _MockReceiver) -> None:
    """统一信封：报告/通知的 invalid_json / invalid_arg / 成功三路。"""
    ok = asyncio_run(
        se.execute_report_tool(
            json.dumps(
                {"task_id": "qa-env", "title": "t", "sections": [{"heading": "h", "content": "c"}]}
            )
        )
    )
    assert ok["ok"] is True
    bad = asyncio_run(se.execute_report_tool("{bad"))
    assert bad["ok"] is False and bad["error_type"] == "invalid_json"
    missing = asyncio_run(se.execute_report_tool('{"title": "t"}'))
    assert missing["ok"] is False and missing["error_type"] == "invalid_arg"

    # 通知信封：端点不通 → 结构化 internal，不异常穿透
    notify_ok = asyncio_run(
        se.execute_notify_http_tool(
            json.dumps({"channel": "team", "message": "m", "idempotency_key": "env-notify-key"})
        )
    )
    assert notify_ok["ok"] is False
    assert notify_ok["error_type"] in ("internal", "timeout")


def asyncio_run(coro: object) -> object:
    import asyncio

    return asyncio.run(coro)  # type: ignore[arg-type]
