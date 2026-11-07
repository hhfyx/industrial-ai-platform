"""进程内通知 mock 接收端（测试与联调脚本共用）。

与 `scripts/mock_notify_receiver.py`（独立进程版）同一逻辑：
记录 payload、按 idempotency_key 去重标记、可注入失败测重试。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class MockReceiver:
    """进程内 HTTP 接收端：记录 payload，按 idempotency_key 去重。"""

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
