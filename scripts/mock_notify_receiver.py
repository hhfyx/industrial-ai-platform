"""通知 mock 接收端：打印收到的 payload，并按 idempotency_key 去重。

启动（演示脚本会自动拉起，也可手动）：
    python scripts/mock_notify_receiver.py [--port 8920]
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

SEEN: set[str] = set()
LOG: list[dict] = []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        key = payload.get("idempotency_key", "")
        duplicate = key in SEEN
        SEEN.add(key)
        LOG.append({**payload, "duplicate_at_receiver": duplicate})
        print(
            f"[receiver] {'DUPLICATE ' if duplicate else ''}{json.dumps(payload, ensure_ascii=False)}",
            flush=True,
        )
        body = json.dumps({"accepted": True, "duplicate": duplicate}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """GET /log：查看接收端收到的全部请求（幂等验证用）。"""
        body = json.dumps(LOG, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass  # 静默默认日志，payload 打印已经够了


if __name__ == "__main__":
    port = 8920
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    print(f"[receiver] listening on 127.0.0.1:{port}", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
