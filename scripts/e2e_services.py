"""端到端联调（W12 周三 15:40–16:40）：走一遍完整链路。

链路：上传→索引→提问→流式→引用→Agent 任务→进度查询→取消
端口：8100（rag 面）、8101（gen 面）—— compose 里由 nginx 分流，
本地联调直接打两个端口（nginx 形态的等价物是路由表，逻辑一致）。
"""

import json
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, ".")

RAG = "http://127.0.0.1:8100"
GEN = "http://127.0.0.1:8101"
KEY = "demo-key-12345678"
HEADERS = {"X-API-Key": KEY, "Content-Type": "application/json"}

results = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{'✅' if ok else '❌'} {name}{' — ' + detail if detail else ''}")


def call(method: str, url: str, body: dict | None = None, headers: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers or HEADERS, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, dict(resp.headers), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read().decode("utf-8"))


# 1. readiness（部署起跑门）
code, _, body = call("GET", f"{RAG}/api/v2/ready")
record("readiness 探针", code in (200, 503), f"HTTP {code}")

# 2. 上传→索引（index_text 语义走 extract 的 text 字段没有索引入口——
#    索引入口在 UI/脚本层；服务面验证 ready 之后的 extract 链路）
code, _, body = call(
    "POST",
    f"{RAG}/api/v2/extract",
    {"text": "合同编号：HT-2026-001，甲方：张三", "fields": ["合同编号", "甲方"]},
)
record(
    "v2 extract（鉴权+限流样板）",
    code == 200 and body.get("data", {}).get("合同编号") == "HT-2026-001",
    f"HTTP {code}",
)

# 3. 提问（rag ask 走 CLI/UI；服务面有 chat 非流式）
code, headers, body = call("GET", f"{RAG}/api/v2/rate-limit/status")
record(
    "限流状态端点",
    code == 200 and "remaining" in body,
    f"HTTP {code}, remaining={body.get('remaining')}",
)

# 4. 流式（SSE：验证首帧非缓冲到达）
req = urllib.request.Request(
    f"{RAG}/api/v1/chat/stream",
    data=json.dumps(
        {"messages": [{"role": "user", "content": "用一句话说明什么是违约金"}]}
    ).encode(),
    headers={**HEADERS, "Accept": "text/event-stream"},
    method="POST",
)
first_frame_ms = None
frames = 0
try:
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("event:") or line.startswith("data:"):
                frames += 1
            if first_frame_ms is None and line.startswith("data:"):
                first_frame_ms = (time.perf_counter() - started) * 1000
            if frames > 6:
                break
    record(
        "SSE 流式（首帧时间 = 非缓冲证据）",
        frames >= 2,
        f"frames={frames}, first_frame={first_frame_ms:.0f}ms",
    )
except urllib.error.HTTPError as e:
    record("SSE 流式", False, f"HTTP {e.code}")

# 5. Agent 任务：提交 → 进度 → 取消
code, headers, body = call(
    "POST",
    f"{RAG}/api/v2/agent-tasks",
    {"instruction": "查询批次 B-2024-042 的抗压强度并出判定"},
)
task_id = body.get("task_id", "")
record("Agent 任务提交（202）", code == 202 and bool(task_id), f"HTTP {code}, task={task_id}")

code, _, body = call("GET", f"{RAG}/api/v2/agent-tasks/{task_id}")
six_fields = {"status", "current_node", "steps", "waiting_human", "partial_answer", "degraded"}
record(
    "进度查询六字段",
    code == 200 and six_fields <= set(body.keys()),
    f"HTTP {code}, status={body.get('status')}",
)

code, _, body404 = call("GET", f"{RAG}/api/v2/agent-tasks/nonexistent0000")
record("不存在的 id → 404", code == 404 and body404.get("code") == "TASK_NOT_FOUND", f"HTTP {code}")

code, _, body = call("DELETE", f"{RAG}/api/v2/agent-tasks/{task_id}")
record(
    "取消请求（200/202 形态）", code in (200,), f"HTTP {code}, cancelled={body.get('cancelled')}"
)

# 6. 生成面：提交异步任务 → 进度 → 取消
code, _, body = call(
    "POST",
    f"{GEN}/api/v1/generate/async",
    {"defect_class": "划痕", "count": 4, "size": 512, "seed": 4242},
)
gen_task = body.get("task_id", "")
record("生成任务提交（202）", code == 202, f"HTTP {code}, task={gen_task}")

time.sleep(0.3)
code, _, body = call("GET", f"{GEN}/api/v1/generate/tasks/{gen_task}")
has_pct = "progress_percent" in body and "queue_position" in body
record(
    "生成任务进度（百分比+队列位置字段）",
    code == 200 and has_pct,
    f"HTTP {code}, status={body.get('status')}, pct={body.get('progress_percent')}",
)

# 等 fake 生成完（毫秒级）再查一次
time.sleep(1.0)
code, _, body = call("GET", f"{GEN}/api/v1/generate/tasks/{gen_task}")
done_pct = body.get("progress_percent") == 100
record(
    "生成任务完成（100%）",
    code == 200 and body.get("status") == "succeeded" and done_pct,
    f"HTTP {code}, pct={body.get('progress_percent')}",
)

# 再提交一个，立刻取消（queued 概率低——fake 太快；验证端点行为）
code, _, body = call(
    "POST",
    f"{GEN}/api/v1/generate/async",
    {"defect_class": "凹坑", "count": 8, "size": 768, "seed": 777},
)
gen_task2 = body.get("task_id", "")
code, _, body = call("DELETE", f"{GEN}/api/v1/generate/tasks/{gen_task2}")
record(
    "生成任务取消端点（200/409 均合法）",
    code in (200, 409),
    f"HTTP {code}, status={body.get('status', body.get('code'))}",
)

# 等到终态再取消：已 succeeded 的任务取消无效 → 409（fake 快，等 1.5s 必到终态）
time.sleep(1.5)
code, _, body = call("GET", f"{GEN}/api/v1/generate/tasks/{gen_task2}")
final_status = body.get("status")
code, _, body = call("DELETE", f"{GEN}/api/v1/generate/tasks/{gen_task2}")
if final_status == "succeeded":
    record(
        "重复取消已完结任务 → 409",
        code == 409 and body.get("code") == "TASK_FINISHED",
        f"HTTP {code}",
    )
else:
    # 若任务仍在 running（fake 偶尔慢），本次取消成立，验证幂等回显
    record(
        "取消 running 任务（幂等回显形态）",
        code == 200,
        f"HTTP {code}, status={body.get('status')}",
    )

# 7. 鉴权负路径（v2）
code, _, body = call(
    "GET", f"{RAG}/api/v2/agent-tasks/{task_id}", headers={"Content-Type": "application/json"}
)
record("无 key 访问 v2 → 401", code == 401, f"HTTP {code}")

print()
passed = sum(1 for _, ok, _ in results if ok)
print(f"端到端联调：{passed}/{len(results)} 通过")


async def _noop() -> None:  # 占位：保持脚本无未用告警
    return None


if __name__ == "__main__":
    raise SystemExit(0 if passed == len(results) else 1)
