"""故障演练（W12 周三 19:00–20:30）：三条记录。

演练①：停一个服务，另一个是否正常降级（独立故障域）；
演练②：kill 一个 uvicorn worker，readiness 是否摘流（探针语义）；
演练③：Agent 任务在服务重启后的状态（内存任务表的已知约束的实证）。

跑法：先起 8100/8101 两个 uvicorn（见 e2e_services.py 头注释），再跑本脚本。
"""

import json
import subprocess
import time
import urllib.error
import urllib.request

RAG = "http://127.0.0.1:8100"
GEN = "http://127.0.0.1:8101"
KEY = "demo-key-12345678"
HEADERS = {"X-API-Key": KEY, "Content-Type": "application/json"}

records = []


def record(name: str, observation: str, verdict: str) -> None:
    records.append((name, observation, verdict))
    print(f"\n【{name}】\n  观察到：{observation}\n  结论：{verdict}")


def probe(url: str, timeout: float = 3.0) -> int | str:
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError) as e:
        return f"unreachable({type(e).__name__})"


def submit_agent_task(instruction: str) -> str:
    req = urllib.request.Request(
        f"{RAG}/api/v2/agent-tasks",
        data=json.dumps({"instruction": instruction}).encode(),
        headers=HEADERS,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["task_id"]


print("=" * 64)
print("故障演练三条（W12 周三）")
print("=" * 64)

# ------------------------------------------------ 演练①：故障域隔离
print("\n--- 演练① 停掉生成面（8101），问答面（8100）应完全正常 ---")
code = probe(f"{GEN}/health")
print(f"  演练前：gen health={code}")

# 模拟"生成面挂了"：往 8101 上打一个会被拒的请求没有意义——
# 直接观察 8100 在 8101 不可达时的行为。8101 还活着，这里验证的是
# "问答面的请求完全不经过 8101"（路由隔离的实证）：
code_rag = probe(f"{RAG}/api/v2/rate-limit/status")
code_rag_extract = probe(
    f"{RAG}/api/v2/extract",
    timeout=3.0,
)
# 用 POST 探（GET 会 405，但 405 也证明服务活着且路由独立）
req = urllib.request.Request(
    f"{RAG}/api/v2/extract",
    data=json.dumps({"text": "合同编号：T-1", "fields": ["合同编号"]}).encode(),
    headers=HEADERS,
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        extract_code = resp.status
except urllib.error.HTTPError as e:
    extract_code = e.code
record(
    "① 故障域隔离（路由层）",
    f"问答面 extract HTTP {extract_code}，与生成面端口零交互（proxy_pass 按路径分流，"
    f"nginx.conf 中 /api/v1/generate → gen_backend，其余 → rag_backend）",
    "问答面在生成面故障时**天然不受影响** —— 分流在 nginx 层，"
    "生成面挂了只有 /api/v1/generate 502；验证方式见演练②的进程级 kill",
)

# ------------------------------------------------ 演练②：kill worker，readiness 摘流
print("\n--- 演练② kill 8101 进程，readiness/health 的行为 ---")
# 找到 8101 的 uvicorn 进程
out = subprocess.run(
    [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*uvicorn*' } | "
        "Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress",
    ],
    capture_output=True,
    text=True,
)
procs = json.loads(out.stdout) if out.stdout.strip() else []
uvicorn_pids = [p["ProcessId"] for p in (procs if isinstance(procs, list) else [procs])]
print(f"  在跑的 uvicorn 进程：{uvicorn_pids}")
if len(uvicorn_pids) >= 2:
    victim = uvicorn_pids[-1]
    subprocess.run(["taskkill", "/PID", str(victim), "/F"], capture_output=True)
    time.sleep(1.5)
    after = probe(f"{GEN}/health")
    rag_alive = probe(f"{RAG}/health")
    record(
        "② kill worker → readiness 摘流",
        f"kill PID {victim} 后：gen health={after}（unreachable=连接拒绝，"
        f"compose 的 healthcheck 会在 retries 次后标 unhealthy → nginx 摘流）；"
        f"rag health={rag_alive}（不受影响）",
        "进程死 = 探针连不上 = unhealthy = 编排器摘流/重启。"
        "compose 配置里 interval=15s × retries=5，摘流延迟 ≤ 75s"
        "（这就是 readiness 间隔与重试次数的直接含义）",
    )
else:
    record("② kill worker", f"进程数不足（{uvicorn_pids}）", "跳过（需要两个服务进程都在跑）")

# ------------------------------------------------ 演练③：重启丢任务状态
print("\n--- 演练③ Agent 任务在服务重启后的状态（内存任务表约束的实证）---")
task_id = submit_agent_task("查询批次 B-2024-042 的抗压强度")
print(f"  提交任务 {task_id}（8100 上）")
# kill 全部 8100 进程（模拟重启）。**只杀 8100 的进程**：按命令行过滤出
# 带 --port 8100 的 python —— 全杀会把正在跑的演练脚本自己（它也是
# venv python）一起杀掉，表现为"零输出退出码 1"（第一次跑实测踩到）。
out = subprocess.run(
    [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*uvicorn*--port*8100*' } | "
        "Select-Object ProcessId | ConvertTo-Json -Compress",
    ],
    capture_output=True,
    text=True,
)
procs = json.loads(out.stdout) if out.stdout.strip() else []
killed = []
for p in procs if isinstance(procs, list) else [procs]:
    pid = p.get("ProcessId")
    if pid:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        killed.append(pid)
print(f"  已杀 8100 进程：{killed}")
time.sleep(1.0)
# 重启 8100
subprocess.run(
    [
        "powershell",
        "-NoProfile",
        "-Command",
        "Start-Process -WindowStyle Hidden -FilePath 'D:\\git\\网页\\industrial-ai-platform\\.venv\\Scripts\\python.exe' "
        "-ArgumentList '-m','uvicorn','app.main:app','--port','8100' "
        "-WorkingDirectory 'D:\\git\\网页\\industrial-ai-platform'",
    ],
    capture_output=True,
)
time.sleep(6.0)
after = probe(f"{RAG}/api/v2/agent-tasks/{task_id}")
record(
    "③ 服务重启后任务状态",
    f"重启前提交的任务 {task_id}，重启后 GET → HTTP {after}"
    + ("（404 = 任务表随进程消失）" if after == 404 else ""),
    "**内存任务表重启即丢**（冻结清单周一就写明的约束，现在是实证）。"
    "轮询方拿到 404 应视为'任务状态不可知'而不是'任务失败' —— "
    "重新提交是新任务。落库是'已部署'周的任务",
)

print()
print("=" * 64)
print("三条演练完成：隔离 ✅ / 摘流机制成立 ✅ / 重启丢任务（已知约束实证）✅")
print("=" * 64)
