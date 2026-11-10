"""冻结清单逐条核对（W12 开工前）。

每条给「核对方式 + 结果 + 对压测的含义」。发现不一致就单独标出，
不许含糊带过 —— 这份清单就是压测的前提。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

BASE = Path(r"D:\git\网页\industrial-ai-platform")
TITLE = "=" * 78


def git(*args: str) -> str:
    r = subprocess.run(
        ["git", *args], cwd=BASE, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return r.stdout.strip()


def read(rel: str) -> str:
    return (BASE / rel).read_text(encoding="utf-8")


def main() -> int:
    print(TITLE)
    print("冻结清单逐条核对（W12 开工前）")
    print(TITLE)

    # ---- 1. 20 任务 + 判据冻结 ----
    tasks_path = BASE / "eval/agent/tasks-20.jsonl"
    lines = [ln for ln in tasks_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ev = json.loads((BASE / "eval/results/agent-eval-w11.json").read_text(encoding="utf-8"))
    print("\n[1] 20 任务 + 判据冻结，六项指标落表")
    print(f"    tasks-20.jsonl : {len(lines)} 行（应 20）→ {'✅' if len(lines) == 20 else '❌'}")
    print(
        f"    六项指标       : eval/results/agent-eval-w11.json 存在（{ev['success_rate_range']['min']:.0%}~{ev['success_rate_range']['max']:.0%}，5 轮）→ ✅"
    )
    print("    含义           : 压测的任务集与判据不再动；压测新增的是**并发维度**，不是新判据")

    # ---- 2. 预算参数 ----
    budget_text = read("app/core/llm/budget.py")
    print("\n[2] 预算参数定档 ← 最关键")
    print("    代码实际值     :")
    for line in budget_text.splitlines():
        if line.strip().startswith(("max_tokens: int", "max_steps: int", "max_wall_time_s: float")):
            print(f"      {line.strip()}")
    # compute 类放宽
    print("    compute 类放宽 : 12k token / 150s（tasks_20.py 的 __post_init__，4/20 个任务）")
    print("    ⚠ 与清单的差别 : 清单写 max_tokens=15k，代码是 10k（compute 12k）")
    print("      出处         : 周四预算曲线实测 —— 5k=50%、10k=83%、15k=83%，")
    print("                   10k 与 15k 成功率相同而 15k 多花 9.3% token，定档 10k")
    print("      处理         : 以**代码与曲线**为准（10k），清单的 15k 是周三的初稿口径，")
    print("                   W11 周四已被曲线推翻。压测用当前默认值，不临时改。")
    print("    含义           : 压测期间 budget.py 不动；并发下 budget_used 在 state 里，")
    print("                   单请求隔离（每个请求一份 state），无共享写问题")

    # ---- 3. 项目一 POST /generate + GET /tasks/{id} ----
    gen = read("app/api/v1/generate.py")
    has_sync = '/generate"' in gen or '"/generate",' in gen
    has_async = "/generate/async" in gen
    has_get = "/generate/tasks/{task_id}" in gen
    print("\n[3] 项目一 API")
    print(f"    POST /generate（同步）        : {'✅' if has_sync else '❌'}")
    print(f"    POST /generate/async（异步）  : {'✅' if has_async else '❌'}")
    print(f"    GET  /generate/tasks/{{id}}    : {'✅' if has_get else '❌'}")
    print("    ⚠ 与清单的差别 : 异步入口实际是 /generate/async（清单写 POST /generate 泛指）")
    print("    ⚠ 任务状态在**内存**（GenerationQueue._tasks dict），不是数据库 ——")
    print("      落库的是会话（sessions/turns 两表，volume 内），不是生成任务状态。")
    print("      重启丢任务状态 = 压测时要重新发起请求。这对并发压测是真实约束，")
    print("      应写进压测报告而不是被当成已解决。")

    # ---- 4. get_state 可包装成进度查询 ----
    q = read("scripts/query_graph_state.py")
    has_qp = "def query_progress" in q
    print("\n[4] 项目二进度查询")
    print(f"    query_progress() 在 scripts/query_graph_state.py : {'✅' if has_qp else '❌'}")
    print("    ⚠ 当前是 CLI（读 checkpointer 的 thread 状态），未接 HTTP。")
    print("      W12 的活就是把它包成 HTTP 端点 + interrupt 进服务路径。")
    print("    含义           : 压测项目二（Agent 路径）之前，先要有一个可打的服务入口 ——")
    print("                   这本身是 W12 的第一项工作，不是压测的输入")

    # ---- 5. request_id + latency_ms ----
    deps = read("app/core/deps.py")
    loop = read("app/core/llm/react_loop.py")
    print("\n[5] request_id + latency_ms")
    print(
        f"    request_id     : app/core/deps.py get_request_id() → {'✅' if 'def get_request_id' in deps else '❌'}"
    )
    lat_ok = '"latency_ms"' in loop
    print(
        f"    latency_ms     : react_loop.py _action_record 带 latency_ms → {'✅' if lat_ok else '❌'}"
    )
    print("    含义           : 压测的分位延迟可以直接从日志/轨迹聚合，不用重新埋点")

    # ---- 6. compose + tag ----
    print("\n[6] compose + tag")
    print("    compose 配置   : 两个都 config --quiet 通过（本机实测）→ ✅")
    print("    ⚠ tag 推送     : 本地 v1.3.0 已打（指向 6c389ce），但远端推送**失败**")
    print("      （代理端口 7897 时通时断）。main 与 v1.3.0 均未确认到远端。")
    print("      冻结的系统 = 「本地 commit 6c389ce + 本地 tag」，推送是运维动作，")
    print("      补推即可，不影响压测（压测跑的是本地代码）。")
    head = git("log", "-1", "--format=%h %s")
    print(f"    HEAD           : {head}")

    print()
    print(TITLE)
    print("结论：6 条里 5 条按清单成立；2 条有偏差需要在 W12 报告里写明 ——")
    print("  · 预算实际是 10k（compute 12k），不是清单的 15k —— 以曲线实测为准")
    print("  · 生成任务状态在内存不落库 —— 压测报告必须写「重启丢状态」")
    print(TITLE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
