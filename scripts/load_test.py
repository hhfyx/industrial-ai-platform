"""压测设计表（W12 周四 09:00–09:40）与执行入口。

## 设计总表

| 档 | 并发 | 流量构成（权重） | 时长 | 目的 |
| --- | --- | --- | --- | --- |
| A1 | 10 | 检索-only 100% | 3 min | 服务层天花板（冷启动基线） |
| A2 | 20 | 检索-only 100% | 3 min | 服务层天花板（中压） |
| A3 | 50 | 检索-only 100% | 3 min | 服务层天花板（高压，预期限流触发） |
| B1 | 10 | 完整链路为主（权重 7:2:1） | 4 min | 端到端基线（含 LLM 延迟） |
| B2 | 20 | 完整链路为主 | 4 min | 端到端 + 限流触发点观察 |

任务权重（周四计划原文：检索 7 : 完整 2 : 任务查询 1）：
- 检索-only：POST /api/v2/search（零 LLM，冷端点）——服务层瓶颈；
- 完整链路：POST /api/v2/extract（走 LLM）——端到端；
- 任务查询：GET /api/v2/agent-tasks/{id}（404 或真实任务）——轻路径。

## 热缓存策略（计划原文：先热缓存再压）

A 档开跑前先用 10 条 query 各打一次 /search（写入检索层缓存），
然后压测的 query 从**同一池子**随机取 —— 命中率接近稳态（反复问的
是同一批问题），测的是"缓存服务能力"而不是"冷 miss 风暴"。
冷 miss 的代价在 W12 周二的成本对比里已经单独测过（冷/热两轮）。

## 指标（周四计划原文）

P50/P95/P99、QPS、错误率、缓存命中率（从 /metrics 读）、
token 消耗（B 档，推算口径）、429 触发档位。

## 必答问题的落点

先压检索-only：完整链路的瓶颈几乎总在上游 LLM，混着压只能得到
"系统很慢"；拆开压才能把"我的问题"（服务层）与"上游的问题"（LLM）
分开 —— A 档测我的天花板，B 档测加上上游之后的真实形态。

跑法：
    python -X utf8 scripts/load_test.py --profile A3          # 检索-only 50 并发
    python -X utf8 scripts/load_test.py --profile B2          # 完整链路 20 并发
    python -X utf8 scripts/load_test.py --profile all         # 全部五档顺序跑
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = Path("eval/results/load")

# 五档定义（上面设计表的机器可读形态）
PROFILES: dict[str, dict] = {
    "A1": {"users": 10, "run_time": "3m", "tags": "search_only", "wait": 0.5},
    "A2": {"users": 20, "run_time": "3m", "tags": "search_only", "wait": 0.5},
    "A3": {"users": 50, "run_time": "3m", "tags": "search_only", "wait": 0.5},
    "B1": {"users": 10, "run_time": "4m", "tags": "", "wait": 1.0},
    "B2": {"users": 20, "run_time": "4m", "tags": "", "wait": 1.0},
}

HOST = "http://127.0.0.1:8100"


def warm_cache(host: str, key: str) -> None:
    """热缓存：10 条 query 各打一次 /search（写检索层缓存）。

    用 urllib 而不是 locust 环境 —— 这一步在 locust 起来之前完成。
    """
    import urllib.request

    queries = [f"压测预热第{i}批合同条款" for i in range(10)]
    ok = 0
    for q in queries:
        req = urllib.request.Request(
            f"{host}/api/v2/search",
            data=json.dumps({"query": q, "k": 3}).encode(),
            headers={"X-API-Key": key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=10)
            ok += 1
        except Exception as exc:  # noqa: BLE001 - 预热失败只警告，压测继续（会变成冷 miss 形态）
            print(f"  预热失败 {q[:10]}…: {exc}")
    print(f"  缓存预热：{ok}/10 条写入检索层缓存")


def run_profile(profile: str, host: str, key: str) -> Path:
    """跑一档 locust（headless），落 CSV 与 JSON 汇总。"""
    spec = PROFILES[profile]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_prefix = RESULTS_DIR / f"load-{profile}"

    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        "scripts/locustfile.py",
        "--headless",
        "-u",
        str(spec["users"]),
        "-r",
        "5",  # 每秒起 5 个用户（爬坡，避免瞬时打满）
        "--run-time",
        spec["run_time"],
        "--csv",
        str(csv_prefix),
        "--json",
        str(RESULTS_DIR / f"load-{profile}.json"),
        "--only-summary",
        "--host",
        host,
        profile,
    ]
    # locust 不认未知参数：key/profile 通过 LOCUST_BENCH_* 环境变量传递
    # （locustfile.py 的 init listener 从环境读）—— 比魔改 argv 可靠。
    env = {**os.environ, "LOCUST_BENCH_KEY": key, "LOCUST_BENCH_PROFILE": profile}
    print(
        f"\n=== 档 {profile}: users={spec['users']} time={spec['run_time']} tags={spec['tags'] or 'all'} ==="
    )
    started = time.perf_counter()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env
    )
    elapsed = time.perf_counter() - started
    # locust 的 --only-summary 摘要打 stdout，原样落档
    (RESULTS_DIR / f"load-{profile}.log").write_text(
        proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8"
    )
    print(f"  完成（{elapsed:.0f}s），退出码 {proc.returncode}")
    # 摘要的关键行直接打出来
    for line in proc.stdout.splitlines():
        if line.strip().startswith(("Aggregated", "POST", "GET", "Type")) or "Aggregated" in line:
            print(f"  {line}")
    return RESULTS_DIR / f"load-{profile}.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="all", choices=[*PROFILES.keys(), "all"])
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--key", default="bench-key-0000000000")
    args = parser.parse_args()

    profiles = list(PROFILES.keys()) if args.profile == "all" else [args.profile]
    print(f"压测计划：{profiles} → {args.host}")

    for profile in profiles:
        if PROFILES[profile].get("tags") == "search_only":
            print("  热缓存…")
            warm_cache(args.host, args.key)
        run_profile(profile, args.host, args.key)

    print(f"\n产物：{RESULTS_DIR}/（json 汇总 + csv 明细 + log 摘要）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
