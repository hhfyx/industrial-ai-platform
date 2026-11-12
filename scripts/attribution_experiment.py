"""归因实验（W12 周四 16:40）：两处优化各贡献多少 —— 一次一处，分批测。

矩阵（同条件 = 同并发 20、同时长 60s、同预热）：

| 组 | 响应瘦身（500→120 字） | 结果缓存接入 | 预期 |
| --- | --- | --- | --- |
| 基线（周四 A2） | 否 | 否 | P50=4ms P95=8ms QPS 27.1 |
| 仅瘦身 | 是 | 否 | P95 下降（序列化量减半） |
| 全开（周五 A2-opt） | 是 | 是 | QPS 上升 |

瘦身的开关 = 环境变量 SEARCH_SUMMARY_CHARS（默认 120，设 500 回基线）。
缓存开关 = 环境变量 SEARCH_CACHE_OFF=1（cold 端点跳过结果缓存）。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

sys.path.insert(0, ".")

PY = sys.executable
RESULTS = []


def run_combo(
    name: str, summary_chars: int, cache_off: bool, users: int = 20, seconds: int = 60
) -> dict:
    import os

    env = {
        **os.environ,
        "PLATFORM_API_KEYS": "bench-key-0000000000,demo-key-12345678",
        "PLATFORM_GLOBAL_QPS": "200",
        "PLATFORM_KEY_QPS": "30",
        "SEARCH_SUMMARY_CHARS": str(summary_chars),
        "SEARCH_CACHE_OFF": "1" if cache_off else "",
        "PYTHONIOENCODING": "utf-8",
        "LOCUST_BENCH_KEY": "bench-key-0000000000",
        "LOCUST_BENCH_PROFILE": "A2",
    }
    # 起独立服务进程
    import urllib.request

    server = subprocess.Popen(
        [PY, "-m", "uvicorn", "app.main:app", "--port", "8200"],
        cwd=".",
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        # 等就绪
        for _ in range(30):
            try:
                urllib.request.urlopen("http://127.0.0.1:8200/health", timeout=2)
                break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        # 预热（10 条）
        for i in range(10):
            req = urllib.request.Request(
                "http://127.0.0.1:8200/api/v2/search",
                data=json.dumps({"query": f"压测预热第{i}批合同条款", "k": 3}).encode(),
                headers={"X-API-Key": "bench-key-0000000000", "Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
        # locust 60s
        proc = subprocess.run(
            [
                PY,
                "-m",
                "locust",
                "-f",
                "scripts/locustfile.py",
                "--headless",
                "-u",
                str(users),
                "-r",
                "5",
                "--run-time",
                f"{seconds}s",
                "--only-summary",
                "--host",
                "http://127.0.0.1:8200",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        # 解析 Aggregated 行
        qps = p50 = p95 = None
        for line in proc.stderr.splitlines():
            if "Aggregated" in line:
                # 列：type name reqs fails avg min max med req/s fail/s
                cells = [c.strip() for c in line.split() if c.strip()]
                # cells: Aggregated, reqs, fails(x%), avg, min, max, med, req/s, fails/s
                qps = cells[7]
        for line in proc.stderr.splitlines():
            if "Aggregated" in line and "%" in line:
                cells = [c.strip() for c in line.split() if c.strip()]
                # 分位表：Aggregated 50% 66% ... # reqs
                # cells[0]=Aggregated, 后面是各分位值
                nums = [c for c in cells[1:] if c.replace(",", "").isdigit()]
                if len(nums) >= 6:
                    p50, p95 = nums[0], nums[5]
        RESULTS.append({"name": name, "qps": qps, "p50": p50, "p95": p95})
        print(f"{name}: QPS={qps} P50={p50}ms P95={p95}ms")
        return {"name": name, "qps": qps, "p50": p50, "p95": p95}
    finally:
        server.terminate()
        server.wait(timeout=10)


def main() -> int:
    print("=" * 64)
    print("归因实验：同条件（20 并发 / 60s / 同预热）分批测两处优化")
    print("=" * 64)
    # 组1：基线复刻（500 字 + 缓存关）—— 验证能复现周四 A2 的形态
    run_combo("基线复刻（500字+无缓存）", summary_chars=500, cache_off=True)
    # 组2：仅瘦身
    run_combo("仅瘦身（120字+无缓存）", summary_chars=120, cache_off=True)
    # 组3：全开（120 字 + 缓存）= 周五正式形态
    run_combo("全开（120字+缓存）", summary_chars=120, cache_off=False)

    print()
    print("注意：SEARCH_SUMMARY_CHARS/SEARCH_CACHE_OFF 是本实验的开关，")
    print("正文实现里瘦身是常量 120、缓存恒开 —— 开关只为了归因矩阵能跑。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
