"""locust 压测脚本（W12 周四 09:40–10:40）。

## 任务权重（周四计划原文：检索 7 : 完整 2 : 任务查询 1）

`@task(7)` 检索-only / `@task(2)` 完整链路 / `@task(1)` 任务查询。
`--tags search_only` 只跑检索类（A 档），不加标签跑全权重（B 档）。

## 每请求带 API Key

key 从 `--key` 参数进 environment（locust 的自定义参数机制）；
`PLATFORM_API_KEYS` 里要有它 —— 压测专用 key，与开发 key 分开，
metrics 的按 key 指纹计数才能把压测流量与手工流量分开。

## 断言状态码分布（计划原文）

`EventHook` 里对每类请求统计 2xx/4xx/429 占比，`test_stop` 时汇总打印 ——
"状态码分布"不是靠事后翻 CSV，是压测产物的第一等公民。

## 与服务端的约定

- 检索 query 从预热池取（`WARM_QUERIES` 与 load_test.py 的 warm_cache
  同源）—— 命中率接近稳态，测的是缓存服务能力（设计表 §热缓存）；
- wait_time 固定小间隔（0.5s/1.0s）：固定到达率让 QPS 可预期，
  constant_pacing 才是"并发数×节奏"的可控实验；
- 429 是**预期内的合法响应**（限流档位观察点），不标 failure ——
  把 429 记成 failure 会让错误率混入"压得太狠"，分不清背压与故障。
"""

from __future__ import annotations

import random

from locust import HttpUser, between, events, task

# 预热池：与 load_test.py 的 warm_cache 同源（命中缓存的 query 池）
WARM_QUERIES = [f"压测预热第{i}批合同条款" for i in range(10)]
# 冷查询池：专门打 miss 的 query（观察"缓存未命中时服务层多贵"）
COLD_QUERIES = [f"冷查询液位传感器第{i}号规范条款" for i in range(50)]

# 状态码分布统计（跨用户全局，events 级共享）
_status_counts: dict[str, dict[int, int]] = {}
# 每请求分段耗时（服务端 X-Elapsed-Ms 头 + 客户端总耗时）
_elapsed_server: list[int] = []


def _record(name: str, status_code: int) -> None:
    bucket = _status_counts.setdefault(name, {})
    bucket[status_code] = bucket.get(status_code, 0) + 1


@events.init.add_listener
def _on_init(environment, **kwargs) -> None:  # noqa: ANN001 - locust 签名
    """从 LOCUST_BENCH_KEY 环境变量取压测 key（load_test.py 传入）。

    locust 不认未知 CLI 参数（实测 `--key` 直接报 unrecognized arguments），
    环境变量是它官方支持的传参方式。
    """
    import os

    environment.api_key = os.environ.get("LOCUST_BENCH_KEY", "bench-key-0000000000")
    print(f"[locust] api key 前缀: {environment.api_key[:5]}…")


@events.test_stop.add_listener
def _on_test_stop(environment, **kwargs) -> None:  # noqa: ANN001
    """压测结束：打印状态码分布（断言的落点）。"""
    print("\n===== 状态码分布 =====")
    total_429 = 0
    total_all = 0
    for name, bucket in sorted(_status_counts.items()):
        parts = ", ".join(f"{code}×{cnt}" for code, cnt in sorted(bucket.items()))
        print(f"  {name}: {parts}")
        total_429 += bucket.get(429, 0)
        total_all += sum(bucket.values())
    if total_all:
        print(f"  429 占比: {total_429}/{total_all} = {total_429 / total_all:.1%}")
    if _elapsed_server:
        _elapsed_server.sort()
        p50 = _elapsed_server[len(_elapsed_server) // 2]
        p95 = _elapsed_server[int(len(_elapsed_server) * 0.95)]
        print(f"  服务端 elapsed_ms（X-Elapsed-Ms 头）：P50={p50}ms P95={p95}ms")


class BenchUser(HttpUser):
    """压测虚拟用户。wait 用 between 固定区间 —— 到达率可控。

    ## --tags 与权重为 0 的任务

    `--tags search_only` 过滤后，BenchUser 里只剩 `@task(0)` 的 search_cold
    和 `@task(0)` 的 submit_task —— 权重 0 不等于"不注册"，locust 的 tags
    过滤发生在任务选择之前，过滤后类里没有任何正权重任务就会炸
    "No tasks defined"。所以 A 档不用 tags 过滤，改用 **profile 切换**：
    `LOCUST_BENCH_PROFILE=A*` 时只执行 search_warm（其余任务直接 return）。
    这样一个 User 类服务两档，权重逻辑在任务内部 —— tags 机制对
    "0 权重任务也要参与过滤"这个边界不友好，绕开它。
    """

    wait_time = between(0.4, 0.6)

    def on_start(self) -> None:
        self.api_key = self.environment.api_key
        import os

        # profile：A* = 检索-only（B 档跑全权重）
        self.profile = os.environ.get("LOCUST_BENCH_PROFILE", "")
        self.headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
        self.cold_cursor = 0

    def _post(self, path: str, body: dict, name: str) -> None:
        with self.client.post(
            path, json=body, headers=self.headers, name=name, catch_response=True
        ) as resp:
            _record(name, resp.status_code)
            server_ms = resp.headers.get("X-Elapsed-Ms")
            if server_ms and server_ms.isdigit():
                _elapsed_server.append(int(server_ms))
            if resp.status_code == 429:
                # 429 = 限流背压，预期内：不标 failure（见模块 docstring）
                resp.success()
            elif resp.status_code >= 500:
                resp.failure(f"server error {resp.status_code}")

    # ------------------------------------------------ 检索-only（权重 7）

    @task(7)
    def search_warm(self) -> None:
        """检索-only：预热池 query（缓存命中路径为主）。"""
        q = random.choice(WARM_QUERIES)
        self._post("/api/v2/search", {"query": q, "k": 3}, "POST /search (warm)")

    @task(0)
    def search_cold(self) -> None:
        """冷查询（默认关 —— 冷 miss 形态单独测，见 W12 周二成本对比）。"""
        q = COLD_QUERIES[self.cold_cursor % len(COLD_QUERIES)]
        self.cold_cursor += 1
        self._post("/api/v2/search", {"query": q, "k": 3}, "POST /search (cold)")

    # ------------------------------------------------ 完整链路（权重 2）

    @task(2)
    def extract_full(self) -> None:
        """完整链路：v2 extract（走 LLM 抽取）。A 档跳过（profile 切换）。"""
        if self.profile.startswith("A"):
            return
        texts = [
            "合同编号：HT-2026-042，甲方：华东制造，签订日期：2026-01-07，合同金额：¥860,000.00",
            "合同编号：HT-2026-043，甲方：北方重工，签订日期：2026-01-06，合同金额：¥1,250,000.00",
        ]
        self._post(
            "/api/v2/extract",
            {"text": random.choice(texts), "fields": ["合同编号", "甲方", "金额"]},
            "POST /extract (full)",
        )

    # ------------------------------------------------ 任务查询（权重 1）

    @task(1)
    def task_query(self) -> None:
        """任务查询：轮询形态。多数打不存在的 id（404 路径 = 轻路径）。
        A 档跳过。"""
        if self.profile.startswith("A"):
            return
        # 20% 概率打真实任务（若本轮有提交过），否则打不存在的 id
        tid = getattr(self, "last_task_id", None)
        if tid is None or random.random() < 0.8:
            tid = f"bench{random.randrange(10**12):012d}"
        name = "GET /agent-tasks (poll)"
        with self.client.get(
            f"/api/v2/agent-tasks/{tid}", headers=self.headers, name=name, catch_response=True
        ) as resp:
            _record(name, resp.status_code)
            if resp.status_code in (200, 404):
                resp.success()
            else:
                resp.failure(f"unexpected {resp.status_code}")

    @task(0)
    def submit_task_then_poll(self) -> None:
        """提交 Agent 任务（默认关：B 档的完整链路用 extract 代表；
        Agent 任务的真实并发行为在队列观察脚本里单独做）。"""
        if self.profile.startswith("A"):
            return
        with self.client.post(
            "/api/v2/agent-tasks",
            json={"instruction": "查询批次 B-2024-042 的抗压强度"},
            headers=self.headers,
            catch_response=True,
        ) as resp:
            _record("POST /agent-tasks (submit)", resp.status_code)
            if resp.status_code == 202:
                self.last_task_id = resp.json().get("task_id")
                resp.success()
            else:
                resp.failure(f"submit {resp.status_code}")
