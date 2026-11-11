"""v2 契约测试：鉴权 / 双层限流 / 冷端点 / readiness（W12 周一 15:30–16:30）。

## 测什么（契约测试 = 错误码全覆盖 + 响应形状）

| 用例 | 验收点（周一计划） |
| --- | --- |
| 无 key / key 错 → 401 | 401 可验；**两种失败同一个 message**（不泄露 key 存在与否） |
| 合法 key → 200 + `X-RateLimit-Remaining` | 头可读，随请求递减 |
| 连发超配额 → 429 + `Retry-After` | 超限可复现；形状 = 统一错误体 4 字段 |
| 公开端点不带 key → 200 | /health /ready /metrics 绕过鉴权限流（探针不死循环） |
| 422 | 校验失败走统一错误体（与 v1 同形状） |
| 冷端点 200 + hits | 检索-only 可用、零模型字段 |
| readiness | 未就绪 503 `NOT_READY` / 构造后 200 `ready` |

## 离线纪律（conftest 同款）

限流桶是**进程级单例**，测试间必须 `reset_rate_limiter()` 隔离 ——
否则上一个测试打掉的令牌让下一个测试莫名其妙吃 429。
`PLATFORM_API_KEYS` 用 monkeypatch.setenv 设置（自动还原），
鉴权配置是**每次请求时读环境变量**的（require_api_key 里现读现判），
所以 setenv 后无需重启 app。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.security import reset_rate_limiter
from app.main import app

KEY_A = "test-key-aaaaaaaa"
KEY_B = "test-key-bbbbbbbb"
HEADERS_A = {"X-API-Key": KEY_A}


@pytest.fixture(autouse=True)
def _clean_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    """每个测试独立限流器 + 显式 key 集合。测试默认启用鉴权。"""
    reset_rate_limiter()
    monkeypatch.setenv("PLATFORM_API_KEYS", f"{KEY_A},{KEY_B}")
    monkeypatch.setenv("PLATFORM_KEY_QPS", "3")
    yield
    reset_rate_limiter()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------- 401：鉴权


class TestAuth:
    def test_missing_key_returns_401(self, client: TestClient) -> None:
        r = client.post("/api/v2/extract", json={"text": "合同编号：HT-1", "fields": ["合同编号"]})
        assert r.status_code == 401
        body = r.json()
        assert body["code"] == "UNAUTHORIZED"
        # 统一错误体 4 字段
        assert set(body.keys()) == {"code", "message", "request_id", "details"}
        assert r.headers.get("X-Request-ID") == body["request_id"]

    def test_wrong_key_returns_401_with_same_message(self, client: TestClient) -> None:
        """key 错与 key 缺失必须同 message —— 401 不能变成"这把 key 存在"的 oracle。"""
        r_missing = client.post("/api/v2/extract", json={"text": "x", "fields": ["日期"]})
        r_wrong = client.post(
            "/api/v2/extract",
            json={"text": "x", "fields": ["日期"]},
            headers={"X-API-Key": "wrong-key-cccccccc"},
        )
        assert r_missing.status_code == 401
        assert r_wrong.status_code == 401
        assert r_missing.json()["message"] == r_wrong.json()["message"]

    def test_401_does_not_leak_key_in_message(self, client: TestClient) -> None:
        r = client.post(
            "/api/v2/extract",
            json={"text": "x", "fields": ["日期"]},
            headers={"X-API-Key": "secret-value-should-not-appear-999999"},
        )
        assert "secret-value-should-not-appear" not in r.text

    def test_valid_key_passes_auth(self, client: TestClient) -> None:
        r = client.post(
            "/api/v2/extract",
            json={"text": "合同编号：HT-2026-001，甲方：张三", "fields": ["合同编号", "甲方"]},
            headers=HEADERS_A,
        )
        assert r.status_code == 200
        body = r.json()
        # 响应体与 v1 契约完全一致（4 字段 + data 键序与请求一致）
        assert body["data"] == {"合同编号": "HT-2026-001", "甲方": "张三"}
        assert 0.0 <= body["confidence"] <= 1.0

    def test_public_paths_skip_auth(self, client: TestClient) -> None:
        """健康检查/探针/metrics 不带 key 必须 200 —— 探针被 429 挡住 = 编排器
        在高负载重启活着的容器（liveness/readiness 分离的镜像面）。"""
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200
        # /ready 走 v2 前缀但也在白名单：可能 200（就绪）或 503（未就绪），
        # 但绝不能是 401/429 —— 那说明鉴权挡住了探针。
        r = client.get("/api/v2/ready")
        assert r.status_code in (200, 503)


# ---------------------------------------------------------------- 429：双层限流


class TestRateLimit:
    def test_key_burst_returns_429_with_retry_after(self, client: TestClient) -> None:
        """单 key 桶：3 QPS × 2s 突发 = 6 容量，连发 12 次必然超限。"""
        codes = []
        for _ in range(12):
            r = client.post(
                "/api/v2/extract",
                json={"text": "合同编号：HT-1", "fields": ["合同编号"]},
                headers=HEADERS_A,
            )
            codes.append(r.status_code)
        assert 200 in codes  # 前几个必须过（桶初始为满）
        assert 429 in codes  # 后面必须被限
        limited = client.post(
            "/api/v2/extract",
            json={"text": "合同编号：HT-1", "fields": ["合同编号"]},
            headers=HEADERS_A,
        )
        assert limited.status_code == 429
        body = limited.json()
        assert body["code"] == "RATE_LIMITED"
        assert set(body.keys()) == {"code", "message", "request_id", "details"}
        # Retry-After 必须存在且为正整数
        retry_after = limited.headers.get("Retry-After")
        assert retry_after is not None and int(retry_after) >= 1

    def test_remaining_header_decreases(self, client: TestClient) -> None:
        """X-RateLimit-Remaining 随成功请求递减。"""
        seen: list[int] = []
        for _ in range(3):
            r = client.post(
                "/api/v2/extract",
                json={"text": "合同编号：HT-1", "fields": ["合同编号"]},
                headers=HEADERS_A,
            )
            assert r.status_code == 200
            seen.append(int(r.headers["X-RateLimit-Remaining"]))
        assert seen == sorted(seen, reverse=True)
        assert seen[0] > seen[-1]

    def test_different_keys_have_independent_budgets(self, client: TestClient) -> None:
        """双层桶的核心：key A 打光自己的配额，key B 不受影响。
        只做全局限流的坑（周一计划原文）就是这一条测出来的反面。"""
        for _ in range(10):
            client.post(
                "/api/v2/extract",
                json={"text": "合同编号：HT-1", "fields": ["合同编号"]},
                headers=HEADERS_A,
            )
        r_b = client.post(
            "/api/v2/extract",
            json={"text": "合同编号：HT-1", "fields": ["合同编号"]},
            headers={"X-API-Key": KEY_B},
        )
        assert r_b.status_code == 200

    def test_429_body_does_not_contain_key(self, client: TestClient) -> None:
        """日志脱敏的响应面：429 体与头里都不能出现明文 key。"""
        for _ in range(12):
            client.post(
                "/api/v2/extract",
                json={"text": "x", "fields": ["日期"]},
                headers=HEADERS_A,
            )
        r = client.post(
            "/api/v2/extract", json={"text": "x", "fields": ["日期"]}, headers=HEADERS_A
        )
        assert r.status_code == 429
        assert KEY_A not in r.text


# ---------------------------------------------------------------- 422 / 冷端点 / readiness


class TestColdAndReady:
    def test_v2_extract_422_same_error_shape(self, client: TestClient) -> None:
        """校验失败与 v1 同一张错误码表（统一错误体不因 v2 而分叉）。"""
        r = client.post(
            "/api/v2/extract",
            json={"text": "", "fields": ["日期"]},
            headers=HEADERS_A,
        )
        assert r.status_code == 422
        body = r.json()
        assert body["code"] == "INVALID_INPUT"
        assert body["details"][0]["field"] == "text"

    def test_search_only_endpoint(self, client: TestClient) -> None:
        """冷端点：走真检索管线，命中内置语料，响应里没有模型字段。"""
        r = client.post(
            "/api/v2/search",
            json={"query": "抗压强度 判定", "k": 3},
            headers=HEADERS_A,
        )
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) == {"request_id", "hits", "elapsed_ms"}
        assert len(body["hits"]) >= 1
        hit = body["hits"][0]
        assert {"id", "source", "page", "score", "text"} <= set(hit.keys())
        assert "抗压强度" in hit["text"] or "复测" in hit["text"]

    def test_search_low_relevance_scores_for_unknown_query(self, client: TestClient) -> None:
        """无关查询：仍返回 top-k（Retriever 永远给 top-k，与 retrieve_spec_chunks
        同一语义），但分数显著低于命中查询 ——「没有」由调用方按分数阈值判断，
        冷端点如实给分数，不替调用方下结论（与 refusal 闸门分层同一条哲学）。"""
        r_unknown = client.post(
            "/api/v2/search",
            json={"query": "绝对不存在的量子术语xyzzy", "k": 2},
            headers=HEADERS_A,
        )
        r_known = client.post(
            "/api/v2/search",
            json={"query": "抗压强度 判定", "k": 2},
            headers=HEADERS_A,
        )
        assert r_unknown.status_code == 200
        assert r_known.status_code == 200
        top_unknown = r_unknown.json()["hits"][0]["score"]
        top_known = r_known.json()["hits"][0]["score"]
        assert top_known > top_unknown

    def test_search_hits_sorted_by_score_desc(self, client: TestClient) -> None:
        r = client.post(
            "/api/v2/search",
            json={"query": "含水量 限值 复测", "k": 5},
            headers=HEADERS_A,
        )
        assert r.status_code == 200
        scores = [h["score"] for h in r.json()["hits"]]
        assert scores == sorted(scores, reverse=True)

    def test_ready_reflects_dependency_state(self, client: TestClient) -> None:
        """readiness 随依赖单例状态翻转：未构造 → 503，构造后 → 200。

        通过 conftest 的 offline_extractor 已把 get_extractor 覆盖 ——
        但 dependency_overrides 不影响 cache_info（单例本身没构造），
        所以这里直接控制真实单例的构造状态来验证两种响应。
        """
        from app.core.deps import get_extractor, get_generator

        r = client.get("/api/v2/ready")
        # 干净进程里两个单例都没构造 → 503 NOT_READY
        if r.status_code == 503:
            assert r.json()["code"] == "NOT_READY"
            assert r.headers.get("Retry-After") is not None
        # 触发构造（fake 生成器 + 规则版抽取器都是毫秒级）
        get_generator()
        get_extractor()
        r2 = client.get("/api/v2/ready")
        assert r2.status_code == 200
        body = r2.json()
        assert body["status"] == "ready"
        assert body["checks"] == {"generator": True, "extractor": True}

    def test_rate_limit_status_endpoint(self, client: TestClient) -> None:
        r = client.get("/api/v2/rate-limit/status", headers=HEADERS_A)
        assert r.status_code == 200
        body = r.json()
        assert {"remaining", "rejected_by_this_key", "tracked_keys"} <= set(body.keys())


# ---------------------------------------------------------------- 令牌桶单元行为


class TestTokenBucketUnit:
    def test_lazy_refill(self) -> None:
        """惰性补令牌：时间过去 = 令牌回来，不用后台线程。"""
        from app.core.security import TokenBucket

        bucket = TokenBucket(rate=10.0, capacity=10.0)
        for _ in range(10):
            ok, _, _ = bucket.try_take(now=100.0)
            assert ok
        ok, _, retry = bucket.try_take(now=100.0)
        assert not ok
        assert retry == pytest.approx(0.1)  # 差 1 个令牌，10/s → 0.1s

        # 过 0.5 秒 → 补了 5 个令牌
        ok, remaining, _ = bucket.try_take(now=100.5)
        assert ok
        assert remaining == pytest.approx(4.0)  # 5 补充 - 1 取用

    def test_capacity_cap(self) -> None:
        """桶封顶：长时间空闲不会攒出无限突发。"""
        from app.core.security import TokenBucket

        bucket = TokenBucket(rate=2.0, capacity=4.0, tokens=4.0)
        ok, remaining, _ = bucket.try_take(now=1000.0)  # 空闲了很久
        assert ok
        assert remaining == pytest.approx(3.0)  # 不是 4 - 0 + 2000，而是封顶 4 后取 1
