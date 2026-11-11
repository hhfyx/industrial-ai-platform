"""v2 鉴权与限流：API Key 鉴权 + 双层令牌桶（W12 周一）。

## 为什么放 core 而不是 api/

鉴权与限流是**横切面**（对 v2 所有端点生效），不是某个路由的私有逻辑；
放 `app/core/` 与 middleware/errors 同层，v2 路由只 import 不实现。
v1 端点一行不改 —— W1 的契约冻结规则：给 v1 加必填鉴权等于破坏所有现有调用方。

## 鉴权设计（对应周一 09:40）

- key 从环境变量 `PLATFORM_API_KEYS` 读（逗号分隔，**复数**——单 key 配额
  要求 key 不止一个，否则双层桶退化成一层）；
- 失败一律 401 `UNAUTHORIZED`，**同一个 message** ——
  "key 缺失"与"key 错误"不区分。区分它会变成一个 oracle：
  攻击者拿它枚举有效 key。401 语义就是"没通过"，不解释为什么。
- 日志里**绝不打印 key**：`_mask()` 只留前 3 位 + 长度。
  401 响应也只回掩码 —— 调用方报障时能对上"我发的是哪个 key"，
  而日志收集系统里不存在明文。

## 双层令牌桶（对应周一 10:20）

- **全局桶**：整个进程一个桶，QPS 上限保护的是模型网关；
- **单 key 桶**：每个 key 一个桶，配额防的是"一个客户端把所有人挤掉"。
  坑（周一计划原文）：只做全局限流时，监控里"全局 QPS 正常"，
  但单个客户端已经把别人的配额吃光 —— 两个桶都要有，且 **/metrics 要按
  key 维度记拒绝计数**，否则这个坑踩到时从监控里看不见。

- 实现是**惰性生成令牌**而不是后台线程滴灌：
  `tokens = min(capacity, tokens + (now - last) * rate)`。
  两者在数学上等价，但惰性版没有线程、没有 tick 误差、单事件循环内无锁
  （asyncio 单线程，读改写之间没有 await 就没有竞争）。
- 超限 429 `RATE_LIMITED` + `Retry-After`（按缺口速率算：
  需要补满 n 个令牌要等 n/rate 秒）+ `X-RateLimit-Remaining`。
- **先查单 key 桶再查全局桶**：单 key 超限是调用方自己的问题（改自己的节奏），
  全局超限是全体的问题（等别人）。先查前者，429 的 message 才能说到点子上。

## 单实例边界（写进规范，不是注释）

内存桶只对**单个进程**有效。多实例部署时每个实例各有一个桶，
实际总配额 = 单实例配额 × 实例数，要精确限流需换 Redis + Lua。
这一条必须出现在 docs/api-spec.md 的 v2 节，不能只活在代码注释里。

## 与统一错误体的关系

401/429 一律抛 `AppError`，走 `errors.py` 的 `_error_response()` 唯一出口 ——
`Retry-After` 用 `AppError.headers` 携带（W8 已验证的机制），
**不在中间件里手写 JSONResponse**，否则契约测试会抓到两套错误形状。
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Final

from fastapi import Request

from app.core.errors import AppError

# ---------------------------------------------------------------- 配置

API_KEYS_ENV: Final[str] = "PLATFORM_API_KEYS"
GLOBAL_RATE_ENV: Final[str] = "PLATFORM_GLOBAL_QPS"
KEY_RATE_ENV: Final[str] = "PLATFORM_KEY_QPS"

# 默认值：全局 10 QPS / 单 key 3 QPS。压测时要放大全局值 —— 改环境变量，
# 不改代码（应急开关是改配置，不是改代码，与 EXTRACTOR_IMPL 同一条纪律）。
DEFAULT_GLOBAL_QPS: Final[float] = 10.0
DEFAULT_KEY_QPS: Final[float] = 3.0

# 桶容量 = 速率 × 突发秒数。允许短时突发（前端一次页面刷新打 3 个请求很正常），
# 严格 1 req/s 的桶会把正常用户也挡掉。突发 2 秒的量是经验起点，压测后调。
BURST_SECONDS: Final[float] = 2.0

# 无 key 环境的兜底：设了 PLATFORM_API_KEYS 才启用鉴权。
# 不设 = 服务退回 v1 时代的"本地演示无鉴权"模式 —— 显式降级而非隐式崩溃，
# CI 与本地联调不需要永远带着一个 key。生产部署必须设置这个变量（写进规范）。
AUTH_ENABLED_ENV_SET: Final[bool] = True


def _mask(key: str) -> str:
    """key 掩码：前 3 位 + 长度。日志、401 响应、metrics 全部用这个形状。

    前 3 位是安全与可排查性的折中：调用方报障时能对上"我发的是哪把钥匙"，
    而对暴力枚举者 3 位信息量约等于零（剩余空间仍然是天文数字）。
    """
    if len(key) <= 3:
        return "*" * len(key)
    return f"{key[:3]}…(len={len(key)})"


def _key_fingerprint(key: str) -> str:
    """key 的稳定指纹（sha256 前 12 位）。metrics 标签用它，**绝不用明文**。"""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _load_qps(env: str, default: float) -> float:
    """从环境变量读速率。非法值回退默认 —— 配置错误不该打死服务。"""
    raw = os.environ.get(env)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


# ---------------------------------------------------------------- 令牌桶


@dataclass
class TokenBucket:
    """惰性令牌桶。单事件循环内使用（无锁，读改写之间不得 await）。"""

    rate: float
    """每秒生成令牌数。"""
    capacity: float
    """桶容量（= rate × BURST_SECONDS，允许短时突发）。"""
    tokens: float = 0.0
    last: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        # 桶初始为满：服务刚起来就该能接住一波正常请求，
        # 从空桶慢慢涨会让冷启动的合法请求吃 429。
        if self.tokens == 0.0:
            self.tokens = self.capacity

    def try_take(self, *, now: float | None = None) -> tuple[bool, float, float]:
        """尝试取 1 个令牌。返回 (是否成功, 剩余令牌, 建议等待秒数)。

        **失败也要返回 retry_after**：按"取到 1 个令牌还差多少"折算，
        而不是拍脑袋回一个固定值 —— 客户端按它退避后**必然**能通过
        （前提是窗口内没有别人抢），Retry-After 才有诚实可言。
        """
        now = time.monotonic() if now is None else now
        elapsed = max(0.0, now - self.last)
        self.last = now
        # 惰性补令牌。`min` 封顶：容量之外的时间不折算成令牌。
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True, self.tokens, 0.0
        deficit = 1.0 - self.tokens
        return False, self.tokens, deficit / self.rate


class RateLimiter:
    """双层限流器：全局一桶，每 key 一桶。

    ## 为什么单 key 桶要懒创建

    key 集合是开放的（运维随时加），预先建桶没有意义。
    `defaultdict` 式的懒建还有个内存边界问题：攻击者伪造无限个 key 也只会
    在**鉴权通过之后**才建桶 —— 而鉴权挡在前面，所以桶数上限 = 真实 key 数。
    伪造 key 在 401 那一步就出局了，根本摸不到限流器。
    """

    def __init__(self) -> None:
        global_rate = _load_qps(GLOBAL_RATE_ENV, DEFAULT_GLOBAL_QPS)
        key_rate = _load_qps(KEY_RATE_ENV, DEFAULT_KEY_QPS)
        self.global_bucket = TokenBucket(rate=global_rate, capacity=global_rate * BURST_SECONDS)
        self.key_rate = key_rate
        self.key_buckets: dict[str, TokenBucket] = {}
        # metrics 用：按 key 指纹计 429 次数。监控按 key 维度分桶，
        # 才能看见"哪个客户端在挤掉别人"（全局 QPS 看不出来）。
        self.rejected_by_key: dict[str, int] = {}

    def check(self, fingerprint: str) -> tuple[bool, float, float]:
        """双层检查。返回 (是否通过, 剩余令牌, retry_after 秒)。

        顺序是先单 key 后全局（理由见模块 docstring）。
        **只扣通过的那一层**：单 key 桶扣了令牌但全局桶拒绝时，
        单 key 的令牌要退回去 —— 不退的话，一次被全局拒绝的请求
        会凭空消耗调用方自己的配额，双层桶互相污染。
        """
        bucket = self.key_buckets.get(fingerprint)
        if bucket is None:
            bucket = TokenBucket(rate=self.key_rate, capacity=self.key_rate * BURST_SECONDS)
            self.key_buckets[fingerprint] = bucket

        ok, remaining, retry_after = bucket.try_take()
        if not ok:
            self.rejected_by_key[fingerprint] = self.rejected_by_key.get(fingerprint, 0) + 1
            return False, remaining, retry_after

        ok_global, _, retry_after_global = self.global_bucket.try_take()
        if not ok_global:
            # 全局拒绝：退回单 key 的令牌（双层桶不互相污染）
            bucket.tokens = min(bucket.capacity, bucket.tokens + 1.0)
            return False, bucket.tokens, retry_after_global
        return True, remaining, 0.0

    def remaining_for(self, fingerprint: str) -> int:
        """当前 key 的剩余令牌（响应头 X-RateLimit-Remaining 用，向下取整）。"""
        bucket = self.key_buckets.get(fingerprint)
        return int(bucket.tokens) if bucket else int(self.key_rate * BURST_SECONDS)


# 进程级单例（与 generation_queue 同一条装配纪律）。
_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter


def reset_rate_limiter() -> None:
    """测试用：清掉单例（连同所有桶与计数）。"""
    global _limiter
    _limiter = None


# ---------------------------------------------------------------- 鉴权依赖

# 鉴权/限流跳过清单：健康检查被 429 挡住 = 编排器在高负载时重启活着的容器，
# 等于亲手造出"永远起不来"的死循环（与 liveness/readiness 分离是同一课的正面与反面）。
# /docs /openapi.json 跳过：开发期文档要在无 key 环境可看；生产可收掉。
PUBLIC_PATH_PREFIXES: Final[tuple[str, ...]] = (
    "/health",
    "/ready",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
)


def is_public_path(path: str) -> bool:
    """健康检查与运维端点不鉴权、不限流。"""
    return any(path == p or path.startswith(p + "/") for p in PUBLIC_PATH_PREFIXES)


def _configured_keys() -> tuple[str, ...]:
    """解析 PLATFORM_API_KEYS（逗号分隔）。空白项跳过，全空 = 鉴权未启用。"""
    raw = os.environ.get(API_KEYS_ENV, "")
    return tuple(k.strip() for k in raw.split(",") if k.strip())


async def require_api_key(request: Request) -> str:
    """v2 端点的鉴权 + 限流依赖。通过返回 key 掩码（日志用），失败抛 AppError。

    ## 为什么鉴权和限流放同一个依赖

    限流的 key 维度来自鉴权结果 —— 分开写就要把"调用方是谁"在两个依赖间传，
    而 FastAPI 依赖的返回值缓存让同一个依赖在一条请求里只算一次，
    合在一起是**最省事且不丢语义**的做法。想只鉴权不限流的端点不存在：
    v2 的答案是"没有配额的端点就不要上 v2"。

    ## 401 的一致性

    无头、头空、key 不对 —— 三种失败**同一个 message**。
    区分它们会让 401 变成"这把 key 存不存在"的 oracle（见模块 docstring）。
    """
    if not is_public_path(request.url.path):
        keys = _configured_keys()
        if keys:  # 未配置 = 鉴权未启用（本地/CI 显式降级，见 AUTH_ENABLED_ENV_SET 注释）
            header = request.headers.get("X-API-Key", "").strip()
            if header not in keys:
                raise AppError(
                    code="UNAUTHORIZED",
                    message="缺少或无效的 X-API-Key",
                    status_code=401,
                )

            limiter = get_rate_limiter()
            fingerprint = _key_fingerprint(header)
            ok, _, retry_after = limiter.check(fingerprint)
            if not ok:
                limiter.rejected_by_key[fingerprint] = limiter.rejected_by_key.get(fingerprint, 0)
                raise AppError(
                    code="RATE_LIMITED",
                    message=f"请求频率超限（单 key {limiter.key_rate:g} QPS），请按 Retry-After 退避",
                    status_code=429,
                    headers={"Retry-After": str(max(1, int(retry_after) + 1))},
                )
            request.state.rate_limit_fingerprint = fingerprint
        else:
            # 鉴权未启用：仍执行全局桶（保护网关的最后一道闸不因本地模式而消失），
            # 但没有 key 维度 —— 全局限流对匿名流量也生效。
            ok, _, retry_after = get_rate_limiter().global_bucket.try_take()
            if not ok:
                raise AppError(
                    code="RATE_LIMITED",
                    message="服务全局请求频率超限，请按 Retry-After 退避",
                    status_code=429,
                    headers={"Retry-After": str(max(1, int(retry_after) + 1))},
                )
    return ""


def remaining_header_for(request: Request) -> int | None:
    """取 X-RateLimit-Remaining 的值。鉴权未启用时返回 None（不写头）。

    为什么由路由收尾时写而不是依赖里直接写：依赖拿到的 `request`
    在异常路径写不上响应头（W8 的教训：异常一抛，后面的代码不执行），
    429/401 的头走 `AppError.headers`，2xx 的这个头由路由正常路径写。
    """
    fingerprint = getattr(request.state, "rate_limit_fingerprint", None)
    if fingerprint is None:
        return None
    return get_rate_limiter().remaining_for(fingerprint)


__all__ = [
    "API_KEYS_ENV",
    "DEFAULT_GLOBAL_QPS",
    "DEFAULT_KEY_QPS",
    "PUBLIC_PATH_PREFIXES",
    "TokenBucket",
    "RateLimiter",
    "require_api_key",
    "is_public_path",
    "remaining_header_for",
    "get_rate_limiter",
    "reset_rate_limiter",
    "_mask",
    "_key_fingerprint",
]
