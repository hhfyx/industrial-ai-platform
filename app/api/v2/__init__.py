"""API v2 路由汇总（W12 周一：规范 v2 · 鉴权 · 限流）。

v2 与 v1 平级（与 v1 包的 `__init__` 同一条接线纪律，`main.py` 只挂一次）。
v2 的三个交付：

1. `extract_v2`  —— 鉴权 + 双层限流的样板端点（请求/响应体与 v1 extract 相同）；
2. `cold`        —— /api/v2/search（零 LLM 检索）+ /ready（readiness）+ 配额状态；
3. 限流/鉴权契约 —— 401/429/`Retry-After`/`X-RateLimit-Remaining` 的行为定义
   在 `app/core/security.py`，规范文本在 `docs/api-spec.md` v2 节。

v1 端点一行不改：给 v1 加必填鉴权等于破坏所有现有调用方（api-spec 冻结规则）。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v2.cold import router as cold_router
from app.api.v2.extract_v2 import router as extract_v2_router

router = APIRouter()
router.include_router(extract_v2_router)
router.include_router(cold_router)

__all__ = ["router"]
