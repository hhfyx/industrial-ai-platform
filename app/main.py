"""FastAPI application entry point.

只做接线，不含业务逻辑：

    1. configure_logging              装日志 handler（否则 logger.info 全被丢弃）
    2. FastAPI(...)                   实例化
    3. register_cors                  跨域（前端从 file:// 或另一个端口访问时需要）
    4. register_middlewares           请求上下文（request_id / 契约头）
    5. register_exception_handlers    统一错误体
    6. include_router                 挂 /api/v1
    7. mount_static                   挂 static/（前端的单页 UI）

启动：

    uv run uvicorn app.main:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.v1 import router as v1_router
from app.api.v2 import router as v2_router
from app.core.errors import register_exception_handlers
from app.core.logging import setup_from_settings
from app.core.middleware import register_middlewares

# 必须在任何业务代码之前装好 handler。
# 少了这一步，项目里所有 `logger.info(...)` 都会被静默丢弃 ——
# 排查时表现为"检测没生效"，实际是"日志没输出"。
setup_from_settings()

app = FastAPI(
    title="Industrial AI Platform",
    version=__version__,
    description=(
        "工业 AI 平台后端。\n\n"
        "**统一错误体**：所有非 2xx 都是 `{code, message, request_id, details}`。\n\n"
        "**契约冻结**：`POST /api/v1/extract` 的形状见 `docs/api-spec.md`，"
        "实现（规则版 → LLM）可换，契约不变。\n\n"
        "**v2 接口线**：`/api/v2/*` 需要 `X-API-Key` 鉴权并受双层限流"
        "（单 key 配额 + 全局 QPS），见 `docs/api-spec.md` v2 节。"
    ),
)

# ---------------------------------------------------------------- CORS
#
# **为什么需要**：前端页面可能来自
#   - `file://` 直接打开（origin 为 `null`）
#   - 另一个端口（如 Live Server 的 5500）
# 而 API 在 8000。浏览器会拦下跨域请求。
#
# **为什么放开 `allow_origins=["*"]`**：这是本地演示用的工具，
# 没有凭证、没有用户数据。生产环境必须收敛到具体域名 ——
# `*` 配合 `allow_credentials=True` 会被浏览器直接拒绝（规范不允许），
# 所以这里也不开 credentials，避免"看起来能配实际配不成"。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    # 让前端能读到耗时相关的响应头（需要在浏览器里看时有用）
    expose_headers=["X-Request-ID", "X-Extract-Model", "X-Elapsed-Ms"],
)

register_middlewares(app)
register_exception_handlers(app)
app.include_router(v1_router)
# v2（W12 周一）：鉴权 + 限流的新接口线。v1 端点一行不改 —— 契约冻结规则。
app.include_router(v2_router)


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness probe used by deployment and CI."""
    return {"status": "ok"}


@app.get("/metrics", tags=["ops"])
def metrics() -> dict[str, object]:
    """错误分类计数（W8 周日）+ 限流拒绝计数（W12 周一）。

    进程内计数器，单实例演示服务的轻量可观测——
    上 Prometheus 的时机是"有第二个实例"那天（见 logging.py 的说明）。

    限流计数按 key 指纹分桶：只看全局会漏掉"一个客户端挤掉所有人"的坑 ——
    全局 QPS 正常时单 key 的 429 可能已经打爆，key 维度才看得见。
    """
    from app.core.logging import error_counts, get_request_id
    from app.core.security import get_rate_limiter

    limiter = get_rate_limiter()
    return {
        "errors": error_counts(),
        # 当前进程最近一次请求的 id（诊断用：容器日志里 grep 它）
        "last_request_id": get_request_id(),
        # 限流：总量 + 按 key 指纹分桶（无明文 key）
        "rate_limit_rejections": {
            "total": sum(limiter.rejected_by_key.values()),
            "by_key_fingerprint": dict(limiter.rejected_by_key),
        },
    }


# ---------------------------------------------------------------- 静态前端
#
# `mount` 放在最后：StaticFiles 挂在 "/" 会兜住所有未匹配路径，
# 如果它先注册，后面的 API 路由就都被它抢走了。
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
if STATIC_DIR.is_dir():
    # html=True 让 "/" 自动返回 index.html
    app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")


__all__ = ["app"]
