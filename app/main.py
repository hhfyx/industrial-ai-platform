"""FastAPI application entry point.

只做接线，不含业务逻辑：

    1. FastAPI(...)                 实例化
    2. register_middlewares         请求上下文（request_id / 契约头）
    3. register_exception_handlers  统一错误体
    4. include_router               挂 /api/v1

启动：

    uv run uvicorn app.main:app --reload
"""

from fastapi import FastAPI

from app import __version__
from app.api.v1 import router as v1_router
from app.core.errors import register_exception_handlers
from app.core.middleware import register_middlewares

app = FastAPI(
    title="Industrial AI Platform",
    version=__version__,
    description=(
        "工业 AI 平台后端。\n\n"
        "**统一错误体**：所有非 2xx 都是 `{code, message, request_id, details}`。\n\n"
        "**契约冻结**：`POST /api/v1/extract` 的形状见 `docs/api-spec.md`，"
        "实现（规则版 → LLM）可换，契约不变。"
    ),
)

register_middlewares(app)
register_exception_handlers(app)
app.include_router(v1_router)


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness probe used by deployment and CI."""
    return {"status": "ok"}


__all__ = ["app"]
