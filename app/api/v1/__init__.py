"""API v1 路由汇总。

`include_router` 集中在这里，`main.py` 只挂一次 —— 加 v2 就是加一个平级包，
不用回 main 里翻路径。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.extract import router as extract_router

router = APIRouter()
router.include_router(extract_router)

__all__ = ["router"]
