"""API v1 路由汇总。

`include_router` 集中在这里，`main.py` 只挂一次 —— 加 v2 就是加一个平级包，
不用回 main 里翻路径。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.chat import router as chat_router
from app.api.v1.extract import router as extract_router
from app.api.v1.stream import router as stream_router

router = APIRouter()
router.include_router(extract_router)
# 流式是**新增**端点：/api/v1/extract 未改动，v1 契约保持不变（见 docs/api-spec.md §4）
router.include_router(stream_router)
# 聊天流式是另一条独立接口线：抽取走结构化契约，聊天走开放对话，
# 两者的帧格式与终止符刻意不同（见 chat.py 的模块 docstring）
router.include_router(chat_router)

__all__ = ["router"]
