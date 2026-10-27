"""`POST /api/v1/summarize/stream` —— 上传文档并流式返回摘要。

**这个端点是"本周主交付"的收口**：把 loader（10 月 31 日 09:00–10:30）、
map-reduce（10:30–11:20）、SSE（周二/周三的积累）串成一条真实可用的链路。

## 为什么是 multipart 上传 + SSE

- 上传必须 POST + multipart（因为要传文件），**这正是 EventSource 做不到的**
  —— 所以前端只能用 `fetch` + `ReadableStream` 手工解 SSE。
- 响应是 SSE，因为摘要要逐字出现，用户不能等 30 秒。

## 事件协议

    data: {"progress": {"done": 1, "total": 9, "label": "第 1–6 页"}}
    data: {"delta": "该文档"}                       ← 汇总阶段的逐字输出
    data: {"meta": {"chunks": 9, "api_calls": 10}}
    data: [DONE]

**进度和正文在同一个流里**，这是刻意的：50 页文档的 map 阶段要跑几十秒，
那期间如果一个事件都没有，前端只能干等 —— 用户会以为卡死了。
把 `progress` 混进流里，前端就能显示"正在总结第 3/9 段"。

## 错误处理的两个阶段

1. **流开始前**（文件格式、大小、解析失败）：可以返回正常的 JSON 错误体
   （422/415），前端能读到 `{code, message, ...}`。
2. **流开始后**（模型出错）：状态码已经是 200，只能 `data: {"error": ...}`。

这个分界由 `StreamingResponse` 决定：**第一个 yield 之前**都还能改状态码。
所以文件读取/解析必须放在**生成器之前** —— 放进去就晚了。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.core.llm.client import get_llm_client
from app.core.sse import DisconnectWatcher
from app.rag.loaders import SUPPORTED_EXTENSIONS, load_document
from app.rag.replay import is_replay_enabled, replay_stream
from app.rag.summarize import chunk_documents, summarize_streaming

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["summarize"])

SSE_MEDIA_TYPE = "text/event-stream"
DONE_MARKER = "[DONE]"

# 上传大小上限：20 MB。够放一份几百页的报告，又能挡住"传了个 2GB 视频"。
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

PING_INTERVAL = 15.0

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    # nginx 反代必须关缓冲，否则流式会被攒成一坨（详见 docs/sse.md）
    "X-Accel-Buffering": "no",
}

# 允许的扩展名（从 loader 注册表推导，避免两处漂移）
ALLOWED_EXT = set(SUPPORTED_EXTENSIONS)


def sse_frame(payload: Any) -> str:
    """`data: {json}` + 双换行。"""
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {body}\n\n"


def done_frame() -> str:
    return f"data: {DONE_MARKER}\n\n"


@router.post(
    "/summarize/stream",
    summary="上传文档并流式返回摘要",
    description=(
        "multipart 上传（字段名 `file`），支持 PDF / DOCX / Markdown。\n\n"
        "**为什么前端的 EventSource 用不了**：EventSource 只支持 GET 且不能设请求头，"
        "而这里必须 POST 上传文件。所以前端要用 `fetch` + `ReadableStream` 手工解 SSE。\n\n"
        "事件：`progress`（分段进度）→ `delta`（摘要逐字）→ `meta` → `[DONE]`。\n"
        "流开始后的错误以 `data: {error}` 帧返回（此时状态码已是 200，无法再改）。"
    ),
    response_class=StreamingResponse,
    responses={
        415: {"description": "文件格式不支持"},
        413: {"description": "文件过大"},
        422: {"description": "文档无法解析或无文本"},
    },
)
async def summarize_stream(
    request: Request,
    file: UploadFile = File(..., description="待摘要的文档"),  # noqa: B008 - FastAPI 依赖注入的标准写法
    chunk_tokens: int = 800,
) -> StreamingResponse:
    """上传 → 解析 → map-reduce → 流式摘要。

    **解析放在生成器之前**：这样"格式不支持""文件太大""解析失败"都能返回
    正常的 HTTP 状态码 + JSON 错误体，前端能结构化处理。
    一旦进了生成器，就只能靠 SSE 帧报告错误了。
    """
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXT:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"不支持的文件格式 {suffix or '(无后缀)'}；支持：{sorted(ALLOWED_EXT)}",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"文件过大（{len(content) // 1024 // 1024} MB），上限 {MAX_UPLOAD_BYTES // 1024 // 1024} MB",
        )
    if not content:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="文件为空")

    # 写到临时目录再交给 loader：pypdf / python-docx 都需要真实路径
    import tempfile

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            result = await asyncio.to_thread(load_document, tmp_path)
        except ValueError as exc:
            # loader 的"真失败"（坏文件/打不开）→ 422，让前端能显示原因
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc

        # loader 内部用的是临时文件名（tmpXXXX.pdf），替换成用户上传的名字 ——
        # 摘要和前端要显示原名，临时名对用户毫无意义。
        # Document 是 frozen 的，用 dataclasses.replace 造新对象而不是就地改。
        from dataclasses import replace

        result.documents = [replace(doc, source=filename) for doc in result.documents]

    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    if not result.documents:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="文档没有可提取的文本（可能是扫描件，需要 OCR）",
        )

    settings = get_settings()
    llm = settings.llm
    client = get_llm_client(settings)
    extra: dict[str, Any] = {}
    if "glm" in llm.model.lower():
        extra["reasoning_effort"] = "none"

    chunks = chunk_documents(result.documents, chunk_tokens=chunk_tokens)

    async def event_stream() -> AsyncIterator[str]:
        stats_started = time.perf_counter()
        watcher = DisconnectWatcher(request)
        watcher.start()

        # 先发一个元信息帧：前端立刻知道"文件收到了、有几段"
        yield sse_frame(
            {
                "accepted": {
                    "filename": filename,
                    "pages": result.page_count,
                    "chars": result.total_chars,
                    "chunks": len(chunks),
                    "warnings": result.warnings[:3],
                }
            }
        )

        try:
            # 演示回放模式（DEMO_REPLAY=1）：用预置节奏产出与真实链路**同构**的事件。
            # 为什么要有这个：录屏时打真模型每次画面都不同（文本/速度/偶发超时），
            # 结果不可复现且每次真花钱。回放只替换"产出来源"，
            # 不碰网络层也不碰前端 —— 前端仍走 fetch + ReadableStream 的真代码路径。
            if is_replay_enabled():
                source = replay_stream(
                    filename=filename,
                    page_count=result.page_count,
                    total_chars=result.total_chars,
                    chunk_count=len(chunks),
                )
            else:
                source = summarize_streaming(
                    result.documents,
                    client=client,
                    model=llm.model,
                    chunk_tokens=chunk_tokens,
                    extra_request=extra,
                )

            async for kind, payload in source:
                if watcher.disconnected:
                    logger.info("summarize/stream 客户端断开，停止生成")
                    return

                if kind == "progress":
                    yield sse_frame({"progress": payload})
                elif kind == "delta":
                    yield sse_frame({"delta": payload["text"]})
                elif kind == "done":
                    yield sse_frame(
                        {
                            "meta": {
                                "chunks": payload.get("chunks", 0),
                                "api_calls": payload.get("api_calls", 0),
                                "elapsed_ms": int((time.perf_counter() - stats_started) * 1000),
                                "replay": is_replay_enabled(),
                            }
                        }
                    )
                    yield done_frame()
                    return

        except asyncio.CancelledError:
            logger.info("summarize/stream 被取消（客户端断开）")
            raise
        except Exception as exc:  # noqa: BLE001 - 流已开始，只能走数据帧
            logger.exception("summarize/stream 出错")
            yield sse_frame(
                {"error": {"code": "STREAM_ERROR", "message": f"{type(exc).__name__}: {exc}"}}
            )
            yield done_frame()
        finally:
            if watcher.disconnected:
                logger.info(
                    "summarize/stream 客户端断开（%.1fs）",
                    time.perf_counter() - stats_started,
                )
            await watcher.stop()

    return StreamingResponse(
        event_stream(),
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS,
    )


__all__ = [
    "ALLOWED_EXT",
    "DONE_MARKER",
    "MAX_UPLOAD_BYTES",
    "SSE_HEADERS",
    "router",
    "sse_frame",
]
