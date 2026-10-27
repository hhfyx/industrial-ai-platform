"""演示回放：让 `/api/v1/summarize/stream` 走**预置数据**而非真模型。

**为什么需要这个**：录 demo 时如果打真模型，每次画面都不一样 ——
摘要文本变、出字速度变、偶尔超时。录制结果不可复现，
而且录一次要真花钱跑一次 50 页文档。

但**前端代码路径必须继续保持真实**：它仍然要经过
`fetch` → `response.body.getReader()` → SSE 解析 → 逐字渲染。
所以回放只替换**服务端的产出来源**，不碰网络层、不碰前端。

实现方式：`DEMO_REPLAY=1` 环境变量开启。开启后端点走 `replay_stream()`，
按固定节奏产出与真实链路完全相同的事件序列 ——
同样的 `accepted` / `progress` / `delta` / `meta` / `[DONE]`。

**这本身也是有用的能力**，不只是为了录屏：
- 前端开发和联调不需要 API key
- 压测前端渲染性能时可以无限次重复
- 做 e2e 测试时不需要模型，测试才能进 CI
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

# 环境变量名。设成 1/true/yes 即开启。
DEMO_REPLAY_ENV = "DEMO_REPLAY"

# 回放用的摘要正文：**直接取自真实跑出来的结果**（50 页合同 PDF），
# 这样演示画面与真实效果一致，不是编出来的漂亮话。
# 长度约 240 字 —— 足够展示"逐字滚动 + 自动滚动"的完整观感。
DEMO_SUMMARY = (
    "该文档系统规定了合同第 1 至第 50 项交付物的验收标准及对应的付款与违约条款，"
    "共 50 项，整体结构完全一致，仅交付物编号、里程碑次数及违约金比例逐项递增。"
    "付款条件方面，甲方应在每一项交付物对应的第 1 至第 50 次里程碑验收完成后"
    "十五个工作日内完成付款。违约责任方面，若第 N 期交付延期，"
    "乙方按合同总额的 N‰ 承担违约责任，比例自第 1 期的 1‰ 起逐期递增至第 50 期的 50‰，"
    "其中第 11 至 20 期为 11‰ 至 20‰，第 21 至 30 期为 21‰ 至 30‰。"
)

# 节奏参数：与真模型的实测节奏接近（TTFT 约 1.5s，出字约 28 字/秒）。
# 出字速度比真实略快 —— 演示要在 20 秒内看完，慢速逐字会让 GIF 又大又长。
PROGRESS_START = 0.35
PROGRESS_STEP = 0.5
DELTA_START = 1.6
DELTA_STEP = 0.022


def is_replay_enabled() -> bool:
    import os

    return os.environ.get(DEMO_REPLAY_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


async def replay_stream(
    *,
    filename: str,
    page_count: int,
    total_chars: int,
    chunk_count: int,
    speed: float = 1.0,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """按固定节奏产出与真实链路同构的事件。

    `speed > 1` 可加速（测试用）。
    产出 `(kind, payload)`，与 `summarize_streaming` 的接口一致 —— 调用方无需分支。
    """

    def scaled(seconds: float) -> float:
        return seconds / speed

    # ① 文件已接收
    yield (
        "accepted",
        {
            "filename": filename,
            "pages": page_count,
            "chars": total_chars,
            "chunks": chunk_count,
            "warnings": [],
        },
    )
    await asyncio.sleep(scaled(0.1))

    # ② 分段进度
    for i in range(1, chunk_count + 1):
        label = f"第 {i} 段"
        yield ("progress", {"done": i, "total": chunk_count, "label": label})
        await asyncio.sleep(scaled(PROGRESS_STEP))

    # ③ 逐字输出（真实链路里这也是流式的）
    await asyncio.sleep(scaled(max(DELTA_START - PROGRESS_START - chunk_count * PROGRESS_STEP, 0)))
    for char in DEMO_SUMMARY:
        yield ("delta", {"text": char})
        await asyncio.sleep(scaled(DELTA_STEP))

    # ④ 元信息
    yield (
        "done",
        {
            "summary": DEMO_SUMMARY,
            "chunks": chunk_count,
            "api_calls": chunk_count + 1,
        },
    )


def replay_meta(chunk_count: int) -> dict[str, Any]:
    """回放模式下给前端的 meta（与真实链路字段一致）。"""
    return {
        "chunks": chunk_count,
        "api_calls": chunk_count + 1,
        "replay": True,
    }


def demo_sse_preview() -> str:
    """生成一段可打印的 SSE 预览（用于文档/排查，不依赖浏览器）。"""
    lines: list[str] = []

    def frame(payload: Any) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}"

    lines.append(frame({"accepted": {"filename": "sample50.pdf", "pages": 50, "chunks": 5}}))
    lines.append(frame({"progress": {"done": 1, "total": 5, "label": "第 1 段"}}))
    lines.append(frame({"delta": DEMO_SUMMARY[:20]}))
    lines.append(frame({"meta": replay_meta(5)}))
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


__all__ = [
    "DEMO_REPLAY_ENV",
    "DEMO_SUMMARY",
    "demo_sse_preview",
    "is_replay_enabled",
    "replay_meta",
    "replay_stream",
]
