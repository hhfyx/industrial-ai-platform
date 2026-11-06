"""压测：10 并发 query（W8 周六 16:30 格）。

容器内运行：docker exec industrial-ai-api python /app/scripts/stress_test.py
统计 P50 / P95 / 错误率 / 瓶颈段判定（上游限流 vs 本地 CPU）。
"""

import asyncio
import statistics
import sys
import time

sys.path.insert(0, "/app")

from app.core.config import get_settings  # noqa: E402
from app.core.llm.client import get_llm_client  # noqa: E402
from app.rag.pipeline import build_pipeline  # noqa: E402

QUESTIONS = [
    "缺陷类型有哪些",
    "数据量变化",
    "训练损失",
    "基座模型是什么",
    "标注流程",
    "预处理步骤",
    "切片尺寸",
    "数据增强",
    "损失曲线",
    "训练进度",
]


async def one(p, client, model, q, out, i):
    t0 = time.perf_counter()
    try:
        ans = await p.answer(q, client=client, model=model)
        out[i] = (time.perf_counter() - t0, False, ans.refused)
    except Exception:  # noqa: BLE001
        out[i] = (time.perf_counter() - t0, True, False)


async def main() -> None:
    p = build_pipeline(backend="chroma")
    client = get_llm_client(get_settings())
    model = get_settings().llm.model
    out: list = [None] * 10
    wall0 = time.perf_counter()
    await asyncio.gather(*[one(p, client, model, q, out, i) for i, q in enumerate(QUESTIONS)])
    wall = time.perf_counter() - wall0
    times = sorted(d for d, e, _r in out)
    errors = sum(1 for _, e, _ in out if e)
    refusals = sum(1 for _, _, r in out if r)
    print(f"wall={wall:.1f}s errors={errors} refusals={refusals}")
    print(
        f"P50={statistics.median(times) * 1000:.0f}ms "
        f"P95={times[int(len(times) * 0.95) - 1] * 1000:.0f}ms "
        f"min={times[0] * 1000:.0f}ms max={times[-1] * 1000:.0f}ms"
    )
    for (d, e, r), q in zip(out, QUESTIONS, strict=True):
        tag = "ERR" if e else ("REF" if r else "OK ")
        print(f"  {tag} {d * 1000:6.0f}ms  {q}")


if __name__ == "__main__":
    asyncio.run(main())
