"""端到端冒烟：检索结果（带 overlap 重复）→ 去重 → 编号 → 生成 → 双向校验。

W8 周二验收的最后一环：引用层改动只影响展示，不影响链路行为。
运行：python -X utf8 scripts/smoke_citation_chain.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.generation.base import build_context, generate_answer
from app.rag.schema import Chunk


class FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


class FakeComp:
    def __init__(self):
        self.calls = []

    async def create(self, **kw):
        self.calls.append(kw)
        return FakeResp(
            "TTFT（Time To First Token）= 从发起请求到收到第一个非空 content 的时间 [1]。"
        )


class FakeClient:
    def __init__(self):
        self.completions = FakeComp()
        self.chat = type("Chat", (), {"completions": self.completions})()


# 构造带 overlap 重复的检索结果（模拟真实场景：相邻块共享尾巴）
t = (
    'TTFT（Time To First Token）= 从发起请求到收到第一个"非空 content"的时间。'
    "六轮基准里 3209 ms → 1567 ms，体感约 2.0 倍。" * 4
)
c1 = Chunk(text=t, page=3, source="sse.md", index=0)
c2 = Chunk(text=" " + t, page=4, source="sse.md", index=1)  # overlap 近重复
c3 = Chunk(
    text="完全无关的另一段内容，讲的是部署与容器网络。", page=9, source="deploy-notes.md", index=2
)


async def main() -> int:
    client = FakeClient()
    answer = await generate_answer(
        "TTFT 的口径是怎么定的？",
        [c1, c2, c3],
        client=client,
        model="m",
        min_score=0.0,
        scores=[0.85, 0.83, 0.1],
    )
    print("dedupe:", answer.dedupe)
    print("used_chunks:", answer.used_chunks)
    print("citations:", [(c.index, c.source, c.page) for c in answer.citations])
    print(
        "invalid_refs:",
        answer.invalid_refs,
        "degraded:",
        answer.degraded,
        "unreferenced:",
        answer.unreferenced,
    )

    assert answer.dedupe["near_removed"] == 1, "overlap 重复必须被合并"
    assert answer.used_chunks == 2, "去重后只剩 2 块进上下文"
    assert answer.invalid_refs == [] and answer.degraded is False
    assert answer.unreferenced == [2], "无关块未被引用应被记录"

    # 编号连续性：available 的编号必须是 1..n（去重后重排的间接验证）
    _, available = build_context([c1, c3])
    assert [c.index for c in available] == [1, 2]
    assert len(client.completions.calls) == 1, "无越界时只调一次模型"
    print("OK 端到端：去重合并 overlap 重复、编号连续、引用映射正确、零越界")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
