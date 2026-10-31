"""端到端演示：用假 LLM 跑完整问答链路，验证"带引用"与"拒答"。

**为什么不用真模型**：本机网关对 chat 端点返回 502（未配置凭据），
而真实答案的内容**不是这条链路的验收标准** —— 验收标准是四件结构性的事：

1. 检索到的块进了上下文，且带编号
2. prompt 里明确"只允许引用给定编号"
3. 答案里的 [n] 能映射回 source + page
4. 检索无结果/低分时**不调模型**，直接拒答

这四件事用假 client 就能验，而且**可复现**。真模型输出每次都不一样，
拿它验收链路等于把"链路对不对"和"模型答得好不好"混在一起。

真模型的接入在 `docs/rag-query-chain.md` §6 记为"未验证"——
它需要能调 chat 端点的凭据，这是环境限制，不是代码问题。

运行：

    uv run python -X utf8 scripts/rag_e2e_demo.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.embedding import HashingEmbedder  # noqa: E402
from app.rag.generation import REFUSAL_TEXT  # noqa: E402
from app.rag.pipeline import RagPipeline  # noqa: E402

# 知识库：3 条合同条款
CLAUSES = {
    "合同.pdf": [
        ("第五条 付款计划：验收合格后十五个工作日内付款。", 1),
        ("第六条 违约责任：乙方逾期交付的，按日万分之五支付违约金。", 2),
        ("第十三条 质保期：系统验收合格之日起 12 个月。", 3),
    ],
}


class ScriptedClient:
    """按提问返回预设答案，并记录收到的 prompt。

    **记录 prompt 是关键**：验收"引用约束写进了 prompt"必须有证据，
    而这个证据只在请求体里。没有它，"我加了约束"只是一句声称。
    """

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict] = []
        self.completions = self._Completions(self)
        self.chat = type("Chat", (), {"completions": self.completions})()

    class _Completions:
        def __init__(self, outer: ScriptedClient) -> None:
            self.outer = outer

        async def create(self, **kwargs):  # noqa: ANN003, ANN201
            self.outer.calls.append(kwargs)
            return _ScriptedResponse(self.outer.reply)


class _ScriptedResponse:
    def __init__(self, content: str) -> None:
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


def build_pipeline() -> RagPipeline:
    pipeline = RagPipeline(embedder=HashingEmbedder(dimension=256), top_k=3)
    for source, clauses in CLAUSES.items():
        for text, page in clauses:
            from app.rag.schema import Document

            pipeline.indexer.index_documents(
                [Document(source=source, page=page, text=text, pages=(page,))],
                source=source,
            )
    return pipeline


async def main() -> int:
    pipeline = build_pipeline()
    print(f"知识库：{pipeline.store.count()} 条（{len(CLAUSES)} 份文档）\n")

    results: dict[str, object] = {}

    # ---------------------------------------------------------- ① 带引用回答
    #
    # **编号由检索排序决定，不是写死的**：这里先检索一次、看真正的排序，
    # 再据此构造"模型会怎么引用"。我第一版直接把回复写成 `[2]`（假设违约金
    # 排第二），结果 `[2]` 映射到的是 p1 的付款条款 —— 断言失败。
    # 这正好说明**编号是动态分配的**：写死编号的测试验不到映射对不对，
    # 只是碰巧可能对上。
    question = "逾期交付的违约金怎么算"
    probe = pipeline.retrieve(question)
    relevant_rank = next(
        (rank for rank, hit in enumerate(probe.hits, start=1) if "违约金" in hit.chunk.text),
        None,
    )
    assert relevant_rank is not None, "检索没找到违约金条款"
    relevant_hit = probe.hits[relevant_rank - 1]

    client = ScriptedClient(f"乙方逾期交付的，按日万分之五支付违约金 [{relevant_rank}]。")
    answer = await pipeline.answer(
        question, client=client, model="demo-model", top_k=3, min_score=0.0
    )

    print(f"① 提问：{question}")
    print("   检索排序：" + " | ".join(f"[{i}]{h.chunk.page}" for i, h in enumerate(probe.hits, 1)))
    print(f"   答案：{answer.text}")
    print(f"   引用：{answer.citation_block()}")
    print(
        f"   上下文 {answer.context_chars} 字 / {answer.used_chunks} 块，"
        f"最高分 {answer.top_score:.4f}"
    )
    assert answer.citations, "带引用的回答必须解析出 citation"
    assert answer.citations[0].page == relevant_hit.chunk.page, (
        f"引用应映射回 p{relevant_hit.chunk.page}，实际 p{answer.citations[0].page}"
    )
    assert client.calls, "应当调用了模型"

    prompt = client.calls[0]["messages"][1]["content"]
    assert "只允许引用" in prompt, "引用约束没写进 prompt"
    assert "[1]" in prompt and "[3]" in prompt, "可用编号没枚举进 prompt"
    print("   ✓ prompt 里含「只允许引用给定编号」且枚举了 [1][2][3]")

    results["with_citation"] = {
        "question": question,
        "retrieval_order": [{"rank": i, "page": h.chunk.page} for i, h in enumerate(probe.hits, 1)],
        "answer": answer.text,
        "citations": [
            {"index": c.index, "label": c.label, "page": c.page} for c in answer.citations
        ],
        "prompt_has_constraint": "只允许引用" in prompt,
    }

    # ---------------------------------------------------------- ② 越界编号
    print("\n② 模型编了一个不存在的编号")
    client2 = ScriptedClient("违约金按日万分之五计算 [2]，另见 [9]。")
    answer2 = await pipeline.answer(
        question, client=client2, model="demo-model", top_k=3, min_score=0.0
    )
    print(f"   答案：{answer2.text}")
    print(f"   有效引用：{[c.index for c in answer2.citations]}")
    print(f"   ⚠ 无效编号：{answer2.invalid_refs}")
    assert answer2.invalid_refs == [9], "越界编号必须被记录"
    results["invalid_refs"] = {
        "answer": answer2.text,
        "valid": [c.index for c in answer2.citations],
        "invalid": answer2.invalid_refs,
    }

    # ---------------------------------------------------------- ③ 空库拒答
    print("\n③ 检索无结果（空知识库）")
    empty = RagPipeline(embedder=HashingEmbedder(dimension=256), top_k=3)

    class NeverCall:
        @property
        def chat(self):  # noqa: ANN201
            raise AssertionError("拒答路径不该调用模型")

    answer3 = await empty.answer("知识库里没有的问题", client=NeverCall(), model="demo-model")
    print(f"   答案：{answer3.text.splitlines()[0]}")
    print(f"   refused={answer3.refused}")
    assert answer3.refused is True
    assert answer3.text == REFUSAL_TEXT
    results["refuse_empty"] = {"refused": answer3.refused, "text": answer3.text}

    # ---------------------------------------------------------- ④ 低分拒答
    print("\n④ 分数低于阈值")
    answer4 = await pipeline.answer(
        "完全不相关的问题：今天天气如何",
        client=NeverCall(),
        model="demo-model",
        top_k=3,
        min_score=0.99,  # 高到必然触发
    )
    print(f"   最高分 {answer4.top_score:.4f} < 阈值 0.99")
    print(f"   refused={answer4.refused}")
    assert answer4.refused is True
    results["refuse_low_score"] = {
        "refused": answer4.refused,
        "top_score": answer4.top_score,
        "threshold": 0.99,
    }

    out = PROJECT_ROOT / "scripts" / "rag_e2e_demo.json"
    out.write_text(
        json.dumps(
            {
                "meta": {
                    "client": "ScriptedClient（假模型，本机网关 502 未配置凭据）",
                    "embedder": "hashing-256",
                    "corpus_size": pipeline.store.count(),
                    "note": "验的是链路结构，不是答案质量；真模型接入尚未验证",
                },
                "cases": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n全部通过。已写入 {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
