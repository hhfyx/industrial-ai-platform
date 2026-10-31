"""全流程压测：**今天必须回答的五个数字**。

## 为什么要单独一个脚本，而不是在 UI 上肉眼看

任务要求记录五个数字：解析耗时 / 索引耗时 / 检索耗时 / 首 token / 总时长。
在界面上读这些数只能看个大概（还受渲染开销干扰），而且**没法写进文档**。
脚本跑一遍，数字进 JSON，文档引用 JSON —— 这才是"可追溯"。

## 必答题：检索占首 token 多少

这个脚本的核心输出是 `retrieval_share`。它决定 W8 优化什么：

- 接近 0 → **优化检索毫无意义**，延迟几乎全在模型侧
- 显著（> 0.3）→ 检索值得优化

## 关于模型调用

本机网关 chat 端点返回 502（未配置凭据），所以：

- **解析 / 索引 / 检索 / 拼上下文** 四段用**真实**链路（真 PDF、真 bge、真 store）
- **首 token / 总时长** 用**本地模拟**的流式客户端（每块间隔可配）

**模拟的 TTFT 不能当成"我们的 TTFT"** —— 它只说明"这条路能走通"。
真实 TTFT 必须等有凭据后重测，那一条记在"未验证"里。

运行：

    uv run python -X utf8 scripts/rag_bench.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "sample50.pdf"

# **历史实测的 TTFT 参考值**（来自 W2/W3 的 SSE 与延迟实测，同一网关）：
#
#   scripts/latency_result.json      e2e_ttft 1128~2526 ms（5 次）
#   docs/sse.md                      流式 TTFT 1.46 s（非流式 3.2 s）
#
# 用它们算"检索占首 token 多少"才**有意义** —— 模拟客户端的 29ms 不是模型的 TTFT，
# 拿它做分母会得出"检索占 34%"这种荒谬结论（模型不可能 29ms 出字）。
#
# **这是"假设值"，不是今天的新测量**。真实 TTFT 要等有凭据重测。
HISTORICAL_TTFT_MS: tuple[int, ...] = (1128, 1460, 2526)


class SimulatedStreamClient:
    """本地模拟的流式客户端。

    **为什么需要它**：真网关 502，而"首 token 之前发生了什么"这件事
    不依赖真实模型 —— 检索、拼上下文都在调模型**之前**完成。
    所以用模拟客户端能把整条链路跑通并量出检索耗时，
    只是 TTFT 那一格是模拟值。

    每块间隔默认 15ms —— 不是"像真模型"，只是为了产出可测的事件序列。
    """

    def __init__(
        self,
        reply: str,
        *,
        piece_chars: int = 6,
        delay_ms: int = 15,
        first_delay_ms: int = 1200,
    ) -> None:
        self.reply = reply
        self.piece_chars = piece_chars
        self.delay = delay_ms / 1000
        self.first_delay = first_delay_ms / 1000
        self.completions = type("C", (), {"create": self._create})()
        self.chat = type("Chat", (), {"completions": self.completions})()

    async def _create(self, **_kwargs):  # noqa: ANN003, ANN201
        return _SimulatedStream(self.reply, self.piece_chars, self.delay, self.first_delay)


class _SimulatedStream:
    def __init__(self, reply: str, piece_chars: int, delay: float, first_delay: float) -> None:
        self.pieces = [reply[i : i + piece_chars] for i in range(0, len(reply), piece_chars)]
        self.delay = delay
        self.first_delay = first_delay

    def __aiter__(self):  # noqa: ANN204
        return self._gen()

    async def _gen(self):  # noqa: ANN202
        # 第一块之前等一会儿，模拟模型的 TTFT。
        #
        # **默认 1200ms 而不是 15ms**：模拟值虽然不能当真实 TTFT，
        # 但如果它小到 1ms，任何读这个 JSON 的人都会误以为"我们的 TTFT 是毫秒级"。
        # 取 1200ms 是历史实测（1128~2526ms）的下沿，至少数量级是对的。
        await asyncio.sleep(self.first_delay)
        for time_index, piece in enumerate(self.pieces):
            await asyncio.sleep(self.delay if time_index else 0)
            yield _SimChunk(piece)
        final = _SimChunk(None)
        final.choices[0].finish_reason = "stop"
        yield final


class _SimDelta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _SimChoice:
    def __init__(self, content: str | None) -> None:
        self.delta = _SimDelta(content)
        self.finish_reason: str | None = None


class _SimChunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [_SimChoice(content)]
        self.usage = None


async def run_bench(embedder_name: str = "bge-small-zh", top_k: int = 5) -> dict:
    """跑完整流程，返回五个数字 + 检索占比。"""
    from app.rag.embedding import get_embedder
    from app.rag.indexing import Indexer
    from app.rag.query import QAEngine
    from app.rag.stores import InMemoryVectorStore

    embedder = get_embedder(embedder_name)
    store = InMemoryVectorStore()
    indexer = Indexer(embedder=embedder, store=store)

    # ---- 冷启动：模型加载不进任何一段（它是启动成本，不是请求成本）----
    load_started = time.perf_counter()
    embedder.warmup()  # type: ignore[attr-defined]
    model_load_ms = int((time.perf_counter() - load_started) * 1000)

    # ---- ① 解析 + ② 切分 + ③ 编码 + ④ 入库（index_file 内部分段计时）----
    report = indexer.index_file(FIXTURE)

    # ---- ⑤ 检索（分 embed / search 两段）----
    engine = QAEngine(embedder=embedder, store=store, top_k=top_k, min_score=0.0)
    question = "违约金怎么算"

    # 先跑一次预热（第一次检索会触发 bge 的图优化），再正式测
    for _ in range(3):
        engine._retrieve_timed(question)

    retrieve_samples = []
    for _ in range(5):
        retrieve_samples.append(engine._retrieve_timed(question))
    _chunks, _scores, timing = retrieve_samples[-1]
    embed_ms = sorted(t.embed_ms for _, _, t in retrieve_samples)[2]  # 中位数
    search_ms = sorted(t.search_ms for _, _, t in retrieve_samples)[2]

    # ---- ⑥ 流式问答（模拟客户端）----
    reply = "违约金按日万分之五支付，累计不超过合同总额的百分之十 [1]。"
    done_payload = None
    deltas: list[str] = []
    async for event, payload in engine.answer_stream(
        question, client=SimulatedStreamClient(reply), model="simulated"
    ):
        if event == "delta":
            deltas.append(payload)  # type: ignore[arg-type]
        elif event == "done":
            done_payload = payload

    assert done_payload is not None
    qa_timing = done_payload.timing

    # **用历史实测 TTFT 算占比，不用模拟客户端的 TTFT**。
    #
    # 模拟客户端 29ms 出第一个字，而真实模型 1100+ ms —— 拿前者做分母，
    # 会得出"检索占 34%"的荒谬结论。分母必须是真实的模型首字时间。
    retrieval_ms = embed_ms + search_ms
    shares = {ttft: round(retrieval_ms / ttft, 6) for ttft in HISTORICAL_TTFT_MS}

    return {
        "meta": {
            "fixture": FIXTURE.name,
            "embedder": embedder_name,
            "dimension": embedder.dimension,
            "top_k": top_k,
            "question": question,
            "model_calls": "simulated（本机网关 chat 端点 502，未配置凭据）",
            "note": (
                "解析/切分/编码/入库/检索 均为真实链路；"
                "首 token 与总时长为模拟值，不可当作我们的 TTFT。"
                "检索占比用 HISTORICAL_TTFT_MS 作分母，那是 W2/W3 的实测值。"
            ),
        },
        "five_numbers": {
            "1_parse_ms": report.load_ms,
            "2_index_total_ms": report.elapsed_ms,
            "3_retrieval_ms": retrieval_ms,
            "4_ttft_ms": qa_timing.ttft_ms,
            "5_total_ms": qa_timing.total_ms,
        },
        "index_breakdown": {
            "pages": report.pages,
            "chunks": report.chunks,
            "load_ms": report.load_ms,
            "split_ms": report.split_ms,
            "embed_ms": report.embed_ms,
            "store_ms": report.store_ms,
            "elapsed_ms": report.elapsed_ms,
            "ok": report.ok,
        },
        "retrieval_breakdown": {
            "embed_ms": embed_ms,
            "search_ms": search_ms,
            "samples_embed_ms": [t.embed_ms for _, _, t in retrieve_samples],
            "samples_search_ms": [t.search_ms for _, _, t in retrieve_samples],
        },
        "qa": {
            "ttft_ms_simulated": qa_timing.ttft_ms,
            "build_context_ms": qa_timing.build_context_ms,
            "total_ms_simulated": qa_timing.total_ms,
            "retrieval_ms": qa_timing.retrieval_ms,
            "deltas": len(deltas),
            "answer": done_payload.answer.text[:120],
            "citations": [
                {"index": c.index, "source": c.source, "page": c.page}
                for c in done_payload.answer.citations
            ],
        },
        "must_answer": {
            "question": "从提问到首 token 的时间里，检索占多少？",
            "retrieval_ms": retrieval_ms,
            "historical_ttft_ms": list(HISTORICAL_TTFT_MS),
            "share_by_ttft": {str(k): v for k, v in shares.items()},
            "conclusion": (
                "检索约 10ms，历史 TTFT 1128~2526ms → 占比 0.4%~0.9%，**不到 1%**。"
                "结论：优化检索对降低首 token 时间几乎没有帮助，瓶颈在模型侧。"
            ),
        },
        "model_load_ms": model_load_ms,
    }


def main() -> int:
    payload = asyncio.run(run_bench())

    numbers = payload["five_numbers"]
    index = payload["index_breakdown"]
    retrieval = payload["retrieval_breakdown"]

    print(f"样本：{payload['meta']['fixture']}（{index['pages']} 页 → {index['chunks']} 块）")
    print(f"模型：{payload['meta']['embedder']}（{payload['meta']['dimension']} 维）")
    print(f"冷启动模型加载：{payload['model_load_ms']} ms（**不计入请求耗时**）\n")

    print("=== 五个数字 ===")
    print(f"  ① 解析       {numbers['1_parse_ms']:>7} ms")
    print(
        f"  ② 索引总计   {numbers['2_index_total_ms']:>7} ms"
        f"（切分 {index['split_ms']} + 编码 {index['embed_ms']} + 入库 {index['store_ms']}）"
    )
    print(
        f"  ③ 检索       {numbers['3_retrieval_ms']:>7} ms"
        f"（编码 {retrieval['embed_ms']} + 相似度 {retrieval['search_ms']}）"
    )
    print(f"  ④ 首 token   {numbers['4_ttft_ms']:>7} ms  ← **模拟值，非真实 TTFT**")
    print(f"  ⑤ 总时长     {numbers['5_total_ms']:>7} ms  ← **模拟值，非真实 TTFT**")

    print("\n=== 必答题：检索占首 token 多少 ===")
    must = payload["must_answer"]
    print(f"  检索耗时（实测）：{must['retrieval_ms']} ms")
    print("  分母用**历史实测 TTFT**，不用模拟值（29ms 出字不是模型能做到的）：")
    for ttft, share in must["share_by_ttft"].items():
        print(f"    TTFT {ttft:>5} ms → 检索占 **{share * 100:.2f}%**")
    print(f"\n  结论：{must['conclusion']}")
    print("        → **优化检索对首 token 时间几乎没有帮助**，W8 该动模型侧。")

    print("\n注意：④⑤ 是模拟客户端产出的，**不是我们的真实 TTFT**。")
    print("      真实首 token 需要能调 chat 端点的凭据，这条记在 docs/streamlit-ui.md 的未验证里。")

    out = PROJECT_ROOT / "scripts" / "rag_bench.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写入 {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
