"""冷/热两轮成本对比（W12 周二 16:30–17:00）。

同一批 20 条查询跑两轮：
- 冷轮：清缓存后全链路（embed + search，真实 CPU 编码）；
- 热轮：命中缓存查表（sha256 + LRU + 回查 store）。

记录每轮延迟分布与缓存命中率。「token 差」部分：检索层缓存不省 token
（省的是 embed CPU），答案层缓存省的是模型 token —— 离线用 hashing
embedder 无法产生真实 token 账单，所以 token 数据用 usage_log 口径的
推算公式给出（每条答案 ~2500 token 上下文 + 200 输出），
真实账单需要联网跑（脚本退出码 0 = 延迟数据有效，token 是推算值）。

跑法：python -X utf8 scripts/cache_cost_compare.py
"""

from __future__ import annotations

import statistics
import sys
import time

sys.path.insert(0, ".")

from app.rag.cached_qa import CachedQAEngine  # noqa: E402
from app.rag.embedding.hashing import HashingEmbedder  # noqa: E402
from app.rag.indexing import Indexer  # noqa: E402
from app.rag.query import QAEngine  # noqa: E402
from app.rag.result_cache import ResultCache, reset_result_cache  # noqa: E402
from app.rag.schema import Document  # noqa: E402
from app.rag.settings import RagConfig  # noqa: E402
from app.rag.splitters.recursive import RecursiveSplitter  # noqa: E402
from app.rag.stores.memory import InMemoryVectorStore  # noqa: E402

# 20 条查询：10 条可答（内置语料主题内）+ 10 条主题外（负缓存的主力）
ANSWERABLE = [
    "违约金怎么计算",
    "违约金的上限是多少",
    "甲方逾期支付怎么办",
    "含水率的合格线是多少",
    "含水量测定用什么方法",
    "超差了怎么处理",
    "复测的规则是什么",
    "烘干温度是多少",
    "合同总额的比例限制",
    "不合格判定条件",
]
UNANSWERABLE = [
    "抗压强度的判定标准",
    "外观检验的裂纹标准",
    "劳动合同的解除条件",
    "数据隐私的保护要求",
    "光通量的测量方法",
    "同轴电缆的规格",
    "背光模组的良率",
    "景深计算的公式",
    "锈蚀等级的划分",
    "气泡缺陷的验收标准",
]


def build() -> tuple[QAEngine, ResultCache]:
    reset_result_cache()
    embedder = HashingEmbedder()
    store = InMemoryVectorStore()
    indexer = Indexer(embedder=embedder, store=store, splitter=RecursiveSplitter())
    corpus = [
        Document(
            source="合同.md",
            page=None,
            text=(
                "违约金条款：甲方逾期支付的，违约金按日万分之五计算，"
                "上限为合同总额的百分之十。复测规则：单块低于限值但平均值达标时"
                "允许复测一次，复测仍不达标判为不合格。"
            ),
        ),
        Document(
            source="规范.md",
            page=None,
            text=(
                "含水量限值：烘干法测定，105±5℃ 烘至恒重。"
                "含水率 ≤10.0% 判为合格。含水率超过限值应复测一次，"
                "复测仍超差判为不合格并通知质检主管。"
            ),
        ),
    ]
    indexer.index_documents(corpus, source="mixed")
    engine = QAEngine(embedder=embedder, store=store, config=RagConfig(score_threshold=0.0))
    return engine, ResultCache()


def run_round(engine: QAEngine, cache: ResultCache, queries: list[str]) -> dict:
    shell = CachedQAEngine(engine, cache=cache)
    latencies: list[float] = []
    hits_before = cache.stats()["retrieval"]["hits"]
    misses_before = cache.stats()["retrieval"]["misses"]
    for q in queries:
        started = time.perf_counter()
        shell.retrieve_chunks(q)
        latencies.append((time.perf_counter() - started) * 1000)
    stats = cache.stats()["retrieval"]
    # LRU 计数是**累计**的，本轮的量 = 差值（直接读总数会把上一轮算进来）
    round_hits = stats["hits"] - hits_before
    round_misses = stats["misses"] - misses_before
    return {
        "latencies_ms": latencies,
        "p50": statistics.median(latencies),
        "p99": sorted(latencies)[int(len(latencies) * 0.99) - 1],
        "mean": statistics.mean(latencies),
        "hits": round_hits,
        "misses": round_misses,
    }


def main() -> int:
    engine, cache = build()
    queries = ANSWERABLE + UNANSWERABLE  # 20 条

    # 冷轮：缓存已空
    cold = run_round(engine, cache, queries)
    # 热轮：同 query 再跑一遍（全部应命中）
    hot = run_round(engine, cache, queries)

    hot_rate = hot["hits"] / (hot["hits"] + hot["misses"]) if (hot["hits"] + hot["misses"]) else 0.0

    print("=" * 64)
    print("缓存成本对比（20 条 query：10 可答 + 10 主题外，hashing 离线）")
    print("=" * 64)
    print(f"冷轮：mean={cold['mean']:.2f}ms  p50={cold['p50']:.2f}ms  p99={cold['p99']:.2f}ms")
    print(f"热轮：mean={hot['mean']:.2f}ms  p50={hot['p50']:.2f}ms  p99={hot['p99']:.2f}ms")
    print(f"热轮命中率：{hot['hits']}/{hot['hits'] + hot['misses']} = {hot_rate:.0%}")
    print(f"检索延迟下降：{(1 - hot['mean'] / cold['mean']) * 100:.1f}%")
    print()
    print("token 影响（推算口径，非账单）：")
    print("  检索层缓存：不省模型 token（省 embed CPU；bge 上每条查询 ~10ms → hashing 这里太便宜，")
    print("  真实收益要在 bge 语料 500+ 块的索引重建时才显著 —— embedding/cache.py 已有数据）")
    print("  答案层缓存：命中一次 = 省 1 次模型调用 ≈ 2700 token（6000 字上下文 + 输出）。")
    print("  按答案层命中率 70% 估算：每 100 次问答省 70 × 2700 ≈ 189k token。")
    print()
    print("注意：本脚本离线（hashing embedder 毫秒级），延迟差的真实量级要在")
    print("bge + Chroma 链路上复测（检索 ~11ms → 缓存命中 ~1ms 仍有 90% 下降）。")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
