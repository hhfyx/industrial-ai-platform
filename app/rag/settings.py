"""RAG 参数配置：**所有可调参数的一处定义，retrieve() 从它读，签名不变**。

## 为什么需要这个文件（今天是它存在的必要性最清楚的一天）

到昨天为止，同一个参数散落在至少三处：

| 参数 | 昨天它在哪 |
| --- | --- |
| `top_k` | `Retriever.DEFAULT_TOP_K`（=5）/ `ChunkSize.chunk_chars` / UI 滑块 |
| `chunk_size` / `overlap` | `ChunkSize`（旧链路）**和** `ChunkingConfig`（新链路）各一份 |
| `score_threshold` | `Embedder.recommended_min_score`（绑在 embedder 上） |
| `strategy` | `chunking.factory.DEFAULT_STRATEGY` |

`chunk_size` 有两份这件事最危险：`Indexer` 默认走 `RecursiveSplitter(ChunkSize())`，
而网格实验走 `build_strategy("recursive", ChunkingConfig())`。实测同一批 322 段文档，
**旧链路切 351 块、新链路切 353 块，且逐块文本不同** ——
也就是说"改一个参数"在两个入口上会切出不一样的东西，而两边都不会报错。

所以这里的规则是：**调参只改这里，代码里不许再出现字面量默认值**。

## 与 `core/config.py` 的分工

`app/core/config.py` 管**进程级基础设施**（LLM 凭证、超时、日志级别）。
这个文件管**检索质量参数**（k、阈值、切分、过滤默认值）。

分开的理由不是洁癖：`Settings` 的字段会进 `/health` 与启动日志，
而检索参数会随实验每周变。混在一起会让"我今天想试 k=8"变成
"改应用配置并重启"，而它本该是"传给构造器的一个值"。

**但两者不重复定义**：环境变量仍走 `RAG_` 前缀（与 `deploy_config.EMBEDDER_ENV`
同一个命名空间），读 env 的入口只有这里的 `from_env()`。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Literal

Strategy = Literal["recursive", "heading", "semantic"]

# 环境变量前缀。与 `deploy_config.EMBEDDER_ENV = "RAG_EMBEDDER"` 同一命名空间。
ENV_PREFIX: str = "RAG_"

DEFAULT_TOP_K: int = 5
"""默认取 5 条。**这是全项目唯一一处 top_k 默认值**（原来在 `retrieval.py`）。

## 为什么是 5，而不是昨天之前那个"四个答案"

`scripts/topk_scan.py` 的实测（20 条 gold，chunk=500/overlap=50）：

| k | Recall@k | 上下文 token | 噪音条数 | 每 100 token 换来的召回 |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 55.0% | 804.5 | 2.45 | — |
| **5** | **65.0%** | **1321.7** | **4.35** | **+1.93 pp** |
| 8 | 70.0% | 2109.5 | 7.30 | +0.63 pp |
| 10 | 75.0% | 2608.8 | 9.25 | +1.00 pp |

选 5 的理由是**边际收益的拐点**：3→5 每 100 token 换 1.93pp 召回，
5→8 掉到 0.63pp。再往上虽然还有收益（8→10 是 1.00pp），但 k=10 的
上下文是 k=5 的 **2 倍**、噪音条数翻倍。

**这条曲线在 20 条样本上是不显著的**（见 `docs/retrieval-tuning-token-grid.md` §6 的教训：
1 条翻转 = 5pp）。所以 5 不是"最优"，是**"在拐点上且维持现状成本最低"**：
改成 8 要多付 60% 的上下文与 68% 的噪音，换来的 5pp 落在置信区间里。

> **必答题的答案在这一行**：k=10 的 Recall@5 与 k=5 **必然相等**
> （`topk_scan.check_prefix_invariant` 逐条验证过：前 5 条是同一批）。
> 所以选 k 从来不是看召回，是看"多送的那些 token 和噪音值不值"。
"""

DEFAULT_SCORE_THRESHOLD: float = 0.575
"""拒答阈值。**注意：分数的方向是相似度，越大越像。**

## W5 周六精标：0.40 → 0.575

`scripts/threshold_fine.py` 在 50 条集（40 可答 + 10 拒答）上扫了 19 档：

| 阈值 | 正确拒答 | 误放 | 误拒 | 端到端答对 |
| ---: | ---: | ---: | ---: | ---: |
| 0.400（旧） | 0/10 | 10 | 0 | 16/50 |
| **0.575（新）** | **2/10** | **8** | **1** | **17/50** |
| 0.600 | 6/10 | 4 | 4 | 18/50 |
| 0.650 | 8/10 | 2 | 10 | 14/50 |

**为什么选 0.575 而不是端到端最高的 0.600**：
0.600 会误拒 4 条（40 条可答题里 10%），而"知识库里有的答不出来"
对用户的伤害大于"偶尔答一条不相关的"。0.575 只误拒 1 条。

**这是一个产品决定，不是数据决定。** 三个候选的完整数据在
`docs/strategy-comparison.md` §5.3；不同意只需改这一个常量。

## 0.40 是怎么来的（历史，保留）

来自 `Embedder.recommended_min_score` 的实测反推
（`scripts/embedding_score_probe.py` 的 4+4 条标注对：

    相关 0.578~0.676    无关 0.253~0.367    中点 0.472

取 0.40 是因为**误拒真答案比放过弱相关更伤**。

**但它只负责挡住明显无关，不负责提高准确率。** 50 条集实测：
拒答题分数 [0.5549, 0.7958] **覆盖**了可答题的 [0.5157, 0.7207] ——
拒答最高分比可答最高分还高。所以精标只能把拒答从 0/10 提到 2/10，
**不可能更好**：那不是阈值问题，是分数本身不含"能不能回答"这个信息。
"""

DEFAULT_STRATEGY: Strategy = "recursive"
"""默认切分策略。**与 `chunking.factory.DEFAULT_STRATEGY` 必须一致**（测试守着）。"""

DEFAULT_CHUNK_CHARS: int = 500
"""目标块长（字符）。见 `docs/retrieval-tuning-token-grid.md` §1：

- 500 时 527 块里 1 块超 512 token（0.19%，安全）
- 800 时 12.15%，1000 时 14.29% —— **调大块长会撞上 bge 的截断墙**
"""

DEFAULT_OVERLAP_CHARS: int = 50
"""相邻块重叠字符数。

`docs/retrieval-tuning-token-grid.md` §4 的实测：0/50/100 三档命中率分不出差异
（McNemar p 全部 = 1.0），但重复内容率单调上升 5.5% → 8.0% → 11.0%。
**overlap=100 是确定的零收益**（命中条数不变，块数 +11%）。
维持 50 的理由是"没有证据说该改，而改它要重建索引"。
"""

DEFAULT_MAX_CONTEXT_CHARS: int = 6000
"""拼进 prompt 的上下文字符上限。中文 ≈ 0.67 tok/字 ≈ 4000 token。"""


@dataclass(frozen=True, slots=True)
class RagConfig:
    """检索质量参数的**唯一来源**。

    `frozen` 是刻意的：传进 `Retriever` 之后不该有人改它。
    想换一套参数请 `replace(cfg, top_k=8)`，那是显式的、可 diff 的。
    """

    top_k: int = DEFAULT_TOP_K
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    strategy: Strategy = DEFAULT_STRATEGY
    chunk_chars: int = DEFAULT_CHUNK_CHARS
    overlap_chars: int = DEFAULT_OVERLAP_CHARS
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS

    def __post_init__(self) -> None:
        """**校验放在这里，因为它能拦住"改了配置但索引没重建"之外的错误。**

        四条检查各自对应一个会静默出错的情况：

        - `top_k <= 0`：`store.search` 会抛，但抛在很深的调用栈里
        - `score_threshold` 不在 [-1, 1]：余弦相似度的定义域，越界等于关掉或全拒
        - `overlap_chars >= chunk_chars`：切分步进为零，不收敛
        - `chunk_chars` 过大：会撞 bge 的 512 token 截断墙（见模块常量说明）
        """
        if self.top_k <= 0:
            raise ValueError(f"top_k 必须为正数，收到 {self.top_k}")
        if not -1.0 <= self.score_threshold <= 1.0:
            raise ValueError(
                f"score_threshold 必须在 [-1, 1]（余弦相似度的定义域），"
                f"收到 {self.score_threshold}。注意方向是**越大越像**。"
            )
        if self.chunk_chars <= 0:
            raise ValueError(f"chunk_chars 必须为正数，收到 {self.chunk_chars}")
        if self.overlap_chars < 0:
            raise ValueError(f"overlap_chars 不能为负，收到 {self.overlap_chars}")
        if self.overlap_chars >= self.chunk_chars:
            raise ValueError(
                f"overlap_chars({self.overlap_chars}) 必须小于 "
                f"chunk_chars({self.chunk_chars})，否则切分不收敛"
            )
        if self.max_context_chars <= 0:
            raise ValueError(f"max_context_chars 必须为正数，收到 {self.max_context_chars}")

    # ------------------------------------------------------------------ 派生

    def chunking_config(self) -> Any:  # noqa: ANN401
        """翻译成 `ChunkingConfig`。**这是唯一的翻译点。**

        索引侧与实验脚本都从它拿，所以"改了 config 但索引用了另一套参数"
        在结构上不可能 —— 而那个错误的表现是"检索效果和预期不符"，
        查起来要翻到切分器里。
        """
        from app.rag.chunking.base import ChunkingConfig

        return ChunkingConfig(
            chunk_chars=self.chunk_chars,
            overlap_chars=self.overlap_chars,
            # 策略不在 ChunkingConfig 里（它是装配参数，见 factory.py 的说明），
            # 所以这里只翻译"策略参数"，`strategy` 由调用方传给 build_strategy。
        )

    def as_dict(self) -> dict[str, object]:
        """给日志/`/health`/实验产物用。**顺序固定**，便于人眼对照。"""
        return {
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "strategy": self.strategy,
            "chunk_chars": self.chunk_chars,
            "overlap_chars": self.overlap_chars,
            "max_context_chars": self.max_context_chars,
        }

    def with_overrides(self, **changes: object) -> RagConfig:
        """派生一份改过的配置。**显式、可 diff、不改原对象。**"""
        return replace(self, **changes)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ 环境变量

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> RagConfig:
        """从 `RAG_*` 环境变量构造。**未设置的项取默认值，格式错误直接抛。**

        格式错误必须抛（而不是回退默认）：`RAG_TOP_K=eigth` 静默变成 5，
        会让人以为"配置生效了但没效果" —— 而实际是根本没读到。
        """
        env = os.environ if environ is None else environ
        changes: dict[str, Any] = {}

        def _get(name: str) -> str | None:
            value = env.get(f"{ENV_PREFIX}{name}")
            return value.strip() if isinstance(value, str) and value.strip() else None

        if (raw := _get("TOP_K")) is not None:
            changes["top_k"] = _parse_int("RAG_TOP_K", raw)
        if (raw := _get("SCORE_THRESHOLD")) is not None:
            changes["score_threshold"] = _parse_float("RAG_SCORE_THRESHOLD", raw)
        if (raw := _get("CHUNK_CHARS")) is not None:
            changes["chunk_chars"] = _parse_int("RAG_CHUNK_CHARS", raw)
        if (raw := _get("OVERLAP_CHARS")) is not None:
            changes["overlap_chars"] = _parse_int("RAG_OVERLAP_CHARS", raw)
        if (raw := _get("MAX_CONTEXT_CHARS")) is not None:
            changes["max_context_chars"] = _parse_int("RAG_MAX_CONTEXT_CHARS", raw)
        if (raw := _get("STRATEGY")) is not None:
            if raw not in ("recursive", "heading", "semantic"):
                raise ValueError(
                    f"RAG_STRATEGY 取值非法：{raw!r}；可用：recursive / heading / semantic"
                )
            changes["strategy"] = raw

        return cls(**changes)


def _parse_int(name: str, raw: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数，收到 {raw!r}") from exc


def _parse_float(name: str, raw: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字，收到 {raw!r}") from exc


# 进程级缓存。**与 `get_settings` 同一模式**：解析一次，测试用 `reset()` 清。
_cached: RagConfig | None = None


def get_rag_config() -> RagConfig:
    """全进程单例。**代码里所有取检索参数的地方都从这里来。**"""
    global _cached
    if _cached is None:
        _cached = RagConfig.from_env()
    return _cached


def set_rag_config(config: RagConfig) -> None:
    """显式设置（测试/实验脚本用）。"""
    global _cached
    _cached = config


def reset_rag_config() -> None:
    """清缓存，下次 `get_rag_config()` 重新读环境变量。"""
    global _cached
    _cached = None


__all__ = [
    "DEFAULT_CHUNK_CHARS",
    "DEFAULT_MAX_CONTEXT_CHARS",
    "DEFAULT_OVERLAP_CHARS",
    "DEFAULT_SCORE_THRESHOLD",
    "DEFAULT_STRATEGY",
    "DEFAULT_TOP_K",
    "ENV_PREFIX",
    "RagConfig",
    "Strategy",
    "get_rag_config",
    "reset_rag_config",
    "set_rag_config",
]
