"""工厂：按配置选策略。**三种策略的唯一装配点。**

## 为什么需要工厂，而不是让调用方自己 `new`

三种策略的构造签名不同：

| 策略 | 构造需要什么 |
| --- | --- |
| `recursive` | 什么都不用（配置在 `chunk()` 里） |
| `heading` | 什么都不用 |
| `semantic` | **必须一个 `Embedder`** |

如果调用方各自 `new`，那"选语义策略但忘了传 embedder"这件事就会在
**某个分支上变成 `TypeError`**，而那个分支通常在实验脚本里 ——
报错时间点离真正的原因（配置写错）很远。

工厂把这件事收进一个地方：`build_strategy("semantic", embedder=...)`，
缺 embedder 时**立刻给一句人话**，而不是等 10 分钟后在 `chunk()` 里炸。

## `from_config` 为什么接受两种形态

`strategy` 可以是一个 `ChunkingStrategy` 实例（测试要注入假实现），
也可以是一个字符串（配置文件的形态）。两种都收，是为了让
"注入替身"与"读配置"不必写两套装配代码 —— 而两套装配代码必然漂移。

## 为什么不让 `ChunkingConfig` 自己也带 `strategy` 字段

那样工厂就不需要 `strategy=` 参数了，看起来更简单。但代价是
**配置对象同时承担"选哪个"和"参数多少"两件事** ——
而 `ChunkingConfig` 是三种策略共用的（见 `base.py` 的说明），
让它带一个"只对工厂有意义"的字段，会让"cfg 被策略读取"这件事出现两种含义。
分开：`strategy` 是装配参数，`cfg` 是策略参数。
"""

from __future__ import annotations

from app.rag.chunking.base import ChunkingConfig, ChunkingStrategy
from app.rag.chunking.heading import HeadingChunking
from app.rag.chunking.recursive import RecursiveChunking
from app.rag.chunking.semantic import SemanticChunking
from app.rag.embedding import Embedder

# 已注册的策略名。**新增策略改这一处**（与 `stores/__init__.py` 的 get_store 同一个模式）。
STRATEGIES: tuple[str, ...] = ("recursive", "heading", "semantic")

DEFAULT_STRATEGY: str = "recursive"
"""默认策略 = 递归字符。

理由与 W4 默认 `DelimiterSplitter` 相同：它**只依赖字符**，任何文本都能跑，
不会因缺元数据（PDF 无标题）或缺 embedder（CI / 免费层）而静默降级。
另外两种策略在特定文档上更好，但它们都有"前提条件不满足就悄悄变差"的问题 ——
默认值必须是"最不容易静默变差"的那个。
"""


def build_strategy(
    name: str = DEFAULT_STRATEGY,
    *,
    embedder: Embedder | None = None,
    cfg: ChunkingConfig | None = None,
) -> ChunkingStrategy:
    """按名构造策略。

    `cfg` 参数在这里**不参与构造**，存在的唯一目的是提前校验配置合法性
    （见 `validate_config`）—— 让"配置写错"在装配阶段就暴露，
    而不是在切完 5000 页之后才发现块长是负的。
    """
    if cfg is not None:
        validate_config(cfg)

    if name == "recursive":
        return RecursiveChunking()
    if name == "heading":
        return HeadingChunking()
    if name == "semantic":
        if embedder is None:
            raise ValueError(
                "semantic 策略必须传 embedder：语义切分的断点完全由句向量决定，"
                "给它一个默认实现等于让『用哪个模型切的』变成隐式状态。"
                "请显式传 embedder=get_embedder(...)"
            )
        return SemanticChunking(embedder)
    raise ValueError(f"未知的切分策略：{name}（当前可用：{'、'.join(STRATEGIES)}）")


def from_config(
    strategy: str | ChunkingStrategy,
    *,
    embedder: Embedder | None = None,
    cfg: ChunkingConfig | None = None,
) -> ChunkingStrategy:
    """接受"策略名"或"策略实例"，统一返回实例。

    `str` 走工厂；实例原样返回。这条分支是给测试注入替身用的 ——
    但**同一个函数**，不是两套装配代码（见模块 docstring）。
    """
    if isinstance(strategy, str):
        return build_strategy(strategy, embedder=embedder, cfg=cfg)
    if cfg is not None:
        validate_config(cfg)
    return strategy


def validate_config(cfg: ChunkingConfig) -> None:
    """配置合法性检查。**在装配阶段调用，不在切分循环里**。

    这里只校验"会让切分不收敛或产出空块"的硬约束：

    - `chunk_chars <= 0`：预算为零，任何文本都切不出块
    - `overlap_chars >= chunk_chars`：步进为零，合并循环要么死锁、要么
      被边界条件静默吞成一块（后者更糟：看起来跑通了，实际全文挤进一块）
    - `semantic_min_chars >= semantic_max_chars`：兜底逻辑自相矛盾 ——
      "过短要合并"的门槛高于"过长要断开"的门槛，两条规则会互相抵消
    - `heading_max_level` 不在 1~6：loader 只产出 1~6，别的值等于关掉标题切分

    **这些检查为什么不写进 `ChunkingConfig.__post_init__`**：
    `ChunkingConfig` 是 `frozen` 的纯数据，在那里抛异常会让"构造一个配置对象"
    变成可能失败的表达式 —— 而 dataclass 的惯例是纯赋值。
    更重要的是：**策略参数之间的一致性**（最后一条）不属于任何单个字段，
    它是跨字段约束，天然属于装配期校验。
    """
    if cfg.chunk_chars <= 0:
        raise ValueError(f"chunk_chars 必须为正数，收到 {cfg.chunk_chars}")
    if cfg.overlap_chars < 0:
        raise ValueError(f"overlap_chars 不能为负，收到 {cfg.overlap_chars}")
    if cfg.overlap_chars >= cfg.chunk_chars:
        raise ValueError(
            f"overlap_chars({cfg.overlap_chars}) 必须小于 chunk_chars({cfg.chunk_chars})；"
            "否则合并步进为零，切分不收敛"
        )
    if not 1 <= cfg.heading_max_level <= 6:
        raise ValueError(f"heading_max_level 必须在 1~6，收到 {cfg.heading_max_level}")
    if cfg.semantic_min_chars >= cfg.semantic_max_chars:
        raise ValueError(
            f"semantic_min_chars({cfg.semantic_min_chars}) 必须小于 "
            f"semantic_max_chars({cfg.semantic_max_chars})；"
            "否则『过短合并』与『过长断开』互相抵消"
        )
    if not 0.0 <= cfg.semantic_breakpoint_percentile <= 100.0:
        raise ValueError(
            f"semantic_breakpoint_percentile 必须在 0~100，收到 {cfg.semantic_breakpoint_percentile}"
        )


def describe_strategies() -> dict[str, str]:
    """列出可用策略。给 CLI / 报告用，避免它们各写一份 if-chain。"""
    return {
        "recursive": "递归字符：分隔符优先级逐级降级（保底，任何文本都能跑）",
        "heading": "按标题层级：结构优先，大节在节内二次切（PDF 会降级为按页）",
        "semantic": "语义：句向量相似度找断点（需 embedder，hashing 下无参考价值）",
    }


__all__ = [
    "DEFAULT_STRATEGY",
    "STRATEGIES",
    "build_strategy",
    "describe_strategies",
    "from_config",
    "validate_config",
]
