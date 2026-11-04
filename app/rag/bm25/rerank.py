"""Reranker（Cross-Encoder）：**在候选里做精排的第二段**。

## 为什么必须是两段式（必答问题 A）

| 结构 | 怎么算 | 可预计算 | 复杂度 | 精度 |
| --- | --- | --- | --- | --- |
| **Bi-Encoder**（现在的 bge-small-zh） | query 与 doc **各自**编码成向量，再算余弦 | ✅ doc 向量可离线算好 | 检索 O(log n) | 粗 |
| **Cross-Encoder**（本模块） | query 与 doc **拼接**后过同一个模型打分 | ❌ 每对都要现算 | 每对 O(1) 但很慢 | 精 |

关键在注意力：Cross-Encoder 里 query 与 doc 的 token **在同一段注意力里互相看见**，
所以它能判断"这句话是否真的回答了这个问题"。
Bi-Encoder 把 doc 压成一个向量时就丢掉了与具体 query 的交互 —— 它只能保证
"这个 doc 整体上像这个 query"，做不到"这段是否对得上"。

**这就是两段式的理由**：全库 1363 块如果都 cross-encode，单对一个 ~30 ms
（CPU 实测），全库 40 秒 —— 一个 query 40 秒，不可用。
所以先用 Bi-Encoder 把候选压到 20~50 条（毫秒级），再对这几十条精排（秒级）。

## 必答问题 B：Cross-Encoder 的分数能不能跨 query 比较？

**基本不能。** 三个理由：

1. **尺度由 query 决定**：同一对 (query, doc)，document 侧相同、query 换一个，
   分数会整体平移。实测不同 query 的 top1 分可以在 0.3~0.99 之间大幅波动。
2. **sigmoid 之后的"0.5"没有绝对含义**：它是模型在训练分布上的一个切点，
   不是"相关/不相关"的物理边界。
3. **候选集组成会改变分布**：Top50 里如果全是弱相关块，最高分可能只有 0.4；
   如果有一条明显相关，最高分 0.95。**同一句 query 换个候选集，最高分就变了。**

所以本模块提供 `normalize_within_query()` —— 与 `fusion.py` 里那个同名函数
是**同一条纪律的第二次落地**：只用同一 query 内部的相对排序，
想用阈值就必须先按 query 内归一化（min-max 或 percentile）。

> 这是本周第五次遇到"分数尺度"问题（Chroma distance、BM25 无界、
> 加权融合的量纲、RRF 免疫量纲、现在是 Cross-Encoder）。
> 每次都指向同一个结论：**跨分布比较绝对分数，是这类系统里最容易犯且最难发现的错。**

## 为什么用 sigmoid 而不是直接 logits

bge-reranker 系列输出的是**一个 logit**（不是概率）。原论文/官方示例用 sigmoid 归一：
`score = sigmoid(logit)`。它的实际作用是**把无界 logit 压到 (0,1)**，
让"分数"看起来像概率 —— 但**这不改变它的跨 query 不可比性**，
因为 sigmoid 是单调变换，保序但不保尺度可比。

保留 sigmoid 的理由：官方口径一致，便于和别人的数对比。**不是**因为它让分数可解释。

## 截断：长 doc 会被切，切掉的部分不参与打分

`bge-reranker-v2-m3` 的 `max_length` 是 8192 token（很大），但**这里仍然要显式设**：

- 设 512 时，一个 500 字的块（约 350 token）+ query 完全放得下
- 但如果块里有表格/代码，token 数会远超字数估计 —— **超了就被切**
- 被切掉的部分**不参与打分**，于是"块后半段才是答案"的那条会被误判为不相关

三种截断策略（截尾/截头/中段）各自会让不同的块受益，
`scripts/rerank_probe.py --truncation` 会实测三种下的打分差异。

## 纯 CPU 的现实

本机 `torch.cuda.is_available()` 是 False（2.15.0+cpu）。
所以本模块的所有延迟数字都是 CPU 的 —— 这一点必须写进结论，
否则"10 ms/对"这种数会被当成 GPU 机器的预期。
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

#: 模型默认走本地路径。**不写 HF repo id**：本机 huggingface.co 不可达
#: （实测 `getaddrinfo failed`），用 repo id 会触发网络校验并卡在超时上。
#:
#: ## 路径解析顺序（W7 修正：原来硬编码 `d:/cache/...`）
#:
#: 1. 环境变量 `RERANKER_MODEL_DIR` —— 显式指定，最高优先级
#: 2. `HF_HOME` / `HF_HUB_CACHE` 下的 `modelscope/bge-reranker-v2-m3`
#:    （尊重该环境变量，就不再假设盘符）
#: 3. **本仓约定的位置**：项目根的上两级里的 `cache/...`
#:    （本仓布局是 `<盘>/git/网页/industrial-ai-platform` -> `<盘>/cache`）
#: 4. 兜底：`Path.home() / "cache" / ...`
#:
#: ## 为什么原来是硬编码，为什么必须改
#:
#: 原写法 `Path("d:/cache/modelscope/bge-reranker-v2-m3")` 在本机是对的
#: （`HF_HOME=d:\cache`，模型 2.27 GB 不该落 C 盘）。但它是**绝对路径常量**，
#: 换一台机器、换一个盘符、或在 Linux/CI 上跑，它就指向一个不存在的位置。
#:
#: 症状是"精排加载失败" —— 而排查的人会先怀疑模型没下载、怀疑显存，
#: 很难想到是代码里写死了 D 盘。**这类错误只在别人的机器上发作。**
#:
#: 改成"候选列表 + 取第一个存在的"之后，本机行为不变（第 3 条命中），
#: 但换环境时不再直接失败。
DEFAULT_MODEL_SUBPATH: Path = Path("modelscope") / "bge-reranker-v2-m3"


def _model_dir_candidates() -> list[Path]:
    """按优先级列出可能的模型目录。**第一个真实存在的才被用。**

    暴露成函数（而不是内联）是为了可测：测试里可以 monkeypatch 环境变量，
    断言候选顺序正确 —— 这类"路径拼接"逻辑光看代码很容易觉得对。
    """
    import os

    candidates: list[Path] = []

    # ① 显式环境变量
    explicit = os.environ.get("RERANKER_MODEL_DIR")
    if explicit:
        candidates.append(Path(explicit))

    # ② HF_HOME / HF_HUB_CACHE（不假设盘符，跟着用户配置走）
    for var in ("HF_HUB_CACHE", "HF_HOME", "MODELSCOPE_CACHE"):
        value = os.environ.get(var)
        if value:
            base = Path(value)
            # HF_HOME 指的是 "cache 根"，模型在 hub 子目录；这里两种都试，
            # 因为用户可能把它设成任意一级
            candidates.append(base / DEFAULT_MODEL_SUBPATH)
            candidates.append(base / "hub" / DEFAULT_MODEL_SUBPATH)

    # ③ 本仓约定的位置：项目根的上两级 + cache
    #
    # 用 `__file__` 推出项目根，**不写盘符**（FOLDERS.md 第七节也禁止硬编码盘符）。
    project_root = Path(__file__).resolve().parents[3]  # app/rag/bm25 -> 项目根
    candidates.append(project_root.parent.parent / "cache" / DEFAULT_MODEL_SUBPATH)

    # ④ 兜底：用户主目录
    candidates.append(Path.home() / "cache" / DEFAULT_MODEL_SUBPATH)

    return candidates


def resolve_model_dir() -> Path:
    """返回**实际可用的**模型目录。

    ## 一个都不存在时怎么办

    返回**第一个候选**（而不是抛异常）。理由：调用方的错误信息里
    需要包含"我们尝试过哪里"，而返回一个不存在的路径能让那句报错
    变得有信息量（`rerank.py` 的加载失败分支会打印候选列表）。

    在这里抛异常会让"只是没用精排"的场景也崩掉，而精排是可选的。
    """
    candidates = _model_dir_candidates()
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else Path("bge-reranker-v2-m3")


DEFAULT_MODEL_DIR: Path = resolve_model_dir()
r"""本地模型目录（由 `scripts/fetch_reranker.py` 从 ModelScope 拉下来）。

解析规则见 `_model_dir_candidates()`：环境变量 > HF_HOME > 本仓约定 > 用户目录。
**W7 之前这里是硬编码 `d:/cache/...`，换机器就失效。**
"""

DEFAULT_MAX_LENGTH: int = 512
"""截断长度。**512 是权衡后的选择**：

- bge-reranker-v2-m3 支持 8192，但 CPU 上注意力是 O(L²) —— 8192 太慢
- 本项目块长 500 字 ≈ 350 token，加 query 后 < 512，绝大多数不截断
- 取 512 而不是 8192：**延迟换精度**，且实测（见 `scripts/rerank_probe.py`）
  截断影响远小于预期

这个数进产物元数据，换值必须重跑测延迟。
"""

DEFAULT_BATCH_SIZE: int = 16
"""批量大小。CPU 上 batch=16 是吞吐与内存的折中（实测见 `scripts/rerank_probe.py`）。

**必须批量**，理由与 `bge.py` 的 embedder 相同：循环单条会让吞吐掉一个量级。
"""


class Truncation(StrEnum):
    """长 doc 的截断策略。**被截掉的部分完全不参与打分。**

    ## 实测：这不是"分数偏低一点"，是"从相关直接归零"

    同一句 golden 答案（「正文超长返回 422 INVALID_INPUT」）放在长文档的三个位置，
    用三种策略各打一次分（`max_length=512`）：

    | 策略 | 答案在开头 | 答案居中 | 答案在末尾 |
    | --- | ---: | ---: | ---: |
    | `TAIL`（保留开头） | **0.9235** | 0.0003 | 0.0003 |
    | `HEAD`（保留结尾） | 0.0003 | 0.0003 | **0.6463** |
    | `MIDDLE`（保留中段） | 0.0003 | **0.7796** | 0.0003 |

    **每一行只有自己覆盖的那一段有分，其余两段全部跌到 0.0003** ——
    那是"sigmoid 之后的数值噪声"，等价于"判为无关"。

    ## 所以：没有单一策略能覆盖所有位置

    这也是为什么 `max_length` 不能设得太小。若 `max_length` 足够大（本项目 512
    相对块长 500 字已经够用），**根本不会触发截断**，这个风险就不存在。
    实测语料里只有 3/1404 块超过 512 token。

    真正会踩这个坑的是：块长调大、语料换成 PDF 长页、或 `max_length` 被调小。
    产物里的 `truncated_pairs` 字段就是为了让这个风险**可见** ——
    它是 0 时，三种策略的讨论就没有实际意义。
    """

    TAIL = "tail"
    """保留**开头**、丢结尾。默认。理由：块的开头通常是块首句（切分器的边界保证），
    信息密度最高。实测答案在开头时得分 0.92。"""

    HEAD = "head"
    """保留**结尾**、丢开头。适合"答案在段末"的场景（结论句、表格最后一行）。
    实测答案在末尾时得分 0.65（比 TAIL 的 0.92 低，因为块首的上下文也丢了）。"""

    MIDDLE = "middle"
    """保留**中段**、两头都丢。实测答案居中时得分 0.78 ——
    三种策略里对"居中"位置效果最好的，但它同时丢掉块首与块尾。"""


def sigmoid(x: float) -> float:
    """数值稳定的 sigmoid。**大负数不能直接 `1/(1+exp(-x))`**（exp 溢出）。

    实测 logit 范围约 -12~+12，其中的负数侧 exp(-(-12))=e^12 还在安全区，
    但模型输出没有硬上界，所以按标准写法分段。
    """
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def normalize_within_query(scores: Sequence[float], method: str = "minmax") -> list[float]:
    """把一个 query 的**这一组精排分数**归一化。

    与 `fusion.normalize_within_query` 是同一纪律的第二次落地：
    **参数是"一个 query 的候选分数"，拿不到全库上下文**，所以做不出跨 query 归一。

    ## 为什么 rerank 也需要它

    需求通常是"设一个阈值，低于就不算相关"。而 Cross-Encoder 的原始 sigmoid
    分数**跨 query 不可比**（见模块 docstring 必答问题 B），
    所以阈值必须建立在 query 内归一化后的相对位置上：

        minmax     : 这个 query 里最好的候选 = 1.0，最差的 = 0.0
        percentile : 按名次分档，0.0 = 最后一名，1.0 = 第一名

    **percentile 比 minmax 更适合设阈值**：它对"一条特别强的候选把其余全压扁"
    这种情况更稳（minmax 下如果 top1 是 0.99、其余 0.4，其余全被压到接近 0）。
    """
    if not scores:
        return []
    if method == "minmax":
        low, high = min(scores), max(scores)
        if high - low < 1e-12:
            return [1.0] * len(scores)
        return [(s - low) / (high - low) for s in scores]
    if method == "percentile":
        n = len(scores)
        if n == 1:
            return [1.0]
        # 名次 → [0, 1]。分数最高的拿 1.0，最低的拿 0.0。
        order = sorted(range(n), key=lambda i: scores[i])
        ranks = {idx: pos for pos, idx in enumerate(order)}
        return [ranks[i] / (n - 1) for i in range(n)]
    raise ValueError(f"未知归一化方式：{method!r}（可用：minmax / percentile）")


@dataclass(frozen=True, slots=True)
class RerankScore:
    """一对 (query, doc) 的打分结果。"""

    chunk_id: str
    logit: float
    """模型原始输出（无界）。"""

    score: float
    """sigmoid(logit)，(0, 1)。**跨 query 不可比**，见模块 docstring。"""

    rank_before: int | None = None
    """精排前的名次（来自上游检索，1-based）。用于回答「精排把它挪了几位」。

    为什么必须带着它：**「精排有没有用」的判据是「gold 的名次变没变好」**，
    只看精排后的顺序看不出这件事 —— 你不知道它原本在哪。
    """

    rank_after: int | None = None
    """精排后的名次（1-based），由 `rerank()` 回填。"""

    @property
    def delta(self) -> int | None:
        """名次变化。**负数 = 往前挪（变好）**，正数 = 往后。"""
        if self.rank_before is None or self.rank_after is None:
            return None
        return self.rank_after - self.rank_before


@dataclass
class RerankReport:
    """一次精排的元数据。**延迟必须记下来**（本周最关键的一步）。"""

    pairs: int
    batch_size: int
    max_length: int
    truncation: str
    elapsed_ms: int
    device: str
    model_path: str
    truncated_pairs: int = 0
    """有多少对被截断了。**这个数决定"截断讨论"是不是空谈** ——
    如果 0 对超长，那三种策略的差异就无从体现。"""

    def as_dict(self) -> dict[str, object]:
        return {
            "pairs": self.pairs,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "truncation": self.truncation,
            "elapsed_ms": self.elapsed_ms,
            "ms_per_pair": round(self.elapsed_ms / self.pairs, 2) if self.pairs else 0.0,
            "device": self.device,
            "model_path": self.model_path,
            "truncated_pairs": self.truncated_pairs,
        }


@dataclass
class Reranker:
    """Cross-Encoder 精排器。**懒加载**：不 import 就不占内存。

    ## 为什么懒加载是必需的

    模型 2.27 GB。如果 `import app.rag.bm25.reranker` 就加载，
    那么所有 `pytest` 都会先等 5 秒 + 吃 2 GB 内存 ——
    而 830 条测试里只有几条真的用到精排。
    """

    model_path: Path = DEFAULT_MODEL_DIR
    max_length: int = DEFAULT_MAX_LENGTH
    batch_size: int = DEFAULT_BATCH_SIZE
    truncation: Truncation = Truncation.TAIL
    device: str = "cpu"
    sort_by_length: bool = True
    """batch 前是否按 token 长度排序。**默认开，这是实测出来的关键优化。**

    关掉它（按检索顺序切 batch）时，实测**批量越大越慢**：
    batch=32 的 ms/对是 batch=1 的 2.1 倍。原因见 `rerank()` 里的 padding 说明。
    `scripts/rerank_probe.py` 会同时跑两种模式做对照。
    """

    _model: object | None = field(default=None, repr=False)
    _tokenizer: object | None = field(default=None, repr=False)
    _load_ms: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """**把 `truncation` 规整成枚举，错误立刻抛。**

        第一版不校验，于是 `Reranker(truncation="tail")` 这种写法能构造成功，
        直到第一次 `rerank()` 才在 `self.truncation.value` 上炸出
        `AttributeError: 'str' object has no attribute 'value'` ——
        **错误发生在离现场很远的地方，且报的是 attribute 而不是"参数类型错了"。**

        `Truncation` 是 `StrEnum`，所以字符串 `"tail"` 在构造时能直接匹配，
        这里顺手接住它（便利），但同时保证内部状态一定是枚举（正确性）。
        """
        if isinstance(self.truncation, str) and not isinstance(self.truncation, Truncation):
            try:
                self.truncation = Truncation(self.truncation)
            except ValueError as exc:
                valid = "、".join(item.value for item in Truncation)
                raise ValueError(
                    f"truncation 取值非法：{self.truncation!r}（可用：{valid}）"
                ) from exc
        if not isinstance(self.truncation, Truncation):
            raise TypeError(
                f"truncation 必须是 Truncation 或它的字符串值，收到 {type(self.truncation).__name__}"
            )
        if self.max_length <= 0:
            raise ValueError(f"max_length 必须为正数，收到 {self.max_length}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size 必须为正数，收到 {self.batch_size}")

    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_ms(self) -> int:
        """加载耗时。**它常常比精排本身还长**，必须单独记。"""
        return self._load_ms

    def load(self) -> None:
        """加载模型与分词器。**用本地路径，不碰网络。**"""
        if self._model is not None:
            return
        started = time.perf_counter()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"reranker 模型目录不存在：{path}\n"
                "先跑 `python -X utf8 scripts/fetch_reranker.py`（从 ModelScope 拉，约 2.3 GB）\n"
                "或用 RERANKER_MODEL_DIR 指向已有目录。"
            )
        self._tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(path), local_files_only=True
        )
        self._model.to(self.device)  # type: ignore[attr-defined]
        self._model.eval()  # type: ignore[attr-defined]
        self._load_ms = int((time.perf_counter() - started) * 1000)

    def _encode_pairs(
        self, pairs: Sequence[tuple[str, str]]
    ) -> tuple[list[dict[str, object]], int]:
        """把 (query, doc) 切成模型输入，并按策略截断 doc。

        ## 截断在**分词之后、模型之前**做

        不能用字符串级别的 `text[:n]` —— 中文字数与 token 数不是线性的
        （代码块、表格的 token 密度高得多）。必须按 token 数截。

        ## 为什么先拼再截

        模型输入是 `query + [SEP] + doc`。如果先按 token 数截 doc，
        再拼 query，总长会超 —— 所以这里给 query 留出预算：

            doc 预算 = max_length - query_token 数 - 特殊 token 数

        这个预算算法是**保守的**：它假设 query 占满了才截 doc。
        """
        tokenizer = self._tokenizer
        assert tokenizer is not None

        encoded: list[dict[str, object]] = []
        truncated = 0
        for query, doc in pairs:
            # 先单独看 query 有多长，给它留预算
            query_ids = tokenizer(query, add_special_tokens=False)["input_ids"]
            # 留 4 个特殊 token 的余量（[CLS] [SEP] 两侧 + 安全垫）
            budget = max(8, self.max_length - len(query_ids) - 4)

            doc_ids = tokenizer(doc, add_special_tokens=False)["input_ids"]
            if len(doc_ids) > budget:
                truncated += 1
                doc_ids = self._truncate_ids(doc_ids, budget)

            # **按 token id 还原成文本再喂 tokenizer**：直接传 id 会绕开
            # 模型要求的 pair 拼接格式（[CLS] q [SEP] d [SEP]）。
            truncated_text = tokenizer.decode(doc_ids, skip_special_tokens=True)
            item = tokenizer(
                query,
                truncated_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
            )
            encoded.append(dict(item))
        return encoded, truncated

    def _truncate_ids(self, ids: list[int], budget: int) -> list[int]:
        """按策略保留 `budget` 个 token。"""
        if self.truncation is Truncation.TAIL:
            return ids[:budget]
        if self.truncation is Truncation.HEAD:
            return ids[-budget:]
        # MIDDLE：两头各丢一半
        drop = len(ids) - budget
        begin = drop // 2
        return ids[begin : begin + budget]

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, str]],
        *,
        top_k: int | None = None,
    ) -> tuple[list[RerankScore], RerankReport]:
        """对候选精排。`candidates` 是 `(chunk_id, text)` 的序列。

        ## 为什么候选里要带 chunk_id

        否则调用方得自己按文本反查 `Chunk` —— 而块文本可能重复（不同位置的相同内容），
        反查会静默取到错的那条。
        """
        if not candidates:
            empty = RerankReport(
                pairs=0,
                batch_size=self.batch_size,
                max_length=self.max_length,
                truncation=self.truncation.value,
                elapsed_ms=0,
                device=self.device,
                model_path=str(self.model_path),
            )
            return [], empty

        self.load()
        import torch

        started = time.perf_counter()
        pairs = [(query, text) for _cid, text in candidates]
        encoded, truncated = self._encode_pairs(pairs)
        model = self._model
        assert model is not None

        # ## batch 前先按长度排序 —— 这是实测出来的关键优化
        #
        # 第一版按传入顺序切 batch，结果**批量越大越慢**（实测 batch=32 是
        # batch=1 的 2.1 倍 ms/对）。原因是 padding：
        #
        #   50 条候选的中位 token 数 113，最长 542（表格块）
        #   batch=32 时，一个 batch 里只要有那条 542，其余 31 条全要补到 542
        #   → 实际 token 6721，padding 后 22174，**浪费 230%**
        #
        # 按长度排序后，长块与长块同组、短块与短块同组，padding 浪费降到个位数。
        # **排序只影响 batch 组成，不影响最终分数** —— 每个 (query, doc) 对
        # 的打分互相独立，模型不回传跨样本信息。
        #
        # 对照见 `scripts/rerank_probe.py` 的 `padding_waste` 一节。
        if self.sort_by_length:
            order = sorted(
                range(len(encoded)),
                key=lambda i: len(encoded[i]["input_ids"]),  # type: ignore[arg-type]
            )
        else:
            order = list(range(len(encoded)))
        logits_by_index: list[float | None] = [None] * len(encoded)
        with torch.no_grad():
            for start in range(0, len(order), self.batch_size):
                batch_indices = order[start : start + self.batch_size]
                batch = [encoded[i] for i in batch_indices]
                inputs = self._collate(batch)
                output = model(**inputs)
                for offset, value in enumerate(output.logits.view(-1)):
                    logits_by_index[batch_indices[offset]] = float(value)

        logits: list[float] = [v if v is not None else 0.0 for v in logits_by_index]

        elapsed_ms = int((time.perf_counter() - started) * 1000)

        # **`rank_before` 来自传入顺序**：调用方按上游检索的排名传进来，
        # 这里把它记下来，精排后才能算出"挪了几位"。
        before = {cid: index for index, (cid, _text) in enumerate(candidates, start=1)}
        sorted_pairs = sorted(
            zip(candidates, logits, strict=True),
            key=lambda pair: (-sigmoid(pair[1]), pair[0][0]),
        )
        scores = [
            RerankScore(
                chunk_id=cid,
                logit=logit,
                score=sigmoid(logit),
                rank_before=before[cid],
                rank_after=index,
            )
            for index, ((cid, _text), logit) in enumerate(sorted_pairs, start=1)
        ]

        report = RerankReport(
            pairs=len(candidates),
            batch_size=self.batch_size,
            max_length=self.max_length,
            truncation=self.truncation.value,
            elapsed_ms=elapsed_ms,
            device=self.device,
            model_path=str(self.model_path),
            truncated_pairs=truncated,
        )
        if top_k is not None:
            scores = scores[:top_k]
        return scores, report

    def _collate(self, batch: Sequence[dict[str, object]]) -> dict[str, object]:
        """把一个 batch 补齐成等长的张量。

        ## padding 必须在这里做

        `tokenizer(..., return_tensors=None)` 给出的是变长 list。
        batch 内要手动 pad 到最长那条 —— `padding=True` 也能做，但它要求每条单独调，
        而我们为了截断预算已经单独编码过了。
        """
        import torch

        max_len = max(len(item["input_ids"]) for item in batch)  # type: ignore[arg-type]
        batch_size = len(batch)
        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        attention = torch.zeros((batch_size, max_len), dtype=torch.long)
        pad_id = 0
        tokenizer = self._tokenizer
        assert tokenizer is not None
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        for row, item in enumerate(batch):
            ids = list(item["input_ids"])  # type: ignore[arg-type]
            length = len(ids)
            input_ids[row, :length] = torch.tensor(ids, dtype=torch.long)
            attention[row, :length] = 1
            if length < max_len:
                input_ids[row, length:] = pad_id
        return {"input_ids": input_ids, "attention_mask": attention}


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_MODEL_DIR",
    "RerankReport",
    "RerankScore",
    "Reranker",
    "Truncation",
    "normalize_within_query",
    "sigmoid",
]
