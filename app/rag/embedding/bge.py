"""BGE 中文小模型（sentence-transformers）：真正的语义向量化。

## 一句话定位

这是把 `HashingEmbedder` 换掉的那个实现。**换掉它之前，本项目所有召回率数字
都没有参考价值**（理由见 `hashing.py` 的 docstring）；换上它之后，数字才开始有意义。

## 三个必须写清的实测事实

### 一、维度 512、`max_seq_length` 512 token

    实测：model.get_embedding_dimension() == 512
          model.max_seq_length == 512

两个 512 是**不相干的两件事**，很容易被混成"512 维 512 字"：
- **维度 512** → 存储成本：每条向量 512 × 4 字节 = 2 KB（fp32）
- **最大长度 512 token** → 超了会被**静默截断**，不是报错

按中文 0.676 tok/字算，512 token ≈ **757 字**。本项目切块是 500 字
（`ChunkSize.chunk_chars = 500`）≈ 350 token，**安全范围内**。
但表格块不切（见 `splitters/recursive.py`）—— 一张 2000 字的大表会被截断到
前 757 字，这是**已知的、必须写进文档的损失**，不是 bug。

### 二、query 侧要加前缀，passage 侧不加

bge-zh v1.5 的检索用法：

    query   = "为这个句子生成表示以用于检索相关文章：" + 原始问题
    passage = 原文，不加任何前缀

这个约定只写在模型的 README 里，**不会在代码层强制**。忘了加前缀不会有任何报错，
只会让命中率低一点 —— 而"低一点"最容易被归因到别处（切分、top_k、模型不行）。

所以这里把它做成**接口级**的事：`embed()` 只用于 passage，
`embed_query()` 负责加前缀。`Retriever` 只调 `embed_query` ——
**让"忘加前缀"在结构上不可能发生**。

### 三、输出**已经是归一化的**，但本实现仍然再归一化一次

实测：sentence-transformers 加载后 `encode()` 出来的向量模长全 == 1.000000
（`modules.json` 里有 Normalize 模块）。用 transformers 直调模型 + CLS pooling
拿到的是**裸向量**，模长 6.36~6.92 —— 两者归一化后完全一致（atol 1e-5）。

那为什么还要再归一化一次？两个理由：
1. **幂等**，成本是 O(d)，可以忽略
2. **不依赖对上游行为的记忆**。哪天 sentence-transformers 改了模块链、
   或者有人换成 `transformers` 直调，归一化这一层要能自己兜住。

## 延迟与内存（实测，CPU / Ryzen 级 / torch 2.15+cpu）

| 项 | 实测 |
| --- | --- |
| import sentence-transformers | **65 s**（首次，含 torch 加载） |
| 模型加载 | 17 s |
| 首次 encode（含图优化） | 77 ms |
| 单条 encode P50 / P95（预热后） | **10.1 ms / 10.6 ms** |
| 批量吞吐 batch=1 / 8 / 32 / 64 | 103 / 284 / 371 / **408** 条/秒 |
| 模型文件 | 91.9 MB |
| 进程常驻内存 | 约 400 MB（torch + 模型） |

**结论：批量必须开。** batch=64 比 batch=1 快 4 倍 —— 这正是
`Embedder` 接口里没有 `embed_one` 的原因（见 `base.py`）。
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence

from app.rag.embedding.vector import l2_normalize

logger = logging.getLogger(__name__)

# bge-zh v1.5 的官方检索指令前缀。
DEFAULT_QUERY_INSTRUCTION: str = "为这个句子生成表示以用于检索相关文章："

# 模型 ID。放在常量里而不是散落在调用处 —— 换模型时只改这里，
# 而且它会进索引元数据（`name` 属性），"这批向量是谁产的"可查。
DEFAULT_MODEL_NAME: str = "BAAI/bge-small-zh-v1.5"

# 国内直连 huggingface.co 实测不通（DNS 解析失败），hf-mirror.com 可用。
# 用环境变量而不是硬编：硬编会让"公司在用私有镜像"的人被迫改代码。
HF_ENDPOINT_ENV: str = "HF_ENDPOINT"
DEFAULT_HF_ENDPOINT: str = "https://hf-mirror.com"


class BGEEmbedder:
    """bge-small-zh-v1.5 的 sentence-transformers 封装。

    **模型是懒加载的**（`_model` 在第一次 `embed` 时才构造）：
    `import` 这个模块不该花 65 秒，也不该在 CI 里触发一次模型下载。
    这条和 `app/core/config.py` 的"import 不解析配置"是同一个原则 ——
    早炸好过半夜炸，但**不该在不需要的时候炸**。
    """

    name = "bge-small-zh"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str = "cpu",
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        batch_size: int = 32,
        max_seq_length: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self._max_seq_length = max_seq_length
        self._model = None
        # 512 是 bge-small-zh-v1.5 的**实测**维度。在这里写死一个常量，
        # 让 `dimension` 在模型加载前就能回答（建集合需要它）。
        # 真实值在 `_ensure_model` 里会与模型自报的维度核对 —— 不一致就炸。
        self.dimension = 512

    # ------------------------------------------------------------------ 生命周期

    def _ensure_model(self) -> object:
        """懒加载模型。**首次调用会下载约 92 MB**，且加载耗 17 秒。"""
        if self._model is not None:
            return self._model

        # 镜像地址必须在 import sentence_transformers 之前设好 ——
        # huggingface_hub 在模块导入时就读走环境变量。
        os.environ.setdefault(HF_ENDPOINT_ENV, DEFAULT_HF_ENDPOINT)

        started = time.perf_counter()
        from sentence_transformers import SentenceTransformer

        logger.info("加载 embedding 模型 %s（首次会下载约 92 MB）…", self.model_name)
        model = SentenceTransformer(self.model_name, device=self.device)
        if self._max_seq_length is not None:
            model.max_seq_length = self._max_seq_length

        # **核对维度**：写死的 512 与模型自报的值不一致时立刻炸。
        # 不一致意味着"已入库的向量维度不匹配"—— 那会让检索静默返回错结果，
        # 比加载失败糟得多。
        # `get_embedding_dimension` 是 sentence-transformers 6.x 的新名，
        # 旧版只有 `get_sentence_embedding_dimension`。两个都试：
        # **不能只写旧名** —— 6.x 会在每次调用时刷 FutureWarning，
        # 而"每次跑测试都刷一条废警告"的结果是所有人开始无视警告。
        getter = (
            getattr(model, "get_embedding_dimension", None)
            or model.get_sentence_embedding_dimension
        )
        actual = getter()
        if actual != self.dimension:
            raise RuntimeError(
                f"模型 {self.model_name} 维度为 {actual}，与声明的 {self.dimension} 不一致；"
                "换模型必须同时重建索引"
            )

        self._model = model
        logger.info(
            "模型加载完成：%d 维，%d token，耗时 %.1f s",
            actual,
            model.max_seq_length,
            time.perf_counter() - started,
        )
        return model

    def warmup(self) -> None:
        """预热：加载模型 + 跑一次 encode。

        **为什么要单独给一个预热入口**：首次 encode 比预热后慢（实测 77 ms vs 10 ms，
        差 7 倍，来自图优化与内存分配）。线上第一个真实请求不该承担这个成本 ——
        那会让"服务刚启动时特别慢"变成一个查不出原因的玄学问题。

        在 `on_startup` 里调一次即可。测试不调（没必要为测试付 17 秒）。
        """
        self._ensure_model()
        self.embed(["预热"])

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载。测试与健康检查用它区分"没加载"和"加载失败"。"""
        return self._model is not None

    @property
    def max_seq_length(self) -> int:
        """最大 token 数。**超过会被静默截断**，不是报错。"""
        return self._max_seq_length or 512

    # ------------------------------------------------------------------ 向量化

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """**passage 侧**向量化：不加任何前缀。

        返回顺序与入参严格一致、每个向量已 L2 归一化（见 `base.Embedder` 的契约）。
        """
        if not texts:
            return []
        model = self._ensure_model()
        # `normalize_embeddings=True` 让 sentence-transformers 自己归一化；
        # 我们再过一遍 `l2_normalize` 是**有意的双保险**（见模块 docstring 第三节），
        # 成本 O(d)，换来的是"上游改了模块链也不会静默出错"。
        vectors = model.encode(  # type: ignore[attr-defined]
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [l2_normalize([float(x) for x in row]) for row in vectors]

    def embed_query(self, queries: Sequence[str]) -> list[list[float]]:
        """**query 侧**向量化：自动加检索指令前缀。

        `Retriever` 只调这个方法。**把前缀的处理放进接口**，
        是为了让"忘了加前缀"在结构上不可能发生 —— 而不是靠注释提醒。

        已经带前缀的输入不会被重复加（幂等）：调用方手工拼过前缀时不会得到
        `前缀前缀问题`。这个检查很便宜，但它省掉的是"前缀加两次"这种
        只有看日志才能发现的诡异现象。
        """
        if not queries:
            return []
        prefixed = [self._with_instruction(q) for q in queries]
        return self.embed(prefixed)

    def _with_instruction(self, text: str) -> str:
        """加前缀（幂等）。"""
        stripped = text.lstrip()
        if not self.query_instruction:
            return text
        if stripped.startswith(self.query_instruction.strip()):
            return text
        return self.query_instruction + text


__all__ = [
    "DEFAULT_HF_ENDPOINT",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_QUERY_INSTRUCTION",
    "BGEEmbedder",
]
