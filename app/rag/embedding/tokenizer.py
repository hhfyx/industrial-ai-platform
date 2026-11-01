"""用 **bge 自己的 tokenizer** 数 token。不是 tiktoken，也不是"字符数 × 系数"。

## 为什么必须用真 tokenizer

`docs/embedding.md` 里那条"500 字 ≈ 350 token，在安全范围内"是拿
**0.676 tok/字** 的系数折算出来的 —— 而那个系数来自另一个模型网关的
`usage` 反推（`scripts/token_result.json`），**不是 bge 的分词结果**。

两件事足以让这个折算失效：

1. **bge 的 vocab 是字级的**。实测 `付款计划` → `['付','款','计','划']`，
   即 4 个汉字 4 个 token（1.0 tok/字），而不是 0.676。
2. **ASCII / 数字 / 标点会合并**，而中文标点、全角符号、markdown 表格竖线
   往往单独成 token。同一批块里混着这三类文本时，**单一系数必然错**。

系数错的方向是可以预先判断的：0.676 < 1.0 意味着用 0.676 折算会**低估** token 数，
也就是**高估**"500 字有多安全"。这正是 W4 的 C3 题（超长块被静默截断）要落地的地方。

所以这里直接调 bge 的 tokenizer，`max_length` 取模型自报的 `max_seq_length`，
一条一条数出真实长度。

## 为什么 `add_special_tokens=True`

bge 的输入实际是 `[CLS] ... [SEP]`，这两个 token 占位置。切成 500 字后
真实长度是 `body_tokens + 2`。差 2 对 512 的边界判断不是无关紧要的：
越界判定的口径必须是"**喂给模型的序列有多长**"，而不是"正文有多少 token"。

## 为什么要单独抽一个函数而不是在脚本里写

两个调用方：`scripts/token_audit.py`（数全部块）和 `scripts/grid_search.py`
（每组参数报告"平均块 token"）。两处各写一遍 `tokenizer(text)["input_ids"]`
就会各写一遍 `add_special_tokens` 的选择 —— 而那种不一致只在边界值上暴露
（499 字的块一个算 501 一个算 499），最难发现。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

# 与 `bge.py` 同一个模型名。**不重复定义常量**，从那里取。
from app.rag.embedding.bge import DEFAULT_HF_ENDPOINT, DEFAULT_MODEL_NAME, HF_ENDPOINT_ENV

logger = logging.getLogger(__name__)

# bge-small-zh-v1.5 的 max_seq_length。**这是兜底值，不是真相**：
# 真相由 `tokenizer.model_max_length` 给出，加载后会核对（见 `BGETokenizer.max_seq_length`）。
FALLBACK_MAX_TOKENS: int = 512


class BGETokenizer:
    """bge-small-zh-v1.5 的分词器封装。**懒加载**：`import` 它不该拉 transformers。"""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._tokenizer: Any = None
        self._max_seq_length: int | None = None

    def _ensure(self) -> Any:  # noqa: ANN401
        if self._tokenizer is not None:
            return self._tokenizer
        import os

        # 镜像地址必须在 import transformers 之前设好 —— 与 `bge.py::_ensure_model` 同一条纪律。
        os.environ.setdefault(HF_ENDPOINT_ENV, DEFAULT_HF_ENDPOINT)
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的清晰报错
            raise RuntimeError(
                f"缺 transformers（{exc}）。token 审计需要 bge 自己的分词器，"
                "不能用字符数折算代替 —— 理由见 `app/rag/embedding/tokenizer.py` 的模块说明。"
            ) from exc

        logger.info("加载 tokenizer %s …", self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._tokenizer = tokenizer
        # `model_max_length` 在个别版本上是 `int` 上限常量（1e30），那种值不可用，
        # 回退到 bge 的实测值 512 并**记一条 warning**（不是静默）。
        raw = getattr(tokenizer, "model_max_length", None)
        if isinstance(raw, int) and 0 < raw <= 1_000_000:
            self._max_seq_length = raw
        else:
            logger.warning(
                "tokenizer 的 model_max_length=%r 不可用，回退到 bge 实测值 %d",
                raw,
                FALLBACK_MAX_TOKENS,
            )
            self._max_seq_length = FALLBACK_MAX_TOKENS
        return tokenizer

    @property
    def max_seq_length(self) -> int:
        """模型能接受的最大 token 数。超过会被**静默截断**（不是报错）。"""
        self._ensure()
        return self._max_seq_length or FALLBACK_MAX_TOKENS

    def count(self, texts: Sequence[str]) -> list[int]:
        """批量数 token。**返回顺序与入参严格一致**，长度相同。

        空输入返回空列表（与 `Embedder.embed` 同口径，不抛异常）。
        """
        if not texts:
            return []
        tokenizer = self._ensure()
        # 不传 `truncation` / `padding`：这里要的正是**未截断的真实长度**。
        # 传了 `truncation=True` 会得到一堆正好 512 的数，那等于把要测的东西测没了。
        encoded = tokenizer(list(texts), add_special_tokens=True)
        return [len(ids) for ids in encoded["input_ids"]]

    def count_one(self, text: str) -> int:
        return self.count([text])[0]


__all__ = ["FALLBACK_MAX_TOKENS", "BGETokenizer"]
