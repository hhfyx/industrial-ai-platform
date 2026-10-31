"""可选重依赖的可用性探测。

**为什么单独一个模块，而不是在测试里 `try: import torch`**：

1. `tests/test_embedding.py` 与 `scripts/*.py` 都要判断，各写一份会漂移
2. **探测本身要便宜**：不能为了判断"torch 装没装"就真的 `import torch`（6 秒起）。
   这里用 `importlib.util.find_spec` —— 只查模块是否存在，不执行它

## 为什么不把 torch 做成可选依赖组（`[project.optional-dependencies]`）

本项目要"拔网线 `pytest -q` 全绿"（`tests/conftest.py` 的硬规矩）。
torch 装进主依赖之后，CI 上 `uv sync` 会拉 900 MB —— 而 CI 里
**没有一个测试需要它**（bge 相关测试全部 skip）。

但"移出主依赖"也有代价：`uv sync` 之后本地跑不了 `bge_prefix_ab.py`，
得记得 `uv sync --extra embedding`。对一个还在演进的项目，**默认能跑**比**默认精简**重要。

所以现在的选择是：**保留在主依赖里，但测试对它完全免疫**。
代价写在 `docs/design-decisions.md` 第二条（镜像 +900 MB）。
"""

from __future__ import annotations

import importlib.util


def _module_exists(name: str) -> bool:
    """模块是否**存在**（不执行它）。`find_spec` 只查 sys.path，不 import。"""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # 父包缺失时 find_spec 会抛，视同"不存在"
        return False


def bge_available() -> bool:
    """bge embedder 能否工作：需要 torch + sentence-transformers + 模型文件。

    **不检查模型文件是否已下载**：那要访问 HF 缓存目录，会引入"测试依赖
    某个人的本地目录结构"这种脆弱性。模型没下载时，
    第一个测试会因为加载失败而报错 —— 那个错误信息比"skip"更有用，
    因为它告诉你"环境不全"，而不是让你以为"这些测试本来就不跑"。
    """
    return _module_exists("torch") and _module_exists("sentence_transformers")


def numpy_available() -> bool:
    """numpy 是否可用。`test_embedding` 里手写余弦的对照测试需要它。"""
    return _module_exists("numpy")


def chroma_available() -> bool:
    """chromadb 是否可用。

    **`ChromaStore` 的测试全部挂在它上面**：没装 chromadb 的环境里那批测试 skip，
    而不是让整套测试红 —— 与 `bge_available` 同一个处理。

    注意这**不是**在削弱测试：装了 chromadb 的机器（本机、完整镜像）
    会跑全部测试。"可选依赖缺失时降级"和"测试覆盖被永久削弱"是两件事。
    """
    return _module_exists("chromadb")


__all__ = ["bge_available", "chroma_available", "numpy_available"]
