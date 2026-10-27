"""Loader 注册表：按文件后缀分发到具体 loader。

**设计要点：一张表决定一切。**

加新格式只在 `LOADERS` 里加一行，`load_document` 和 CLI/API 都不用改。
这跟 `TOOL_REGISTRY` 是同一个思路 —— "能力和它的分发方式绑在一起"，
避免出现"支持了格式但忘了注册"这类低级错误。

**为什么 `load_document` 返回 `LoadResult` 而不是抛异常**：
只有"文件打不开/格式不支持"才抛（那是真失败）；
"某页没文本""整份是扫描件"返回 warning ——
调用方需要区分"处理失败"和"处理成功但内容有问题"，这两件事的应对完全不同。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from app.rag import LoadResult
from app.rag.loaders.docx import load_docx
from app.rag.loaders.markdown import load_markdown
from app.rag.loaders.pdf import load_pdf

logger = logging.getLogger(__name__)

# 后缀 -> loader。加新格式就加一行。
LOADERS: dict[str, Callable[[str | Path], LoadResult]] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".md": load_markdown,
    ".markdown": load_markdown,
}

SUPPORTED_EXTENSIONS = tuple(sorted(LOADERS))


def load_document(path: str | Path) -> LoadResult:
    """按后缀分发加载。不支持的格式立刻抛 `ValueError`。

    后缀大小写不敏感（`.PDF` 也要认）：Windows 上用户经常拿到大写的文件，
    为此报错只会让人觉得工具不好用。
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    loader = LOADERS.get(suffix)
    if loader is None:
        supported = "、".join(SUPPORTED_EXTENSIONS)
        raise ValueError(f"不支持的格式 {suffix or '(无后缀)'}；当前支持：{supported}")

    if not file_path.exists():
        raise ValueError(f"文件不存在：{file_path}")

    logger.info("加载 %s（%s）", file_path.name, suffix)
    return loader(file_path)


__all__ = ["LOADERS", "SUPPORTED_EXTENSIONS", "load_document"]
