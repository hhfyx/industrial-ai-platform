"""容器内健康检查：GET http://127.0.0.1:$PORT/health。

用 Python 标准库而不是 curl，理由：python:3.11-slim 不带 curl，
为了一行探针去 apt-get 装 curl + libcurl 会让镜像白胖十几 MB。

注意：这里的 127.0.0.1 是**容器自己的**回环地址 —— 这正是探针想要的效果
（探的是容器内进程有没有活着），和"宿主机怎么访问容器"是两件事。
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request


def probe(port: str | int, timeout: float = 2.0) -> int:
    """返回 0 表示健康，1 表示不健康。"""
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - 固定回环地址
            return 0 if resp.status == 200 else 1
    except (urllib.error.URLError, OSError, ValueError):
        return 1


if __name__ == "__main__":
    sys.exit(probe(os.environ.get("PORT", "8000")))
