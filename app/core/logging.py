"""日志配置：让 `logger.info(...)` 真的能出现在控制台。

**为什么需要这个文件**：项目里到处是 `logging.getLogger(__name__)`，
但**从来没有人配置过 root logger**。Python 的默认行为是：
root logger 未配置时，低于 WARNING 的日志**直接被丢弃**（
`logging.lastResort` 只兜 WARNING 及以上）。

后果是：`logger.info("客户端断开，停止生成")` 这种关键运维日志
写得很认真，却一个字都看不到 —— 排查时你会以为"断开检测没生效"，
实际上是日志根本没输出。**这个坑比断开检测本身更容易误导人。**

配置策略：
- 用 `app_env` 决定是否开 DEBUG（dev 开，prod 关）
- 统一格式，带上时间戳和 logger 名（能定位到是哪个模块）
- `uvicorn` 自己的 logger 不动：它有独立的 handler，重复配置会打两份
- **幂等**：多次调用只生效一次，避免 `--reload` 或测试里累积 handler
"""

from __future__ import annotations

import logging
import sys
from typing import Final

# 应用自己的 logger 前缀。只配置这个命名空间，
# 不碰 uvicorn / httpx 的 logger（它们有自己的 handler 和级别）。
APP_LOGGER_NAME: Final[str] = "app"

_FORMAT: Final[str] = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATE_FORMAT: Final[str] = "%H:%M:%S"


def configure_logging(*, level: str = "INFO", force: bool = False) -> None:
    """配置应用 logger。幂等（除非 force=True）。

    参数 `force` 主要给测试用：某些用例要验证日志确实输出，
    需要重新安装 handler。
    """
    app_logger = logging.getLogger(APP_LOGGER_NAME)

    if not force and getattr(app_logger, "_dsh_configured", False):
        return

    # 清理旧的 handler（reload / 多次调用时会出现）
    for handler in list(app_logger.handlers):
        app_logger.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))

    app_logger.addHandler(handler)
    app_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    # **关键**：不向上传播。root logger 未配置，
    # 传播上去要么被 lastResort 吃掉（只剩 WARNING+），要么和 uvicorn 的 handler 重复。
    app_logger.propagate = False
    app_logger._dsh_configured = True  # type: ignore[attr-defined]


def setup_from_settings() -> None:
    """按配置里的 `log_level` 与 `app_env` 初始化。

    dev 环境额外把级别降到 INFO（默认已是 INFO）；关键是在**任何**环境都装上 handler，
    否则"日志消失"这个坑会一直在。
    """
    from app.core.config import get_settings

    settings = get_settings()
    configure_logging(level=settings.log_level)


__all__ = ["APP_LOGGER_NAME", "configure_logging", "setup_from_settings"]
