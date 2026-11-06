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

## W8 周日新增：结构化日志（JSON 行）

`LOG_FORMAT=json` 时每条日志一行 JSON：
`{"time","level","logger","message","request_id","extra..."}`。

**为什么 request_id 用 contextvar 而不是参数**：
贯穿意味着"中间件生成的 id，在**任意深处**的 logger 调用里都出现"——
param 传递要求每个函数签名都带 request_id 再透传，那是侵入式改造。
`contextvars.ContextVar` 是 asyncio 原生的请求级上下文：
中间件 set 一次，同一请求的所有协程（含 to_thread 之外的部分）读到的都是它，
不同请求之间天然隔离。uvicorn 的每个请求跑在自己的 task 上下文里，
这正是 contextvar 的设计场景。

**错误分类计数**（`/metrics` 的数据源）：`count_error(code)` 按
错误码累计（进程内计数器）。轻量做的理由：这是单实例演示服务，
不是集群——上 Prometheus 的时机是"有第二个实例"那天。
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from collections import Counter
from typing import Any, Final

# 应用自己的 logger 前缀。只配置这个命名空间，
# 不碰 uvicorn / httpx 的 logger（它们有自己的 handler 和级别）。
APP_LOGGER_NAME: Final[str] = "app"

_FORMAT: Final[str] = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATE_FORMAT: Final[str] = "%H:%M:%S"

# 请求级上下文：中间件 set，全链路读（见模块 docstring 的"为什么"）
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")

# 错误分类计数（进程内）。key = 错误码（如 INVALID_INPUT / GPU_OOM）
_error_counts: Counter[str] = Counter()


def set_request_id(request_id: str) -> None:
    """中间件在请求入口调用。整个请求链路的日志都带这个 id。"""
    request_id_var.set(request_id)


def get_request_id() -> str:
    return request_id_var.get()


def count_error(code: str) -> None:
    """按错误码累计。供 `/metrics` 或日志审计读（见 `_error_counts` 的说明）。"""
    _error_counts[code] += 1


def error_counts() -> dict[str, int]:
    """当前累计的错误分类计数（拷贝，防外部改动）。"""
    return dict(_error_counts)


class _JsonFormatter(logging.Formatter):
    """JSON 行格式：一条日志 = 一行可 grep 的 JSON。

    `extra` 里的键平铺进顶层（logging 的惯例是塞进 `record.__dict__`，
    这里挑出非标准属性），消息本体放 `message`。
    """

    # logging.Record 的标准属性（不进 extra 平铺）
    _RESERVED: Final[frozenset[str]] = frozenset(
        {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "taskName",
            "message",
            "asctime",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _make_handler(level: str, fmt: str) -> logging.Handler:
    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def configure_logging(*, level: str = "INFO", force: bool = False, fmt: str | None = None) -> None:
    """配置应用 logger。幂等（除非 force=True）。

    参数 `force` 主要给测试用：某些用例要验证日志确实输出，
    需要重新安装 handler。

    `fmt`：`"json"` = JSON 行；`"text"`（默认）= 人读格式。
    不传读环境变量 `LOG_FORMAT`。
    """
    import os

    resolved = fmt or os.environ.get("LOG_FORMAT", "text")
    app_logger = logging.getLogger(APP_LOGGER_NAME)

    if not force and getattr(app_logger, "_dsh_configured", False):
        return

    # 清理旧的 handler（reload / 多次调用时会出现）
    for handler in list(app_logger.handlers):
        app_logger.removeHandler(handler)
        handler.close()

    app_logger.addHandler(_make_handler(level, resolved))
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


__all__ = [
    "APP_LOGGER_NAME",
    "configure_logging",
    "count_error",
    "error_counts",
    "get_request_id",
    "set_request_id",
    "setup_from_settings",
]
