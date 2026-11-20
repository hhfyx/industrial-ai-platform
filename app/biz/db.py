"""MySQL 引擎与 Session 装配（docs/module-contract-approval.md §八 Day1 上午）。

装配纪律与 `app/core/deps.py` 同款：

- 引擎与 sessionmaker 都是 `lru_cache` 进程级单例 —— 连接池绝不能每请求新建；
- 配置从环境变量 `BIZ_DB_URL` 读，**未设置时引擎拒绝构造**（显式失败优于
  悄悄退化成 SQLite —— 本模块的价值全在 MySQL 的真实并发行为上，
  见方案文档「提醒 1」）。测试用 `reset_db_cache()` 换实现。
"""

from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

DB_URL_ENV = "BIZ_DB_URL"
POOL_SIZE_ENV = "BIZ_DB_POOL_SIZE"

# 默认连接池 20：并发测试要同时开 20 个事务（方案文档 §5.1），
# 池小于并发数时请求会排队等连接 —— 不是死锁，但会让 409 变成慢响应。
DEFAULT_POOL_SIZE = 20


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """进程级唯一的 async 引擎。url 缺失时抛错 —— 配置错误挡在启动时。"""
    url = os.environ.get(DB_URL_ENV, "").strip()
    if not url:
        raise RuntimeError(
            f"环境变量 {DB_URL_ENV} 未设置。"
            "本模块只支持 MySQL（并发/ENUM/外键行为必须在 MySQL 上验证），"
            "不要用 SQLite 替代 —— 见 docs/module-contract-approval.md 提醒 1。"
        )
    return create_async_engine(
        url,
        pool_size=_int_env(POOL_SIZE_ENV, DEFAULT_POOL_SIZE),
        pool_pre_ping=True,  # MySQL 8 小时断连（wait_timeout）用 pre_ping 自愈
        pool_recycle=3600,
    )


@lru_cache(maxsize=1)
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """进程级唯一的 sessionmaker。事务边界由调用方 `async with session.begin()` 控制。"""
    return async_sessionmaker(get_engine(), expire_on_commit=False)


def reset_db_cache() -> None:
    """清掉引擎/sessionmaker 单例。测试切换 BIZ_DB_URL 后必须调用。"""
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


async def dispose_engine() -> None:
    """优雅关闭连接池（应用 shutdown 时调用；测试兜底也用它）。"""
    engine = get_engine()
    await engine.dispose()
    reset_db_cache()


__all__ = [
    "DB_URL_ENV",
    "POOL_SIZE_ENV",
    "dispose_engine",
    "get_engine",
    "get_sessionmaker",
    "reset_db_cache",
]
