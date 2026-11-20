"""biz 数据库测试的共享夹具（tests/test_approve_concurrency.py 等用）。

三个纪律：

1. **没起 MySQL 容器就 skip，不是 fail** —— 离线套件（conftest.py 的总开关）
   必须永远绿；数据库测试显式加入（设 BIZ_DB_URL 或本地容器在跑）。
2. **每条测试独立数据 + 跑完清场** —— TRUNCATE 全部 5 张表，
   并发测试对"日志只多 1 条"的断言才不被历史数据污染。
3. **引擎随事件循环生灭（重要教训）** —— aiomysql 的池化连接绑定在
   创建它的循环上；跨 `asyncio.run()` 复用引擎 = 上个循环的连接悬空，
   析构时 RuntimeError: Event loop is closed。所以 helper 是
   `run_biz(fn)`：循环内建引擎 → 跑完 fn → dispose，全部在同一个循环里。

容器与 URL（见 docker-compose.biz.yml）：

    docker compose -f docker-compose.biz.yml up -d mysql
    $env:BIZ_DB_URL="mysql+aiomysql://biz:biz_dev@127.0.0.1:3307/biz_contracts"
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import TypeVar

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

BIZ_DB_URL_DEFAULT = "mysql+aiomysql://biz:biz_dev@127.0.0.1:3307/biz_contracts"

TABLES_IN_TEARDOWN_ORDER = (
    "approval_logs",
    "approval_nodes",
    "idempotency_keys",
    "contracts",
    "customers",
)

T = TypeVar("T")

_AVAILABLE_CACHE: bool | None = None


def _db_url() -> str:
    return os.environ.get("BIZ_DB_URL", "").strip() or BIZ_DB_URL_DEFAULT


def _make_engine(*, pool_size: int) -> AsyncEngine:
    from sqlalchemy.ext.asyncio import create_async_engine

    return create_async_engine(_db_url(), pool_size=pool_size, pool_pre_ping=True)


def db_available() -> bool:
    """连得上才算可用（容器没起时默认 URL 也不可用）。进程内探测一次并缓存。"""
    global _AVAILABLE_CACHE
    if _AVAILABLE_CACHE is None:

        async def _probe() -> bool:
            engine = _make_engine(pool_size=1)
            try:
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                return True
            except Exception:
                return False
            finally:
                await engine.dispose()  # 同一个循环内 await dispose

        try:
            _AVAILABLE_CACHE = asyncio.run(_probe())
        except Exception:
            _AVAILABLE_CACHE = False
    return _AVAILABLE_CACHE


def run_biz(fn: Callable[[AsyncEngine], Awaitable[T]], *, pool_size: int = 25) -> T:
    """在一个专用事件循环里：建引擎 → 跑 fn → dispose。

    并发测试要同时开 20+ 个事务，池必须盖得住（pool_size 默认 25）。
    """

    async def _runner() -> T:
        engine = _make_engine(pool_size=pool_size)
        try:
            return await fn(engine)
        finally:
            await engine.dispose()

    return asyncio.run(_runner())


def truncate_all(engine: AsyncEngine) -> None:
    """TRUNCATE 全部 5 张表（子表在前）。TRUNCATE 重置自增，
    让"日志只多 1 条"这类断言有确定起点。"""

    async def _do(engine: AsyncEngine) -> None:
        async with engine.begin() as conn:
            await conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
            for table in TABLES_IN_TEARDOWN_ORDER:
                await conn.execute(text(f"TRUNCATE TABLE {table}"))
            await conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

    run_biz(_do, pool_size=2)


def seed_contract(*, amount_cents: int, prefix: str) -> dict:
    """同步外壳：造 客户 + 合同（级数由金额定）并 submit，返回响应 dict。"""
    from app.biz.schemas import ContractCreate, CustomerCreate
    from app.biz.service import ContractService

    async def _seed(engine: AsyncEngine) -> dict:
        svc = ContractService(sessionmaker=async_sessionmaker(engine, expire_on_commit=False))
        cust, _ = await svc.create_customer(CustomerCreate(name=f"{prefix}-{os.urandom(4).hex()}"))
        contract, _ = await svc.create_contract(
            ContractCreate(
                contract_no=f"HT-{prefix}-{os.urandom(6).hex().upper()}",
                customer_id=cust["id"],
                title=f"{prefix} 测试合同",
                amount_cents=amount_cents,
            )
        )
        # submit 返回的是提交后的响应（status=pending, version=1）——
        # 返回旧 dict 会让调用方拿到 draft 的过期快照
        submitted = await svc.submit(contract["id"], actor="test-seeder")
        return submitted

    return run_biz(_seed)


def count_logs(contract_id: int) -> int:
    """审计日志计数（并发测试的验收锚点）。"""

    async def _count(engine: AsyncEngine) -> int:
        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT COUNT(*) FROM approval_logs WHERE contract_id = :cid"),
                {"cid": contract_id},
            )
            return int(result.scalar())

    return run_biz(_count, pool_size=2)


__all__ = [
    "BIZ_DB_URL_DEFAULT",
    "TABLES_IN_TEARDOWN_ORDER",
    "count_logs",
    "db_available",
    "run_biz",
    "seed_contract",
    "truncate_all",
]
