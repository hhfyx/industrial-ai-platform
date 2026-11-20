"""Alembic 环境：URL 从 BIZ_DB_URL（同步形态）读，元数据来自 app.biz.models。

BIZ_DB_URL 是 aiomysql 的异步 URL（mysql+aiomysql://...），
Alembic 的 env.py 默认走同步连接 —— 这里在入口把 driver 段替换成 pymysql，
两种 URL 共用同一个环境变量，不引入第二个配置项。
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from app.biz.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _sync_url() -> str:
    url = os.environ.get("BIZ_DB_URL", "").strip()
    if not url:
        raise RuntimeError(
            "BIZ_DB_URL 未设置。迁移与运行时共用同一个变量；"
            "MySQL 容器起法见 docker-compose.biz.yml 头部注释。"
        )
    # mysql+aiomysql:// -> mysql+pymysql://（Alembic 用同步连接执行迁移）
    return url.replace("+aiomysql", "+pymysql")


def run_migrations_offline() -> None:
    context.configure(
        url=_sync_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_sync_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
