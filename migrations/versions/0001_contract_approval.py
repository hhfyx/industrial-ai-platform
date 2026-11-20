"""contract approval: 5 tables (W14, docs/module-contract-approval.md §二)

Revision ID: 0001
Revises:
Create Date: 2026-11-16

5 张表全部由 ORM 元数据 autogenerate 而来（env.py 的 target_metadata），
这里保留手写版 —— 迁移文件是历史记录，必须可独立读。

设计要点（迁移里能直接看到的）：
- 主键 BIGINT unsigned：MySQL 方言类型，SQLAlchemy 2.1 起通用类型不再收 unsigned 参数
- contracts.version：乐观锁载体，NOT NULL DEFAULT 0
- contracts.amount_cents：金额用「分」存 BIGINT，不用 DECIMAL/FLOAT
- approval_logs：只增不改的审计表，(contract_id, created_at) 联合索引
- ENUM 直接用 MySQL ENUM（本模块只跑 MySQL，方言类型即显式前提）

Revision ID: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mysql import BIGINT as MySQL_BIGINT

revision: str = "0001"
down_revision: str | None = None
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "customers",
        sa.Column("id", MySQL_BIGINT(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("tax_no", sa.String(32), nullable=True),
        sa.Column("contact", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(3), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    op.create_table(
        "contracts",
        sa.Column("id", MySQL_BIGINT(), autoincrement=True, nullable=False),
        sa.Column("contract_no", sa.String(32), nullable=False),
        sa.Column("customer_id", MySQL_BIGINT(), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("amount_cents", MySQL_BIGINT(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "draft",
                "pending",
                "approved",
                "rejected",
                "cancelled",
                name="contract_status",
            ),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_by", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(3), server_default=sa.func.now(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(3),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.id"], name="fk_contracts_customer"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("contract_no"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_contracts_customer", "contracts", ["customer_id"])
    op.create_index("idx_contracts_status_created", "contracts", ["status", "created_at"])

    op.create_table(
        "approval_nodes",
        sa.Column("id", MySQL_BIGINT(), autoincrement=True, nullable=False),
        sa.Column("contract_id", MySQL_BIGINT(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column(
            "status",
            sa.Enum("waiting", "approved", "rejected", "skipped", name="node_status"),
            nullable=False,
        ),
        sa.Column("approver", sa.String(64), nullable=True),
        sa.Column("decided_at", sa.DateTime(3), nullable=True),
        sa.ForeignKeyConstraint(["contract_id"], ["contracts.id"], name="fk_nodes_contract"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("contract_id", "seq", name="uk_nodes_contract_seq"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    op.create_table(
        "approval_logs",
        sa.Column("id", MySQL_BIGINT(), autoincrement=True, nullable=False),
        sa.Column("contract_id", MySQL_BIGINT(), nullable=False),
        sa.Column("from_status", sa.String(16), nullable=False),
        sa.Column("to_status", sa.String(16), nullable=False),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("actor", sa.String(64), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(3), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["contract_id"], ["contracts.id"], name="fk_logs_contract"),
        sa.PrimaryKeyConstraint("id"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_logs_contract", "approval_logs", ["contract_id", "created_at"])

    op.create_table(
        "idempotency_keys",
        sa.Column("key_hash", sa.CHAR(64), nullable=False),
        sa.Column("scope", sa.String(32), nullable=False),
        sa.Column("response_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(3), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("key_hash"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_idem_created", "idempotency_keys", ["created_at"])


def downgrade() -> None:
    # 倒序删：子表在前，父表在后（外键约束）
    op.drop_index("idx_idem_created", table_name="idempotency_keys")
    op.drop_table("idempotency_keys")
    op.drop_index("idx_logs_contract", table_name="approval_logs")
    op.drop_table("approval_logs")
    op.drop_table("approval_nodes")
    op.drop_index("idx_contracts_status_created", table_name="contracts")
    op.drop_index("idx_contracts_customer", table_name="contracts")
    op.drop_table("contracts")
    op.drop_table("customers")
