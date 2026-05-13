"""Initial schema with TimescaleDB hypertables.

Revision ID: 0001
Revises:
Create Date: 2025-01-01 00:00:00
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Extension (idempotent)
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")

    # ── options_chain ────────────────────────────────────────────────────────
    op.create_table(
        "options_chain",
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("expiration", sa.Date(), nullable=False),
        sa.Column("strike", sa.Numeric(20, 6), nullable=False),
        sa.Column("option_type", sa.CHAR(1), nullable=False),
        sa.Column("oi", sa.BigInteger(), nullable=True),
        sa.Column("volume", sa.BigInteger(), nullable=True),
        sa.Column("iv", sa.Numeric(20, 8), nullable=True),
        sa.Column("delta", sa.Numeric(20, 8), nullable=True),
        sa.Column("gamma", sa.Numeric(20, 8), nullable=True),
        sa.Column("last_price", sa.Numeric(20, 6), nullable=True),
        sa.Column("bid", sa.Numeric(20, 6), nullable=True),
        sa.Column("ask", sa.Numeric(20, 6), nullable=True),
        sa.Column("underlying_price", sa.Numeric(20, 6), nullable=True),
        sa.PrimaryKeyConstraint("ts", "symbol", "expiration", "strike", "option_type"),
    )
    op.create_index("ix_options_chain_symbol_ts", "options_chain", ["symbol", "ts"])
    op.create_index(
        "ix_options_chain_symbol_expiry", "options_chain", ["symbol", "expiration"]
    )

    op.execute(
        "SELECT create_hypertable('options_chain', 'ts', if_not_exists => TRUE, "
        "migrate_data => TRUE);"
    )
    op.execute(
        "SELECT add_retention_policy('options_chain', INTERVAL '7 days', "
        "if_not_exists => TRUE);"
    )

    # ── computed_metrics ─────────────────────────────────────────────────────
    op.create_table(
        "computed_metrics",
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("metric_type", sa.Text(), nullable=False),
        sa.Column(
            "strike",
            sa.Numeric(20, 6),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "expiration",
            sa.Date(),
            nullable=False,
            server_default=sa.text("'1970-01-01'::date"),
        ),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("value", sa.Numeric(30, 8), nullable=True),
        sa.Column("extra_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("ts", "symbol", "metric_type", "strike", "expiration"),
    )
    op.create_index(
        "ix_computed_metrics_symbol_type_ts",
        "computed_metrics",
        ["symbol", "metric_type", "ts"],
    )
    op.execute(
        "SELECT create_hypertable('computed_metrics', 'ts', if_not_exists => TRUE, "
        "migrate_data => TRUE);"
    )
    op.execute(
        "SELECT add_retention_policy('computed_metrics', INTERVAL '7 days', "
        "if_not_exists => TRUE);"
    )

    # ── api_keys ─────────────────────────────────────────────────────────────
    op.create_table(
        "api_keys",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("key_hash", sa.Text(), nullable=False),
        sa.Column("key_prefix", sa.String(32), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column(
            "allowed_symbols",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("TRUE"),
        ),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "usage_count",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key_hash", name="uq_api_keys_key_hash"),
    )
    op.create_index("ix_api_keys_key_prefix", "api_keys", ["key_prefix"])

    # ── admin_users ──────────────────────────────────────────────────────────
    op.create_table(
        "admin_users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("username", sa.Text(), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
    )

    # Compress data older than 1 day on options_chain (best-effort).
    op.execute(
        """
        ALTER TABLE options_chain SET (
          timescaledb.compress,
          timescaledb.compress_segmentby = 'symbol, option_type'
        );
        """
    )
    op.execute(
        "SELECT add_compression_policy('options_chain', INTERVAL '1 day', "
        "if_not_exists => TRUE);"
    )


def downgrade() -> None:
    op.drop_table("admin_users")
    op.drop_index("ix_api_keys_key_prefix", table_name="api_keys")
    op.drop_table("api_keys")
    op.drop_index("ix_computed_metrics_symbol_type_ts", table_name="computed_metrics")
    op.drop_table("computed_metrics")
    op.drop_index("ix_options_chain_symbol_expiry", table_name="options_chain")
    op.drop_index("ix_options_chain_symbol_ts", table_name="options_chain")
    op.drop_table("options_chain")
