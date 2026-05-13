"""Phase-2 storage: futures ticks, options trades, flow events,
liquidity snapshots, alert rules + events.

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-04 01:00:00
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── futures_ticks ────────────────────────────────────────────────────
    op.create_table(
        "futures_ticks",
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("seq", sa.BigInteger(), nullable=False),
        sa.Column("price", sa.Numeric(20, 6), nullable=False),
        sa.Column("size", sa.BigInteger(), nullable=False),
        sa.Column("aggressor", sa.SmallInteger(), nullable=True),
        sa.Column("bid", sa.Numeric(20, 6), nullable=True),
        sa.Column("ask", sa.Numeric(20, 6), nullable=True),
        sa.Column("venue", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("ts", "symbol", "seq"),
    )
    op.create_index("ix_futures_ticks_symbol_ts", "futures_ticks", ["symbol", "ts"])
    op.execute(
        "SELECT create_hypertable('futures_ticks', 'ts', if_not_exists => TRUE, "
        "migrate_data => TRUE);"
    )
    op.execute(
        "SELECT add_retention_policy('futures_ticks', INTERVAL '14 days', "
        "if_not_exists => TRUE);"
    )

    # ── options_trades ───────────────────────────────────────────────────
    op.create_table(
        "options_trades",
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("expiration", sa.Date(), nullable=False),
        sa.Column("strike", sa.Numeric(20, 6), nullable=False),
        sa.Column("option_type", sa.CHAR(1), nullable=False),
        sa.Column("seq", sa.BigInteger(), nullable=False),
        sa.Column("price", sa.Numeric(20, 6), nullable=False),
        sa.Column("size", sa.BigInteger(), nullable=False),
        sa.Column("bid", sa.Numeric(20, 6), nullable=True),
        sa.Column("ask", sa.Numeric(20, 6), nullable=True),
        sa.Column("exchange", sa.Text(), nullable=True),
        sa.Column("side", sa.SmallInteger(), nullable=True),
        sa.Column("signed_premium", sa.Numeric(30, 6), nullable=True),
        sa.PrimaryKeyConstraint(
            "ts", "symbol", "expiration", "strike", "option_type", "seq"
        ),
    )
    op.create_index("ix_options_trades_symbol_ts", "options_trades", ["symbol", "ts"])
    op.create_index(
        "ix_options_trades_contract", "options_trades",
        ["symbol", "expiration", "strike", "option_type"],
    )
    op.execute(
        "SELECT create_hypertable('options_trades', 'ts', if_not_exists => TRUE, "
        "migrate_data => TRUE);"
    )
    op.execute(
        "SELECT add_retention_policy('options_trades', INTERVAL '14 days', "
        "if_not_exists => TRUE);"
    )

    # ── flow_events (regular table — low volume, not time-series-ish) ────
    op.create_table(
        "flow_events",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("expiration", sa.Date(), nullable=False),
        sa.Column("strike", sa.Numeric(20, 6), nullable=False),
        sa.Column("option_type", sa.CHAR(1), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column(
            "side", sa.SmallInteger(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "size", sa.BigInteger(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("price", sa.Numeric(20, 6), nullable=True),
        sa.Column(
            "legs", sa.Integer(), nullable=False, server_default=sa.text("1")
        ),
        sa.Column(
            "venues",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_flow_events_symbol_ts", "flow_events", ["symbol", "ts"])
    op.create_index("ix_flow_events_type_ts", "flow_events", ["event_type", "ts"])

    # ── liquidity_snapshots (hypertable, JSONB-heavy) ────────────────────
    op.create_table(
        "liquidity_snapshots",
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("bids", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("asks", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "depth_levels",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("10"),
        ),
        sa.PrimaryKeyConstraint("ts", "symbol"),
    )
    op.create_index(
        "ix_liquidity_snapshots_symbol_ts", "liquidity_snapshots", ["symbol", "ts"]
    )
    op.execute(
        "SELECT create_hypertable('liquidity_snapshots', 'ts', if_not_exists => TRUE, "
        "migrate_data => TRUE);"
    )
    op.execute(
        "SELECT add_retention_policy('liquidity_snapshots', INTERVAL '7 days', "
        "if_not_exists => TRUE);"
    )

    # ── alert_rules ──────────────────────────────────────────────────────
    op.create_table(
        "alert_rules",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("rule", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "severity", sa.Text(), nullable=False, server_default=sa.text("'info'")
        ),
        sa.Column(
            "enabled", sa.Boolean(), nullable=False, server_default=sa.text("TRUE")
        ),
        sa.Column(
            "cooldown_seconds",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("300"),
        ),
        sa.Column("last_fired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "symbol", name="uq_alert_rules_name_symbol"),
    )

    # ── alert_events ─────────────────────────────────────────────────────
    op.create_table(
        "alert_events",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("rule_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "ts",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column(
            "matched",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["rule_id"], ["alert_rules.id"], ondelete="CASCADE",
            name="fk_alert_events_rule_id",
        ),
    )
    op.create_index("ix_alert_events_symbol_ts", "alert_events", ["symbol", "ts"])
    op.create_index("ix_alert_events_rule_ts", "alert_events", ["rule_id", "ts"])


def downgrade() -> None:
    op.drop_index("ix_alert_events_rule_ts", table_name="alert_events")
    op.drop_index("ix_alert_events_symbol_ts", table_name="alert_events")
    op.drop_table("alert_events")

    op.drop_table("alert_rules")

    op.drop_index("ix_liquidity_snapshots_symbol_ts", table_name="liquidity_snapshots")
    op.drop_table("liquidity_snapshots")

    op.drop_index("ix_flow_events_type_ts", table_name="flow_events")
    op.drop_index("ix_flow_events_symbol_ts", table_name="flow_events")
    op.drop_table("flow_events")

    op.drop_index("ix_options_trades_contract", table_name="options_trades")
    op.drop_index("ix_options_trades_symbol_ts", table_name="options_trades")
    op.drop_table("options_trades")

    op.drop_index("ix_futures_ticks_symbol_ts", table_name="futures_ticks")
    op.drop_table("futures_ticks")
