"""Rev 4 — bbo-1s persisted cache + Lee-Ready tick-rule fallback wiring.

Adds a dedicated ``bbo_book`` hypertable that captures BBO snapshots
keyed by symbol + instrument_id + ts. Used by:

* The live Lee-Ready classifier (production path) — current best bid/ask
  is cached in-memory from the cmbp-1 / bbo-1s feed and now also
  persisted so a restart does not lose context.
* The pipeline backfill / replay path — :func:`app.processing.bbo_cache
  .enrich_with_bbo` performs a point-in-time as-of merge between
  persisted trades and BBO snapshots, allowing Lee-Ready to be re-run
  over any historical window with the original mid recovered.

This migration is strictly additive — no destructive ALTERs.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-01 00:00:00
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "bbo_book",
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("instrument_id", sa.BigInteger(), nullable=False),
        sa.Column("expiration", sa.Date(), nullable=True),
        sa.Column("strike", sa.Numeric(20, 6), nullable=True),
        sa.Column("option_type", sa.CHAR(1), nullable=True),
        sa.Column("bid_px", sa.Numeric(20, 6), nullable=True),
        sa.Column("bid_sz", sa.BigInteger(), nullable=True),
        sa.Column("ask_px", sa.Numeric(20, 6), nullable=True),
        sa.Column("ask_sz", sa.BigInteger(), nullable=True),
        sa.Column("source", sa.Text(), nullable=True),  # "cmbp-1" | "bbo-1s"
        sa.PrimaryKeyConstraint("ts", "symbol", "instrument_id"),
    )
    op.create_index(
        "ix_bbo_book_symbol_ts",
        "bbo_book",
        ["symbol", "ts"],
    )
    op.create_index(
        "ix_bbo_book_contract_ts",
        "bbo_book",
        ["symbol", "expiration", "strike", "option_type", "ts"],
    )

    op.execute(
        "SELECT create_hypertable('bbo_book', 'ts', if_not_exists => TRUE, "
        "migrate_data => TRUE);"
    )
    # bbo-1s for both indices yields ~50 messages/s sustained; 7-day
    # retention keeps the table compact while leaving enough history for
    # ad-hoc Lee-Ready re-runs.
    op.execute(
        "SELECT add_retention_policy('bbo_book', INTERVAL '7 days', "
        "if_not_exists => TRUE);"
    )
    op.execute(
        "ALTER TABLE bbo_book SET ("
        "  timescaledb.compress, "
        "  timescaledb.compress_segmentby = 'symbol, instrument_id'"
        ");"
    )
    op.execute(
        "SELECT add_compression_policy('bbo_book', INTERVAL '1 day', "
        "if_not_exists => TRUE);"
    )


def downgrade() -> None:
    op.execute("SELECT remove_compression_policy('bbo_book', if_exists => TRUE);")
    op.execute("SELECT remove_retention_policy('bbo_book', if_exists => TRUE);")
    op.drop_index("ix_bbo_book_contract_ts", table_name="bbo_book")
    op.drop_index("ix_bbo_book_symbol_ts", table_name="bbo_book")
    op.drop_table("bbo_book")
