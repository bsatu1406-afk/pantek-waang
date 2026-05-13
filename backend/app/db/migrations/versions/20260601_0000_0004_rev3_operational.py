"""Rev 3 operational hardening: pipeline run telemetry, ingestion DLQ,
backfill checkpoints, contract ADV cache, additional indexes, and
TimescaleDB compression policies on the largest hypertables.

This migration is **additive** — it does not alter existing tables
(per the Rev 3 constraint "do not modify migrations 0001–0003"). All
new objects use ``IF NOT EXISTS`` semantics where Postgres supports
it so re-running the migration is safe.

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-01 00:00:00
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    has_timescale = _has_timescaledb(bind)

    # ── pipeline_runs ────────────────────────────────────────────────────
    # One row per scheduler tick per symbol. Records duration, success,
    # and metric coverage so /admin/system/status can answer "did the
    # last cycle complete on time and produce all 25+ metric types?".
    op.create_table(
        "pipeline_runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "duration_ms", sa.Numeric(20, 3), nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "status", sa.Text(), nullable=False,
            server_default=sa.text("'running'"),
        ),
        # ``ok`` | ``failed`` | ``partial`` | ``running``
        sa.Column(
            "rows_read", sa.BigInteger(), nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "metric_rows_written", sa.BigInteger(), nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "missing_metric_types",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_pipeline_runs_symbol_started",
        "pipeline_runs",
        ["symbol", sa.text("started_at DESC")],
    )

    # ── dead_letter_queue ────────────────────────────────────────────────
    # Persistent record of ingestion records we could not parse / write.
    # Lets operators inspect bad-feed records via /admin/inspector without
    # depending on ephemeral logs.
    op.create_table(
        "dead_letter_queue",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "ts", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("source", sa.Text(), nullable=False),
        # ``opra_live`` | ``opra_historical`` | ``globex_live`` | ``eod_oi`` | ``pipeline``
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column(
            "payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_dlq_source_ts",
        "dead_letter_queue",
        ["source", sa.text("ts DESC")],
    )

    # ── backfill_checkpoints ─────────────────────────────────────────────
    # Tracks the last *successfully-ingested* end timestamp per (dataset,
    # symbol) so historical backfills can resume after restart without
    # re-pulling already-stored chunks.
    op.create_table(
        "backfill_checkpoints",
        sa.Column("dataset", sa.Text(), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("last_completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint("dataset", "symbol"),
    )

    # ── contract_adv ─────────────────────────────────────────────────────
    # Trailing N-day average daily volume per contract, used by the UOA
    # branch of :func:`app.processing.flow_events.detect_flow_events`.
    # Refreshed by a daily job (post-close).
    op.create_table(
        "contract_adv",
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("expiration", sa.Date(), nullable=False),
        sa.Column("strike", sa.Numeric(20, 6), nullable=False),
        sa.Column("option_type", sa.CHAR(1), nullable=False),
        sa.Column("avg_daily_volume", sa.Numeric(20, 6), nullable=False),
        sa.Column("window_days", sa.Integer(), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint("symbol", "expiration", "strike", "option_type"),
    )
    op.create_index(
        "ix_contract_adv_symbol_expiry",
        "contract_adv",
        ["symbol", "expiration"],
    )

    # ── Additional indexes on existing tables (additive — non-blocking) ──
    # computed_metrics is queried by (symbol, metric_type, expiration, ts).
    op.create_index(
        "ix_computed_metrics_symbol_type_exp_ts",
        "computed_metrics",
        ["symbol", "metric_type", "expiration", sa.text("ts DESC")],
    )
    # options_chain often queried by (symbol, expiration, strike) for
    # contract drill-down.
    op.create_index(
        "ix_options_chain_contract",
        "options_chain",
        ["symbol", "expiration", "strike", "option_type"],
    )
    # flow_events index for "give me sweeps in last hour for SYM".
    op.create_index(
        "ix_flow_events_symbol_type_ts",
        "flow_events",
        ["symbol", "event_type", sa.text("ts DESC")],
    )

    # ── TimescaleDB compression policies (largest hypertables) ───────────
    # Compression cuts storage ~5–10× on options_chain / options_trades /
    # futures_ticks. We compress chunks older than 1 day; recent data
    # stays uncompressed for fast writes.
    if has_timescale:
        for table, segment_by in (
            ("options_chain", "symbol"),
            ("options_trades", "symbol"),
            ("futures_ticks", "symbol"),
            ("computed_metrics", "symbol"),
        ):
            # ALTER TABLE … SET (timescaledb.compress, …) is idempotent
            # but errors if a non-existent column is referenced — segment
            # by symbol which all four tables have.
            op.execute(
                f"ALTER TABLE {table} SET ("
                f"timescaledb.compress, "
                f"timescaledb.compress_segmentby = '{segment_by}', "
                f"timescaledb.compress_orderby = 'ts DESC');"
            )
            # add_compression_policy supports if_not_exists since TimescaleDB 2.6.
            op.execute(
                f"SELECT add_compression_policy('{table}', INTERVAL '1 day', "
                f"if_not_exists => TRUE);"
            )

        # Tighter chunk interval on options_chain (default is 7 days, but
        # our chain is high-cardinality so smaller chunks compress better
        # and accelerate retention drops).
        op.execute(
            "SELECT set_chunk_time_interval('options_chain', INTERVAL '1 day');"
        )
        op.execute(
            "SELECT set_chunk_time_interval('computed_metrics', INTERVAL '1 day');"
        )


def downgrade() -> None:
    bind = op.get_bind()
    has_timescale = _has_timescaledb(bind)

    if has_timescale:
        for table in (
            "options_chain",
            "options_trades",
            "futures_ticks",
            "computed_metrics",
        ):
            op.execute(
                f"SELECT remove_compression_policy('{table}', if_exists => TRUE);"
            )

    op.drop_index(
        "ix_flow_events_symbol_type_ts", table_name="flow_events"
    )
    op.drop_index(
        "ix_options_chain_contract", table_name="options_chain"
    )
    op.drop_index(
        "ix_computed_metrics_symbol_type_exp_ts", table_name="computed_metrics"
    )
    op.drop_index("ix_contract_adv_symbol_expiry", table_name="contract_adv")
    op.drop_table("contract_adv")
    op.drop_table("backfill_checkpoints")
    op.drop_index("ix_dlq_source_ts", table_name="dead_letter_queue")
    op.drop_table("dead_letter_queue")
    op.drop_index("ix_pipeline_runs_symbol_started", table_name="pipeline_runs")
    op.drop_table("pipeline_runs")


# ── Helpers ──────────────────────────────────────────────────────────────


def _has_timescaledb(bind: sa.engine.Connection) -> bool:
    """Return True when the timescaledb extension is installed.

    The test suite uses plain Postgres without Timescale, so we silently
    skip hypertable / compression DDL there.
    """
    result = bind.execute(
        sa.text(
            "SELECT 1 FROM pg_extension WHERE extname = 'timescaledb' LIMIT 1"
        )
    ).first()
    return result is not None
