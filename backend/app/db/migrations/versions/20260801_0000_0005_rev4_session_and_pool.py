"""Rev 4 — session lifecycle, 0DTE/spot telemetry, Databento key pool.

This migration is **strictly additive** to the Rev 3 schema:

* ``session_events``       — lightweight audit trail of session open/close/reset
* ``metric_type_registry`` — human-readable catalogue of every ``metric_type``
* ``databento_api_keys``   — encrypted pool of fallback Databento credentials
* ``pipeline_runs`` gains four new columns describing the 0DTE / spot
  state at run time.
* A partial index on ``computed_metrics`` accelerates the 0DTE query path.

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-01 00:00:00
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# ──────────────────────────────────────────────────────────────────────────
# Metric-type catalogue seed. Kept inline so re-running the migration
# repopulates rows that humans may have deleted by hand. Categories follow
# the rough analytic family of each metric.
# ──────────────────────────────────────────────────────────────────────────


_METRIC_TYPE_SEED: list[dict[str, str | bool]] = [
    # ── GEX family (Rev 1 + Rev 2 + Rev 3 + Rev 4) ──────────────────────
    {"metric_type": "GEX_NET_TOTAL", "category": "gex", "is_0dte": False,
     "description": "Aggregate dealer gamma exposure (OI-weighted).", "added_in_rev": "rev1"},
    {"metric_type": "GEX_LEVEL", "category": "gex", "is_0dte": False,
     "description": "Per-strike GEX curve point (OI-weighted).", "added_in_rev": "rev1"},
    {"metric_type": "GEX_NET_TOTAL_VOL", "category": "gex", "is_0dte": False,
     "description": "Aggregate dealer gamma exposure (volume-weighted).", "added_in_rev": "rev2"},
    {"metric_type": "GEX_LEVEL_VOL", "category": "gex", "is_0dte": False,
     "description": "Per-strike GEX curve point (volume-weighted).", "added_in_rev": "rev2"},
    {"metric_type": "GEX_0DTE_NET_TOTAL", "category": "0dte", "is_0dte": True,
     "description": "Aggregate dealer gamma exposure restricted to 0DTE contracts.", "added_in_rev": "rev4"},
    {"metric_type": "GEX_0DTE_LEVEL", "category": "0dte", "is_0dte": True,
     "description": "Per-strike 0DTE GEX curve point (OI-weighted).", "added_in_rev": "rev4"},
    {"metric_type": "GEX_0DTE_NET_TOTAL_VOL", "category": "0dte", "is_0dte": True,
     "description": "Aggregate dealer gamma (0DTE, volume-weighted).", "added_in_rev": "rev4"},
    {"metric_type": "GEX_0DTE_LEVEL_VOL", "category": "0dte", "is_0dte": True,
     "description": "Per-strike 0DTE GEX (volume-weighted).", "added_in_rev": "rev4"},
    {"metric_type": "GEX_BACK_NET_TOTAL", "category": "gex", "is_0dte": False,
     "description": "Aggregate GEX excluding 0DTE (back-month only).", "added_in_rev": "rev4"},
    {"metric_type": "GEX_BACK_LEVEL", "category": "gex", "is_0dte": False,
     "description": "Per-strike back-month GEX curve point.", "added_in_rev": "rev4"},
    {"metric_type": "GEX_0DTE_FLIP_SPEED", "category": "0dte", "is_0dte": True,
     "description": "Points of spot movement required to flip 0DTE regime sign.", "added_in_rev": "rev4"},
    # ── Vanna / Charm ────────────────────────────────────────────────────
    {"metric_type": "VANNA_NET_TOTAL", "category": "vanna", "is_0dte": False,
     "description": "Aggregate dealer vanna exposure (∂Δ/∂σ).", "added_in_rev": "rev3"},
    {"metric_type": "VANNA_LEVEL", "category": "vanna", "is_0dte": False,
     "description": "Per-strike vanna curve point.", "added_in_rev": "rev3"},
    {"metric_type": "CHARM_NET_TOTAL", "category": "charm", "is_0dte": False,
     "description": "Aggregate dealer charm (∂Δ/∂t).", "added_in_rev": "rev3"},
    {"metric_type": "CHARM_LEVEL", "category": "charm", "is_0dte": False,
     "description": "Per-strike charm curve point.", "added_in_rev": "rev3"},
    {"metric_type": "CHARM_0DTE_DECAY_RATE", "category": "0dte", "is_0dte": True,
     "description": "Near-ATM 0DTE dealer delta change per hour from time decay.", "added_in_rev": "rev4"},
    # ── Max pain / Walls / IV / Move / Pin / Regime / Term / Risk reversal
    {"metric_type": "MAX_PAIN", "category": "max_pain", "is_0dte": False,
     "description": "Per-expiry max-pain strike.", "added_in_rev": "rev1"},
    {"metric_type": "MAX_PAIN_AGG", "category": "max_pain", "is_0dte": False,
     "description": "Multi-expiry aggregate max-pain strike.", "added_in_rev": "rev3"},
    {"metric_type": "CALL_WALL_OI", "category": "walls", "is_0dte": False,
     "description": "Top call wall by open interest.", "added_in_rev": "rev2"},
    {"metric_type": "PUT_WALL_OI", "category": "walls", "is_0dte": False,
     "description": "Top put wall by open interest.", "added_in_rev": "rev2"},
    {"metric_type": "CALL_WALL_VOL", "category": "walls", "is_0dte": False,
     "description": "Top call wall by volume.", "added_in_rev": "rev2"},
    {"metric_type": "PUT_WALL_VOL", "category": "walls", "is_0dte": False,
     "description": "Top put wall by volume.", "added_in_rev": "rev2"},
    {"metric_type": "ATM_IV", "category": "iv", "is_0dte": False,
     "description": "At-the-money implied volatility.", "added_in_rev": "rev1"},
    {"metric_type": "IV_SKEW", "category": "iv", "is_0dte": False,
     "description": "25Δ put IV − 25Δ call IV.", "added_in_rev": "rev1"},
    {"metric_type": "IV_SURFACE", "category": "iv", "is_0dte": False,
     "description": "Per-strike IV surface point.", "added_in_rev": "rev1"},
    {"metric_type": "IV_TERM_STRUCTURE", "category": "iv", "is_0dte": False,
     "description": "Per-expiry ATM IV slice.", "added_in_rev": "rev3"},
    {"metric_type": "RISK_REVERSAL_25D", "category": "iv", "is_0dte": False,
     "description": "25Δ risk reversal premium.", "added_in_rev": "rev3"},
    {"metric_type": "MOVE_TRACKER", "category": "move", "is_0dte": False,
     "description": "Spot move tracking summary (intraday).", "added_in_rev": "rev3"},
    {"metric_type": "PIN_PROBABILITY", "category": "pin", "is_0dte": False,
     "description": "Per-strike pin probability at expiry.", "added_in_rev": "rev3"},
    {"metric_type": "REGIME_OI", "category": "regime", "is_0dte": False,
     "description": "Bullish/neutral/bearish regime (OI-weighted GEX).", "added_in_rev": "rev2"},
    {"metric_type": "REGIME_VOL", "category": "regime", "is_0dte": False,
     "description": "Bullish/neutral/bearish regime (volume-weighted GEX).", "added_in_rev": "rev2"},
    # ── Flow / basis ─────────────────────────────────────────────────────
    {"metric_type": "HIRO", "category": "hiro", "is_0dte": False,
     "description": "Hedging-impact reaction-oriented signed-premium tape.", "added_in_rev": "rev2"},
    {"metric_type": "BASIS_SPX_ES", "category": "basis", "is_0dte": False,
     "description": "SPX cash − ES front-month futures basis.", "added_in_rev": "rev3"},
    {"metric_type": "BASIS_NDX_NQ", "category": "basis", "is_0dte": False,
     "description": "NDX cash − NQ front-month futures basis.", "added_in_rev": "rev4"},
    {"metric_type": "VOLUME_PROFILE_ES", "category": "volume_profile", "is_0dte": False,
     "description": "ES futures intraday volume profile.", "added_in_rev": "rev3"},
    # ── Session sentinels ────────────────────────────────────────────────
    {"metric_type": "HIRO_EOD", "category": "session", "is_0dte": False,
     "description": "End-of-day HIRO summary persisted at session close.", "added_in_rev": "rev4"},
]


def upgrade() -> None:
    # ── session_events ───────────────────────────────────────────────────
    op.create_table(
        "session_events",
        sa.Column(
            "id", sa.BigInteger(), primary_key=True, autoincrement=True
        ),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=True),
        sa.Column(
            "ts",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("extra_json", postgresql.JSONB(), nullable=True),
    )
    op.create_index(
        "ix_session_events_ts",
        "session_events",
        [sa.text("ts DESC")],
    )

    # ── metric_type_registry ─────────────────────────────────────────────
    op.create_table(
        "metric_type_registry",
        sa.Column("metric_type", sa.Text(), primary_key=True),
        sa.Column("category", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "is_0dte",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
        sa.Column("added_in_rev", sa.Text(), nullable=True),
    )
    op.bulk_insert(
        sa.table(
            "metric_type_registry",
            sa.column("metric_type", sa.Text()),
            sa.column("category", sa.Text()),
            sa.column("description", sa.Text()),
            sa.column("is_0dte", sa.Boolean()),
            sa.column("added_in_rev", sa.Text()),
        ),
        _METRIC_TYPE_SEED,
    )

    # ── databento_api_keys ────────────────────────────────────────────────
    # Encrypted Databento credentials. ``api_key_encrypted`` is a Fernet
    # token whose key is derived from JWT_SECRET — see
    # ``app.core.crypto.derive_fernet_from_jwt_secret``. We never store
    # plaintext keys; the masked prefix is held separately for the admin
    # UI list view.
    op.create_table(
        "databento_api_keys",
        sa.Column(
            "id", sa.BigInteger(), primary_key=True, autoincrement=True
        ),
        sa.Column("label", sa.Text(), nullable=False),
        # 'OPRA.PILLAR' | 'GLBX.MDP3' | 'BOTH'
        sa.Column("dataset", sa.Text(), nullable=False),
        sa.Column("api_key_encrypted", sa.Text(), nullable=False),
        sa.Column("api_key_prefix", sa.Text(), nullable=False),
        sa.Column(
            "priority", sa.Integer(), nullable=False, server_default=sa.text("100")
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("TRUE"),
        ),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error_msg", sa.Text(), nullable=True),
        sa.Column(
            "error_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.UniqueConstraint("label", name="uq_databento_keys_label"),
        sa.CheckConstraint(
            "dataset IN ('OPRA.PILLAR', 'GLBX.MDP3', 'BOTH')",
            name="ck_databento_keys_dataset",
        ),
    )
    op.create_index(
        "ix_databento_keys_dataset_priority",
        "databento_api_keys",
        ["dataset", "priority"],
    )

    # ── pipeline_runs: 0DTE + spot diagnostics columns ──────────────────
    op.add_column(
        "pipeline_runs",
        sa.Column(
            "is_expiration_day",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
    )
    op.add_column(
        "pipeline_runs",
        sa.Column("spot_source", sa.Text(), nullable=True),
    )
    op.add_column(
        "pipeline_runs",
        sa.Column("spot_price", sa.Numeric(20, 6), nullable=True),
    )
    op.add_column(
        "pipeline_runs",
        sa.Column("tau_0dte_years", sa.Numeric(20, 10), nullable=True),
    )

    # ── 0DTE partial index on computed_metrics ──────────────────────────
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_computed_metrics_0dte
          ON computed_metrics(symbol, ts DESC)
          WHERE metric_type LIKE 'GEX_0DTE%%'
             OR metric_type LIKE 'CHARM_0DTE%%';
        """
    )

    _ = datetime.now(UTC)  # noqa: F841  — keep import path live for downgrade


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_computed_metrics_0dte;")
    op.drop_column("pipeline_runs", "tau_0dte_years")
    op.drop_column("pipeline_runs", "spot_price")
    op.drop_column("pipeline_runs", "spot_source")
    op.drop_column("pipeline_runs", "is_expiration_day")
    op.drop_index(
        "ix_databento_keys_dataset_priority",
        table_name="databento_api_keys",
    )
    op.drop_table("databento_api_keys")
    op.drop_table("metric_type_registry")
    op.drop_index("ix_session_events_ts", table_name="session_events")
    op.drop_table("session_events")
