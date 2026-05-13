"""Add eod_open_interest table for daily OI snapshots.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-04 00:00:00
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "eod_open_interest",
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("expiration", sa.Date(), nullable=False),
        sa.Column("strike", sa.Numeric(20, 6), nullable=False),
        sa.Column("option_type", sa.CHAR(1), nullable=False),
        sa.Column("oi_date", sa.Date(), nullable=False),
        sa.Column(
            "open_interest",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint(
            "symbol", "expiration", "strike", "option_type"
        ),
    )
    op.create_index(
        "ix_eod_oi_symbol_expiry", "eod_open_interest", ["symbol", "expiration"]
    )


def downgrade() -> None:
    op.drop_index("ix_eod_oi_symbol_expiry", table_name="eod_open_interest")
    op.drop_table("eod_open_interest")
