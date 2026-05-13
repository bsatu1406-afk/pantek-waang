"""SQLAlchemy ORM models for the options analytics platform."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    ARRAY,
    CHAR,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class OptionsChain(Base):
    """Time-series options chain snapshots from the OPRA feed.

    Promoted to a TimescaleDB hypertable in the initial migration.
    """

    __tablename__ = "options_chain"

    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )
    symbol: Mapped[str] = mapped_column(Text, primary_key=True, nullable=False)
    expiration: Mapped[datetime] = mapped_column(Date, primary_key=True, nullable=False)
    strike: Mapped[float] = mapped_column(Numeric(20, 6), primary_key=True, nullable=False)
    option_type: Mapped[str] = mapped_column(CHAR(1), primary_key=True, nullable=False)

    oi: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    iv: Mapped[float | None] = mapped_column(Numeric(20, 8), nullable=True)
    delta: Mapped[float | None] = mapped_column(Numeric(20, 8), nullable=True)
    gamma: Mapped[float | None] = mapped_column(Numeric(20, 8), nullable=True)
    last_price: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    bid: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    ask: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    underlying_price: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)

    __table_args__ = (
        Index("ix_options_chain_symbol_ts", "symbol", "ts"),
        Index("ix_options_chain_symbol_expiry", "symbol", "expiration"),
    )


class ComputedMetric(Base):
    """Time-series storage for computed metrics (GEX, max pain, walls, IV, etc.).

    Promoted to a TimescaleDB hypertable in the initial migration.
    """

    __tablename__ = "computed_metrics"

    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )
    symbol: Mapped[str] = mapped_column(Text, primary_key=True, nullable=False)
    metric_type: Mapped[str] = mapped_column(Text, primary_key=True, nullable=False)
    strike: Mapped[float | None] = mapped_column(
        Numeric(20, 6), primary_key=True, nullable=False, default=0
    )
    expiration: Mapped[datetime | None] = mapped_column(
        Date, primary_key=True, nullable=False
    )

    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    value: Mapped[float | None] = mapped_column(Numeric(30, 8), nullable=True)
    extra_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_computed_metrics_symbol_type_ts", "symbol", "metric_type", "ts"),
    )


class EodOpenInterest(Base):
    """End-of-day Open Interest snapshots.

    OPRA Pillar's ``definition`` schema doesn't include open interest, so
    intraday OI is frequently missing on a fresh deployment. This table is
    populated by a daily ingestion job that pulls the most recent OI per
    contract (best-effort — falls back to "no rows" if the data source isn't
    available). The compute pipeline merges these snapshots back into the
    options chain whenever live OI is null/zero.
    """

    __tablename__ = "eod_open_interest"

    symbol: Mapped[str] = mapped_column(Text, primary_key=True, nullable=False)
    expiration: Mapped[datetime] = mapped_column(Date, primary_key=True, nullable=False)
    strike: Mapped[float] = mapped_column(Numeric(20, 6), primary_key=True, nullable=False)
    option_type: Mapped[str] = mapped_column(CHAR(1), primary_key=True, nullable=False)

    oi_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    open_interest: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_eod_oi_symbol_expiry", "symbol", "expiration"),
    )


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    key_hash: Mapped[str] = mapped_column(Text, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(32), nullable=False)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    allowed_symbols: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    usage_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    __table_args__ = (
        UniqueConstraint("key_hash", name="uq_api_keys_key_hash"),
        Index("ix_api_keys_key_prefix", "key_prefix"),
    )


class AdminUser(Base):
    __tablename__ = "admin_users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    username: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )


# ── Phase 2: futures + flow + alerts ────────────────────────────────────────


class FuturesTick(Base):
    """Globex MDP 3.0 futures trade tape.

    One row per trade event. Promoted to a TimescaleDB hypertable in
    migration 0003. Volume can balloon quickly (~5M ES trades / day); a
    short retention window is applied.
    """

    __tablename__ = "futures_ticks"

    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )
    symbol: Mapped[str] = mapped_column(Text, primary_key=True, nullable=False)
    seq: Mapped[int] = mapped_column(BigInteger, primary_key=True, nullable=False)

    price: Mapped[float] = mapped_column(Numeric(20, 6), nullable=False)
    size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    aggressor: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    """+1 = buyer-aggressor, -1 = seller-aggressor, NULL = unknown."""
    bid: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    ask: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    venue: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_futures_ticks_symbol_ts", "symbol", "ts"),
    )


class OptionsTrade(Base):
    """OPRA trade tape, classified via Lee-Ready downstream.

    One row per trade message. Promoted to a hypertable.
    """

    __tablename__ = "options_trades"

    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )
    symbol: Mapped[str] = mapped_column(Text, primary_key=True, nullable=False)
    expiration: Mapped[datetime] = mapped_column(Date, primary_key=True, nullable=False)
    strike: Mapped[float] = mapped_column(Numeric(20, 6), primary_key=True, nullable=False)
    option_type: Mapped[str] = mapped_column(CHAR(1), primary_key=True, nullable=False)
    seq: Mapped[int] = mapped_column(BigInteger, primary_key=True, nullable=False)

    price: Mapped[float] = mapped_column(Numeric(20, 6), nullable=False)
    size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    bid: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    ask: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    exchange: Mapped[str | None] = mapped_column(Text, nullable=True)
    side: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    """Lee-Ready customer side: +1 (buy), -1 (sell), 0 (unclassified)."""
    signed_premium: Mapped[float | None] = mapped_column(Numeric(30, 6), nullable=True)

    __table_args__ = (
        Index("ix_options_trades_symbol_ts", "symbol", "ts"),
        Index("ix_options_trades_contract", "symbol", "expiration", "strike", "option_type"),
    )


class FlowEvent(Base):
    """Detected sweeps / blocks / UOA. Persisted for the website + alerts."""

    __tablename__ = "flow_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    symbol: Mapped[str] = mapped_column(Text, nullable=False)
    expiration: Mapped[datetime] = mapped_column(Date, nullable=False)
    strike: Mapped[float] = mapped_column(Numeric(20, 6), nullable=False)
    option_type: Mapped[str] = mapped_column(CHAR(1), nullable=False)

    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    """``SWEEP`` | ``BLOCK`` | ``UOA``."""
    side: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    size: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    price: Mapped[float | None] = mapped_column(Numeric(20, 6), nullable=True)
    legs: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    venues: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list
    )
    meta: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_flow_events_symbol_ts", "symbol", "ts"),
        Index("ix_flow_events_type_ts", "event_type", "ts"),
    )


class LiquiditySnapshot(Base):
    """Globex MDP 3.0 MBO order-book depth snapshot.

    Stored as compact JSONB rather than per-level rows: order books
    update tens of thousands of times per second on ES, and we only need
    point-in-time snapshots for analytics. Snapshot frequency is
    configured by the ingester (default 1 / second).
    """

    __tablename__ = "liquidity_snapshots"

    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )
    symbol: Mapped[str] = mapped_column(Text, primary_key=True, nullable=False)

    bids: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)
    """``[{price, size, orders}, ...]`` highest bid first."""
    asks: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)
    """``[{price, size, orders}, ...]`` lowest ask first."""
    depth_levels: Mapped[int] = mapped_column(Integer, nullable=False, default=10)

    __table_args__ = (
        Index("ix_liquidity_snapshots_symbol_ts", "symbol", "ts"),
    )


class AlertRule(Base):
    """User-defined alert rule expressed as a JSON predicate tree."""

    __tablename__ = "alert_rules"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    symbol: Mapped[str] = mapped_column(Text, nullable=False)
    rule: Mapped[dict] = mapped_column(JSONB, nullable=False)
    severity: Mapped[str] = mapped_column(Text, nullable=False, default="info")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    cooldown_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=300)
    last_fired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint("name", "symbol", name="uq_alert_rules_name_symbol"),
    )


class AlertEvent(Base):
    """An alert firing produced by an AlertRule on a specific snapshot."""

    __tablename__ = "alert_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    rule_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("alert_rules.id", ondelete="CASCADE"),
        nullable=False,
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    symbol: Mapped[str] = mapped_column(Text, nullable=False)
    matched: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list
    )
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_alert_events_symbol_ts", "symbol", "ts"),
        Index("ix_alert_events_rule_ts", "rule_id", "ts"),
    )


# ── Rev 3: operational telemetry / ingestion safety net ─────────────────────


class PipelineRun(Base):
    """One row per scheduler tick per symbol — runtime audit log.

    Used by :func:`app.api.endpoints.admin.system_status` to answer
    "did the last cycle complete cleanly and produce all 25+ metric
    types?" without re-deriving from raw ``computed_metrics``.
    """

    __tablename__ = "pipeline_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    symbol: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    duration_ms: Mapped[float] = mapped_column(
        Numeric(20, 3), nullable=False, default=0
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, default="running")
    """``running`` | ``ok`` | ``partial`` | ``failed``."""
    rows_read: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    metric_rows_written: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0
    )
    missing_metric_types: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class DeadLetterEntry(Base):
    """Records ingestion payloads we could not parse / write.

    Surfaced via /admin/inspector so operators can diagnose feed issues
    without trawling logs.
    """

    __tablename__ = "dead_letter_queue"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    source: Mapped[str] = mapped_column(Text, nullable=False)
    """``opra_live`` | ``opra_historical`` | ``globex_live`` | ``eod_oi`` | ``pipeline``."""
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class BackfillCheckpoint(Base):
    """Per (dataset, symbol) bookmark for resumable historical backfills."""

    __tablename__ = "backfill_checkpoints"

    dataset: Mapped[str] = mapped_column(Text, primary_key=True, nullable=False)
    symbol: Mapped[str] = mapped_column(Text, primary_key=True, nullable=False)
    last_completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )


class ContractAdv(Base):
    """Trailing-N-day average daily volume per contract.

    Consumed by :func:`app.processing.flow_events.detect_flow_events` for
    the UOA branch. Refreshed by a daily post-close job.
    """

    __tablename__ = "contract_adv"

    symbol: Mapped[str] = mapped_column(Text, primary_key=True, nullable=False)
    expiration: Mapped[datetime] = mapped_column(
        Date, primary_key=True, nullable=False
    )
    strike: Mapped[float] = mapped_column(
        Numeric(20, 6), primary_key=True, nullable=False
    )
    option_type: Mapped[str] = mapped_column(
        CHAR(1), primary_key=True, nullable=False
    )

    avg_daily_volume: Mapped[float] = mapped_column(
        Numeric(20, 6), nullable=False
    )
    window_days: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_contract_adv_symbol_expiry", "symbol", "expiration"),
    )
