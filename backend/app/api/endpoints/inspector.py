"""Admin "Data Inspector" endpoint.

Exposes a single read-only payload that the admin dashboard polls so the
operator can verify, in one glance, that every Phase-2 ingestion + compute
pipeline is producing rows. Combines:

* Row counts + freshness (lag) per ingestion table.
* ``computed_metrics`` breakdown by ``metric_type`` (count + last_seen).
* The most recent payload for every key metric type (GEX-Vol top long/short,
  Vanna, Charm, IV term-structure, Pin probability, Move tracker, basis,
  HIRO, ES Volume Profile).
* The most recent flow events + alert events.

The payload is intentionally JSON-flexible (Mapping[str, Any]) so future
metrics can be surfaced without breaking the schema.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import authenticate_admin
from app.api.schemas import DlqEntry, DlqPage
from app.config import get_settings
from app.db.models import (
    AlertEvent,
    AlertRule,
    ComputedMetric,
    DeadLetterEntry,
    EodOpenInterest,
    FlowEvent,
    FuturesTick,
    LiquiditySnapshot,
    OptionsChain,
    OptionsTrade,
)
from app.db.session import get_db
from app.ingestion.databento_globex import get_globex_live_ingester
from app.ingestion.databento_live import get_live_ingester

router = APIRouter(prefix="/admin/inspector", tags=["admin"])

# Tables surfaced in the row-count card. Keep ``OptionsChain`` first so the
# UI can highlight it as the primary live feed.
_TABLES: list[tuple[str, Any, Any]] = [
    ("options_chain", OptionsChain, OptionsChain.ts),
    ("options_trades", OptionsTrade, OptionsTrade.ts),
    ("futures_ticks", FuturesTick, FuturesTick.ts),
    ("liquidity_snapshots", LiquiditySnapshot, LiquiditySnapshot.ts),
    ("computed_metrics", ComputedMetric, ComputedMetric.ts),
    ("flow_events", FlowEvent, FlowEvent.ts),
    ("alert_events", AlertEvent, AlertEvent.ts),
    ("eod_open_interest", EodOpenInterest, EodOpenInterest.updated_at),
]

# Metric types displayed individually. Order is the same as the screenshot
# panels in the UI.
_LATEST_METRIC_TYPES: list[str] = [
    "GEX_NET_TOTAL_VOL",
    "GEX_NET_TOTAL",
    "VANNA_NET_TOTAL",
    "CHARM_NET_TOTAL",
    "MAX_PAIN",
    "ATM_IV",
    "MOVE_TRACKER",
    "REGIME_VOL",
    "REGIME_OI",
    "HIRO",
    "BASIS_SPX_ES",
    "VOLUME_PROFILE_ES",
]


def _lag_seconds(ts: datetime | None) -> float | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - ts).total_seconds())


@router.get("")
async def data_inspector(
    _admin: Annotated[str, Depends(authenticate_admin)],
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    settings = get_settings()
    now = datetime.now(UTC)

    # ── 1. Row counts + freshness per table ───────────────────────────────
    tables: list[dict[str, Any]] = []
    for name, model, ts_col in _TABLES:
        cnt_row = await session.execute(
            select(func.count()).select_from(model)
        )
        latest_row = await session.execute(select(func.max(ts_col)))
        cnt = int(cnt_row.scalar_one() or 0)
        latest: datetime | None = latest_row.scalar_one_or_none()
        tables.append({
            "name": name,
            "rows": cnt,
            "latest_ts": latest.isoformat() if latest else None,
            "lag_seconds": _lag_seconds(latest),
        })

    # ── 2. metric_type breakdown ──────────────────────────────────────────
    breakdown_rows = (
        await session.execute(
            select(
                ComputedMetric.metric_type,
                func.count().label("rows"),
                func.max(ComputedMetric.ts).label("last_seen"),
                func.min(ComputedMetric.ts).label("first_seen"),
            )
            .group_by(ComputedMetric.metric_type)
            .order_by(ComputedMetric.metric_type)
        )
    ).all()
    metric_breakdown = [
        {
            "metric_type": row.metric_type,
            "rows": int(row.rows or 0),
            "first_seen": row.first_seen.isoformat() if row.first_seen else None,
            "last_seen": row.last_seen.isoformat() if row.last_seen else None,
            "lag_seconds": _lag_seconds(row.last_seen),
        }
        for row in breakdown_rows
    ]

    # ── 3. Latest snapshot per key metric_type ────────────────────────────
    latest_metrics: list[dict[str, Any]] = []
    for metric_type in _LATEST_METRIC_TYPES:
        # one row per (symbol, metric_type) — most recent ts.
        for symbol in settings.supported_symbols:
            row = (
                await session.execute(
                    select(ComputedMetric)
                    .where(
                        ComputedMetric.metric_type == metric_type,
                        ComputedMetric.symbol == symbol,
                    )
                    .order_by(desc(ComputedMetric.ts))
                    .limit(1)
                )
            ).scalar_one_or_none()
            if row is None:
                continue
            latest_metrics.append({
                "metric_type": metric_type,
                "symbol": row.symbol,
                "ts": row.ts.isoformat() if row.ts else None,
                "lag_seconds": _lag_seconds(row.ts),
                "value": float(row.value) if row.value is not None else None,
                "expiration": row.expiration.isoformat() if row.expiration else None,
                "strike": float(row.strike) if row.strike is not None else None,
                "extra": row.extra_json or {},
            })

    # ── 4. Latest IV term-structure (one row per expiration) ──────────────
    term_struct_latest_ts = (
        await session.execute(
            select(func.max(ComputedMetric.ts)).where(
                ComputedMetric.metric_type == "IV_TERM_STRUCTURE"
            )
        )
    ).scalar_one_or_none()

    term_structure: list[dict[str, Any]] = []
    if term_struct_latest_ts is not None:
        rows = (
            await session.execute(
                select(ComputedMetric)
                .where(
                    ComputedMetric.metric_type == "IV_TERM_STRUCTURE",
                    ComputedMetric.ts == term_struct_latest_ts,
                )
                .order_by(ComputedMetric.symbol, ComputedMetric.expiration)
            )
        ).scalars().all()
        for row in rows:
            extra = row.extra_json or {}
            term_structure.append({
                "symbol": row.symbol,
                "expiration": (
                    row.expiration.isoformat() if row.expiration else None
                ),
                "days_to_expiry": extra.get("days_to_expiry"),
                "atm_iv": float(row.value) if row.value is not None else None,
                "call_25d_iv": extra.get("call_25d_iv"),
                "put_25d_iv": extra.get("put_25d_iv"),
                "risk_reversal_25d": extra.get("risk_reversal_25d"),
            })

    # ── 5. Latest Pin Probability heatmap (one row per strike) ────────────
    pin_latest_ts = (
        await session.execute(
            select(func.max(ComputedMetric.ts)).where(
                ComputedMetric.metric_type == "PIN_PROBABILITY"
            )
        )
    ).scalar_one_or_none()

    pin_probability: list[dict[str, Any]] = []
    if pin_latest_ts is not None:
        rows = (
            await session.execute(
                select(ComputedMetric)
                .where(
                    ComputedMetric.metric_type == "PIN_PROBABILITY",
                    ComputedMetric.ts == pin_latest_ts,
                )
                .order_by(desc(ComputedMetric.value))
                .limit(15)
            )
        ).scalars().all()
        for row in rows:
            extra = row.extra_json or {}
            pin_probability.append({
                "symbol": row.symbol,
                "strike": float(row.strike) if row.strike is not None else None,
                "probability": float(row.value) if row.value is not None else None,
                "oi": extra.get("oi"),
                "abs_charm": extra.get("abs_charm"),
                "atm_iv": extra.get("atm_iv"),
            })

    # ── 6. Recent flow events ─────────────────────────────────────────────
    flow_rows = (
        await session.execute(
            select(FlowEvent).order_by(desc(FlowEvent.ts)).limit(50)
        )
    ).scalars().all()
    flow_events = [
        {
            "id": str(r.id),
            "ts": r.ts.isoformat() if r.ts else None,
            "symbol": r.symbol,
            "expiration": r.expiration.isoformat() if r.expiration else None,
            "strike": float(r.strike) if r.strike is not None else None,
            "option_type": r.option_type,
            "event_type": r.event_type,
            "side": int(r.side or 0),
            "size": int(r.size or 0),
            "price": float(r.price) if r.price is not None else None,
            "legs": int(r.legs or 1),
            "venues": list(r.venues or []),
        }
        for r in flow_rows
    ]

    # ── 7. Recent alert events + rules summary ────────────────────────────
    alert_rules_total = int(
        (await session.execute(
            select(func.count()).select_from(AlertRule)
        )).scalar_one() or 0
    )
    alert_rules_enabled = int(
        (await session.execute(
            select(func.count())
            .select_from(AlertRule)
            .where(AlertRule.enabled.is_(True))
        )).scalar_one() or 0
    )
    alert_rows = (
        await session.execute(
            select(AlertEvent).order_by(desc(AlertEvent.ts)).limit(20)
        )
    ).scalars().all()
    alert_events = [
        {
            "id": str(r.id),
            "ts": r.ts.isoformat() if r.ts else None,
            "rule_id": str(r.rule_id) if r.rule_id else None,
            "symbol": r.symbol,
            "matched": r.matched or {},
            "payload": r.payload or {},
        }
        for r in alert_rows
    ]

    # ── 8. Chain data quality (last 1 hour, then full table fallback) ─────
    # The pipeline silently produces zero metrics when bid/ask/iv are null,
    # so we expose coverage % per symbol so the operator can immediately
    # see whether the upstream feed is delivering quote data at all.
    chain_quality: list[dict[str, Any]] = []
    cutoff_1h = now - timedelta(hours=1)
    for symbol in settings.supported_symbols:
        row = (
            await session.execute(
                select(
                    func.count().label("rows"),
                    func.count(OptionsChain.bid).label("with_bid"),
                    func.count(OptionsChain.ask).label("with_ask"),
                    func.count(OptionsChain.last_price).label("with_last"),
                    func.count(OptionsChain.iv).label("with_iv"),
                    func.count(OptionsChain.delta).label("with_delta"),
                    func.count(OptionsChain.gamma).label("with_gamma"),
                    func.count(OptionsChain.oi).label("with_oi"),
                    func.count(OptionsChain.volume).label("with_volume"),
                    func.count(OptionsChain.underlying_price).label("with_underlying"),
                    func.max(OptionsChain.ts).label("latest_ts"),
                )
                .where(
                    OptionsChain.symbol == symbol,
                    OptionsChain.ts > cutoff_1h,
                )
            )
        ).first()
        # If no rows in last hour, fall back to the most recent rows so we
        # can still compute coverage when the market is closed.
        if row is None or int(row.rows or 0) == 0:
            row = (
                await session.execute(
                    select(
                        func.count().label("rows"),
                        func.count(OptionsChain.bid).label("with_bid"),
                        func.count(OptionsChain.ask).label("with_ask"),
                        func.count(OptionsChain.last_price).label("with_last"),
                        func.count(OptionsChain.iv).label("with_iv"),
                        func.count(OptionsChain.delta).label("with_delta"),
                        func.count(OptionsChain.gamma).label("with_gamma"),
                        func.count(OptionsChain.oi).label("with_oi"),
                        func.count(OptionsChain.volume).label("with_volume"),
                        func.count(OptionsChain.underlying_price).label("with_underlying"),
                        func.max(OptionsChain.ts).label("latest_ts"),
                    ).where(OptionsChain.symbol == symbol)
                )
            ).first()
        rows_count = int(row.rows or 0) if row else 0
        latest_ts = row.latest_ts if row else None
        chain_quality.append({
            "symbol": symbol,
            "rows_last_hour": rows_count,
            "latest_ts": latest_ts.isoformat() if latest_ts else None,
            "lag_seconds": _lag_seconds(latest_ts) if latest_ts else None,
            "coverage": {
                "bid": _pct(row.with_bid, rows_count) if row else None,
                "ask": _pct(row.with_ask, rows_count) if row else None,
                "last_price": _pct(row.with_last, rows_count) if row else None,
                "iv": _pct(row.with_iv, rows_count) if row else None,
                "delta": _pct(row.with_delta, rows_count) if row else None,
                "gamma": _pct(row.with_gamma, rows_count) if row else None,
                "oi": _pct(row.with_oi, rows_count) if row else None,
                "volume": _pct(row.with_volume, rows_count) if row else None,
                "underlying_price": _pct(row.with_underlying, rows_count) if row else None,
            },
        })

    # ── 9. Live ingester diagnostics ─────────────────────────────────────
    try:
        opra_diag = get_live_ingester().diagnostics()
    except Exception as exc:  # noqa: BLE001
        opra_diag = {"error": str(exc)}
    try:
        globex_diag = get_globex_live_ingester().diagnostics()
    except Exception as exc:  # noqa: BLE001
        globex_diag = {"error": str(exc)}

    return {
        "now": now.isoformat(),
        "supported_symbols": settings.supported_symbols,
        "tables": tables,
        "metric_breakdown": metric_breakdown,
        "latest_metrics": latest_metrics,
        "term_structure": term_structure,
        "pin_probability": pin_probability,
        "flow_events": flow_events,
        "alerts": {
            "rules_total": alert_rules_total,
            "rules_enabled": alert_rules_enabled,
            "events": alert_events,
        },
        "chain_quality": chain_quality,
        "ingesters": {
            "opra": opra_diag,
            "globex": globex_diag,
        },
    }


def _pct(part: Any, total: int) -> float | None:
    if total <= 0:
        return None
    try:
        p = int(part or 0)
    except (TypeError, ValueError):
        return None
    return round(100.0 * p / total, 1)


# ── DLQ paginated inspector ────────────────────────────────────────────────


@router.get("/dlq", response_model=DlqPage)
async def dlq_inspector(
    _admin: Annotated[str, Depends(authenticate_admin)],
    session: AsyncSession = Depends(get_db),
    limit: int = Query(50, gt=0, le=500),
    offset: int = Query(0, ge=0),
) -> DlqPage:
    """Paginated read-only view of the dead-letter queue.

    ``limit`` defaults to 50 and is capped at 500; ``offset`` defaults to 0.
    Out-of-range values return ``422 Unprocessable Entity`` via FastAPI's
    built-in query-param validation (``gt=0``, ``le=500``, ``ge=0``).
    """
    total = int(
        (
            await session.execute(
                select(func.count()).select_from(DeadLetterEntry)
            )
        ).scalar_one()
        or 0
    )
    rows = (
        await session.execute(
            select(DeadLetterEntry)
            .order_by(desc(DeadLetterEntry.ts))
            .limit(limit)
            .offset(offset)
        )
    ).scalars().all()
    items = [
        DlqEntry(
            id=row.id,
            ts=row.ts,
            source=row.source,
            reason=row.reason,
            payload=row.payload,
        )
        for row in rows
    ]
    return DlqPage(total=total, limit=limit, offset=offset, items=items)
