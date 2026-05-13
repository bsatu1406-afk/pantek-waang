"""Compute pipeline: load the latest snapshot, run all metrics, persist results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.logging import get_logger
from app.db.models import ComputedMetric
from app.db.session import get_session_factory
from app.processing.gex import GexSummary, compute_gex
from app.processing.iv import IVSummary, compute_iv_summary, fill_missing_iv
from app.processing.loader import load_latest_snapshot
from app.processing.max_pain import MaxPainSummary, compute_max_pain
from app.processing.move_tracker import MoveSnapshot, compute_move_tracker
from app.processing.pin_probability import compute_pin_probability
from app.processing.regime import RegimeSummary, compute_regime
from app.processing.term_structure import compute_term_structure
from app.processing.vanna_charm import GreekSummary, compute_charm, compute_vanna
from app.processing.walls import WallsSummary, compute_walls

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    symbol: str
    ts: datetime
    duration_ms: float
    rows: int
    gex: GexSummary
    gex_volume: GexSummary
    max_pain: MaxPainSummary
    walls: WallsSummary
    iv: IVSummary
    regime: RegimeSummary
    vanna: GreekSummary
    charm: GreekSummary
    term_structure: list[dict]
    move_tracker: MoveSnapshot
    pin_probability: list[dict]


async def _persist_metrics(
    session: AsyncSession, *, symbol: str, ts: datetime, result: PipelineResult
) -> int:
    """Upsert all metrics into ``computed_metrics``. Returns rows inserted."""
    rows: list[dict] = []
    sentinel_expiry = pd.Timestamp("1970-01-01").date()

    # GEX rows — both OI-weighted and Volume-weighted variants are persisted
    # under distinct metric_type discriminators so the API can expose both
    # in /v1/{symbol}/snapshot.
    for gex_summary, total_type, level_type in (
        (result.gex,        "GEX_NET_TOTAL",     "GEX_LEVEL"),
        (result.gex_volume, "GEX_NET_TOTAL_VOL", "GEX_LEVEL_VOL"),
    ):
        rows.append(
            {
                "ts": ts,
                "symbol": symbol,
                "metric_type": total_type,
                "strike": 0,
                "expiration": sentinel_expiry,
                "computed_at": ts,
                "value": gex_summary.net_total,
                "extra_json": {
                    "underlying_price": gex_summary.underlying_price,
                    "curve": gex_summary.curve,
                    "top_positive": gex_summary.top_positive,
                    "top_negative": gex_summary.top_negative,
                    "zero_gamma": gex_summary.zero_gamma,
                    "weight_col": gex_summary.weight_col,
                },
            }
        )
        for level in gex_summary.curve:
            rows.append(
                {
                    "ts": ts,
                    "symbol": symbol,
                    "metric_type": level_type,
                    "strike": level["strike"],
                    "expiration": sentinel_expiry,
                    "computed_at": ts,
                    "value": level.get("net_gex", 0.0),
                    "extra_json": level,
                }
            )

    # Max pain
    for entry in result.max_pain.per_expiry:
        if entry["strike"] is None:
            continue
        rows.append(
            {
                "ts": ts,
                "symbol": symbol,
                "metric_type": "MAX_PAIN",
                "strike": entry["strike"],
                "expiration": pd.Timestamp(entry["expiration"]).date(),
                "computed_at": ts,
                "value": entry.get("pain"),
                "extra_json": {"curve": entry.get("curve", [])},
            }
        )
    if result.max_pain.aggregate_strike is not None:
        rows.append(
            {
                "ts": ts,
                "symbol": symbol,
                "metric_type": "MAX_PAIN_AGG",
                "strike": result.max_pain.aggregate_strike,
                "expiration": sentinel_expiry,
                "computed_at": ts,
                "value": result.max_pain.aggregate_value,
                "extra_json": {"window_expiries": 5},
            }
        )

    # Walls
    for kind, payload in (("OI", result.walls.by_oi), ("VOL", result.walls.by_volume)):
        for side, arr in (("CALL_WALL", payload.get("call_wall", [])),
                          ("PUT_WALL", payload.get("put_wall", []))):
            metric_type = f"{side}_{kind}"
            for rank, entry in enumerate(arr, start=1):
                rows.append(
                    {
                        "ts": ts,
                        "symbol": symbol,
                        "metric_type": metric_type,
                        "strike": entry["strike"],
                        "expiration": sentinel_expiry,
                        "computed_at": ts,
                        "value": entry["value"],
                        "extra_json": {"rank": rank},
                    }
                )

    # IV
    if result.iv.atm_iv is not None:
        rows.append(
            {
                "ts": ts,
                "symbol": symbol,
                "metric_type": "ATM_IV",
                "strike": 0,
                "expiration": sentinel_expiry,
                "computed_at": ts,
                "value": result.iv.atm_iv,
                "extra_json": None,
            }
        )
    for expiry, skew_value in result.iv.skew_per_expiry.items():
        rows.append(
            {
                "ts": ts,
                "symbol": symbol,
                "metric_type": "IV_SKEW",
                "strike": 0,
                "expiration": pd.Timestamp(expiry).date(),
                "computed_at": ts,
                "value": skew_value,
                "extra_json": None,
            }
        )
    if result.iv.surface:
        rows.append(
            {
                "ts": ts,
                "symbol": symbol,
                "metric_type": "IV_SURFACE",
                "strike": 0,
                "expiration": sentinel_expiry,
                "computed_at": ts,
                "value": None,
                "extra_json": {"surface": result.iv.surface},
            }
        )

    # Regime (one row per mode, score in [-1, +1] in `value`).
    for mode_name, mode_payload in (("REGIME_OI", result.regime.oi),
                                    ("REGIME_VOL", result.regime.vol)):
        rows.append(
            {
                "ts": ts,
                "symbol": symbol,
                "metric_type": mode_name,
                "strike": 0,
                "expiration": sentinel_expiry,
                "computed_at": ts,
                "value": mode_payload.score,
                "extra_json": {
                    "label": mode_payload.label,
                    "call_wall_total": mode_payload.call_wall_total,
                    "put_wall_total": mode_payload.put_wall_total,
                    "net_gex": mode_payload.net_gex,
                },
            }
        )

    # ── Vanna & Charm (mirror of GEX persistence layout) ─────────────────
    for greek_summary, total_type, level_type in (
        (result.vanna, "VANNA_NET_TOTAL", "VANNA_LEVEL"),
        (result.charm, "CHARM_NET_TOTAL", "CHARM_LEVEL"),
    ):
        rows.append(
            {
                "ts": ts,
                "symbol": symbol,
                "metric_type": total_type,
                "strike": 0,
                "expiration": sentinel_expiry,
                "computed_at": ts,
                "value": greek_summary.net_total,
                "extra_json": {
                    "underlying_price": greek_summary.underlying_price,
                    "curve": greek_summary.curve,
                    "top_positive": greek_summary.top_positive,
                    "top_negative": greek_summary.top_negative,
                    "weight_col": greek_summary.weight_col,
                },
            }
        )
        for level in greek_summary.curve:
            value = level.get("vanna_exposure",
                              level.get("charm_exposure", 0.0))
            rows.append(
                {
                    "ts": ts,
                    "symbol": symbol,
                    "metric_type": level_type,
                    "strike": level["strike"],
                    "expiration": sentinel_expiry,
                    "computed_at": ts,
                    "value": value,
                    "extra_json": level,
                }
            )

    # ── Term-structure (one row per expiration) ──────────────────────────
    for entry in result.term_structure:
        try:
            exp_date = pd.Timestamp(entry["expiration"]).date()
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "ts": ts,
                "symbol": symbol,
                "metric_type": "IV_TERM_STRUCTURE",
                "strike": 0,
                "expiration": exp_date,
                "computed_at": ts,
                "value": entry.get("atm_iv"),
                "extra_json": entry,
            }
        )
        if entry.get("risk_reversal_25d") is not None:
            rows.append(
                {
                    "ts": ts,
                    "symbol": symbol,
                    "metric_type": "RISK_REVERSAL_25D",
                    "strike": 0,
                    "expiration": exp_date,
                    "computed_at": ts,
                    "value": entry["risk_reversal_25d"],
                    "extra_json": {
                        "call_25d_iv": entry.get("call_25d_iv"),
                        "put_25d_iv": entry.get("put_25d_iv"),
                    },
                }
            )

    # ── Realized vs Implied Move tracker (single row) ────────────────────
    if (
        result.move_tracker.realized_move is not None
        or result.move_tracker.implied_move is not None
    ):
        rows.append(
            {
                "ts": ts,
                "symbol": symbol,
                "metric_type": "MOVE_TRACKER",
                "strike": 0,
                "expiration": sentinel_expiry,
                "computed_at": ts,
                "value": result.move_tracker.ratio,
                "extra_json": {
                    "underlying_price": result.move_tracker.underlying_price,
                    "open_price": result.move_tracker.open_price,
                    "realized_move": result.move_tracker.realized_move,
                    "implied_move": result.move_tracker.implied_move,
                    "implied_dte": result.move_tracker.implied_dte,
                    "ratio": result.move_tracker.ratio,
                },
            }
        )

    # ── Pin probability heatmap (one row per 0DTE strike) ────────────────
    for entry in result.pin_probability:
        rows.append(
            {
                "ts": ts,
                "symbol": symbol,
                "metric_type": "PIN_PROBABILITY",
                "strike": entry["strike"],
                "expiration": sentinel_expiry,
                "computed_at": ts,
                "value": entry["prob"],
                "extra_json": entry,
            }
        )

    if not rows:
        return 0

    stmt = insert(ComputedMetric).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["ts", "symbol", "metric_type", "strike", "expiration"],
        set_={
            "computed_at": stmt.excluded.computed_at,
            "value": stmt.excluded.value,
            "extra_json": stmt.excluded.extra_json,
        },
    )
    await session.execute(stmt)
    await session.commit()
    return len(rows)


async def run_pipeline_for_symbol(symbol: str) -> PipelineResult | None:
    settings = get_settings()
    factory = get_session_factory()
    started = perf_counter()
    ts = datetime.now(UTC).replace(microsecond=0)

    async with factory() as session:
        df = await load_latest_snapshot(session, symbol)

    if df.empty:
        logger.info("pipeline_no_data", symbol=symbol)
        return None

    # Diagnostic: surface upstream feed-quality issues loudly. The pipeline
    # silently emits zero-valued metrics when iv/greeks/spot can't be
    # derived, which has historically masked subscription problems
    # (e.g. the OPRA Pillar Standard plan ships trades + statistics +
    # definitions but NOT cmbp-1 NBBO updates, so bid/ask stay null).
    rows_total = int(len(df))
    have_bid = int(df["bid"].notna().sum()) if "bid" in df.columns else 0
    have_ask = int(df["ask"].notna().sum()) if "ask" in df.columns else 0
    have_last = int(df["last_price"].notna().sum()) if "last_price" in df.columns else 0
    have_underlying = (
        int(df["underlying_price"].notna().sum()) if "underlying_price" in df.columns else 0
    )

    df = fill_missing_iv(df, risk_free_rate=settings.risk_free_rate)

    have_iv = int(df["iv"].notna().sum()) if "iv" in df.columns else 0
    have_gamma = int(df["gamma"].notna().sum()) if "gamma" in df.columns else 0

    if have_underlying == 0:
        logger.warning(
            "pipeline_no_underlying",
            symbol=symbol,
            rows=rows_total,
            have_bid=have_bid,
            have_ask=have_ask,
            have_last=have_last,
            hint=(
                "Spot synthesis failed — chain has no usable bid/ask or last_price. "
                "Check ingester diagnostics in /admin/inspector for dropped schemas "
                "(cmbp-1 not available?) or live record_counts."
            ),
        )
    elif have_iv == 0 or have_gamma == 0:
        logger.warning(
            "pipeline_low_greek_coverage",
            symbol=symbol,
            rows=rows_total,
            have_iv=have_iv,
            have_gamma=have_gamma,
            have_underlying=have_underlying,
        )

    gex = compute_gex(df, weight_col="oi", risk_free_rate=settings.risk_free_rate)
    gex_vol = compute_gex(
        df, weight_col="volume", risk_free_rate=settings.risk_free_rate
    )
    mp = compute_max_pain(df)
    walls = compute_walls(df)
    iv = compute_iv_summary(df)
    regime = compute_regime(walls, gex, gex_vol)
    vanna = compute_vanna(df, weight_col="oi", risk_free_rate=settings.risk_free_rate)
    charm = compute_charm(df, weight_col="oi", risk_free_rate=settings.risk_free_rate)
    term_structure = compute_term_structure(df)
    pin_probability = compute_pin_probability(
        df, risk_free_rate=settings.risk_free_rate
    )
    # Open price is supplied externally — for now we don't have a session
    # cache, so pass None (move_tracker will surface implied side only).
    move_tracker = compute_move_tracker(df, open_price=None)

    result = PipelineResult(
        symbol=symbol,
        ts=ts,
        duration_ms=0.0,
        rows=int(len(df)),
        gex=gex,
        gex_volume=gex_vol,
        max_pain=mp,
        walls=walls,
        iv=iv,
        regime=regime,
        vanna=vanna,
        charm=charm,
        term_structure=term_structure,
        move_tracker=move_tracker,
        pin_probability=pin_probability,
    )

    async with factory() as session:
        inserted = await _persist_metrics(session, symbol=symbol, ts=ts, result=result)

    duration_ms = (perf_counter() - started) * 1000
    result.duration_ms = duration_ms
    logger.info(
        "pipeline_complete",
        symbol=symbol,
        duration_ms=duration_ms,
        snapshot_rows=int(len(df)),
        metric_rows=inserted,
    )
    return result
