"""Compute pipeline: load the latest snapshot, run all metrics, persist results.

Rev 3 hardening (Agent 7):

* Every ``_persist_metrics`` call is wrapped in a single DB transaction so a
  partial failure rolls back cleanly — there is no half-written metric set.
* The loader's snapshot is sanity-checked for minimum coverage (bid+ask **or**
  IV present on ≥30% of rows). If neither holds, the tick is recorded as
  ``partial`` in ``pipeline_runs`` and metric computation is skipped.
* Every scheduler tick per symbol now persists a row to ``pipeline_runs``
  with ``started_at`` / ``finished_at`` / ``duration_ms`` / ``status`` /
  ``rows_read`` / ``metric_rows_written`` / ``missing_metric_types`` /
  ``error``.
* After ``_persist_metrics``, the latest ``metric_type`` set for the run's
  ``(symbol, ts)`` is diffed against :data:`EXPECTED_METRIC_TYPES`. Any
  shortfall is surfaced via ``missing_metric_types`` and downgrades the run
  status from ``ok`` to ``partial``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

import pandas as pd
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.logging import get_logger
from app.db.models import ComputedMetric, PipelineRun, SessionEvent
from app.db.session import get_session_factory
from app.processing.gex import GexSummary, compute_gex
from app.processing.iv import IVSummary, compute_iv_summary, fill_missing_iv
from app.processing.loader import load_latest_snapshot
from app.processing.max_pain import MaxPainSummary, compute_max_pain
from app.processing.move_tracker import MoveSnapshot, compute_move_tracker
from app.processing.pin_probability import compute_pin_probability
from app.processing.regime import RegimeSummary, compute_regime
from app.processing.session import (
    is_expiration_day,
    is_rth_now,
    session_snapshot,
    time_to_expiry_0dte_years,
)
from app.processing.spot import (
    SpotResult,
    reset_basis_cache,
    resolve_spot,
    spot_result_to_payload,
)
from app.processing.term_structure import compute_term_structure
from app.processing.vanna_charm import GreekSummary, compute_charm, compute_vanna
from app.processing.walls import WallsSummary, compute_walls
from app.processing.zero_dte import (
    BackMonthSummary,
    ZeroDteSummary,
    compute_back_month_summary,
    compute_zero_dte_summary,
)

logger = get_logger(__name__)


# ── Completeness contract ────────────────────────────────────────────────────
#
# The canonical list of ``metric_type`` discriminators that a single chain-
# pipeline tick is expected to produce for a "healthy" symbol. The list is
# derived from the ``metric_type`` literals written by :func:`_persist_metrics`
# below plus the Rev 3 additions (vanna/charm/term-structure/move-tracker/
# pin-probability). After every tick we diff the latest persisted set against
# this contract and surface the shortfall as ``pipeline_runs.missing_metric_types``.
#
# Metric types produced by *other* pipelines (``HIRO``, ``BASIS_SPX_ES``,
# ``VOLUME_PROFILE_ES`` from the flow pipeline) are intentionally **not**
# part of this contract — they run on their own cadence and we don't want
# a slow flow pipeline to mark the chain pipeline as partial.
EXPECTED_METRIC_TYPES: frozenset[str] = frozenset(
    {
        "GEX_NET_TOTAL",
        "GEX_LEVEL",
        "GEX_NET_TOTAL_VOL",
        "GEX_LEVEL_VOL",
        "MAX_PAIN",
        "MAX_PAIN_AGG",
        "CALL_WALL_OI",
        "PUT_WALL_OI",
        "CALL_WALL_VOL",
        "PUT_WALL_VOL",
        "ATM_IV",
        "IV_SKEW",
        "IV_SURFACE",
        "REGIME_OI",
        "REGIME_VOL",
        "VANNA_NET_TOTAL",
        "VANNA_LEVEL",
        "CHARM_NET_TOTAL",
        "CHARM_LEVEL",
        "IV_TERM_STRUCTURE",
        "RISK_REVERSAL_25D",
        "MOVE_TRACKER",
        "PIN_PROBABILITY",
        # Rev 4 — 0DTE + back-month split. These rows are always written;
        # on non-0DTE days every 0DTE row has value=0 and an explanatory
        # ``extra_json.reason`` so subscribers don't see gaps.
        "GEX_0DTE_NET_TOTAL",
        "GEX_0DTE_LEVEL",
        "GEX_0DTE_NET_TOTAL_VOL",
        "GEX_0DTE_LEVEL_VOL",
        "GEX_BACK_NET_TOTAL",
        "GEX_BACK_LEVEL",
        "GEX_BACK_NET_TOTAL_VOL",
        "GEX_BACK_LEVEL_VOL",
        "CHARM_0DTE_NET_TOTAL",
        "CHARM_0DTE_LEVEL",
        "CHARM_0DTE_DECAY_RATE",
        "GEX_0DTE_FLIP_SPEED",
        # Spot resolution snapshot — value=price, extra_json carries
        # source/futures/basis diagnostics.
        "SPOT",
    }
)

# Minimum fraction of rows that must carry usable bid+ask **or** IV for a
# snapshot to be considered worth computing on. Set deliberately low — we
# want to compute when the feed is at least partially healthy, but flag
# obviously-broken snapshots before they emit a fleet of zero metrics.
MIN_COVERAGE_FRACTION: float = 0.30


# ── Rev 4: flip-speed cache (symbol → (prev_net_gex_0dte, prev_ts_seconds)) ─
# Module-level so the next tick can compute Δ/Δt. Reset in
# :func:`reset_session_state` so flip-speed doesn't carry overnight noise.
_flip_speed_cache: dict[str, tuple[float, float]] = {}


def reset_flip_speed_cache(symbol: str | None = None) -> None:
    """Drop cached previous-tick GEX (used at session open + in tests)."""
    if symbol is None:
        _flip_speed_cache.clear()
    else:
        _flip_speed_cache.pop(symbol.upper(), None)


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
    spot: SpotResult | None = None
    session_state: dict[str, object] | None = None
    zero_dte: ZeroDteSummary | None = None
    back_month: BackMonthSummary | None = None


def _coverage_ok(df: pd.DataFrame) -> tuple[bool, dict[str, float]]:
    """Return (acceptable, diagnostics) for the loader's chain snapshot.

    A snapshot is acceptable when **either**:

    * ``bid`` *and* ``ask`` are present on at least
      :data:`MIN_COVERAGE_FRACTION` of rows, **or**
    * ``iv`` is present on at least :data:`MIN_COVERAGE_FRACTION` of rows.

    Diagnostics are returned alongside so callers can log them as
    structured context on the partial-run warning.
    """
    total = int(len(df))
    if total == 0:
        return False, {"rows_total": 0.0}

    have_bid = float(df["bid"].notna().sum()) if "bid" in df.columns else 0.0
    have_ask = float(df["ask"].notna().sum()) if "ask" in df.columns else 0.0
    have_iv = float(df["iv"].notna().sum()) if "iv" in df.columns else 0.0
    bid_ask_present = (
        float(((df["bid"].notna()) & (df["ask"].notna())).sum())
        if {"bid", "ask"}.issubset(df.columns)
        else 0.0
    )

    quote_frac = bid_ask_present / total
    iv_frac = have_iv / total
    diagnostics = {
        "rows_total": float(total),
        "rows_with_bid": have_bid,
        "rows_with_ask": have_ask,
        "rows_with_bid_and_ask": bid_ask_present,
        "rows_with_iv": have_iv,
        "quote_fraction": round(quote_frac, 4),
        "iv_fraction": round(iv_frac, 4),
        "min_required_fraction": MIN_COVERAGE_FRACTION,
    }
    acceptable = (
        quote_frac >= MIN_COVERAGE_FRACTION or iv_frac >= MIN_COVERAGE_FRACTION
    )
    return acceptable, diagnostics


async def _persist_metrics(
    session: AsyncSession, *, symbol: str, ts: datetime, result: PipelineResult
) -> int:
    """Upsert all metrics into ``computed_metrics`` inside a single transaction.

    If any statement in the transaction fails the entire upsert is rolled
    back so the next tick observes the prior state, not a half-written one.
    Returns the number of rows that would have been inserted.
    """
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

    # ── Rev 4: 0DTE-specific + back-month split ──────────────────────────
    # Always written, even on non-0DTE days, with value=0 and an
    # explanatory ``extra_json.reason``. This keeps the completeness
    # check happy and lets the UI distinguish "no 0DTE today" from
    # "computation failed".
    if result.zero_dte is not None:
        zdte = result.zero_dte
        reason = None if zdte.has_0dte else "no_0dte_today"
        # Net totals (always one row, even when has_0dte=False).
        for summary, total_type, level_type in (
            (zdte.gex_oi, "GEX_0DTE_NET_TOTAL", "GEX_0DTE_LEVEL"),
            (zdte.gex_vol, "GEX_0DTE_NET_TOTAL_VOL", "GEX_0DTE_LEVEL_VOL"),
        ):
            rows.append(
                {
                    "ts": ts,
                    "symbol": symbol,
                    "metric_type": total_type,
                    "strike": 0,
                    "expiration": sentinel_expiry,
                    "computed_at": ts,
                    "value": summary.net_total,
                    "extra_json": {
                        "underlying_price": summary.underlying_price,
                        "curve": summary.curve,
                        "top_positive": summary.top_positive,
                        "top_negative": summary.top_negative,
                        "zero_gamma": summary.zero_gamma,
                        "tau_years": zdte.tau_years,
                        "reason": reason,
                    },
                }
            )
            for level in summary.curve:
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

        # Charm rows (0DTE cohort only).
        rows.append(
            {
                "ts": ts,
                "symbol": symbol,
                "metric_type": "CHARM_0DTE_NET_TOTAL",
                "strike": 0,
                "expiration": sentinel_expiry,
                "computed_at": ts,
                "value": zdte.charm.net_total,
                "extra_json": {
                    "underlying_price": zdte.charm.underlying_price,
                    "curve": zdte.charm.curve,
                    "tau_years": zdte.tau_years,
                    "reason": reason,
                },
            }
        )
        for level in zdte.charm.curve:
            rows.append(
                {
                    "ts": ts,
                    "symbol": symbol,
                    "metric_type": "CHARM_0DTE_LEVEL",
                    "strike": level["strike"],
                    "expiration": sentinel_expiry,
                    "computed_at": ts,
                    "value": level.get("charm_exposure", 0.0),
                    "extra_json": level,
                }
            )

        # Scalars: decay rate + flip speed.
        rows.append(
            {
                "ts": ts,
                "symbol": symbol,
                "metric_type": "CHARM_0DTE_DECAY_RATE",
                "strike": 0,
                "expiration": sentinel_expiry,
                "computed_at": ts,
                "value": zdte.charm_decay_rate,
                "extra_json": {"reason": reason, "tau_years": zdte.tau_years},
            }
        )
        rows.append(
            {
                "ts": ts,
                "symbol": symbol,
                "metric_type": "GEX_0DTE_FLIP_SPEED",
                "strike": 0,
                "expiration": sentinel_expiry,
                "computed_at": ts,
                "value": zdte.flip_speed,
                "extra_json": {"reason": reason},
            }
        )

    if result.back_month is not None:
        bm = result.back_month
        for summary, total_type, level_type in (
            (bm.gex_oi, "GEX_BACK_NET_TOTAL", "GEX_BACK_LEVEL"),
            (bm.gex_vol, "GEX_BACK_NET_TOTAL_VOL", "GEX_BACK_LEVEL_VOL"),
        ):
            rows.append(
                {
                    "ts": ts,
                    "symbol": symbol,
                    "metric_type": total_type,
                    "strike": 0,
                    "expiration": sentinel_expiry,
                    "computed_at": ts,
                    "value": summary.net_total,
                    "extra_json": {
                        "underlying_price": summary.underlying_price,
                        "curve": summary.curve,
                        "top_positive": summary.top_positive,
                        "top_negative": summary.top_negative,
                        "zero_gamma": summary.zero_gamma,
                    },
                }
            )
            for level in summary.curve:
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

    # ── Rev 4: persist spot resolution result so /v1/{symbol}/spot can
    # serve the most recent reading without re-running the resolver.
    if result.spot is not None:
        rows.append(
            {
                "ts": ts,
                "symbol": symbol,
                "metric_type": "SPOT",
                "strike": 0,
                "expiration": sentinel_expiry,
                "computed_at": ts,
                "value": float(result.spot.price),
                "extra_json": spot_result_to_payload(result.spot),
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
    # Atomicity: a single execute is already a single statement, but we
    # bracket commit/rollback explicitly so any future multi-statement
    # additions inherit the same "all or nothing" semantics.
    try:
        await session.execute(stmt)
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    return len(rows)


async def _latest_persisted_metric_types(
    session: AsyncSession, *, symbol: str, ts: datetime
) -> set[str]:
    """Return the distinct ``metric_type`` set persisted at (symbol, ts)."""
    stmt = (
        select(ComputedMetric.metric_type)
        .where(ComputedMetric.symbol == symbol)
        .where(ComputedMetric.ts == ts)
        .distinct()
    )
    res = await session.execute(stmt)
    return {row[0] for row in res.all()}


def _missing_metric_types(persisted: set[str]) -> list[str]:
    """Diff a persisted set against :data:`EXPECTED_METRIC_TYPES`."""
    return sorted(EXPECTED_METRIC_TYPES - persisted)


async def _insert_pipeline_run(
    *, run_id: uuid.UUID, symbol: str, started_at: datetime
) -> None:
    """Insert the initial ``pipeline_runs`` row with status='running'.

    Uses its own session/transaction so the audit trail survives any
    later rollback of the metrics transaction.
    """
    factory = get_session_factory()
    async with factory() as s:
        s.add(
            PipelineRun(
                id=run_id,
                symbol=symbol,
                started_at=started_at,
                status="running",
            )
        )
        await s.commit()


async def _finalize_pipeline_run(
    *,
    run_id: uuid.UUID,
    status: str,
    started_at: datetime,
    finished_at: datetime,
    duration_ms: float,
    rows_read: int,
    metric_rows_written: int,
    missing_metric_types: list[str],
    error: str | None,
    is_expiration_day: bool = False,
    spot_source: str | None = None,
    spot_price: float | None = None,
    tau_0dte_years: float | None = None,
) -> None:
    """Update the previously-inserted ``pipeline_runs`` row with the result."""
    factory = get_session_factory()
    async with factory() as s:
        try:
            await s.execute(
                update(PipelineRun)
                .where(PipelineRun.id == run_id)
                .values(
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=duration_ms,
                    status=status,
                    rows_read=rows_read,
                    metric_rows_written=metric_rows_written,
                    missing_metric_types=missing_metric_types,
                    error=error,
                    is_expiration_day=is_expiration_day,
                    spot_source=spot_source,
                    spot_price=spot_price,
                    tau_0dte_years=tau_0dte_years,
                )
            )
            await s.commit()
        except Exception:
            await s.rollback()
            # Never let audit-log persistence kill the pipeline tick.
            logger.exception(
                "pipeline_run_persist_error", symbol=None, run_id=str(run_id)
            )


async def run_pipeline_for_symbol(symbol: str) -> PipelineResult | None:
    """Run the chain pipeline for ``symbol`` once.

    Every invocation persists exactly one ``pipeline_runs`` row regardless
    of whether the tick produced metrics. The row's ``status`` follows:

    * ``ok``       — snapshot loaded, coverage OK, ``_persist_metrics`` returned
      a complete metric set.
    * ``partial``  — snapshot was empty / under-covered / metric set missed
      some of :data:`EXPECTED_METRIC_TYPES`.
    * ``failed``   — an exception escaped one of the steps.
    """
    settings = get_settings()
    factory = get_session_factory()
    started = perf_counter()
    started_at = datetime.now(UTC)
    ts = started_at.replace(microsecond=0)

    run_id = uuid.uuid4()
    await _insert_pipeline_run(run_id=run_id, symbol=symbol, started_at=started_at)

    status: str = "ok"
    error_msg: str | None = None
    rows_read: int = 0
    metric_rows_written: int = 0
    missing: list[str] = []
    result: PipelineResult | None = None
    spot: SpotResult | None = None
    sess_state = session_snapshot(symbol=symbol)
    is_exp_today = bool(sess_state.get("is_expiration_day", False))
    tau_years = float(sess_state.get("time_to_expiry_0dte_years", 0.0))

    try:
        async with factory() as session:
            df = await load_latest_snapshot(session, symbol)
            # ── Rev 4: resolve spot via futures-first chain BEFORE metrics.
            #     The result overrides ``underlying_price`` on every row so
            #     every Greek computation downstream sees the same S.
            spot = await resolve_spot(symbol, df, session)
        rows_read = int(len(df))

        if spot is not None and not df.empty:
            df = df.copy()
            df["underlying_price"] = float(spot.price)

        if df.empty:
            logger.info("pipeline_no_data", symbol=symbol)
            status = "partial"
            missing = sorted(EXPECTED_METRIC_TYPES)
        else:
            # Run IV inversion before the coverage check so synthesized IV
            # also counts toward the threshold.
            df = fill_missing_iv(df, risk_free_rate=settings.risk_free_rate)
            cov_ok, cov_diag = _coverage_ok(df)
            if not cov_ok:
                logger.warning(
                    "pipeline_low_coverage",
                    symbol=symbol,
                    **cov_diag,
                    hint=(
                        "Skipping metric computation: neither bid+ask nor IV "
                        f"meets the {MIN_COVERAGE_FRACTION:.0%} coverage "
                        "threshold. Check feed health (cmbp-1 NBBO present?)."
                    ),
                )
                status = "partial"
                missing = sorted(EXPECTED_METRIC_TYPES)
            else:
                result = _compute_metrics(df=df, symbol=symbol, ts=ts, settings=settings)
                result.spot = spot
                result.session_state = sess_state

                async with factory() as session:
                    metric_rows_written = await _persist_metrics(
                        session, symbol=symbol, ts=ts, result=result
                    )

                async with factory() as session:
                    persisted = await _latest_persisted_metric_types(
                        session, symbol=symbol, ts=ts
                    )
                missing = _missing_metric_types(persisted)
                status = "ok" if not missing else "partial"

    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error_msg = f"{type(exc).__name__}: {exc}"
        # Drop the partially-computed result: callers must not consume
        # metrics that were never persisted.
        result = None
        logger.exception("pipeline_error", symbol=symbol)

    finished_at = datetime.now(UTC)
    duration_ms = (perf_counter() - started) * 1000

    await _finalize_pipeline_run(
        run_id=run_id,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        rows_read=rows_read,
        metric_rows_written=metric_rows_written,
        missing_metric_types=missing,
        error=error_msg,
        is_expiration_day=is_exp_today,
        spot_source=spot.source if spot is not None else None,
        spot_price=float(spot.price) if spot is not None else None,
        tau_0dte_years=tau_years,
    )

    if result is not None:
        result.duration_ms = duration_ms
        logger.info(
            "pipeline_complete",
            symbol=symbol,
            status=status,
            duration_ms=duration_ms,
            snapshot_rows=rows_read,
            metric_rows=metric_rows_written,
            missing=missing,
        )

    # ── Agent 5 streaming hook ───────────────────────────────────────────
    # Best-effort fan-out to any WS / SSE subscribers. We publish on any
    # tick that produced metrics (ok or partial) so subscribers continue
    # receiving frames even when one slow metric type (e.g. PIN_PROBABILITY,
    # which needs futures data) is missing. Failures here must never
    # poison the pipeline tick — log and move on.
    if status in ("ok", "partial") and result is not None:
        try:
            await _publish_streaming_snapshot(symbol)
        except Exception:  # noqa: BLE001
            logger.exception("streaming_publish_failed", symbol=symbol)

    return result


async def _publish_streaming_snapshot(symbol: str) -> None:
    """Build the comprehensive snapshot payload and broadcast to subscribers.

    Imported lazily so the processing module stays decoupled from the API
    surface at import time. The notifier itself drops the oldest queued
    frame for slow subscribers, so this never blocks the pipeline.
    """
    from app.api.endpoints.snapshot import build_snapshot_payload
    from app.api.stream_notifier import get_stream_notifier

    notifier = get_stream_notifier()
    if notifier.subscriber_count(symbol) == 0:
        return
    factory = get_session_factory()
    async with factory() as session:
        payload, computed_at = await build_snapshot_payload(session, symbol)
    await notifier.publish(
        symbol, {"data": payload, "computed_at": computed_at}
    )


def _compute_metrics(
    *,
    df: pd.DataFrame,
    symbol: str,
    ts: datetime,
    settings,
) -> PipelineResult:
    """Pure-CPU portion of the tick: compute every metric from the snapshot."""
    rows_total = int(len(df))
    have_underlying = (
        int(df["underlying_price"].notna().sum()) if "underlying_price" in df.columns else 0
    )
    have_iv = int(df["iv"].notna().sum()) if "iv" in df.columns else 0
    have_gamma = int(df["gamma"].notna().sum()) if "gamma" in df.columns else 0

    if have_underlying == 0:
        logger.warning(
            "pipeline_no_underlying",
            symbol=symbol,
            rows=rows_total,
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
    move_tracker = compute_move_tracker(df, open_price=None)

    # Rev 4 — 0DTE / back-month split. Pull the prior tick's 0DTE net GEX
    # from the symbol-local cache so we can derive flip speed Δ/Δt.
    prev = _flip_speed_cache.get(symbol)
    now_ts_seconds = ts.timestamp()
    prev_net_gex = prev[0] if prev is not None else None
    prev_ts_seconds = prev[1] if prev is not None else None

    zero_dte = compute_zero_dte_summary(
        df,
        risk_free_rate=settings.risk_free_rate,
        atm_band_pct=getattr(settings, "atm_band_pct_0dte", 0.005),
        prev_net_gex=prev_net_gex,
        prev_ts_seconds=prev_ts_seconds,
        now_ts_seconds=now_ts_seconds,
    )
    back_month = compute_back_month_summary(
        df, risk_free_rate=settings.risk_free_rate
    )
    # Update flip-speed cache with this tick's OI-weighted 0DTE net GEX.
    _flip_speed_cache[symbol] = (zero_dte.gex_oi.net_total, now_ts_seconds)

    return PipelineResult(
        symbol=symbol,
        ts=ts,
        duration_ms=0.0,
        rows=rows_total,
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
        zero_dte=zero_dte,
        back_month=back_month,
    )


# ────────────────────────────────────────────────────────────────────────────
# Rev 4 — session lifecycle hooks
# ────────────────────────────────────────────────────────────────────────────


async def _record_session_event(
    *,
    event_type: str,
    symbol: str | None,
    extra: dict[str, object] | None = None,
) -> None:
    """Insert a row into ``session_events`` so the admin/inspector knows
    when the scheduler last opened / closed / reset state."""
    factory = get_session_factory()
    async with factory() as s:
        try:
            s.add(
                SessionEvent(
                    event_type=event_type,
                    symbol=symbol,
                    extra_json=extra or {},
                )
            )
            await s.commit()
        except Exception:
            await s.rollback()
            logger.exception(
                "session_event_persist_error",
                event_type=event_type,
                symbol=symbol,
            )


async def reset_session_state(symbols: list[str]) -> None:
    """Wipe per-session caches at 09:29 ET.

    * Clears the futures-basis EMA cache (each new session needs to
      re-establish basis as the carry / dividend assumption may have
      changed overnight).
    * Inserts a ``session_open`` audit row per symbol so the timeline
      view in /admin/inspector lines up cleanly.

    HIRO accumulators live in :mod:`app.processing.hiro` and reset
    automatically on the first call of a new session because that
    module keys its bucket cumulative by trade-date.
    """
    logger.info("session.reset", symbols=symbols)
    for symbol in symbols:
        reset_basis_cache(symbol)
        reset_flip_speed_cache(symbol)
        await _record_session_event(
            event_type="session_open",
            symbol=symbol,
            extra={"reset_basis_cache": True, "reset_flip_speed_cache": True},
        )

    # Sentinel pipeline_runs row so /admin/system/status can show
    # "last session opened at HH:MM" without joining session_events.
    factory = get_session_factory()
    now = datetime.now(UTC)
    async with factory() as s:
        try:
            for symbol in symbols:
                s.add(
                    PipelineRun(
                        id=uuid.uuid4(),
                        symbol=symbol,
                        started_at=now,
                        finished_at=now,
                        duration_ms=0,
                        status="session_open",
                        is_expiration_day=is_expiration_day(symbol),
                        tau_0dte_years=time_to_expiry_0dte_years(),
                    )
                )
            await s.commit()
        except Exception:
            await s.rollback()
            logger.exception("session_open_sentinel_persist_error")


async def finalize_session(symbols: list[str]) -> None:
    """End-of-session hook called at 16:16 ET.

    Today this only records the close in ``session_events`` and writes a
    sentinel ``pipeline_runs`` row. The richer end-of-day HIRO summary
    is computed by the flow pipeline; this hook is the synchronization
    point that tells everyone "no more frames after this".
    """
    logger.info("session.finalize", symbols=symbols)
    for symbol in symbols:
        await _record_session_event(
            event_type="session_close",
            symbol=symbol,
            extra=None,
        )

    factory = get_session_factory()
    now = datetime.now(UTC)
    async with factory() as s:
        try:
            for symbol in symbols:
                s.add(
                    PipelineRun(
                        id=uuid.uuid4(),
                        symbol=symbol,
                        started_at=now,
                        finished_at=now,
                        duration_ms=0,
                        status="session_close",
                        is_expiration_day=is_expiration_day(symbol),
                        tau_0dte_years=0.0,
                    )
                )
            await s.commit()
        except Exception:
            await s.rollback()
            logger.exception("session_close_sentinel_persist_error")


__all__ = [
    "EXPECTED_METRIC_TYPES",
    "MIN_COVERAGE_FRACTION",
    "PipelineResult",
    "finalize_session",
    "is_rth_now",
    "reset_session_state",
    "run_pipeline_for_symbol",
    "spot_result_to_payload",
]
