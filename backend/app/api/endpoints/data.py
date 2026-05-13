"""End-user data endpoints (require X-API-Key)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import limiter, require_symbol_access
from app.api.schemas import (
    DataEnvelope,
    GexResponse,
    IvResponse,
    MaxPainResponse,
    WallsResponse,
)
from app.config import get_settings
from app.db.models import ComputedMetric
from app.db.session import get_db

router = APIRouter()


_SYMBOL_PATTERN = r"^[A-Za-z0-9_.-]+$"

# GEX metric type per mode.
_GEX_METRIC_BY_MODE: dict[str, str] = {
    "oi": "GEX_NET_TOTAL",
    "volume": "GEX_NET_TOTAL_VOL",
}


def _parse_iso_date_or_400(value: str, field: str) -> date:
    """Parse ``YYYY-MM-DD`` strictly; raise HTTP 400 on malformed input.

    We use 400 (not 422) because the value lives inside an open-ended
    string ``Query`` rather than a typed parameter, so FastAPI's default
    422 validation does not fire — we surface a deliberate ``400 Bad
    Request`` so the client can distinguish "the date you sent is not a
    valid ISO-8601 date" from a generic schema violation.
    """
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{field} must be 'nearest', 'all', or an ISO-8601 date (YYYY-MM-DD)",
        ) from exc


async def _latest_metrics(
    session: AsyncSession,
    symbol: str,
    metric_type: str,
) -> list[ComputedMetric]:
    """Return rows of a given metric_type for the most recent ts of that type."""
    latest_ts_q = (
        select(ComputedMetric.ts)
        .where(ComputedMetric.symbol == symbol, ComputedMetric.metric_type == metric_type)
        .order_by(desc(ComputedMetric.ts))
        .limit(1)
    )
    latest_ts = (await session.execute(latest_ts_q)).scalar_one_or_none()
    if latest_ts is None:
        return []
    rows_q = select(ComputedMetric).where(
        ComputedMetric.symbol == symbol,
        ComputedMetric.metric_type == metric_type,
        ComputedMetric.ts == latest_ts,
    )
    return list((await session.execute(rows_q)).scalars().all())


def _envelope(symbol: str, computed_at: datetime | None, data: dict[str, Any]) -> DataEnvelope:
    settings = get_settings()
    next_in = settings.compute_interval_seconds
    if computed_at is not None:
        elapsed = (datetime.now(UTC) - computed_at).total_seconds()
        next_in = max(0, int(settings.compute_interval_seconds - elapsed))
    return DataEnvelope(
        symbol=symbol.upper(),
        computed_at=computed_at,
        next_update_in_seconds=next_in,
        data=data,
    )


# ── /v1/{symbol}/gex ────────────────────────────────────────────────────────


@router.get("/v1/{symbol}/gex", response_model=DataEnvelope[GexResponse])
@limiter.limit(lambda: f"{get_settings().rate_limit_per_minute}/minute")
async def get_gex(
    request: Request,  # noqa: ARG001
    symbol: str = Path(..., min_length=1, max_length=20, pattern=_SYMBOL_PATTERN),
    mode: str = Query("oi", pattern="^(oi|volume)$"),
    expiry: str = Query("all"),  # noqa: ARG001 - reserved for future use
    session: AsyncSession = Depends(get_db),
    _api_key=Depends(require_symbol_access()),
) -> DataEnvelope:
    metric_type = _GEX_METRIC_BY_MODE[mode]
    rows = await _latest_metrics(session, symbol.upper(), metric_type)
    if not rows:
        return _envelope(symbol, None, {"net_total": 0.0, "curve": [], "top_positive": [], "top_negative": []})
    row = rows[0]
    payload = dict(row.extra_json or {})
    payload["net_total"] = float(row.value or 0)
    payload.setdefault("curve", [])
    payload.setdefault("top_positive", [])
    payload.setdefault("top_negative", [])
    return _envelope(symbol, row.ts, payload)


# ── /v1/{symbol}/max-pain ───────────────────────────────────────────────────


@router.get("/v1/{symbol}/max-pain", response_model=DataEnvelope[MaxPainResponse])
@limiter.limit(lambda: f"{get_settings().rate_limit_per_minute}/minute")
async def get_max_pain(
    request: Request,  # noqa: ARG001
    symbol: str = Path(..., min_length=1, max_length=20, pattern=_SYMBOL_PATTERN),
    expiry: str = Query("nearest", min_length=1, max_length=10),
    session: AsyncSession = Depends(get_db),
    _api_key=Depends(require_symbol_access()),
) -> DataEnvelope:
    sym = symbol.upper()

    # Strict validation: expiry must be 'nearest', 'all' or a real ISO date.
    target_date: date | None = None
    if expiry not in ("nearest", "all"):
        target_date = _parse_iso_date_or_400(expiry, "expiry")

    per_expiry_rows = await _latest_metrics(session, sym, "MAX_PAIN")
    aggregate_rows = await _latest_metrics(session, sym, "MAX_PAIN_AGG")

    per_expiry = sorted(
        [
            {
                "expiration": str(r.expiration),
                "strike": float(r.strike),
                "pain": float(r.value or 0),
            }
            for r in per_expiry_rows
        ],
        key=lambda x: x["expiration"],
    )
    if target_date is not None:
        per_expiry = [e for e in per_expiry if e["expiration"] == target_date.isoformat()]
    elif expiry == "nearest":
        per_expiry = per_expiry[:1]

    aggregate = None
    if aggregate_rows:
        r = aggregate_rows[0]
        aggregate = {"strike": float(r.strike), "value": float(r.value or 0)}

    computed_at = (
        per_expiry_rows[0].ts if per_expiry_rows else (aggregate_rows[0].ts if aggregate_rows else None)
    )
    return _envelope(symbol, computed_at, {"per_expiry": per_expiry, "aggregate": aggregate})


# ── /v1/{symbol}/walls ──────────────────────────────────────────────────────


async def _walls_payload(session: AsyncSession, symbol: str, mode: str) -> dict:
    payload: dict[str, Any] = {}
    items: list[tuple[str, str]] = []
    if mode in ("oi", "both"):
        items.extend([("CALL_WALL_OI", "call_wall_oi"), ("PUT_WALL_OI", "put_wall_oi")])
    if mode in ("volume", "both"):
        items.extend([("CALL_WALL_VOL", "call_wall_volume"), ("PUT_WALL_VOL", "put_wall_volume")])

    computed_at: datetime | None = None
    for metric_type, key in items:
        rows = await _latest_metrics(session, symbol, metric_type)
        if rows and computed_at is None:
            computed_at = rows[0].ts
        payload[key] = sorted(
            [
                {
                    "rank": int((r.extra_json or {}).get("rank", 0)),
                    "strike": float(r.strike),
                    "value": float(r.value or 0),
                }
                for r in rows
            ],
            key=lambda x: x["rank"] or 999,
        )
    return {"computed_at": computed_at, "payload": payload}


@router.get("/v1/{symbol}/walls", response_model=DataEnvelope[WallsResponse])
@limiter.limit(lambda: f"{get_settings().rate_limit_per_minute}/minute")
async def get_walls(
    request: Request,  # noqa: ARG001
    symbol: str = Path(..., min_length=1, max_length=20, pattern=_SYMBOL_PATTERN),
    mode: str = Query("both", pattern="^(oi|volume|both)$"),
    session: AsyncSession = Depends(get_db),
    _api_key=Depends(require_symbol_access()),
) -> DataEnvelope:
    res = await _walls_payload(session, symbol.upper(), mode)
    payload: dict[str, Any] = {
        "call_wall_oi": [],
        "put_wall_oi": [],
        "call_wall_volume": [],
        "put_wall_volume": [],
    }
    payload.update(res["payload"])
    return _envelope(symbol, res["computed_at"], payload)


# ── /v1/{symbol}/iv ─────────────────────────────────────────────────────────


@router.get("/v1/{symbol}/iv", response_model=DataEnvelope[IvResponse])
@limiter.limit(lambda: f"{get_settings().rate_limit_per_minute}/minute")
async def get_iv(
    request: Request,  # noqa: ARG001
    symbol: str = Path(..., min_length=1, max_length=20, pattern=_SYMBOL_PATTERN),
    session: AsyncSession = Depends(get_db),
    _api_key=Depends(require_symbol_access()),
) -> DataEnvelope:
    sym = symbol.upper()
    atm_rows = await _latest_metrics(session, sym, "ATM_IV")
    skew_rows = await _latest_metrics(session, sym, "IV_SKEW")
    surface_rows = await _latest_metrics(session, sym, "IV_SURFACE")

    atm = float(atm_rows[0].value) if atm_rows and atm_rows[0].value is not None else None
    skew = {str(r.expiration): float(r.value or 0) for r in skew_rows}
    surface = (surface_rows[0].extra_json or {}).get("surface") if surface_rows else []

    computed_at = (
        atm_rows[0].ts if atm_rows else (skew_rows[0].ts if skew_rows else (surface_rows[0].ts if surface_rows else None))
    )
    # ``skew_per_expiry`` is kept for backward compatibility with existing
    # consumers; ``skew`` is the new typed-schema name.
    return _envelope(
        symbol,
        computed_at,
        {
            "atm_iv": atm,
            "skew": skew,
            "skew_per_expiry": skew,
            "surface": surface,
        },
    )


# ── /v1/{symbol}/snapshot ───────────────────────────────────────────────────


@router.get("/v1/{symbol}/snapshot", response_model=DataEnvelope)
@limiter.limit(lambda: f"{get_settings().rate_limit_per_minute}/minute")
async def get_snapshot(
    request: Request,  # noqa: ARG001
    symbol: str = Path(..., min_length=1, max_length=20, pattern=_SYMBOL_PATTERN),
    session: AsyncSession = Depends(get_db),
    _api_key=Depends(require_symbol_access()),
) -> DataEnvelope:
    sym = symbol.upper()
    if sym not in [s.upper() for s in get_settings().supported_symbols]:
        raise HTTPException(status_code=404, detail=f"Unsupported symbol {sym}")

    gex_rows = await _latest_metrics(session, sym, "GEX_NET_TOTAL")
    gex_payload: dict[str, Any] = {
        "net_total": 0.0,
        "curve": [],
        "top_positive": [],
        "top_negative": [],
        "zero_gamma": None,
    }
    if gex_rows:
        r = gex_rows[0]
        gex_payload = dict(r.extra_json or {})
        gex_payload["net_total"] = float(r.value or 0)
        gex_payload.setdefault("zero_gamma", None)

    gex_vol_rows = await _latest_metrics(session, sym, "GEX_NET_TOTAL_VOL")
    gex_vol_payload: dict[str, Any] = {
        "net_total": 0.0,
        "curve": [],
        "top_positive": [],
        "top_negative": [],
        "zero_gamma": None,
    }
    if gex_vol_rows:
        r = gex_vol_rows[0]
        gex_vol_payload = dict(r.extra_json or {})
        gex_vol_payload["net_total"] = float(r.value or 0)
        gex_vol_payload.setdefault("zero_gamma", None)

    # Top-level zero_gamma block: expose both flavours so consumers don't
    # have to dig into the nested gex/gex_volume payloads. Volume-weighted
    # is the variant the MotiveWave indicator renders by default.
    zero_gamma_payload = {
        "oi": gex_payload.get("zero_gamma"),
        "volume": gex_vol_payload.get("zero_gamma"),
        "underlying_price": (
            gex_vol_payload.get("underlying_price")
            or gex_payload.get("underlying_price")
        ),
    }

    mp_rows = await _latest_metrics(session, sym, "MAX_PAIN")
    mp_agg_rows = await _latest_metrics(session, sym, "MAX_PAIN_AGG")
    max_pain_payload = {
        "per_expiry": sorted(
            [
                {
                    "expiration": str(r.expiration),
                    "strike": float(r.strike),
                    "pain": float(r.value or 0),
                }
                for r in mp_rows
            ],
            key=lambda x: x["expiration"],
        ),
        "aggregate": (
            {"strike": float(mp_agg_rows[0].strike), "value": float(mp_agg_rows[0].value or 0)}
            if mp_agg_rows
            else None
        ),
    }

    walls = await _walls_payload(session, sym, "both")

    iv_atm = await _latest_metrics(session, sym, "ATM_IV")
    iv_skew = await _latest_metrics(session, sym, "IV_SKEW")
    iv_surface = await _latest_metrics(session, sym, "IV_SURFACE")
    iv_payload = {
        "atm_iv": float(iv_atm[0].value) if iv_atm and iv_atm[0].value is not None else None,
        "skew_per_expiry": {str(r.expiration): float(r.value or 0) for r in iv_skew},
        "surface": (iv_surface[0].extra_json or {}).get("surface") if iv_surface else [],
    }

    regime_oi_rows = await _latest_metrics(session, sym, "REGIME_OI")
    regime_vol_rows = await _latest_metrics(session, sym, "REGIME_VOL")

    def _regime_entry(rows):
        if not rows:
            return None
        r = rows[0]
        extra = dict(r.extra_json or {})
        return {
            "score": float(r.value or 0.0),
            "label": extra.get("label", "neutral"),
            "call_wall_total": float(extra.get("call_wall_total") or 0.0),
            "put_wall_total": float(extra.get("put_wall_total") or 0.0),
            "net_gex": float(extra.get("net_gex") or 0.0),
        }

    regime_payload = {
        "oi": _regime_entry(regime_oi_rows),
        "vol": _regime_entry(regime_vol_rows),
    }

    candidates = [
        r for r in (
            gex_rows + gex_vol_rows + mp_rows + mp_agg_rows
            + iv_atm + iv_skew + iv_surface
            + regime_oi_rows + regime_vol_rows
        ) if r
    ]
    computed_at = max([r.ts for r in candidates], default=None) if candidates else None

    return _envelope(
        symbol,
        computed_at,
        {
            "gex": gex_payload,
            "gex_volume": gex_vol_payload,
            "zero_gamma": zero_gamma_payload,
            "max_pain": max_pain_payload,
            "walls": walls["payload"],
            "iv": iv_payload,
            "regime": regime_payload,
        },
    )
