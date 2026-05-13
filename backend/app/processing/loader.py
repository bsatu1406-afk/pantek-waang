"""Load the latest options-chain snapshot per (expiration, strike, option_type) for a symbol.

When live OI is missing/zero (common for OPRA Pillar definition-only feeds),
we fall back to the most recently ingested end-of-day Open Interest snapshot
from ``eod_open_interest`` so downstream metrics (GEX-by-OI, walls-by-OI) still
have meaningful weights to use.

When ``underlying_price`` is missing (OPRA Pillar does not publish the SPX/NDX
cash index), we resolve it in this priority order:

1. Put-call parity from the freshest near-the-money pair (see :mod:`app.processing.spot`).
2. Front-month CME futures last price minus a cached basis (ES → SPX, NQ → NDX).

Both fallbacks are necessary because parity drifts when the ATM spread is
wide / one-sided, while the futures price needs basis context to translate
to cash-index space.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.processing.spot import synthesize_underlying_price

SNAPSHOT_QUERY = text(
    """
    SELECT DISTINCT ON (expiration, strike, option_type)
        ts, symbol, expiration, strike, option_type,
        oi, volume, iv, delta, gamma, last_price, bid, ask, underlying_price
    FROM options_chain
    WHERE symbol = :symbol
      AND ts > NOW() - INTERVAL '2 days'
    ORDER BY expiration, strike, option_type, ts DESC
    """
)


EOD_OI_QUERY = text(
    """
    SELECT expiration, strike, option_type, open_interest
    FROM eod_open_interest
    WHERE symbol = :symbol
    """
)


# Cash index ↔ front-month CME futures root mapping.
_FUTURES_ROOT_FOR_SYMBOL = {
    "SPXW": "ES",
    "SPX": "ES",
    "NDXP": "NQ",
    "NDX": "NQ",
}


# Latest known cash-minus-futures basis per cash symbol.
# Populated whenever both the parity-derived spot and the futures last
# price are known in the same load_latest_snapshot call. Used to translate
# a future-only price back to cash-index space when parity fails next time.
_BASIS_CACHE: dict[str, float] = {}


_FUTURES_LAST_QUERY = text(
    """
    SELECT price
    FROM futures_ticks
    WHERE symbol LIKE :prefix
      AND ts > NOW() - INTERVAL '15 minutes'
    ORDER BY ts DESC
    LIMIT 1
    """
)


async def load_latest_snapshot(session: AsyncSession, symbol: str) -> pd.DataFrame:
    result = await session.execute(SNAPSHOT_QUERY, {"symbol": symbol})
    rows = result.mappings().all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame.from_records([dict(r) for r in rows])
    # Coerce numeric columns to floats for downstream math.
    for col in ("strike", "iv", "delta", "gamma", "last_price", "bid", "ask", "underlying_price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("oi", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fold in EOD OI fallback for any rows where live OI is null/zero.
    df = await _apply_eod_oi_fallback(session, symbol, df)

    # Synthesize underlying spot via put-call parity if the feed didn't
    # provide one, then fall back to ES/NQ futures + cached basis when
    # parity also fails.
    df = await _apply_underlying_synthesis(session, symbol, df)
    return df


async def _apply_underlying_synthesis(
    session: AsyncSession, symbol: str, df: pd.DataFrame
) -> pd.DataFrame:
    if df.empty or "underlying_price" not in df.columns:
        return df
    have_spot = df["underlying_price"].dropna()
    if not have_spot.empty and float(have_spot.iloc[-1] or 0) > 0:
        return df

    settings = get_settings()
    fut_last = await _latest_front_month_future(session, symbol)
    parity_spot = synthesize_underlying_price(df, risk_free_rate=settings.risk_free_rate)

    # Refresh the cached basis whenever both legs are known.
    if parity_spot is not None and fut_last is not None:
        _BASIS_CACHE[symbol.upper()] = float(fut_last) - float(parity_spot)

    spot: float | None = parity_spot
    if spot is None and fut_last is not None:
        cached_basis = _BASIS_CACHE.get(symbol.upper(), 0.0)
        spot = float(fut_last) - cached_basis

    if spot is None:
        return df
    df = df.copy()
    df["underlying_price"] = spot
    return df


async def _latest_front_month_future(
    session: AsyncSession, symbol: str
) -> float | None:
    """Return the most recent ES/NQ outright tick price, or ``None``."""
    root = _FUTURES_ROOT_FOR_SYMBOL.get(symbol.upper())
    if root is None:
        return None
    # Match outright contracts (e.g. ``ESM6``) but not calendar spreads
    # (which embed a ``-`` such as ``ESH7-ESM7``). The size-3-letter pattern
    # ``ESxN`` doesn't fit a single LIKE so we intentionally widen and let
    # the LIMIT 1 ORDER BY ts pick the freshest single trade — outrights
    # dominate volume so the freshest tick is overwhelmingly an outright.
    res = await session.execute(_FUTURES_LAST_QUERY, {"prefix": f"{root}%"})
    row = res.first()
    if row is None or row[0] is None:
        return None
    try:
        price = float(row[0])
    except (TypeError, ValueError):
        return None
    if not (price > 0):
        return None
    return price


async def _apply_eod_oi_fallback(
    session: AsyncSession, symbol: str, df: pd.DataFrame
) -> pd.DataFrame:
    """Fill rows where ``oi`` is null or zero from ``eod_open_interest``."""
    if df.empty or "oi" not in df.columns:
        return df

    needs_fill = df["oi"].isna() | (df["oi"].fillna(0) == 0)
    if not needs_fill.any():
        return df

    result = await session.execute(EOD_OI_QUERY, {"symbol": symbol})
    rows = result.mappings().all()
    if not rows:
        return df

    eod = pd.DataFrame.from_records([dict(r) for r in rows])
    if eod.empty:
        return df
    eod["strike"] = pd.to_numeric(eod["strike"], errors="coerce")
    eod["open_interest"] = pd.to_numeric(eod["open_interest"], errors="coerce").fillna(0)
    eod["option_type"] = eod["option_type"].astype(str).str.upper()
    df["option_type"] = df["option_type"].astype(str).str.upper()

    merged = df.merge(
        eod[["expiration", "strike", "option_type", "open_interest"]],
        on=["expiration", "strike", "option_type"],
        how="left",
    )
    fallback = merged["open_interest"].fillna(0)
    df.loc[needs_fill, "oi"] = fallback[needs_fill].values
    return df
