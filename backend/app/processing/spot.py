"""Synthetic underlying spot via put-call parity.

OPRA Pillar **does not publish the underlying index spot** for SPX / NDX —
those are computed indices, not exchange-traded contracts. To keep GEX /
IV / regime metrics working without an extra data subscription, we
recover the spot from the option chain itself using put-call parity:

    C - P = S - K * exp(-r * T)
    => S = K * exp(-r * T) + (C - P)

We pick the strike closest to the per-expiry forward midpoint of all
available strikes, which is approximately ATM. A noisy/stale quote is
filtered out by requiring positive bid + ask on **both** legs.

The synthetic spot is averaged across the nearest few expiries to dampen
single-expiry quote noise. If the chain has fewer than two valid
call/put pairs, ``synthesize_underlying_price`` returns ``None``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _mid(row: pd.Series) -> float | None:
    """Best available reference price for one option leg.

    Prefer the bid/ask midpoint when both quotes exist. When the upstream
    feed only delivers trade prints (e.g. the Databento OPRA Pillar
    Standard plan does not include cmbp-1 NBBO updates), fall back to the
    last-trade price so put-call parity still has something to work with.
    """
    bid = row.get("bid")
    ask = row.get("ask")
    if (
        bid is not None
        and ask is not None
        and not pd.isna(bid)
        and not pd.isna(ask)
        and bid > 0
        and ask > 0
    ):
        return float((bid + ask) / 2.0)
    last = row.get("last_price")
    if last is not None and not pd.isna(last) and last > 0:
        return float(last)
    return None


def _years_to_expiry(today: pd.Timestamp, expiry) -> float:
    today_d = today.date() if hasattr(today, "date") else today
    expiry_d = expiry.date() if hasattr(expiry, "date") else pd.Timestamp(expiry).date()
    days = max(1, (expiry_d - today_d).days)
    return days / 365.0


def synthesize_underlying_price(
    df: pd.DataFrame,
    *,
    risk_free_rate: float = 0.05,
    today: pd.Timestamp | None = None,
    max_expiries: int = 3,
) -> float | None:
    """Recover spot via put-call parity from the freshest near-the-money pair.

    Returns ``None`` when no usable call/put pair exists. Values that fall
    outside a sanity range (``< 1`` or ``> 1e6``) are also filtered out as
    likely artefacts of stale / one-sided quotes.
    """
    if df.empty:
        return None
    needed = {"strike", "expiration", "option_type"}
    if not needed.issubset(df.columns):
        return None
    # Either {bid, ask} or last_price must be present; otherwise we cannot
    # build a leg price for parity.
    if not (
        ({"bid", "ask"}.issubset(df.columns)) or ("last_price" in df.columns)
    ):
        return None

    if today is None:
        today = pd.Timestamp.utcnow()
        if today.tzinfo is not None:
            today = today.tz_convert(None)

    work = df.copy()
    work["mid"] = work.apply(_mid, axis=1)
    work = work.dropna(subset=["mid"])
    if work.empty:
        return None
    work["option_type_u"] = work["option_type"].astype(str).str.upper()

    expiries = sorted(pd.to_datetime(work["expiration"].unique()))[:max_expiries]
    candidates: list[float] = []

    for expiry in expiries:
        T = _years_to_expiry(today, expiry)
        sub = work[pd.to_datetime(work["expiration"]) == expiry]
        calls = sub[sub["option_type_u"] == "C"][["strike", "mid"]]
        puts = sub[sub["option_type_u"] == "P"][["strike", "mid"]]
        if calls.empty or puts.empty:
            continue
        merged = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
        if merged.empty:
            continue
        # ATM ≈ strike where call mid is closest to put mid (so C - P ≈ 0
        # ⇒ S ≈ K * e^{-rT}). This is the safest pivot when we have no
        # spot estimate yet.
        merged = merged.assign(diff=lambda d: (d["mid_c"] - d["mid_p"]).abs())
        atm = merged.sort_values("diff").iloc[0]
        K = float(atm["strike"])
        spot = K * math.exp(-risk_free_rate * T) + float(atm["mid_c"]) - float(atm["mid_p"])
        if 1.0 < spot < 1e6 and math.isfinite(spot):
            candidates.append(spot)

    if not candidates:
        return None
    return float(np.median(candidates))
