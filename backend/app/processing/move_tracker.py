"""Realized vs Implied move tracker.

At session open we extract the *implied* daily move from the front-month
ATM straddle:

    implied_move_$ ≈ ATM_call_price + ATM_put_price        (front expiry)

This is the market's price of one standard deviation of underlying move
through expiration, scaled to a single trading day if the front expiry
is not 0DTE (using ``√(1/dte)`` rescaling).

Throughout the session, *realized* move = ``|last_price − open_price|``.

Output: a single record with both numbers and a ``ratio`` field. The
website surfaces this as a "vol crush" (ratio < 0.5) or "vol expansion"
(ratio > 1.5) signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MoveSnapshot:
    underlying_price: float | None
    open_price: float | None
    realized_move: float | None
    implied_move: float | None
    implied_dte: int | None
    ratio: float | None


def compute_move_tracker(
    chain: pd.DataFrame,
    *,
    open_price: float | None,
    today: pd.Timestamp | None = None,
) -> MoveSnapshot:
    """Compute realized vs implied move for the front expiry.

    ``chain`` is the latest options chain DataFrame with columns
    ``strike, expiration, option_type, last_price, underlying_price``.

    ``open_price`` is the underlying's session-opening price (caller's
    responsibility — the chain DataFrame doesn't carry it).
    """
    if chain.empty:
        return MoveSnapshot(None, open_price, None, None, None, None)

    spot_series = chain["underlying_price"].dropna()
    if spot_series.empty:
        return MoveSnapshot(None, open_price, None, None, None, None)
    S = float(spot_series.iloc[-1])

    realized = None
    if open_price is not None and open_price > 0 and np.isfinite(S):
        realized = abs(S - open_price)

    if today is None:
        today = pd.Timestamp.utcnow()
        if today.tzinfo is not None:
            today = today.tz_convert(None)
    today_d = today.date() if hasattr(today, "date") else today

    work = chain.copy()
    work["last_price"] = pd.to_numeric(work["last_price"], errors="coerce")
    work["strike"] = pd.to_numeric(work["strike"], errors="coerce")
    work = work[work["last_price"].notna() & work["strike"].notna()]
    if work.empty:
        return MoveSnapshot(S, open_price, realized, None, None, None)

    # Find front expiry (smallest dte > 0).
    front_exp = None
    front_dte = None
    for exp in sorted(work["expiration"].unique()):
        try:
            exp_d = pd.Timestamp(exp).date()
        except (TypeError, ValueError):
            continue
        dte = (exp_d - today_d).days
        if dte >= 0:
            front_exp = exp
            front_dte = dte
            break
    if front_exp is None:
        return MoveSnapshot(S, open_price, realized, None, None, None)

    front = work[work["expiration"] == front_exp]
    if front.empty:
        return MoveSnapshot(S, open_price, realized, None, None, None)

    nearest = front.iloc[(front["strike"] - S).abs().argsort()]
    if nearest.empty:
        return MoveSnapshot(S, open_price, realized, None, None, None)
    atm_strike = float(nearest.iloc[0]["strike"])

    atm_call = front[
        (front["strike"] == atm_strike)
        & (front["option_type"].astype(str).str.upper() == "C")
    ]
    atm_put = front[
        (front["strike"] == atm_strike)
        & (front["option_type"].astype(str).str.upper() == "P")
    ]
    if atm_call.empty or atm_put.empty:
        return MoveSnapshot(S, open_price, realized, None, front_dte, None)

    call_p = float(atm_call.iloc[0]["last_price"])
    put_p = float(atm_put.iloc[0]["last_price"])
    if not (np.isfinite(call_p) and np.isfinite(put_p)):
        return MoveSnapshot(S, open_price, realized, None, front_dte, None)

    implied_total = call_p + put_p
    # Rescale to a single-day move when the front expiry is multi-day.
    daily_implied = implied_total
    if front_dte and front_dte > 1:
        daily_implied = implied_total / np.sqrt(front_dte)

    ratio = None
    if realized is not None and daily_implied > 0:
        ratio = float(realized / daily_implied)

    return MoveSnapshot(
        underlying_price=S,
        open_price=open_price,
        realized_move=realized,
        implied_move=float(daily_implied),
        implied_dte=int(front_dte) if front_dte is not None else None,
        ratio=ratio,
    )
