"""HIRO — Hedging Impact Reaction-Oriented signed-premium tape.

Concept:
* Every option trade is classified as buyer-initiated or seller-initiated
  by ``classify_lee_ready``.
* Each classified trade is converted into a *dealer signed premium*::

      premium_$ = side · size · price · 100

  where ``side`` is the **dealer's** sign (opposite to the customer's):
  if the customer bought (side = +1), the dealer sold and is now short
  the option, so the dealer's premium delta is **−** size·price·100.

* HIRO is the running cumulative sum of dealer signed premium, broken
  out by call vs put because the hedge implications are different:
  dealer-short calls hedges by *buying* the underlying (positive flow);
  dealer-short puts hedges by *selling* the underlying (negative flow).

* Net HIRO ≈ cumulative dealer hedging force on the underlying::

      HIRO = HIRO_call_buy_pressure − HIRO_put_sell_pressure

This module is pure / vectorised and stateless. Caller supplies a
DataFrame of classified option trades; we return aggregated time-bucketed
signed-premium series suitable for storing in ``computed_metrics`` or
streaming to the website.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CONTRACT_MULTIPLIER = 100


@dataclass
class HiroSeries:
    """Result of a HIRO computation over a window."""

    bucket_size: str
    """pandas frequency alias used for resampling (e.g. ``"1min"``)."""

    series: list[dict] = field(default_factory=list)
    """One entry per time bucket: ``ts``, ``call_premium``, ``put_premium``,
    ``net_premium``, ``cumulative``."""

    cumulative: float = 0.0
    """Last value of the running net cumulative dealer signed premium."""


def compute_hiro(
    df: pd.DataFrame,
    *,
    bucket: str = "1min",
) -> HiroSeries:
    """Aggregate classified option trades into a HIRO time series.

    Required input columns:

    * ``ts``          — trade timestamp (datetime-like).
    * ``side``        — +1 (customer buy) / -1 (customer sell), integer.
    * ``size``        — contracts traded.
    * ``price``       — trade price (per contract).
    * ``option_type`` — 'C' or 'P'.

    The dealer-side sign is the negation of customer ``side``.
    """
    expected = {"ts", "side", "size", "price", "option_type"}
    if df.empty:
        return HiroSeries(bucket_size=bucket)
    missing = expected.difference(df.columns)
    if missing:
        raise KeyError(f"HIRO requires {expected}; missing {missing}")

    work = df.copy()
    work = work[pd.to_numeric(work["side"], errors="coerce").fillna(0) != 0]
    if work.empty:
        return HiroSeries(bucket_size=bucket)

    work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce")
    work = work.dropna(subset=["ts"])
    if work.empty:
        return HiroSeries(bucket_size=bucket)

    customer_side = pd.to_numeric(work["side"], errors="coerce").fillna(0).astype(int)
    dealer_side = -customer_side
    size = pd.to_numeric(work["size"], errors="coerce").fillna(0)
    price = pd.to_numeric(work["price"], errors="coerce").fillna(0)
    is_call = work["option_type"].astype(str).str.upper() == "C"

    premium = dealer_side * size * price * CONTRACT_MULTIPLIER
    call_prem = np.where(is_call, premium, 0.0)
    put_prem = np.where(~is_call, premium, 0.0)

    work = work.assign(_call=call_prem, _put=put_prem)
    work = work.set_index("ts")

    grouped = work.resample(bucket).agg({"_call": "sum", "_put": "sum"})
    grouped["net"] = grouped["_call"] + grouped["_put"]
    grouped["cumulative"] = grouped["net"].cumsum()

    series_payload = [
        {
            "ts": ts.isoformat(),
            "call_premium": float(row["_call"]),
            "put_premium": float(row["_put"]),
            "net_premium": float(row["net"]),
            "cumulative": float(row["cumulative"]),
        }
        for ts, row in grouped.iterrows()
        if not np.isnan(row["net"])
    ]
    last_cum = (
        float(grouped["cumulative"].iloc[-1])
        if not grouped.empty
        else 0.0
    )
    return HiroSeries(
        bucket_size=bucket,
        series=series_payload,
        cumulative=last_cum,
    )
