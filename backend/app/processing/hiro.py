"""HIRO — Hedging Impact Reaction-Oriented signed-premium tape.

Concept
-------
* Every option trade is classified as buyer-initiated or seller-initiated
  by :func:`app.processing.lee_ready.classify_lee_ready`.
* Each classified trade is converted into a *hedge-flow signed premium*::

      premium_$ = customer_side · size · price · 100 · option_sign

  where ``option_sign`` is ``+1`` for calls and ``-1`` for puts.

  The sign captures what the dealer must do to hedge in the underlying:

  * Customer **buys** a **call** → dealer is short the call (negative
    delta) → dealer hedges by **buying** the underlying → **positive**
    hedge flow.
  * Customer **buys** a **put** → dealer is short the put (positive
    delta) → dealer hedges by **selling** the underlying → **negative**
    hedge flow.
  * The mirror holds for customer sells. The net effect is that customer
    bullish flow (long calls / short puts) yields **positive** HIRO and
    customer bearish flow (long puts / short calls) yields **negative**
    HIRO — directly interpretable as cumulative dealer-hedging buy
    pressure on the underlying.

Bucketing
---------
The output is broken into time buckets at the resample frequency
specified by ``bucket`` (default ``1min``). **The ``cumulative`` field on
each bucket row resets at the start of every bucket** — it represents
the running signed premium *within* the bucket, not a session-wide
cumsum. Consumers that want a session-wide running total can ``cumsum``
the per-bucket ``net_premium`` themselves; storing per-bucket cumulative
keeps each bucket independently meaningful and avoids carry-over from
stale data when the window slides forward.

``HiroSeries.cumulative`` reports the most-recent bucket's signed
premium (i.e. the last bucket's ``net_premium``).
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
    ``net_premium``, ``cumulative`` (resets at the start of each bucket)."""

    cumulative: float = 0.0
    """Signed premium of the most-recent bucket (per-bucket reset)."""


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

    See the module docstring for the sign convention.
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
    size = pd.to_numeric(work["size"], errors="coerce").fillna(0)
    price = pd.to_numeric(work["price"], errors="coerce").fillna(0)
    is_call = work["option_type"].astype(str).str.upper() == "C"

    base_premium = customer_side * size * price * CONTRACT_MULTIPLIER
    # Calls: +customer_side (customer-buy-call → +). Puts: -customer_side.
    call_prem = np.where(is_call, base_premium, 0.0)
    put_prem = np.where(~is_call, -base_premium, 0.0)

    work = work.assign(_call=call_prem, _put=put_prem)
    work = work.set_index("ts")

    grouped = work.resample(bucket).agg({"_call": "sum", "_put": "sum"})
    grouped["net"] = grouped["_call"] + grouped["_put"]
    # ``cumulative`` resets at the start of each bucket. Since the
    # resample collapses all intra-bucket trades into a single row, the
    # bucket's running cumulative simply equals its net signed premium.
    grouped["cumulative"] = grouped["net"]

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
