"""Lee-Ready trade-direction classifier.

The canonical algorithm from Lee & Ready (1991), *Inferring Trade
Direction from Intraday Data*, J. of Finance 46(2). For each trade,
classify whether it was buyer-initiated (+1) or seller-initiated (-1):

1.  **Quote-rule**: if the trade price is **above** the prevailing midpoint,
    classify as a buy (+1); if below, classify as a sell (-1).
2.  **Tick-rule** (only used when the trade lands exactly on the
    midpoint): use the sign of the change relative to the previous
    *different* trade price. ``+1`` if the trade is higher than the last
    different trade price, ``-1`` if lower. If still tied (e.g. session
    open with no history) the trade is left unclassified (0).

Inputs (DataFrame columns expected, all required):

* ``ts`` — trade timestamp (any monotone column); rows must be sorted on
  this column before calling, OR ``sort=True`` (default) lets the function
  sort defensively.
* ``price`` — trade price (float).
* ``bid`` / ``ask`` — prevailing best quotes at trade time (float).

Returns a copy of the input DataFrame with three new columns added:

* ``mid``       — midpoint at trade time.
* ``side``      — +1 (buy), -1 (sell), 0 (unclassified).
* ``signed_qty``— ``side`` times ``size`` if ``size`` is present in the
                   input, otherwise the bare ``side``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def classify_lee_ready(
    df: pd.DataFrame,
    *,
    sort: bool = True,
) -> pd.DataFrame:
    required = {"price", "bid", "ask"}
    if df.empty:
        out = df.copy()
        out["mid"] = pd.Series(dtype=float)
        out["side"] = pd.Series(dtype="int8")
        out["signed_qty"] = pd.Series(dtype=float)
        return out
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Lee-Ready classifier requires {required}; missing {missing}")

    work = df.copy()
    if sort and "ts" in work.columns:
        work = work.sort_values("ts").reset_index(drop=True)

    bid = pd.to_numeric(work["bid"], errors="coerce").to_numpy(dtype=float)
    ask = pd.to_numeric(work["ask"], errors="coerce").to_numpy(dtype=float)
    price = pd.to_numeric(work["price"], errors="coerce").to_numpy(dtype=float)

    mid = (bid + ask) / 2.0
    # Quote rule with a small epsilon so floating-point round-trips don't
    # mis-classify trades that landed exactly on the mid.
    eps = 1e-9
    side = np.zeros_like(price, dtype=np.int8)
    side[price > mid + eps] = 1
    side[price < mid - eps] = -1

    # Tick rule fallback. Walk through unclassified trades and look back
    # to the most recent **different** trade price.
    if (side == 0).any():
        last_diff_price = np.nan
        for i in range(price.size):
            if side[i] != 0:
                # quote-rule already classified — update last_diff_price
                if np.isfinite(price[i]):
                    last_diff_price = price[i]
                continue
            if not np.isfinite(price[i]):
                continue
            if np.isfinite(last_diff_price):
                if price[i] > last_diff_price:
                    side[i] = 1
                elif price[i] < last_diff_price:
                    side[i] = -1
                # Equal -> leave as 0 (zero-tick); update last_diff_price
                # only on a real move.
                else:
                    pass
            # Update last_diff_price only when the current trade has a
            # different price from the last seen.
            if np.isfinite(last_diff_price) and price[i] != last_diff_price:
                last_diff_price = price[i]
            elif not np.isfinite(last_diff_price):
                last_diff_price = price[i]

    work["mid"] = mid
    work["side"] = side
    if "size" in work.columns:
        size = pd.to_numeric(work["size"], errors="coerce").fillna(0).to_numpy()
        work["signed_qty"] = side.astype(float) * size
    else:
        work["signed_qty"] = side.astype(float)
    return work
