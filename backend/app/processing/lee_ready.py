"""Lee-Ready trade-direction classifier.

The canonical algorithm from Lee & Ready (1991), *Inferring Trade
Direction from Intraday Data*, J. of Finance 46(2). For each trade,
classify whether it was buyer-initiated (+1) or seller-initiated (-1):

1.  **Quote-rule**: if the trade price is **above** the prevailing midpoint,
    classify as a buy (+1); if below, classify as a sell (-1).
2.  **Tick-rule** (only used when the trade lands exactly on the
    midpoint, the spread is zero, or quotes are missing): use the sign
    of the change relative to the previous *different* trade price.
    ``+1`` if the trade is higher than the last different trade price,
    ``-1`` if lower. If still tied (e.g. session open with no history,
    or a non-finite price) the trade is left unclassified (``side=0``).

Inputs (DataFrame columns expected, all required):

* ``ts`` — trade timestamp (any monotone column); rows must be sorted on
  this column before calling, OR ``sort=True`` (default) lets the function
  sort defensively.
* ``price`` — trade price (float).
* ``bid`` / ``ask`` — prevailing best quotes at trade time (float). Missing
  values (NaN / None) are tolerated and trigger the tick-rule fallback.

Returns a copy of the input DataFrame with three new columns added:

* ``mid``       — midpoint at trade time (NaN if either quote was missing).
* ``side``      — +1 (buy), -1 (sell), 0 (unclassified).
* ``signed_qty``— ``side`` times ``size`` if ``size`` is present in the
                   input, otherwise the bare ``side``.

After classification, a structured ``lee_ready_classified`` log event is
emitted with counts by classification method (quote vs tick vs unclassified)
to make diagnostics easy from production telemetry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)


def classify_lee_ready(
    df: pd.DataFrame,
    *,
    sort: bool = True,
) -> pd.DataFrame:
    """Classify trades by initiator side using the Lee-Ready algorithm.

    See module docstring for the rule precedence and edge-case behavior.
    """
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

    # ── Quote rule ───────────────────────────────────────────────────────
    # ``mid`` is NaN whenever either quote is missing — those rows fall
    # straight through to the tick rule because ``price > mid`` is False
    # for any NaN comparand. We also explicitly route zero-spread quotes
    # (bid == ask) through the tick rule: with the spread collapsed, the
    # quote rule degenerates to a strict price ≠ mid test which is too
    # noisy to be useful.
    mid = (bid + ask) / 2.0
    spread = ask - bid
    eps = 1e-9
    quote_eligible = np.isfinite(mid) & np.isfinite(spread) & (spread > eps)

    side = np.zeros_like(price, dtype=np.int8)
    side[quote_eligible & (price > mid + eps)] = 1
    side[quote_eligible & (price < mid - eps)] = -1
    quote_classified_mask = side != 0

    # ── Tick rule fallback ───────────────────────────────────────────────
    # Walk through unclassified trades and look back to the most recent
    # **different** trade price. Equal-to-previous (zero-tick) trades
    # remain unclassified.
    tick_classified_mask = np.zeros_like(price, dtype=bool)
    if (side == 0).any():
        last_diff_price = np.nan
        for i in range(price.size):
            if side[i] != 0:
                if np.isfinite(price[i]):
                    last_diff_price = price[i]
                continue
            if not np.isfinite(price[i]):
                # cannot apply tick rule to a non-finite price
                continue
            if np.isfinite(last_diff_price):
                if price[i] > last_diff_price:
                    side[i] = 1
                    tick_classified_mask[i] = True
                elif price[i] < last_diff_price:
                    side[i] = -1
                    tick_classified_mask[i] = True
                # equal → leave as 0 (zero-tick); do not update history.
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

    # ── Structured diagnostic log ────────────────────────────────────────
    total = int(price.size)
    quote_n = int(quote_classified_mask.sum())
    tick_n = int(tick_classified_mask.sum())
    unclassified_n = total - quote_n - tick_n
    logger.debug(
        "lee_ready_classified",
        total=total,
        quote_rule=quote_n,
        tick_rule=tick_n,
        unclassified=unclassified_n,
    )
    return work
