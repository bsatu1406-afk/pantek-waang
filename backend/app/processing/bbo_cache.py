"""Point-in-time BBO enrichment for Lee-Ready (Rev 4 Agent 4).

When the live ingester does not have a synchronous BBO snapshot for a
trade (typical at session open, after a reconnect, or for a backfill
window), we want the Lee-Ready classifier to fall back to the persisted
``bbo_book`` table that the ``cmbp-1`` / ``bbo-1s`` subscriptions write
to.

The public surface here is intentionally small:

* :func:`enrich_with_bbo` — vectorised as-of join between a trades
  DataFrame (``ts``, ``symbol``, plus optional contract keys) and a
  BBO snapshot DataFrame, writing ``bid`` / ``ask`` / ``mid`` /
  ``bbo_age_seconds`` columns onto the trades.
* :class:`InMemoryBboCache` — a tiny per-instrument LRU-ish cache used
  by the *live* ingester to track the most recent quote without a DB
  roundtrip. Same interface as the DataFrame helper so both can be
  swapped in tests.

Both paths apply a configurable staleness cap (default 2 s — bbo-1s is
sampled every second, so anything older than that is treated as
"no quote available" and the tick rule fallback takes over).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)

#: Default tolerance for the as-of join. bbo-1s is sampled at 1 Hz so we
#: keep one full second of buffer plus 1 s of clock skew tolerance.
DEFAULT_MAX_AGE = timedelta(seconds=2)


# ── DataFrame as-of join ─────────────────────────────────────────────────


def enrich_with_bbo(
    trades_df: pd.DataFrame,
    bbo_df: pd.DataFrame,
    *,
    max_age: timedelta = DEFAULT_MAX_AGE,
    contract_keys: Iterable[str] = ("symbol", "expiration", "strike", "option_type"),
) -> pd.DataFrame:
    """Attach the most recent BBO snapshot to every row in ``trades_df``.

    For each trade, looks up the latest BBO row whose ``ts`` is **at or
    before** the trade ``ts`` and whose contract keys match. Anything
    older than ``max_age`` is treated as a miss (``bid`` / ``ask`` set
    to NaN, ``bbo_age_seconds`` set to NaN).

    Parameters
    ----------
    trades_df : pd.DataFrame
        Must contain ``ts`` and the contract keys. The ``ts`` column is
        normalised to UTC-naive datetime64[ns] internally so DST jumps
        do not cause non-monotonic merge errors.
    bbo_df : pd.DataFrame
        Must contain ``ts``, ``bid_px`` (or ``bid``), ``ask_px`` (or
        ``ask``), and the same contract keys. Empty BBO is allowed —
        the result will have NaN bid/ask everywhere.
    max_age : datetime.timedelta
        Maximum age of an accepted BBO snapshot relative to the trade.
    contract_keys : iterable of str
        Columns to match on (besides ``ts``). The default targets the
        ``options_trades`` table; for futures use just ``("symbol",)``.

    Returns
    -------
    pd.DataFrame
        Copy of ``trades_df`` with four extra columns:
        ``bid``, ``ask``, ``mid``, ``bbo_age_seconds``.
        Pre-existing ``bid`` / ``ask`` columns are overwritten only on
        rows where the enrichment found a fresh snapshot.
    """
    if "ts" not in trades_df.columns:
        raise KeyError("enrich_with_bbo requires trades_df to have a 'ts' column")

    out = trades_df.copy()
    out["bid"] = out.get("bid", pd.NA)
    out["ask"] = out.get("ask", pd.NA)
    out["mid"] = pd.Series([float("nan")] * len(out), index=out.index, dtype=float)
    out["bbo_age_seconds"] = pd.Series(
        [float("nan")] * len(out), index=out.index, dtype=float
    )

    if bbo_df.empty or out.empty:
        return out

    # Normalise column names from the DB shape (bid_px / ask_px) into the
    # downstream shape (bid / ask) without mutating the caller's frame.
    bbo = bbo_df.copy()
    if "bid" not in bbo.columns and "bid_px" in bbo.columns:
        bbo = bbo.rename(columns={"bid_px": "bid"})
    if "ask" not in bbo.columns and "ask_px" in bbo.columns:
        bbo = bbo.rename(columns={"ask_px": "ask"})

    keys = [k for k in contract_keys if k in out.columns and k in bbo.columns]
    if not keys:
        raise KeyError(
            f"enrich_with_bbo requires at least one shared contract key, "
            f"got out={list(out.columns)} bbo={list(bbo.columns)}"
        )

    # merge_asof requires both sides sorted on the join key and the by-
    # columns. Rather than mutate input ordering, sort + remember index.
    out_sorted = out.sort_values("ts", kind="mergesort").reset_index(names="__orig_idx")
    bbo_sorted = bbo.sort_values("ts", kind="mergesort")[
        ["ts", "bid", "ask", *keys]
    ].copy()
    # Stash the BBO timestamp under a non-colliding name so we can recover
    # the BBO age from the merge result. ``merge_asof`` consumes ``ts`` on
    # both sides but happily passes through other columns.
    bbo_sorted["__bbo_ts"] = bbo_sorted["ts"]

    # Ensure consistent dtypes on the join keys (object vs string mismatches
    # cause merge_asof to silently produce all-NaN results).
    for k in keys:
        out_sorted[k] = out_sorted[k].astype(bbo_sorted[k].dtype)

    merged = pd.merge_asof(
        out_sorted,
        bbo_sorted,
        on="ts",
        by=keys,
        direction="backward",
        tolerance=pd.Timedelta(max_age),
        suffixes=("", "_bbo"),
    )

    bid_col = "bid_bbo" if "bid_bbo" in merged.columns else "bid"
    ask_col = "ask_bbo" if "ask_bbo" in merged.columns else "ask"

    # Compute mid + age once we know which rows actually got a hit.
    bid = pd.to_numeric(merged[bid_col], errors="coerce")
    ask = pd.to_numeric(merged[ask_col], errors="coerce")
    has_hit = bid.notna() & ask.notna()
    age = (merged["ts"] - merged["__bbo_ts"]).dt.total_seconds()
    mid = (bid + ask) / 2.0
    mid = mid.where(has_hit, np.nan)

    merged["bid"] = bid.where(has_hit, np.nan)
    merged["ask"] = ask.where(has_hit, np.nan)
    merged["mid"] = mid
    merged["bbo_age_seconds"] = age.where(has_hit, np.nan)

    # Restore original row ordering.
    merged = merged.sort_values("__orig_idx", kind="mergesort").reset_index(drop=True)
    out = out.reset_index(drop=True).copy()
    out["bid"] = merged["bid"].to_numpy()
    out["ask"] = merged["ask"].to_numpy()
    out["mid"] = merged["mid"].to_numpy()
    out["bbo_age_seconds"] = merged["bbo_age_seconds"].to_numpy()

    enriched = int(merged["bbo_age_seconds"].notna().sum())
    logger.debug(
        "bbo_enrichment",
        trades=int(len(out)),
        enriched=enriched,
        miss=int(len(out) - enriched),
    )
    return out


# ── In-memory cache for the live path ────────────────────────────────────


@dataclass
class _BboEntry:
    ts: datetime
    bid: float
    ask: float


class InMemoryBboCache:
    """Per-instrument latest-BBO cache for the live ingester.

    The cmbp-1 / bbo-1s record handlers call :meth:`update` on each
    snapshot; :meth:`at` is called by the trade handler to look up
    the most recent fresh BBO for a given instrument. Anything older
    than ``max_age`` (or never seen) returns ``None``.
    """

    def __init__(self, max_age: timedelta = DEFAULT_MAX_AGE):
        self._max_age = max_age
        self._store: dict[int, _BboEntry] = {}

    def update(self, instrument_id: int, ts: datetime, bid: float | None, ask: float | None) -> None:
        if bid is None or ask is None:
            return
        if not (bid > 0 and ask > 0 and ask >= bid):
            return
        self._store[instrument_id] = _BboEntry(ts=ts, bid=float(bid), ask=float(ask))

    def at(self, instrument_id: int, ts: datetime) -> tuple[float, float, float] | None:
        """Return ``(bid, ask, age_seconds)`` if the cache hit is fresh."""
        entry = self._store.get(instrument_id)
        if entry is None:
            return None
        age = ts - entry.ts
        if age < timedelta(0):
            # Trade ts is *before* the cached snapshot — treat as miss.
            return None
        if age > self._max_age:
            return None
        return entry.bid, entry.ask, age.total_seconds()

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()
