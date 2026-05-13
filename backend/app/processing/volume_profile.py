"""ES futures volume profile (per session footprint).

Aggregates Globex MDP 3.0 futures trades into a price-binned histogram::

    POC   — Price of Control: the bin with the highest traded volume.
    VAH   — Value Area High: top of the contiguous price range that
             contains 70% of total volume above POC.
    VAL   — Value Area Low: matching lower bound.
    bins  — [{price, total, buy, sell}] one entry per bin.

Tick-size and bin-size default to 0.25 (ES tick), but callers can pass
``bin_size`` for coarser/finer aggregation. We accept ``side`` (Lee-Ready
or Aggressor flag from the feed) to populate buy / sell volumes; if
absent everything is dumped into ``total`` only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VolumeProfile:
    bin_size: float
    poc: float | None
    vah: float | None
    val: float | None
    total_volume: int
    bins: list[dict]


def compute_volume_profile(
    trades: pd.DataFrame,
    *,
    bin_size: float = 0.25,
    value_area_pct: float = 0.70,
) -> VolumeProfile:
    """Build a price-binned volume profile from a flat trade tape.

    Required columns: ``price``, ``size``. Optional columns:
    ``side`` (``+1`` buy, ``-1`` sell), ``ts``.
    """
    if trades.empty:
        return VolumeProfile(bin_size=bin_size, poc=None, vah=None, val=None,
                             total_volume=0, bins=[])

    work = trades.copy()
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["size"] = pd.to_numeric(work["size"], errors="coerce").fillna(0).astype(int)
    work = work[work["price"].notna() & (work["size"] > 0)]
    if work.empty:
        return VolumeProfile(bin_size=bin_size, poc=None, vah=None, val=None,
                             total_volume=0, bins=[])

    # Quantise price into bins.
    bin_centres = np.round(work["price"] / bin_size) * bin_size
    work["_bin"] = bin_centres

    if "side" in work.columns:
        side = pd.to_numeric(work["side"], errors="coerce").fillna(0).astype(int)
        work["_buy"] = np.where(side > 0, work["size"], 0)
        work["_sell"] = np.where(side < 0, work["size"], 0)
    else:
        work["_buy"] = 0
        work["_sell"] = 0

    grouped = (
        work.groupby("_bin", as_index=False)
        .agg(total=("size", "sum"), buy=("_buy", "sum"), sell=("_sell", "sum"))
        .rename(columns={"_bin": "price"})
        .sort_values("price")
        .reset_index(drop=True)
    )

    if grouped.empty or grouped["total"].sum() == 0:
        return VolumeProfile(bin_size=bin_size, poc=None, vah=None, val=None,
                             total_volume=0, bins=[])

    total_vol = int(grouped["total"].sum())
    poc_idx = int(grouped["total"].idxmax())
    poc_price = float(grouped.iloc[poc_idx]["price"])

    # Value area: expand symmetrically around POC until we cover ``value_area_pct``.
    target = value_area_pct * total_vol
    accumulated = int(grouped.iloc[poc_idx]["total"])
    lo, hi = poc_idx, poc_idx
    while accumulated < target and (lo > 0 or hi < len(grouped) - 1):
        next_lo_total = (
            int(grouped.iloc[lo - 1]["total"]) if lo > 0 else -1
        )
        next_hi_total = (
            int(grouped.iloc[hi + 1]["total"]) if hi < len(grouped) - 1 else -1
        )
        if next_hi_total >= next_lo_total and next_hi_total >= 0:
            hi += 1
            accumulated += next_hi_total
        elif next_lo_total >= 0:
            lo -= 1
            accumulated += next_lo_total
        else:
            break
    vah = float(grouped.iloc[hi]["price"])
    val = float(grouped.iloc[lo]["price"])

    bins = [
        {
            "price": float(row["price"]),
            "total": int(row["total"]),
            "buy": int(row["buy"]),
            "sell": int(row["sell"]),
        }
        for _, row in grouped.iterrows()
    ]
    return VolumeProfile(
        bin_size=bin_size,
        poc=poc_price,
        vah=vah,
        val=val,
        total_volume=total_vol,
        bins=bins,
    )
