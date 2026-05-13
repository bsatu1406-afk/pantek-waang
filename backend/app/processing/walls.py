"""Call / Put wall detection by Open Interest and Volume.

A "wall" is the strike that the dealer book has the largest gross exposure
at, either by resting Open Interest (``by_oi``) or by today's traded volume
(``by_volume``). The top-N strikes are returned per option type with
strictly positive aggregated weight.

Sanity guarantees applied here:

* Non-finite OI/volume (NaN/inf) inputs are coerced to zero before
  aggregation, so they cannot rank above legitimate strikes.
* Strikes that are missing or non-finite are dropped entirely.
* When every strike has zero (or non-finite) weight, the wall list is
  empty rather than a meaningless ordering of zeros.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WallsSummary:
    by_oi: dict
    by_volume: dict


def _top_strikes(
    df: pd.DataFrame, *, value_col: str, option_type: str, top_n: int = 3
) -> list[dict]:
    sub = df[df["option_type"].astype(str).str.upper() == option_type].copy()
    if sub.empty:
        return []

    sub["strike"] = pd.to_numeric(sub.get("strike"), errors="coerce")
    sub = sub[np.isfinite(sub["strike"])].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce").fillna(0.0)
    sub.loc[~np.isfinite(sub[value_col]), value_col] = 0.0
    if sub.empty:
        return []

    grouped = (
        sub.groupby("strike", as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
    )
    # Drop zero-value strikes — when the underlying weight column is all zero
    # the order is arbitrary and call/put walls degenerate to the same strikes,
    # which is misleading. Better to return an empty list and let the caller
    # render "no walls available yet".
    grouped = grouped[grouped[value_col] > 0].head(top_n)
    return [
        {"strike": float(r["strike"]), "value": float(r[value_col] or 0)}
        for _, r in grouped.iterrows()
    ]


def compute_walls(df: pd.DataFrame, *, top_n: int = 3) -> WallsSummary:
    if df.empty or "option_type" not in df.columns:
        return WallsSummary(by_oi={}, by_volume={})

    df = df.copy()
    for col in ("oi", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            df.loc[~np.isfinite(df[col]), col] = 0.0
        else:
            df[col] = 0.0

    return WallsSummary(
        by_oi={
            "call_wall": _top_strikes(df, value_col="oi", option_type="C", top_n=top_n),
            "put_wall": _top_strikes(df, value_col="oi", option_type="P", top_n=top_n),
        },
        by_volume={
            "call_wall": _top_strikes(df, value_col="volume", option_type="C", top_n=top_n),
            "put_wall": _top_strikes(df, value_col="volume", option_type="P", top_n=top_n),
        },
    )
