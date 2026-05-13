"""Call / Put wall detection by Open Interest and Volume."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class WallsSummary:
    by_oi: dict
    by_volume: dict


def _top_strikes(
    df: pd.DataFrame, *, value_col: str, option_type: str, top_n: int = 3
) -> list[dict]:
    sub = df[df["option_type"].str.upper() == option_type].copy()
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
    if df.empty:
        return WallsSummary(by_oi={}, by_volume={})

    df = df.copy()
    df["oi"] = df["oi"].fillna(0)
    df["volume"] = df["volume"].fillna(0)

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
