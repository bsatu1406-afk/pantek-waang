"""Gamma Exposure (GEX) calculations.

Following Squeeze Metrics / SpotGamma methodology:

    GEX_per_strike = gamma * <weight> * 100 * underlying_price^2 * 0.01

where ``<weight>`` is **open interest** for the classical (resting) GEX, or
**volume** for the intraday flow-weighted GEX. Both expose the same shape
(curve / net_total / top_positive / top_negative) so the front-end and
indicator can render them identically.

Calls contribute positive GEX (dealer hedging convention), puts negative.
Net GEX = call GEX − |put GEX|.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.processing.zero_gamma import compute_zero_gamma

CONTRACT_MULTIPLIER = 100  # standard equity/index option contract size
ONE_PERCENT = 0.01


@dataclass
class GexSummary:
    underlying_price: float | None
    net_total: float
    curve: list[dict]
    top_positive: list[dict]
    top_negative: list[dict]
    zero_gamma: float | None = None
    weight_col: str = "oi"


def _empty(weight_col: str) -> GexSummary:
    return GexSummary(
        underlying_price=None,
        net_total=0.0,
        curve=[],
        top_positive=[],
        top_negative=[],
        zero_gamma=None,
        weight_col=weight_col,
    )


def _gex_per_row(row: pd.Series, S: float, weight_col: str) -> float:
    gamma = row.get("gamma")
    weight = row.get(weight_col)
    if gamma is None or weight is None or pd.isna(gamma) or pd.isna(weight):
        return 0.0
    sign = 1.0 if str(row.get("option_type", "")).upper() == "C" else -1.0
    return float(sign * gamma * weight * CONTRACT_MULTIPLIER * (S**2) * ONE_PERCENT)


def compute_gex(
    df: pd.DataFrame,
    *,
    top_n: int = 5,
    weight_col: str = "oi",
    risk_free_rate: float = 0.05,
) -> GexSummary:
    """Compute GEX curve, net total, top +/- levels, and zero-gamma level.

    Expects a DataFrame with: ``strike``, ``option_type``, ``gamma``,
    ``underlying_price`` and the requested ``weight_col`` (``oi`` or ``volume``).
    For zero-gamma the ``iv`` and ``expiration`` columns are also required;
    if they are missing/null on every row the level is omitted (None).

    If the weight column is missing or fully null, returns an empty summary.
    """
    if df.empty or weight_col not in df.columns:
        return _empty(weight_col)

    spot_series = df["underlying_price"].dropna()
    if spot_series.empty:
        return _empty(weight_col)
    S = float(spot_series.iloc[-1])

    # Skip computation entirely if the requested weight is entirely null/zero.
    weight_series = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)
    if weight_series.abs().sum() == 0:
        return GexSummary(
            underlying_price=S,
            net_total=0.0,
            curve=[],
            top_positive=[],
            top_negative=[],
            zero_gamma=None,
            weight_col=weight_col,
        )

    df = df.copy()
    df[weight_col] = weight_series
    df["gex"] = df.apply(lambda r: _gex_per_row(r, S, weight_col), axis=1)
    df["option_type_u"] = df["option_type"].astype(str).str.upper()

    call_sum = (
        df.loc[df["option_type_u"] == "C"]
        .groupby("strike", as_index=False)["gex"]
        .sum()
        .rename(columns={"gex": "call_gex"})
    )
    put_sum = (
        df.loc[df["option_type_u"] == "P"]
        .groupby("strike", as_index=False)["gex"]
        .sum()
        .rename(columns={"gex": "put_gex"})
    )
    curve_df = (
        pd.merge(call_sum, put_sum, on="strike", how="outer")
        .fillna({"call_gex": 0.0, "put_gex": 0.0})
    )
    curve_df["strike"] = curve_df["strike"].astype(float)
    curve_df["net_gex"] = curve_df["call_gex"] - curve_df["put_gex"].abs()
    curve_df = curve_df.sort_values("strike").reset_index(drop=True)
    curve_df = curve_df.replace({np.nan: 0.0})

    top_pos = (
        curve_df.sort_values("net_gex", ascending=False).head(top_n).to_dict(orient="records")
    )
    top_neg = (
        curve_df.sort_values("net_gex", ascending=True).head(top_n).to_dict(orient="records")
    )

    zg = compute_zero_gamma(
        df, weight_col=weight_col, risk_free_rate=risk_free_rate
    )

    return GexSummary(
        underlying_price=S,
        net_total=float(curve_df["net_gex"].sum()),
        curve=curve_df.to_dict(orient="records"),
        top_positive=top_pos,
        top_negative=top_neg,
        zero_gamma=zg,
        weight_col=weight_col,
    )
