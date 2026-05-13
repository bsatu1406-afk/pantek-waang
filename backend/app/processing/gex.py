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
    """Per-row dollar GEX with strict NaN/inf coercion.

    Non-finite gamma/weight inputs collapse to 0 so they cannot leak into the
    per-strike aggregate.
    """
    gamma = row.get("gamma")
    weight = row.get(weight_col)
    try:
        gamma_f = float(gamma) if gamma is not None else float("nan")
        weight_f = float(weight) if weight is not None else float("nan")
    except (TypeError, ValueError):
        return 0.0
    if not (np.isfinite(gamma_f) and np.isfinite(weight_f)):
        return 0.0
    sign = 1.0 if str(row.get("option_type", "")).upper() == "C" else -1.0
    value = sign * gamma_f * weight_f * CONTRACT_MULTIPLIER * (S * S) * ONE_PERCENT
    if not np.isfinite(value):
        return 0.0
    return float(value)


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

    spot_series = pd.to_numeric(df["underlying_price"], errors="coerce").dropna()
    spot_series = spot_series[np.isfinite(spot_series)]
    if spot_series.empty:
        return _empty(weight_col)
    S = float(spot_series.iloc[-1])
    if not np.isfinite(S) or S <= 0:
        return _empty(weight_col)

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
    # Defensive: pandas can produce NaN if a row missed the weight column.
    df["gex"] = pd.to_numeric(df["gex"], errors="coerce").fillna(0.0)
    df.loc[~np.isfinite(df["gex"]), "gex"] = 0.0

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
    curve_df["strike"] = pd.to_numeric(curve_df["strike"], errors="coerce")
    curve_df = curve_df[np.isfinite(curve_df["strike"])].copy()
    curve_df["strike"] = curve_df["strike"].astype(float)
    curve_df["net_gex"] = curve_df["call_gex"] - curve_df["put_gex"].abs()
    curve_df = curve_df.sort_values("strike").reset_index(drop=True)
    # Collapse any residual non-finite values to 0 so NaN/inf cannot reach the DB.
    for col in ("call_gex", "put_gex", "net_gex"):
        curve_df[col] = pd.to_numeric(curve_df[col], errors="coerce").fillna(0.0)
        curve_df.loc[~np.isfinite(curve_df[col]), col] = 0.0

    top_pos = (
        curve_df.sort_values("net_gex", ascending=False).head(top_n).to_dict(orient="records")
    )
    top_neg = (
        curve_df.sort_values("net_gex", ascending=True).head(top_n).to_dict(orient="records")
    )

    zg = compute_zero_gamma(
        df, weight_col=weight_col, risk_free_rate=risk_free_rate
    )

    net_total_raw = float(curve_df["net_gex"].sum())
    net_total = net_total_raw if np.isfinite(net_total_raw) else 0.0

    return GexSummary(
        underlying_price=S,
        net_total=net_total,
        curve=curve_df.to_dict(orient="records"),
        top_positive=top_pos,
        top_negative=top_neg,
        zero_gamma=zg,
        weight_col=weight_col,
    )
