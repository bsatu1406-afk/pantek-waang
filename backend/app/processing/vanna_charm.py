"""Aggregate Vanna and Charm dealer-exposure surfaces.

These complement the GEX module: while GEX captures dealer hedging
sensitivity to *spot moves*, Vanna captures sensitivity to *implied
volatility moves* (a major flow driver around Fed days, CPI, and other
vol shocks), and Charm captures sensitivity to *time decay* (the dominant
0DTE pinning mechanic in the last 90 minutes of the session).

Sign convention (dealer's book is short customer flow):

* Calls contribute **+** vanna and **+** charm (dealer holds a short call).
* Puts contribute **−** vanna and **−** charm (dealer holds a short put).

This matches the SqueezeMetrics convention used by ``compute_gex``.

Output schema mirrors ``GexSummary`` so persistence / API code can stay
uniform: net total + per-strike curve + top-3 long/short levels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.processing import bsm

CONTRACT_MULTIPLIER = 100


@dataclass
class GreekSummary:
    underlying_price: float | None
    net_total: float
    curve: list[dict]
    top_positive: list[dict]
    top_negative: list[dict]
    weight_col: str = "oi"


def _empty(weight_col: str) -> GreekSummary:
    return GreekSummary(
        underlying_price=None,
        net_total=0.0,
        curve=[],
        top_positive=[],
        top_negative=[],
        weight_col=weight_col,
    )


def _prepare(
    df: pd.DataFrame,
    *,
    weight_col: str,
    today: pd.Timestamp | None,
) -> tuple[pd.DataFrame | None, float | None]:
    """Validate the chain DataFrame and return an array-friendly subset.

    Returns ``(work_df, S)`` or ``(None, None)`` if computation is not
    possible (missing columns, empty chain, all-zero weights, non-finite
    spot, …).
    """
    required = {"strike", "option_type", "iv", "expiration", "underlying_price", weight_col}
    if df.empty or not required.issubset(df.columns):
        return None, None

    spot_series = pd.to_numeric(df["underlying_price"], errors="coerce").dropna()
    spot_series = spot_series[np.isfinite(spot_series)]
    if spot_series.empty:
        return None, None
    S = float(spot_series.iloc[-1])
    if not np.isfinite(S) or S <= 0:
        return None, None

    if today is None:
        today = pd.Timestamp.utcnow()
        if today.tzinfo is not None:
            today = today.tz_convert(None)
    today_d = today.date() if hasattr(today, "date") else today

    work = df[["strike", "option_type", "iv", "expiration", weight_col]].copy()
    work["weight"] = pd.to_numeric(work[weight_col], errors="coerce").fillna(0.0)
    work.loc[~np.isfinite(work["weight"]), "weight"] = 0.0
    work["iv"] = pd.to_numeric(work["iv"], errors="coerce")
    work["strike"] = pd.to_numeric(work["strike"], errors="coerce")

    def _tau(exp) -> float:  # type: ignore[no-untyped-def]
        try:
            d = pd.Timestamp(exp).date()
            # 0DTE -> 1/365 (one day floor) so charm/vanna don't blow up.
            return max(1, (d - today_d).days) / 365.0
        except (TypeError, ValueError):
            return 0.0

    work["tau"] = work["expiration"].apply(_tau)
    work = work[
        (work["weight"].abs() > 0)
        & np.isfinite(work["weight"])
        & work["iv"].notna()
        & np.isfinite(work["iv"])
        & (work["iv"] > 0)
        & work["strike"].notna()
        & np.isfinite(work["strike"])
        & (work["strike"] > 0)
        & (work["tau"] > 0)
        & np.isfinite(work["tau"])
    ]
    if work.empty:
        return None, None

    return work, S


def _signed_aggregate(work: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Sum ``value_col`` per strike, with calls positive and puts negative.

    Non-finite per-row values (NaN/inf produced by extreme BSM inputs) are
    zero-filled so they cannot poison the per-strike aggregate. This matches
    the dealer-hedging sign convention shared with ``compute_gex``: long-call
    customer flow is positive and long-put customer flow is negative.
    """
    sign = np.where(
        work["option_type"].astype(str).str.upper() == "C", 1.0, -1.0
    )
    work = work.copy()
    raw = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)
    raw = raw.where(np.isfinite(raw), 0.0)
    work["_signed"] = sign * raw
    out = (
        work.groupby("strike", as_index=False)["_signed"]
        .sum()
        .rename(columns={"_signed": value_col})
        .sort_values("strike")
        .reset_index(drop=True)
    )
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce").fillna(0.0)
    out.loc[~np.isfinite(out[value_col]), value_col] = 0.0
    return out


def _summarise(curve_df: pd.DataFrame, value_col: str, *,
               S: float, weight_col: str, top_n: int) -> GreekSummary:
    if curve_df.empty:
        return _empty(weight_col)
    top_pos = (
        curve_df.sort_values(value_col, ascending=False).head(top_n)
        .to_dict(orient="records")
    )
    top_neg = (
        curve_df.sort_values(value_col, ascending=True).head(top_n)
        .to_dict(orient="records")
    )
    net_total = float(curve_df[value_col].sum())
    if not np.isfinite(net_total):
        net_total = 0.0
    return GreekSummary(
        underlying_price=S,
        net_total=net_total,
        curve=curve_df.to_dict(orient="records"),
        top_positive=top_pos,
        top_negative=top_neg,
        weight_col=weight_col,
    )


def compute_vanna(
    df: pd.DataFrame,
    *,
    weight_col: str = "oi",
    risk_free_rate: float = 0.05,
    today: pd.Timestamp | None = None,
    top_n: int = 5,
) -> GreekSummary:
    """Aggregate dealer **Vanna** exposure curve, in dollars per +1 vol point."""
    work, S = _prepare(df, weight_col=weight_col, today=today)
    if work is None:
        return _empty(weight_col)

    K = work["strike"].to_numpy(dtype=float)
    sigma = work["iv"].to_numpy(dtype=float)
    tau = work["tau"].to_numpy(dtype=float)
    weight = work["weight"].to_numpy(dtype=float)

    # Per-option vanna (∂Δ/∂σ). Convert to per-1-vol-point by ×0.01.
    vanna_per = bsm.vanna(S, K, tau, sigma, r=risk_free_rate)
    # Dollar-vanna = vanna × M × weight × 0.01 (per +1 vol pt) × S
    dollar_vanna = vanna_per * weight * CONTRACT_MULTIPLIER * S * 0.01

    work["_vanna"] = dollar_vanna
    curve_df = _signed_aggregate(work, "_vanna").rename(
        columns={"_vanna": "vanna_exposure"}
    )
    return _summarise(curve_df, "vanna_exposure",
                      S=S, weight_col=weight_col, top_n=top_n)


def compute_charm(
    df: pd.DataFrame,
    *,
    weight_col: str = "oi",
    risk_free_rate: float = 0.05,
    today: pd.Timestamp | None = None,
    top_n: int = 5,
) -> GreekSummary:
    """Aggregate dealer **Charm** exposure curve, in dollars per business day."""
    work, S = _prepare(df, weight_col=weight_col, today=today)
    if work is None:
        return _empty(weight_col)

    K = work["strike"].to_numpy(dtype=float)
    sigma = work["iv"].to_numpy(dtype=float)
    tau = work["tau"].to_numpy(dtype=float)
    weight = work["weight"].to_numpy(dtype=float)
    is_call = work["option_type"].astype(str).str.upper().to_numpy() == "C"

    # Per-option charm is in delta-per-year. Multiply by S to convert to
    # dollar-delta change per year, then scale to per-business-day.
    charm_per = bsm.charm(S, K, tau, sigma, r=risk_free_rate,
                          option_type=np.where(is_call, "C", "P"))
    business_days_per_year = 252.0
    dollar_charm = (
        charm_per * weight * CONTRACT_MULTIPLIER * S
    ) / business_days_per_year

    work["_charm"] = dollar_charm
    curve_df = _signed_aggregate(work, "_charm").rename(
        columns={"_charm": "charm_exposure"}
    )
    return _summarise(curve_df, "charm_exposure",
                      S=S, weight_col=weight_col, top_n=top_n)
