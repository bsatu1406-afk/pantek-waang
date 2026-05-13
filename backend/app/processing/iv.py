"""Implied Volatility utilities (Black-Scholes inversion + skew/ATM aggregates).

Also provides analytical Black-Scholes ``gamma`` and ``delta`` computation
so the pipeline can populate greeks even when the upstream feed only
publishes mid prices (OPRA Pillar does not transmit greeks; SqueezeMetrics
GEX requires them).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

# Reasonable IV bounds: 1% – 500% annualized.
IV_LOWER_BOUND = 0.01
IV_UPPER_BOUND = 5.0


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float | None:
    """Analytical Black-Scholes gamma. Same for calls and puts."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    g = pdf / (S * sigma * math.sqrt(T))
    if not math.isfinite(g):
        return None
    return float(g)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float | None:
    """Analytical Black-Scholes delta."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    if is_call:
        d = norm.cdf(d1)
    else:
        d = norm.cdf(d1) - 1.0
    if not math.isfinite(d):
        return None
    return float(d)


def _bs_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # Intrinsic value at expiry / degenerate
        intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
        return intrinsic
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(
    *,
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    is_call: bool,
) -> float | None:
    """Return implied volatility via Black-Scholes inversion using brentq.

    Returns ``None`` when the price is non-arbitrageable or no solution exists in bounds.
    """
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None

    intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
    if price < intrinsic:
        return None

    def objective(sigma: float) -> float:
        return _bs_price(S, K, T, r, sigma, is_call) - price

    try:
        f_lo = objective(IV_LOWER_BOUND)
        f_hi = objective(IV_UPPER_BOUND)
    except (ValueError, OverflowError):
        return None

    if f_lo * f_hi > 0:
        return None
    try:
        iv = brentq(objective, IV_LOWER_BOUND, IV_UPPER_BOUND, maxiter=64, xtol=1e-6)
    except (ValueError, RuntimeError):
        return None

    if iv < IV_LOWER_BOUND or iv > IV_UPPER_BOUND or not math.isfinite(iv):
        return None
    return float(iv)


@dataclass
class IVSummary:
    atm_iv: float | None
    skew_per_expiry: dict[str, float]
    surface: list[dict]


def _years_to_expiry(today: pd.Timestamp, expiry: pd.Timestamp) -> float:
    # Compare on a date basis to avoid tz-naive vs tz-aware subtraction issues
    # (the DB ``expiration`` column round-trips as a tz-naive date, while
    # ``today`` is sometimes tz-aware).
    today_d = today.date() if hasattr(today, "date") else today
    expiry_d = expiry.date() if hasattr(expiry, "date") else expiry
    days = max(1, (expiry_d - today_d).days)
    return days / 365.0


def _row_price(row: pd.Series) -> float:
    """Pick the best available reference price: last → mid(bid,ask) → 0."""
    last = row.get("last_price")
    if last is not None and not pd.isna(last) and last > 0:
        return float(last)
    bid = row.get("bid")
    ask = row.get("ask")
    if (
        bid is not None
        and ask is not None
        and not pd.isna(bid)
        and not pd.isna(ask)
        and bid > 0
        and ask > 0
    ):
        return float((bid + ask) / 2.0)
    return 0.0


def fill_missing_iv(
    df: pd.DataFrame,
    *,
    risk_free_rate: float,
    today: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute IV via Black-Scholes when the feed-provided IV is missing/invalid.

    Expects columns: ``strike``, ``expiration``, ``option_type``, ``last_price``,
    ``underlying_price``, ``iv``. Optionally consumes ``bid``/``ask`` to use a
    mid-price when ``last_price`` is missing.

    Side effect: populates ``gamma`` and ``delta`` analytically wherever they
    are missing/zero and a valid IV is available, so downstream GEX
    computation can run even when the upstream feed (e.g. OPRA Pillar) does
    not publish greeks.
    """
    if df.empty:
        return df

    df = df.copy()
    if today is None:
        today = pd.Timestamp.utcnow()
        if today.tzinfo is not None:
            today = today.tz_convert(None)

    needs_iv = df["iv"].isna() | (df["iv"] <= 0) | (df["iv"] > IV_UPPER_BOUND)
    for idx in df.index[needs_iv]:
        row = df.loc[idx]
        S = float(row.get("underlying_price") or 0)
        K = float(row.get("strike") or 0)
        price = _row_price(row)
        if not (S and K and price):
            continue
        T = _years_to_expiry(today, pd.Timestamp(row["expiration"]))
        is_call = str(row["option_type"]).upper() == "C"
        iv = implied_vol(price=price, S=S, K=K, T=T, r=risk_free_rate, is_call=is_call)
        if iv is not None:
            df.at[idx, "iv"] = iv

    # Analytical greeks fill: gamma/delta from (S, K, T, sigma).
    if "gamma" not in df.columns:
        df["gamma"] = np.nan
    if "delta" not in df.columns:
        df["delta"] = np.nan
    df["gamma"] = pd.to_numeric(df["gamma"], errors="coerce")
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")

    iv_ok = df["iv"].notna() & (df["iv"] > 0)
    spot_ok = df["underlying_price"].notna() & (df["underlying_price"] > 0)
    strike_ok = df["strike"].notna() & (df["strike"] > 0)
    gamma_missing = df["gamma"].isna() | (df["gamma"].fillna(0).abs() == 0)
    delta_missing = df["delta"].isna() | (df["delta"].fillna(0).abs() == 0)
    needs_greeks = iv_ok & spot_ok & strike_ok & (gamma_missing | delta_missing)

    for idx in df.index[needs_greeks]:
        row = df.loc[idx]
        S = float(row["underlying_price"])
        K = float(row["strike"])
        sigma = float(row["iv"])
        T = _years_to_expiry(today, pd.Timestamp(row["expiration"]))
        is_call = str(row["option_type"]).upper() == "C"
        cur_g = df.at[idx, "gamma"]
        if pd.isna(cur_g) or float(cur_g) == 0:
            g = bs_gamma(S, K, T, risk_free_rate, sigma)
            if g is not None:
                df.at[idx, "gamma"] = g
        cur_d = df.at[idx, "delta"]
        if pd.isna(cur_d) or float(cur_d) == 0:
            d = bs_delta(S, K, T, risk_free_rate, sigma, is_call)
            if d is not None:
                df.at[idx, "delta"] = d
    return df


def compute_iv_summary(df: pd.DataFrame) -> IVSummary:
    """Return ATM IV, skew per expiry, and a flattened IV surface."""
    if df.empty or df["underlying_price"].dropna().empty:
        return IVSummary(atm_iv=None, skew_per_expiry={}, surface=[])

    spot = float(df["underlying_price"].dropna().iloc[-1])

    # ATM IV: average of nearest-strike call and put IVs (pooled across nearest expiry).
    expiries_sorted = sorted(pd.to_datetime(df["expiration"].unique()))
    atm_iv: float | None = None
    if expiries_sorted:
        nearest_expiry = expiries_sorted[0]
        sub = df[pd.to_datetime(df["expiration"]) == nearest_expiry].dropna(subset=["iv"])
        if not sub.empty:
            sub = sub.assign(dist=lambda d: (d["strike"] - spot).abs())
            min_dist = sub["dist"].min()
            atm_rows = sub[sub["dist"] == min_dist]
            atm_iv = float(atm_rows["iv"].mean())

    # Skew per expiry: 25-delta call IV − 25-delta put IV (pick rows closest to 0.25 / -0.25).
    skew: dict[str, float] = {}
    for expiry, sub in df.dropna(subset=["iv", "delta"]).groupby("expiration"):
        calls = sub[sub["option_type"].str.upper() == "C"]
        puts = sub[sub["option_type"].str.upper() == "P"]
        if calls.empty or puts.empty:
            continue
        c_row = calls.iloc[(calls["delta"] - 0.25).abs().argsort()[:1]]
        p_row = puts.iloc[(puts["delta"] - (-0.25)).abs().argsort()[:1]]
        if c_row.empty or p_row.empty:
            continue
        skew[str(pd.Timestamp(expiry).date())] = float(
            c_row["iv"].iloc[0] - p_row["iv"].iloc[0]
        )

    # Surface: flatten valid rows.
    surface_df = df.dropna(subset=["iv"])[
        ["expiration", "strike", "option_type", "iv", "delta"]
    ].copy()
    surface_df["expiration"] = surface_df["expiration"].apply(
        lambda d: str(pd.Timestamp(d).date())
    )
    surface_df["strike"] = surface_df["strike"].astype(float)
    surface_df["iv"] = surface_df["iv"].astype(float)
    surface_df["delta"] = surface_df["delta"].astype(float).where(surface_df["delta"].notna(), None)
    surface_df = surface_df.replace({np.nan: None})

    return IVSummary(
        atm_iv=atm_iv,
        skew_per_expiry=skew,
        surface=surface_df.to_dict(orient="records"),
    )
