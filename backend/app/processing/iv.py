"""Implied Volatility utilities (Black-Scholes inversion + skew/ATM aggregates).

Also provides analytical Black-Scholes ``gamma`` and ``delta`` computation
so the pipeline can populate greeks even when the upstream feed only
publishes mid prices (OPRA Pillar does not transmit greeks; SqueezeMetrics
GEX requires them).

Inversion strategy (in priority order):

1. Robust bracketed root finder (``scipy.optimize.brentq``) on the BSM
   pricing function in ``[IV_LOWER_BOUND, IV_UPPER_BOUND]``.
2. Newton-Raphson fallback using analytical vega when the brentq bracket
   fails — common for deep ITM/OTM contracts where the price is dominated
   by intrinsic and vega is tiny but nonzero.

References:
* Hull, *Options, Futures, and Other Derivatives*, 10th ed., §19.
* Haug, *The Complete Guide to Option Pricing Formulas*, 2nd ed.
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

# Convergence tolerances for the root finder.
IV_XTOL = 1e-7
IV_RTOL = 1e-5
IV_NEWTON_MAX_ITER = 32
IV_NEWTON_TOL = 1e-6

# Below ~1 day to expiry brentq can become unstable on extreme prices.
# We do not lift the cap here (which would hide problems), but record the
# constant so callers can short-circuit.
SHORT_DATED_TAU_CUTOFF = 1.0 / 365.0


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


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float | None:
    """Analytical Black-Scholes vega (∂price/∂σ). Same for calls and puts.

    Returned in raw price-per-unit-sigma — multiply by 0.01 to get
    vega-per-1-vol-point if needed by the caller.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    v = S * pdf * math.sqrt(T)
    if not math.isfinite(v):
        return None
    return float(v)


def _bs_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # Discounted intrinsic value at expiry / degenerate. Using the
        # *discounted* strike here is the correct lower bound for an
        # European option price under positive ``r`` (otherwise we falsely
        # flag near-ITM contracts as no-arbitrage violations when only
        # the time value is small).
        discount = math.exp(-r * max(T, 0.0))
        intrinsic = (
            max(0.0, S - K * discount)
            if is_call
            else max(0.0, K * discount - S)
        )
        return intrinsic
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _newton_implied_vol(
    *,
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    is_call: bool,
    initial_sigma: float = 0.25,
) -> float | None:
    """Newton-Raphson IV fallback when brentq fails to bracket a root.

    Uses analytical vega as the derivative. Diverges quickly on bad
    starting points so we cap iterations and bail to ``None`` on overflow.
    """
    sigma = initial_sigma
    for _ in range(IV_NEWTON_MAX_ITER):
        try:
            f = _bs_price(S, K, T, r, sigma, is_call) - price
        except (ValueError, OverflowError):
            return None
        if not math.isfinite(f):
            return None
        if abs(f) < IV_NEWTON_TOL:
            if IV_LOWER_BOUND <= sigma <= IV_UPPER_BOUND:
                return float(sigma)
            return None
        v = bs_vega(S, K, T, r, sigma)
        if v is None or v < 1e-12:
            return None
        sigma = sigma - f / v
        if not math.isfinite(sigma) or sigma <= 0 or sigma > IV_UPPER_BOUND:
            return None
    return None


def implied_vol(
    *,
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    is_call: bool,
) -> float | None:
    """Return implied volatility via Black-Scholes inversion.

    Strategy:

    1. Reject obviously non-arbitrageable prices (price < discounted intrinsic).
    2. Try ``scipy.optimize.brentq`` on ``[IV_LOWER_BOUND, IV_UPPER_BOUND]``.
    3. If brentq fails to bracket (function has the same sign at both
       endpoints), fall back to Newton-Raphson from σ = 25 %.

    Returns ``None`` when both root finders fail.
    """
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None

    discount = math.exp(-r * T)
    intrinsic = (
        max(0.0, S - K * discount)
        if is_call
        else max(0.0, K * discount - S)
    )
    if price + 1e-12 < intrinsic:
        return None

    def objective(sigma: float) -> float:
        return _bs_price(S, K, T, r, sigma, is_call) - price

    try:
        f_lo = objective(IV_LOWER_BOUND)
        f_hi = objective(IV_UPPER_BOUND)
    except (ValueError, OverflowError):
        return None

    iv: float | None = None
    if math.isfinite(f_lo) and math.isfinite(f_hi) and f_lo * f_hi <= 0:
        try:
            iv = brentq(
                objective,
                IV_LOWER_BOUND,
                IV_UPPER_BOUND,
                maxiter=64,
                xtol=IV_XTOL,
                rtol=IV_RTOL,
            )
        except (ValueError, RuntimeError):
            iv = None

    if iv is None:
        iv = _newton_implied_vol(
            price=price, S=S, K=K, T=T, r=r, is_call=is_call
        )

    if iv is None or not math.isfinite(iv):
        return None
    if iv < IV_LOWER_BOUND or iv > IV_UPPER_BOUND:
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
