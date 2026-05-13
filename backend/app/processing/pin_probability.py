"""0DTE pin-probability heatmap.

Models the probability that the underlying *closes the session* at each
0DTE strike, weighted by:

* **Open interest** — the larger the dealer's net long-gamma position at
  a strike, the stronger the *gamma-pin* effect (dealers buy weakness /
  sell strength to stay flat as expiry approaches).
* **Charm** — toward expiry, dealer hedges roll into the underlying at a
  rate proportional to charm. We use ``|charm|`` magnitude per strike as
  a *dynamic* pinning force.
* **Distance from current spot** — exponential decay weighted by the
  remaining ATM straddle implied move (i.e. P(close near K) ∝
  exp(−½ · ((K-S)/σ_remaining)²) ).

Final score per strike::

    raw_i      = (oi_call + oi_put + charm_weight · |charm_i|) · gauss((K_i-S)/σ)
    prob_i     = raw_i / Σ_j raw_j

This is **not** a calibrated probability — it's a relative likelihood
heatmap, and that is what we expose. Callers normalise on display.

Returns a list of ``{strike, prob, contributors}`` dicts sorted by
probability descending.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.processing import bsm


def compute_pin_probability(
    df: pd.DataFrame,
    *,
    today: pd.Timestamp | None = None,
    sigma_floor: float = 1e-4,
    charm_weight: float = 0.5,
    risk_free_rate: float = 0.05,
) -> list[dict]:
    """Return a probability-weighted list of 0DTE pin candidates.

    Required columns: ``strike, expiration, option_type, oi, iv,
    underlying_price``. ``charm`` is computed analytically inside.
    """
    needed = {"strike", "expiration", "option_type", "oi", "iv", "underlying_price"}
    if df.empty or not needed.issubset(df.columns):
        return []

    spot_series = df["underlying_price"].dropna()
    if spot_series.empty:
        return []
    S = float(spot_series.iloc[-1])

    if today is None:
        today = pd.Timestamp.utcnow()
        if today.tzinfo is not None:
            today = today.tz_convert(None)
    today_d = today.date() if hasattr(today, "date") else today

    # 0DTE-only filter.
    work = df.copy()
    work["expiration_d"] = work["expiration"].apply(
        lambda x: pd.Timestamp(x).date() if pd.notna(x) else None
    )
    work = work[work["expiration_d"] == today_d]
    if work.empty:
        return []

    work["iv"] = pd.to_numeric(work["iv"], errors="coerce")
    work["strike"] = pd.to_numeric(work["strike"], errors="coerce")
    work["oi"] = pd.to_numeric(work["oi"], errors="coerce").fillna(0)
    work = work[
        work["iv"].notna()
        & (work["iv"] > 0)
        & work["strike"].notna()
        & (work["strike"] > 0)
    ]
    if work.empty:
        return []

    # Use a 1-day floor for tau so charm doesn't blow up when we compute
    # at session open (intra-day, fractional days remaining is fine).
    tau = max(1, 1) / 365.0

    K = work["strike"].to_numpy(dtype=float)
    sigma = work["iv"].to_numpy(dtype=float)
    is_call = work["option_type"].astype(str).str.upper().to_numpy() == "C"

    charm_arr = bsm.charm(
        S, K, tau, sigma, r=risk_free_rate,
        option_type=np.where(is_call, "C", "P"),
    )
    work["_abs_charm"] = np.abs(charm_arr)

    by_strike = work.groupby("strike", as_index=False).agg(
        oi=("oi", "sum"),
        abs_charm=("_abs_charm", "sum"),
        atm_iv=("iv", "median"),
    )

    # Remaining one-σ band for distance kernel. Use the median ATM IV on
    # 0DTE chain as the daily σ proxy (in absolute pts).
    median_iv = float(np.nanmedian(work["iv"])) if work["iv"].notna().any() else 0.20
    sigma_pts = max(sigma_floor, S * median_iv * np.sqrt(tau))

    distance = (by_strike["strike"].to_numpy(dtype=float) - S) / sigma_pts
    kernel = np.exp(-0.5 * distance * distance)

    raw = (
        by_strike["oi"].to_numpy(dtype=float)
        + charm_weight * by_strike["abs_charm"].to_numpy(dtype=float)
    ) * kernel
    total = raw.sum()
    if total <= 0 or not np.isfinite(total):
        return []
    prob = raw / total

    payload = [
        {
            "strike": float(by_strike.iloc[i]["strike"]),
            "prob": float(prob[i]),
            "oi": int(by_strike.iloc[i]["oi"]),
            "abs_charm": float(by_strike.iloc[i]["abs_charm"]),
            "atm_iv": (
                float(by_strike.iloc[i]["atm_iv"])
                if not np.isnan(by_strike.iloc[i]["atm_iv"])
                else None
            ),
        }
        for i in range(len(by_strike))
    ]
    payload.sort(key=lambda r: r["prob"], reverse=True)
    return payload
