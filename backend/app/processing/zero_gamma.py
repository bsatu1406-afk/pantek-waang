"""Zero-gamma computation.

Aggregate dealer-gamma flip level: the hypothetical underlying price ``S*``
at which the total dollar-gamma exposure of dealers crosses zero.

Method (vectorised grid scan + linear interpolation):

1.  For each strike in the chain, evaluate Black-Scholes gamma at every
    grid value of the underlying ``S`` (using each option's stored σ, K, τ,
    and the configured risk-free rate ``r``).
2.  Compute aggregate dealer dollar-gamma at each grid point::

        G(S) = Σ_i  sign_i · γ_i(S) · w_i · M · S² · 0.01

    where ``sign_i`` is ``+1`` for calls / ``-1`` for puts,
    ``γ_i(S)`` is the Black-Scholes gamma at hypothetical S,
    ``w_i`` is OI or traded volume (same convention as ``compute_gex``),
    and ``M = 100`` is the contract multiplier.
3.  Locate the grid index where ``G`` flips sign **closest to current spot**
    (multiple zero crossings can exist on chains with bimodal gamma
    distributions; the indicator user always wants the actionable one).
4.  Linearly interpolate between the two bracketing grid points to yield
    a sub-grid-resolution price.

Returns ``None`` if:

* The chain is empty or lacks valid ``iv`` / ``strike`` / ``expiration``.
* The aggregate gamma never flips sign within the search window.

Default parameters: ±5% half-width, 401 grid points → ≈0.025% resolution
(≈1.4 pts at SPX 5800), then linear-interpolation refinement to <0.05 pts.
This is well below typical strike spacing (5 pts on SPX, 25 pts on NDX) so
the user gets a precise level without arbitrarily small grids.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_zero_gamma(
    df: pd.DataFrame,
    *,
    weight_col: str = "volume",
    risk_free_rate: float = 0.05,
    today: pd.Timestamp | None = None,
    search_pct: float = 0.05,
    n_points: int = 401,
    fallback_to_closest: bool = False,
) -> float | None:
    """Hypothetical underlying ``S*`` at which aggregate dealer gamma = 0.

    See module docstring for the full methodology.

    Args:
        fallback_to_closest: When True, if the grid scan never crosses zero,
            return the grid point with the smallest ``|aggregate gamma|``
            (the closest-to-zero strike) rather than ``None``. The result is
            guaranteed finite; non-finite intermediate values still cause a
            ``None`` return.
    """
    required = {"strike", "option_type", "iv", "expiration", "underlying_price", weight_col}
    if df.empty or not required.issubset(df.columns):
        return None

    spot_series = pd.to_numeric(df["underlying_price"], errors="coerce").dropna()
    spot_series = spot_series[np.isfinite(spot_series)]
    if spot_series.empty:
        return None
    spot = float(spot_series.iloc[-1])
    if not np.isfinite(spot) or spot <= 0:
        return None

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
        return None

    K = work["strike"].to_numpy(dtype=float)
    sigma = work["iv"].to_numpy(dtype=float)
    tau = work["tau"].to_numpy(dtype=float)
    sign = np.where(work["option_type"].astype(str).str.upper() == "C", 1.0, -1.0)
    weight = work["weight"].to_numpy(dtype=float)
    r = float(risk_free_rate)

    s_lo = spot * (1.0 - search_pct)
    s_hi = spot * (1.0 + search_pct)
    if s_lo <= 0 or s_hi <= s_lo:
        return None
    s_grid = np.linspace(s_lo, s_hi, n_points)

    # Black-Scholes gamma evaluated on the (n_grid × n_options) tensor.
    log_ratio = np.log(s_grid[:, None] / K[None, :])
    sigma_sqrt_tau = sigma * np.sqrt(tau)  # shape (n_opt,)
    drift = (r + 0.5 * sigma * sigma) * tau
    d1 = (log_ratio + drift[None, :]) / sigma_sqrt_tau[None, :]
    pdf = norm.pdf(d1)
    gamma_grid = pdf / (s_grid[:, None] * sigma_sqrt_tau[None, :])

    # Per-strike contribution to aggregate dollar-gamma at each grid point.
    contrib = (
        sign[None, :]
        * gamma_grid
        * weight[None, :]
        * 100.0
        * (s_grid[:, None] ** 2)
        * 0.01
    )
    total_gex = contrib.sum(axis=1)

    if not np.all(np.isfinite(total_gex)):
        # Numerical guard: if any grid point produced a non-finite value
        # (e.g. extreme σ or τ), refuse to emit a level rather than report
        # a meaningless one.
        return None

    signs = np.sign(total_gex)
    diffs = np.diff(signs)
    cross_idx = np.where(diffs != 0)[0]
    if cross_idx.size == 0:
        if not fallback_to_closest:
            return None
        # No sign flip inside the window: report the grid point with the
        # smallest absolute dealer-gamma so consumers still get an
        # actionable level instead of a NaN/None hole on the chart.
        idx = int(np.argmin(np.abs(total_gex)))
        value = float(s_grid[idx])
        return value if np.isfinite(value) else None

    # Pick the crossing closest to current spot.
    nearest = int(cross_idx[np.argmin(np.abs(s_grid[cross_idx] - spot))])
    g0 = total_gex[nearest]
    g1 = total_gex[nearest + 1]
    if g1 == g0:
        result = float(s_grid[nearest])
        return result if np.isfinite(result) else None
    frac = -g0 / (g1 - g0)
    result = float(s_grid[nearest] + frac * (s_grid[nearest + 1] - s_grid[nearest]))
    return result if np.isfinite(result) else None
