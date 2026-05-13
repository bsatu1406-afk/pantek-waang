"""Black-Scholes-Merton greek primitives.

Centralised so every processing module (GEX, Vanna, Charm, Pin) uses
the exact same conventions:

* All inputs in absolute units (decimal IV, year fraction τ, dollar prices).
* Calls and puts share the same gamma / vanna; charm differs by sign.
* Dealer-side hedging convention applied at the *aggregator* level
  (sign by option_type), NOT here — this module returns option-side greeks.

References:
* Hull, *Options, Futures, and Other Derivatives*, ch. 19 (Greeks).
* Wilmott, *Paul Wilmott on Quantitative Finance*, ch. 8 (Greeks formulas).

Notation throughout::

    S       spot underlying price
    K       strike
    tau     time to expiration in years
    sigma   implied volatility (decimal, e.g. 0.20 for 20%)
    r       continuously-compounded risk-free rate (decimal)
    q       continuous dividend yield (decimal); we default to 0 for
            equity-index options (carry is bundled into the futures basis).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, tau, sigma, r, q=0.0):  # type: ignore[no-untyped-def]
    """Compute (d1, d2) on broadcast-compatible array inputs.

    Caller is responsible for masking τ ≤ 0 / σ ≤ 0 / S ≤ 0 / K ≤ 0
    rows beforehand. We do not silently substitute 0s here because
    that would yield misleading divide-by-zero greeks downstream.
    """
    sigma_sqrt_tau = sigma * np.sqrt(tau)
    log_ratio = np.log(S / K)
    drift = (r - q + 0.5 * sigma * sigma) * tau
    d1 = (log_ratio + drift) / sigma_sqrt_tau
    d2 = d1 - sigma_sqrt_tau
    return d1, d2


def gamma(S, K, tau, sigma, r=0.0, q=0.0):  # type: ignore[no-untyped-def]
    """BSM gamma (call and put are identical).

    γ = e^{-qτ} · φ(d1) / (S · σ · √τ)
    """
    d1, _ = _d1_d2(S, K, tau, sigma, r, q)
    return np.exp(-q * tau) * norm.pdf(d1) / (S * sigma * np.sqrt(tau))


def vanna(S, K, tau, sigma, r=0.0, q=0.0):  # type: ignore[no-untyped-def]
    """BSM vanna = ∂Δ/∂σ = ∂ν/∂S (call and put are identical).

    Vanna = -e^{-qτ} · φ(d1) · d2 / σ

    Interpretation: how much delta moves per +1 vol-point change in σ.
    Important for vol-sensitive hedging — long vega traders accumulate
    delta when vol rises and shed it when vol falls.
    """
    d1, d2 = _d1_d2(S, K, tau, sigma, r, q)
    return -np.exp(-q * tau) * norm.pdf(d1) * d2 / sigma


def charm(S, K, tau, sigma, r=0.0, q=0.0, *, option_type="C"):  # type: ignore[no-untyped-def]
    """BSM charm = ∂Δ/∂t (calls and puts differ by an extra qN(d1) / -qN(-d1) term).

    For a call::

        charm_C = -e^{-qτ} · [ φ(d1) · ((r - q)/(σ√τ) − d2/(2τ)) − q · N(d1) ]

    For a put, replace ``q · N(d1)`` with ``−q · N(-d1)``.

    Note: returned value is **per year**. To compare against business-time
    expectations downstream, multiply by elapsed (or remaining) days/365.

    The ``option_type`` argument may be a scalar ('C'/'P') or a vectorised
    object array of equal length to S/K/τ/σ.
    """
    d1, d2 = _d1_d2(S, K, tau, sigma, r, q)
    pdf = norm.pdf(d1)
    sigma_sqrt_tau = sigma * np.sqrt(tau)
    common = -np.exp(-q * tau) * pdf * ((r - q) / sigma_sqrt_tau - d2 / (2 * tau))

    is_call = _is_call_mask(option_type)
    extra_call = q * np.exp(-q * tau) * norm.cdf(d1)
    extra_put = -q * np.exp(-q * tau) * norm.cdf(-d1)
    return common + np.where(is_call, extra_call, extra_put)


def delta(S, K, tau, sigma, r=0.0, q=0.0, *, option_type="C"):  # type: ignore[no-untyped-def]
    """BSM delta. Useful for DEX (dollar-delta exposure) and 25Δ-skew location."""
    d1, _ = _d1_d2(S, K, tau, sigma, r, q)
    is_call = _is_call_mask(option_type)
    call_delta = np.exp(-q * tau) * norm.cdf(d1)
    put_delta = -np.exp(-q * tau) * norm.cdf(-d1)
    return np.where(is_call, call_delta, put_delta)


def vega(S, K, tau, sigma, r=0.0, q=0.0):  # type: ignore[no-untyped-def]
    """BSM vega per +1.00 σ unit (i.e. raw, not 1-vol-point scaled)."""
    d1, _ = _d1_d2(S, K, tau, sigma, r, q)
    return S * np.exp(-q * tau) * norm.pdf(d1) * np.sqrt(tau)


def _is_call_mask(option_type):  # type: ignore[no-untyped-def]
    """Convert scalar or array option_type into a boolean is-call mask."""
    arr = np.asarray(option_type)
    if arr.dtype.kind in {"U", "S", "O"}:
        return np.char.upper(arr.astype(str)) == "C"
    return arr.astype(bool)
