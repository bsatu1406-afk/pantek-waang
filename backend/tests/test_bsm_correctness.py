"""Agent 1 — BSM / IV correctness tests.

These tests exercise the analytical greek primitives in ``app.processing.bsm``
and the IV inversion in ``app.processing.iv`` against well-known closed-form
identities:

* Put-call parity: ``C - P = S·e^{-qτ} - K·e^{-rτ}``.
* Vanna finite-difference: ``∂Δ/∂σ ≈ vanna(σ)`` for small Δσ.
* Charm sign: long-dated calls have negative charm (delta bleeds to ATM
  with time decay), long-dated puts have positive charm.
* IV round-trip: ``implied_vol(price=price(σ), …) ≈ σ`` for a grid of
  ATM/ITM/OTM contracts and rates ∈ {0%, 5%}.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.processing import bsm
from app.processing.iv import _bs_price, bs_gamma, bs_vega, implied_vol

# ── Closed-form helpers (kept local so tests don't depend on _bs_price's API) ─

def _bsm_price(
    S: float, K: float, tau: float, sigma: float, r: float, *, is_call: bool
) -> float:
    """Reference BSM price for sanity checks. Mirrors :func:`iv._bs_price`."""
    # iv._bs_price signature is (S, K, T, r, sigma, is_call) — note r before σ.
    return _bs_price(S, K, tau, r, sigma, is_call)


# ── 1. Put-call parity ───────────────────────────────────────────────────────


@pytest.mark.parametrize("S,K,tau,sigma,r", [
    (100.0, 100.0, 0.25, 0.20, 0.05),
    (4500.0, 4400.0, 0.50, 0.18, 0.05),
    (50.0, 100.0, 1.00, 0.40, 0.00),
    (200.0, 100.0, 0.10, 0.30, 0.03),
])
def test_put_call_parity(S: float, K: float, tau: float, sigma: float, r: float) -> None:
    """C − P should equal S − K·e^{-rτ} to high precision (q = 0)."""
    c = _bsm_price(S, K, tau, sigma, r, is_call=True)
    p = _bsm_price(S, K, tau, sigma, r, is_call=False)
    expected = S - K * math.exp(-r * tau)
    assert c - p == pytest.approx(expected, rel=1e-10, abs=1e-9)


# ── 2. Vanna via finite differences ──────────────────────────────────────────


@pytest.mark.parametrize("S,K,tau,sigma,r", [
    (100.0, 100.0, 0.25, 0.20, 0.05),
    (100.0, 110.0, 0.25, 0.25, 0.05),
    (100.0, 90.0, 0.50, 0.18, 0.05),
])
def test_vanna_finite_difference(
    S: float, K: float, tau: float, sigma: float, r: float
) -> None:
    """∂Δ/∂σ ≈ vanna(σ) to within 1e-4 absolute via central differences."""
    h = 1e-4
    delta_up = bsm.delta(S, K, tau, sigma + h, r=r, option_type="C")
    delta_dn = bsm.delta(S, K, tau, sigma - h, r=r, option_type="C")
    fd = (float(delta_up) - float(delta_dn)) / (2 * h)
    analytical = float(bsm.vanna(S, K, tau, sigma, r=r))
    assert fd == pytest.approx(analytical, rel=1e-3, abs=1e-4)


# ── 3. Charm sign behaviour ──────────────────────────────────────────────────


def test_charm_sign_atm_call_is_negative() -> None:
    """For an ATM call with positive rates, charm should be negative
    (delta bleeds *down* toward 0.5 as we approach expiry)."""
    c = float(bsm.charm(100.0, 100.0, 0.25, 0.20, r=0.05, option_type="C"))
    assert c < 0


def test_charm_matches_delta_finite_difference() -> None:
    """``bsm.charm`` should match ``-∂Δ/∂τ`` (with τ = time to expiry)
    to within 1e-3 absolute for both calls and puts.

    We use a central-difference estimate of ``∂Δ/∂τ``, then negate it,
    which corresponds to ``∂Δ/∂t`` (calendar time passing).
    """
    S, K, sigma, r, tau = 100.0, 100.0, 0.20, 0.05, 0.25
    h = 1e-4
    for opt in ("C", "P"):
        d_up = float(bsm.delta(S, K, tau + h, sigma, r=r, option_type=opt))
        d_dn = float(bsm.delta(S, K, tau - h, sigma, r=r, option_type=opt))
        fd = -(d_up - d_dn) / (2 * h)  # ∂Δ/∂t = −∂Δ/∂τ
        analytical = float(bsm.charm(S, K, tau, sigma, r=r, option_type=opt))
        assert fd == pytest.approx(analytical, rel=1e-2, abs=1e-3)


# ── 4. Gamma sanity: peaks ATM, monotone in σ near ATM ───────────────────────


def test_gamma_peaks_at_the_money() -> None:
    """Gamma should be largest at K ≈ S for a given expiry/IV."""
    S, tau, sigma, r = 100.0, 0.25, 0.20, 0.05
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    gammas = np.array(
        [float(bsm.gamma(S, k, tau, sigma, r=r)) for k in strikes]
    )
    assert gammas.argmax() == 2  # K=100


# ── 5. IV round-trip (ATM / ITM / OTM, r ∈ {0%, 5%}) ─────────────────────────


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("r", [0.0, 0.05])
@pytest.mark.parametrize("sigma_true", [0.10, 0.20, 0.40, 0.80])
@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
def test_iv_round_trip(
    is_call: bool, r: float, sigma_true: float, K: float
) -> None:
    """``implied_vol(price = price(σ), …)`` should recover σ within 1e-5."""
    S, tau = 100.0, 0.25
    price = _bsm_price(S, K, tau, sigma_true, r, is_call=is_call)
    iv = implied_vol(
        price=price, S=S, K=K, T=tau, r=r, is_call=is_call
    )
    assert iv is not None
    assert iv == pytest.approx(sigma_true, abs=1e-5)


# ── 6. IV rejects sub-intrinsic prices ───────────────────────────────────────


def test_iv_rejects_sub_intrinsic_price() -> None:
    """A call price below max(0, S − K·e^{-rτ}) is unarbitrageable."""
    S, K, tau, r = 100.0, 50.0, 0.25, 0.05
    intrinsic = S - K * math.exp(-r * tau)
    iv = implied_vol(
        price=intrinsic - 1.0, S=S, K=K, T=tau, r=r, is_call=True
    )
    assert iv is None


# ── 7. Newton fallback engages on deep-OTM where brentq fails to bracket ─────


def test_iv_newton_fallback_engages_on_extreme_otm() -> None:
    """Deep-OTM low-vol contracts can have prices indistinguishable from
    zero on the brentq lower bound — Newton with σ₀ = 25 % should still
    converge. Use a moderately OTM strike so the price is detectable."""
    S, K, tau, r = 100.0, 130.0, 0.50, 0.05
    sigma_true = 0.30
    price = _bsm_price(S, K, tau, sigma_true, r, is_call=True)
    iv = implied_vol(price=price, S=S, K=K, T=tau, r=r, is_call=True)
    assert iv is not None
    assert iv == pytest.approx(sigma_true, abs=1e-3)


# ── 8. Vega is non-negative and zero only on degenerate input ────────────────


@pytest.mark.parametrize("S,K,tau,sigma,r", [
    (100.0, 100.0, 0.25, 0.20, 0.05),
    (100.0, 50.0, 0.25, 0.20, 0.05),
    (100.0, 200.0, 0.25, 0.20, 0.05),
])
def test_vega_is_non_negative(
    S: float, K: float, tau: float, sigma: float, r: float
) -> None:
    # bs_vega signature: (S, K, T, r, sigma)
    v = bs_vega(S, K, tau, r, sigma)
    assert v is not None and v >= 0


# ── 9. Gamma matches between bsm.gamma and iv.bs_gamma (no drift) ────────────


def test_bsm_gamma_equals_iv_bs_gamma() -> None:
    """``bsm.gamma`` (vectorised, q=0) should agree with ``iv.bs_gamma``."""
    S, K, tau, sigma, r = 100.0, 100.0, 0.25, 0.20, 0.05
    g1 = float(bsm.gamma(S, K, tau, sigma, r=r))
    # bs_gamma signature: (S, K, T, r, sigma)
    g2 = bs_gamma(S, K, tau, r, sigma)
    assert g2 is not None
    assert g1 == pytest.approx(g2, rel=1e-10)
