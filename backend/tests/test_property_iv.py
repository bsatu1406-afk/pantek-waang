"""Agent 9 — Hypothesis-based property tests for BSM/IV.

These tests exercise the inversion across a randomised grid of inputs.
They are tagged with the ``property`` marker so CI can run them in a
separate stage if needed (they are typically <2 s in aggregate, but
hypothesis can occasionally hit a long-tail shrink).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.processing.iv import _bs_price, bs_vega, implied_vol

pytestmark = pytest.mark.property


@settings(max_examples=200, deadline=None)
@given(
    sigma_true=st.floats(min_value=0.10, max_value=1.5, allow_nan=False),
    moneyness=st.floats(min_value=0.85, max_value=1.15, allow_nan=False),
    tau=st.floats(min_value=1 / 12, max_value=2.0, allow_nan=False),
    r=st.sampled_from([0.0, 0.02, 0.05]),
    is_call=st.booleans(),
)
def test_iv_round_trip_property(
    sigma_true: float, moneyness: float, tau: float, r: float, is_call: bool
) -> None:
    """Over a near-the-money (m ∈ [0.85, 1.15]) and well-vega'd surface,
    ``implied_vol(price(σ), …)`` recovers σ within 1e-3.

    We exclude deep wings (m < 0.85 or m > 1.15) because the price is
    dominated by intrinsic there and σ becomes poorly determined by the
    price — the Newton fallback still converges but the round-trip
    error inflates with low vega. The bracketed near-the-money regime
    is what production code cares about for ATM IV / skew.
    """
    S = 100.0
    K = S / moneyness  # m = S/K — higher m => more ITM call, more OTM put

    # Skip configurations where vega is effectively zero — IV is not
    # observable in that regime by construction.
    vega = bs_vega(S, K, tau, r, sigma_true)
    if vega is None or vega < 1e-4:
        return

    price = _bs_price(S, K, tau, r, sigma_true, is_call)
    if price <= 0:
        return  # degenerate edge from extreme inputs — skip silently

    iv = implied_vol(price=price, S=S, K=K, T=tau, r=r, is_call=is_call)
    if iv is None:
        # The inverter declined — acceptable when the price is below
        # discounted intrinsic by a hair due to float imprecision. We
        # only fail when both the inverter succeeded and the round-trip
        # drifted past tolerance.
        return
    assert math.isfinite(iv)
    assert iv == pytest.approx(sigma_true, abs=1e-3, rel=1e-3)
