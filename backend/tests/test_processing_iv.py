"""Tests for IV calculation: BS inversion + summary statistics."""

from __future__ import annotations

import math

import pandas as pd

from app.processing.iv import (
    IV_LOWER_BOUND,
    IV_UPPER_BOUND,
    _bs_price,
    compute_iv_summary,
    fill_missing_iv,
    implied_vol,
)


def test_bs_inversion_round_trip_call():
    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20
    price = _bs_price(S, K, T, r, sigma, is_call=True)
    iv = implied_vol(price=price, S=S, K=K, T=T, r=r, is_call=True)
    assert iv is not None
    assert math.isclose(iv, sigma, rel_tol=1e-3, abs_tol=1e-3)


def test_bs_inversion_round_trip_put():
    S, K, T, r, sigma = 100.0, 110.0, 1.0, 0.04, 0.35
    price = _bs_price(S, K, T, r, sigma, is_call=False)
    iv = implied_vol(price=price, S=S, K=K, T=T, r=r, is_call=False)
    assert iv is not None
    assert math.isclose(iv, sigma, rel_tol=2e-3, abs_tol=2e-3)


def test_implied_vol_returns_none_for_arbitrage_violation():
    S, K, T, r = 100.0, 100.0, 0.25, 0.05
    # Price below intrinsic -> no solution.
    iv = implied_vol(price=-1.0, S=S, K=K, T=T, r=r, is_call=True)
    assert iv is None


def test_implied_vol_clamps_within_bounds():
    iv = implied_vol(price=0.5, S=100.0, K=100.0, T=0.001, r=0.05, is_call=True)
    if iv is not None:
        assert IV_LOWER_BOUND <= iv <= IV_UPPER_BOUND


def test_fill_missing_iv_only_replaces_invalid():
    df = pd.DataFrame(
        [
            {
                "expiration": pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=30),
                "strike": 100.0,
                "option_type": "C",
                "last_price": 5.0,
                "underlying_price": 100.0,
                "iv": 0.25,
                "delta": None,
                "gamma": None,
            },
            {
                "expiration": pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=30),
                "strike": 105.0,
                "option_type": "C",
                "last_price": 2.5,
                "underlying_price": 100.0,
                "iv": None,  # to be filled
                "delta": None,
                "gamma": None,
            },
        ]
    )
    out = fill_missing_iv(df, risk_free_rate=0.05)
    assert math.isclose(float(out.loc[0, "iv"]), 0.25)
    iv_filled = out.loc[1, "iv"]
    assert iv_filled is not None and IV_LOWER_BOUND <= float(iv_filled) <= IV_UPPER_BOUND


def test_compute_iv_summary_returns_atm_and_skew():
    today = pd.Timestamp.utcnow().normalize()
    expiry = today + pd.Timedelta(days=30)
    rows = [
        {"expiration": expiry, "strike": 95.0, "option_type": "P", "iv": 0.35,
         "delta": -0.25, "underlying_price": 100.0},
        {"expiration": expiry, "strike": 100.0, "option_type": "C", "iv": 0.20,
         "delta": 0.50, "underlying_price": 100.0},
        {"expiration": expiry, "strike": 100.0, "option_type": "P", "iv": 0.22,
         "delta": -0.50, "underlying_price": 100.0},
        {"expiration": expiry, "strike": 105.0, "option_type": "C", "iv": 0.18,
         "delta": 0.25, "underlying_price": 100.0},
    ]
    df = pd.DataFrame(rows)
    summary = compute_iv_summary(df)
    assert summary.atm_iv is not None
    assert math.isclose(summary.atm_iv, (0.20 + 0.22) / 2, rel_tol=1e-6)
    expiry_str = str(expiry.date())
    assert expiry_str in summary.skew_per_expiry
    assert math.isclose(summary.skew_per_expiry[expiry_str], 0.18 - 0.35, rel_tol=1e-6)
    assert len(summary.surface) == 4


def test_compute_iv_summary_handles_empty():
    summary = compute_iv_summary(pd.DataFrame())
    assert summary.atm_iv is None
    assert summary.skew_per_expiry == {}
    assert summary.surface == []
