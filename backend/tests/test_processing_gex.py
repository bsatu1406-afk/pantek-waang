"""Tests for the GEX (gamma exposure) computation."""

from __future__ import annotations

import math

import pandas as pd

from app.processing.gex import compute_gex


def test_compute_gex_handles_empty():
    summary = compute_gex(pd.DataFrame())
    assert summary.curve == []
    assert summary.top_positive == []
    assert summary.top_negative == []
    assert summary.net_total == 0.0


def test_compute_gex_call_positive_put_negative():
    """A pure-call book yields positive net GEX; a pure-put book yields negative."""
    S = 100.0
    df_call = pd.DataFrame(
        [
            {
                "strike": 100,
                "option_type": "C",
                "oi": 1000,
                "gamma": 0.05,
                "underlying_price": S,
            }
        ]
    )
    summary = compute_gex(df_call)
    expected = 1 * 0.05 * 1000 * 100 * (S**2) * 0.01
    assert math.isclose(summary.curve[0]["call_gex"], expected, rel_tol=1e-9)
    assert summary.net_total > 0
    assert summary.top_positive[0]["strike"] == 100

    df_put = pd.DataFrame(
        [
            {
                "strike": 100,
                "option_type": "P",
                "oi": 1000,
                "gamma": 0.05,
                "underlying_price": S,
            }
        ]
    )
    summary = compute_gex(df_put)
    assert summary.net_total < 0
    assert summary.top_negative[0]["strike"] == 100


def test_compute_gex_ranks_top_levels():
    S = 100.0
    rows = []
    # Many strikes with varying gamma * OI to produce distinct GEX values.
    for strike, gamma, oi in [
        (95, 0.05, 100),
        (100, 0.10, 500),
        (105, 0.04, 200),
        (110, 0.02, 800),
    ]:
        rows.append(
            {
                "strike": strike,
                "option_type": "C",
                "oi": oi,
                "gamma": gamma,
                "underlying_price": S,
            }
        )
    df = pd.DataFrame(rows)
    summary = compute_gex(df, top_n=2)
    assert len(summary.curve) == 4
    assert len(summary.top_positive) == 2
    assert summary.top_positive[0]["strike"] == 100  # highest gamma*OI
    assert summary.top_positive[0]["net_gex"] > summary.top_positive[1]["net_gex"]


def test_compute_gex_skips_when_no_underlying_price():
    df = pd.DataFrame(
        [{"strike": 100, "option_type": "C", "oi": 100, "gamma": 0.05, "underlying_price": None}]
    )
    summary = compute_gex(df)
    assert summary.underlying_price is None
    assert summary.curve == []


def test_compute_gex_volume_mode_uses_volume_weight():
    """Volume-weighted GEX flips magnitude when OI != volume."""
    S = 100.0
    df = pd.DataFrame(
        [
            {"strike": 100, "option_type": "C", "oi": 100, "volume": 5000,
             "gamma": 0.05, "underlying_price": S},
            {"strike": 100, "option_type": "P", "oi": 5000, "volume": 100,
             "gamma": 0.05, "underlying_price": S},
        ]
    )
    by_oi = compute_gex(df, weight_col="oi")
    by_vol = compute_gex(df, weight_col="volume")
    # Under OI: puts dominate (5000 vs 100) -> negative net GEX.
    assert by_oi.net_total < 0
    assert by_oi.weight_col == "oi"
    # Under Volume: calls dominate (5000 vs 100) -> positive net GEX.
    assert by_vol.net_total > 0
    assert by_vol.weight_col == "volume"


def test_compute_gex_returns_empty_when_weight_column_all_zero():
    df = pd.DataFrame(
        [{"strike": 100, "option_type": "C", "oi": 0, "volume": 0,
          "gamma": 0.05, "underlying_price": 100.0}]
    )
    by_oi = compute_gex(df, weight_col="oi")
    by_vol = compute_gex(df, weight_col="volume")
    assert by_oi.net_total == 0.0
    assert by_oi.curve == []
    assert by_vol.net_total == 0.0
    assert by_vol.curve == []
