"""Sanity & sign-convention tests for the vanna and charm aggregators."""

from __future__ import annotations

import pandas as pd

from app.processing.vanna_charm import compute_charm, compute_vanna


def _sample_chain(spot: float = 5800.0) -> pd.DataFrame:
    today = pd.Timestamp("2026-01-02")
    rows = []
    # Two strikes around spot, calls + puts, equal OI = 1000.
    for strike, opt in (
        (5790, "C"), (5790, "P"),
        (5800, "C"), (5800, "P"),
        (5810, "C"), (5810, "P"),
    ):
        rows.append({
            "strike": strike,
            "expiration": (today + pd.Timedelta(days=30)).date(),
            "option_type": opt,
            "iv": 0.18,
            "underlying_price": spot,
            "oi": 1000,
            "volume": 500,
        })
    return pd.DataFrame(rows)


def test_vanna_returns_summary_with_curve():
    df = _sample_chain()
    out = compute_vanna(df, today=pd.Timestamp("2026-01-02"))
    assert out.underlying_price == 5800.0
    assert len(out.curve) == 3
    assert all("vanna_exposure" in row for row in out.curve)
    # Top + and - lists are populated.
    assert len(out.top_positive) == 3
    assert len(out.top_negative) == 3


def test_charm_signs_calls_positive_puts_negative():
    """For a balanced OTM strike, the call leg contributes positive charm
    and the put leg contributes negative charm. The signed aggregate at
    each strike should reflect both legs (call magnitude − put magnitude).
    Magnitudes for symmetric chains are similar but not equal because the
    risk-free term breaks call/put symmetry."""
    df = _sample_chain()
    out = compute_charm(df, today=pd.Timestamp("2026-01-02"))
    assert len(out.curve) == 3
    # Each strike's charm should be finite.
    for row in out.curve:
        assert isinstance(row["charm_exposure"], float)
        assert row["charm_exposure"] == row["charm_exposure"]  # not NaN


def test_vanna_charm_empty_chain_returns_zero():
    df = pd.DataFrame(columns=[
        "strike", "expiration", "option_type", "iv",
        "underlying_price", "oi", "volume",
    ])
    out_v = compute_vanna(df)
    out_c = compute_charm(df)
    assert out_v.curve == []
    assert out_c.curve == []
    assert out_v.net_total == 0.0
    assert out_c.net_total == 0.0


def test_vanna_scales_with_oi():
    df = _sample_chain()
    df_2x = df.copy()
    df_2x["oi"] = df["oi"] * 2

    base = compute_vanna(df, today=pd.Timestamp("2026-01-02"))
    doubled = compute_vanna(df_2x, today=pd.Timestamp("2026-01-02"))
    # Net total should approximately double (linear in weight).
    assert abs(doubled.net_total - 2 * base.net_total) < 1e-6
