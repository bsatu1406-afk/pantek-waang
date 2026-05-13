"""Pin probability heatmap sanity tests."""

from __future__ import annotations

import pandas as pd

from app.processing.pin_probability import compute_pin_probability


def _zero_dte_chain(spot: float = 5800.0):
    today = pd.Timestamp("2026-01-02")
    rows = []
    for strike in (5790, 5795, 5800, 5805, 5810):
        for opt in ("C", "P"):
            rows.append({
                "strike": strike,
                "expiration": today.date(),
                "option_type": opt,
                "iv": 0.18,
                "underlying_price": spot,
                "oi": 1000 if strike == 5800 else 200,
            })
    return pd.DataFrame(rows)


def test_distribution_normalises_to_one():
    out = compute_pin_probability(
        _zero_dte_chain(),
        today=pd.Timestamp("2026-01-02"),
    )
    assert out, "expected non-empty pin probability"
    total = sum(entry["prob"] for entry in out)
    assert abs(total - 1.0) < 1e-9


def test_atm_strike_gets_highest_probability():
    out = compute_pin_probability(
        _zero_dte_chain(),
        today=pd.Timestamp("2026-01-02"),
    )
    top = out[0]
    assert top["strike"] == 5800.0  # large OI + zero distance


def test_no_zero_dte_returns_empty():
    today = pd.Timestamp("2026-01-02")
    chain = pd.DataFrame([
        {
            "strike": 5800,
            "expiration": (today + pd.Timedelta(days=30)).date(),
            "option_type": "C",
            "iv": 0.18,
            "underlying_price": 5800.0,
            "oi": 1000,
        }
    ])
    assert compute_pin_probability(chain, today=today) == []
