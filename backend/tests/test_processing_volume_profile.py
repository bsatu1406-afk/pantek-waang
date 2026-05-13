"""ES volume profile tests."""

from __future__ import annotations

import pandas as pd

from app.processing.volume_profile import compute_volume_profile


def test_poc_at_highest_volume_bin():
    trades = pd.DataFrame([
        {"price": 5800.00, "size": 5},
        {"price": 5800.00, "size": 5},
        {"price": 5800.25, "size": 1},
        {"price": 5800.50, "size": 1},
    ])
    profile = compute_volume_profile(trades, bin_size=0.25, value_area_pct=0.70)
    assert profile.poc == 5800.00
    assert profile.total_volume == 12


def test_value_area_envelopes_70pct():
    trades = pd.DataFrame(
        [{"price": 5800 + (i % 5) * 0.25, "size": 1} for i in range(100)]
    )
    profile = compute_volume_profile(trades, bin_size=0.25, value_area_pct=0.70)
    assert profile.poc is not None
    # VAH/VAL should bracket the POC.
    assert profile.val <= profile.poc <= profile.vah


def test_buy_sell_split_on_side_column():
    trades = pd.DataFrame([
        {"price": 5800.00, "size": 5, "side": 1},
        {"price": 5800.00, "size": 3, "side": -1},
        {"price": 5800.25, "size": 4, "side": 1},
    ])
    profile = compute_volume_profile(trades, bin_size=0.25)
    bin_5800 = next(b for b in profile.bins if b["price"] == 5800.00)
    assert bin_5800["buy"] == 5
    assert bin_5800["sell"] == 3


def test_empty_input():
    profile = compute_volume_profile(pd.DataFrame(columns=["price", "size"]))
    assert profile.bins == []
    assert profile.poc is None
