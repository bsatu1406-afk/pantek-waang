"""Tests for call/put wall detection."""

from __future__ import annotations

import pandas as pd

from app.processing.walls import compute_walls


def test_walls_top_strikes_by_oi_and_volume():
    rows = [
        {"strike": 100, "option_type": "C", "oi": 1000, "volume": 50},
        {"strike": 105, "option_type": "C", "oi": 5000, "volume": 200},  # call wall by OI
        {"strike": 110, "option_type": "C", "oi": 200, "volume": 800},  # call wall by volume
        {"strike": 95, "option_type": "P", "oi": 6000, "volume": 100},  # put wall by OI
        {"strike": 90, "option_type": "P", "oi": 200, "volume": 1500},  # put wall by volume
    ]
    df = pd.DataFrame(rows)
    summary = compute_walls(df, top_n=3)
    assert summary.by_oi["call_wall"][0]["strike"] == 105
    assert summary.by_volume["call_wall"][0]["strike"] == 110
    assert summary.by_oi["put_wall"][0]["strike"] == 95
    assert summary.by_volume["put_wall"][0]["strike"] == 90


def test_walls_aggregates_duplicates():
    rows = [
        {"strike": 100, "option_type": "C", "oi": 500, "volume": 0},
        {"strike": 100, "option_type": "C", "oi": 500, "volume": 0},
        {"strike": 105, "option_type": "C", "oi": 800, "volume": 0},
    ]
    df = pd.DataFrame(rows)
    summary = compute_walls(df, top_n=2)
    # Two rows at strike 100 should be summed -> 1000 > 800 at 105.
    assert summary.by_oi["call_wall"][0]["strike"] == 100
    assert summary.by_oi["call_wall"][0]["value"] == 1000


def test_walls_handles_empty():
    summary = compute_walls(pd.DataFrame())
    assert summary.by_oi == {}
    assert summary.by_volume == {}
