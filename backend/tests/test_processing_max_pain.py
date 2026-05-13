"""Tests for max-pain computation."""

from __future__ import annotations

import pandas as pd

from app.processing.max_pain import compute_max_pain


def test_max_pain_picks_minimum_loss_strike():
    """A symmetric chain centered on 100 should peg max pain at 100."""
    today = pd.Timestamp.utcnow().normalize()
    expiry = today + pd.Timedelta(days=7)
    strikes = [90, 95, 100, 105, 110]
    rows = []
    for K in strikes:
        rows.append(
            {"expiration": expiry, "strike": K, "option_type": "C", "oi": 1000, "volume": 0}
        )
        rows.append(
            {"expiration": expiry, "strike": K, "option_type": "P", "oi": 1000, "volume": 0}
        )
    df = pd.DataFrame(rows)
    summary = compute_max_pain(df)
    assert len(summary.per_expiry) == 1
    assert summary.per_expiry[0]["strike"] == 100


def test_max_pain_per_expiry_independent():
    today = pd.Timestamp.utcnow().normalize()
    e1 = today + pd.Timedelta(days=7)
    e2 = today + pd.Timedelta(days=14)
    rows = []
    # Expiry 1: heavy call OI at 105 -> max pain skews lower
    for K, oi_c, oi_p in [(95, 100, 100), (100, 100, 100), (105, 5000, 100)]:
        rows.append({"expiration": e1, "strike": K, "option_type": "C", "oi": oi_c, "volume": 0})
        rows.append({"expiration": e1, "strike": K, "option_type": "P", "oi": oi_p, "volume": 0})
    # Expiry 2: heavy put OI at 95 -> max pain skews higher
    for K, oi_c, oi_p in [(95, 100, 5000), (100, 100, 100), (105, 100, 100)]:
        rows.append({"expiration": e2, "strike": K, "option_type": "C", "oi": oi_c, "volume": 0})
        rows.append({"expiration": e2, "strike": K, "option_type": "P", "oi": oi_p, "volume": 0})
    df = pd.DataFrame(rows)
    summary = compute_max_pain(df)
    assert len(summary.per_expiry) == 2
    by_expiry = {e["expiration"]: e["strike"] for e in summary.per_expiry}
    # First expiry: heavy call OI at 105 means call holders lose more if S > 105,
    # so max pain (min loss to all holders) is at or below 100.
    assert by_expiry[str(e1.date())] <= 100
    # Second expiry: heavy put OI at 95 means put holders lose more if S < 95,
    # so max pain is at or above 100.
    assert by_expiry[str(e2.date())] >= 100
    assert summary.aggregate_strike is not None


def test_max_pain_empty_dataframe():
    summary = compute_max_pain(pd.DataFrame())
    assert summary.per_expiry == []
    assert summary.aggregate_strike is None
