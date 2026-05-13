"""Tests for the regime score computation."""

from __future__ import annotations

from app.processing.gex import GexSummary
from app.processing.regime import compute_regime
from app.processing.walls import WallsSummary


def _gex(net_total: float, gross: float = 100.0) -> GexSummary:
    """Build a GexSummary with a single curve point summing to ``net_total``."""
    return GexSummary(
        underlying_price=100.0,
        net_total=net_total,
        curve=[
            {
                "strike": 100.0,
                "call_gex": max(net_total, 0.0) + gross / 2,
                "put_gex": -max(-net_total, 0.0) - gross / 2,
                "net_gex": net_total,
            }
        ],
        top_positive=[],
        top_negative=[],
    )


def test_regime_bullish_when_calls_dominate_walls():
    walls = WallsSummary(
        by_oi={
            "call_wall": [{"strike": 105, "value": 8000}],
            "put_wall": [{"strike": 95, "value": 1000}],
        },
        by_volume={
            "call_wall": [{"strike": 105, "value": 8000}],
            "put_wall": [{"strike": 95, "value": 1000}],
        },
    )
    summary = compute_regime(walls, _gex(50.0), _gex(50.0))
    assert summary.oi.label == "bullish"
    assert summary.vol.label == "bullish"
    assert summary.oi.score > 0.2


def test_regime_bearish_when_puts_dominate():
    walls = WallsSummary(
        by_oi={
            "call_wall": [{"strike": 105, "value": 1000}],
            "put_wall": [{"strike": 95, "value": 8000}],
        },
        by_volume={
            "call_wall": [{"strike": 105, "value": 1000}],
            "put_wall": [{"strike": 95, "value": 8000}],
        },
    )
    summary = compute_regime(walls, _gex(-50.0), _gex(-50.0))
    assert summary.oi.label == "bearish"
    assert summary.vol.label == "bearish"
    assert summary.oi.score < -0.2


def test_regime_neutral_when_walls_balanced_and_no_gex():
    walls = WallsSummary(
        by_oi={
            "call_wall": [{"strike": 105, "value": 1000}],
            "put_wall": [{"strike": 95, "value": 1000}],
        },
        by_volume={
            "call_wall": [{"strike": 105, "value": 1000}],
            "put_wall": [{"strike": 95, "value": 1000}],
        },
    )
    flat = GexSummary(
        underlying_price=100.0,
        net_total=0.0,
        curve=[],
        top_positive=[],
        top_negative=[],
    )
    summary = compute_regime(walls, flat, flat)
    assert summary.oi.label == "neutral"
    assert summary.vol.label == "neutral"
    assert abs(summary.oi.score) < 0.2


def test_regime_handles_empty_walls():
    walls = WallsSummary(by_oi={}, by_volume={})
    flat = GexSummary(
        underlying_price=100.0, net_total=0.0, curve=[], top_positive=[], top_negative=[]
    )
    summary = compute_regime(walls, flat, flat)
    assert summary.oi.score == 0.0
    assert summary.oi.label == "neutral"
    assert summary.oi.call_wall_total == 0.0
    assert summary.oi.put_wall_total == 0.0
