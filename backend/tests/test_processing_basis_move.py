"""Basis tracker + Realized vs Implied move tests."""

from __future__ import annotations

import pandas as pd

from app.processing.basis import compute_basis
from app.processing.move_tracker import compute_move_tracker


def test_basis_simple():
    out = compute_basis(spot=5800.0, futures=5810.0)
    assert out.basis == 10.0
    assert out.basis_pct is not None
    assert abs(out.basis_pct - 10.0 / 5800.0) < 1e-9


def test_basis_handles_missing_inputs():
    out = compute_basis(spot=None, futures=5810.0)
    assert out.basis is None
    out = compute_basis(spot=0.0, futures=5810.0)
    assert out.basis is None


def test_move_tracker_returns_implied_only_when_open_missing():
    today = pd.Timestamp("2026-01-02")
    chain = pd.DataFrame([
        {"strike": 5800, "expiration": today.date(), "option_type": "C",
         "last_price": 30.0, "underlying_price": 5800.0},
        {"strike": 5800, "expiration": today.date(), "option_type": "P",
         "last_price": 28.0, "underlying_price": 5800.0},
    ])
    out = compute_move_tracker(chain, open_price=None, today=today)
    assert out.implied_move is not None
    # 0DTE -> 1-day floor -> implied_total / sqrt(1) -> 58.0
    assert out.implied_move == 58.0
    assert out.realized_move is None
    assert out.ratio is None


def test_move_tracker_computes_realized_and_ratio():
    today = pd.Timestamp("2026-01-02")
    chain = pd.DataFrame([
        {"strike": 5800, "expiration": today.date(), "option_type": "C",
         "last_price": 30.0, "underlying_price": 5810.0},
        {"strike": 5800, "expiration": today.date(), "option_type": "P",
         "last_price": 28.0, "underlying_price": 5810.0},
    ])
    out = compute_move_tracker(chain, open_price=5800.0, today=today)
    assert out.realized_move == 10.0
    assert out.implied_move == 58.0
    assert abs(out.ratio - (10.0 / 58.0)) < 1e-6
