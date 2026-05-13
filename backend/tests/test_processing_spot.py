"""Tests for put-call parity spot synthesis."""

from datetime import date

import pandas as pd

from app.processing.spot import synthesize_underlying_price


def _build_chain(spot: float = 7000.0, expiry=date(2026, 12, 31)) -> pd.DataFrame:
    """Build a stylised SPX chain with mids consistent with ``spot``."""
    rows = []
    # At-the-money strike — put/call near parity.
    rows.append(
        {"strike": spot, "expiration": expiry, "option_type": "C", "bid": 19.95, "ask": 20.05}
    )
    rows.append(
        {"strike": spot, "expiration": expiry, "option_type": "P", "bid": 19.95, "ask": 20.05}
    )
    # ITM call / OTM put 50 below.
    rows.append(
        {"strike": spot - 50, "expiration": expiry, "option_type": "C", "bid": 60.0, "ask": 60.4}
    )
    rows.append(
        {"strike": spot - 50, "expiration": expiry, "option_type": "P", "bid": 9.9, "ask": 10.1}
    )
    return pd.DataFrame(rows)


def test_synthesize_returns_value_close_to_spot():
    df = _build_chain(spot=7000.0)
    out = synthesize_underlying_price(df, risk_free_rate=0.05)
    assert out is not None
    # Allow a couple percent tolerance — synthetic spot via parity drifts
    # with discount factor across long-dated expiries.
    assert abs(out - 7000.0) / 7000.0 < 0.05


def test_synthesize_returns_none_for_empty():
    assert synthesize_underlying_price(pd.DataFrame(), risk_free_rate=0.05) is None


def test_synthesize_returns_none_when_only_calls():
    df = _build_chain()
    df = df[df["option_type"] == "C"].copy()
    assert synthesize_underlying_price(df, risk_free_rate=0.05) is None


def test_synthesize_falls_back_to_last_price():
    """When bid/ask are absent (e.g. cmbp-1 not subscribed) but trade
    prints exist, parity should still produce a usable spot."""
    df = _build_chain(spot=7000.0)
    df = df.assign(last_price=lambda d: (d["bid"] + d["ask"]) / 2.0)
    df["bid"] = pd.NA
    df["ask"] = pd.NA
    out = synthesize_underlying_price(df, risk_free_rate=0.05)
    assert out is not None
    assert abs(out - 7000.0) / 7000.0 < 0.05


def test_synthesize_returns_none_when_no_quotes_at_all():
    df = _build_chain(spot=7000.0)
    df["bid"] = pd.NA
    df["ask"] = pd.NA
    # No last_price either.
    assert synthesize_underlying_price(df, risk_free_rate=0.05) is None
