"""HIRO signed-premium aggregator tests.

HIRO uses the **underlying-hedge-flow** convention:

* Customer-buy CALL  → dealer-short call  → dealer buys underlying  → +HIRO
* Customer-buy PUT   → dealer-short put   → dealer sells underlying → -HIRO

The mirror holds for customer sells.
"""

from __future__ import annotations

import pandas as pd

from app.processing.hiro import compute_hiro


def _trade(ts: str, side: int, size: int, price: float, opt: str):
    return {
        "ts": pd.Timestamp(ts, tz="UTC"),
        "side": side,
        "size": size,
        "price": price,
        "option_type": opt,
    }


def test_customer_buy_call_is_positive_hedge_flow():
    """A customer BUY of 1 call @ $1.00 (size=10) means the dealer is
    short and must buy the underlying to hedge → positive hedge flow of
    +10 * 100 * 1.00 = +1000."""
    df = pd.DataFrame([_trade("2026-01-02T14:30:00", side=1, size=10, price=1.00, opt="C")])
    out = compute_hiro(df, bucket="1min")
    assert len(out.series) == 1
    bucket = out.series[0]
    assert bucket["call_premium"] == 1000.0
    assert bucket["put_premium"] == 0.0
    assert bucket["net_premium"] == 1000.0
    # Per-bucket reset: cumulative == net for a single-bucket window.
    assert bucket["cumulative"] == 1000.0
    assert out.cumulative == 1000.0


def test_calls_and_puts_separated():
    df = pd.DataFrame([
        _trade("2026-01-02T14:30:00", side=1, size=10, price=1.00, opt="C"),  # +1000 hedge
        _trade("2026-01-02T14:30:01", side=-1, size=5, price=2.00, opt="P"),  # +1000 hedge
    ])
    out = compute_hiro(df, bucket="1min")
    assert out.series[0]["call_premium"] == 1000.0
    # Customer sell of put → dealer long put → dealer buys underlying → +HIRO.
    assert out.series[0]["put_premium"] == 1000.0
    assert out.series[0]["net_premium"] == 2000.0


def test_unclassified_trades_excluded():
    df = pd.DataFrame([
        _trade("2026-01-02T14:30:00", side=0, size=10, price=1.00, opt="C"),
    ])
    out = compute_hiro(df)
    assert out.series == []
    assert out.cumulative == 0.0


def test_empty_input():
    out = compute_hiro(pd.DataFrame(), bucket="1min")
    assert out.series == []
    assert out.cumulative == 0.0
