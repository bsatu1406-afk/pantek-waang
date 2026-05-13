"""Lee-Ready classifier sanity tests."""

from __future__ import annotations

import pandas as pd

from app.processing.lee_ready import classify_lee_ready


def _trade(ts: int, price: float, bid: float, ask: float, size: int = 10):
    return {"ts": ts, "price": price, "bid": bid, "ask": ask, "size": size}


def test_quote_rule_classifies_above_below_mid():
    df = pd.DataFrame([
        _trade(1, 5.10, 5.00, 5.10),  # at-ask, bid<ask -> price>mid -> +1
        _trade(2, 5.00, 5.00, 5.10),  # at-bid -> -1
    ])
    out = classify_lee_ready(df)
    assert list(out["side"]) == [1, -1]


def test_tick_rule_resolves_midpoint_trades():
    df = pd.DataFrame([
        _trade(1, 5.05, 5.00, 5.10),  # at mid -> needs tick rule, no history -> 0
        _trade(2, 5.05, 5.00, 5.10),  # equal to last -> still 0
        _trade(3, 5.04, 5.00, 5.10),  # below last different (5.05) -> -1
        _trade(4, 5.05, 5.00, 5.10),  # above last (5.04) -> +1
    ])
    out = classify_lee_ready(df)
    assert list(out["side"]) == [0, 0, -1, 1]


def test_signed_qty_uses_size():
    df = pd.DataFrame([
        _trade(1, 5.10, 5.00, 5.10, size=7),
        _trade(2, 5.00, 5.00, 5.10, size=11),
    ])
    out = classify_lee_ready(df)
    assert list(out["signed_qty"]) == [7.0, -11.0]


def test_empty_input_returns_typed_empty():
    out = classify_lee_ready(pd.DataFrame(columns=["price", "bid", "ask"]))
    assert list(out.columns) >= ["mid", "side", "signed_qty"]
    assert out.empty
