"""BBO cache + Lee-Ready integration (Rev 4 Agent 4)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from app.processing.bbo_cache import (
    DEFAULT_MAX_AGE,
    InMemoryBboCache,
    enrich_with_bbo,
)
from app.processing.lee_ready import (
    TickRuleState,
    classify_lee_ready,
    classify_lee_ready_with_bbo,
    quote_or_tick_rule,
)

# ── enrich_with_bbo (as-of join) ─────────────────────────────────────────


def _ts(seconds: float) -> pd.Timestamp:
    """Build a UTC timestamp at the given offset from a fixed anchor."""
    return pd.Timestamp("2026-09-01 14:30:00", tz="UTC") + pd.Timedelta(seconds=seconds)


def _trade(ts: float, symbol: str, price: float, size: int = 10) -> dict:
    return {
        "ts": _ts(ts),
        "symbol": symbol,
        "expiration": pd.Timestamp("2026-09-01"),
        "strike": 5000.0,
        "option_type": "C",
        "price": price,
        "size": size,
    }


def _bbo(ts: float, symbol: str, bid: float, ask: float) -> dict:
    return {
        "ts": _ts(ts),
        "symbol": symbol,
        "expiration": pd.Timestamp("2026-09-01"),
        "strike": 5000.0,
        "option_type": "C",
        "bid_px": bid,
        "ask_px": ask,
    }


def test_enrich_with_bbo_attaches_latest_snapshot() -> None:
    trades = pd.DataFrame([_trade(0.5, "SPXW", 1.05), _trade(1.5, "SPXW", 1.07)])
    bbo = pd.DataFrame(
        [
            _bbo(0.0, "SPXW", 1.00, 1.10),
            _bbo(1.0, "SPXW", 1.02, 1.12),
        ]
    )
    out = enrich_with_bbo(trades, bbo)
    # Trade @ 0.5s -> BBO from 0.0s ; trade @ 1.5s -> BBO from 1.0s.
    assert out["bid"].tolist() == [1.00, 1.02]
    assert out["ask"].tolist() == [1.10, 1.12]
    assert out["mid"].tolist() == pytest.approx([1.05, 1.07])
    assert out["bbo_age_seconds"].tolist() == pytest.approx([0.5, 0.5])


def test_enrich_with_bbo_drops_stale_quotes() -> None:
    trades = pd.DataFrame([_trade(60.0, "SPXW", 1.05)])
    bbo = pd.DataFrame([_bbo(0.0, "SPXW", 1.00, 1.10)])
    out = enrich_with_bbo(trades, bbo, max_age=timedelta(seconds=5))
    assert out["bid"].isna().all()
    assert out["ask"].isna().all()
    assert out["mid"].isna().all()
    assert out["bbo_age_seconds"].isna().all()


def test_enrich_with_bbo_partitions_by_contract() -> None:
    trades = pd.DataFrame(
        [
            {**_trade(0.5, "SPXW", 1.05), "strike": 5000.0},
            {**_trade(0.5, "SPXW", 0.95), "strike": 4950.0},
        ]
    )
    bbo = pd.DataFrame(
        [
            {**_bbo(0.0, "SPXW", 1.00, 1.10), "strike": 5000.0},
            {**_bbo(0.0, "SPXW", 0.90, 1.00), "strike": 4950.0},
        ]
    )
    out = enrich_with_bbo(trades, bbo)
    # Each trade picks up only its own strike's BBO.
    assert out.loc[out["strike"] == 5000.0, "bid"].iloc[0] == 1.00
    assert out.loc[out["strike"] == 4950.0, "bid"].iloc[0] == 0.90


def test_enrich_with_bbo_empty_bbo_returns_passthrough() -> None:
    trades = pd.DataFrame([_trade(0.5, "SPXW", 1.05)])
    bbo = pd.DataFrame(columns=["ts", "symbol", "expiration", "strike", "option_type", "bid_px", "ask_px"])
    out = enrich_with_bbo(trades, bbo)
    assert out["bid"].isna().all()
    assert out["mid"].isna().all()
    # Other columns preserved.
    assert out["price"].tolist() == [1.05]


def test_enrich_with_bbo_accepts_already_renamed_columns() -> None:
    trades = pd.DataFrame([_trade(0.5, "SPXW", 1.05)])
    bbo = pd.DataFrame(
        [
            {**_bbo(0.0, "SPXW", 1.00, 1.10)},
        ]
    ).rename(columns={"bid_px": "bid", "ask_px": "ask"})
    out = enrich_with_bbo(trades, bbo)
    assert out["bid"].iloc[0] == 1.00
    assert out["ask"].iloc[0] == 1.10


def test_enrich_with_bbo_requires_ts() -> None:
    with pytest.raises(KeyError):
        enrich_with_bbo(pd.DataFrame({"symbol": ["SPXW"]}), pd.DataFrame())


# ── classify_lee_ready_with_bbo (integration) ────────────────────────────


def test_classify_lee_ready_with_bbo_end_to_end() -> None:
    trades = pd.DataFrame(
        [
            _trade(0.5, "SPXW", 1.10),  # above mid -> +1
            _trade(1.5, "SPXW", 1.02),  # below mid -> -1
            _trade(2.5, "SPXW", 1.07),  # at mid -> tick rule (prev=1.02) -> +1
        ]
    )
    bbo = pd.DataFrame(
        [
            _bbo(0.0, "SPXW", 1.00, 1.10),
            _bbo(1.0, "SPXW", 1.02, 1.12),
            _bbo(2.0, "SPXW", 1.04, 1.10),
        ]
    )
    out = classify_lee_ready_with_bbo(trades, bbo)
    assert out["side"].tolist() == [1, -1, 1]


# ── InMemoryBboCache (live-path cache) ───────────────────────────────────


def test_in_memory_cache_returns_fresh_quote() -> None:
    cache = InMemoryBboCache()
    base = datetime(2026, 9, 1, 14, 30, tzinfo=UTC)
    cache.update(42, base, 1.00, 1.10)
    hit = cache.at(42, base + timedelta(milliseconds=500))
    assert hit is not None
    bid, ask, age = hit
    assert bid == 1.00 and ask == 1.10
    assert age == pytest.approx(0.5)


def test_in_memory_cache_rejects_stale() -> None:
    cache = InMemoryBboCache(max_age=timedelta(seconds=1))
    base = datetime(2026, 9, 1, 14, 30, tzinfo=UTC)
    cache.update(42, base, 1.00, 1.10)
    hit = cache.at(42, base + timedelta(seconds=5))
    assert hit is None


def test_in_memory_cache_rejects_invalid_quotes() -> None:
    cache = InMemoryBboCache()
    base = datetime(2026, 9, 1, 14, 30, tzinfo=UTC)
    cache.update(42, base, None, 1.10)  # missing bid -> drop
    cache.update(42, base, 1.50, 1.40)  # ask < bid -> drop
    cache.update(42, base, 0.0, 1.10)  # non-positive bid -> drop
    assert cache.at(42, base) is None


def test_in_memory_cache_returns_none_for_future_trade() -> None:
    cache = InMemoryBboCache()
    base = datetime(2026, 9, 1, 14, 30, tzinfo=UTC)
    cache.update(42, base + timedelta(seconds=2), 1.00, 1.10)
    # Trade arrived *before* the cached BBO — treat as miss.
    assert cache.at(42, base) is None


# ── quote_or_tick_rule (single-record API used by the live ingester) ──────


def test_quote_or_tick_rule_uses_quote_when_spread_present() -> None:
    state = TickRuleState()
    side = quote_or_tick_rule(
        price=1.10, bid=1.00, ask=1.10, tick_state=state, instrument_id=1
    )
    # at-ask -> price > mid -> +1
    assert side == 1


def test_quote_or_tick_rule_falls_back_to_tick_when_spread_zero() -> None:
    state = TickRuleState()
    # Seed history with a prior trade (no BBO).
    quote_or_tick_rule(price=1.00, bid=None, ask=None, tick_state=state, instrument_id=1)
    side = quote_or_tick_rule(
        price=1.05, bid=1.05, ask=1.05, tick_state=state, instrument_id=1
    )
    # zero spread -> tick rule -> 1.05 > 1.00 -> +1
    assert side == 1


def test_quote_or_tick_rule_first_trade_with_no_bbo_unclassified() -> None:
    state = TickRuleState()
    side = quote_or_tick_rule(
        price=1.05, bid=None, ask=None, tick_state=state, instrument_id=1
    )
    # No quote, no history -> unclassified.
    assert side == 0


def test_quote_or_tick_rule_keeps_tick_history_after_quote_hit() -> None:
    state = TickRuleState()
    # Quote-rule hits first (records 1.05 in tick history).
    quote_or_tick_rule(
        price=1.05, bid=1.00, ask=1.10, tick_state=state, instrument_id=1
    )
    # Subsequent zero-spread trade can now use tick rule.
    side = quote_or_tick_rule(
        price=1.06, bid=1.06, ask=1.06, tick_state=state, instrument_id=1
    )
    assert side == 1


# ── classify_lee_ready stays unchanged for callers that don't use BBO ────


def test_classify_lee_ready_backward_compatible() -> None:
    df = pd.DataFrame(
        [
            {"ts": 1, "price": 1.10, "bid": 1.00, "ask": 1.10},
            {"ts": 2, "price": 1.00, "bid": 1.00, "ask": 1.10},
        ]
    )
    out = classify_lee_ready(df)
    assert out["side"].tolist() == [1, -1]


def test_default_max_age_constant_advertised() -> None:
    # bbo-1s is sampled at 1 Hz; default tolerance must cover at least 1s.
    assert DEFAULT_MAX_AGE >= timedelta(seconds=1)
