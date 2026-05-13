"""End-to-end correctness tests for the trade-side flow pipeline.

Covers the modules in :mod:`app.processing.lee_ready`,
:mod:`app.processing.hiro`, and :mod:`app.processing.flow_events`. The
suite exercises:

* Lee-Ready edge cases (mid-price + uptick, at-bid, at-ask, missing
  quotes, zero spread, side=0 fall-through).
* HIRO sign convention (customer-buy calls → positive cumulative) and
  the per-bucket reset semantic.
* Flow-event detection (SWEEP / BLOCK / UOA) under the new
  :class:`FlowEventConfig` thresholds.
* Threshold sensitivity (lowering ``flow_sweep_min_premium`` makes more
  trades qualify as sweeps).
* Idempotency: replaying the same trades twice does NOT duplicate events.
"""

from __future__ import annotations

import pandas as pd

from app.processing.flow_events import FlowEventConfig, detect_flow_events
from app.processing.hiro import compute_hiro
from app.processing.lee_ready import classify_lee_ready

# ─── Helpers ───────────────────────────────────────────────────────────────


def _quote_trade(
    ts: int | float | pd.Timestamp,
    price: float,
    bid: float | None,
    ask: float | None,
    size: int = 10,
) -> dict:
    return {"ts": ts, "price": price, "bid": bid, "ask": ask, "size": size}


def _opt_trade(
    ts: str | pd.Timestamp,
    side: int,
    size: int,
    price: float,
    opt: str = "C",
) -> dict:
    return {
        "ts": pd.Timestamp(ts, tz="UTC"),
        "side": side,
        "size": size,
        "price": price,
        "option_type": opt,
    }


def _flow_row(
    *,
    ts: pd.Timestamp,
    size: int = 10,
    price: float = 1.00,
    side: int = 1,
    exchange: str = "CBOE",
    strike: float = 5800.0,
    option_type: str = "C",
    symbol: str = "SPXW",
    expiration: object = pd.Timestamp("2026-01-02").date(),
) -> dict:
    return {
        "ts": ts,
        "symbol": symbol,
        "expiration": expiration,
        "strike": strike,
        "option_type": option_type,
        "price": price,
        "size": size,
        "side": side,
        "exchange": exchange,
    }


# ─── Lee-Ready edge cases ──────────────────────────────────────────────────


def test_lee_ready_trade_at_ask_is_buy() -> None:
    df = pd.DataFrame([_quote_trade(1, price=5.10, bid=5.00, ask=5.10)])
    out = classify_lee_ready(df)
    assert int(out.loc[0, "side"]) == 1


def test_lee_ready_trade_at_bid_is_sell() -> None:
    df = pd.DataFrame([_quote_trade(1, price=5.00, bid=5.00, ask=5.10)])
    out = classify_lee_ready(df)
    assert int(out.loc[0, "side"]) == -1


def test_lee_ready_trade_at_mid_with_prior_uptick_is_buy() -> None:
    """A trade lands exactly on the midpoint — the quote rule cannot
    classify it, so the tick rule falls back on the previous different
    trade price. If that price was below the at-mid trade, the at-mid
    trade is classified as a BUY (uptick)."""
    df = pd.DataFrame(
        [
            _quote_trade(1, price=5.00, bid=5.00, ask=5.10),  # below mid → -1 (sets history)
            _quote_trade(2, price=5.05, bid=5.00, ask=5.10),  # at mid, uptick vs 5.00 → +1
            _quote_trade(3, price=5.05, bid=5.00, ask=5.10),  # at mid, equal to history → 0
        ]
    )
    out = classify_lee_ready(df)
    assert list(out["side"]) == [-1, 1, 0]


def test_lee_ready_missing_quotes_falls_back_to_tick_rule() -> None:
    """Trades whose bid/ask are missing should NOT be classified by the
    quote rule; they fall through to the tick rule."""
    df = pd.DataFrame(
        [
            _quote_trade(1, price=5.00, bid=None, ask=None),  # no quote, no tick history → 0
            _quote_trade(2, price=5.10, bid=None, ask=None),  # uptick vs 5.00 → +1
            _quote_trade(3, price=5.05, bid=None, ask=None),  # downtick vs 5.10 → -1
        ]
    )
    out = classify_lee_ready(df)
    assert list(out["side"]) == [0, 1, -1]


def test_lee_ready_zero_spread_uses_tick_rule() -> None:
    """When bid == ask the quote rule is degenerate; classify by tick."""
    df = pd.DataFrame(
        [
            _quote_trade(1, price=5.00, bid=5.00, ask=5.00),  # no history → 0
            _quote_trade(2, price=5.10, bid=5.10, ask=5.10),  # uptick → +1
            _quote_trade(3, price=5.05, bid=5.05, ask=5.05),  # downtick → -1
        ]
    )
    out = classify_lee_ready(df)
    assert list(out["side"]) == [0, 1, -1]


def test_lee_ready_no_rule_applies_yields_side_zero() -> None:
    """Single mid-price trade with no history and no quotes → unclassified."""
    df = pd.DataFrame([_quote_trade(1, price=5.00, bid=None, ask=None)])
    out = classify_lee_ready(df)
    assert int(out.loc[0, "side"]) == 0
    assert float(out.loc[0, "signed_qty"]) == 0.0


# ─── HIRO ──────────────────────────────────────────────────────────────────


def test_hiro_all_customer_buy_calls_yields_positive_cumulative() -> None:
    """100 % customer-buy calls → dealer is short calls → dealer must buy
    underlying to hedge → strictly POSITIVE HIRO cumulative."""
    df = pd.DataFrame(
        [
            _opt_trade("2026-01-02T14:30:00", side=1, size=10, price=1.00, opt="C"),
            _opt_trade("2026-01-02T14:30:15", side=1, size=20, price=1.50, opt="C"),
            _opt_trade("2026-01-02T14:30:45", side=1, size=5, price=0.75, opt="C"),
        ]
    )
    out = compute_hiro(df, bucket="1min")
    assert len(out.series) == 1
    # 10*100*1 + 20*100*1.5 + 5*100*0.75 = 1000 + 3000 + 375 = 4375
    assert out.cumulative > 0
    assert out.series[0]["cumulative"] == 4375.0
    assert out.series[0]["call_premium"] == 4375.0
    assert out.series[0]["put_premium"] == 0.0


def test_hiro_cumulative_resets_per_bucket() -> None:
    """The per-bucket ``cumulative`` field equals that bucket's net
    premium (i.e. it RESETS at the start of every bucket — it is NOT a
    session-wide running total)."""
    df = pd.DataFrame(
        [
            _opt_trade("2026-01-02T14:30:00", side=1, size=10, price=1.00, opt="C"),  # +1000
            _opt_trade("2026-01-02T14:31:00", side=1, size=20, price=1.00, opt="C"),  # +2000
            _opt_trade("2026-01-02T14:32:00", side=1, size=5, price=2.00, opt="C"),   # +1000
        ]
    )
    out = compute_hiro(df, bucket="1min")
    cums = [b["cumulative"] for b in out.series]
    nets = [b["net_premium"] for b in out.series]
    # Per-bucket cumulative equals per-bucket net (reset across buckets).
    assert cums == nets == [1000.0, 2000.0, 1000.0]
    # HiroSeries.cumulative reports the LAST bucket's signed premium.
    assert out.cumulative == 1000.0


# ─── Flow events: SWEEP ────────────────────────────────────────────────────


def test_sweep_three_legs_within_one_second_above_premium_threshold() -> None:
    """3 legs of the same contract, same side, on distinct venues within
    1 second, with total premium ≥ threshold → exactly one SWEEP event."""
    base = pd.Timestamp("2026-01-02T14:30:00.000Z")
    trades = pd.DataFrame(
        [
            _flow_row(ts=base, size=200, price=1.00, exchange="CBOE"),
            _flow_row(ts=base + pd.Timedelta(milliseconds=200), size=200, price=1.00, exchange="ARCA"),
            _flow_row(ts=base + pd.Timedelta(milliseconds=600), size=200, price=1.00, exchange="ISE"),
        ]
    )
    cfg = FlowEventConfig(
        sweep_window_ms=1000,
        sweep_min_legs=3,
        sweep_min_premium=50_000.0,  # 600 * 1 * 100 = 60_000 ≥ 50_000
        block_min_size=10_000,        # disable BLOCK
        uoa_min_absolute_volume=10_000_000,
        uoa_volume_multiplier=1e9,
    )
    events = detect_flow_events(trades, config=cfg)
    sweeps = [e for e in events if e["event_type"] == "SWEEP"]
    assert len(sweeps) == 1
    assert sweeps[0]["legs"] == 3
    assert sweeps[0]["size"] == 600
    assert sweeps[0]["meta"]["premium"] == 60_000.0
    assert set(sweeps[0]["venues"]) == {"CBOE", "ARCA", "ISE"}


def test_sweep_below_premium_threshold_is_not_flagged() -> None:
    """Same shape as above but total premium below the threshold → no SWEEP."""
    base = pd.Timestamp("2026-01-02T14:30:00.000Z")
    trades = pd.DataFrame(
        [
            _flow_row(ts=base, size=10, price=1.00, exchange="CBOE"),
            _flow_row(ts=base + pd.Timedelta(milliseconds=200), size=10, price=1.00, exchange="ARCA"),
            _flow_row(ts=base + pd.Timedelta(milliseconds=600), size=10, price=1.00, exchange="ISE"),
        ]
    )
    cfg = FlowEventConfig(
        sweep_window_ms=1000,
        sweep_min_legs=3,
        sweep_min_premium=50_000.0,  # 30 * 1 * 100 = 3000 ≪ 50_000
        block_min_size=10_000,
        uoa_min_absolute_volume=10_000_000,
        uoa_volume_multiplier=1e9,
    )
    events = detect_flow_events(trades, config=cfg)
    assert [e for e in events if e["event_type"] == "SWEEP"] == []


def test_lowering_sweep_min_premium_admits_smaller_sweeps() -> None:
    """Same trades, two configs: with the production default the small
    cluster is NOT a sweep; lowering ``sweep_min_premium`` makes it one."""
    base = pd.Timestamp("2026-01-02T14:30:00.000Z")
    trades = pd.DataFrame(
        [
            _flow_row(ts=base, size=10, price=1.00, exchange="CBOE"),
            _flow_row(ts=base + pd.Timedelta(milliseconds=200), size=10, price=1.00, exchange="ARCA"),
            _flow_row(ts=base + pd.Timedelta(milliseconds=600), size=10, price=1.00, exchange="ISE"),
        ]
    )
    strict = FlowEventConfig(
        sweep_window_ms=1000, sweep_min_legs=3, sweep_min_premium=50_000.0,
        block_min_size=10_000, uoa_min_absolute_volume=10_000_000,
        uoa_volume_multiplier=1e9,
    )
    lenient = FlowEventConfig(
        sweep_window_ms=1000, sweep_min_legs=3, sweep_min_premium=100.0,
        block_min_size=10_000, uoa_min_absolute_volume=10_000_000,
        uoa_volume_multiplier=1e9,
    )
    strict_sweeps = [
        e for e in detect_flow_events(trades, config=strict) if e["event_type"] == "SWEEP"
    ]
    lenient_sweeps = [
        e for e in detect_flow_events(trades, config=lenient) if e["event_type"] == "SWEEP"
    ]
    assert len(strict_sweeps) == 0
    assert len(lenient_sweeps) == 1


# ─── Flow events: BLOCK ────────────────────────────────────────────────────


def test_block_single_trade_above_threshold() -> None:
    """A single print at exactly ``flow_block_min_size`` is a BLOCK."""
    base = pd.Timestamp("2026-01-02T14:30:00Z")
    trades = pd.DataFrame([_flow_row(ts=base, size=100, price=2.50)])
    cfg = FlowEventConfig(
        block_min_size=100, sweep_min_premium=1e12,
        uoa_min_absolute_volume=10_000_000, uoa_volume_multiplier=1e9,
    )
    events = detect_flow_events(trades, config=cfg)
    blocks = [e for e in events if e["event_type"] == "BLOCK"]
    assert len(blocks) == 1
    assert blocks[0]["size"] == 100
    assert blocks[0]["price"] == 2.50


def test_block_below_threshold_is_not_flagged() -> None:
    base = pd.Timestamp("2026-01-02T14:30:00Z")
    trades = pd.DataFrame([_flow_row(ts=base, size=99, price=2.50)])
    cfg = FlowEventConfig(
        block_min_size=100, sweep_min_premium=1e12,
        uoa_min_absolute_volume=10_000_000, uoa_volume_multiplier=1e9,
    )
    events = detect_flow_events(trades, config=cfg)
    assert [e for e in events if e["event_type"] == "BLOCK"] == []


# ─── Flow events: UOA ──────────────────────────────────────────────────────


def test_uoa_when_no_sweep_or_block_and_volume_above_absolute() -> None:
    """A contract with elevated volume but no sweep/block prints fires UOA
    via the absolute-volume fallback."""
    base = pd.Timestamp("2026-01-02T14:30:00Z")
    trades = pd.DataFrame(
        [
            _flow_row(ts=base, size=3000, price=1.00),
            _flow_row(ts=base + pd.Timedelta(seconds=1), size=3000, price=1.00),
        ]
    )
    cfg = FlowEventConfig(
        block_min_size=10_000,           # no block
        sweep_min_premium=1e12,          # no sweep
        uoa_min_absolute_volume=5000,
        uoa_volume_multiplier=1e9,
    )
    events = detect_flow_events(trades, config=cfg)
    uoas = [e for e in events if e["event_type"] == "UOA"]
    assert len(uoas) == 1
    assert uoas[0]["size"] == 6000
    assert uoas[0]["meta"]["method"] == "absolute"


def test_uoa_uses_vol_oi_ratio_when_adv_missing() -> None:
    """With no ADV row but OI known, the vol/OI ratio gate fires UOA."""
    base = pd.Timestamp("2026-01-02T14:30:00Z")
    trades = pd.DataFrame([_flow_row(ts=base, size=300, price=1.00)])
    oi = pd.DataFrame(
        [
            {
                "symbol": "SPXW",
                "expiration": pd.Timestamp("2026-01-02").date(),
                "strike": 5800.0,
                "option_type": "C",
                "open_interest": 100,
            }
        ]
    )
    cfg = FlowEventConfig(
        block_min_size=10_000, sweep_min_premium=1e12,
        uoa_min_absolute_volume=10_000_000,    # disable absolute fallback
        uoa_volume_multiplier=1e9,             # disable ADV branch (no ADV anyway)
        uoa_vol_oi_ratio=2.0,                  # 300 / 100 = 3.0 ≥ 2.0
    )
    events = detect_flow_events(trades, contract_oi=oi, config=cfg)
    uoas = [e for e in events if e["event_type"] == "UOA"]
    assert len(uoas) == 1
    assert uoas[0]["meta"]["method"] == "vol_oi_ratio"
    assert uoas[0]["meta"]["ratio"] == 3.0


def test_uoa_prefers_adv_over_oi_when_both_available() -> None:
    """ADV takes precedence: if a contract has both an ADV row and an OI
    row, the UOA decision uses ADV × multiplier (the OI gate is skipped)."""
    base = pd.Timestamp("2026-01-02T14:30:00Z")
    trades = pd.DataFrame([_flow_row(ts=base, size=400, price=1.00)])
    adv = pd.DataFrame(
        [
            {
                "symbol": "SPXW",
                "expiration": pd.Timestamp("2026-01-02").date(),
                "strike": 5800.0,
                "option_type": "C",
                "avg_daily_volume": 50.0,
            }
        ]
    )
    oi = pd.DataFrame(
        [
            {
                "symbol": "SPXW",
                "expiration": pd.Timestamp("2026-01-02").date(),
                "strike": 5800.0,
                "option_type": "C",
                "open_interest": 100,
            }
        ]
    )
    cfg = FlowEventConfig(
        block_min_size=10_000, sweep_min_premium=1e12,
        uoa_min_absolute_volume=10_000_000,
        uoa_volume_multiplier=5.0,
        uoa_vol_oi_ratio=2.0,
    )
    events = detect_flow_events(trades, contract_adv=adv, contract_oi=oi, config=cfg)
    uoas = [e for e in events if e["event_type"] == "UOA"]
    assert len(uoas) == 1
    assert uoas[0]["meta"]["method"] == "adv"


def test_uoa_suppressed_when_sweep_or_block_already_on_contract() -> None:
    """UOA is a *residual* signal: if a BLOCK was already detected on a
    contract, we don't double-fire UOA on the same contract."""
    base = pd.Timestamp("2026-01-02T14:30:00Z")
    trades = pd.DataFrame(
        [
            _flow_row(ts=base, size=500, price=1.00),  # → BLOCK at 500 ≥ 100
            _flow_row(ts=base + pd.Timedelta(seconds=1), size=5000, price=1.00),
        ]
    )
    cfg = FlowEventConfig(
        block_min_size=100, sweep_min_premium=1e12,
        uoa_min_absolute_volume=1000, uoa_volume_multiplier=1e9,
    )
    events = detect_flow_events(trades, config=cfg)
    assert any(e["event_type"] == "BLOCK" for e in events)
    assert [e for e in events if e["event_type"] == "UOA"] == []


# ─── Threshold-respect smoke test ──────────────────────────────────────────


def test_threshold_respect_block_size() -> None:
    """A trade at ``block_min_size - 1`` is not a block, raising the
    threshold by 1 prevents the previously-flagged trade from firing."""
    base = pd.Timestamp("2026-01-02T14:30:00Z")
    trades = pd.DataFrame([_flow_row(ts=base, size=100, price=1.00)])
    permissive = FlowEventConfig(
        block_min_size=100, sweep_min_premium=1e12,
        uoa_min_absolute_volume=10_000_000, uoa_volume_multiplier=1e9,
    )
    strict = FlowEventConfig(
        block_min_size=101, sweep_min_premium=1e12,
        uoa_min_absolute_volume=10_000_000, uoa_volume_multiplier=1e9,
    )
    assert any(
        e["event_type"] == "BLOCK"
        for e in detect_flow_events(trades, config=permissive)
    )
    assert not any(
        e["event_type"] == "BLOCK"
        for e in detect_flow_events(trades, config=strict)
    )


# ─── Idempotency ───────────────────────────────────────────────────────────


def test_idempotency_duplicate_trade_rows_do_not_duplicate_events() -> None:
    """Feeding the exact same trade row twice (e.g. a re-tick from the
    live feed) must NOT produce duplicate events."""
    base = pd.Timestamp("2026-01-02T14:30:00Z")
    one = _flow_row(ts=base, size=500, price=1.00)
    trades = pd.DataFrame([one, dict(one)])  # exact duplicate
    cfg = FlowEventConfig(
        block_min_size=100, sweep_min_premium=1e12,
        uoa_min_absolute_volume=10_000_000, uoa_volume_multiplier=1e9,
    )
    events = detect_flow_events(trades, config=cfg)
    blocks = [e for e in events if e["event_type"] == "BLOCK"]
    assert len(blocks) == 1
    assert blocks[0]["size"] == 500  # not 1000 — the duplicate is dropped


def test_idempotency_repeated_calls_are_stable() -> None:
    """Calling :func:`detect_flow_events` twice on the same DataFrame
    produces the same events (the detector is stateless)."""
    base = pd.Timestamp("2026-01-02T14:30:00Z")
    trades = pd.DataFrame(
        [
            _flow_row(ts=base, size=200, price=1.00, exchange="CBOE"),
            _flow_row(ts=base + pd.Timedelta(milliseconds=200), size=200, price=1.00, exchange="ARCA"),
            _flow_row(ts=base + pd.Timedelta(milliseconds=600), size=200, price=1.00, exchange="ISE"),
        ]
    )
    cfg = FlowEventConfig(
        sweep_window_ms=1000, sweep_min_legs=3, sweep_min_premium=50_000.0,
        block_min_size=10_000, uoa_min_absolute_volume=10_000_000,
        uoa_volume_multiplier=1e9,
    )
    a = detect_flow_events(trades, config=cfg)
    b = detect_flow_events(trades, config=cfg)
    assert a == b
