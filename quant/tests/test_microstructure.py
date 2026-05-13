"""Unit tests for the Microstructure specialist (Agent #5).

Uses synthetic bars + TBBO frames so the tests are hermetic — no Databento,
no disk loader. Each detector has at least one dedicated test, plus tests
for throttling, the time gate, and the empty-data edge case.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import polars as pl

from src.common.types import Side
from src.strategy.specialists.microstructure import (
    MicrostructureParams,
    _generate_signals_from_data,
    _resample_tbbo_to_1s,
)

SD = date(2024, 5, 1)
# Session-open time in UTC matches 09:30 ET on 2024-05-01 (EDT, UTC-4)
OPEN_UTC = datetime(2024, 5, 1, 13, 30, tzinfo=UTC)
CLOSE_UTC = datetime(2024, 5, 1, 20, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Synthetic frame builders
# ---------------------------------------------------------------------------


def _make_bars(
    n_seconds: int = 3600,
    start_ts: datetime = OPEN_UTC,
    start_price: float = 5000.0,
    step: float = 0.0,
) -> pl.DataFrame:
    rows = []
    px = start_price
    for i in range(n_seconds):
        ts = start_ts + timedelta(seconds=i)
        rows.append({
            "ts": ts,
            "session_date": SD,
            "open": px,
            "high": px + 0.25,
            "low": px - 0.25,
            "close": px,
            "volume": 10,
        })
        px += step
    return pl.DataFrame(
        rows,
        schema={
            "ts": pl.Datetime("us", "UTC"),
            "session_date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        },
    )


def _tbbo_schema() -> dict:
    return {
        "ts": pl.Datetime("us", "UTC"),
        "session_date": pl.Date,
        "bid_px": pl.Float64,
        "bid_sz": pl.Int64,
        "ask_px": pl.Float64,
        "ask_sz": pl.Int64,
        "price": pl.Float64,
        "size": pl.Int64,
        "side": pl.Utf8,
        "action": pl.Utf8,
    }


def _quote_row(ts, bid_px, bid_sz, ask_px, ask_sz):
    return {
        "ts": ts, "session_date": SD,
        "bid_px": bid_px, "bid_sz": bid_sz,
        "ask_px": ask_px, "ask_sz": ask_sz,
        "price": None, "size": None, "side": "N", "action": "A",
    }


def _trade_row(ts, price, size, side, bid_px=5000.0, ask_px=5000.25,
               bid_sz=20, ask_sz=20):
    return {
        "ts": ts, "session_date": SD,
        "bid_px": bid_px, "bid_sz": bid_sz,
        "ask_px": ask_px, "ask_sz": ask_sz,
        "price": price, "size": size, "side": side, "action": "T",
    }


def _df(rows):
    return pl.DataFrame(rows, schema=_tbbo_schema())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_bars_returns_no_signals() -> None:
    bars = pl.DataFrame(
        [],
        schema={
            "ts": pl.Datetime("us", "UTC"),
            "session_date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        },
    )
    tbbo = _df([])
    out = _generate_signals_from_data(bars, tbbo, MicrostructureParams())
    assert out == []


def test_empty_tbbo_returns_no_signals() -> None:
    bars = _make_bars(60)
    tbbo = _df([])
    out = _generate_signals_from_data(bars, tbbo, MicrostructureParams())
    assert out == []


def test_resample_to_1s_carries_forward_quote_and_aggregates_trades() -> None:
    t0 = OPEN_UTC + timedelta(minutes=10)
    rows = [
        _quote_row(t0, 5000.0, 100, 5000.25, 20),
        _trade_row(t0 + timedelta(milliseconds=100), 5000.25, 5, "B",
                    bid_sz=100, ask_sz=20),
        _trade_row(t0 + timedelta(milliseconds=200), 5000.25, 7, "B",
                    bid_sz=100, ask_sz=20),
        # next second: no new quote — should forward-fill
        _trade_row(t0 + timedelta(seconds=1, milliseconds=50), 5000.0, 3, "S",
                    bid_sz=100, ask_sz=20),
    ]
    micro = _resample_tbbo_to_1s(_df(rows))
    assert micro.height == 2
    r0 = micro.row(0, named=True)
    r1 = micro.row(1, named=True)
    assert r0["buy_vol"] == 12 and r0["sell_vol"] == 0
    assert r1["bid_sz"] == 100  # forward-filled
    assert r1["sell_vol"] == 3
    # Microprice = (bid*ask_sz + ask*bid_sz)/(bid_sz+ask_sz): heavy bid SIZE
    # pulls microprice TOWARD the ask (bullish pressure), not toward the bid.
    assert r0["microprice"] > r0["mid"]
    # delta column is buy - sell
    assert r0["delta"] == 12


# ---------------------------------------------------------------------------
# Detector: book imbalance
# ---------------------------------------------------------------------------


def test_book_imbalance_emits_long_when_bids_dominate_for_5s() -> None:
    bars = _make_bars(60 * 20)  # 20 minutes
    t0 = OPEN_UTC + timedelta(minutes=10)
    rows = []
    # 6 seconds of heavy bid stacking (ratio = 200/20 = 10)
    for i in range(6):
        rows.append(_quote_row(
            t0 + timedelta(seconds=i),
            bid_px=5000.0, bid_sz=200, ask_px=5000.25, ask_sz=20,
        ))
    out = _generate_signals_from_data(bars, _df(rows), MicrostructureParams())
    long_imb = [s for s in out if s.setup_id == "book_imbalance" and s.side == Side.LONG]
    assert len(long_imb) >= 1
    s = long_imb[0]
    assert s.entry_price == 5000.25  # ask
    assert s.stop_price < s.entry_price
    assert s.target_price > s.entry_price
    assert 0.0 <= s.confidence <= 1.0


def test_book_imbalance_emits_short_when_asks_dominate() -> None:
    bars = _make_bars(60 * 20)
    t0 = OPEN_UTC + timedelta(minutes=10)
    rows = [
        _quote_row(t0 + timedelta(seconds=i),
                   bid_px=5000.0, bid_sz=20, ask_px=5000.25, ask_sz=200)
        for i in range(6)
    ]
    out = _generate_signals_from_data(bars, _df(rows), MicrostructureParams())
    short_imb = [s for s in out if s.setup_id == "book_imbalance" and s.side == Side.SHORT]
    assert len(short_imb) >= 1
    assert short_imb[0].entry_price == 5000.0  # bid


def test_book_imbalance_does_not_fire_if_top_sz_too_small() -> None:
    p = MicrostructureParams(imbalance_min_top_sz=40)
    bars = _make_bars(60 * 20)
    t0 = OPEN_UTC + timedelta(minutes=10)
    # ratio is 10 but absolute size 10/1 is below floor
    rows = [
        _quote_row(t0 + timedelta(seconds=i),
                   bid_px=5000.0, bid_sz=10, ask_px=5000.25, ask_sz=1)
        for i in range(6)
    ]
    out = _generate_signals_from_data(bars, _df(rows), p)
    assert not any(s.setup_id == "book_imbalance" for s in out)


# ---------------------------------------------------------------------------
# Detector: microprice deviation
# ---------------------------------------------------------------------------


def test_microprice_deviation_emits_short_when_asks_dominate() -> None:
    """Heavy ask size pulls microprice toward the bid => negative dev => SHORT."""
    bars = _make_bars(60 * 20)
    t0 = OPEN_UTC + timedelta(minutes=10)
    # mid = 5000.125. With bid_sz=10, ask_sz=200:
    # microprice = (5000*200 + 5000.25*10) / 210 ≈ 5000.0119 -> dev = -0.113 ≈ -0.45 tk
    # That's < 1.5 tk. Use a wider spread so microprice can deviate more.
    rows = [
        _quote_row(t0 + timedelta(seconds=i),
                   bid_px=5000.0, bid_sz=10, ask_px=5001.0, ask_sz=200)
        for i in range(3)
    ]
    p = replace(MicrostructureParams(),
                imbalance_threshold=999.0,  # disable imbalance to isolate
                microprice_dev_ticks=0.5)
    out = _generate_signals_from_data(bars, _df(rows), p)
    micro_sigs = [s for s in out if s.setup_id == "microprice_dev"]
    assert len(micro_sigs) >= 1
    assert micro_sigs[0].side == Side.SHORT


# ---------------------------------------------------------------------------
# Detector: sweep
# ---------------------------------------------------------------------------


def test_sweep_emits_short_for_buy_sweep() -> None:
    """Multiple aggressive buys within 500ms walking price by 3 ticks => SHORT (fade)."""
    bars = _make_bars(60 * 20)
    t0 = OPEN_UTC + timedelta(minutes=10)
    rows = [
        _quote_row(t0, 5000.0, 20, 5000.25, 20),
    ]
    # 4 aggressive buy prints stepping up the ladder within 400ms
    for i, price in enumerate([5000.25, 5000.50, 5000.75, 5001.00]):
        rows.append(_trade_row(
            t0 + timedelta(milliseconds=50 + i * 100),
            price=price, size=15, side="B",
            bid_px=5000.0 + i * 0.25, ask_px=5000.25 + i * 0.25,
            bid_sz=20, ask_sz=20,
        ))
    p = replace(MicrostructureParams(),
                imbalance_threshold=999.0,
                microprice_dev_ticks=99.0)  # disable other detectors
    out = _generate_signals_from_data(bars, _df(rows), p)
    sweeps = [s for s in out if s.setup_id == "sweep"]
    assert len(sweeps) >= 1
    assert sweeps[0].side == Side.SHORT
    assert sweeps[0].metadata["p_range_ticks"] >= 3.0


def test_sweep_emits_long_for_sell_sweep() -> None:
    bars = _make_bars(60 * 20)
    t0 = OPEN_UTC + timedelta(minutes=12)
    rows = [_quote_row(t0, 5000.0, 20, 5000.25, 20)]
    for i, price in enumerate([5000.0, 4999.75, 4999.50, 4999.25]):
        rows.append(_trade_row(
            t0 + timedelta(milliseconds=50 + i * 80),
            price=price, size=15, side="S",
            bid_px=5000.0 - i * 0.25, ask_px=5000.25 - i * 0.25,
            bid_sz=20, ask_sz=20,
        ))
    p = replace(MicrostructureParams(),
                imbalance_threshold=999.0, microprice_dev_ticks=99.0)
    out = _generate_signals_from_data(bars, _df(rows), p)
    sweeps = [s for s in out if s.setup_id == "sweep"]
    assert len(sweeps) >= 1
    assert sweeps[0].side == Side.LONG


# ---------------------------------------------------------------------------
# Detector: trapped traders
# ---------------------------------------------------------------------------


def test_trapped_long_print_revisited_emits_short() -> None:
    """Big buy print at P, price drops, returns to P => SHORT."""
    t0 = OPEN_UTC + timedelta(minutes=8)
    # Bars: at t0 price = 5000, then drops to 4998, then returns to 5000.5
    bar_rows = []
    base = OPEN_UTC
    for i in range(60 * 20):
        ts = base + timedelta(seconds=i)
        secs_in = (ts - t0).total_seconds()
        if secs_in < 0:
            px = 5000.0
        elif secs_in < 60:
            px = 5000.0  # price holds
        elif secs_in < 180:
            px = 4998.0  # drops 2 points = 8 ticks (> 4-tick adverse threshold)
        elif secs_in < 240:
            px = 5000.25  # returns to print level
        else:
            px = 5000.0
        bar_rows.append({
            "ts": ts, "session_date": SD,
            "open": px, "high": px + 0.25, "low": px - 0.25,
            "close": px, "volume": 10,
        })
    bars = pl.DataFrame(bar_rows, schema={
        "ts": pl.Datetime("us", "UTC"), "session_date": pl.Date,
        "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
        "close": pl.Float64, "volume": pl.Int64,
    })
    # Big buy print at t0
    rows = [
        _quote_row(t0, 5000.0, 20, 5000.25, 20),
        _trade_row(t0 + timedelta(milliseconds=10), price=5000.0,
                    size=100, side="B"),
    ]
    p = replace(MicrostructureParams(),
                imbalance_threshold=999.0, microprice_dev_ticks=99.0,
                hole_flow_min_abs=10**9)  # disable other detectors
    out = _generate_signals_from_data(bars, _df(rows), p)
    traps = [s for s in out if s.setup_id == "trapped"]
    assert len(traps) == 1
    assert traps[0].side == Side.SHORT
    assert traps[0].entry_price == 5000.0


def test_trapped_does_not_fire_without_adverse_move() -> None:
    """Big buy print where price goes UP immediately => no trap fired."""
    bar_rows = []
    base = OPEN_UTC
    for i in range(60 * 20):
        ts = base + timedelta(seconds=i)
        px = 5000.0 + (ts - base).total_seconds() / 600 * 2  # drifts up
        bar_rows.append({
            "ts": ts, "session_date": SD,
            "open": px, "high": px + 0.25, "low": px - 0.25,
            "close": px, "volume": 10,
        })
    bars = pl.DataFrame(bar_rows, schema={
        "ts": pl.Datetime("us", "UTC"), "session_date": pl.Date,
        "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
        "close": pl.Float64, "volume": pl.Int64,
    })
    t0 = OPEN_UTC + timedelta(minutes=8)
    rows = [
        _quote_row(t0, 5000.0, 20, 5000.25, 20),
        _trade_row(t0 + timedelta(milliseconds=10), price=5000.0,
                    size=100, side="B"),
    ]
    p = replace(MicrostructureParams(),
                imbalance_threshold=999.0, microprice_dev_ticks=99.0,
                hole_flow_min_abs=10**9)
    out = _generate_signals_from_data(bars, _df(rows), p)
    assert not any(s.setup_id == "trapped" for s in out)


# ---------------------------------------------------------------------------
# Throttling + time gate
# ---------------------------------------------------------------------------


def test_per_setup_cooldown_drops_duplicate_imbalance_signals() -> None:
    bars = _make_bars(60 * 60)  # 1 hour
    t0 = OPEN_UTC + timedelta(minutes=10)
    rows = []
    # Two bursts of bid imbalance: at t0 and t0+60s
    for offset in (0, 60):
        for i in range(6):
            rows.append(_quote_row(
                t0 + timedelta(seconds=offset + i),
                bid_px=5000.0, bid_sz=200, ask_px=5000.25, ask_sz=20,
            ))
    p = replace(MicrostructureParams(),
                min_seconds_between_signals=600,  # 10 min cooldown
                microprice_dev_ticks=99.0)
    out = _generate_signals_from_data(bars, _df(rows), p)
    imb = [s for s in out if s.setup_id == "book_imbalance"]
    # Only one signal should survive: the cooldown is 10 min and bursts are 60s apart
    assert len(imb) == 1


def test_signals_before_open_cutoff_are_filtered() -> None:
    bars = _make_bars(60 * 20)
    t0 = OPEN_UTC + timedelta(minutes=1)  # before 5-min open cutoff
    rows = [
        _quote_row(t0 + timedelta(seconds=i),
                   bid_px=5000.0, bid_sz=200, ask_px=5000.25, ask_sz=20)
        for i in range(6)
    ]
    out = _generate_signals_from_data(bars, _df(rows), MicrostructureParams())
    # Trigger times are in the first 5 minutes; should be filtered out
    assert all(
        s.timestamp >= OPEN_UTC + timedelta(minutes=5)
        for s in out
    )


def test_max_signals_per_day_caps_output() -> None:
    bars = _make_bars(60 * 60 * 4)  # 4-hour synthetic session
    rows = []
    # Many widely-spaced imbalance triggers
    for k in range(40):
        t0 = OPEN_UTC + timedelta(minutes=10 + k * 8)
        for i in range(6):
            rows.append(_quote_row(
                t0 + timedelta(seconds=i),
                bid_px=5000.0, bid_sz=200, ask_px=5000.25, ask_sz=20,
            ))
    p = replace(MicrostructureParams(),
                max_signals_per_day=5,
                min_seconds_between_signals=120,
                microprice_dev_ticks=99.0)
    out = _generate_signals_from_data(bars, _df(rows), p)
    assert len(out) <= 5


# ---------------------------------------------------------------------------
# Signal contract conformance
# ---------------------------------------------------------------------------


def test_emitted_signals_obey_contract() -> None:
    bars = _make_bars(60 * 20)
    t0 = OPEN_UTC + timedelta(minutes=10)
    rows = [
        _quote_row(t0 + timedelta(seconds=i),
                   bid_px=5000.0, bid_sz=200, ask_px=5000.25, ask_sz=20)
        for i in range(6)
    ]
    out = _generate_signals_from_data(bars, _df(rows), MicrostructureParams())
    assert len(out) >= 1
    for s in out:
        assert 0.0 <= s.confidence <= 1.0
        assert s.specialist == "micro"
        if s.side == Side.LONG:
            assert s.stop_price < s.entry_price
            assert s.target_price is None or s.target_price > s.entry_price
        else:
            assert s.stop_price > s.entry_price
            assert s.target_price is None or s.target_price < s.entry_price
