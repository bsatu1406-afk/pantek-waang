"""Unit tests for the Momentum / Breakout specialist (Agent #8).

ORB geometry is tested FIRST (the brief asks for that explicitly), followed by
the EOD-drive and multi-bar contraction setups, the signal contract, and the
empty-input edge cases.

All tests construct synthetic 1-second RTH bars in-memory. We never depend on
real Databento data here — the standalone audit is where real data is
exercised.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from src.common.types import Side
from src.strategy.specialists.momentum import (
    MomentumParams,
    generate_signals,
)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# --- bar builders -----------------------------------------------------------


def _bar_ts(sd: date, t: time) -> datetime:
    """Convert a session-date + ET time into a tz-aware UTC datetime."""
    return datetime(sd.year, sd.month, sd.day, t.hour, t.minute, t.second, tzinfo=ET).astimezone(UTC)


def _rth_seconds(sd: date) -> list[datetime]:
    """All 09:30-15:59:59 ET seconds for ``sd`` as UTC tz-aware datetimes."""
    out: list[datetime] = []
    cur = datetime(sd.year, sd.month, sd.day, 9, 30, 0, tzinfo=ET)
    end = datetime(sd.year, sd.month, sd.day, 16, 0, 0, tzinfo=ET)
    while cur < end:
        out.append(cur.astimezone(UTC))
        cur += timedelta(seconds=1)
    return out


def _build_bars(sd: date, price_path) -> pl.DataFrame:
    """Construct a Polars 1-s bar frame from a callable ``price_path(ts_et)``.

    The callable returns ``(open, high, low, close, volume)``. We just use
    ``close`` of the previous bar as the next ``open`` to keep things tight.
    """
    seconds = _rth_seconds(sd)
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    vols: list[int] = []
    last_close: float | None = None
    for ts in seconds:
        ts_et = ts.astimezone(ET)
        o, h, lo, c, v = price_path(ts_et)
        if last_close is not None:
            o = last_close
        # ensure h >= max(o,c), lo <= min(o,c)
        h = max(h, o, c)
        lo = min(lo, o, c)
        opens.append(o)
        highs.append(h)
        lows.append(lo)
        closes.append(c)
        vols.append(v)
        last_close = c
    return pl.DataFrame(
        {
            "ts": seconds,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
            "session_date": [sd] * len(seconds),
        }
    ).with_columns(
        pl.col("ts").cast(pl.Datetime("us", "UTC")),
        pl.col("session_date").cast(pl.Date),
    )


# --- price-path factories ---------------------------------------------------


def _orb_long_breakout(sd: date):
    """Build an OR with high≈5005, low≈5000 then a sharp breakout up at 09:45 ET."""

    def f(ts_et: datetime):
        m = (ts_et.hour - 9) * 60 + (ts_et.minute - 30) + ts_et.second / 60.0
        if m < 5:
            # OR: oscillate between 5000 and 5005
            base = 5002.5 + 2.5 * ((m % 1) - 0.5) * 2.0
            return (base, 5005.0, 5000.0, base, 100)
        if m < 15:
            # consolidate just under or_high (no breakout yet)
            return (5003.0, 5004.5, 5001.0, 5003.0, 50)
        if m < 16:
            # explosive break above 5005 + 2 ticks = 5005.50
            return (5006.0, 5008.0, 5005.5, 5008.0, 500)
        # drift up afterwards
        target = 5008.0 + min(20.0, (m - 16) * 0.1)
        return (target, target + 0.5, target - 0.5, target, 80)

    return f


def _orb_short_breakout(sd: date):
    def f(ts_et: datetime):
        m = (ts_et.hour - 9) * 60 + (ts_et.minute - 30) + ts_et.second / 60.0
        if m < 5:
            base = 5002.5 + 2.5 * ((m % 1) - 0.5) * 2.0
            return (base, 5005.0, 5000.0, base, 100)
        if m < 15:
            return (5001.5, 5004.0, 5000.5, 5001.5, 50)
        if m < 16:
            return (4998.0, 4999.0, 4993.0, 4993.0, 500)
        target = 4993.0 - min(20.0, (m - 16) * 0.1)
        return (target, target + 0.5, target - 0.5, target, 80)

    return f


def _tight_range_no_break(sd: date):
    def f(ts_et: datetime):
        # OR of 1 tick only (4999.75 / 5000.00), then nothing happens
        if ts_et.minute % 2 == 0:
            return (5000.0, 5000.0, 4999.75, 5000.0, 10)
        return (4999.75, 5000.0, 4999.75, 4999.75, 10)

    return f


def _eod_long_drive(sd: date):
    """Strong up-trend from open, dip THROUGH VWAP around 14:35, then continuation.

    VWAP at 14:30 sits near 5025 (midpoint of the linear ramp). The dip
    intentionally crosses below VWAP so the pullback detector trips.
    """

    def f(ts_et: datetime):
        m = (ts_et.hour - 9) * 60 + (ts_et.minute - 30) + ts_et.second / 60.0
        # Ramp 5000 → 5050 from 09:30 → 14:30 (linear)
        if m < 300:
            p = 5000.0 + 50.0 * (m / 300.0)
            return (p, p + 0.5, p - 0.5, p, 100)
        if m < 308:  # 14:30 → 14:38 ET: dip THROUGH VWAP (~5025) to 5018
            p = 5050.0 - 32.0 * ((m - 300) / 8.0)  # down to ~5018
            return (p, p + 0.5, p - 0.5, p, 200)
        if m < 360:  # 14:38 → 15:30 ET: drive back up
            p = 5018.0 + 38.0 * ((m - 308) / 52.0)  # up to ~5056
            return (p, p + 0.5, p - 0.5, p, 200)
        # tail: hold
        return (5056.0, 5056.5, 5055.5, 5056.0, 100)

    return f


def _multibar_contraction_then_break(sd: date):
    """Wide chop for 30 min, 20-min tight contraction, then sharp break up.

    Layout chosen so that at break time both windows hold the right state:

    * ``rolling-5min range`` sees only the contraction bars (low rng)
    * ``ATR_30m`` still has half its window in the wide chop (high baseline)
    """

    def f(ts_et: datetime):
        m = (ts_et.hour - 9) * 60 + (ts_et.minute - 30) + ts_et.second / 60.0
        if m < 30:
            # WIDE chop: h-l = 8.0 per bar so ATR_30m stays elevated
            p = 5000.0 + 8.0 * ((int(m) % 5) - 2)
            return (p, p + 4.0, p - 4.0, p, 100)
        if m < 50:
            # tight 20-min contraction band 4999.75-5000.25 (0.5 wide)
            p = 5000.0 + 0.125 * ((int(m * 2) % 2) - 0.5)
            return (p, 5000.25, 4999.75, p, 50)
        if m < 50.5:
            # explosive long break above 5000.25 + 2 ticks = 5000.75
            return (5001.5, 5002.5, 5001.0, 5002.5, 500)
        p = 5002.5 + (m - 50.5) * 0.05
        return (p, p + 0.5, p - 0.5, p, 100)

    return f


# --- tests ------------------------------------------------------------------


SD = date(2024, 3, 14)


def test_orb_geometry_long_breakout_fires_with_stop_below_entry():
    bars = _build_bars(SD, _orb_long_breakout(SD))
    params = MomentumParams()
    sigs = generate_signals(bars, params)
    orb_longs = [s for s in sigs if s.setup_id.startswith("orb_") and s.side == Side.LONG]
    assert orb_longs, "expected at least one ORB long signal"
    s = orb_longs[0]
    # Stop is on the opposite side of entry
    assert s.stop_price < s.entry_price
    assert s.target_price is not None and s.target_price > s.entry_price
    # Stop matches OR low (≈ 5000)
    assert 4999.0 <= s.stop_price <= 5001.0
    # Entry sits above OR high (≈ 5005)
    assert s.entry_price > 5005.0
    # Confidence is in [0, 1]
    assert 0.0 <= s.confidence <= 1.0


def test_orb_geometry_short_breakout_fires_with_stop_above_entry():
    bars = _build_bars(SD, _orb_short_breakout(SD))
    sigs = generate_signals(bars, MomentumParams())
    orb_shorts = [s for s in sigs if s.setup_id.startswith("orb_") and s.side == Side.SHORT]
    assert orb_shorts, "expected at least one ORB short signal"
    s = orb_shorts[0]
    assert s.stop_price > s.entry_price
    assert s.target_price is not None and s.target_price < s.entry_price
    assert 4999.0 <= s.stop_price <= 5006.0
    assert s.entry_price < 5000.0


def test_orb_no_signal_on_tight_dead_range():
    bars = _build_bars(SD, _tight_range_no_break(SD))
    sigs = generate_signals(bars, MomentumParams())
    # No ORB signal because OR_range is 1 tick < orb_min_range_ticks (4)
    assert not [s for s in sigs if s.setup_id.startswith("orb_")]


def test_orb_buffer_respected_no_signal_when_close_within_buffer():
    """Close that pierces OR_high by exactly 1 tick must NOT fire when buffer = 2 ticks."""

    def path(ts_et):
        m = (ts_et.hour - 9) * 60 + (ts_et.minute - 30) + ts_et.second / 60.0
        if m < 5:
            base = 5002.5 + 2.5 * ((m % 1) - 0.5) * 2.0
            return (base, 5005.0, 5000.0, base, 100)
        # After OR: close stays exactly 1 tick above 5005 ⇒ NO signal (buffer=2 ticks)
        return (5005.25, 5005.25, 5005.0, 5005.25, 50)

    bars = _build_bars(SD, path)
    sigs = generate_signals(bars, MomentumParams(breakout_buffer_ticks=2))
    assert not [s for s in sigs if s.setup_id.startswith("orb_") and s.side == Side.LONG]


def test_eod_drive_long_continuation_fires_in_window():
    bars = _build_bars(SD, _eod_long_drive(SD))
    sigs = generate_signals(bars, MomentumParams())
    eod = [s for s in sigs if s.setup_id == "eod_drive_long"]
    assert eod, "expected an EOD long drive signal"
    s = eod[0]
    # Within the 14:30-15:30 ET window
    et = s.timestamp.astimezone(ET)
    assert et.time() >= time(14, 30)
    assert et.time() <= time(15, 30)
    assert s.stop_price < s.entry_price
    assert s.target_price is not None and s.target_price > s.entry_price


def test_multibar_contraction_breakout_fires():
    bars = _build_bars(SD, _multibar_contraction_then_break(SD))
    sigs = generate_signals(bars, MomentumParams())
    mb = [s for s in sigs if s.setup_id.startswith("multibar_break_")]
    assert mb, "expected at least one multibar breakout signal"
    s = mb[0]
    # Stop is on the opposite side of entry regardless of LONG/SHORT
    if s.side == Side.LONG:
        assert s.stop_price < s.entry_price
        assert s.target_price is not None and s.target_price > s.entry_price
    else:
        assert s.stop_price > s.entry_price
        assert s.target_price is not None and s.target_price < s.entry_price


def test_empty_bars_returns_empty_list():
    empty = pl.DataFrame(
        schema={
            "ts": pl.Datetime("us", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        }
    )
    assert generate_signals(empty, MomentumParams()) == []


def test_signals_returned_in_chronological_order():
    bars = _build_bars(SD, _orb_long_breakout(SD))
    sigs = generate_signals(bars, MomentumParams())
    timestamps = [s.timestamp for s in sigs]
    assert timestamps == sorted(timestamps)


def test_signal_contract_stop_opposite_side_of_entry_and_confidence_bounded():
    """The engine relies on stop being on the opposite side of entry. This test
    enforces the contract across every signal type the specialist can emit."""
    bars_long = _build_bars(SD, _orb_long_breakout(SD))
    bars_short = _build_bars(SD, _orb_short_breakout(SD))
    bars_mb = _build_bars(SD, _multibar_contraction_then_break(SD))
    sigs = (
        generate_signals(bars_long, MomentumParams())
        + generate_signals(bars_short, MomentumParams())
        + generate_signals(bars_mb, MomentumParams())
    )
    assert sigs, "test fixtures should produce at least one signal"
    for s in sigs:
        assert s.specialist == "momentum"
        assert 0.0 <= s.confidence <= 1.0
        if s.side == Side.LONG:
            assert s.stop_price < s.entry_price
            if s.target_price is not None:
                assert s.target_price > s.entry_price
        else:
            assert s.stop_price > s.entry_price
            if s.target_price is not None:
                assert s.target_price < s.entry_price


def test_missing_required_columns_raises():
    bad = pl.DataFrame({"ts": [datetime(2024, 3, 14, 14, 30, tzinfo=UTC)], "open": [1.0]})
    with pytest.raises(ValueError, match="missing required bar columns"):
        generate_signals(bad, MomentumParams())


def test_per_session_independence_no_carry_over():
    """Calling generate_signals twice on the same bars must yield identical results
    (deterministic + no hidden state)."""
    bars = _build_bars(SD, _orb_long_breakout(SD))
    p = MomentumParams()
    a = generate_signals(bars, p)
    b = generate_signals(bars, p)
    assert len(a) == len(b)
    for sa, sb in zip(a, b, strict=True):
        assert sa.timestamp == sb.timestamp
        assert sa.side == sb.side
        assert sa.entry_price == sb.entry_price
        assert sa.stop_price == sb.stop_price
        assert sa.setup_id == sb.setup_id
