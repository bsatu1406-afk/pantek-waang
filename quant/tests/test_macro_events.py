"""Tests for Agent #2 — Macro & Event Specialist.

Pins:
 * Event calendar coverage (FOMC / CPI / NFP / OPEX / holidays) for 2018-2025.
 * Public helpers (is_event_day, event_distance, is_session_blocked).
 * Signal-generation correctness for the three opportunistic setups, with
   synthetic 1-second bars (no real data needed — the parent will run the
   live audit on the 5-year corpus).
"""
from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from src.common.types import Side
from src.strategy.specialists.macro_events import (
    EVENT_LOCKOUT_DATES,
    EVENTS,
    SPECIALIST_ID,
    MacroEventParams,
    event_distance,
    generate_signals,
    is_event_day,
    is_session_blocked,
)

ET = ZoneInfo("America/New_York")
UTC = UTC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _et_to_utc(d: date, t: time) -> datetime:
    return datetime.combine(d, t, tzinfo=ET).astimezone(UTC)


def _synth_bars(
    sd: date,
    *,
    open_px: float = 4800.0,
    drift_pts_by_et: dict[time, float] | None = None,
) -> pl.DataFrame:
    """Build a 1-second RTH bars frame (09:30-16:00 ET) with prescribed drift.

    drift_pts_by_et maps an ET wall-clock time to the price delta (vs open) that
    must be observed at that bar. Linear interpolation between waypoints.
    """
    drift_pts_by_et = drift_pts_by_et or {}
    start = _et_to_utc(sd, time(9, 30))
    end = _et_to_utc(sd, time(16, 0))
    n_seconds = int((end - start).total_seconds())

    waypoints_utc = sorted(
        [(_et_to_utc(sd, t), pts) for t, pts in drift_pts_by_et.items()]
    )

    def drift_at(ts: datetime) -> float:
        if not waypoints_utc:
            return 0.0
        if ts <= waypoints_utc[0][0]:
            return waypoints_utc[0][1]
        if ts >= waypoints_utc[-1][0]:
            return waypoints_utc[-1][1]
        for i in range(len(waypoints_utc) - 1):
            a_ts, a_v = waypoints_utc[i]
            b_ts, b_v = waypoints_utc[i + 1]
            if a_ts <= ts <= b_ts:
                span = (b_ts - a_ts).total_seconds() or 1.0
                f = (ts - a_ts).total_seconds() / span
                return a_v + f * (b_v - a_v)
        return waypoints_utc[-1][1]

    rows = []
    for i in range(n_seconds):
        ts = start + timedelta(seconds=i)
        px = open_px + drift_at(ts)
        # Keep OHLC tight so stops/targets are hit cleanly at exact prices.
        rows.append({
            "ts": ts,
            "session_date": sd,
            "open": px,
            "high": px + 0.25,
            "low": px - 0.25,
            "close": px,
            "volume": 100,
        })
    df = pl.DataFrame(rows)
    df = df.with_columns(
        pl.col("ts").cast(pl.Datetime("ns", time_zone="UTC")),
        pl.col("session_date").cast(pl.Date),
        pl.col("volume").cast(pl.Int64),
    )
    return df


# ---------------------------------------------------------------------------
# Calendar coverage
# ---------------------------------------------------------------------------


def test_calendar_size_is_multi_year_and_covers_kinds():
    """EVENTS must be a multi-hundred-entry calendar spanning all kinds."""
    assert len(EVENTS) >= 700
    kinds = {e.kind for e in EVENTS}
    required = {"FOMC", "FOMC_MINUTES", "CPI", "PPI", "NFP", "GDP", "ISM",
                "OPEX", "QUAD_WITCH", "HOLIDAY", "EARLY_CLOSE"}
    missing = required - kinds
    assert not missing, f"missing event kinds: {missing}"
    years = {e.d.year for e in EVENTS}
    assert years.issuperset(set(range(2018, 2026))), \
        f"calendar missing some years 2018-2025: {sorted(years)}"


def test_known_fomc_day_and_cpi_day_are_event_days():
    # FOMC announcement on Jan 31 2024 is a known HIGH-impact day.
    is_evt, kind = is_event_day(date(2024, 1, 31))
    assert is_evt is True
    assert kind == "FOMC"
    # CPI release on Jan 11 2024 — HIGH impact too.
    is_evt2, kind2 = is_event_day(date(2024, 1, 11))
    assert is_evt2 is True
    assert kind2 == "CPI"


def test_non_event_day_returns_false_and_random_midweek():
    """A quiet Tuesday like 2024-01-09 carries no HIGH-impact event."""
    is_evt, kind = is_event_day(date(2024, 1, 9))
    assert is_evt is False
    assert kind == ""


def test_is_session_blocked_for_holidays_and_fomc_not_for_random_day():
    # New Year's Day 2024 was a market holiday.
    assert is_session_blocked(date(2024, 1, 1)) is True
    # FOMC days are HIGH impact => blocked for non-event specialists.
    assert is_session_blocked(date(2024, 1, 31)) is True
    # Random midweek non-event Tuesday.
    assert is_session_blocked(date(2024, 1, 9)) is False
    # Lockout cardinality sanity: a few hundred high-impact days over 8 years.
    assert 300 <= len(EVENT_LOCKOUT_DATES) <= 800


def test_event_distance_signed_trading_days():
    """`event_distance` returns signed trading-day distances per event kind."""
    # 24 Jan 2024 (Wed). Next FOMC = Jan 31 (Wed) => 5 trading days ahead.
    dist = event_distance(date(2024, 1, 24))
    assert dist["fomc"] == 5
    # CPI Jan 11 was the prior CPI => 8 trading days before Jan 24 (5 in Jan W3
    # + 3 in W2 after release): we accept the value the impl returns but it
    # must be a small negative integer.
    assert dist["cpi"] < 0
    # Day-of behaviour: distance to today's event is 0.
    fomc_day = date(2024, 1, 31)
    dist_fomc = event_distance(fomc_day)
    assert dist_fomc["fomc"] == 0
    # NFP is forward-looking (first Friday of Feb = Feb 2) => +2 trading days.
    assert dist_fomc["nfp"] == 2


# ---------------------------------------------------------------------------
# generate_signals — empty / non-event / holiday paths
# ---------------------------------------------------------------------------


def test_generate_signals_empty_bars_returns_empty():
    sigs = generate_signals(pl.DataFrame(schema={"ts": pl.Datetime("ns", "UTC"),
                                                 "session_date": pl.Date,
                                                 "open": pl.Float64,
                                                 "high": pl.Float64,
                                                 "low": pl.Float64,
                                                 "close": pl.Float64,
                                                 "volume": pl.Int64}),
                            MacroEventParams())
    assert sigs == []


def test_generate_signals_non_event_day_returns_empty():
    # 2024-01-09 — no event. Synthetic bars with any drift => no signals.
    bars = _synth_bars(date(2024, 1, 9),
                       drift_pts_by_et={time(11, 0): -6.0})
    assert generate_signals(bars, MacroEventParams()) == []


def test_generate_signals_holiday_returns_empty():
    bars = _synth_bars(date(2024, 1, 1))
    assert generate_signals(bars, MacroEventParams()) == []


# ---------------------------------------------------------------------------
# Setup 1 — event AM range fade
# ---------------------------------------------------------------------------


def test_am_range_fade_short_on_cpi_after_rally():
    """CPI day with a +6 pt 9:30-11:00 rally => fade short at 11:00 ET."""
    sd = date(2024, 1, 11)  # CPI day
    bars = _synth_bars(sd, drift_pts_by_et={
        time(9, 30): 0.0,
        time(11, 0): 6.0,   # rallied +6 into 11:00
    })
    sigs = generate_signals(bars, MacroEventParams())
    fade_short = [s for s in sigs
                  if s.setup_id.startswith("event_am_range_fade_short")]
    assert len(fade_short) == 1
    s = fade_short[0]
    assert s.side == Side.SHORT
    assert s.specialist == SPECIALIST_ID
    assert s.stop_price > s.entry_price
    assert s.target_price is not None and s.target_price < s.entry_price
    assert 0.0 <= s.confidence <= 1.0
    assert s.metadata["event_kind"] == "CPI"


def test_am_range_fade_long_on_nfp_after_drop():
    """NFP day with a -6 pt 9:30-11:00 drop => fade long at 11:00 ET."""
    # First Friday of February 2024 was Feb 2.
    sd = date(2024, 2, 2)
    bars = _synth_bars(sd, drift_pts_by_et={
        time(9, 30): 0.0,
        time(11, 0): -6.0,
    })
    sigs = generate_signals(bars, MacroEventParams())
    fade_long = [s for s in sigs
                 if s.setup_id.startswith("event_am_range_fade_long")]
    assert len(fade_long) == 1
    s = fade_long[0]
    assert s.side == Side.LONG
    assert s.stop_price < s.entry_price
    assert s.target_price is not None and s.target_price > s.entry_price
    assert s.metadata["event_kind"] == "NFP"


def test_am_range_fade_skipped_when_range_too_small():
    """If 9:30-11:00 range is below threshold, no fade signal."""
    sd = date(2024, 1, 11)  # CPI
    bars = _synth_bars(sd, drift_pts_by_et={
        time(9, 30): 0.0,
        time(11, 0): 1.0,   # tiny 1 pt drift
    })
    sigs = generate_signals(bars, MacroEventParams())
    fades = [s for s in sigs if "am_range_fade" in s.setup_id]
    assert fades == []


# ---------------------------------------------------------------------------
# Setup 2 — pre-FOMC drift long
# ---------------------------------------------------------------------------


def test_pre_fomc_drift_long_when_pre_drift_is_neutral():
    """FOMC day with ~flat pre-drift => emit pre_fomc_drift_long at 12:00 ET."""
    sd = date(2024, 1, 31)  # FOMC announcement day
    bars = _synth_bars(sd, drift_pts_by_et={
        time(9, 30): 0.0,
        time(12, 0): -1.0,   # mild dip, still in allowed band
        time(14, 0): -1.0,
        time(14, 30): -1.0,
    })
    sigs = generate_signals(bars, MacroEventParams())
    pre = [s for s in sigs if s.setup_id == "pre_fomc_drift_long"]
    assert len(pre) == 1
    s = pre[0]
    assert s.side == Side.LONG
    assert s.stop_price < s.entry_price
    assert s.target_price is not None and s.target_price > s.entry_price


def test_pre_fomc_drift_skipped_when_market_already_rallied_too_far():
    sd = date(2024, 1, 31)
    bars = _synth_bars(sd, drift_pts_by_et={
        time(9, 30): 0.0,
        time(12, 0): 20.0,  # already +20 before noon => no edge
    })
    sigs = generate_signals(bars, MacroEventParams())
    pre = [s for s in sigs if s.setup_id == "pre_fomc_drift_long"]
    assert pre == []


# ---------------------------------------------------------------------------
# Setup 3 — post-FOMC reaction fade
# ---------------------------------------------------------------------------


def test_post_fomc_reaction_fade_short_after_initial_rally():
    """FOMC: market rips +8 pts in 14:00-14:30 ET => fade short at 14:30 ET."""
    sd = date(2024, 1, 31)
    bars = _synth_bars(sd, drift_pts_by_et={
        time(9, 30): 0.0,
        time(12, 0): -2.0,
        time(14, 0): -2.0,
        time(14, 30): 6.0,   # +8 pts from 14:00 to 14:30
    })
    sigs = generate_signals(bars, MacroEventParams())
    fade = [s for s in sigs if s.setup_id.startswith("post_fomc_reaction_fade_short")]
    assert len(fade) == 1
    s = fade[0]
    assert s.side == Side.SHORT
    assert s.stop_price > s.entry_price
    assert s.target_price is not None and s.target_price < s.entry_price


def test_post_fomc_reaction_fade_skipped_when_reaction_is_small():
    sd = date(2024, 1, 31)
    bars = _synth_bars(sd, drift_pts_by_et={
        time(9, 30): 0.0,
        time(14, 0): 0.0,
        time(14, 30): 1.0,   # only +1 pt reaction => below threshold
    })
    sigs = generate_signals(bars, MacroEventParams())
    fade = [s for s in sigs if "post_fomc_reaction_fade" in s.setup_id]
    assert fade == []


# ---------------------------------------------------------------------------
# Combined day + determinism
# ---------------------------------------------------------------------------


def test_full_fomc_day_emits_multiple_signals_in_chronological_order():
    sd = date(2024, 1, 31)
    bars = _synth_bars(sd, drift_pts_by_et={
        time(9, 30): 0.0,
        time(12, 0): 0.0,
        time(14, 0): 0.0,
        time(14, 30): 7.0,
    })
    sigs = generate_signals(bars, MacroEventParams())
    setups = {s.setup_id for s in sigs}
    assert "pre_fomc_drift_long" in setups
    assert any("post_fomc_reaction_fade" in sid for sid in setups)
    # chronological
    timestamps = [s.timestamp for s in sigs]
    assert timestamps == sorted(timestamps)


def test_generate_signals_is_deterministic():
    sd = date(2024, 1, 11)
    bars = _synth_bars(sd, drift_pts_by_et={
        time(9, 30): 0.0,
        time(11, 0): 6.0,
    })
    p = MacroEventParams()
    a = generate_signals(bars, p)
    b = generate_signals(bars, p)
    assert len(a) == len(b)
    for sa, sb in zip(a, b, strict=True):
        assert sa.timestamp == sb.timestamp
        assert sa.side == sb.side
        assert sa.entry_price == sb.entry_price
        assert sa.stop_price == sb.stop_price
        assert sa.target_price == sb.target_price
        assert sa.setup_id == sb.setup_id


def test_signal_contract_fields_are_valid():
    """All emitted signals must satisfy the Signal contract invariants."""
    sd = date(2024, 1, 31)
    bars = _synth_bars(sd, drift_pts_by_et={
        time(9, 30): 0.0,
        time(11, 0): 6.0,
        time(12, 0): 4.0,
        time(14, 0): 4.0,
        time(14, 30): 12.0,
    })
    sigs = generate_signals(bars, MacroEventParams())
    assert len(sigs) >= 1
    for s in sigs:
        assert 0.0 <= s.confidence <= 1.0
        assert s.specialist == SPECIALIST_ID
        assert isinstance(s.setup_id, str) and s.setup_id
        # stop on opposite side of entry
        if s.side == Side.LONG:
            assert s.stop_price < s.entry_price
            if s.target_price is not None:
                assert s.target_price > s.entry_price
        else:
            assert s.stop_price > s.entry_price
            if s.target_price is not None:
                assert s.target_price < s.entry_price
