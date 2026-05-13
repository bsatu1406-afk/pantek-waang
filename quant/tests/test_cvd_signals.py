"""Unit tests for the CVD specialist (Agent #4)."""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise

import polars as pl

from src.common.types import Side
from src.strategy.specialists.cvd_signals import (
    CVDParams,
    _enrich,
    detect_absorption,
    detect_divergence,
    detect_exhaustion,
    generate_signals,
)

SD = date(2024, 1, 2)
# RTH session opens 09:30 ET == 14:30 UTC (DST off in January).
SESSION_OPEN = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def _empty_bars() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ts": pl.Datetime("us", "UTC"),
            "session_date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
            "delta": pl.Float64,
            "buy_vol": pl.Float64,
            "sell_vol": pl.Float64,
            "cvd": pl.Float64,
        }
    )


def _make_bars(
    prices, deltas, *,
    start: datetime = SESSION_OPEN,
    sd: date = SD,
    bar_half_range: float = 0.25,
) -> pl.DataFrame:
    """Build a deterministic 1-second bar frame for one session day.

    ``prices[i]`` is the close at second i (open == close, high/low ==
    close ± ``bar_half_range``). ``deltas[i]`` is the per-second signed volume.
    """
    assert len(prices) == len(deltas), "prices and deltas must have equal length"
    rows = []
    cvd = 0
    for i, (p, d) in enumerate(zip(prices, deltas, strict=True)):
        cvd += d
        rows.append(
            {
                "ts": start + timedelta(seconds=i),
                "session_date": sd,
                "open": float(p),
                "high": float(p) + bar_half_range,
                "low": float(p) - bar_half_range,
                "close": float(p),
                "volume": int(abs(d) + 10),
                "delta": float(d),
                "buy_vol": float(max(d, 0) + 5),
                "sell_vol": float(max(-d, 0) + 5),
                "cvd": float(cvd),
            }
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sanity / contract tests
# ---------------------------------------------------------------------------


def test_empty_bars_returns_no_signals():
    assert generate_signals(_empty_bars(), CVDParams()) == []


def test_bars_without_cvd_returns_no_signals(monkeypatch):
    """When the session has no cvd_1s parquet, no signals are emitted."""
    n = 3600
    bars = _make_bars([4700 + 0.01 * i for i in range(n)], [0] * n)
    bars = bars.drop(["delta", "buy_vol", "sell_vol", "cvd"])
    # Patch the loader to simulate a missing parquet.
    monkeypatch.setattr(
        "src.strategy.specialists.cvd_signals.load_cvd_1s", lambda sd: None
    )
    sigs = generate_signals(bars, CVDParams())
    assert sigs == []


def test_default_params_have_correct_specialist_id():
    p = CVDParams()
    assert p.specialist_id == "cvd"


# ---------------------------------------------------------------------------
# Detector unit tests — use _enrich() directly to verify each detector fires
# ---------------------------------------------------------------------------


def test_detect_bearish_divergence_fires_on_hh_with_lh_cvd():
    """Price HH + CVD LH → exactly one short divergence candidate near the HH."""
    n = 1800
    # First half: aggressive buying drives price + CVD up.
    # Second half: price grinds even higher but CVD bleeds down (LH).
    prices = []
    deltas = []
    for i in range(n):
        if i < 900:
            prices.append(4700 + 0.02 * i)  # 4700 → 4718
            deltas.append(50)
        else:
            prices.append(4718 + 0.005 * (i - 900))  # creeps higher
            deltas.append(-30)
    df = _make_bars(prices, deltas)
    df = _enrich(df, CVDParams())
    hits = detect_divergence(df, CVDParams())
    shorts = [h for h in hits if h[1] == Side.SHORT]
    assert shorts, f"expected at least one short divergence, got {hits}"


def test_detect_bullish_divergence_fires_on_ll_with_hl_cvd():
    n = 1800
    prices, deltas = [], []
    for i in range(n):
        if i < 900:
            prices.append(4700 - 0.02 * i)  # 4700 → 4682
            deltas.append(-50)
        else:
            prices.append(4682 - 0.005 * (i - 900))
            deltas.append(30)
    df = _make_bars(prices, deltas)
    df = _enrich(df, CVDParams())
    hits = detect_divergence(df, CVDParams())
    longs = [h for h in hits if h[1] == Side.LONG]
    assert longs, f"expected at least one long divergence, got {hits}"


def test_detect_absorption_fires_on_heavy_one_sided_flat_range():
    """Heavy buy aggression inside a tight range → short absorption."""
    n = 600
    prices = [4700.0] * n  # totally flat range
    deltas = [60] * n      # one-sided buying for 10 minutes
    # Tight bar range so range_atr falls below the threshold.
    df = _make_bars(prices, deltas, bar_half_range=0.0)
    df = _enrich(df, CVDParams())
    hits = detect_absorption(df, CVDParams(absorption_min_volume=500))
    shorts = [h for h in hits if h[1] == Side.SHORT]
    assert shorts, "expected at least one absorption-short"
    # And the symmetric case.
    df2 = _make_bars(prices, [-60] * n, bar_half_range=0.0)
    df2 = _enrich(df2, CVDParams())
    hits2 = detect_absorption(df2, CVDParams(absorption_min_volume=500))
    assert any(h[1] == Side.LONG for h in hits2), "expected absorption-long"


def test_detect_exhaustion_spike_short_when_spike_fails_to_make_new_high():
    n = 1200
    # Baseline noise; insert one giant +delta spike at t=600; price *immediately*
    # rolls over and never re-tests the high.
    prices = []
    deltas = []
    for i in range(n):
        if i < 600:
            prices.append(4700 + 0.005 * i)
            deltas.append(2 * math.sin(i / 30.0))
        elif i == 600:
            prices.append(4703.0)
            deltas.append(2000.0)  # massive +Z spike
        else:
            # roll over and stay well below the spike high
            prices.append(4703.0 - 0.01 * (i - 600))
            deltas.append(2 * math.sin(i / 30.0))
    df = _make_bars(prices, deltas)
    df = _enrich(df, CVDParams(exhaustion_zscore=3.0, exhaustion_confirm_bars=30))
    hits = detect_exhaustion(df, CVDParams(exhaustion_zscore=3.0, exhaustion_confirm_bars=30))
    shorts = [h for h in hits if h[1] == Side.SHORT]
    assert shorts, f"expected exhaustion short, got {hits}"


# ---------------------------------------------------------------------------
# generate_signals — end-to-end contract enforcement
# ---------------------------------------------------------------------------


def test_signal_contract_stop_on_opposite_side_and_confidence_bounded():
    n = 3600
    prices, deltas = [], []
    for i in range(n):
        if i < 1800:
            prices.append(4700 + 0.01 * i)
            deltas.append(60)
        else:
            prices.append(4718 + 0.002 * (i - 1800))
            deltas.append(-40)
    bars = _make_bars(prices, deltas)
    sigs = generate_signals(bars, CVDParams())
    assert sigs, "expected at least one signal on this synthetic divergence"
    for s in sigs:
        assert 0.0 <= s.confidence <= 1.0
        assert s.specialist == "cvd"
        assert s.setup_id in {
            "cvd_bearish_divergence", "cvd_bullish_divergence",
            "absorption_short", "absorption_long",
            "exhaustion_spike_short", "exhaustion_spike_long",
        }
        if s.side == Side.LONG:
            assert s.stop_price < s.entry_price < (s.target_price or float("inf"))
        else:
            assert s.stop_price > s.entry_price > (s.target_price or float("-inf"))


def test_min_seconds_between_signals_dedups_back_to_back_candidates():
    n = 3600
    prices = [4700 + 0.01 * i if i < 1800 else 4718 + 0.001 * (i - 1800) for i in range(n)]
    deltas = [60 if i < 1800 else -40 for i in range(n)]
    bars = _make_bars(prices, deltas)
    sigs_loose = generate_signals(bars, replace(CVDParams(), min_seconds_between_signals=10))
    sigs_tight = generate_signals(bars, replace(CVDParams(), min_seconds_between_signals=600))
    assert len(sigs_tight) <= len(sigs_loose)
    # Tight gating must produce a strictly monotonic >= 600s gap between emits.
    for a, b in pairwise(sigs_tight):
        assert (b.timestamp - a.timestamp).total_seconds() >= 600


def test_max_signals_per_day_caps_output():
    n = 3600
    prices = [4700 + 0.01 * i if i < 1800 else 4718 + 0.001 * (i - 1800) for i in range(n)]
    deltas = [60 if i < 1800 else -40 for i in range(n)]
    bars = _make_bars(prices, deltas)
    sigs = generate_signals(
        bars,
        replace(CVDParams(), max_signals_per_day=2, min_seconds_between_signals=60),
    )
    assert len(sigs) <= 2


def test_signals_are_in_chronological_order():
    n = 4800
    prices, deltas = [], []
    for i in range(n):
        if i < 2400:
            prices.append(4700 + 0.01 * i)
            deltas.append(60)
        else:
            prices.append(4724 + 0.001 * (i - 2400))
            deltas.append(-40)
    bars = _make_bars(prices, deltas)
    sigs = generate_signals(bars, CVDParams())
    for a, b in pairwise(sigs):
        assert a.timestamp <= b.timestamp


def test_session_gating_skips_first_and_last_window():
    """No signal should be emitted in the first ``min_minutes_since_open`` minutes."""
    n = 3600
    prices, deltas = [], []
    for i in range(n):
        if i < 1800:
            prices.append(4700 + 0.01 * i)
            deltas.append(60)
        else:
            prices.append(4718 + 0.001 * (i - 1800))
            deltas.append(-40)
    bars = _make_bars(prices, deltas)
    params = replace(CVDParams(), min_minutes_since_open=30, max_minutes_before_close=5)
    sigs = generate_signals(bars, params)
    open_ts = bars[0, "ts"]
    close_ts = bars[-1, "ts"]
    for s in sigs:
        assert (s.timestamp - open_ts).total_seconds() >= 30 * 60
        assert (close_ts - s.timestamp).total_seconds() >= 5 * 60


def test_does_not_mutate_input_bars():
    n = 1200
    bars = _make_bars(
        [4700 + 0.01 * i for i in range(n)],
        [50 if i < 600 else -50 for i in range(n)],
    )
    before_cols = list(bars.columns)
    before_height = bars.height
    _ = generate_signals(bars, CVDParams())
    assert list(bars.columns) == before_cols
    assert bars.height == before_height


def test_deterministic_for_same_inputs():
    n = 2400
    bars = _make_bars(
        [4700 + 0.01 * i if i < 1200 else 4712 + 0.002 * (i - 1200) for i in range(n)],
        [60 if i < 1200 else -40 for i in range(n)],
    )
    s1 = generate_signals(bars, CVDParams())
    s2 = generate_signals(bars, CVDParams())
    assert [(s.timestamp, s.side, s.setup_id) for s in s1] == [
        (s.timestamp, s.side, s.setup_id) for s in s2
    ]
