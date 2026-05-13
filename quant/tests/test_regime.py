"""Unit tests for Agent #3 — Market Regime Classifier.

Tests build synthetic long-form 1-second bars in memory so they run in well
under a second and don't depend on the Databento sample being downloaded.
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.strategy.specialists.regime import (
    Regime,
    RegimeParams,
    VolBucket,
    classify_regimes,
    generate_signals,
    lookup_regime,
    persist_regime_table,
    stability_stddev_pp,
    yearly_distribution,
)

# -- helpers ------------------------------------------------------------------


def _make_session_bars(
    sd: date,
    open_p: float,
    high_p: float,
    low_p: float,
    close_p: float,
    n_bars: int = 6,
) -> pl.DataFrame:
    """Synthesize 1-second-spaced bars whose first.open / max.high / min.low /
    last.close match the requested daily OHLC. Volume is irrelevant.

    All bars are placed inside the RTH window (09:30 ET => 14:30 UTC).
    """
    base = datetime(sd.year, sd.month, sd.day, 14, 30, 0, tzinfo=UTC)
    ts = [base + timedelta(seconds=i) for i in range(n_bars)]
    # Spread OHLC across n_bars: bar 0 sets open, bar 1 sets high, bar 2 sets
    # low, last bar sets close; middle bars repeat open price.
    opens = [open_p] * n_bars
    highs = [open_p] * n_bars
    lows = [open_p] * n_bars
    closes = [open_p] * n_bars
    if n_bars >= 4:
        highs[1] = high_p
        lows[2] = low_p
        closes[-1] = close_p
        opens[-1] = close_p  # last open ~ close to keep continuity
    return pl.DataFrame(
        {
            "ts": ts,
            "session_date": [sd] * n_bars,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100] * n_bars,
        },
        schema={
            "ts": pl.Datetime("ns", "UTC"),
            "session_date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        },
    )


def _trend_series(n_days: int, start_close: float, daily_drift: float,
                  daily_range: float, base_date: date) -> pl.DataFrame:
    """Build a multi-day trending OHLC stream as 1-second bars."""
    frames = []
    prev_close = start_close
    d = base_date
    weekdays_added = 0
    while weekdays_added < n_days:
        if d.weekday() >= 5:  # skip weekends
            d = d + timedelta(days=1)
            continue
        open_p = prev_close
        close_p = open_p + daily_drift
        high_p = max(open_p, close_p) + daily_range / 2
        low_p = min(open_p, close_p) - daily_range / 2
        frames.append(_make_session_bars(d, open_p, high_p, low_p, close_p))
        prev_close = close_p
        d = d + timedelta(days=1)
        weekdays_added += 1
    return pl.concat(frames, how="vertical")


# -- tests --------------------------------------------------------------------


def test_generate_signals_stub_returns_empty() -> None:
    """The regime specialist is a labeller, not a signaller."""
    df = _make_session_bars(date(2023, 6, 1), 4500, 4510, 4490, 4505)
    assert generate_signals(df, RegimeParams()) == []
    # also tolerant of None params
    assert generate_signals(df) == []  # type: ignore[arg-type]


def test_classify_regimes_returns_one_row_per_session() -> None:
    bars = pl.concat(
        [
            _make_session_bars(date(2023, 6, 1), 4500, 4510, 4490, 4505),
            _make_session_bars(date(2023, 6, 2), 4505, 4515, 4495, 4510),
            _make_session_bars(date(2023, 6, 5), 4510, 4520, 4500, 4515),
        ],
        how="vertical",
    )
    out = classify_regimes(bars)
    assert out.height == 3
    # mandatory columns from the briefing contract
    for col in ("session_date", "regime", "vol_regime", "open", "high", "low",
                "close", "atr20"):
        assert col in out.columns, f"missing {col}"
    # all regime/vol labels must be from the enums
    regimes = set(out["regime"].to_list())
    vols = set(out["vol_regime"].to_list())
    assert regimes <= {r.value for r in Regime}
    assert vols <= {v.value for v in VolBucket}


def test_classify_regimes_detects_gap_day() -> None:
    bars = pl.concat(
        [
            _make_session_bars(date(2023, 6, 1), 4500.0, 4505.0, 4495.0, 4500.0),
            # Open jumps 1% above previous close — clear GAP_DAY
            _make_session_bars(date(2023, 6, 2), 4545.0, 4550.0, 4540.0, 4548.0),
        ],
        how="vertical",
    )
    out = classify_regimes(bars).sort("session_date")
    assert out["regime"][1] == Regime.GAP_DAY.value
    assert abs(out["gap_pct"][1] - 0.01) < 1e-9


def test_classify_regimes_detects_reversal_day() -> None:
    # Build 21 narrow-range days then one wide-range day with a tiny body.
    days = _trend_series(
        n_days=21,
        start_close=4500.0,
        daily_drift=1.0,
        daily_range=5.0,
        base_date=date(2023, 5, 1),
    )
    # Wide-range, tiny-body day: close approx open, but H-L is ~5x avg range
    last_sd = date(2023, 6, 5)
    while last_sd.weekday() >= 5:
        last_sd += timedelta(days=1)
    # find prev_close from synth data
    prev_close = days["close"].to_list()[-1]
    wide_day = _make_session_bars(
        last_sd,
        open_p=prev_close,
        high_p=prev_close + 20.0,
        low_p=prev_close - 20.0,
        close_p=prev_close + 0.05,  # ~1bp body
    )
    bars = pl.concat([days, wide_day], how="vertical")
    out = classify_regimes(bars).sort("session_date")
    # Last row should be REVERSAL_DAY (precedes HIGH_VOL_EVENT in cascade)
    assert out["regime"][-1] == Regime.REVERSAL_DAY.value


def test_classify_regimes_trending_bull_after_ramp() -> None:
    # 30 trading days with steady uptrend should produce TRENDING_BULL
    bars = _trend_series(
        n_days=30,
        start_close=4500.0,
        daily_drift=5.0,
        daily_range=8.0,
        base_date=date(2023, 4, 3),
    )
    out = classify_regimes(bars).sort("session_date")
    # Final day should be TRENDING_BULL once SMAs and slope have stabilized.
    assert out["regime"][-1] == Regime.TRENDING_BULL.value
    assert out["sma5"][-1] > out["sma20"][-1]


def test_classify_regimes_trending_bear_after_ramp_down() -> None:
    bars = _trend_series(
        n_days=30,
        start_close=4500.0,
        daily_drift=-5.0,
        daily_range=8.0,
        base_date=date(2023, 4, 3),
    )
    out = classify_regimes(bars).sort("session_date")
    assert out["regime"][-1] == Regime.TRENDING_BEAR.value
    assert out["sma5"][-1] < out["sma20"][-1]


def test_vol_bucket_quartile_assignment() -> None:
    """LOW_VOL bottom quartile, HIGH_VOL top quartile of ATR-20."""
    # 40 days with monotonically-increasing daily range — ATR-20 should be
    # increasing too, so the last 25% of rows should land in HIGH_VOL.
    frames = []
    prev_close = 4500.0
    d = date(2023, 1, 2)  # Monday
    for i in range(40):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        rng = 2.0 + i * 0.5  # widening range
        frames.append(_make_session_bars(d, prev_close,
                                         prev_close + rng / 2,
                                         prev_close - rng / 2,
                                         prev_close + 0.1))
        prev_close += 0.1
        d += timedelta(days=1)
    bars = pl.concat(frames, how="vertical")
    out = classify_regimes(bars).sort("session_date")
    head_buckets = set(out.head(8)["vol_regime"].to_list())
    tail_buckets = set(out.tail(8)["vol_regime"].to_list())
    assert VolBucket.LOW_VOL.value in head_buckets
    assert VolBucket.HIGH_VOL.value in tail_buckets
    # And every value is a legal VolBucket
    assert set(out["vol_regime"].to_list()) <= {v.value for v in VolBucket}


def test_persist_and_lookup_round_trip(tmp_path: Path) -> None:
    bars = pl.concat(
        [
            _make_session_bars(date(2023, 6, 1), 4500, 4510, 4490, 4505),
            _make_session_bars(date(2023, 6, 2), 4505, 4515, 4495, 4510),
        ],
        how="vertical",
    )
    table = classify_regimes(bars)
    out_path = tmp_path / "regime_table.parquet"
    persist_regime_table(table, out_path)
    assert out_path.exists()
    reg, vol = lookup_regime(date(2023, 6, 1), out_path)
    assert isinstance(reg, Regime)
    assert isinstance(vol, VolBucket)
    with pytest.raises(KeyError):
        lookup_regime(date(2099, 1, 1), out_path)
    with pytest.raises(FileNotFoundError):
        lookup_regime(date(2023, 6, 1), tmp_path / "does_not_exist.parquet")


def test_stability_metric_constant_distribution_is_zero() -> None:
    """If every year has identical regime shares, stddev should be 0 pp."""
    # Build a 3-year frame with identical fake regime shares.
    rows = []
    for yr in (2020, 2021, 2022):
        for reg, n in [("range", 8), ("trending_bull", 2)]:
            for i in range(n):
                rows.append({
                    "session_date": date(yr, 6, 1 + i),
                    "regime": reg,
                    "vol_regime": "med_vol",
                })
    table = pl.DataFrame(rows)
    dist = yearly_distribution(table)
    assert dist.height > 0
    assert stability_stddev_pp(table) == pytest.approx(0.0, abs=1e-9)


def test_classify_regimes_rejects_missing_columns() -> None:
    df = pl.DataFrame({"session_date": [date(2023, 6, 1)], "close": [4500.0]})
    with pytest.raises(ValueError):
        classify_regimes(df)


def test_classify_regimes_handles_empty_frame() -> None:
    empty = pl.DataFrame(
        schema={
            "ts": pl.Datetime("ns", "UTC"),
            "session_date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        }
    )
    out = classify_regimes(empty)
    assert out.is_empty()
    assert "regime" in out.columns
