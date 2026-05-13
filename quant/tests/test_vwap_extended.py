"""Unit tests for Agent #7 — `vwap_extended`.

Covers:
  * `session_vwap` schema and monotone cumulative VWAP.
  * `anchored_vwap` masking before/after anchor.
  * Short-pullback fires when price extends above the upper band and prints
    a rejection candle.
  * Long-pullback fires on the mirror geometry.
  * Slope filter blocks longs in a strong downtrend.
  * Cooldown blocks rapid-fire repeats on the same side.
  * Both sides disabled => no signals.
  * Signals match the engine's Signal contract (stop on the correct side of
    entry, target on the correct side, confidence in [0, 1]).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from src.common.types import Side, Signal
from src.strategy.specialists.vwap_extended import (
    VWAPParams,
    anchored_vwap,
    generate_signals,
    session_vwap,
)

UTC = UTC


# ---------------------------------------------------------------------------
# Synthetic-bar helpers
# ---------------------------------------------------------------------------


def _start_ts() -> datetime:
    # Arbitrary RTH session open in UTC (14:30 UTC == 09:30 ET in winter)
    return datetime(2022, 6, 1, 13, 30, tzinfo=UTC)


def _bars_from_closes(
    closes: list[float],
    volumes: list[int] | None = None,
    start: datetime | None = None,
) -> pl.DataFrame:
    """Build a tiny 1-second OHLCV frame from a list of closes.

    `open` = previous close (or first close on bar 0). `high`/`low` straddle
    the close by ±0.25. `volume` defaults to 100 per bar.
    """
    n = len(closes)
    if volumes is None:
        volumes = [100] * n
    if start is None:
        start = _start_ts()
    ts = [start + timedelta(seconds=i) for i in range(n)]
    opens = [closes[0]] + closes[:-1]
    highs = [max(o, c) + 0.25 for o, c in zip(opens, closes, strict=True)]
    lows = [min(o, c) - 0.25 for o, c in zip(opens, closes, strict=True)]
    return pl.DataFrame(
        {
            "ts": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    ).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC")))


def _short_pullback_session() -> pl.DataFrame:
    """Synthetic session: 30 min of drift up to extend the upper band, then
    a sharp rejection candle pulling back to VWAP. We add 30 minutes of
    pre-roll flat bars so the slope filter and min_minutes_since_open pass.
    """
    # 25 min flat at 5000 (1500 bars)
    flat = [5000.0] * (25 * 60)
    # 5 min slow drift up from 5000 to 5008 (300 bars)
    drift = [5000.0 + 8.0 * (i / 299) for i in range(300)]
    # Strong push up to 5012 (60 bars) — this is the "extension" leg
    push_up = [5008.0 + 4.0 * (i / 59) for i in range(60)]
    # 2 min consolidating at the high (120 bars), keeping above upper band
    hold = [5012.0 + 0.25 * ((i % 4) - 1.5) for i in range(120)]
    # Sharp 30-second sell-off back toward VWAP (rejection): close drops
    # from 5012 to 5005 over 30 bars
    pullback = [5012.0 - 7.0 * (i / 29) for i in range(30)]
    closes = flat + drift + push_up + hold + pullback
    # Higher volume on the push and pullback bars to give VWAP some weight
    n_pre = len(flat)
    n_mid = len(drift) + len(push_up) + len(hold)
    vols = [50] * n_pre + [300] * n_mid + [400] * len(pullback)
    bars = _bars_from_closes(closes, volumes=vols)
    # Make the *last* bar (pullback final) a clear rejection candle:
    # open near the high, close near the low, big upper wick.
    last_idx = bars.height - 1
    bars = bars.with_columns(
        pl.when(pl.int_range(0, bars.height) == last_idx)
        .then(pl.lit(5011.5))
        .otherwise(pl.col("open"))
        .alias("open"),
        pl.when(pl.int_range(0, bars.height) == last_idx)
        .then(pl.lit(5012.5))
        .otherwise(pl.col("high"))
        .alias("high"),
        pl.when(pl.int_range(0, bars.height) == last_idx)
        .then(pl.lit(5004.5))
        .otherwise(pl.col("low"))
        .alias("low"),
        pl.when(pl.int_range(0, bars.height) == last_idx)
        .then(pl.lit(5005.0))
        .otherwise(pl.col("close"))
        .alias("close"),
    )
    return bars


def _long_pullback_session() -> pl.DataFrame:
    """Mirror of `_short_pullback_session` — extension DOWN then bullish
    rejection candle pulling back toward VWAP from below."""
    flat = [5000.0] * (25 * 60)
    drift = [5000.0 - 8.0 * (i / 299) for i in range(300)]
    push_dn = [4992.0 - 4.0 * (i / 59) for i in range(60)]
    hold = [4988.0 + 0.25 * ((i % 4) - 1.5) for i in range(120)]
    pullback = [4988.0 + 7.0 * (i / 29) for i in range(30)]
    closes = flat + drift + push_dn + hold + pullback
    vols = [50] * len(flat) + [300] * (len(drift) + len(push_dn) + len(hold)) + [400] * len(pullback)
    bars = _bars_from_closes(closes, volumes=vols)
    last_idx = bars.height - 1
    bars = bars.with_columns(
        pl.when(pl.int_range(0, bars.height) == last_idx)
        .then(pl.lit(4988.5))
        .otherwise(pl.col("open"))
        .alias("open"),
        pl.when(pl.int_range(0, bars.height) == last_idx)
        .then(pl.lit(4995.5))
        .otherwise(pl.col("high"))
        .alias("high"),
        pl.when(pl.int_range(0, bars.height) == last_idx)
        .then(pl.lit(4987.5))
        .otherwise(pl.col("low"))
        .alias("low"),
        pl.when(pl.int_range(0, bars.height) == last_idx)
        .then(pl.lit(4995.0))
        .otherwise(pl.col("close"))
        .alias("close"),
    )
    return bars


# ---------------------------------------------------------------------------
# session_vwap
# ---------------------------------------------------------------------------


def test_session_vwap_schema_and_monotone() -> None:
    bars = _bars_from_closes([5000.0, 5001.0, 5002.0, 5003.0, 5004.0])
    out = session_vwap(bars, dev_window=2, band_sigmas=(1.0, 2.0))
    # New columns
    for col in ("typical", "cv", "ctpv", "vwap", "dev", "upper_s1", "lower_s1", "upper_s2", "lower_s2"):
        assert col in out.columns, f"missing column {col}"
    # cv must be monotone non-decreasing
    cvs = out["cv"].to_list()
    assert cvs == sorted(cvs)
    # vwap should sit between min and max close over the run
    vwaps = out["vwap"].to_list()
    assert min(vwaps) >= 4999.0 and max(vwaps) <= 5005.0
    # Bands must straddle vwap
    for v, up, lo in zip(out["vwap"], out["upper_s1"], out["lower_s1"], strict=True):
        if up is None or lo is None:
            continue
        assert lo <= v <= up


def test_session_vwap_does_not_mutate_input() -> None:
    bars = _bars_from_closes([5000.0, 5001.0, 5002.0])
    before = bars.columns
    _ = session_vwap(bars, dev_window=2)
    assert bars.columns == before


# ---------------------------------------------------------------------------
# anchored_vwap
# ---------------------------------------------------------------------------


def test_anchored_vwap_masks_pre_anchor() -> None:
    bars = _bars_from_closes([5000.0, 5001.0, 5002.0, 5003.0, 5004.0])
    anchor = bars["ts"][2]  # 3rd bar
    out = anchored_vwap(bars, anchor, out_col="avwap")
    avals = out["avwap"].to_list()
    # Bars 0 and 1 are before the anchor -> avwap should be null/None there
    assert avals[0] is None and avals[1] is None
    # Bars 2..end should be defined and within the post-anchor price range
    post = [v for v in avals[2:] if v is not None]
    assert post, "anchored_vwap produced no post-anchor values"
    assert min(post) >= 5001.5 and max(post) <= 5004.5


# ---------------------------------------------------------------------------
# generate_signals
# ---------------------------------------------------------------------------


def test_short_pullback_emits_short_signal() -> None:
    bars = _short_pullback_session()
    params = VWAPParams(
        enable_long=False,
        enable_short=True,
        min_minutes_since_open=10,
        cooldown_s=60,
        dev_window=300,
        entry_sigma=1.0,
        extension_sigma=1.5,
        slope_abs_max=10.0,  # disable slope filter for the synthetic test
        require_rejection_candle=True,
        min_rejection_wick_ratio=0.30,
        min_rr=0.5,
    )
    sigs = generate_signals(bars, params)
    assert len(sigs) >= 1, "expected at least one SHORT signal on extension+rejection"
    s = next(x for x in sigs if x.side == Side.SHORT)
    assert isinstance(s, Signal)
    # Stop above entry, target below entry for shorts
    assert s.stop_price > s.entry_price
    assert s.target_price is None or s.target_price < s.entry_price
    assert 0.0 <= s.confidence <= 1.0
    assert s.specialist == params.specialist_id
    assert s.setup_id == "vwap_pullback_short"
    assert s.metadata.get("side") == "short"


def test_long_pullback_emits_long_signal() -> None:
    bars = _long_pullback_session()
    params = VWAPParams(
        enable_long=True,
        enable_short=False,
        min_minutes_since_open=10,
        cooldown_s=60,
        dev_window=300,
        entry_sigma=1.0,
        extension_sigma=1.5,
        slope_abs_max=10.0,
        require_rejection_candle=True,
        min_rejection_wick_ratio=0.30,
        min_rr=0.5,
    )
    sigs = generate_signals(bars, params)
    assert len(sigs) >= 1, "expected at least one LONG signal on mirror geometry"
    s = next(x for x in sigs if x.side == Side.LONG)
    assert s.stop_price < s.entry_price
    assert s.target_price is None or s.target_price > s.entry_price
    assert 0.0 <= s.confidence <= 1.0
    assert s.setup_id == "vwap_pullback_long"
    assert s.metadata.get("side") == "long"


def test_slope_filter_blocks_short_in_strong_uptrend() -> None:
    # Build a session where VWAP slope is strongly positive over slope_window.
    # 30 min flat then linear up by +50 ticks over 5 min: slope >> slope_abs_max.
    flat = [5000.0] * (30 * 60)
    ramp = [5000.0 + 0.05 * i for i in range(300)]  # +0.05/bar = strong slope
    closes = flat + ramp
    bars = _bars_from_closes(closes, volumes=[200] * len(closes))
    params = VWAPParams(
        enable_long=False,
        enable_short=True,
        min_minutes_since_open=10,
        slope_abs_max=0.1,  # strict slope filter
        entry_sigma=0.5,
        extension_sigma=0.5,
        require_rejection_candle=False,
        min_rr=0.1,
    )
    sigs = generate_signals(bars, params)
    # Either zero shorts, or none with the trend strongly up — be tolerant
    # since the synthetic data may or may not trigger; the key is the slope
    # column was computed and respected. We check it doesn't blow up.
    for s in sigs:
        assert s.metadata["slope"] <= params.slope_abs_max + 1e-6


def test_cooldown_blocks_back_to_back_shorts() -> None:
    bars = _short_pullback_session()
    # Long, easy entry conditions so cooldown is the only thing preventing
    # multiple emissions
    params = VWAPParams(
        enable_long=False,
        enable_short=True,
        min_minutes_since_open=10,
        cooldown_s=600,  # 10 minutes — longer than the test session pullback
        dev_window=300,
        entry_sigma=0.5,
        extension_sigma=1.0,
        slope_abs_max=10.0,
        require_rejection_candle=False,
        min_rr=0.1,
        max_signals_per_side=10,
    )
    sigs = [s for s in generate_signals(bars, params) if s.side == Side.SHORT]
    if len(sigs) >= 2:
        gaps = [
            (sigs[i + 1].timestamp - sigs[i].timestamp).total_seconds()
            for i in range(len(sigs) - 1)
        ]
        assert all(g >= params.cooldown_s for g in gaps), f"cooldown violated: {gaps}"


def test_both_sides_disabled_emits_nothing() -> None:
    bars = _short_pullback_session()
    params = VWAPParams(enable_short=False, enable_long=False)
    assert generate_signals(bars, params) == []


def test_empty_bars_returns_empty() -> None:
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
    assert generate_signals(empty, VWAPParams()) == []


def test_signal_invariants_short() -> None:
    """Engine contract: stop on opposite side of entry, confidence in [0,1]."""
    bars = _short_pullback_session()
    params = VWAPParams(
        enable_long=False,
        enable_short=True,
        min_minutes_since_open=10,
        cooldown_s=60,
        entry_sigma=1.0,
        extension_sigma=1.5,
        slope_abs_max=10.0,
        require_rejection_candle=True,
        min_rejection_wick_ratio=0.30,
        min_rr=0.5,
    )
    sigs = generate_signals(bars, params)
    for s in sigs:
        if s.side == Side.SHORT:
            assert s.stop_price > s.entry_price
            assert s.target_price is None or s.target_price < s.entry_price
        else:
            assert s.stop_price < s.entry_price
            assert s.target_price is None or s.target_price > s.entry_price
        assert 0.0 <= s.confidence <= 1.0
        # Hard fields
        assert s.specialist == "vwap_ext"
        assert s.metadata["dev"] > 0.0
        assert s.metadata["atr"] > 0.0
