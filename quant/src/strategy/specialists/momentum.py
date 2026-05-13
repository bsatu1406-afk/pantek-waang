"""Momentum / Breakout specialist (Agent #8).

Three complementary setups, each emitting LONG and SHORT signals:

1. **Opening Range Breakout (ORB)** — at 09:30 ET, freeze the high/low of the
   first ``N`` minutes (default candidates: 5, 15, 30). After the OR closes,
   the first bar whose close clears ``OR_high + buffer`` is a LONG; the first
   to break below ``OR_low - buffer`` is a SHORT. One LONG + one SHORT signal
   per ``(window, side)`` per session day (so at most 6 ORB signals per day).
   Stop is parked at the opposite OR extreme; target is ``1.5 * OR_range``
   beyond entry.

2. **EOD drive (continuation)** — between ``eod_drive_start_et`` (default
   14:30 ET) and ``eod_drive_end_et`` (default 15:30 ET). If session trend is
   intact (close has been on one side of session VWAP for the last
   ``trend_lookback_min`` minutes by a clean majority) AND the last
   ``pullback_min`` minutes traced a pullback back to/through VWAP that failed
   (price reclaimed VWAP on the trend side), emit a continuation signal in
   the trend direction. Stop is parked 1 * ATR_5 against entry past VWAP,
   target is ``2 * ATR_5`` favourable. Throttled to one LONG + one SHORT in
   the EOD window per session.

3. **Multi-bar contraction breakout** — rolling 5-minute high/low range over
   the last 300 bars compared to ``ATR_atr_window_min * atr_window_min``. When
   the recent range is in the bottom ``compression_pct`` of recent volatility
   and the next bar's close pierces the contracted high (LONG) or low (SHORT)
   by at least ``breakout_buffer_ticks``, fire. Stop at the opposite extreme
   of the contracted range. Target is ``1.5 * contracted_range`` favourable.
   Global cooldown of ``multibar_cooldown_min`` minutes between signals.

The module follows the standard specialist contract:

    def generate_signals(bars: pl.DataFrame, params: MomentumParams) -> list[Signal]

The function accepts ONE RTH session day of 1-second bars, never mutates the
input, returns chronologically sorted ``Signal`` objects, and is deterministic
for any given ``(bars, params)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time

import polars as pl

from src.common.types import Side, Signal
from src.strategy.specialists._interface import SpecialistParams

# MES tick economics — kept in sync with src.simulation.engine but imported
# directly to avoid a circular dep on the engine module from a specialist.
TICK_SIZE = 0.25
ET_TZ = "America/New_York"


@dataclass
class MomentumParams(SpecialistParams):
    """Tunable parameters for the momentum specialist.

    The defaults below were chosen by inspection of ORB / breakout literature
    plus the v3 session's notes on ES. They are intentionally on the
    conservative side so the dev-period audit is not flattered by overfit.
    """

    specialist_id: str = "momentum"

    # ORB
    or_minutes: tuple[int, ...] = (5, 15, 30)
    breakout_buffer_ticks: int = 2
    orb_target_range_mult: float = 1.5
    orb_min_range_ticks: int = 4          # don't trade ORB on a dead range
    orb_max_minutes_after: int = 180      # ignore ORB break that fires post-12:30 ET

    # EOD drive
    eod_drive_start_et: str = "14:30"
    eod_drive_end_et: str = "15:30"
    trend_lookback_min: int = 30
    trend_min_pct_on_side: float = 0.60
    pullback_min: int = 5
    eod_target_atr_mult: float = 2.0
    eod_stop_atr_mult: float = 1.0

    # Multi-bar contraction breakout
    contraction_window_min: int = 5
    atr_window_min: int = 30
    compression_ratio: float = 0.45       # rolling_range/atr_baseline must be <= this
    multibar_target_range_mult: float = 1.5
    multibar_cooldown_min: int = 10
    multibar_start_et: str = "10:00"      # don't fire during OR construction window
    multibar_end_et: str = "15:45"        # stop before the EOD flat

    # Confidence shaping
    confidence_floor: float = 0.30
    confidence_ceiling: float = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_et_time(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _prep(bars: pl.DataFrame) -> pl.DataFrame:
    """Add ``et_time`` and ``minutes_since_open`` columns; sort by ``ts``.

    ``bars`` must contain ``ts`` (tz-aware UTC) plus OHLCV. ``session_date``
    is optional — if absent, we derive it from ts in ET.
    """
    df = bars.sort("ts")
    df = df.with_columns(
        pl.col("ts").dt.convert_time_zone(ET_TZ).dt.time().alias("et_time"),
        pl.col("ts").dt.convert_time_zone(ET_TZ).dt.date().alias("_et_date"),
    )
    # minutes since 09:30 ET on the session date — robust to DST
    open_et = pl.datetime(
        pl.col("ts").dt.convert_time_zone(ET_TZ).dt.year(),
        pl.col("ts").dt.convert_time_zone(ET_TZ).dt.month(),
        pl.col("ts").dt.convert_time_zone(ET_TZ).dt.day(),
        9,
        30,
        time_zone=ET_TZ,
    )
    df = df.with_columns(
        ((pl.col("ts").dt.convert_time_zone(ET_TZ) - open_et).dt.total_seconds() / 60.0)
        .alias("min_since_open")
    )
    return df


def _atr_n_minutes(df: pl.DataFrame, minutes: int) -> pl.DataFrame:
    """Append ``atr_{minutes}m`` column: rolling mean of (high-low) over
    ``minutes`` * 60 seconds of 1-s bars. Falls back to expanding mean for the
    first window."""
    w = max(1, minutes * 60)
    return df.with_columns(
        (pl.col("high") - pl.col("low"))
        .rolling_mean(window_size=w, min_samples=10)
        .alias(f"atr_{minutes}m")
    )


def _session_vwap(df: pl.DataFrame) -> pl.DataFrame:
    """Append ``vwap`` column (cumulative typical*volume / cumulative volume)."""
    return (
        df.with_columns(
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("_typical"),
        )
        .with_columns(
            (pl.col("_typical") * pl.col("volume")).cum_sum().alias("_ctpv"),
            pl.col("volume").cum_sum().alias("_cv"),
        )
        .with_columns(
            pl.when(pl.col("_cv") > 0)
            .then(pl.col("_ctpv") / pl.col("_cv"))
            .otherwise(pl.col("_typical"))
            .alias("vwap")
        )
        .drop(["_typical", "_ctpv", "_cv"])
    )


# ---------------------------------------------------------------------------
# Setup builders — each returns list[Signal]
# ---------------------------------------------------------------------------


def _orb_signals(df: pl.DataFrame, params: MomentumParams) -> list[Signal]:
    """Opening-range breakouts for each window in ``params.or_minutes``."""
    if df.is_empty():
        return []
    buf = params.breakout_buffer_ticks * TICK_SIZE
    min_range = params.orb_min_range_ticks * TICK_SIZE
    sigs: list[Signal] = []

    for n in params.or_minutes:
        or_df = df.filter(
            (pl.col("min_since_open") >= 0) & (pl.col("min_since_open") < n)
        )
        if or_df.is_empty():
            continue
        or_high = float(or_df["high"].max())
        or_low = float(or_df["low"].min())
        or_range = or_high - or_low
        if or_range < min_range:
            continue
        after = df.filter(
            (pl.col("min_since_open") >= n)
            & (pl.col("min_since_open") <= params.orb_max_minutes_after)
        )
        if after.is_empty():
            continue

        # First LONG break: close > or_high + buf
        long_hits = after.filter(pl.col("close") > or_high + buf)
        if not long_hits.is_empty():
            row = long_hits.row(0, named=True)
            entry = float(row["close"])
            stop = or_low
            target = entry + params.orb_target_range_mult * or_range
            impulse = (entry - or_high) / or_range  # fraction of range beyond
            conf = _clip(
                0.45 + 0.25 * (or_range / max(min_range, 0.25)) ** 0.5 + 0.20 * impulse,
                params.confidence_floor,
                params.confidence_ceiling,
            )
            sigs.append(
                Signal(
                    timestamp=row["ts"],
                    side=Side.LONG,
                    confidence=float(conf),
                    specialist="momentum",
                    setup_id=f"orb_{n}m_long",
                    entry_price=entry,
                    stop_price=float(stop),
                    target_price=float(target),
                    metadata={
                        "or_window_min": n,
                        "or_high": or_high,
                        "or_low": or_low,
                        "or_range": or_range,
                        "impulse_frac": float(impulse),
                    },
                )
            )

        # First SHORT break: close < or_low - buf
        short_hits = after.filter(pl.col("close") < or_low - buf)
        if not short_hits.is_empty():
            row = short_hits.row(0, named=True)
            entry = float(row["close"])
            stop = or_high
            target = entry - params.orb_target_range_mult * or_range
            impulse = (or_low - entry) / or_range
            conf = _clip(
                0.45 + 0.25 * (or_range / max(min_range, 0.25)) ** 0.5 + 0.20 * impulse,
                params.confidence_floor,
                params.confidence_ceiling,
            )
            sigs.append(
                Signal(
                    timestamp=row["ts"],
                    side=Side.SHORT,
                    confidence=float(conf),
                    specialist="momentum",
                    setup_id=f"orb_{n}m_short",
                    entry_price=entry,
                    stop_price=float(stop),
                    target_price=float(target),
                    metadata={
                        "or_window_min": n,
                        "or_high": or_high,
                        "or_low": or_low,
                        "or_range": or_range,
                        "impulse_frac": float(impulse),
                    },
                )
            )

    return sigs


def _eod_drive_signals(df: pl.DataFrame, params: MomentumParams) -> list[Signal]:
    """Trend-continuation entries after a failed pullback to VWAP."""
    if df.is_empty():
        return []
    start = _parse_et_time(params.eod_drive_start_et)
    end = _parse_et_time(params.eod_drive_end_et)
    window = df.filter((pl.col("et_time") >= start) & (pl.col("et_time") <= end))
    if window.is_empty():
        return []

    # Need full session up to each bar for VWAP + ATR — pre-compute on `df`.
    enriched = _session_vwap(df)
    enriched = _atr_n_minutes(enriched, 5)
    # boolean: close > vwap for LONG-trend confirmation
    enriched = enriched.with_columns(
        (pl.col("close") > pl.col("vwap")).alias("_above"),
        (pl.col("close") < pl.col("vwap")).alias("_below"),
    )
    # rolling fraction on each side over `trend_lookback_min`
    w = max(1, params.trend_lookback_min * 60)
    enriched = enriched.with_columns(
        pl.col("_above").cast(pl.Float64).rolling_mean(window_size=w, min_samples=30).alias("_above_pct"),
        pl.col("_below").cast(pl.Float64).rolling_mean(window_size=w, min_samples=30).alias("_below_pct"),
    )
    # pullback indicator: low <= vwap in last `pullback_min` minutes
    pw = max(1, params.pullback_min * 60)
    enriched = enriched.with_columns(
        (pl.col("low") <= pl.col("vwap")).cast(pl.Int64).rolling_sum(window_size=pw, min_samples=10).alias("_pullback_long_cnt"),
        (pl.col("high") >= pl.col("vwap")).cast(pl.Int64).rolling_sum(window_size=pw, min_samples=10).alias("_pullback_short_cnt"),
    )

    # Restrict to EOD drive window
    eod = enriched.filter((pl.col("et_time") >= start) & (pl.col("et_time") <= end))
    if eod.is_empty():
        return []

    sigs: list[Signal] = []
    # LONG continuation: trend up + pullback to/through vwap + reclaim
    long_candidates = eod.filter(
        (pl.col("_above_pct") >= params.trend_min_pct_on_side)
        & (pl.col("_pullback_long_cnt") >= 5)
        & (pl.col("close") > pl.col("vwap"))
    ).head(1)
    if not long_candidates.is_empty():
        row = long_candidates.row(0, named=True)
        entry = float(row["close"])
        atr = float(row["atr_5m"] or 0.5)
        stop = entry - params.eod_stop_atr_mult * atr
        target = entry + params.eod_target_atr_mult * atr
        conf = _clip(
            0.40 + 0.35 * (row["_above_pct"] - 0.5) + 0.20 * _clip((entry - row["vwap"]) / max(atr, 0.25), 0.0, 1.0),
            params.confidence_floor,
            params.confidence_ceiling,
        )
        sigs.append(
            Signal(
                timestamp=row["ts"],
                side=Side.LONG,
                confidence=float(conf),
                specialist="momentum",
                setup_id="eod_drive_long",
                entry_price=entry,
                stop_price=float(stop),
                target_price=float(target),
                metadata={
                    "vwap": float(row["vwap"]),
                    "above_pct": float(row["_above_pct"]),
                    "atr_5m": atr,
                },
            )
        )

    # SHORT continuation: trend down + pullback up to vwap + rejection
    short_candidates = eod.filter(
        (pl.col("_below_pct") >= params.trend_min_pct_on_side)
        & (pl.col("_pullback_short_cnt") >= 5)
        & (pl.col("close") < pl.col("vwap"))
    ).head(1)
    if not short_candidates.is_empty():
        row = short_candidates.row(0, named=True)
        entry = float(row["close"])
        atr = float(row["atr_5m"] or 0.5)
        stop = entry + params.eod_stop_atr_mult * atr
        target = entry - params.eod_target_atr_mult * atr
        conf = _clip(
            0.40 + 0.35 * (row["_below_pct"] - 0.5) + 0.20 * _clip((row["vwap"] - entry) / max(atr, 0.25), 0.0, 1.0),
            params.confidence_floor,
            params.confidence_ceiling,
        )
        sigs.append(
            Signal(
                timestamp=row["ts"],
                side=Side.SHORT,
                confidence=float(conf),
                specialist="momentum",
                setup_id="eod_drive_short",
                entry_price=entry,
                stop_price=float(stop),
                target_price=float(target),
                metadata={
                    "vwap": float(row["vwap"]),
                    "below_pct": float(row["_below_pct"]),
                    "atr_5m": atr,
                },
            )
        )

    return sigs


def _multibar_breakout_signals(df: pl.DataFrame, params: MomentumParams) -> list[Signal]:
    """Volatility-contraction → impulse breakouts (5-min range)."""
    if df.is_empty():
        return []
    start = _parse_et_time(params.multibar_start_et)
    end = _parse_et_time(params.multibar_end_et)

    enriched = _atr_n_minutes(df, params.atr_window_min)
    w_contract = max(1, params.contraction_window_min * 60)
    enriched = enriched.with_columns(
        pl.col("high").rolling_max(window_size=w_contract, min_samples=30).alias("_rng_high"),
        pl.col("low").rolling_min(window_size=w_contract, min_samples=30).alias("_rng_low"),
    )
    enriched = enriched.with_columns(
        (pl.col("_rng_high") - pl.col("_rng_low")).alias("_rng"),
    )
    enriched = enriched.with_columns(
        (pl.col("_rng") / pl.col(f"atr_{params.atr_window_min}m")).alias("_compression"),
    )

    buf = params.breakout_buffer_ticks * TICK_SIZE
    # Previous-bar contracted range for breakout detection (avoid lookahead)
    enriched = enriched.with_columns(
        pl.col("_rng_high").shift(1).alias("_rng_high_prev"),
        pl.col("_rng_low").shift(1).alias("_rng_low_prev"),
        pl.col("_rng").shift(1).alias("_rng_prev"),
        pl.col("_compression").shift(1).alias("_compression_prev"),
    )

    in_window = (pl.col("et_time") >= start) & (pl.col("et_time") <= end)
    compressed = pl.col("_compression_prev") <= params.compression_ratio
    long_break = pl.col("close") > pl.col("_rng_high_prev") + buf
    short_break = pl.col("close") < pl.col("_rng_low_prev") - buf

    cand = enriched.filter(in_window & compressed & (long_break | short_break))
    if cand.is_empty():
        return []

    # Apply global cooldown
    cooldown_s = params.multibar_cooldown_min * 60
    sigs: list[Signal] = []
    last_ts: datetime | None = None
    for row in cand.iter_rows(named=True):
        if last_ts is not None and (row["ts"] - last_ts).total_seconds() < cooldown_s:
            continue
        entry = float(row["close"])
        rng = float(row["_rng_prev"] or 0.5)
        if row["close"] > row["_rng_high_prev"]:
            side = Side.LONG
            stop = float(row["_rng_low_prev"])
            target = entry + params.multibar_target_range_mult * rng
            setup_id = "multibar_break_long"
        else:
            side = Side.SHORT
            stop = float(row["_rng_high_prev"])
            target = entry - params.multibar_target_range_mult * rng
            setup_id = "multibar_break_short"
        compression = float(row["_compression_prev"] or 1.0)
        conf = _clip(
            0.40 + 0.30 * (1.0 - compression / max(params.compression_ratio, 0.05))
            + 0.20 * _clip(abs(entry - (row["_rng_high_prev"] if side == Side.LONG else row["_rng_low_prev"])) / max(rng, 0.25), 0.0, 1.0),
            params.confidence_floor,
            params.confidence_ceiling,
        )
        sigs.append(
            Signal(
                timestamp=row["ts"],
                side=side,
                confidence=float(conf),
                specialist="momentum",
                setup_id=setup_id,
                entry_price=entry,
                stop_price=stop,
                target_price=float(target),
                metadata={
                    "contracted_range": rng,
                    "compression_ratio": compression,
                    "atr_baseline": float(row[f"atr_{params.atr_window_min}m"] or 0.0),
                },
            )
        )
        last_ts = row["ts"]
    return sigs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_signals(
    bars: pl.DataFrame, params: MomentumParams | None = None
) -> list[Signal]:
    """Emit momentum/breakout signals for one RTH session day of 1-s bars."""
    if params is None:
        params = MomentumParams()
    if bars.is_empty():
        return []
    required = {"ts", "open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"momentum: missing required bar columns: {sorted(missing)}")

    df = _prep(bars)

    orb = _orb_signals(df, params)
    eod = _eod_drive_signals(df, params)
    mb = _multibar_breakout_signals(df, params)

    all_sigs = orb + eod + mb
    all_sigs.sort(key=lambda s: s.timestamp)
    return all_sigs


__all__ = ["MomentumParams", "generate_signals"]
