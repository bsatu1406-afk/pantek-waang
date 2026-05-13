"""Agent #3 — Market Regime Classifier.

Produces one regime label + one vol-bucket label per RTH session date for
ES futures. Other specialists / the ensemble can condition on these.

Public API (read-only contract):

    class Regime(str, Enum): TRENDING_BULL | TRENDING_BEAR | RANGE
                              | HIGH_VOL_EVENT | LOW_VOL_DRIFT
                              | GAP_DAY | REVERSAL_DAY
    class VolBucket(str, Enum): LOW_VOL | MED_VOL | HIGH_VOL

    classify_regimes(bars: pl.DataFrame) -> pl.DataFrame
        Input: long-form 1-second (or any intraday) bars with
            ts, session_date, open, high, low, close, volume
        Output: one row per session_date with columns
            session_date, regime, vol_regime, open, high, low, close,
            atr20, daily_range, range_ratio_20, sma5, sma20, sma5_slope,
            gap_pct, body_pct, prev_close.

    lookup_regime(d) -> (Regime, VolBucket)
        Reads the persisted regime table at
        ``data/processed/regime/regime_table.parquet``.

    generate_signals(bars, params) -> []
        REQUIRED STUB so the standalone audit harness doesn't crash.
        The regime specialist does NOT emit trade signals.

All math is done with Polars expressions over per-day aggregates — no per-row
Python loops.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path

import polars as pl

from src.common.types import Signal
from src.strategy.specialists._interface import SpecialistParams

REPO_ROOT = Path(__file__).resolve().parents[3]
REGIME_TABLE_PATH = REPO_ROOT / "data" / "processed" / "regime" / "regime_table.parquet"


class Regime(str, Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGE = "range"
    HIGH_VOL_EVENT = "high_vol_event"
    LOW_VOL_DRIFT = "low_vol_drift"
    GAP_DAY = "gap_day"
    REVERSAL_DAY = "reversal_day"


class VolBucket(str, Enum):
    LOW_VOL = "low_vol"
    MED_VOL = "med_vol"
    HIGH_VOL = "high_vol"


@dataclass
class RegimeParams(SpecialistParams):
    """Tunable thresholds for the regime classifier.

    All defaults were selected from prior-period (pre-2024) ES behaviour and
    sanity-checked against published ATR characteristics. They are intentionally
    conservative so the regime label is stable.
    """

    specialist_id: str = "regime"
    atr_window: int = 20                 # daily ATR lookback
    sma_short: int = 5                   # short trend MA length
    sma_long: int = 20                   # long trend MA length
    sma_slope_window: int = 5            # bars over which SMA5 slope is measured
    gap_threshold_pct: float = 0.005     # 0.5% gap on the open
    reversal_body_bp: float = 10.0       # |close-open|/open < 10bp = doji body
    reversal_range_mult: float = 2.0     # day range > 2x 20d avg = wide-range bar
    high_vol_event_range_mult: float = 1.5
    low_vol_drift_range_mult: float = 0.5
    trend_min_separation_pct: float = 0.0  # SMA5 - SMA20 must differ by this much


# -- core classification ------------------------------------------------------


def _aggregate_daily(bars: pl.DataFrame) -> pl.DataFrame:
    """Collapse intraday bars to per-session-date OHLCV.

    The aggregation is order-aware: open = first open in the RTH session, close
    = last close, high/low are full-session extremes, volume is the sum.
    """
    if bars.is_empty():
        return pl.DataFrame(
            schema={
                "session_date": pl.Date,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            }
        )
    # Make sure rows are time-ordered within each session before agg.first/last.
    bars = bars.sort(["session_date", "ts"])
    daily = bars.group_by("session_date").agg(
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    ).sort("session_date")
    return daily


def _annotate_features(
    daily: pl.DataFrame,
    params: RegimeParams,
) -> pl.DataFrame:
    """Add trend / vol / gap / reversal features (all Polars expressions)."""
    if daily.is_empty():
        return daily
    # True range = max(high-low, |high-prev_close|, |low-prev_close|)
    daily = daily.with_columns(
        pl.col("close").shift(1).alias("prev_close"),
    )
    daily = daily.with_columns(
        pl.max_horizontal([
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("prev_close")).abs(),
            (pl.col("low") - pl.col("prev_close")).abs(),
        ]).alias("true_range"),
        (pl.col("high") - pl.col("low")).alias("daily_range"),
    )
    # ATR-20 (simple mean of true_range over window). Use min_periods=1 to keep
    # early-session rows usable; they'll be classified as RANGE until enough
    # history is built up.
    daily = daily.with_columns(
        pl.col("true_range")
        .rolling_mean(window_size=params.atr_window, min_periods=1)
        .alias("atr20"),
        pl.col("daily_range")
        .rolling_mean(window_size=params.atr_window, min_periods=1)
        .alias("range_avg_20"),
        pl.col("close")
        .rolling_mean(window_size=params.sma_short, min_periods=1)
        .alias("sma5"),
        pl.col("close")
        .rolling_mean(window_size=params.sma_long, min_periods=1)
        .alias("sma20"),
    )
    # SMA5 slope = sma5 today - sma5 N days ago (price units)
    daily = daily.with_columns(
        (pl.col("sma5") - pl.col("sma5").shift(params.sma_slope_window))
        .alias("sma5_slope"),
        (pl.col("daily_range") / pl.col("range_avg_20")).alias("range_ratio_20"),
        # Gap pct vs prev_close
        ((pl.col("open") - pl.col("prev_close")) / pl.col("prev_close"))
        .alias("gap_pct"),
        # Body (close - open) as fraction of open
        ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("body_pct"),
    )
    return daily


def _label_vol_bucket(daily: pl.DataFrame) -> pl.DataFrame:
    """Tag each row LOW_VOL / MED_VOL / HIGH_VOL by ATR-20 quartile.

    Quartiles are computed across the entire input frame (the caller is
    expected to pass a frame large enough — at least one full year — for the
    quartile to be meaningful). When the frame is too small, everything falls
    back to MED_VOL.
    """
    if daily.is_empty():
        return daily.with_columns(pl.lit(None, dtype=pl.Utf8).alias("vol_regime"))
    n = daily.height
    if n < 8:
        return daily.with_columns(
            pl.lit(VolBucket.MED_VOL.value, dtype=pl.Utf8).alias("vol_regime")
        )
    q1 = daily.select(pl.col("atr20").quantile(0.25)).item()
    q3 = daily.select(pl.col("atr20").quantile(0.75)).item()
    if q1 is None or q3 is None:
        return daily.with_columns(
            pl.lit(VolBucket.MED_VOL.value, dtype=pl.Utf8).alias("vol_regime")
        )
    bucket = (
        pl.when(pl.col("atr20") <= q1)
        .then(pl.lit(VolBucket.LOW_VOL.value))
        .when(pl.col("atr20") >= q3)
        .then(pl.lit(VolBucket.HIGH_VOL.value))
        .otherwise(pl.lit(VolBucket.MED_VOL.value))
    )
    return daily.with_columns(bucket.alias("vol_regime"))


def _label_regime(daily: pl.DataFrame, params: RegimeParams) -> pl.DataFrame:
    """Assign a single Regime per row using a priority cascade.

    Precedence (top-most wins):
        1. GAP_DAY      — open gaps > threshold vs prev close
        2. REVERSAL_DAY — tiny body but wide range
        3. HIGH_VOL_EVENT — day range >> 20d average range
        4. LOW_VOL_DRIFT — day range << 20d average range
        5. TRENDING_BULL — SMA5 > SMA20 and SMA5 slope > 0
        6. TRENDING_BEAR — SMA5 < SMA20 and SMA5 slope < 0
        7. RANGE         — fallback
    """
    if daily.is_empty():
        return daily.with_columns(pl.lit(None, dtype=pl.Utf8).alias("regime"))

    body_thresh = params.reversal_body_bp / 10_000.0  # bp -> fraction
    gap_thresh = params.gap_threshold_pct

    is_gap = pl.col("gap_pct").abs() > gap_thresh
    is_reversal = (
        (pl.col("body_pct").abs() < body_thresh)
        & (pl.col("range_ratio_20") > params.reversal_range_mult)
    )
    is_high_vol_event = pl.col("range_ratio_20") > params.high_vol_event_range_mult
    is_low_vol_drift = pl.col("range_ratio_20") < params.low_vol_drift_range_mult
    sma_diff = pl.col("sma5") - pl.col("sma20")
    min_sep = params.trend_min_separation_pct * pl.col("sma20")
    is_bull = (sma_diff > min_sep) & (pl.col("sma5_slope") > 0)
    is_bear = (sma_diff < -min_sep) & (pl.col("sma5_slope") < 0)

    regime_expr = (
        pl.when(is_gap)
        .then(pl.lit(Regime.GAP_DAY.value))
        .when(is_reversal)
        .then(pl.lit(Regime.REVERSAL_DAY.value))
        .when(is_high_vol_event)
        .then(pl.lit(Regime.HIGH_VOL_EVENT.value))
        .when(is_low_vol_drift)
        .then(pl.lit(Regime.LOW_VOL_DRIFT.value))
        .when(is_bull)
        .then(pl.lit(Regime.TRENDING_BULL.value))
        .when(is_bear)
        .then(pl.lit(Regime.TRENDING_BEAR.value))
        .otherwise(pl.lit(Regime.RANGE.value))
    )
    return daily.with_columns(regime_expr.alias("regime"))


def classify_regimes(
    bars: pl.DataFrame,
    params: RegimeParams | None = None,
) -> pl.DataFrame:
    """Run the full regime + vol bucket pipeline over long-form intraday bars.

    Parameters
    ----------
    bars: long-form Polars DataFrame with columns
        ts, session_date, open, high, low, close, volume.
    params: optional RegimeParams. Defaults are conservative and tuned on
        2020-2023 data only (no 2024+ leakage).

    Returns
    -------
    pl.DataFrame, one row per session_date, columns:
        session_date, regime, vol_regime, open, high, low, close, volume,
        prev_close, true_range, daily_range, atr20, range_avg_20,
        range_ratio_20, sma5, sma20, sma5_slope, gap_pct, body_pct.
    """
    if params is None:
        params = RegimeParams()
    if bars.is_empty():
        return pl.DataFrame(
            schema={
                "session_date": pl.Date,
                "regime": pl.Utf8,
                "vol_regime": pl.Utf8,
            }
        )
    required = {"session_date", "open", "high", "low", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"classify_regimes: bars missing columns: {missing}")
    daily = _aggregate_daily(bars)
    daily = _annotate_features(daily, params)
    daily = _label_vol_bucket(daily)
    daily = _label_regime(daily, params)
    # Stable column ordering for downstream consumers.
    cols_first = ["session_date", "regime", "vol_regime",
                  "open", "high", "low", "close", "volume",
                  "prev_close", "true_range", "daily_range",
                  "atr20", "range_avg_20", "range_ratio_20",
                  "sma5", "sma20", "sma5_slope",
                  "gap_pct", "body_pct"]
    cols = [c for c in cols_first if c in daily.columns]
    cols += [c for c in daily.columns if c not in cols]
    return daily.select(cols)


# -- persistence + lookup -----------------------------------------------------


def persist_regime_table(
    table: pl.DataFrame,
    path: Path | None = None,
) -> Path:
    """Write the regime table to Parquet for fast cross-specialist lookup."""
    out = Path(path) if path is not None else REGIME_TABLE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out)
    return out


def lookup_regime(
    d: date,
    path: Path | None = None,
) -> tuple[Regime, VolBucket]:
    """Read the persisted regime table and return (regime, vol_bucket) for ``d``.

    Raises ``FileNotFoundError`` if the table hasn't been built yet, and
    ``KeyError`` if ``d`` isn't a classified session day.
    """
    src = Path(path) if path is not None else REGIME_TABLE_PATH
    if not src.exists():
        raise FileNotFoundError(
            f"regime_table.parquet missing at {src} — run classify_regimes "
            "and persist_regime_table first"
        )
    df = pl.scan_parquet(src).filter(pl.col("session_date") == d).collect()
    if df.is_empty():
        raise KeyError(f"no regime classified for session_date={d}")
    row = df.row(0, named=True)
    return Regime(row["regime"]), VolBucket(row["vol_regime"])


# -- stability metric (briefing-specific verdict) -----------------------------


def yearly_distribution(table: pl.DataFrame) -> pl.DataFrame:
    """Return a (year, regime) -> share matrix for stability inspection."""
    if table.is_empty():
        return pl.DataFrame(schema={"year": pl.Int32, "regime": pl.Utf8, "share": pl.Float64})
    df = table.with_columns(pl.col("session_date").dt.year().alias("year"))
    counts = df.group_by(["year", "regime"]).agg(pl.len().alias("n"))
    totals = df.group_by("year").agg(pl.len().alias("total"))
    return counts.join(totals, on="year").with_columns(
        (pl.col("n") / pl.col("total")).alias("share")
    ).sort(["year", "regime"])


def stability_stddev_pp(table: pl.DataFrame) -> float:
    """Pooled per-regime year-over-year share standard deviation, in PP.

    For each regime, compute the std-dev of its yearly share across all years
    present in ``table``. Average those std-devs and return as percentage points
    (e.g. 0.06 -> 6.0pp).
    """
    dist = yearly_distribution(table)
    if dist.is_empty():
        return 0.0
    per_regime = dist.group_by("regime").agg(pl.col("share").std().alias("std"))
    avg_std = per_regime["std"].mean()
    if avg_std is None:
        return 0.0
    return float(avg_std) * 100.0  # fraction -> percentage points


# -- signal stub --------------------------------------------------------------


def generate_signals(bars: pl.DataFrame, params: RegimeParams | None = None) -> list[Signal]:
    """Required by the specialist contract; regime never emits trade signals."""
    _ = bars
    _ = params
    return []
