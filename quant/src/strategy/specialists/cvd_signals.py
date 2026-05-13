"""Order Flow / CVD specialist (Agent #4).

Emits trading signals based on three named order-flow setups:

* ``cvd_bearish_divergence`` / ``cvd_bullish_divergence``: price makes a fresh
  swing high (low) inside a 5-minute window while cumulative volume delta
  fails to confirm the swing — i.e. the move is *not* being supported by
  net aggressive buying (selling). We fade the unsupported leg.

* ``absorption_short`` / ``absorption_long``: heavy one-sided aggressive
  volume inside a tight ATR-normalised price range. The dominant side is
  *failing to move price* — taken as institutional absorption — so we
  trade against the exhausted side.

* ``exhaustion_spike_short`` / ``exhaustion_spike_long``: an extreme
  per-second signed-volume burst (z-score against trailing window) that
  is *not* immediately followed by a new price extreme. The buyer (seller)
  has spent their ammunition and price reverses.

The standalone audit harness only feeds ``bars_1s`` rows to specialists, so
:func:`generate_signals` lazy-loads the matching ``cvd_1s`` parquet for the
session day via :func:`src.data.loader.load_cvd_1s` and joins on ``ts``.
Tests can short-circuit the loader by passing bars that already contain the
``delta`` / ``cvd`` / ``buy_vol`` / ``sell_vol`` columns.

All heavy computations are Polars expressions; only the per-row scan that
applies the per-detector gating logic falls back to Python.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import polars as pl

from src.common.types import Side, Signal
from src.data.loader import load_cvd_1s
from src.strategy.specialists._interface import SpecialistParams

CVD_COLUMNS = ("delta", "cvd", "buy_vol", "sell_vol")


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


@dataclass
class CVDParams(SpecialistParams):
    specialist_id: str = "cvd"

    # Session gating
    min_minutes_since_open: int = 15
    max_minutes_before_close: int = 10
    max_signals_per_day: int = 8
    min_seconds_between_signals: int = 180

    # Risk
    atr_window: int = 300  # rolling mean of (high - low) over this many 1s bars
    atr_floor: float = 0.25  # protect against degenerate ATR (<1 tick)
    stop_atr_mult: float = 1.0
    target_atr_mult: float = 1.5

    # CVD divergence
    enable_divergence: bool = True
    divergence_lookback: int = 300              # length of the comparison window
    divergence_pivot_window: int = 60           # window for the "current" swing
    divergence_min_cvd_drop_pct: float = 0.15   # CVD must regress at least this %

    # Absorption
    enable_absorption: bool = True
    absorption_window: int = 60                 # rolling volume / range window
    absorption_min_volume: int = 1500
    absorption_max_range_atr: float = 0.40
    absorption_min_imbalance: float = 0.62

    # Exhaustion spike
    enable_exhaustion: bool = True
    exhaustion_zscore_window: int = 300
    exhaustion_zscore: float = 3.0
    exhaustion_confirm_bars: int = 30           # bars after spike that must fail


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_date_of(bars: pl.DataFrame) -> date | None:
    """Pull the session date from the first row, tolerating missing column."""
    if "session_date" not in bars.columns or bars.is_empty():
        return None
    val = bars[0, "session_date"]
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    return None


def _attach_cvd(bars: pl.DataFrame) -> pl.DataFrame:
    """Return ``bars`` with cvd_1s columns joined in (inner join on ``ts``)."""
    if all(c in bars.columns for c in CVD_COLUMNS):
        return bars
    sd = _session_date_of(bars)
    if sd is None:
        return bars
    cvd = load_cvd_1s(sd)
    if cvd is None or cvd.is_empty():
        return bars  # caller checks for missing CVD cols
    keep = ["ts"] + [c for c in CVD_COLUMNS if c in cvd.columns]
    cvd = cvd.select(keep).unique(subset=["ts"], keep="last")
    return bars.join(cvd, on="ts", how="left").with_columns(
        [pl.col(c).fill_null(0) for c in CVD_COLUMNS if c in cvd.columns]
    )


def _enrich(df: pl.DataFrame, params: CVDParams) -> pl.DataFrame:
    """Compute rolling features used by all three detectors in a single pass."""
    aw = params.atr_window
    aw_w = max(aw, 5)
    div_lb = params.divergence_lookback
    pv = max(params.divergence_pivot_window, 5)
    abw = max(params.absorption_window, 5)
    zw = max(params.exhaustion_zscore_window, 30)

    # Re-build cumulative CVD inside the dataframe so a join with NULLs can't
    # break monotonicity. We treat absent ``delta`` as 0.
    df = df.with_columns(
        [
            pl.col("delta").cast(pl.Float64).fill_null(0).alias("delta"),
            pl.col("buy_vol").cast(pl.Float64).fill_null(0).alias("buy_vol"),
            pl.col("sell_vol").cast(pl.Float64).fill_null(0).alias("sell_vol"),
        ]
    )
    df = df.with_columns(pl.col("delta").cum_sum().alias("cvd"))

    df = df.with_columns(
        [
            (pl.col("high") - pl.col("low"))
            .rolling_mean(window_size=aw_w, min_samples=1)
            .alias("atr"),
            # Divergence: rolling max/min of price + CVD inside the *current* and
            # *prior* swing windows. The prior swing is the window that ends
            # (lookback - pivot) bars ago — implemented via shift.
            pl.col("close").rolling_max(window_size=pv, min_samples=1).alias("close_hi_now"),
            pl.col("close").rolling_min(window_size=pv, min_samples=1).alias("close_lo_now"),
            pl.col("cvd").rolling_max(window_size=pv, min_samples=1).alias("cvd_hi_now"),
            pl.col("cvd").rolling_min(window_size=pv, min_samples=1).alias("cvd_lo_now"),
            pl.col("close")
            .rolling_max(window_size=pv, min_samples=1)
            .shift(div_lb - pv)
            .alias("close_hi_prev"),
            pl.col("close")
            .rolling_min(window_size=pv, min_samples=1)
            .shift(div_lb - pv)
            .alias("close_lo_prev"),
            pl.col("cvd")
            .rolling_max(window_size=pv, min_samples=1)
            .shift(div_lb - pv)
            .alias("cvd_hi_prev"),
            pl.col("cvd")
            .rolling_min(window_size=pv, min_samples=1)
            .shift(div_lb - pv)
            .alias("cvd_lo_prev"),
            # Absorption window: 1-min sums and range.
            pl.col("buy_vol").rolling_sum(window_size=abw, min_samples=1).alias("buy_sum"),
            pl.col("sell_vol").rolling_sum(window_size=abw, min_samples=1).alias("sell_sum"),
            pl.col("high").rolling_max(window_size=abw, min_samples=1).alias("win_high"),
            pl.col("low").rolling_min(window_size=abw, min_samples=1).alias("win_low"),
            # Exhaustion: rolling stats on per-second signed delta.
            pl.col("delta").rolling_mean(window_size=zw, min_samples=10).alias("delta_mu"),
            pl.col("delta").rolling_std(window_size=zw, min_samples=10).alias("delta_sd"),
        ]
    )
    df = df.with_columns(
        [
            pl.when(pl.col("atr") < params.atr_floor)
            .then(pl.lit(params.atr_floor))
            .otherwise(pl.col("atr"))
            .alias("atr"),
            (pl.col("buy_sum") + pl.col("sell_sum")).alias("vol_sum"),
            (pl.col("win_high") - pl.col("win_low")).alias("win_range"),
        ]
    )
    df = df.with_columns(
        [
            pl.when(pl.col("vol_sum") > 0)
            .then(pl.col("buy_sum") / pl.col("vol_sum"))
            .otherwise(pl.lit(0.5))
            .alias("buy_share"),
            (pl.col("win_range") / pl.col("atr")).alias("range_atr"),
            pl.when(pl.col("delta_sd") > 0)
            .then((pl.col("delta") - pl.col("delta_mu")) / pl.col("delta_sd"))
            .otherwise(pl.lit(0.0))
            .alias("delta_z"),
            pl.col("high").rolling_max(window_size=params.exhaustion_confirm_bars, min_samples=1)
            .shift(-params.exhaustion_confirm_bars)
            .alias("fwd_high"),
            pl.col("low").rolling_min(window_size=params.exhaustion_confirm_bars, min_samples=1)
            .shift(-params.exhaustion_confirm_bars)
            .alias("fwd_low"),
        ]
    )
    return df


# ---------------------------------------------------------------------------
# Public detectors (also re-exported for tests / standalone use)
# ---------------------------------------------------------------------------


def detect_divergence(df: pl.DataFrame, params: CVDParams) -> list[tuple[datetime, Side, float]]:
    """Return candidate ``(ts, side, score)`` triples for CVD-price divergence."""
    if not params.enable_divergence:
        return []
    needed = ("close_hi_now", "close_hi_prev", "cvd_hi_now", "cvd_hi_prev",
              "close_lo_now", "close_lo_prev", "cvd_lo_now", "cvd_lo_prev")
    if any(c not in df.columns for c in needed):
        return []
    pct = params.divergence_min_cvd_drop_pct

    bear = df.filter(
        (pl.col("close_hi_now") > pl.col("close_hi_prev"))
        & (pl.col("cvd_hi_now") < pl.col("cvd_hi_prev"))
        & (pl.col("cvd_hi_prev").abs() > 1)
        & ((pl.col("cvd_hi_prev") - pl.col("cvd_hi_now")).abs()
           >= pct * pl.col("cvd_hi_prev").abs())
    )
    bull = df.filter(
        (pl.col("close_lo_now") < pl.col("close_lo_prev"))
        & (pl.col("cvd_lo_now") > pl.col("cvd_lo_prev"))
        & (pl.col("cvd_lo_prev").abs() > 1)
        & ((pl.col("cvd_lo_now") - pl.col("cvd_lo_prev")).abs()
           >= pct * pl.col("cvd_lo_prev").abs())
    )
    out: list[tuple[datetime, Side, float]] = []
    for row in bear.iter_rows(named=True):
        prev = abs(row["cvd_hi_prev"]) or 1.0
        score = min(1.0, abs(row["cvd_hi_prev"] - row["cvd_hi_now"]) / prev)
        out.append((row["ts"], Side.SHORT, float(score)))
    for row in bull.iter_rows(named=True):
        prev = abs(row["cvd_lo_prev"]) or 1.0
        score = min(1.0, abs(row["cvd_lo_now"] - row["cvd_lo_prev"]) / prev)
        out.append((row["ts"], Side.LONG, float(score)))
    return out


def detect_absorption(df: pl.DataFrame, params: CVDParams) -> list[tuple[datetime, Side, float]]:
    """Return absorption setups: heavy one-sided flow inside a tight range."""
    if not params.enable_absorption:
        return []
    if any(c not in df.columns for c in ("vol_sum", "buy_share", "range_atr")):
        return []
    mv = params.absorption_min_volume
    rmax = params.absorption_max_range_atr
    imb = params.absorption_min_imbalance
    short_df = df.filter(
        (pl.col("vol_sum") >= mv)
        & (pl.col("range_atr") <= rmax)
        & (pl.col("buy_share") >= imb)
    )
    long_df = df.filter(
        (pl.col("vol_sum") >= mv)
        & (pl.col("range_atr") <= rmax)
        & ((1.0 - pl.col("buy_share")) >= imb)
    )
    out: list[tuple[datetime, Side, float]] = []
    for row in short_df.iter_rows(named=True):
        score = float(min(1.0, (row["buy_share"] - 0.5) * 2))
        out.append((row["ts"], Side.SHORT, score))
    for row in long_df.iter_rows(named=True):
        score = float(min(1.0, (0.5 - row["buy_share"]) * 2))
        out.append((row["ts"], Side.LONG, score))
    return out


def detect_exhaustion(df: pl.DataFrame, params: CVDParams) -> list[tuple[datetime, Side, float]]:
    """Return exhaustion spikes that failed to make a new extreme."""
    if not params.enable_exhaustion:
        return []
    if any(c not in df.columns for c in ("delta_z", "fwd_high", "fwd_low", "high", "low")):
        return []
    z = params.exhaustion_zscore
    pos = df.filter(
        (pl.col("delta_z") >= z) & (pl.col("fwd_high").is_not_null())
        & (pl.col("fwd_high") <= pl.col("high"))
    )
    neg = df.filter(
        (pl.col("delta_z") <= -z) & (pl.col("fwd_low").is_not_null())
        & (pl.col("fwd_low") >= pl.col("low"))
    )
    out: list[tuple[datetime, Side, float]] = []
    for row in pos.iter_rows(named=True):
        out.append((row["ts"], Side.SHORT, float(min(1.0, abs(row["delta_z"]) / 6.0))))
    for row in neg.iter_rows(named=True):
        out.append((row["ts"], Side.LONG, float(min(1.0, abs(row["delta_z"]) / 6.0))))
    return out


# ---------------------------------------------------------------------------
# generate_signals — required public entrypoint
# ---------------------------------------------------------------------------


_SETUP_BY_DETECTOR = {
    "divergence": {Side.SHORT: "cvd_bearish_divergence", Side.LONG: "cvd_bullish_divergence"},
    "absorption": {Side.SHORT: "absorption_short", Side.LONG: "absorption_long"},
    "exhaustion": {Side.SHORT: "exhaustion_spike_short", Side.LONG: "exhaustion_spike_long"},
}


def _signal_from_row(
    detector: str, ts: datetime, side: Side, score: float,
    row: dict, params: CVDParams,
) -> Signal:
    atr = max(float(row.get("atr") or params.atr_floor), params.atr_floor)
    entry = float(row["close"])
    if side == Side.LONG:
        stop = entry - atr * params.stop_atr_mult
        target = entry + atr * params.target_atr_mult
    else:
        stop = entry + atr * params.stop_atr_mult
        target = entry - atr * params.target_atr_mult
    setup = _SETUP_BY_DETECTOR[detector][side]
    return Signal(
        timestamp=ts,
        side=side,
        confidence=float(max(0.0, min(1.0, score))),
        specialist="cvd",
        setup_id=setup,
        entry_price=entry,
        stop_price=float(stop),
        target_price=float(target),
        metadata={
            "detector": detector,
            "atr": atr,
            "cvd": float(row.get("cvd") or 0.0),
            "delta_z": float(row.get("delta_z") or 0.0),
            "buy_share": float(row.get("buy_share") or 0.5),
            "range_atr": float(row.get("range_atr") or 0.0),
        },
    )


def _within_session_window(ts: datetime, open_ts: datetime, close_ts: datetime,
                           params: CVDParams) -> bool:
    if (ts - open_ts).total_seconds() < params.min_minutes_since_open * 60:
        return False
    if (close_ts - ts).total_seconds() < params.max_minutes_before_close * 60:
        return False
    return True


def generate_signals(bars: pl.DataFrame, params: CVDParams | None = None) -> list[Signal]:
    """Emit Signals for one RTH session day.

    ``bars`` is a 1-second OHLCV frame for a single session. CVD aggregates
    are joined in either from the provided dataframe (test path) or from
    ``data/processed/cvd_1s/<sd>.parquet`` via :func:`load_cvd_1s` (audit path).
    """
    if params is None:
        params = CVDParams()
    if bars is None or bars.is_empty():
        return []
    df = bars.sort("ts")
    df = _attach_cvd(df)
    if not all(c in df.columns for c in ("delta", "buy_vol", "sell_vol")):
        return []  # no CVD data available for this session
    df = _enrich(df, params)

    open_ts = df[0, "ts"]
    close_ts = df[-1, "ts"]

    candidates: list[tuple[datetime, Side, float, str]] = []
    for ts, side, score in detect_divergence(df, params):
        candidates.append((ts, side, score, "divergence"))
    for ts, side, score in detect_absorption(df, params):
        candidates.append((ts, side, score, "absorption"))
    for ts, side, score in detect_exhaustion(df, params):
        candidates.append((ts, side, score, "exhaustion"))

    if not candidates:
        return []
    candidates.sort(key=lambda t: (t[0], -t[2]))

    row_by_ts: dict[datetime, dict] = {r["ts"]: r for r in df.iter_rows(named=True)}

    signals: list[Signal] = []
    last_emit: datetime | None = None
    gap = timedelta(seconds=params.min_seconds_between_signals)
    for ts, side, score, detector in candidates:
        if not _within_session_window(ts, open_ts, close_ts, params):
            continue
        if last_emit is not None and ts - last_emit < gap:
            continue
        row = row_by_ts.get(ts)
        if row is None:
            continue
        sig = _signal_from_row(detector, ts, side, score, row, params)
        signals.append(sig)
        last_emit = ts
        if len(signals) >= params.max_signals_per_day:
            break
    return signals


__all__ = [
    "CVDParams",
    "detect_absorption",
    "detect_divergence",
    "detect_exhaustion",
    "generate_signals",
]
