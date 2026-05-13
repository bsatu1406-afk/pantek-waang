"""Agent #7 — VWAP & Mean Reversion specialist (extended).

Builds on the Orchestrator's `baseline_short_vwap` smoke specialist and adds:

* Session VWAP with a rolling-band std proxy (`session_vwap`).
* Anchored VWAP for arbitrary anchor timestamps (`anchored_vwap`).
* Both SHORT and LONG pullback setups with symmetric geometry.
* Rejection-candle confirmation (we don't have CVD in `bars`, so candle
  shape is the aggression proxy).
* VWAP slope filter — avoid shorting strong uptrends / longing strong
  downtrends.
* Per-setup metadata so the ensemble runner / ML ranker can mask sides.

v3-session note: the LONG side of VWAP pullbacks was structurally negative
in the 2025 sample. We emit both sides honestly here; the standalone audit
reports per-side WR/PF so the ensemble can disable the loser side via
`metadata.side` masking.

Contract (per `src/strategy/specialists/_interface.py`):

    def generate_signals(bars, params: VWAPParams) -> list[Signal]

`bars` is a 1-second OHLCV frame for ONE session day (RTH-filtered, UTC ts).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl

from src.common.types import Side, Signal
from src.strategy.specialists._interface import SpecialistParams

# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


@dataclass
class VWAPParams(SpecialistParams):
    """Tuning knobs for `vwap_extended`.

    Defaults are picked from the v3-session SHORT-only baseline and then
    softened for the LONG mirror. They are dev-period (2020-2023) tunings;
    NEVER tune these against 2024+ data and report as final.
    """

    specialist_id: str = "vwap_ext"

    # --- band geometry ---
    band_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0)
    # dev_window: rolling window (in 1s bars) used to compute the band std
    dev_window: int = 300
    # entry band: signal fires when |close - vwap| / dev >= entry_sigma
    entry_sigma: float = 1.5
    # extension band: a recent bar must have crossed at least this many sigmas
    extension_sigma: float = 2.0
    # extension lookback (seconds) — how recently the extension must have happened
    extension_lookback_s: int = 300

    # --- exit geometry ---
    stop_atr_mult: float = 1.0
    target_atr_mult: float = 1.5
    atr_window: int = 60  # bars used to compute ATR proxy
    # alternative: use swing high/low for stop; if False, use atr stop
    use_swing_stop: bool = True
    swing_lookback: int = 60  # seconds of recent bars to compute swing hi/lo

    # --- filters ---
    min_minutes_since_open: int = 30
    # cooldown between consecutive signals (same side)
    cooldown_s: int = 300
    # max signals per side per day
    max_signals_per_side: int = 4
    # VWAP slope filter: short only when slope <= +slope_abs_max,
    # long only when slope >= -slope_abs_max (units: price per bar).
    slope_window: int = 300
    slope_abs_max: float = 0.5
    # require rejection-candle confirmation (close < open for shorts; close > open for longs)
    require_rejection_candle: bool = True
    # rejection wick threshold (upper_wick / range for shorts)
    min_rejection_wick_ratio: float = 0.35
    # minimum reward:risk to accept a signal
    min_rr: float = 1.0

    # --- sides ---
    enable_short: bool = True
    enable_long: bool = True

    # --- numeric guards ---
    min_atr: float = 0.25      # 1 tick on ES; avoids degenerate stops
    min_dev: float = 0.10      # avoid div-by-zero on extremely tight ranges


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def session_vwap(
    bars: pl.DataFrame,
    dev_window: int = 300,
    band_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> pl.DataFrame:
    """Annotate `bars` with session VWAP and rolling-dev bands.

    Adds columns:
      * `typical`       — (h + l + c) / 3
      * `cv`, `ctpv`    — running cumulative volume / typical*volume
      * `vwap`          — cumulative session VWAP
      * `dev`           — rolling std of (close - vwap) over `dev_window` bars
      * `upper_sN`      — vwap + N * dev for each N in band_sigmas
      * `lower_sN`      — vwap - N * dev for each N in band_sigmas

    Assumes `bars` is one contiguous session day. We do NOT modify the input.
    """
    if bars.is_empty():
        return bars

    df = bars.sort("ts").with_columns(
        [
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("typical"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("volume").cum_sum().alias("cv"),
            (pl.col("typical") * pl.col("volume")).cum_sum().alias("ctpv"),
        ]
    )
    # Guard against zero cumulative volume (extremely thin first bars)
    df = df.with_columns(
        (pl.col("ctpv") / pl.when(pl.col("cv") > 0).then(pl.col("cv")).otherwise(1)).alias("vwap")
    )
    # Rolling band std proxy: stddev of (close - vwap) over dev_window
    dev_min = max(1, min(dev_window, max(10, dev_window // 10)))
    df = df.with_columns(
        ((pl.col("close") - pl.col("vwap")) ** 2)
        .rolling_mean(window_size=dev_window, min_samples=dev_min)
        .sqrt()
        .alias("dev")
    )
    band_exprs: list[pl.Expr] = []
    for s in band_sigmas:
        tag = _sigma_tag(s)
        band_exprs.append((pl.col("vwap") + s * pl.col("dev")).alias(f"upper_{tag}"))
        band_exprs.append((pl.col("vwap") - s * pl.col("dev")).alias(f"lower_{tag}"))
    df = df.with_columns(band_exprs)
    return df


def anchored_vwap(
    bars: pl.DataFrame,
    anchor_ts: datetime | pl.Expr,
    out_col: str = "avwap",
) -> pl.DataFrame:
    """Compute an anchored VWAP starting from `anchor_ts`.

    Returns the input bars sorted by `ts` with a new column `out_col`
    containing the anchored VWAP from `anchor_ts` forward. Rows strictly
    before `anchor_ts` have null in that column.

    `anchor_ts` may be a Python datetime or a Polars expression that
    evaluates to a single scalar timestamp.
    """
    if bars.is_empty():
        return bars

    df = bars.sort("ts").with_columns(
        [
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("_avwap_typ"),
        ]
    )
    # Mask rows >= anchor_ts; cumulative on the masked typical*vol / vol
    in_anchor = pl.col("ts") >= anchor_ts
    df = df.with_columns(
        [
            pl.when(in_anchor).then(pl.col("volume")).otherwise(0).alias("_avwap_v"),
            pl.when(in_anchor)
            .then(pl.col("_avwap_typ") * pl.col("volume"))
            .otherwise(0.0)
            .alias("_avwap_vp"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("_avwap_v").cum_sum().alias("_avwap_cv"),
            pl.col("_avwap_vp").cum_sum().alias("_avwap_cvp"),
        ]
    )
    df = df.with_columns(
        pl.when(pl.col("_avwap_cv") > 0)
        .then(pl.col("_avwap_cvp") / pl.col("_avwap_cv"))
        .otherwise(None)
        .alias(out_col)
    )
    return df.drop(["_avwap_typ", "_avwap_v", "_avwap_vp", "_avwap_cv", "_avwap_cvp"])


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------


def generate_signals(bars: pl.DataFrame, params: VWAPParams) -> list[Signal]:
    """Emit LONG and SHORT VWAP-pullback signals for one session day.

    Pipeline:
      1. Annotate bars with session VWAP, dev bands, ATR, slope.
      2. Mark recent-extension flags (close went beyond `extension_sigma` in
         the last `extension_lookback_s` bars, per side).
      3. Walk bars in chronological order, emit a signal when the current
         bar is a rejection candle inside the [entry_sigma .. extension_sigma]
         band AND a recent extension occurred AND filters pass.
      4. Apply per-side cooldown and per-side daily cap.
    """
    if bars.is_empty() or bars.height < max(params.dev_window, params.atr_window) // 2:
        return []
    if not (params.enable_short or params.enable_long):
        return []

    df = session_vwap(bars, params.dev_window, params.band_sigmas)

    # ATR proxy: rolling mean of (high - low) over atr_window bars
    atr_min = max(1, min(params.atr_window, max(5, params.atr_window // 4)))
    df = df.with_columns(
        (pl.col("high") - pl.col("low"))
        .rolling_mean(window_size=params.atr_window, min_samples=atr_min)
        .alias("atr")
    )

    # VWAP slope: vwap - vwap.shift(slope_window), in price units per window
    df = df.with_columns(
        (pl.col("vwap") - pl.col("vwap").shift(params.slope_window)).alias("vwap_slope")
    )

    # Swing hi / lo over swing_lookback bars (inclusive of current bar)
    df = df.with_columns(
        [
            pl.col("high")
            .rolling_max(window_size=params.swing_lookback, min_samples=1)
            .alias("swing_hi"),
            pl.col("low")
            .rolling_min(window_size=params.swing_lookback, min_samples=1)
            .alias("swing_lo"),
        ]
    )

    # Recent-extension flags: did any bar in the last extension_lookback_s
    # close beyond +/- extension_sigma * dev?
    ext_up = (pl.col("close") - pl.col("vwap")) >= (params.extension_sigma * pl.col("dev"))
    ext_dn = (pl.col("vwap") - pl.col("close")) >= (params.extension_sigma * pl.col("dev"))
    df = df.with_columns(
        [
            ext_up.cast(pl.Int8).alias("_ext_up"),
            ext_dn.cast(pl.Int8).alias("_ext_dn"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("_ext_up")
            .rolling_max(window_size=params.extension_lookback_s, min_samples=1)
            .alias("recent_ext_up"),
            pl.col("_ext_dn")
            .rolling_max(window_size=params.extension_lookback_s, min_samples=1)
            .alias("recent_ext_dn"),
        ]
    )

    # Session open ts (first bar) — used for min_minutes_since_open
    open_ts = df.select(pl.col("ts").min()).item()

    sigs: list[Signal] = []
    last_short_ts: datetime | None = None
    last_long_ts: datetime | None = None
    n_short = 0
    n_long = 0

    cols = (
        "ts",
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "dev",
        "atr",
        "vwap_slope",
        "swing_hi",
        "swing_lo",
        "recent_ext_up",
        "recent_ext_dn",
    )

    for row in df.select(list(cols)).iter_rows(named=True):
        ts = row["ts"]
        if (ts - open_ts).total_seconds() / 60.0 < params.min_minutes_since_open:
            continue

        dev = row["dev"]
        atr = row["atr"]
        if dev is None or atr is None:
            continue
        if dev < params.min_dev or atr < params.min_atr:
            continue

        vwap = row["vwap"]
        close = row["close"]
        open_px = row["open"]
        high = row["high"]
        low = row["low"]
        slope = row["vwap_slope"] if row["vwap_slope"] is not None else 0.0
        rng = max(high - low, 1e-9)

        # Distance from VWAP in sigma units
        up_sigma = (close - vwap) / dev
        dn_sigma = (vwap - close) / dev

        # ---- SHORT pullback ----
        if (
            params.enable_short
            and n_short < params.max_signals_per_side
            and row["recent_ext_up"] == 1
            and up_sigma >= params.entry_sigma
            and slope <= params.slope_abs_max
            and (
                last_short_ts is None
                or (ts - last_short_ts).total_seconds() >= params.cooldown_s
            )
        ):
            upper_wick = high - max(open_px, close)
            is_rejection = (close <= open_px) and (upper_wick / rng >= params.min_rejection_wick_ratio)
            if (not params.require_rejection_candle) or is_rejection:
                entry = float(close)
                if params.use_swing_stop:
                    stop = float(row["swing_hi"]) + atr * 0.10
                else:
                    stop = entry + atr * params.stop_atr_mult
                target = float(vwap)
                risk = stop - entry
                reward = entry - target
                if risk > 0 and reward / risk >= params.min_rr:
                    confidence = _clip01(
                        0.4
                        + 0.15 * min(up_sigma / max(params.extension_sigma, 1e-9), 2.0)
                        + 0.10 * min(reward / max(risk, 1e-9) - params.min_rr, 1.0)
                    )
                    sigs.append(
                        Signal(
                            timestamp=ts,
                            side=Side.SHORT,
                            confidence=confidence,
                            specialist=params.specialist_id,
                            setup_id="vwap_pullback_short",
                            entry_price=entry,
                            stop_price=stop,
                            target_price=target,
                            metadata={
                                "side": "short",
                                "vwap": float(vwap),
                                "dev": float(dev),
                                "atr": float(atr),
                                "up_sigma": float(up_sigma),
                                "slope": float(slope),
                                "rr": float(reward / risk),
                                "wick_ratio": float(upper_wick / rng),
                            },
                        )
                    )
                    last_short_ts = ts
                    n_short += 1

        # ---- LONG pullback (mirror) ----
        if (
            params.enable_long
            and n_long < params.max_signals_per_side
            and row["recent_ext_dn"] == 1
            and dn_sigma >= params.entry_sigma
            and slope >= -params.slope_abs_max
            and (
                last_long_ts is None
                or (ts - last_long_ts).total_seconds() >= params.cooldown_s
            )
        ):
            lower_wick = min(open_px, close) - low
            is_rejection = (close >= open_px) and (lower_wick / rng >= params.min_rejection_wick_ratio)
            if (not params.require_rejection_candle) or is_rejection:
                entry = float(close)
                if params.use_swing_stop:
                    stop = float(row["swing_lo"]) - atr * 0.10
                else:
                    stop = entry - atr * params.stop_atr_mult
                target = float(vwap)
                risk = entry - stop
                reward = target - entry
                if risk > 0 and reward / risk >= params.min_rr:
                    confidence = _clip01(
                        0.4
                        + 0.15 * min(dn_sigma / max(params.extension_sigma, 1e-9), 2.0)
                        + 0.10 * min(reward / max(risk, 1e-9) - params.min_rr, 1.0)
                    )
                    sigs.append(
                        Signal(
                            timestamp=ts,
                            side=Side.LONG,
                            confidence=confidence,
                            specialist=params.specialist_id,
                            setup_id="vwap_pullback_long",
                            entry_price=entry,
                            stop_price=stop,
                            target_price=target,
                            metadata={
                                "side": "long",
                                "vwap": float(vwap),
                                "dev": float(dev),
                                "atr": float(atr),
                                "dn_sigma": float(dn_sigma),
                                "slope": float(slope),
                                "rr": float(reward / risk),
                                "wick_ratio": float(lower_wick / rng),
                            },
                        )
                    )
                    last_long_ts = ts
                    n_long += 1

    # Sort just in case — emission is already chronological, but defensive
    sigs.sort(key=lambda s: s.timestamp)
    return sigs


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _sigma_tag(s: float) -> str:
    """Sigma -> column-name suffix, e.g. 1.0 -> 's1', 2.5 -> 's2_5'."""
    if float(s).is_integer():
        return f"s{int(s)}"
    return "s" + str(s).replace(".", "_")


__all__ = ["VWAPParams", "anchored_vwap", "generate_signals", "session_vwap"]
