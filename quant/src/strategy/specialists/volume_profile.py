"""Volume Profile specialist — Agent #6.

Builds intraday volume profile structure features and emits signals around them:

- Point of Control (POC): single highest-volume price bin in the session-to-date
  profile.
- Value Area High / Low (VAH / VAL): expanded around POC until covering
  `va_volume_pct` (default 70%) of session volume.
- High-Volume Nodes (HVNs) and Low-Volume Nodes (LVNs) — local extrema of the
  profile vs a smoothed neighborhood.
- Naked POC: a prior-session POC whose level has not been traded back through
  intraday (a magnet for current-day price).

Signal hypotheses (from `briefings/agent_06_volume_profile.md`):

1. Naked POC magnet — price approaches an unfilled prior-day POC; LONG/SHORT
   toward it depending on side.
2. LVN slip-through — price punches through an LVN with momentum; trend
   continuation in that direction.
3. VAH / VAL bounce — first test of the developing value area boundary in RTH
   gets bought at VAL / sold at VAH.

Contract (per `src/strategy/specialists/_interface.py`):

    def generate_signals(bars: pl.DataFrame, params: VPParams) -> list[Signal]

The function MUST be deterministic for a given (bars, params) — it relies on
session-internal information only when emitting current-day signals. For naked
POCs, the module keeps a small process-local cache of prior session profiles
keyed by date, so backtest harnesses that iterate sessions chronologically
build up state naturally without leaking future information.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl

from src.common.types import Side, Signal
from src.strategy.specialists._interface import SpecialistParams

# ES tick economics: 1 tick = 0.25 points
ES_TICK_SIZE = 0.25

# Minimum bar count to compute a meaningful intraday profile (~10 min @ 1s).
_MIN_BARS_FOR_PROFILE = 60


@dataclass
class VPParams(SpecialistParams):
    """Tunable parameters for the volume profile specialist.

    Defaults were chosen to be conservative on 2020-2023 ES synthetic stress
    tests; the orchestrator's full 5y audit will retune if needed.
    """

    specialist_id: str = "vp"

    # --- profile construction
    bin_ticks: int = 4                  # 1 tick = 0.25 → 4 ticks = 1.0 point bins
    va_volume_pct: float = 0.70
    naked_poc_lookback_days: int = 5

    # --- HVN / LVN detection
    hvn_zscore: float = 1.0             # bins ≥ mean + z*std volume → HVN
    lvn_zscore: float = -0.8            # bins ≤ mean + z*std volume → LVN
    node_neighbor_bins: int = 3         # local-extremum half-window

    # --- signal gating
    min_minutes_since_open: int = 30    # don't fire in opening drive
    max_signals_per_day: int = 6
    min_seconds_between_signals: int = 180

    # --- entry/stop sizing
    naked_poc_proximity_pct: float = 0.0015   # within 0.15% to flag magnet
    stop_atr_mult: float = 1.0
    target_atr_mult: float = 1.5
    lvn_breakout_ticks: int = 4         # bars must close beyond LVN edge by this
    va_bounce_max_excess_ticks: int = 8  # rejection from VAH/VAL within N ticks

    # --- confidence shaping
    min_confidence: float = 0.30
    max_confidence: float = 0.90


# --------------------------------------------------------------------------- #
# Module-local cache of prior-session profiles for naked-POC lookups.
# Keyed by session date isoformat string -> dict produced by build_profile.
# The cache is bounded by `naked_poc_lookback_days` * 2 entries to avoid
# unbounded growth across long backtests.
# --------------------------------------------------------------------------- #
_PROFILE_CACHE: dict[str, dict] = {}


def _atr_points(df: pl.DataFrame, window: int = 60) -> float:
    """Approximate intraday ATR in price points from a tail of `window` bars."""
    if df.height == 0:
        return 1.0
    tail = df.tail(window)
    rng = (tail["high"] - tail["low"]).mean()
    return float(rng or 0.5) * max(window, 1) / max(window, 1)


def _bin_size_points(params: VPParams) -> float:
    return params.bin_ticks * ES_TICK_SIZE


def _build_histogram(bars: pl.DataFrame, params: VPParams) -> pl.DataFrame:
    """Bin volume by the bar's typical price.

    The 1s OHLCV bars from Databento are tight enough (most bars have
    high == low within 1-2 ticks for liquid ES RTH) that allocating the
    bar's volume to a single bin at the typical price is a faithful enough
    approximation of the underlying TPO/volume profile. A full-fledged
    tick-volume distribution requires TBBO, which Agent #4 owns.
    """
    if bars.is_empty():
        return pl.DataFrame({"bin": [], "volume": []})
    binp = _bin_size_points(params)
    df = bars.with_columns(
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("typical")
    )
    df = df.with_columns(
        ((pl.col("typical") / binp).floor() * binp).alias("bin"),
    )
    hist = (
        df.group_by("bin")
        .agg(pl.col("volume").sum().alias("volume"))
        .sort("bin")
    )
    return hist


def _poc_and_value_area(hist: pl.DataFrame, params: VPParams) -> tuple[float, float, float]:
    """Return (poc, val, vah) from a (bin, volume) histogram.

    POC is the bin with maximum volume. The value area grows symmetrically
    outward from POC, taking whichever adjacent bin (above or below) has
    more volume, until cumulative volume covers `va_volume_pct`.
    """
    if hist.is_empty():
        return float("nan"), float("nan"), float("nan")
    rows = hist.sort("bin").to_dicts()
    bins = [r["bin"] for r in rows]
    vols = [r["volume"] for r in rows]
    total = sum(vols)
    if total <= 0:
        return float("nan"), float("nan"), float("nan")
    poc_idx = max(range(len(vols)), key=lambda i: vols[i])
    poc = bins[poc_idx]
    target_vol = total * params.va_volume_pct
    covered = vols[poc_idx]
    lo, hi = poc_idx, poc_idx
    while covered < target_vol and (lo > 0 or hi < len(vols) - 1):
        up_vol = vols[hi + 1] if hi < len(vols) - 1 else -1
        dn_vol = vols[lo - 1] if lo > 0 else -1
        if up_vol < 0 and dn_vol < 0:
            break
        if up_vol >= dn_vol:
            hi += 1
            covered += vols[hi]
        else:
            lo -= 1
            covered += vols[lo]
    return float(poc), float(bins[lo]), float(bins[hi])


def _hvn_lvn(hist: pl.DataFrame, params: VPParams) -> tuple[list[float], list[float]]:
    """Identify HVN/LVN bins.

    A bin qualifies as an HVN if it is a local maximum in a neighborhood of
    `node_neighbor_bins` AND its z-scored volume vs all bins exceeds
    `hvn_zscore`. LVN is symmetric on the low side.
    """
    if hist.is_empty():
        return [], []
    rows = hist.sort("bin").to_dicts()
    bins = [r["bin"] for r in rows]
    vols = [r["volume"] for r in rows]
    n = len(vols)
    if n < 3:
        return [], []
    mean_v = sum(vols) / n
    var = sum((v - mean_v) ** 2 for v in vols) / n
    std_v = (var ** 0.5) or 1.0
    half = params.node_neighbor_bins
    hvns: list[float] = []
    lvns: list[float] = []
    for i in range(n):
        v = vols[i]
        z = (v - mean_v) / std_v
        lo = max(0, i - half)
        hi = min(n - 1, i + half)
        local_max = v >= max(vols[lo : hi + 1])
        local_min = v <= min(vols[lo : hi + 1])
        if z >= params.hvn_zscore and local_max:
            hvns.append(bins[i])
        if z <= params.lvn_zscore and local_min and v > 0:
            # an empty bin in the middle of the range is also "low", but we want
            # bins with at least *some* trade — pure zeros are no-mans-land.
            lvns.append(bins[i])
    return hvns, lvns


def _shape_label(hist: pl.DataFrame) -> str:
    """Crude profile shape classifier (D / b / p / double).

    Used only as diagnostic `metadata`; signals do not depend on it.
    """
    if hist.is_empty():
        return "empty"
    rows = hist.sort("bin").to_dicts()
    n = len(rows)
    if n < 5:
        return "thin"
    vols = [r["volume"] for r in rows]
    third = n // 3
    top = sum(vols[: third])
    mid = sum(vols[third : 2 * third])
    bot = sum(vols[2 * third :])
    total = sum(vols) or 1
    top_f, mid_f, bot_f = top / total, mid / total, bot / total
    # Look for a clear double-distribution: two peaks separated by a valley
    peaks = sum(
        1
        for i in range(1, n - 1)
        if vols[i] > vols[i - 1] and vols[i] > vols[i + 1] and vols[i] > 0.7 * max(vols)
    )
    if peaks >= 2:
        return "double"
    if mid_f >= 0.45:
        return "D"
    if bot_f >= 0.40 and bot_f > top_f:
        return "b"
    if top_f >= 0.40 and top_f > bot_f:
        return "p"
    return "D"


def build_profile(bars: pl.DataFrame, params: VPParams) -> dict:
    """Return the full profile dict for one session day's bars.

    Output keys:
        poc        : float — point of control price level
        val        : float — value area low
        vah        : float — value area high
        hvns       : list[float] — HVN bin prices
        lvns       : list[float] — LVN bin prices
        naked_pocs : list[float] — prior-day POCs unfilled by *this* session
        shape      : str — "D"/"b"/"p"/"double"/"thin"/"empty"
        total_vol  : int — sum of volume in the session
        bin_size   : float — bin size in price points
        n_bars     : int — number of bars used
    """
    hist = _build_histogram(bars, params)
    poc, val, vah = _poc_and_value_area(hist, params)
    hvns, lvns = _hvn_lvn(hist, params)
    shape = _shape_label(hist)
    naked_pocs = _resolve_naked_pocs(bars, params)
    total_vol = int(bars["volume"].sum()) if not bars.is_empty() else 0
    return {
        "poc": poc,
        "val": val,
        "vah": vah,
        "hvns": hvns,
        "lvns": lvns,
        "naked_pocs": naked_pocs,
        "shape": shape,
        "total_vol": total_vol,
        "bin_size": _bin_size_points(params),
        "n_bars": int(bars.height),
    }


def _session_date_of(bars: pl.DataFrame) -> str | None:
    if bars.is_empty() or "session_date" not in bars.columns:
        return None
    sd = bars["session_date"][0]
    if hasattr(sd, "isoformat"):
        return sd.isoformat()
    return str(sd)


def _resolve_naked_pocs(bars: pl.DataFrame, params: VPParams) -> list[float]:
    """Pull naked POCs from prior cached sessions.

    A prior-day POC counts as "naked" if the current session has not traded
    through that price level (i.e. min(low) > poc or max(high) < poc as of the
    last bar). We look back up to `naked_poc_lookback_days` cached sessions.
    """
    if bars.is_empty():
        return []
    sd_str = _session_date_of(bars)
    if sd_str is None:
        return []
    candidates: list[tuple[str, float]] = []
    for k, prof in _PROFILE_CACHE.items():
        if k >= sd_str:
            continue
        poc = prof.get("poc")
        if poc is None or poc != poc:  # NaN check
            continue
        candidates.append((k, float(poc)))
    candidates.sort(reverse=True)
    candidates = candidates[: params.naked_poc_lookback_days]

    cur_low = float(bars["low"].min())
    cur_high = float(bars["high"].max())
    naked: list[float] = []
    for _k, poc in candidates:
        if poc < cur_low or poc > cur_high:
            naked.append(poc)
    return naked


def _register_profile(bars: pl.DataFrame, profile: dict) -> None:
    sd = _session_date_of(bars)
    if sd is None:
        return
    _PROFILE_CACHE[sd] = profile
    if len(_PROFILE_CACHE) > 64:
        # Keep cache bounded; drop oldest entries.
        for stale in sorted(_PROFILE_CACHE.keys())[: len(_PROFILE_CACHE) - 32]:
            _PROFILE_CACHE.pop(stale, None)


def reset_cache() -> None:
    """Clear the module-local profile cache. Tests use this to isolate runs."""
    _PROFILE_CACHE.clear()


# --------------------------------------------------------------------------- #
# Signal generation
# --------------------------------------------------------------------------- #


def _clamp_confidence(c: float, params: VPParams) -> float:
    return float(max(params.min_confidence, min(params.max_confidence, c)))


def _approx_developing_profile(
    bars_so_far: pl.DataFrame, params: VPParams
) -> tuple[float, float, float] | None:
    """Compute developing (POC, VAL, VAH) from session-to-date bars."""
    if bars_so_far.height < _MIN_BARS_FOR_PROFILE:
        return None
    hist = _build_histogram(bars_so_far, params)
    poc, val, vah = _poc_and_value_area(hist, params)
    if poc != poc:  # NaN
        return None
    return poc, val, vah


def generate_signals(bars: pl.DataFrame, params: VPParams) -> list[Signal]:
    """Emit Signals from VA/POC/HVN/LVN setups + naked POC magnet.

    Specialist runs on ONE session's bars. We walk the day in 1-minute
    sampling steps (every 60 bars) so we don't fire every second — VP setups
    are slow features. We require `min_minutes_since_open` before the first
    fire to skip the opening drive.
    """
    if bars.is_empty():
        return []
    df = bars.sort("ts")
    sigs: list[Signal] = []

    # Cache prior naked POCs BEFORE registering today's profile.
    naked_pocs = _resolve_naked_pocs(df, params)

    open_ts: datetime | None = None
    last_ts: datetime | None = None
    n_sig = 0

    # Track a per-side "tested" flag for VAH/VAL bounce so we only signal first
    # touch in each direction per day.
    vah_tested = False
    val_tested = False
    # Track LVN levels that have already produced a breakout signal today.
    lvn_signaled: set[float] = set()

    rows = df.iter_rows(named=True)
    bars_so_far_list: list[dict] = []
    for i, row in enumerate(rows):
        bars_so_far_list.append(row)
        if open_ts is None:
            open_ts = row["ts"]
        mins_since_open = (row["ts"] - open_ts).total_seconds() / 60.0
        if mins_since_open < params.min_minutes_since_open:
            continue
        # Sample every 60s
        if i % 60 != 0:
            continue
        if last_ts is not None and (
            row["ts"] - last_ts
        ).total_seconds() < params.min_seconds_between_signals:
            continue
        if n_sig >= params.max_signals_per_day:
            break

        # Build a developing profile from session-to-date bars.
        bars_so_far = df.head(i + 1)
        dev = _approx_developing_profile(bars_so_far, params)
        if dev is None:
            continue
        poc, val, vah = dev

        atr = _atr_points(bars_so_far, window=60)
        if atr <= 0:
            atr = 0.5
        bin_size = _bin_size_points(params)
        proximity = max(atr * 0.25, bin_size * 1.5)
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        ts = row["ts"]

        new_sig: Signal | None = None

        # --- 1) Naked POC magnet ----------------------------------------------
        for npoc in naked_pocs:
            band = max(npoc * params.naked_poc_proximity_pct, proximity)
            if abs(close - npoc) <= band and abs(close - npoc) > bin_size * 0.5:
                side = Side.LONG if close < npoc else Side.SHORT
                entry = close
                target = npoc
                if side == Side.LONG:
                    stop = entry - atr * params.stop_atr_mult
                else:
                    stop = entry + atr * params.stop_atr_mult
                conf = _clamp_confidence(
                    0.55 + 0.25 * (1.0 - abs(close - npoc) / max(band, bin_size)),
                    params,
                )
                new_sig = Signal(
                    timestamp=ts,
                    side=side,
                    confidence=conf,
                    specialist="vp",
                    setup_id="naked_poc_magnet",
                    entry_price=entry,
                    stop_price=stop,
                    target_price=target,
                    metadata={
                        "naked_poc": float(npoc),
                        "atr": atr,
                        "distance_pts": abs(close - npoc),
                    },
                )
                break

        # --- 2) VAH / VAL bounce (first test in each direction) ---------------
        if new_sig is None and val == val and vah == vah:
            bounce_band = max(
                bin_size * params.va_bounce_max_excess_ticks / params.bin_ticks,
                atr * 0.5,
            )
            # VAL bounce → LONG if price tags VAL from above and rejects
            if (
                not val_tested
                and low <= val + bin_size
                and close > val
                and (close - val) <= bounce_band
            ):
                entry = close
                stop = val - atr * params.stop_atr_mult
                target = poc
                conf = _clamp_confidence(0.55 + 0.10 * ((close - val) / bin_size), params)
                if target > entry:
                    new_sig = Signal(
                        timestamp=ts,
                        side=Side.LONG,
                        confidence=conf,
                        specialist="vp",
                        setup_id="val_bounce_long",
                        entry_price=entry,
                        stop_price=stop,
                        target_price=target,
                        metadata={
                            "val": val,
                            "vah": vah,
                            "poc": poc,
                            "atr": atr,
                        },
                    )
                    val_tested = True
            # VAH bounce → SHORT if price tags VAH from below and rejects
            elif (
                new_sig is None
                and not vah_tested
                and high >= vah - bin_size
                and close < vah
                and (vah - close) <= bounce_band
            ):
                entry = close
                stop = vah + atr * params.stop_atr_mult
                target = poc
                conf = _clamp_confidence(0.55 + 0.10 * ((vah - close) / bin_size), params)
                if target < entry:
                    new_sig = Signal(
                        timestamp=ts,
                        side=Side.SHORT,
                        confidence=conf,
                        specialist="vp",
                        setup_id="vah_bounce_short",
                        entry_price=entry,
                        stop_price=stop,
                        target_price=target,
                        metadata={
                            "val": val,
                            "vah": vah,
                            "poc": poc,
                            "atr": atr,
                        },
                    )
                    vah_tested = True

        # --- 3) LVN slip-through (momentum continuation) ----------------------
        if new_sig is None:
            hist_so_far = _build_histogram(bars_so_far, params)
            _, lvns = _hvn_lvn(hist_so_far, params)
            for lvn in lvns:
                if lvn in lvn_signaled:
                    continue
                edge = params.lvn_breakout_ticks * ES_TICK_SIZE
                if close > lvn + edge and low > lvn:
                    entry = close
                    stop = lvn - atr * params.stop_atr_mult * 0.5
                    target = entry + atr * params.target_atr_mult
                    conf = _clamp_confidence(
                        0.45 + 0.10 * ((close - lvn) / max(edge, bin_size)),
                        params,
                    )
                    new_sig = Signal(
                        timestamp=ts,
                        side=Side.LONG,
                        confidence=conf,
                        specialist="vp",
                        setup_id="lvn_slip_long",
                        entry_price=entry,
                        stop_price=stop,
                        target_price=target,
                        metadata={"lvn": float(lvn), "atr": atr},
                    )
                    lvn_signaled.add(lvn)
                    break
                if close < lvn - edge and high < lvn:
                    entry = close
                    stop = lvn + atr * params.stop_atr_mult * 0.5
                    target = entry - atr * params.target_atr_mult
                    conf = _clamp_confidence(
                        0.45 + 0.10 * ((lvn - close) / max(edge, bin_size)),
                        params,
                    )
                    new_sig = Signal(
                        timestamp=ts,
                        side=Side.SHORT,
                        confidence=conf,
                        specialist="vp",
                        setup_id="lvn_slip_short",
                        entry_price=entry,
                        stop_price=stop,
                        target_price=target,
                        metadata={"lvn": float(lvn), "atr": atr},
                    )
                    lvn_signaled.add(lvn)
                    break

        if new_sig is not None:
            # Final sanity: stop must be on the opposite side of entry.
            if new_sig.side == Side.LONG and new_sig.stop_price >= new_sig.entry_price:
                continue
            if new_sig.side == Side.SHORT and new_sig.stop_price <= new_sig.entry_price:
                continue
            sigs.append(new_sig)
            last_ts = ts
            n_sig += 1

    # Register today's full-session profile in cache for downstream naked-POC
    # lookups (subsequent sessions will see this as prior data).
    full_profile = build_profile(df, params)
    _register_profile(df, full_profile)

    return sigs


__all__ = [
    "VPParams",
    "build_profile",
    "generate_signals",
    "reset_cache",
]
