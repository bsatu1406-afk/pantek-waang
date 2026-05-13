"""Microstructure specialist — TBBO L1-derived signals (Agent #5).

Emits trading signals from top-of-book microstructure features:

    * book_imbalance  — sustained bid/ask size imbalance (momentum, trade WITH)
    * microprice_dev  — microprice deviation from mid > N ticks (momentum, WITH)
    * sweep           — multi-trade aggressive walk in <500 ms (fade)
    * trapped         — big-print level revisit after adverse move (fade)
    * liq_hole        — thin top-of-book + directional aggressor flow (momentum)

The detectors compose into ONE specialist; signals are emitted with a setup_id
that identifies which detector fired so downstream ranking / risk can route
them differently.

No lookahead: every feature for a signal at second T uses TBBO events
with ts <= T. The standalone harness enters at the first bar strictly
AFTER signal.timestamp, i.e. at T+1s, so a same-second feature is safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl

from src.common.types import Side, Signal
from src.data.loader import load_tbbo
from src.strategy.specialists._interface import SpecialistParams

# ES/MES tick economics
TICK_SIZE = 0.25
TICKS_PER_POINT = 4


@dataclass
class MicrostructureParams(SpecialistParams):
    """Knobs for the microstructure specialist. Defaults tuned on 2020-2023."""

    specialist_id: str = "micro"

    # Book imbalance ----------------------------------------------------
    imbalance_threshold: float = 3.0     # max(bid_sz, ask_sz) / min(...)
    imbalance_window_s: int = 5          # require sustained over rolling window
    imbalance_min_top_sz: int = 40       # ignore tiny top-of-book

    # Microprice deviation ----------------------------------------------
    microprice_dev_ticks: float = 1.5    # |microprice - mid| in ticks

    # Sweep -------------------------------------------------------------
    sweep_window_ms: int = 500
    sweep_min_levels: int = 3            # price walk in ticks
    sweep_min_trades: int = 3
    sweep_dom_side_frac: float = 0.7     # >=70% same-side aggressor

    # Trapped traders ---------------------------------------------------
    trap_min_size: int = 50              # large print threshold (contracts)
    trap_adverse_ticks: float = 4.0      # must move N ticks against print first
    trap_zone_ticks: float = 0.5         # revisit tolerance
    trap_max_lookback_min: int = 30      # only consider prints within last N min

    # Liquidity hole ----------------------------------------------------
    hole_size_pct: float = 0.25          # top_sz < this * rolling median
    hole_flow_lookback_s: int = 10
    hole_flow_min_abs: int = 50

    # Risk / exit -------------------------------------------------------
    stop_ticks: float = 4.0              # 1.00 point ES = $5/MES
    target_ticks: float = 6.0            # 1.5R

    # Throttling --------------------------------------------------------
    min_seconds_between_signals: int = 360   # 6 min cooldown per setup
    max_signals_per_day: int = 16
    min_minutes_since_open: int = 5
    min_minutes_to_close: int = 5            # no new entries near close


# --------------------------------------------------------------------------
# Public entrypoint
# --------------------------------------------------------------------------


def generate_signals(
    bars: pl.DataFrame, params: MicrostructureParams,
) -> list[Signal]:
    """Generate microstructure signals for one RTH session day.

    The bars dataframe is 1-second OHLCV. We fetch TBBO for the same session
    date through `load_tbbo` (Polars-cached). If TBBO is unavailable for the
    day we return an empty list — this specialist requires L1 microstructure.
    """
    if bars.is_empty():
        return []
    session_date = bars["session_date"][0]
    tbbo = load_tbbo(session_date)
    if tbbo is None or tbbo.is_empty():
        return []
    return _generate_signals_from_data(bars, tbbo, params)


def _generate_signals_from_data(
    bars: pl.DataFrame,
    tbbo: pl.DataFrame,
    params: MicrostructureParams,
) -> list[Signal]:
    """Pure compute path; takes both bars and tbbo so it is unit-testable
    without filesystem state."""
    bars = bars.sort("ts")
    tbbo = tbbo.sort("ts")
    micro = _resample_tbbo_to_1s(tbbo)
    if micro.is_empty():
        return []

    session_open = bars["ts"][0]
    session_close = bars["ts"][-1]
    open_cutoff = session_open + timedelta(minutes=params.min_minutes_since_open)
    close_cutoff = session_close - timedelta(minutes=params.min_minutes_to_close)

    candidates: list[_Candidate] = []
    candidates.extend(_detect_imbalance(micro, params))
    candidates.extend(_detect_microprice(micro, params))
    candidates.extend(_detect_sweeps(tbbo, micro, params))
    candidates.extend(_detect_traps(tbbo, bars, params))
    candidates.extend(_detect_holes(micro, params))

    # Time window: skip first / last minutes of session
    candidates = [c for c in candidates if open_cutoff <= c.ts <= close_cutoff]
    # Stable chronological order
    candidates.sort(key=lambda c: (c.ts, c.setup_id))

    return _throttle_and_finalize(candidates, params)


# --------------------------------------------------------------------------
# 1-second microstructure frame
# --------------------------------------------------------------------------


def _resample_tbbo_to_1s(tbbo: pl.DataFrame) -> pl.DataFrame:
    """Collapse TBBO events to one row per second with last quote + trade aggs."""
    df = tbbo.with_columns(pl.col("ts").dt.truncate("1s").alias("ts_1s"))

    is_quote = pl.col("bid_px").is_not_null() & pl.col("ask_px").is_not_null()
    quotes = df.filter(is_quote)
    last_quote = (
        quotes.group_by("ts_1s")
        .agg(
            pl.col("bid_px").last(),
            pl.col("ask_px").last(),
            pl.col("bid_sz").last(),
            pl.col("ask_sz").last(),
        )
        .sort("ts_1s")
    )

    if "action" in df.columns:
        is_trade = pl.col("action") == "T"
    else:
        is_trade = pl.col("price").is_not_null()
    trades = df.filter(is_trade & pl.col("side").is_in(["B", "S"]))
    buy_vol = pl.when(pl.col("side") == "B").then(pl.col("size")).otherwise(0)
    sell_vol = pl.when(pl.col("side") == "S").then(pl.col("size")).otherwise(0)
    trade_agg = (
        trades.with_columns(buy_vol.alias("buy_vol"), sell_vol.alias("sell_vol"))
        .group_by("ts_1s")
        .agg(
            pl.col("buy_vol").sum(),
            pl.col("sell_vol").sum(),
            pl.len().alias("n_trades"),
            pl.col("size").max().alias("max_trade_size"),
        )
        .sort("ts_1s")
    )
    out = last_quote.join(trade_agg, on="ts_1s", how="left").sort("ts_1s")
    out = out.with_columns(
        pl.col("buy_vol").fill_null(0),
        pl.col("sell_vol").fill_null(0),
        pl.col("n_trades").fill_null(0),
        pl.col("max_trade_size").fill_null(0),
    )
    # Forward-fill quotes (gaps where no quote update fired)
    out = out.with_columns(
        pl.col("bid_px").forward_fill(),
        pl.col("ask_px").forward_fill(),
        pl.col("bid_sz").forward_fill(),
        pl.col("ask_sz").forward_fill(),
    ).drop_nulls(subset=["bid_px", "ask_px", "bid_sz", "ask_sz"])
    if out.is_empty():
        return out
    out = out.with_columns(
        ((pl.col("bid_px") + pl.col("ask_px")) / 2).alias("mid"),
        (
            (pl.col("bid_px") * pl.col("ask_sz") + pl.col("ask_px") * pl.col("bid_sz"))
            / (pl.col("bid_sz") + pl.col("ask_sz"))
        ).alias("microprice"),
        (pl.col("buy_vol") - pl.col("sell_vol")).alias("delta"),
        (pl.col("bid_sz") + pl.col("ask_sz")).alias("top_sz"),
    ).rename({"ts_1s": "ts"})
    return out


# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------


@dataclass
class _Candidate:
    ts: datetime
    side: Side
    setup_id: str
    entry_price: float
    stop_price: float
    target_price: float
    confidence: float
    metadata: dict


def _detect_imbalance(micro: pl.DataFrame, p: MicrostructureParams) -> list[_Candidate]:
    """Bid/ask top-of-book size imbalance sustained for `imbalance_window_s` seconds.

    Trade WITH the imbalance: bids stacked => buying pressure => LONG.
    """
    if micro.height < p.imbalance_window_s:
        return []
    df = micro.with_columns(
        (pl.col("bid_sz") / pl.col("ask_sz")).alias("b_over_a"),
        (pl.col("ask_sz") / pl.col("bid_sz")).alias("a_over_b"),
    )
    # rolling min over window => sustained imbalance
    df = df.with_columns(
        pl.col("b_over_a")
        .rolling_min(window_size=p.imbalance_window_s)
        .alias("b_over_a_min"),
        pl.col("a_over_b")
        .rolling_min(window_size=p.imbalance_window_s)
        .alias("a_over_b_min"),
    )
    long_mask = (
        (pl.col("b_over_a_min") >= p.imbalance_threshold)
        & (pl.col("bid_sz") >= p.imbalance_min_top_sz)
    )
    short_mask = (
        (pl.col("a_over_b_min") >= p.imbalance_threshold)
        & (pl.col("ask_sz") >= p.imbalance_min_top_sz)
    )
    df = df.with_columns(
        pl.when(long_mask).then(pl.lit("long"))
        .when(short_mask).then(pl.lit("short"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("imb_side")
    ).filter(pl.col("imb_side").is_not_null())

    cands: list[_Candidate] = []
    stop_d = p.stop_ticks * TICK_SIZE
    tgt_d = p.target_ticks * TICK_SIZE
    for r in df.iter_rows(named=True):
        side = Side.LONG if r["imb_side"] == "long" else Side.SHORT
        ratio = r["b_over_a_min"] if side == Side.LONG else r["a_over_b_min"]
        # Use ask for long entry, bid for short entry (cross the spread, conservative).
        entry = float(r["ask_px"]) if side == Side.LONG else float(r["bid_px"])
        if side == Side.LONG:
            stop = entry - stop_d
            tgt = entry + tgt_d
        else:
            stop = entry + stop_d
            tgt = entry - tgt_d
        cands.append(_Candidate(
            ts=r["ts"], side=side, setup_id="book_imbalance",
            entry_price=entry, stop_price=stop, target_price=tgt,
            confidence=_squash((float(ratio) - p.imbalance_threshold) / p.imbalance_threshold, 0.4, 0.9),
            metadata={"ratio": float(ratio), "top_sz": int(r["top_sz"])},
        ))
    return cands


def _detect_microprice(micro: pl.DataFrame, p: MicrostructureParams) -> list[_Candidate]:
    """Microprice deviates from mid by more than `microprice_dev_ticks` ticks."""
    threshold = p.microprice_dev_ticks * TICK_SIZE
    df = micro.with_columns(
        (pl.col("microprice") - pl.col("mid")).alias("dev"),
    ).filter(pl.col("dev").abs() >= threshold)
    cands: list[_Candidate] = []
    stop_d = p.stop_ticks * TICK_SIZE
    tgt_d = p.target_ticks * TICK_SIZE
    for r in df.iter_rows(named=True):
        dev = float(r["dev"])
        side = Side.LONG if dev > 0 else Side.SHORT
        entry = float(r["ask_px"]) if side == Side.LONG else float(r["bid_px"])
        if side == Side.LONG:
            stop = entry - stop_d
            tgt = entry + tgt_d
        else:
            stop = entry + stop_d
            tgt = entry - tgt_d
        cands.append(_Candidate(
            ts=r["ts"], side=side, setup_id="microprice_dev",
            entry_price=entry, stop_price=stop, target_price=tgt,
            confidence=_squash(abs(dev) / (threshold * 2), 0.4, 0.85),
            metadata={"dev": dev, "dev_ticks": dev / TICK_SIZE},
        ))
    return cands


def _detect_sweeps(
    tbbo: pl.DataFrame, micro: pl.DataFrame, p: MicrostructureParams,
) -> list[_Candidate]:
    """Multi-trade aggressive walk within sweep_window_ms; FADE direction.

    Buy sweep (price walked up) => SHORT.
    Sell sweep (price walked down) => LONG.
    """
    if "action" in tbbo.columns:
        is_trade = pl.col("action") == "T"
    else:
        is_trade = pl.col("price").is_not_null()
    trades = tbbo.filter(
        is_trade & pl.col("side").is_in(["B", "S"]) & pl.col("price").is_not_null()
    ).sort("ts")
    if trades.height < p.sweep_min_trades:
        return []

    window = f"{p.sweep_window_ms}ms"
    grouped = (
        trades.group_by_dynamic("ts", every=window, period=window, closed="left")
        .agg(
            pl.col("price").min().alias("p_min"),
            pl.col("price").max().alias("p_max"),
            pl.len().alias("n_trades"),
            pl.col("side").filter(pl.col("side") == "B").len().alias("n_buy"),
            pl.col("side").filter(pl.col("side") == "S").len().alias("n_sell"),
            pl.col("size").sum().alias("vol"),
            pl.col("price").last().alias("last_price"),
        )
    )
    grouped = grouped.with_columns(
        (pl.col("p_max") - pl.col("p_min")).alias("p_range"),
    )
    buy_sweep = (
        (pl.col("p_range") >= p.sweep_min_levels * TICK_SIZE)
        & (pl.col("n_trades") >= p.sweep_min_trades)
        & (pl.col("n_buy") >= p.sweep_dom_side_frac * pl.col("n_trades"))
    )
    sell_sweep = (
        (pl.col("p_range") >= p.sweep_min_levels * TICK_SIZE)
        & (pl.col("n_trades") >= p.sweep_min_trades)
        & (pl.col("n_sell") >= p.sweep_dom_side_frac * pl.col("n_trades"))
    )
    grouped = grouped.with_columns(
        pl.when(buy_sweep).then(pl.lit("B"))
        .when(sell_sweep).then(pl.lit("S"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("sweep_side")
    ).filter(pl.col("sweep_side").is_not_null())
    if grouped.is_empty():
        return []
    grouped = grouped.with_columns(pl.col("ts").dt.truncate("1s").alias("ts_1s"))
    # Align with micro (last quote of that second) to derive entry/stop.
    aligned = grouped.join(
        micro.select(["ts", "bid_px", "ask_px", "mid"]).rename({"ts": "ts_1s"}),
        on="ts_1s",
        how="left",
    ).drop_nulls(subset=["bid_px", "ask_px"])

    cands: list[_Candidate] = []
    stop_d = p.stop_ticks * TICK_SIZE
    tgt_d = p.target_ticks * TICK_SIZE
    seen_ts: set[datetime] = set()
    for r in aligned.iter_rows(named=True):
        ts_1s = r["ts_1s"]
        if ts_1s in seen_ts:
            continue
        seen_ts.add(ts_1s)
        if r["sweep_side"] == "B":
            side = Side.SHORT
            entry = float(r["bid_px"])
            stop = entry + stop_d
            tgt = entry - tgt_d
        else:
            side = Side.LONG
            entry = float(r["ask_px"])
            stop = entry - stop_d
            tgt = entry + tgt_d
        cands.append(_Candidate(
            ts=ts_1s, side=side, setup_id="sweep",
            entry_price=entry, stop_price=stop, target_price=tgt,
            confidence=_squash(
                (float(r["p_range"]) / TICK_SIZE - p.sweep_min_levels) / 4.0,
                0.45, 0.85,
            ),
            metadata={
                "p_range_ticks": float(r["p_range"]) / TICK_SIZE,
                "n_trades": int(r["n_trades"]),
                "vol": int(r["vol"]),
            },
        ))
    return cands


def _detect_traps(
    tbbo: pl.DataFrame, bars: pl.DataFrame, p: MicrostructureParams,
) -> list[_Candidate]:
    """Big-print levels revisited after adverse move; fade the original print.

    Big BUY print at P, price drops >= trap_adverse_ticks ticks, then returns
    to P (within trap_zone_ticks) => SHORT on revisit.
    Symmetric for big SELL print => LONG on revisit.
    """
    if "action" in tbbo.columns:
        is_trade = pl.col("action") == "T"
    else:
        is_trade = pl.col("price").is_not_null()
    big = tbbo.filter(
        is_trade
        & pl.col("side").is_in(["B", "S"])
        & (pl.col("size") >= p.trap_min_size)
        & pl.col("price").is_not_null()
    ).select(["ts", "price", "size", "side"]).sort("ts")
    if big.is_empty() or bars.is_empty():
        return []

    bars_sorted = bars.sort("ts")
    rows = bars_sorted.select(["ts", "high", "low"]).to_dicts()
    big_rows = big.to_dicts()
    zone = p.trap_zone_ticks * TICK_SIZE
    adverse = p.trap_adverse_ticks * TICK_SIZE
    cooldown = timedelta(minutes=p.trap_max_lookback_min)
    stop_d = p.stop_ticks * TICK_SIZE
    tgt_d = p.target_ticks * TICK_SIZE

    cands: list[_Candidate] = []
    # State per active print: hit adverse yet? returned yet? Emit at first return.
    for bp in big_rows:
        bp_ts = bp["ts"]
        level = float(bp["price"])
        side_print = bp["side"]
        size_print = int(bp["size"])
        hit_adverse = False
        for br in rows:
            if br["ts"] < bp_ts:
                continue
            if br["ts"] - bp_ts > cooldown:
                break
            hi = float(br["high"])
            lo = float(br["low"])
            if side_print == "B":  # need price to drop below level - adverse
                if not hit_adverse:
                    if lo <= level - adverse:
                        hit_adverse = True
                    continue
                # already trapped; look for return to level
                if hi >= level - zone:
                    entry = level
                    stop = entry + stop_d
                    tgt = entry - tgt_d
                    cands.append(_Candidate(
                        ts=br["ts"], side=Side.SHORT, setup_id="trapped",
                        entry_price=entry, stop_price=stop, target_price=tgt,
                        confidence=_squash(size_print / (4 * p.trap_min_size), 0.4, 0.85),
                        metadata={"level": level, "print_side": side_print,
                                   "print_size": size_print},
                    ))
                    break
            else:  # 'S' print
                if not hit_adverse:
                    if hi >= level + adverse:
                        hit_adverse = True
                    continue
                if lo <= level + zone:
                    entry = level
                    stop = entry - stop_d
                    tgt = entry + tgt_d
                    cands.append(_Candidate(
                        ts=br["ts"], side=Side.LONG, setup_id="trapped",
                        entry_price=entry, stop_price=stop, target_price=tgt,
                        confidence=_squash(size_print / (4 * p.trap_min_size), 0.4, 0.85),
                        metadata={"level": level, "print_side": side_print,
                                   "print_size": size_print},
                    ))
                    break
    return cands


def _detect_holes(micro: pl.DataFrame, p: MicrostructureParams) -> list[_Candidate]:
    """Top-of-book thin (< hole_size_pct * rolling median) with directional flow."""
    if micro.height < 60:
        return []
    df = micro.with_columns(
        pl.col("top_sz").rolling_median(window_size=300, min_samples=60).alias("top_sz_med"),
        pl.col("delta").rolling_sum(window_size=p.hole_flow_lookback_s).alias("flow"),
    )
    is_hole = pl.col("top_sz") < (p.hole_size_pct * pl.col("top_sz_med"))
    df = df.with_columns(
        is_hole.alias("hole"),
        pl.when(pl.col("flow") >= p.hole_flow_min_abs).then(pl.lit("long"))
        .when(pl.col("flow") <= -p.hole_flow_min_abs).then(pl.lit("short"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("flow_side"),
    ).filter(pl.col("hole") & pl.col("flow_side").is_not_null())

    cands: list[_Candidate] = []
    stop_d = p.stop_ticks * TICK_SIZE
    tgt_d = p.target_ticks * TICK_SIZE
    for r in df.iter_rows(named=True):
        side = Side.LONG if r["flow_side"] == "long" else Side.SHORT
        entry = float(r["ask_px"]) if side == Side.LONG else float(r["bid_px"])
        if side == Side.LONG:
            stop = entry - stop_d
            tgt = entry + tgt_d
        else:
            stop = entry + stop_d
            tgt = entry - tgt_d
        cands.append(_Candidate(
            ts=r["ts"], side=side, setup_id="liq_hole",
            entry_price=entry, stop_price=stop, target_price=tgt,
            confidence=_squash(abs(r["flow"]) / (3 * p.hole_flow_min_abs), 0.35, 0.8),
            metadata={"top_sz": int(r["top_sz"]), "top_sz_med": float(r["top_sz_med"]),
                       "flow": int(r["flow"])},
        ))
    return cands


# --------------------------------------------------------------------------
# Throttle + finalize
# --------------------------------------------------------------------------


def _throttle_and_finalize(
    cands: list[_Candidate], p: MicrostructureParams,
) -> list[Signal]:
    """Enforce per-setup cooldown + global max-per-day cap, then emit Signals."""
    cooldown = timedelta(seconds=p.min_seconds_between_signals)
    last_ts_per_setup: dict[str, datetime] = {}
    out: list[Signal] = []
    for c in cands:
        last = last_ts_per_setup.get(c.setup_id)
        if last is not None and (c.ts - last) < cooldown:
            continue
        last_ts_per_setup[c.setup_id] = c.ts
        out.append(Signal(
            timestamp=c.ts,
            side=c.side,
            confidence=c.confidence,
            specialist=p.specialist_id,
            setup_id=c.setup_id,
            entry_price=c.entry_price,
            stop_price=c.stop_price,
            target_price=c.target_price,
            metadata=c.metadata,
        ))
        if len(out) >= p.max_signals_per_day:
            break
    out.sort(key=lambda s: s.timestamp)
    return out


def _squash(x: float, lo: float, hi: float) -> float:
    """Linear squash into [lo, hi]; clamp."""
    if x <= 0.0:
        return lo
    if x >= 1.0:
        return hi
    return lo + x * (hi - lo)
