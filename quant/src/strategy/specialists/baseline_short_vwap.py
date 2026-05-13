"""Baseline SHORT-only VWAP pullback specialist.

Provided by the Orchestrator (NOT a child specialist) as a smoke-test stub
so the runner can be validated before child modules land. It is a
deliberately simple reproduction of the v3 session's edge — extension and
LONG-side handling are owned by Agent #7.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import polars as pl

from src.common.types import Side, Signal
from src.strategy.specialists._interface import SpecialistParams


@dataclass
class BaselineShortVWAPParams(SpecialistParams):
    specialist_id: str = "baseline_short_vwap"
    band_sigma: float = 1.5
    min_minutes_since_open: int = 30
    stop_atr_mult: float = 1.0
    target_atr_mult: float = 1.5
    max_signals_per_day: int = 4


def _atr_5(df: pl.DataFrame) -> float:
    if df.height < 5:
        return 0.5
    last = df.tail(60)
    rng = (last["high"] - last["low"]).mean()
    return float(rng or 0.5)


def generate_signals(
    bars: pl.DataFrame, params: BaselineShortVWAPParams,
) -> list[Signal]:
    if bars.is_empty():
        return []
    df = bars.sort("ts")
    df = df.with_columns([
        (pl.col("high") + pl.col("low") + pl.col("close")).truediv(3).alias("typical"),
        pl.col("volume").cum_sum().alias("cv"),
    ])
    df = df.with_columns(
        (pl.col("typical") * pl.col("volume")).cum_sum().alias("ctpv"),
    )
    df = df.with_columns((pl.col("ctpv") / pl.col("cv")).alias("vwap"))
    # rolling std of close around vwap as a crude band proxy
    df = df.with_columns(
        ((pl.col("close") - pl.col("vwap")) ** 2).rolling_mean(window_size=300).sqrt().alias("dev"),
    )
    df = df.with_columns((pl.col("vwap") + params.band_sigma * pl.col("dev")).alias("upper_band"))
    df = df.with_columns((pl.col("close") - pl.col("upper_band")).alias("upper_excess"))

    sigs: list[Signal] = []
    rows = df.iter_rows(named=True)
    n = 0
    last_ts: datetime | None = None
    open_ts = None
    for row in rows:
        if open_ts is None:
            open_ts = row["ts"]
        mins_since_open = (row["ts"] - open_ts).total_seconds() / 60
        if mins_since_open < params.min_minutes_since_open:
            continue
        excess = row.get("upper_excess")
        if excess is None or excess <= 0:
            continue
        if last_ts is not None and (row["ts"] - last_ts).total_seconds() < 300:
            continue
        # extended above upper band; SHORT pullback toward VWAP
        atr = _atr_5(df.filter(pl.col("ts") <= row["ts"]))
        entry = row["close"]
        stop = entry + atr * params.stop_atr_mult
        target = entry - atr * params.target_atr_mult
        sigs.append(Signal(
            timestamp=row["ts"],
            side=Side.SHORT,
            confidence=min(1.0, float(excess) / max(atr, 0.5)),
            specialist="baseline_short_vwap",
            setup_id="vwap_pullback_short",
            entry_price=float(entry),
            stop_price=float(stop),
            target_price=float(target),
            metadata={"upper_band_excess": float(excess), "atr_5": float(atr)},
        ))
        last_ts = row["ts"]
        n += 1
        if n >= params.max_signals_per_day:
            break
    return sigs
