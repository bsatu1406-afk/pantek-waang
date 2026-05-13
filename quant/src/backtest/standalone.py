"""Standalone 1-contract MES audit runner for a single specialist.

This is the harness every child specialist uses to validate their module
before reporting back to the parent.

Usage:
    PYTHONPATH=. python -m src.backtest.standalone \
        --specialist vwap_extended \
        --period 2020-01:2023-12 \
        --validate-on 2024-01:2024-12 \
        --report

The harness:

1. Imports `src.strategy.specialists.<specialist>` dynamically.
2. Iterates over RTH session days in the period, loads bars, calls
   `generate_signals(bars, params)`.
3. For each signal, simulates a 1-contract MES trade:
     - entry at signal.entry_price
     - exit at first of: stop_price, target_price (if set, else 1×ATR_5
       favorable move), 60s pattern-fail timeout, or EOD flat.
4. Aggregates WR, PF, net PnL, per-side breakdown.
5. Assigns a verdict (POSITIVE_EDGE / MARGINAL / NEGATIVE) using thresholds
   defined in the specialist's brief.

The audit is intentionally simple — the engine (`PropFirmEngine`) is NOT used
here. Engine integration happens in the ensemble runner.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import polars as pl

from src.common.types import EdgeLabel, Side, Signal
from src.data.loader import BarLoader
from src.simulation.engine import MES_PER_POINT

log = logging.getLogger(__name__)


@dataclass
class TradeResult:
    signal: Signal
    exit_ts: datetime
    exit_price: float
    pnl: float
    reason: str


def _parse_period(s: str) -> tuple[date, date]:
    a, b = s.split(":")
    start = date.fromisoformat(a + "-01") if len(a) == 7 else date.fromisoformat(a)
    end_year, end_month = map(int, b.split("-"))
    # last day of `b`
    next_month = end_month + 1
    next_year = end_year
    if next_month > 12:
        next_month = 1
        next_year += 1
    end = date(next_year, next_month, 1) - timedelta(days=1)
    return start, end


def _simulate_trade(signal: Signal, bars: pl.DataFrame) -> TradeResult:
    """Walk forward bar-by-bar from signal.timestamp+1s, exit on stop/target/EOD."""
    df = bars.filter(pl.col("ts") > signal.timestamp).sort("ts")
    if df.is_empty():
        return TradeResult(signal, signal.timestamp, signal.entry_price, 0.0, "no_data")
    target = signal.target_price
    if target is None:
        # default: 1× ATR_5 favourable move from entry
        sign = 1 if signal.side == Side.LONG else -1
        # compute a quick ATR-5 (5 bars range mean) at the signal bar
        recent = bars.filter(pl.col("ts") <= signal.timestamp).tail(5)
        if recent.is_empty():
            atr = 0.5
        else:
            atr = (recent["high"] - recent["low"]).mean() or 0.5
        target = signal.entry_price + sign * atr * 1.0

    for row in df.iter_rows(named=True):
        ts = row["ts"]
        hi = row["high"]
        lo = row["low"]
        if signal.side == Side.LONG:
            if lo <= signal.stop_price:
                px = signal.stop_price
                return TradeResult(signal, ts, px,
                                   (px - signal.entry_price) * MES_PER_POINT, "stop")
            if hi >= target:
                px = target
                return TradeResult(signal, ts, px,
                                   (px - signal.entry_price) * MES_PER_POINT, "tp")
        else:
            if hi >= signal.stop_price:
                px = signal.stop_price
                return TradeResult(signal, ts, px,
                                   (signal.entry_price - px) * MES_PER_POINT, "stop")
            if lo <= target:
                px = target
                return TradeResult(signal, ts, px,
                                   (signal.entry_price - px) * MES_PER_POINT, "tp")
    # Force EOD exit at last bar close
    last = df.tail(1).to_dicts()[0]
    sign = 1 if signal.side == Side.LONG else -1
    pnl = sign * (last["close"] - signal.entry_price) * MES_PER_POINT
    return TradeResult(signal, last["ts"], last["close"], pnl, "eod_flat")


def _verdict(net: float, wr: float, pf: float, n: int) -> EdgeLabel:
    if n < 30:
        return EdgeLabel.MARGINAL
    if pf >= 1.30 and wr >= 0.55:
        return EdgeLabel.POSITIVE_EDGE
    if pf >= 1.05 and wr >= 0.50:
        return EdgeLabel.MARGINAL
    return EdgeLabel.NEGATIVE


def _aggregate(results: list[TradeResult]) -> dict:
    if not results:
        return {"trades": 0, "wins": 0, "wr": 0.0, "pf": 0.0, "net_pnl": 0.0,
                "long_wr": 0.0, "short_wr": 0.0, "long_n": 0, "short_n": 0}
    wins = [r for r in results if r.pnl > 0]
    losses = [r for r in results if r.pnl < 0]
    net = sum(r.pnl for r in results)
    pf = (sum(r.pnl for r in wins) /
          abs(sum(r.pnl for r in losses))) if losses else float("inf")
    longs = [r for r in results if r.signal.side == Side.LONG]
    shorts = [r for r in results if r.signal.side == Side.SHORT]
    return {
        "trades": len(results),
        "wins": len(wins),
        "wr": len(wins) / len(results),
        "pf": pf,
        "net_pnl": net,
        "long_n": len(longs),
        "short_n": len(shorts),
        "long_wr": (sum(1 for r in longs if r.pnl > 0) / len(longs)) if longs else 0.0,
        "short_wr": (sum(1 for r in shorts if r.pnl > 0) / len(shorts)) if shorts else 0.0,
    }


def run(
    specialist_name: str,
    period_dev: tuple[date, date],
    period_holdout: Optional[tuple[date, date]] = None,
    params: Optional[object] = None,
) -> dict:
    mod = importlib.import_module(f"src.strategy.specialists.{specialist_name}")
    gen_fn = getattr(mod, "generate_signals", None)
    if gen_fn is None:
        raise RuntimeError(f"specialist '{specialist_name}' missing generate_signals()")
    if params is None:
        # Try to instantiate a default Params dataclass DEFINED in this module
        # (exclude imports like SpecialistParams).
        params_cls = next(
            (
                v for k, v in vars(mod).items()
                if (
                    k.endswith("Params")
                    and isinstance(v, type)
                    and getattr(v, "__module__", "") == mod.__name__
                )
            ),
            None,
        )
        params = params_cls() if params_cls is not None else None

    loader = BarLoader()
    dev_results: list[TradeResult] = []
    for sd, bars in loader.iter_sessions(period_dev[0], period_dev[1]):
        try:
            sigs = gen_fn(bars, params)
        except Exception as e:
            log.warning("specialist %s failed on %s: %s", specialist_name, sd, e)
            continue
        for s in sigs:
            dev_results.append(_simulate_trade(s, bars))
    dev_agg = _aggregate(dev_results)
    verdict = _verdict(dev_agg["net_pnl"], dev_agg["wr"], dev_agg["pf"], dev_agg["trades"])
    out = {"specialist": specialist_name, "dev_period": [str(period_dev[0]), str(period_dev[1])],
           "dev": dev_agg, "dev_verdict": verdict.value}

    if period_holdout is not None:
        ho_results: list[TradeResult] = []
        for sd, bars in loader.iter_sessions(period_holdout[0], period_holdout[1]):
            try:
                sigs = gen_fn(bars, params)
            except Exception:
                continue
            for s in sigs:
                ho_results.append(_simulate_trade(s, bars))
        ho_agg = _aggregate(ho_results)
        ho_verdict = _verdict(ho_agg["net_pnl"], ho_agg["wr"], ho_agg["pf"], ho_agg["trades"])
        out["holdout_period"] = [str(period_holdout[0]), str(period_holdout[1])]
        out["holdout"] = ho_agg
        out["holdout_verdict"] = ho_verdict.value
    return out


def _print_report(r: dict) -> None:
    print(f"specialist={r['specialist']} dev={r['dev_period'][0]}..{r['dev_period'][1]}")
    d = r["dev"]
    print(f"  trades={d['trades']} wins={d['wins']} wr={d['wr']*100:.2f}% "
          f"pf={d['pf']:.2f} net_pnl=${d['net_pnl']:,.2f}")
    print(f"  long_n={d['long_n']} long_wr={d['long_wr']*100:.1f}%  "
          f"short_n={d['short_n']} short_wr={d['short_wr']*100:.1f}%")
    print(f"  dev_verdict: {r['dev_verdict']}")
    if "holdout" in r:
        h = r["holdout"]
        print(f"  holdout {r['holdout_period'][0]}..{r['holdout_period'][1]}: "
              f"trades={h['trades']} wr={h['wr']*100:.1f}% pf={h['pf']:.2f} "
              f"net=${h['net_pnl']:,.2f} verdict={r['holdout_verdict']}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--specialist", required=True)
    p.add_argument("--period", default="2020-01:2023-12", help="YYYY-MM:YYYY-MM")
    p.add_argument("--validate-on", default=None)
    p.add_argument("--report", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dev = _parse_period(args.period)
    ho = _parse_period(args.validate_on) if args.validate_on else None
    out = run(args.specialist, dev, ho)
    if args.report:
        _print_report(out)
    else:
        import json
        print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
