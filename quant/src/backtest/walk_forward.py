"""Walk-forward validation harness (orchestrator-side fallback).

Agent #11 owns the canonical implementation in this same file. The
Orchestrator initially ships this minimal version so the ensemble runner is
unblocked even before #11 lands.

Train / test fold geometry:

    fold 0: train [s, s+train_months) | purge | test [s+train_months, s+train_months+test_months)
    fold k: shift by step_months

Per fold, `strategy_fn(params, bars_train)` returns trained params; then
`backtest_fn(params, bars_test)` returns a metrics dict. The harness
collects per-fold metrics and aggregates them.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

import polars as pl

log = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    params_used: dict
    train_metrics: dict
    test_metrics: dict


@dataclass
class WalkForwardReport:
    folds: list[WalkForwardFold] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "folds": [
                {
                    "train_start": str(f.train_start),
                    "train_end": str(f.train_end),
                    "test_start": str(f.test_start),
                    "test_end": str(f.test_end),
                    "params_used": f.params_used,
                    "train_metrics": f.train_metrics,
                    "test_metrics": f.test_metrics,
                } for f in self.folds
            ],
            "aggregate": self.aggregate,
        }


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _month_add(d: date, n: int) -> date:
    m = d.month - 1 + n
    return date(d.year + m // 12, (m % 12) + 1, 1)


def _fold_dates(
    start: date,
    end: date,
    train_months: int,
    test_months: int,
    step_months: int,
    purge_days: int,
) -> list[tuple[date, date, date, date]]:
    out: list[tuple[date, date, date, date]] = []
    s = _month_start(start)
    while True:
        tr_end = _month_add(s, train_months)
        te_start = tr_end + timedelta(days=purge_days)
        te_end = _month_add(_month_start(te_start), test_months)
        if te_end > end:
            break
        out.append((s, tr_end - timedelta(days=1), te_start, te_end - timedelta(days=1)))
        s = _month_add(s, step_months)
    return out


def run_walkforward(
    strategy_fn: Callable[[dict, "pl.DataFrame"], dict],
    backtest_fn: Callable[[dict, "pl.DataFrame"], dict],
    bars_by_date: dict[date, "pl.DataFrame"],
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
    purge_days: int = 1,
    base_params: dict | None = None,
) -> WalkForwardReport:
    if not bars_by_date:
        return WalkForwardReport()
    base_params = base_params or {}
    dates_sorted = sorted(bars_by_date.keys())
    universe_start = dates_sorted[0]
    universe_end = dates_sorted[-1]
    folds_geom = _fold_dates(
        universe_start, universe_end + timedelta(days=1),
        train_months, test_months, step_months, purge_days,
    )
    report = WalkForwardReport()
    for trs, tre, tes, tee in folds_geom:
        train_bars = pl.concat([bars_by_date[d] for d in dates_sorted if trs <= d <= tre], how="vertical_relaxed") if any(trs <= d <= tre for d in dates_sorted) else pl.DataFrame()
        test_bars = pl.concat([bars_by_date[d] for d in dates_sorted if tes <= d <= tee], how="vertical_relaxed") if any(tes <= d <= tee for d in dates_sorted) else pl.DataFrame()
        if train_bars.is_empty() or test_bars.is_empty():
            continue
        try:
            trained = strategy_fn(base_params, train_bars)
            train_metrics = backtest_fn(trained, train_bars)
            test_metrics = backtest_fn(trained, test_bars)
        except Exception as e:
            log.exception("fold %s..%s failed: %s", tes, tee, e)
            continue
        report.folds.append(WalkForwardFold(
            train_start=trs, train_end=tre,
            test_start=tes, test_end=tee,
            params_used=trained,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
        ))
    if report.folds:
        for k in ("wr", "pf", "wd_per_month", "net_pnl"):
            vals = [f.test_metrics.get(k) for f in report.folds if f.test_metrics.get(k) is not None]
            if not vals:
                continue
            report.aggregate[f"{k}_mean"] = float(mean(vals))
            report.aggregate[f"{k}_std"] = float(pstdev(vals)) if len(vals) > 1 else 0.0
        report.aggregate["n_folds"] = len(report.folds)
        report.aggregate["n_positive_folds"] = sum(
            1 for f in report.folds if f.test_metrics.get("net_pnl", 0) > 0
        )
    return report


def save_report(rep: WalkForwardReport, path: Path) -> None:
    path.write_text(json.dumps(rep.to_json(), indent=2, default=str))
