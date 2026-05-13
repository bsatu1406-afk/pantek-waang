"""Walk-forward validation harness for the orderflow desk (Agent #11).

Replaces the orchestrator's fallback with a stricter implementation that

* Asserts anti-leakage invariants *before* any user-supplied function runs.
* Never swallows exceptions from the user-supplied callables (per the v4
  briefing's anti-pattern list — silent excepts are auto-rejected).
* Hands strategy_fn / backtest_fn the per-day `dict[date, polars.DataFrame]`
  slice so the callable can iterate session-by-session (matching the
  `BarLoader.iter_sessions` contract).
* Produces the full aggregate column set required by `briefings/
  agent_11_walk_forward.md`:
      mean_wr, std_wr, mean_pf, std_pf,
      wd_per_month_mean, wd_per_month_std,
      breach_rate, n_positive_folds, n_negative_folds, n_folds.

Fold geometry (defaults train=12m, test=3m, step=3m, purge=1d):

    fold k anchor = anchor_0 + k * step_months
        train: [anchor, anchor + train_months - 1 day]
        gap  : purge_days calendar days
        test : [train_end + purge + 1 day, ...+ test_months - 1 day]

A fold is emitted only when the test window ends on or before the last
available bar date. Days absent from `bars_by_date` are simply absent from
the per-fold slices — the harness never synthesizes empty data.

Anti-leakage invariants (`RuntimeError` if violated):

    1. min(test_slice.dates) - max(train_slice.dates) > purge_days
    2. max(train_slice.dates) < min(test_slice.dates)              (strict)
    3. strategy_fn is only handed bars in [train_start, train_end]
    4. backtest_fn(test) is only handed bars in [test_start, test_end]
"""
from __future__ import annotations

import json
import logging
import math
import statistics
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardFold:
    """One train/test fold produced by `run_walkforward`."""

    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_metrics: dict
    test_metrics: dict
    params_used: dict

    def as_dict(self) -> dict:
        return {
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "train_metrics": _jsonable(self.train_metrics),
            "test_metrics": _jsonable(self.test_metrics),
            "params_used": _jsonable(self.params_used),
        }


@dataclass
class WalkForwardReport:
    """All folds + aggregate metrics for a walk-forward run."""

    folds: list[WalkForwardFold] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)

    # geometry echoed back for downstream inspection
    train_months: int = 0
    test_months: int = 0
    step_months: int = 0
    purge_days: int = 0

    def as_dict(self) -> dict:
        return {
            "train_months": self.train_months,
            "test_months": self.test_months,
            "step_months": self.step_months,
            "purge_days": self.purge_days,
            "folds": [f.as_dict() for f in self.folds],
            "aggregate": _jsonable(self.aggregate),
        }

    # kept for source-compat with the orchestrator's fallback
    def to_json(self, path: str | Path | None = None) -> Any:
        """If `path` is given, write JSON there and return the path.
        Otherwise return the JSON-serializable dict (fallback compat)."""
        if path is None:
            return self.as_dict()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2, default=str))
        return p

    def to_markdown(self, path: str | Path) -> Path:
        """Write a human-readable per-fold table + aggregate summary."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = ["# Walk-forward report\n"]
        lines.append(
            f"- folds: **{len(self.folds)}**  "
            f"train={self.train_months}m / test={self.test_months}m / "
            f"step={self.step_months}m / purge={self.purge_days}d\n"
        )
        lines.append("## Aggregate\n")
        agg = self.aggregate
        for k in (
            "mean_wr", "std_wr", "mean_pf", "std_pf",
            "wd_per_month_mean", "wd_per_month_std",
            "breach_rate", "n_positive_folds", "n_negative_folds", "n_folds",
        ):
            v = agg.get(k)
            if isinstance(v, float):
                lines.append(f"- `{k}`: {v:.4f}")
            else:
                lines.append(f"- `{k}`: {v}")
        lines.append("\n## Folds\n")
        lines.append("| # | train | test | test wr | test pf | test net_pnl | breach |")
        lines.append("|---|---|---|---|---|---|---|")
        for i, f in enumerate(self.folds, start=1):
            m = f.test_metrics
            lines.append(
                f"| {i} | {f.train_start}..{f.train_end} | "
                f"{f.test_start}..{f.test_end} | "
                f"{_fmt(m.get('wr'))} | {_fmt(m.get('pf'))} | "
                f"{_fmt(m.get('net_pnl'))} | {m.get('breach', False)} |"
            )
        p.write_text("\n".join(lines) + "\n")
        return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if math.isinf(v):
            return "inf"
        return f"{v:.4f}"
    return str(v)


def _jsonable(obj: Any) -> Any:
    """Recursively coerce to JSON-friendly types (date -> iso, set -> list, ...)."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, float) and math.isinf(obj):
        return "inf"
    return obj


def add_months(d: date, months: int) -> date:
    """Calendar-month addition. Clamps the day to the target month's last day."""
    total = d.year * 12 + (d.month - 1) + months
    y, m = divmod(total, 12)
    m += 1
    day = d.day
    while True:
        try:
            return date(y, m, day)
        except ValueError:
            day -= 1
            if day < 1:
                raise


def _slice_by_date(
    bars_by_date: Mapping[date, pl.DataFrame],
    start: date,
    end_inclusive: date,
) -> dict[date, pl.DataFrame]:
    return {
        d: b for d, b in bars_by_date.items()
        if start <= d <= end_inclusive
    }


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return statistics.fmean(xs) if xs else 0.0


def _stdev(xs: Iterable[float]) -> float:
    xs = list(xs)
    return statistics.pstdev(xs) if xs else 0.0


def _as_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(f):
        return default
    return f


def _months_span(start: date, end_inclusive: date) -> float:
    """Approximate number of months in [start, end] inclusive."""
    days = max((end_inclusive - start).days + 1, 1)
    return days / 30.4375


def _aggregate(folds: list[WalkForwardFold]) -> dict:
    """Aggregate per-fold test metrics into the agent_11 column set."""
    if not folds:
        return {
            "n_folds": 0,
            "mean_wr": 0.0, "std_wr": 0.0,
            "mean_pf": 0.0, "std_pf": 0.0,
            "wd_per_month_mean": 0.0, "wd_per_month_std": 0.0,
            "breach_rate": 0.0,
            "n_positive_folds": 0, "n_negative_folds": 0,
        }

    wrs: list[float] = []
    pfs: list[float] = []
    wd_per_month: list[float] = []
    breaches: list[int] = []
    n_pos = 0
    n_neg = 0
    for f in folds:
        m = f.test_metrics or {}
        wrs.append(_as_float(m.get("wr")))
        pf = _as_float(m.get("pf"))
        # Cap PF at 10 so a fold with zero losses doesn't blow up mean/std.
        if math.isinf(pf) or pf > 10.0:
            pf = 10.0
        pfs.append(pf)

        # Prefer an explicit per-month withdrawal if the user supplied it;
        # else fall back to (total_withdrawn / months) over the test window.
        if "wd_per_month" in m:
            wd_per_month.append(_as_float(m.get("wd_per_month")))
        else:
            months = _months_span(f.test_start, f.test_end)
            wd = _as_float(m.get("total_withdrawn"))
            wd_per_month.append(wd / months if months > 0 else 0.0)

        breaches.append(1 if bool(m.get("breach", False)) else 0)

        net = _as_float(m.get("net_pnl"))
        if net > 0:
            n_pos += 1
        elif net < 0:
            n_neg += 1

    return {
        "n_folds": len(folds),
        "mean_wr": _mean(wrs), "std_wr": _stdev(wrs),
        "mean_pf": _mean(pfs), "std_pf": _stdev(pfs),
        "wd_per_month_mean": _mean(wd_per_month),
        "wd_per_month_std": _stdev(wd_per_month),
        "breach_rate": _mean(breaches),
        "n_positive_folds": n_pos,
        "n_negative_folds": n_neg,
    }


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


StrategyFn = Callable[
    [dict | None, Mapping[date, pl.DataFrame]],
    dict | None,
]
BacktestFn = Callable[
    [dict, Mapping[date, pl.DataFrame]],
    dict | None,
]


def run_walkforward(
    strategy_fn: StrategyFn,
    backtest_fn: BacktestFn,
    bars_by_date: Mapping[date, pl.DataFrame],
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
    purge_days: int = 1,
    *,
    initial_params: dict | None = None,
    start: date | None = None,
    end: date | None = None,
) -> WalkForwardReport:
    """Run a rolling walk-forward with purged train/test folds.

    Geometry (with defaults train=12m, test=3m, step=3m, purge=1d) — fold k:

        train: [anchor + k*step,            anchor + k*step + 12m - 1d]
        gap  : purge_days calendar days
        test : [train_end + purge + 1d,     train_end + purge + 3m]

    Parameters
    ----------
    strategy_fn : ``(params, bars_train_by_date) -> trained_params dict``.
    backtest_fn : ``(params, bars_eval_by_date) -> metrics dict`` (called on
        BOTH the train slice — for in-sample metrics — and the test slice).
    bars_by_date : ``date -> polars.DataFrame`` mapping.
    train_months / test_months / step_months : window geometry in months.
    purge_days : calendar-day gap inserted between train and test.
    initial_params : seed params handed to `strategy_fn` on every fold.
    start, end : optional date clamps for the anchor sweep.

    Returns
    -------
    WalkForwardReport
    """
    if not bars_by_date:
        raise ValueError("bars_by_date is empty")
    if train_months <= 0 or test_months <= 0 or step_months <= 0:
        raise ValueError("train/test/step months must all be > 0")
    if purge_days < 0:
        raise ValueError("purge_days must be >= 0")

    all_dates = sorted(bars_by_date.keys())
    first_bar = all_dates[0]
    last_bar = all_dates[-1]
    anchor = start if start is not None else first_bar
    horizon = end if end is not None else last_bar
    if anchor > horizon:
        raise ValueError(f"start {anchor} after end {horizon}")

    folds: list[WalkForwardFold] = []
    cursor = anchor
    safety = 0  # guard against pathological inputs
    while True:
        safety += 1
        if safety > 10_000:
            raise RuntimeError("walk-forward exceeded 10k fold iterations")

        train_start = cursor
        train_end = add_months(train_start, train_months) - timedelta(days=1)
        test_start = train_end + timedelta(days=purge_days + 1)
        test_end = add_months(test_start, test_months) - timedelta(days=1)

        if test_end > horizon:
            break

        train_slice = _slice_by_date(bars_by_date, train_start, train_end)
        test_slice = _slice_by_date(bars_by_date, test_start, test_end)

        if not train_slice or not test_slice:
            cursor = add_months(cursor, step_months)
            if cursor > horizon:
                break
            continue

        # Anti-leakage invariants — raise BEFORE calling user fns.
        max_train_d = max(train_slice)
        min_test_d = min(test_slice)
        if max_train_d >= min_test_d:
            raise RuntimeError(
                f"leakage: train_end {max_train_d} >= test_start {min_test_d}"
            )
        gap = (min_test_d - max_train_d).days
        if gap <= purge_days:
            raise RuntimeError(
                f"insufficient purge gap: {gap} day(s) between train and test, "
                f"need > {purge_days}"
            )

        # NOTE: exceptions from user fns are intentionally NOT caught — see
        # BRIEFING.md "Anti-patterns" (silent excepts are auto-rejected).
        trained = strategy_fn(initial_params, train_slice)
        if trained is None:
            trained = {}
        if not isinstance(trained, dict):
            trained = {"value": trained}

        train_metrics = backtest_fn(trained, train_slice) or {}
        test_metrics = backtest_fn(trained, test_slice) or {}

        folds.append(
            WalkForwardFold(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_metrics=dict(train_metrics),
                test_metrics=dict(test_metrics),
                params_used=dict(trained),
            )
        )

        cursor = add_months(cursor, step_months)
        if cursor > horizon:
            break

    return WalkForwardReport(
        folds=folds,
        aggregate=_aggregate(folds),
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
        purge_days=purge_days,
    )


__all__ = [
    "WalkForwardFold",
    "WalkForwardReport",
    "add_months",
    "run_walkforward",
]
