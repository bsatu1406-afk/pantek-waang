# Agent #11 — Validation & Walk-Forward Engineer

## Mission

Build the walk-forward backtest framework. You do **not** emit signals.

## Output module

`src/backtest/walk_forward.py`

```python
from dataclasses import dataclass
from datetime import date
import polars as pl

@dataclass
class WalkForwardFold:
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_metrics: dict
    test_metrics: dict
    params_used: dict

@dataclass
class WalkForwardReport:
    folds: list[WalkForwardFold]
    aggregate: dict     # mean/std of WR, PF, WD-per-month, breach-rate

def run_walkforward(
    strategy_fn,                    # callable(params, bars_train) -> trained_params
    backtest_fn,                    # callable(params, bars_test) -> metrics dict
    bars_by_date: dict,             # date -> polars df
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
    purge_days: int = 1,
) -> WalkForwardReport: ...
```

## Required output

After running on the final ensemble, save:

- `reports/walk_forward.json` — per-fold + aggregate metrics
- `reports/walk_forward.md` — human-readable summary

Aggregate columns: `mean_wr`, `std_wr`, `mean_pf`, `std_pf`,
`wd_per_month_mean`, `wd_per_month_std`, `breach_rate`, `n_positive_folds`,
`n_negative_folds`.

## Anti-pattern

- ❌ Using the same data for train + test in any fold.
- ❌ Skipping the purge (insider-info leakage at fold boundaries).
- ❌ Reporting only the aggregate without per-fold table.

## Deliverable checklist

- [ ] `src/backtest/walk_forward.py`
- [ ] `tests/test_walk_forward.py` (≥ 4 tests on fold geometry + leakage)
- [ ] Example run on a synthetic strategy (sanity check that the harness works)
