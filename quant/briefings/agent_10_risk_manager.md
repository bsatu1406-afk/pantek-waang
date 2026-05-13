# Agent #10 — Risk Manager & Multi-Account Optimizer

## Mission

Decide **whether and how much** to trade a given signal for a given account,
in a given state. This is the engine's veto layer.

## Output module

`src/strategy/specialists/risk_manager.py`

```python
from dataclasses import dataclass
from src.simulation.engine import Account
from src.common.types import Signal

@dataclass
class RiskParams:
    max_qty_per_signal: int = 1
    min_floor_distance: float = 60.0   # $ from daily-lock floor to allow new entries
    avoid_close_minutes: int = 30      # do not enter signals after 15:30 ET
    funded_profit_lock_cooldown_minutes: int = 0
    ...

def calculate_size(account: Account, signal: Signal, ensemble_state: dict) -> int: ...

def should_take_signal(
    account: Account, signal: Signal, ensemble_state: dict,
) -> tuple[bool, str]:
    """Returns (allow, reason). 'reason' MUST be a short verb-phrase."""

def monte_carlo_breach_prob(
    account: Account, daily_pnl_samples: list[float], n_iters: int = 5000,
) -> float:
    """Estimate the probability of bucket → 0 over the next 60 trading days."""
```

## Hard rules (always veto)

- account is not in `ACTIVE_EVAL` or `ACTIVE_FUNDED`.
- account.locked_today or account.profit_capped_today
- `account.equity - DAILY_LOCK_DELTA < account.eod_high_prev_day - DAILY_LOCK_DELTA + min_floor_distance`
- bucket remaining ≤ $250 (one lock away from breach) — fall back to `max_qty=1`
  if signal.confidence > 0.8 else veto entirely.
- signal.timestamp ≥ 15:30 ET.

## Multi-account allocation

If multiple alive accounts can take the same signal, prefer accounts with
**larger bucket cushion** and **lower current cycle count** (so we build
endapan before unlocking 80% mode).

```python
def allocate_signal(
    accounts: list[Account], signal: Signal,
) -> list[tuple[Account, int]]: ...
```

## Verdict thresholds

Risk Manager is not signal-generating; verdict is **breach reduction**.
Compare a backtest of the ensemble WITH vs WITHOUT your veto:

- POSITIVE: breach rate drops ≥ 30%, total withdrawal ≥ 90% of baseline.
- MARGINAL: 10-30% breach drop OR > 10% withdrawal loss.
- NEGATIVE: < 10% breach drop OR > 25% withdrawal loss.

## Deliverable checklist

- [ ] `src/strategy/specialists/risk_manager.py`
- [ ] `tests/test_risk_manager.py` (≥ 6 tests, all hard rules pinned)
- [ ] Comparative simulation A/B printed (with vs without veto)
