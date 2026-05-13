# Prop Firm Engine — v4 Rules (PINNED)

This document describes the **exact** simulation rules implemented in
`src/simulation/engine.py`. The rules are PINNED by the v4 briefing — any
change requires an explicit briefing amendment.

The constants are exported as module-level globals so a specialist can
import-and-assert them in its own tests:

```python
from src.simulation.engine import (
    ACCOUNT_COST, STARTING_EQUITY, EVAL_PASS_EQUITY,
    DAILY_LOCK_DELTA, DAILY_PROFIT_CAP_FUNDED,
    BUCKET_START, BUCKET_DECREMENT,
    WD_CYCLE_THRESHOLD, WD_PCT_EARLY, WD_PCT_LATE, WD_PCT_SWITCH_AFTER,
    MES_PER_POINT,
)
```

## 1. Account lifecycle

```
        OPEN_ACCOUNT
             │
             ▼
       ACTIVE_EVAL  ──[equity ≥ 26,500]──►  ACTIVE_FUNDED
             │                                     │
        [bucket→0]                            [bucket→0]
             │                                     │
             ▼                                     ▼
   BREACHED_DURING_EVAL                  PASSED_EVAL_BREACHED_FUNDED
```

- Each account costs **$80** (one-off) and starts with $25,000 paper equity.
- An account is *alive* only while phase ∈ {ACTIVE_EVAL, ACTIVE_FUNDED}.
- Multiple accounts may be alive simultaneously (default cap **8**).

## 2. Eval pass

- Trigger: end-of-day equity ≥ **$26,500** (= STARTING_EQUITY + $1,500).
- Action: transition to `ACTIVE_FUNDED`, reset equity to $25,000, reset
  bucket to $1,000, reset cycles_done to 0, clear winning_days.
- The funded account is treated as a fresh start for purposes of profit
  accounting (`profit_base = equity - 25,000`), but the dynamic high-water
  reference rolls forward.

## 3. Daily lock (the $250 floor)

For every alive account, on every MTM tick during the session:

```
ref = account.eod_high_prev_day      # dynamic high-water from prior session
trigger = ref - 250
if account.equity ≤ trigger AND not account.locked_today:
    force-flat the position
    account.locked_today = True
    account.daily_locks += 1
    account.total_bucket -= 250
    if account.total_bucket ≤ 0:
        account.phase = BREACHED_FUNDED  (or BREACHED_EVAL)
```

- A lock cancels any further entries that day; existing positions are
  closed at the current bid/ask (here approximated by the MTM price).
- Bucket starts at $1,000 → 4 locks → 0 → breach.
- A "winning day" never decrements the bucket.

## 4. Funded daily profit cap (positive lock)

While `phase == ACTIVE_FUNDED`:

```
high_water = account.eod_high_prev_day
cap = high_water + 500
if account.equity ≥ cap AND not account.profit_capped_today:
    force-flat
    account.profit_capped_today = True
    NOTE: bucket is NOT decremented
```

A profit cap is a "positive lock" — it preserves the day's gain and stops
the account from giving back. It does NOT consume any of the bucket.

## 5. Withdrawal cycle

- Counter: `winning_days` — incremented at the end of any session whose
  EOD equity is strictly greater than the start-of-day equity. Non-winning
  days do NOT reset the counter.
- Trigger: `winning_days == 5` (non-consecutive allowed).
- Calculation:

```
profit_base = account.equity - 25,000      # endapan from prior cycles compounds
pct = 0.50 if account.cycles_done < 3 else 0.80
amount = max(0, profit_base * pct)
account.equity -= amount
account.external_cash += amount            # added to ecosystem cash
account.winning_days = 0
account.cycles_done += 1
```

- After 3 completed cycles at 50%, all subsequent cycles use 80%.
- No withdrawal occurs if `profit_base ≤ 0` (the engine returns `None`).

### Reinvestment

External cash is allowed to be reinvested into new accounts at the end of
the day:

```
while external_cash ≥ 80 AND n_active < max_active:
    spawn a new account
    external_cash -= 80
    purchase_cost += 80
```

This is what produces the multi-account compounding behaviour.

## 6. Position & PnL accounting

- Specialists emit Signals with `entry_price` in **ES** points.
- The engine executes **1 MES** per signal by default. PnL is computed at
  $5.00 per ES point (MES contract multiplier).
- Hard stop & target prices on a position are evaluated against the bar's
  high/low; whichever is hit first wins. If both are hit in the same bar
  the engine assumes the stop hit first (conservative).
- All positions are force-flat at session close (16:00 ET).
- Single open position per account at a time (MVP). The Risk Manager
  may add multi-leg logic later.

## 7. RTH boundary

- All bars are filtered to RTH (Mon-Fri 09:30–16:00 ET).
- The engine uses the first/last bar of the session for `on_session_open` /
  `on_session_close`.
- Holidays are excluded by the data pipeline (no bars for those dates).

## 8. Forbidden modifications

- Constants in `engine.py` (Section 1 list) are PINNED.
- `Account.equity`, `Account.total_bucket`, `Account.cycles_done` and
  related counters must only be mutated through `_record_*` and `_maybe_*`
  helpers. Specialists should NEVER touch them.
- The `Signal` dataclass field set is fixed; use `metadata` for additional
  info.

## 9. Verified test coverage

`tests/test_engine.py` includes 14 tests pinning all of the above:

- daily lock at exactly -$250 (and NOT at -$249.99)
- 4 locks → bucket 0 → breach
- profit days don't consume bucket
- funded $500 cap locks without bucket hit
- eval pass resets equity + bucket
- 50% / 80% cycle switch after cycle 3
- winning-day counter resets after withdrawal
- losing day doesn't increment winning counter
- no withdrawal at profit_base ≤ 0
- briefing Section 2 worked example (5 cycles compounding)
- position open/close PnL realization
- EOD force-flat
- reinvest spawns new accounts up to max_active
