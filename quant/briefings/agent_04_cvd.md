# Agent #4 — Order Flow / CVD Specialist

## Mission

Emit trading signals based on cumulative volume delta (CVD) divergences,
absorption, and exhaustion patterns. The CVD aggregates are already
computed and saved by the parent as `data/processed/cvd_1s/*.parquet`.

## Output module

`src/strategy/specialists/cvd_signals.py`

```python
from dataclasses import dataclass
from src.strategy.specialists._interface import SpecialistParams
from src.common.types import Signal
import polars as pl

@dataclass
class CVDParams(SpecialistParams):
    specialist_id: str = "cvd"
    divergence_window: int = 300        # seconds
    absorption_min_volume: int = 2000
    exhaustion_atr_mult: float = 1.5
    ...

def generate_signals(bars: pl.DataFrame, params: CVDParams) -> list[Signal]:
    ...
```

## Hypotheses to test (pick the strongest 2-3)

1. **CVD-price divergence**: 5-min price makes higher high, CVD makes lower
   high → SHORT setup. Mirror for LONG.
2. **Absorption**: heavy volume in a tight price range (low ATR-5) WITHOUT
   price moving → institutional defense; fade the side that exhausted.
3. **Exhaustion spike**: extreme positive delta tick (top decile) immediately
   followed by failure to make new high → reversal.

For each named hypothesis you keep, include a `setup_id` literal in the
emitted Signal.

## Helper utilities

Add to your module (or to a shared `src/strategy/specialists/_orderflow.py`
if you really need to):

```python
def detect_divergence(bars, cvd, window) -> list[(ts, side, score)]: ...
def detect_absorption(quotes_or_cvd, params) -> list[(ts, side, score)]: ...
def detect_exhaustion(bars, cvd, params) -> list[(ts, side, score)]: ...
```

## Validation

Run the standalone audit on 2020-2023, validate on 2024 hold-out:

```bash
PYTHONPATH=. python -m src.backtest.standalone --specialist cvd \
    --period 2020-01:2023-12 --validate-on 2024-01:2024-12
```

- POSITIVE_EDGE: dev PF ≥ 1.30, hold-out PF ≥ 1.10, hold-out WR drop < 7pp.
- MARGINAL: hold-out PF 1.00–1.10.
- NEGATIVE: hold-out PF < 1.00.

## Deliverable checklist

- [ ] `src/strategy/specialists/cvd_signals.py`
- [ ] `tests/test_cvd_signals.py` (≥ 6 tests)
- [ ] Standalone audit printed with verdict
- [ ] No more than ~50 signals/day per setup on average (else tighten thresholds)
