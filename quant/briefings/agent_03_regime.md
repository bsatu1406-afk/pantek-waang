# Agent #3 — Market Regime Classifier

## Mission

For every RTH session day in **2020-05 → 2025-05** produce a regime label
that other specialists / the ensemble can condition on. The classifier
itself does NOT emit signals — but it exposes `lookup_regime(date)`.

## Output module

`src/strategy/specialists/regime.py`

```python
from enum import Enum
from datetime import date
import polars as pl

class Regime(str, Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGE         = "range"
    HIGH_VOL_EVENT= "high_vol_event"
    LOW_VOL_DRIFT = "low_vol_drift"
    GAP_DAY       = "gap_day"
    REVERSAL_DAY  = "reversal_day"

class VolBucket(str, Enum):
    LOW_VOL  = "low_vol"   # bottom quartile ATR-20
    MED_VOL  = "med_vol"   # middle two quartiles
    HIGH_VOL = "high_vol"  # top quartile

def classify_regimes(bars_by_day: pl.DataFrame) -> pl.DataFrame:
    """Input: long-form bars with session_date. Output:
        DataFrame[session_date, regime, vol_regime, open, high, low, close, atr20, ...]
    """

def lookup_regime(d: date) -> tuple[Regime, VolBucket]: ...
```

Persist the classification once computed:

```
data/processed/regime/regime_table.parquet
```

Specialists can mmap this file via Polars `scan_parquet` for fast lookups.

## Suggested features

- 20-period ATR (daily timeframe), quartiles for VolBucket.
- Daily range / 20-day average range ratio.
- Trend: 5-day SMA slope vs ±SMA20.
- Gap: |open_t / close_{t-1} - 1| > 0.5%.
- Reversal: |close_t / open_t - 1| < 10 bp BUT high - low > 2× 20-day avg.

## Validation requirements

- Distribution of regime labels MUST be reported year-over-year. If one
  regime is >40% in 2024 but <10% in 2020, flag it.
- Print a small ASCII heatmap in your final report.

## Verdict thresholds

This specialist itself is NOT signal-generating; its verdict is **stability**:

- POSITIVE_EDGE: year-over-year regime distribution std-dev < 8pp.
- MARGINAL: 8–15pp.
- NEGATIVE: > 15pp — your features are not stable; redesign.

## Deliverable checklist

- [ ] `src/strategy/specialists/regime.py` with classifier + lookup + Enum types.
- [ ] `tests/test_regime.py` — at least 5 tests.
- [ ] Run `classify_regimes` on the available processed bars and save the
      Parquet table.
- [ ] Print regime-year heatmap to stdout in your final report.
