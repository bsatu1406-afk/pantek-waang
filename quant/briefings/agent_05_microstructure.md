# Agent #5 — Microstructure Specialist

## Mission

Emit trading signals derived from L1 TBBO microstructure features:

- Book imbalance (top-of-book bid_size vs ask_size).
- Microprice deviation from mid.
- Sweep detection (multi-level walk in a single tick).
- Trapped traders (price returns to a previous big-print level → stop fill).
- Liquidity holes (low-density orderbook visible in inter-trade quote churn).

## Output module

`src/strategy/specialists/microstructure.py`

```python
@dataclass
class MicrostructureParams(SpecialistParams):
    specialist_id: str = "micro"
    imbalance_threshold: float = 3.0      # bid/ask ratio
    microprice_dev_ticks: float = 1.5
    sweep_min_levels: int = 3
    ...

def generate_signals(bars: pl.DataFrame, params: MicrostructureParams) -> list[Signal]:
    """`bars` is 1s OHLCV; load TBBO via loader for the same session."""
```

## Data

Use `src.data.loader.load_tbbo(date)` to fetch TBBO rows for the same session
as `bars`. **Do not** scan the parquet directly — go through the loader so
caching works.

## Anti-patterns

- ❌ Emitting > 10 signals per hour per setup — your threshold is too loose.
- ❌ Looking at trades on the same tick as the signal (lookahead).
- ❌ Mixing this with CVD signals — that belongs to Agent #4.

## Validation thresholds

Same gating as Agent #4 (POSITIVE / MARGINAL / NEGATIVE based on hold-out
2024).

## Deliverable checklist

- [ ] `src/strategy/specialists/microstructure.py`
- [ ] `tests/test_microstructure.py` (≥ 5 tests)
- [ ] Standalone audit verdict
