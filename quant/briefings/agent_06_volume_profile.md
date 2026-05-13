# Agent #6 — Volume Profile Specialist

## Mission

Build volume profile structure features and emit signals around them:

- Value Area (VA, 70% volume range)
- Point of Control (POC, single max-volume tick)
- High-Volume Nodes (HVN) and Low-Volume Nodes (LVN)
- Naked POC from prior session
- Profile shapes: D (balanced), b (bottom heavy), p (top heavy),
  double distribution

## Output module

`src/strategy/specialists/volume_profile.py`

```python
@dataclass
class VPParams(SpecialistParams):
    specialist_id: str = "vp"
    bin_ticks: int = 4         # 1 tick = 0.25 → 4 ticks = 1 point bins
    va_volume_pct: float = 0.70
    naked_poc_lookback_days: int = 5
    ...

def build_profile(bars: pl.DataFrame, params: VPParams) -> dict:
    """Return a dict with poc, va_high, va_low, hvns, lvns, naked_pocs."""

def generate_signals(bars: pl.DataFrame, params: VPParams) -> list[Signal]:
    ...
```

## Hypotheses

1. **Naked POC magnet**: price approaches an unfilled prior-day POC →
   LONG/SHORT toward it depending on side.
2. **LVN slip-through**: price punches through a LVN with momentum → trend
   continuation in that direction.
3. **VAH/VAL bounce**: first test of value area boundary in RTH gets bought
   (VAL) / sold (VAH).

Use `bars` only — TBBO not needed for profiles.

## Verdict thresholds

Same gating: standalone audit on 2020-2023, validate on 2024.

## Deliverable checklist

- [ ] `src/strategy/specialists/volume_profile.py`
- [ ] `tests/test_volume_profile.py` (≥ 5 tests, profile math first)
- [ ] Standalone audit verdict
