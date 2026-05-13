# Agent #8 — Momentum / Breakout Specialist

## Mission

Opposite philosophy from Agent #7: trade momentum/breakout, not reversion.

## Output module

`src/strategy/specialists/momentum.py`

```python
@dataclass
class MomentumParams(SpecialistParams):
    specialist_id: str = "momentum"
    or_minutes: tuple[int, ...] = (5, 15, 30)   # opening range candidates
    eod_drive_start_et: str = "14:30"
    breakout_buffer_ticks: int = 2
    ...

def generate_signals(bars: pl.DataFrame, params: MomentumParams) -> list[Signal]:
    """LONG and SHORT breakout signals from ORB / EOD drives / multi-bar ranges."""
```

## Hypotheses

1. **Opening Range Breakout** (5, 15, 30 min): break ORB high → LONG;
   break ORB low → SHORT. Stop on the opposite side. Trailing target.
2. **EOD momentum**: between 14:30 and 15:30 ET, if trend is intact
   (close > VWAP for LONG / < VWAP for SHORT and pullback to VWAP fails) →
   continuation entry.
3. **Multi-bar breakout**: tightening 5-min range followed by impulse
   candle in either direction.

Report **per-side** WR and PF. (LONG vs SHORT.)

## Verdict thresholds

- POSITIVE: PF ≥ 1.25 on the dev period AND hold-out PF ≥ 1.05 for the
  same side.
- MARGINAL: PF 1.05–1.25, or hold-out drop > 20%.
- NEGATIVE: hold-out PF < 1.00.

## Deliverable checklist

- [ ] `src/strategy/specialists/momentum.py`
- [ ] `tests/test_momentum.py` (≥ 5 tests, ORB geometry first)
- [ ] Standalone audit verdict, per side
