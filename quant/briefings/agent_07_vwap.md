# Agent #7 — VWAP & Mean Reversion Specialist

## Mission

Generalize the SHORT-only VWAP pullback that worked in the v3 session. The
specialist must produce signals from session VWAP + anchored VWAP bands.

## Output module

`src/strategy/specialists/vwap_extended.py`

```python
@dataclass
class VWAPParams(SpecialistParams):
    specialist_id: str = "vwap_ext"
    band_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0)
    pullback_required_atr_mult: float = 1.0
    slope_min: float = 0.0
    ...

def session_vwap(bars: pl.DataFrame) -> pl.DataFrame: ...
def anchored_vwap(bars: pl.DataFrame, anchor_ts) -> pl.DataFrame: ...

def generate_signals(bars: pl.DataFrame, params: VWAPParams) -> list[Signal]:
    """Both LONG and SHORT pullback setups."""
```

## Hypotheses

1. **VWAP pullback short**: extended ≥ 2σ above VWAP + aggression sell tick
   (CVD turns negative) → SHORT, stop above swing high, target VWAP.
2. **VWAP pullback long** (mirror): extended ≥ 2σ below VWAP +
   aggression buy → LONG.
3. **Anchored VWAP rejection**: price tests prior-session VWAP from below;
   if rejection (red candle + negative delta) → SHORT.

The v3 session found LONG side was structurally negative in 2025 sample.
**Test BOTH sides honestly** and report per-side WR/PF in the audit.

## Verdict thresholds

Per-side reporting required. POSITIVE_EDGE applies to whichever side(s)
pass; the other side(s) drop to NEGATIVE in your output and the ensemble
will mask them off.

## Deliverable checklist

- [ ] `src/strategy/specialists/vwap_extended.py`
- [ ] `tests/test_vwap_extended.py` (≥ 5 tests; include pullback geometry)
- [ ] Standalone audit verdict, per side
