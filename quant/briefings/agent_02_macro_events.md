# Agent #2 — Macro & Event Specialist

## Mission

Build a multi-year US event calendar and a filter-style "specialist" module
that:

1. Loads / hardcodes a calendar of high-impact macro events for ES from
   **2018-01-01 through 2025-12-31** (so we have a buffer either side of the
   backtest window 2020-05 → 2025-05).
2. Provides reusable helpers the other specialists call before deciding to
   take a signal.

## Output module

`src/strategy/specialists/macro_events.py`

```python
from datetime import date
from typing import Optional

EVENT_LOCKOUT_DATES: frozenset[date]   # blacklist any entry

def is_event_day(d: date) -> tuple[bool, str]:
    """(True, 'FOMC') if d is a high-impact event; (False, '') otherwise."""

def event_distance(d: date) -> dict:
    """Per-event distance in trading days, e.g.
       {'fomc': -1, 'cpi': 0, 'nfp': 2, 'opex': 5}
    """

def is_session_blocked(d: date) -> bool:
    """True if any high-impact event hits today (== d in EVENT_LOCKOUT_DATES)."""
```

Plus the standard signal contract — for this specialist, signals are
**event-conditioned opportunistic setups**: e.g. "2 hours pre-FOMC mean
reversion" or "post-CPI fade overreaction". Aim for 1–3 named setups, not
a kitchen sink.

```python
@dataclass
class MacroEventParams(SpecialistParams):
    pre_event_minutes: int = 120
    post_event_minutes: int = 60
    ...

def generate_signals(bars, params) -> list[Signal]: ...
```

## Events to cover (minimum)

- FOMC meetings (8/year) and FOMC minutes
- CPI, PPI, NFP (first Friday), GDP advance/revisions, ISM Manufacturing
- Powell speeches when scheduled (best-effort)
- OPEX (3rd Friday) and quad-witching
- US market holidays (full close + early close days)

## Tips

- The `pandas_market_calendars` package gives you the NYSE/CME calendar.
- For FOMC dates use a hardcoded list (date + estimated time of release).
- Group all event metadata in a single `EVENTS: list[Event]` constant. Each
  `Event` has `(date, kind, impact, et_time, source_url)`.
- Distance helpers should be lazy (compute on demand).

## Verdict thresholds

- POSITIVE_EDGE: standalone audit PF ≥ 1.30 across 2020-2023, WR ≥ 55%, and
  the verdict holds in 2024 hold-out.
- MARGINAL: PF 1.05–1.30 or hold-out drift > 25%.
- NEGATIVE: anything worse.

## Deliverable checklist

- [ ] `src/strategy/specialists/macro_events.py` with the 4 public helpers
      + `generate_signals` + `MacroEventParams`.
- [ ] `tests/test_macro_events.py` — at least 5 tests pinning event lookup
      and signal correctness.
- [ ] Standalone audit verdict printed.
- [ ] Push branch + report back to parent.
