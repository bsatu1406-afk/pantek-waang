# Specialist Briefing — Orderflow Strategy v4

You are a **child specialist agent** in a 10-agent trading desk. The parent
session has cloned this repo, built the simulation engine, downloaded raw ES
data from Databento, and is processing it into Parquet. Your job is to ship
ONE specialist module that emits trading signals from preprocessed bars.

> Read this file completely before writing any code. Then read your specific
> role in `briefings/agent_XX_<role>.md`. Then go.

## Quick context

- Universe: **ES (E-mini S&P 500)** futures, continuous front-month
  (`ES.v.0` from Databento `GLBX.MDP3`).
- Execution sym: **MES** (1/10 size = $5/point, $1.25/tick). Specialists
  reason in ES price terms; the engine handles MES PnL conversion.
- Session: **RTH only** (09:30–16:00 ET, US/Eastern). No overnight hold.
- Data range: **2020-05-13 → 2025-05-13** (5 years).
- Backtests run with **1 MES contract** for standalone audit, and with the
  prop firm engine for ensemble/walk-forward.

## What the parent has already built

| File | Purpose |
|---|---|
| `src/common/types.py` | `Signal`, `Trade`, `Side` enums |
| `src/simulation/engine.py` | `PropFirmEngine` — v4 rules pinned |
| `src/data/processor.py` | DBN.zst → Parquet (per-session-date) |
| `src/data/loader.py` | `load_bars(date)`, `load_tbbo(date)`, `load_cvd_1s(date)` |
| `src/strategy/specialists/_interface.py` | the `generate_signals` contract |
| `tests/test_engine.py` | 14 engine rule unit tests |

**Do not modify** anything in `src/simulation/engine.py` or the
`Signal` dataclass — those are pinned by the v4 briefing.

## Data schema (what `load_bars(date)` returns)

```python
pl.DataFrame with columns:
    ts           : Datetime[ns, UTC]   # 1-second bar timestamp
    session_date : Date                # US/Eastern session date
    open, high, low, close : Float64
    volume       : Int64
```

`load_tbbo(date)` returns top-of-book + trade rows:

```python
    ts, session_date,
    bid_px, bid_sz, ask_px, ask_sz : Float64/Int64
    price, size : Float64/Int64        # trade event (NaN if quote-only)
    side : str ('B' / 'S' / 'N')       # buy/sell aggressor / none
    action : str ('A' / 'C' / 'T')     # add / cancel / trade
```

`load_cvd_1s(date)` returns per-second order-flow aggregates:

```python
    ts, session_date,
    delta : Int64         # buy_vol - sell_vol this second
    cvd   : Int64         # cumulative session delta
    buy_vol, sell_vol, n_trades : Int64
```

## The Signal contract (read this twice)

```python
from src.common.types import Signal, Side

Signal(
    timestamp = datetime(2024, 5, 4, 14, 32, tzinfo=UTC),
    side      = Side.SHORT,
    confidence= 0.72,             # 0..1 self-rated
    specialist= "vwap_ext",       # short id matching your module
    setup_id  = "vwap_pullback_short",
    entry_price = 5234.25,
    stop_price  = 5239.0,         # hard stop in price terms
    target_price= 5226.0,         # None = engine-managed exit
    metadata={"vwap_band": 2.1, "atr": 4.3},
)
```

**Rules**:

- `confidence` ∈ [0, 1]. Be honest. The ML ranker (Agent #9) uses this.
- `stop_price` is required and MUST be on the opposite side of entry.
- Time-stop / pattern-fail exits should NOT be encoded as separate signals —
  use `metadata` and let the ensemble runner manage them.
- One signal = one trade idea. Don't emit duplicates within a 60-second window
  unless the underlying setup has materially changed.

## Your module's contract

Create or fill in:

```
src/strategy/specialists/<your_module>.py
tests/test_<your_module>.py
```

The module MUST export:

```python
@dataclass
class <YourSpecialist>Params(SpecialistParams):
    # ... your concrete params with defaults
    ...

def generate_signals(
    bars: pl.DataFrame,           # contiguous bars for ONE session day
    params: <YourSpecialist>Params,
) -> list[Signal]:
    ...
```

The function should:

1. Accept ONE session day at a time (caller iterates over days).
2. Not mutate `bars`.
3. Return signals in chronological order.
4. Log nothing (use return value + metadata).
5. Be deterministic for a given `(bars, params)`.

## Standalone audit (you MUST run this before reporting done)

Run the parent's standalone backtest harness on your specialist:

```bash
PYTHONPATH=. python -m src.backtest.standalone \
    --specialist <your_module> \
    --period 2024-01:2024-12 \
    --report
```

It prints:

```
specialist=vwap_ext period=2024-01..2024-12
  trades=412 wins=247 wr=59.95% pf=1.78 net_pnl=+$3,124.00
  long_wr=46.0%  short_wr=72.1%
  verdict: POSITIVE_EDGE
```

A specialist with `MARGINAL` or `NEGATIVE` verdict will be dropped by the
parent. If your verdict is negative, iterate on params (max 2-3 iterations)
before reporting done.

## Coordination with the parent

When your module is ready:

1. `pytest tests/test_<your_module>.py -v` — all green.
2. `pre-commit run --all-files` if hook exists (it doesn't yet — skip).
3. Commit to your branch:
   `devin/<timestamp>-orderflow-prop-v4-spec-<your_module>` (NOT to the
   parent branch — the parent will merge you).
4. Push your branch.
5. Reply to parent with: branch name, standalone audit verdict, and any
   caveats.

If you need data the parent hasn't downloaded yet, tell them — DO NOT
download multi-year Databento data yourself (that would duplicate $1k+ of
spend).

## Anti-patterns

- ❌ Tuning params against any 2024+ period and reporting them as final
  (data leakage). Use **2020-2023** for development; 2024 is held out.
- ❌ Catching exceptions silently. Let them propagate.
- ❌ Modifying engine constants or the Signal contract.
- ❌ Importing pandas (use Polars).
- ❌ Slow Python loops over millions of rows. Use Polars expressions.
- ❌ Cherry-picking favourable periods to claim positive edge.

## End

Now read your specific brief in `briefings/agent_XX_<your_role>.md` and start.
