# Rev 3 — Production Hardening

A 10-agent, four-phase hardening pass that brings the platform from
"working in dev" to "ready for production traffic". Each agent owns a
narrow slice of correctness, durability, or operability. The complete
plan and per-agent prompts are checked in under `docs/` (see
[Rev 3 plan](docs/rev3_plan.md)) so future deployments can re-run any
single agent in isolation.

## Highlights

| Agent | Surface area | What landed |
|-------|--------------|-------------|
| 1 | `processing/bsm.py`, `processing/iv.py` | Newton-Raphson IV fallback, discounted-intrinsic floor, analytical vega, 64 new correctness tests |
| 2 | `processing/gex.py`, `vanna_charm.py`, `walls.py`, `max_pain.py`, `regime.py`, `zero_gamma.py` | NaN/inf scrubbing, sign-correctness audit, regime hysteresis via `GEX_REGIME_THRESHOLD` |
| 3 | `processing/lee_ready.py`, `hiro.py`, `flow_events.py`, `flow_pipeline.py` | Lee-Ready edge cases, HIRO bucket reset, SWEEP / BLOCK / UOA threshold tuning |
| 4 | `db/migrations/0004_*`, `db/models.py` | `pipeline_runs`, `dead_letter_queue`, `backfill_checkpoints`, `contract_adv` tables + TimescaleDB compression on hot hypertables |
| 5 | `api/endpoints/snapshot.py`, `stream.py`, `flow.py`, `hiro.py` | `GET /v1/{symbol}/snapshot` (25+ metric types), WebSocket `/stream`, SSE fallback, flow + HIRO history endpoints, in-process notifier |
| 6 | `ingestion/dlq.py`, `writer.py`, `bulk_writers.py`, `databento_live.py` | Dead-letter queue, backpressure shedding, periodic contract registry refresh, graceful shutdown that final-flushes buffers |
| 7 | `processing/pipeline.py`, `scheduler.py`, `loader.py` | Atomic `_persist_metrics` transactions, parallel-symbol `asyncio.gather`, loader coverage gate, alert dedup, `pipeline_runs` writes, completeness check |
| 8 | `api/endpoints/data.py`, `admin.py`, `schemas.py` | GEX `mode={oi,volume}`, max-pain `expiry=` filter, typed request/response schemas, admin telemetry endpoints, strict input validation |
| 9 | `tests/`, `docs/`, `CHANGES.md`, `README.md` | `@integration` marker, hypothesis property tests, coverage report, API reference, runbooks |
| 10 | `frontend/src/**` | Live WebSocket dashboard, GEX area chart, HIRO panel, Walls / Max-Pain cards, Flow feed, Regime badge, `recharts` |

## New configuration (env vars, all optional)

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEX_REGIME_THRESHOLD` | `0.2` | Deadband for regime score hysteresis (Agent 2). |
| `FLOW_SWEEP_MIN_PREMIUM` | `50000` | USD premium floor for sweep detection (Agent 3). |
| `FLOW_BLOCK_MIN_SIZE` | `100` | Contract count floor for block detection (Agent 3). |
| `FLOW_UOA_VOL_OI_RATIO` | `2.0` | Volume/OI ratio for UOA detection (Agent 3). |
| `UPSERT_BATCH_SIZE` | `1000` | Default batch size for buffered DB writers (Agent 4). |
| `INGESTION_MAX_PENDING_ROWS` | `10000` | Per-writer backpressure cap before DLQ shedding (Agent 6). |
| `INGESTION_DLQ_MAX_SIZE` | `1000` | DLQ in-memory ring-buffer cap (Agent 6). |
| `INGESTION_REGISTRY_REFRESH_SECONDS` | `14400` | Live-ingester contract registry re-bootstrap interval (Agent 6). |
| `FUTURES_FEED_LAG_WARN_MS` | `5000` | Warn threshold for stale futures feed (Agent 8). |
| `MAX_WS_CONNECTIONS_PER_KEY` | `5` | Streaming API per-key connection cap (Agent 5). |

## New DB objects (migration 0004)

* `pipeline_runs` — one row per scheduler tick per symbol with
  duration, status, `missing_metric_types[]`, error.
* `dead_letter_queue` — persistent record of unparseable / un-writeable
  ingestion payloads, surfaced via `/admin/inspector/dlq`.
* `backfill_checkpoints` — `(dataset, symbol) → last_completed_at`
  bookmark so historical backfills resume after restart.
* `contract_adv` — trailing-N-day ADV per contract for UOA detection.

Plus additive indexes on `computed_metrics`, `options_chain`, and
`flow_events`, and TimescaleDB **compression policies** on
`options_chain` / `options_trades` / `futures_ticks` /
`computed_metrics` (compress chunks older than 1 day; segment by
symbol). Migration 0004 is fully additive — migrations 0001–0003 are
untouched.

## Backwards compatibility

* All existing API routes keep their current names and response
  envelopes. New endpoints are additive.
* The `mode={oi,volume}` parameter on `/v1/{symbol}/gex` and the
  `expiry=` filter on `/v1/{symbol}/max-pain` are optional with
  unchanged defaults.
* Database compression and chunk-interval changes are no-ops on the
  test fixture (plain Postgres without TimescaleDB) — the migration
  detects the extension via `pg_extension` and skips when absent.

---

# Databento pipeline fixes (rev 2)

## Rev 2 — what changed (this update)

Four issues were diagnosed against a live RTH deployment after the dual-key
refactor and fixed in this revision.

### Fix 1 — Globex registry bootstrap (futures_ticks 0 rows)

`futures_ticks` and `liquidity_snapshots` were both empty despite the Globex
live stream emitting hundreds of thousands of `mbp-10` and `trades` records.
Root cause: `mbp-10` / `trades` payloads only carry `instrument_id`, not the
human-readable contract symbol. Without a registry, `_handle_trade` /
`_handle_mbp` saw `contract is None` and returned early on every record.

The fix mirrors the OPRA ingester: bootstrap the registry from a historical
`definition` snapshot (parents `ES.FUT` + `NQ.FUT`, 2-day window) before the
first stream connects. Verified: `_bootstrap_registry()` populates **63
unique contracts** (ES + NQ outrights and quarterly spreads).

Touched: `backend/app/ingestion/databento_globex.py` (new
`_bootstrap_registry()` invoked before the first `_stream_once()`).

### Fix 2 — `publisher_id` type coercion (options_trades 0 rows)

OPRA `TradeMsg.publisher_id` and Globex `TradeMsg.publisher_id` are int8s,
but the `options_trades.exchange` and `futures_ticks.venue` columns are
`Text`. asyncpg refuses to cast int → text and **silently fails the entire
batch**, which is why `options_trades` showed 0 rows even though
2 843 OPRA `TradeMsg`s had been received.

A `_coerce_str(value)` helper now wraps every `publisher_id` access in
both ingesters before the row reaches the writer.

Touched: `backend/app/ingestion/databento_live.py`,
`backend/app/ingestion/databento_globex.py`.

### Fix 3 — Capture & surface gateway System / Error frames

The 3 `ErrorMsg` records counted in OPRA cumulative were being silently
swallowed by the catch-all branch in `_handle_record`, leaving the operator
guessing why NDXP was stale. We now route `Error*` and `System*` records
into bounded ring buffers (`_error_messages`, `_system_messages`, last 10
each), log them at WARN / INFO, and surface them in `/admin/inspector` so
the Data Inspector renders the actual gateway message text in red.

Touched: `databento_live.py`, `databento_globex.py`,
`frontend/src/lib/api.ts`, `frontend/src/pages/DataInspector.tsx`.

### Fix 4 — Front-month CME futures as SPX/NDX spot fallback

When put-call parity fails (one-sided ATM quotes, holiday opens, illiquid
expiries), the loader now falls back to the latest `futures_ticks` outright
price minus a cached basis:

* `SPXW` / `SPX` → ES (front-month last) − cached `cash − futures` basis
* `NDXP` / `NDX` → NQ (front-month last) − cached basis

The basis cache is refreshed on every load whenever **both** parity and the
futures price are available, so it is always within one scheduler cycle of
the latest market state. The basis pipeline regex was also fixed: it was
greedy-matching ``ES`` → ``ESH`` and dropping every futures row in the
filter, which is why `BASIS_SPX_ES` was 0.

Touched: `backend/app/processing/loader.py` (rewritten
`_apply_underlying_synthesis` + new `_FUTURES_LAST_QUERY` +
`_BASIS_CACHE`), `backend/app/processing/flow_pipeline.py` (regex tightened
to `^([A-Z]{2})`).

### Verification

- `python -m ruff check app tests` → all checks passed
- `python -m pytest` → **91 passed, 7 warnings, 0 failed** (~12 s)
- `python -m _bootstrap_test` against live API → **63 contracts registered**
- `npm run typecheck` + `npm run build` → clean

---

# Databento dual-key + RTH live-feed verification (rev 1)

## What changed

Split the single `DATABENTO_API_KEY` setting into **two dataset-specific keys** so
each ingester authenticates against the subscription tier that actually carries
its data, while keeping full backwards compatibility for existing single-key
deployments.

### New env vars

| Variable                   | Used by                                                      | Dataset      |
|----------------------------|--------------------------------------------------------------|--------------|
| `DATABENTO_API_KEY_OPRA`   | `databento_live`, `databento_historical`, `databento_eod_oi` | OPRA.PILLAR  |
| `DATABENTO_API_KEY_GLOBEX` | `databento_globex`                                           | GLBX.MDP3    |
| `DATABENTO_API_KEY`        | Legacy fallback only — used when either of the above is empty | both         |

### Resolution logic (in `app/config.py`)

```python
@property
def opra_api_key(self) -> str:
    return self.databento_api_key_opra or self.databento_api_key

@property
def globex_api_key(self) -> str:
    return self.databento_api_key_globex or self.databento_api_key
```

So if you set both `DATABENTO_API_KEY_OPRA` and `DATABENTO_API_KEY_GLOBEX` you
can clear the legacy `DATABENTO_API_KEY` entirely. If you set only the legacy
one, both ingesters keep working as before.

### Files touched

- `backend/app/config.py`                       — added 2 new fields + 2 properties
- `backend/app/ingestion/databento_live.py`     — uses `opra_api_key`
- `backend/app/ingestion/databento_historical.py` — uses `opra_api_key`
- `backend/app/ingestion/databento_eod_oi.py`     — uses `opra_api_key`
- `backend/app/ingestion/databento_globex.py`     — uses `globex_api_key`
- `backend/tests/conftest.py`                   — sets the new vars to "" for tests
- `.env.example`, `README.md`                   — documented the split

### Verification

- `ruff check app tests` → all checks passed
- `pytest` → **91 passed, 7 warnings, 0 failed** (in 11 s)

---

## Live RTH verification (5 May 2026 ~17:56 UTC, US options + futures RTH)

Both keys were exercised against Databento's Live + Historical APIs in real time.

### OPRA Pillar (`DATABENTO_API_KEY_OPRA`)

| Test                                          | Result                                                  |
|-----------------------------------------------|---------------------------------------------------------|
| Historical `definition` SPXW.OPT (1d window)  | OK — instrument rows returned (5 sampled) with strike, expiration, instrument_class |
| Live stream SPXW.OPT 25 s drain               | **500 073 records** — `cmbp-1`, `trades`, `definition` all flowing |
|   - `CMBP1Msg` (consolidated MBP-1 quotes)    | 417 560 — top-of-book bid/ask present (`levels[0].bid_px / ask_px`) |
|   - `TradeMsg`                                | 1 157 (last sampled trade premium $1.90)                |
|   - `SymbolMappingMsg` (definition mapping)   | 81 352                                                  |
|   - `StatMsg` (live OI / volume)              | **0** — see "OI is null" below                          |
|   - `SystemMsg`                               | 4                                                       |

### GLBX.MDP3 (`DATABENTO_API_KEY_GLOBEX`)

| Test                                          | Result                                                  |
|-----------------------------------------------|---------------------------------------------------------|
| Historical `definition` ES.FUT (1d window)    | OK — ES contract curve returned (5 sampled)            |
| Historical `trades` ES.c.0 (10-min window)    | OK — front-month ES at $7287                            |
| Live stream ES.FUT 25 s drain                 | **3 099 records**                                        |
|   - `MBP10Msg` (top-10 book)                  | 2 824 — `levels[0].bid_px=$7290.75 / ask_px=$7291.00`   |
|   - `TradeMsg`                                | 149 — last live ES trade $7290.75                       |
|   - `SymbolMappingMsg`                        | 123                                                     |

**Conclusion: both keys are valid, both subscriptions are active, and both
streams flow at expected RTH volumes.** No auth or schema errors were observed.

---

## Why `underlying_price` and greeks are null in `options_chain`

This is **by design** — not a bug. The pipeline is:

```
OPRA Pillar live  ─►  options_chain
                       (raw quotes/trades; underlying_price = NULL,
                        iv/delta/gamma = NULL)

every 60 s scheduler ─►  loader.load_latest_snapshot()
                          ├─ synthesize_underlying_price() via
                          │   put-call parity (spot.py)
                          ├─ fill_missing_iv() via Black-Scholes
                          │   inversion (iv.py)
                          └─ analytical gamma/delta from (S,K,T,σ)
                               ─►  computed_metrics (extra_json carries
                                    underlying_price, gex curve, etc.)
```

Concretely, see:

- `app/ingestion/databento_live.py:610-630` — `_emit_row` writes
  `underlying_price = state.get("underlying_price")` and the live state dict
  is **never populated with one** because OPRA Pillar does not carry the SPX/NDX
  cash index. Same for `iv`, `delta`, `gamma`. This is documented in
  `app/processing/loader.py:8-11`:

  > "When `underlying_price` is missing (OPRA Pillar does not publish the SPX/NDX
  >  cash index), we synthesize it via put-call parity from the freshest
  >  near-the-money quotes — see `app.processing.spot`."

- `app/processing/loader.py:65-81` (`_apply_underlying_synthesis`) — runs every
  60 s inside the compute pipeline and fills the synthetic spot.

- `app/processing/iv.py:139-211` (`fill_missing_iv`) — inverts BS to derive IV
  from mid-quote, then computes analytical `gamma`/`delta`.

- `app/processing/pipeline.py:74,223,297` — the resulting `underlying_price`
  is what ends up persisted alongside GEX / vanna / charm / move tracker
  metrics in `computed_metrics.extra_json`.

### Why live OI is also zero on OPRA

Live OPRA Pillar only emits `StatMsg` records intra-day for things like
opening / closing prices, NOT for `open_interest` (stat_type 9). OPRA
publishes OI **end-of-day**, which is exactly what `databento_eod_oi.py`
already pulls daily at 22:30 UTC and uses as a fallback in
`loader._apply_eod_oi_fallback`. So the design is consistent: live OI = null,
EOD OI = filled in at compute time.

### What to verify *after* deploying with both keys

To confirm the analytics layer is producing non-null values, hit:

```bash
# 1. Liveness — should show non-zero record_counts after RTH start
curl -s -H "Authorization: Bearer <admin JWT>" \
     http://localhost:8000/admin/inspector/databento_live | jq

# 2. Computed metrics — should show non-null underlying_price + IV
curl -s -H "X-API-Key: <api key>" \
     "http://localhost:8000/v1/SPXW/gex?mode=oi&expiry=nearest" | jq '.data.underlying_price'
curl -s -H "X-API-Key: <api key>" \
     "http://localhost:8000/v1/SPXW/iv" | jq '.data.atm_iv'
```

If `data.underlying_price` is non-null and `atm_iv` is in the 0.10–0.50 range,
the full pipeline (live → synth → IV → greeks → metrics) is healthy.

---

## Optional follow-up (NOT in this change)

Now that the GLBX.MDP3 feed is live and we know ES front-month price in
real time, a future PR could **enrich `options_chain.underlying_price`
with the latest ES tick adjusted for SPX/ES basis** — that would give a more
accurate spot than the put-call-parity synthesis (which can drift when the
nearest-the-money quotes are wide). Today the codebase has the building
blocks (`processing/basis.py`, `futures_ticks` table) but does not wire
them into `loader._apply_underlying_synthesis`. Happy to do this as a
follow-up if useful.
