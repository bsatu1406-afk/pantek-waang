# Options Flow Analytics Platform

Backend foundation for an options analytics service: live + historical OPRA Pillar
ingestion via [Databento](https://databento.com), TimescaleDB for time-series
storage, a derivatives-metrics processing engine (GEX, max pain, walls, IV),
and a secured REST API consumed by ATAS.NET / MotiveWave indicator plugins
(out of scope for this phase). A React admin dashboard manages API keys and
exposes pipeline health.

> **Phase scope:** This repo covers the foundation only — pipeline, processing,
> database, REST API, and admin dashboard. The ATAS.NET and MotiveWave indicator
> plugins are deliberately **not** included; they will consume this API in a
> follow-up phase.

---

## Architecture at a glance

```
                         ┌──────────────────────┐
       OPRA Pillar       │     Databento        │
       (live + hist.)──▶ │   ingestion clients  │
                         └────────┬─────────────┘
                                  │ batches
                                  ▼
                         ┌──────────────────────┐        ┌─────────────────┐
                         │   options_chain      │        │  computed_      │
                         │   (TimescaleDB)      │◀──────▶│  metrics        │
                         └────────┬─────────────┘  60s   │  (TimescaleDB)  │
                                  │                      └────────┬────────┘
                                  │                               │
                         ┌────────▼─────────────┐        ┌────────▼─────────┐
                         │  Processing engine   │        │   FastAPI        │
                         │  GEX / MaxPain /     │───────▶│   /v1/* + admin  │
                         │  Walls / IV / skew   │        │   X-API-Key, JWT │
                         └──────────────────────┘        └────────┬─────────┘
                                                                  │
                                                         ┌────────▼─────────┐
                                                         │  React admin UI  │
                                                         │  (Vite + shadcn) │
                                                         └──────────────────┘
```

---

## Tech stack

| Layer            | Choice                                               |
|------------------|------------------------------------------------------|
| Backend          | Python 3.11 + FastAPI + SQLAlchemy 2 (async)         |
| Data store       | PostgreSQL 15 + **TimescaleDB** extension            |
| Migrations       | Alembic (async)                                      |
| Scheduling       | APScheduler `AsyncIOScheduler`                       |
| Ingestion        | `databento` Python client (Live + Historical)        |
| Math             | `numpy`, `pandas`, `scipy` (Black-Scholes inversion) |
| Auth             | bcrypt-hashed API keys + JWT for admin dashboard     |
| Rate limiting    | `slowapi` (per-API-key minute window)                |
| Logging          | Structured JSON via `structlog`                      |
| Frontend         | Vite + React 18 + TypeScript + Tailwind + shadcn/ui  |
| Containerization | Docker + Docker Compose                              |

---

## Running with Docker Compose

```bash
cp .env.example .env
# edit .env — at minimum set the two Databento keys, ADMIN_PASSWORD, JWT_SECRET:
#   DATABENTO_API_KEY_OPRA   → OPRA.PILLAR (options)
#   DATABENTO_API_KEY_GLOBEX → GLBX.MDP3   (CME futures)
# (the legacy single ``DATABENTO_API_KEY`` is still honoured as a fallback)
docker compose up --build
```

Services:

- **db** – TimescaleDB (PostgreSQL 15) – internal only
- **backend** – FastAPI on `http://localhost:8000`
- **frontend** – Admin dashboard on `http://localhost:3000`

The backend container automatically runs `alembic upgrade head` on startup and
then runs the historical backfill, starts the live Databento stream, and starts
the 60s compute scheduler.

If you don't have a Databento key yet, set `DISABLE_LIVE_INGESTION=true` and
`DISABLE_HISTORICAL_BACKFILL=true` in `.env`. The API and admin dashboard will
still come up cleanly; the data endpoints will return empty payloads until data
is ingested.

---

## Configuration (`.env`)

| Variable                       | Default                                | Notes                                                       |
|--------------------------------|----------------------------------------|-------------------------------------------------------------|
| `DATABENTO_API_KEY_OPRA`       | —                                      | API key for OPRA.PILLAR (live + historical options).        |
| `DATABENTO_API_KEY_GLOBEX`     | —                                      | API key for GLBX.MDP3 (CME ES/NQ futures live tape).        |
| `DATABENTO_API_KEY`            | —                                      | Legacy single-key fallback used if the two above are empty. |
| `DATABASE_URL`                 | `postgresql+asyncpg://options:options@db:5432/options_db` | Async SQLAlchemy URL.       |
| `ADMIN_USERNAME`               | `admin`                                | Admin dashboard login.                                      |
| `ADMIN_PASSWORD`               | `changeme`                             | Plain text **or** bcrypt hash starting with `$2`.           |
| `JWT_SECRET`                   | —                                      | HMAC secret for admin JWT tokens.                           |
| `JWT_EXPIRE_MINUTES`           | `480`                                  | Admin token lifetime (8h).                                  |
| `SUPPORTED_SYMBOLS`            | `SPXW,NDXP`                            | Comma-separated underlyings.                                |
| `RISK_FREE_RATE`               | `0.05`                                 | Used as `r` in Black-Scholes IV inversion.                  |
| `DATA_RETENTION_DAYS`          | `7`                                    | TimescaleDB drops data older than this.                     |
| `COMPUTE_INTERVAL_SECONDS`     | `60`                                   | Pipeline cadence.                                           |
| `HISTORICAL_BACKFILL_DAYS`     | `7`                                    | Window pulled on first startup.                             |
| `DISABLE_LIVE_INGESTION`       | `false`                                | Set `true` to skip the live stream (dev/testing).           |
| `DISABLE_HISTORICAL_BACKFILL`  | `false`                                | Set `true` to skip the historical pull (dev/testing).       |
| `RATE_LIMIT_PER_MINUTE`        | `120`                                  | Per-API-key rate limit on `/v1/*`.                          |
| `VITE_API_BASE_URL`            | `http://localhost:8000`                | Built into the frontend at build time.                      |

---

## Database schema

Hypertables (TimescaleDB) with a 7-day retention policy and compression on data
older than 1 day:

- `options_chain` – partitioned by `ts`. Holds the latest snapshot per
  `(symbol, expiration, strike, option_type)` along with OI, volume, IV, greeks,
  bid/ask, last price, and underlying price.
- `computed_metrics` – partitioned by `ts`. One row per metric per cycle.
  `metric_type` ∈ {`GEX_NET_TOTAL`, `GEX_NET_TOTAL_VOL`, `GEX_LEVEL`,
  `GEX_LEVEL_VOL`, `MAX_PAIN`, `MAX_PAIN_AGG`, `CALL_WALL_OI`, `PUT_WALL_OI`,
  `CALL_WALL_VOL`, `PUT_WALL_VOL`, `ATM_IV`, `IV_SKEW`, `IV_SURFACE`,
  `REGIME_OI`, `REGIME_VOL`,
  `VANNA_NET_TOTAL`, `VANNA_LEVEL`,
  `CHARM_NET_TOTAL`, `CHARM_LEVEL`,
  `IV_TERM_STRUCTURE`, `RISK_REVERSAL_25D`,
  `MOVE_TRACKER`, `PIN_PROBABILITY`,
  `HIRO`, `BASIS_SPX_ES`, `VOLUME_PROFILE_ES`}.
- `futures_ticks` – Globex MDP 3.0 trade tape (one row per tick), 14-day
  retention.
- `options_trades` – OPRA Pillar trade tape with Lee-Ready `side` and
  dealer-signed premium pre-computed at ingest, 14-day retention.
- `liquidity_snapshots` – top-N order-book snapshots from MBP-10 (1Hz),
  bids/asks stored as JSONB.

Regular tables:

- `api_keys` – bcrypt-hashed API keys (`key_hash`), display prefix
  (`key_prefix`, e.g. `ak_x7f2…`), per-key allowed symbols, expiry, usage stats.
- `admin_users` – schema for future multi-admin support; the bootstrap admin is
  authenticated against `ADMIN_USERNAME` / `ADMIN_PASSWORD` from the env.
- `flow_events` – sweep / block / UOA detections (one row per detected event).
- `alert_rules` – user-defined alert predicates (JSONB tree).
- `alert_events` – every firing of every alert rule.

Apply migrations with `alembic upgrade head` (the backend container does this
automatically on startup).

---

## API reference

### Public

| Method | Path       | Description                                                    |
|--------|------------|----------------------------------------------------------------|
| GET    | `/health`  | System status + last compute timestamp per symbol. No auth.    |

### End-user data (require `X-API-Key` header)

All responses are wrapped in `{ symbol, computed_at, next_update_in_seconds, data }`.

| Method | Path                          | Query                                          | Returns                                                               |
|--------|-------------------------------|------------------------------------------------|-----------------------------------------------------------------------|
| GET    | `/v1/{symbol}/gex`            | `mode=oi\|volume`, `expiry=YYYY-MM-DD\|all`     | Full GEX curve, top 5 positive / negative levels, net total.          |
| GET    | `/v1/{symbol}/max-pain`       | `expiry=YYYY-MM-DD\|nearest\|all`               | Max-pain strike per expiration + aggregate.                           |
| GET    | `/v1/{symbol}/walls`          | `mode=oi\|volume\|both`                         | Top 3 call & put wall strikes per mode.                               |
| GET    | `/v1/{symbol}/iv`             | —                                              | ATM IV, IV skew per expiry, full IV surface.                          |
| GET    | `/v1/{symbol}/snapshot`       | —                                              | All of the above merged into a single payload.                        |

End-user endpoints are rate-limited to **120 req/min per API key** (configurable).

### Admin (require `Authorization: Bearer <jwt>`)

| Method | Path                              | Body / params                                                       |
|--------|-----------------------------------|---------------------------------------------------------------------|
| POST   | `/admin/login`                    | `{ username, password }` → `{ access_token, expires_in_seconds }` |
| GET    | `/admin/api-keys`                 | List all keys (no plaintext).                                       |
| POST   | `/admin/api-keys`                 | `{ label, allowed_symbols, expires_at? }` → returns plaintext once. |
| PATCH  | `/admin/api-keys/{id}`            | Update label / symbols / expiry / `is_active`.                      |
| DELETE | `/admin/api-keys/{id}`            | Revoke (hard delete).                                               |
| GET    | `/admin/api-keys/{id}/usage`      | Per-key usage stats.                                                |
| GET    | `/admin/system/status`            | Pipeline + ingestion + DB row counts.                               |

---

## Processing engine

All metrics are recomputed every `COMPUTE_INTERVAL_SECONDS` (default 60s) for
each configured symbol.

| Module                       | What it computes                                                                                  |
|------------------------------|---------------------------------------------------------------------------------------------------|
| `app/processing/gex.py`      | GEX per strike (`gamma · OI · 100 · S² · 0.01`), call vs. put, net per strike, top ±5 levels.     |
| `app/processing/max_pain.py` | Classic max-pain per expiration + aggregate over the nearest 5 expiries.                          |
| `app/processing/walls.py`    | Top 3 call & put walls by OI **and** by volume.                                                   |
| `app/processing/iv.py`       | Black-Scholes inversion via `scipy.optimize.brentq`; ATM IV, 25-delta skew per expiry, full surface. |

The `pipeline.run_pipeline_for_symbol()` coroutine loads the latest snapshot,
runs all four calculators, and upserts results into `computed_metrics`.

---

## Admin dashboard

```
frontend/
├── src/pages/Login.tsx
├── src/pages/Dashboard.tsx
├── src/pages/ApiKeys.tsx
└── src/pages/SystemStatus.tsx
```

Features:

- JWT-based admin login (token stored in `localStorage`).
- Dashboard with system health, last compute, total active keys, row counts.
- API Keys table with create/edit/revoke and a one-time plaintext display modal.
- Live System Status polling at 5s.

For local development without Docker:

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000, proxies to VITE_API_BASE_URL
```

---

## Local backend development

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# point at a running Postgres (with TimescaleDB) — e.g. via `docker compose up db`
export DATABASE_URL=postgresql+asyncpg://options:options@localhost:5432/options_db
alembic upgrade head
uvicorn app.main:app --reload
```

### Running tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest                           # always-run pure-function + security tests
TEST_DATABASE_URL=postgresql+asyncpg://...   pytest    # adds DB-backed API/admin tests
```

If `TEST_DATABASE_URL` is not set, the conftest will try to spin up a Postgres
testcontainer. If Docker is not available, DB-backed tests are skipped and the
pure-function tests still run. Pure-function tests cover the full processing
engine (GEX, max pain, walls, IV) and the security primitives.

### Lint

```bash
cd backend && ruff check .
cd frontend && npm run lint && npm run typecheck
```

---

## Security notes

- API keys are generated as `ak_<urlsafe-token>` and stored as a bcrypt hash.
  Only the 11-character `key_prefix` is stored in plaintext for table display.
- Admin JWT tokens are HS256-signed with `JWT_SECRET` and expire after
  `JWT_EXPIRE_MINUTES` (default 8h).
- API key auth checks: existence, `is_active`, expiry, and per-key allowed symbols.
- Rate limiting is keyed on the `X-API-Key` header (falling back to client IP).
- The plaintext API key is shown to the admin **once** at creation time and is
  never stored or logged.

---

## Troubleshooting

**Live ingestion is failing with auth errors.** Confirm `DATABENTO_API_KEY_OPRA`
(for options) and `DATABENTO_API_KEY_GLOBEX` (for futures) are
set in `.env` and that the key has OPRA Pillar live + historical access.

**Compute pipeline reports `pipeline_no_data` for a symbol.** Either no data
has been ingested yet (give the live stream and historical backfill a minute),
or the symbol isn't in `SUPPORTED_SYMBOLS`.

**Admin login returns 401 with the right password.** Ensure the
`ADMIN_PASSWORD` in `.env` matches what you typed. If you store a bcrypt hash,
it must start with `$2`.

**Migrations fail in tests.** The test conftest creates schema via
`Base.metadata.create_all` (skipping TimescaleDB extension calls). For
production deployments, `alembic upgrade head` requires the TimescaleDB extension.
