# API Reference

Base URL: `http://<host>:8000`

All `/v1/*` endpoints require an `X-API-Key` header and are rate-limited by
the value of `Settings.rate_limit_per_minute` (default 120 requests / minute
per API key). Administrative endpoints under `/admin/*` require a JWT bearer
token obtained via `POST /admin/login`.

The response envelope is always:

```json
{
  "symbol": "SPXW",
  "computed_at": "2026-06-01T13:30:00+00:00",
  "next_update_in_seconds": 42,
  "data": { /* endpoint-specific payload */ }
}
```

`computed_at` is `null` when no metrics have been written yet — this is
typically the case right after `docker compose up` before the first
scheduler tick completes.

## Public data endpoints (`X-API-Key`)

### `GET /v1/{symbol}/gex`

Per-strike Gamma Exposure curve plus aggregate top-positive / top-negative
strikes.

| Query param | Type | Default | Description |
|-------------|------|---------|-------------|
| `mode` | `oi` \| `volume` | `oi` | OI-weighted or volume-weighted GEX. |
| `expiry` | `all` \| YYYY-MM-DD | `all` | Filter to a single expiry. |

Response `data`:

```json
{
  "net_total": -1.23e9,
  "curve": [{ "strike": 4500, "gamma_exposure": -1.2e8 }, …],
  "top_positive": [ … ],
  "top_negative": [ … ]
}
```

### `GET /v1/{symbol}/max-pain`

Max-pain strike per expiry plus a chain-wide aggregate.

| Query param | Type | Default | Description |
|-------------|------|---------|-------------|
| `expiry` | `nearest` \| `all` \| YYYY-MM-DD | `nearest` | Which expiries to include. |

Response `data`:

```json
{
  "per_expiry": [
    { "expiration": "2026-06-13", "strike": 4495, "pain": 1.2e7 }
  ],
  "aggregate": { "strike": 4500, "value": 8.4e7 }
}
```

### `GET /v1/{symbol}/walls`

Top OI / volume walls per side.

| Query param | Type | Default | Description |
|-------------|------|---------|-------------|
| `mode` | `oi` \| `volume` \| `both` | `both` | Which walls to return. |

### `GET /v1/{symbol}/iv`

ATM IV, IV skew per expiry, and a sampled volatility surface.

### `GET /v1/{symbol}/snapshot` *(Rev 3)*

Single-call aggregate that returns every metric type the pipeline
produces — GEX (net / curve / top), Vanna, Charm, Walls, Max-Pain,
Zero-Gamma, ATM IV, IV skew, surface, HIRO cumulative, signed premium,
flow event counts, regime score + label, and per-section `computed_at`
timestamps.

Useful for a UI that needs a single consolidated render.

### `GET /v1/{symbol}/flow` *(Rev 3)*

Flow events (SWEEP / BLOCK / UOA) filtered by event type and time range.

| Query param | Type | Default |
|-------------|------|---------|
| `event_type` | `SWEEP` \| `BLOCK` \| `UOA` \| `all` | `all` |
| `since` | ISO-8601 timestamp | last 1 h |
| `limit` | integer | 100 |

### `GET /v1/{symbol}/hiro` *(Rev 3)*

HIRO cumulative signed premium history with optional bucket size.

| Query param | Type | Default |
|-------------|------|---------|
| `bucket` | `1m` \| `5m` \| `15m` | `1m` |
| `since` | ISO-8601 timestamp | last 1 h |

### `WS /v1/{symbol}/stream` *(Rev 3)*

WebSocket endpoint that pushes a JSON frame each time the pipeline
completes a cycle for `{symbol}`. Frame shape mirrors the
`/snapshot` payload.

Authentication: `X-API-Key` is read from the **upgrade request
headers**, or from a `?key=<api-key>` query string if the client
cannot send custom headers.

Connection cap per key: `Settings.max_ws_connections_per_key`
(default 5).

### `GET /v1/{symbol}/stream/sse` *(Rev 3)*

Server-Sent Events fallback for clients (browsers behind corporate
proxies that strip the `Upgrade` header) that cannot connect via
WebSocket. Same payload as `/stream`.

## Administrative endpoints (`Authorization: Bearer <jwt>`)

* `POST /admin/login` — exchange username/password for a JWT.
* `GET  /admin/system/status` — combined health snapshot (live
  ingester diagnostics, last pipeline run per symbol, DLQ depth,
  futures lag, OPRA lag, flow events / hour). *Rev 3.*
* `GET  /admin/inspector` — feed-level diagnostics from the ingesters.
* `GET  /admin/inspector/dlq` — paginated dead-letter queue. *Rev 3.*
* CRUD on `/admin/api-keys`, `/admin/alert-rules`.

## Errors

* `401 Unauthorized` — bad/missing API key or JWT.
* `403 Forbidden` — API key does not have access to the requested symbol.
* `404 Not Found` — symbol is unknown or no data has been computed yet.
* `422 Unprocessable Entity` — invalid query parameter (e.g. malformed
  `expiry` date).
* `429 Too Many Requests` — rate limit exceeded.
* `503 Service Unavailable` — backend not ready (DB unreachable).

All error responses include a JSON body `{"detail": "..."}` per FastAPI
conventions.
