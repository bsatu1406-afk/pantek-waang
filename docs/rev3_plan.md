# Rev 3 multi-agent plan

The Rev 3 hardening pass was executed by **ten specialised agents** in four
phases. Each agent is scoped to a narrow slice of the system so the work can
be re-run, audited, or shipped independently in future cycles.

## Execution model

| Phase | Agents | Why this grouping |
|-------|--------|-------------------|
| 1 (foundations) | 1, 4, 6 | All other agents depend on the math being right (Agent 1), the DB shape being final (Agent 4), and the ingestion contract being firm (Agent 6). |
| 2 (core logic) | 2, 3, 7 | Metric correctness (Agent 2), flow correctness (Agent 3), and pipeline orchestration (Agent 7) can run in parallel once Phase 1 is in. |
| 3 (interfaces) | 5, 8, 10 | Streaming + REST + Frontend all consume the Phase-2 outputs and can land in parallel. |
| 4 (QA) | 9 | Tests, docs, runbooks, and the final acceptance gate. |

Phases 1–3 use child Devin sessions to parallelise truly independent work.
Phase 4 runs in the parent session because it needs the merged view of every
other agent's PR.

## Agent index

* **Agent 1 — BSM / IV correctness:** `processing/bsm.py`, `processing/iv.py`
  + property-based tests for put-call parity, vanna finite-diff, charm sign,
  IV round-trip across ATM / ITM / OTM @ r ∈ {0 %, 5 %}.
* **Agent 2 — Metric audits:** `gex.py`, `vanna_charm.py`, `walls.py`,
  `max_pain.py`, `regime.py`, `zero_gamma.py` — NaN/inf scrubbing, sign
  audits, regime hysteresis.
* **Agent 3 — Flow + Lee-Ready:** `lee_ready.py`, `hiro.py`, `flow_events.py`,
  `flow_pipeline.py` — edge cases, threshold tuning via `Settings.flow_*`.
* **Agent 4 — DB schema:** Migration 0004 (additive only), four new ORM
  models (`PipelineRun`, `DeadLetterEntry`, `BackfillCheckpoint`,
  `ContractAdv`), TimescaleDB compression policies.
* **Agent 5 — Streaming API:** `GET /v1/{symbol}/snapshot`, `WS /v1/{symbol}/stream`,
  SSE fallback, flow + HIRO history endpoints, in-process pub/sub notifier.
* **Agent 6 — Ingestion reliability:** DLQ, backpressure, registry refresh,
  graceful shutdown, EOD OI reconciliation, futures lag monitoring.
* **Agent 7 — Pipeline atomicity:** Transaction-wrapped `_persist_metrics`,
  parallel-symbol scheduler, loader coverage gate, alert dedup,
  `pipeline_runs` writes, completeness diff.
* **Agent 8 — API hardening:** GEX `mode={oi,volume}`, max-pain `expiry=`,
  typed request/response schemas, admin telemetry endpoints
  (`futures_lag_ms`, `opra_lag_ms`, `flow_events_last_hour`), strict input
  validation.
* **Agent 9 — Testing + docs:** `@integration` pytest marker, hypothesis
  property tests, coverage report, this `docs/` tree, refreshed README.
* **Agent 10 — Frontend:** Live WebSocket dashboard, GEX area chart, HIRO
  panel, Walls / Max-Pain cards, Flow feed, Regime badge, `recharts` dep.

## Acceptance criteria

All of the following must pass before the Rev 3 PR is merged:

* `python -m pytest` — 100 % pass rate (incl. all Rev 3 tests).
* `python -m ruff check app tests` — 0 violations.
* `npm run typecheck` — 0 TypeScript errors.
* `npm run build` — successful frontend build.
* `docker compose up --build` — all three services healthy.
* `GET /v1/SPXW/snapshot` returns ≥ 25 metric types with a non-null
  `computed_at` after at least one pipeline cycle.
* `ws://localhost:8000/v1/SPXW/stream` pushes a frame within 5 s of the
  next pipeline cycle.
* BSM IV round-trip recovers σ to within 1e-5 across the ATM/ITM/OTM
  grid.
* GEX net total sign test passes (all-call → positive, all-put → negative,
  or the reverse — whichever the codebase documents).
* HIRO sign test passes: a feed of 100 % customer-buy calls produces a
  strictly positive cumulative signed premium.
