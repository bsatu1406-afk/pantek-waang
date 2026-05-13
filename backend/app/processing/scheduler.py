"""APScheduler wiring for periodic metric recomputation.

Rev 3 hardening (Agent 7):

* Symbols within a single scheduler tick are now processed concurrently via
  :func:`asyncio.gather` so a slow / failing symbol does not block the rest
  of the universe. Concurrency is bounded by a semaphore (default 4) so we
  don't stampede the DB connection pool.
* Every exception that escapes a tick coroutine is caught and logged — the
  scheduler thread is never allowed to die. Failed runs are also recorded
  in ``pipeline_runs`` (status='failed') by ``run_pipeline_for_symbol``
  itself.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import get_settings
from app.core.logging import get_logger
from app.ingestion.databento_eod_oi import run_eod_oi_ingestion
from app.processing.alert_pipeline import run_alert_pipeline
from app.processing.flow_pipeline import run_flow_pipeline
from app.processing.pipeline import run_pipeline_for_symbol

logger = get_logger(__name__)


# Maximum number of symbols processed concurrently within a single scheduler
# tick. Bounded so a wide universe doesn't exhaust the DB connection pool —
# each symbol takes ~3 short-lived sessions (load → persist → completeness),
# so 4 in flight ≈ 12 connections worst case.
DEFAULT_SYMBOL_CONCURRENCY: int = 4


class PipelineRunState:
    """Tracks the most recent successful pipeline run per symbol."""

    def __init__(self) -> None:
        self.last_run: dict[str, datetime] = {}
        self.last_duration_ms: dict[str, float] = {}

    def record(self, symbol: str, ts: datetime, duration_ms: float) -> None:
        self.last_run[symbol] = ts
        self.last_duration_ms[symbol] = duration_ms


_state = PipelineRunState()


def get_pipeline_state() -> PipelineRunState:
    return _state


async def _run_symbol_pipeline(symbol: str) -> None:
    """Run the chain → flow → alert pipeline trio for a single symbol.

    Every leg is wrapped in its own try/except so a failure in one stage
    never prevents the others from running. ``run_pipeline_for_symbol``
    additionally records a ``pipeline_runs`` row even on failure so the
    audit trail is preserved.
    """
    try:
        result = await run_pipeline_for_symbol(symbol)
    except Exception:  # noqa: BLE001
        logger.exception("pipeline_error", symbol=symbol)
    else:
        if result is not None:
            _state.record(symbol, result.ts, result.duration_ms)

    try:
        await run_flow_pipeline(symbol=symbol)
    except Exception:  # noqa: BLE001
        logger.exception("flow_pipeline_error", symbol=symbol)

    try:
        await run_alert_pipeline(symbol=symbol)
    except Exception:  # noqa: BLE001
        logger.exception("alert_pipeline_error", symbol=symbol)


async def _run_all_symbols(concurrency: int = DEFAULT_SYMBOL_CONCURRENCY) -> None:
    """Fan out the per-symbol pipeline across the supported universe.

    Uses ``asyncio.gather(..., return_exceptions=True)`` plus a bounded
    semaphore so a slow / failing symbol does not block the others. Any
    exception that escapes :func:`_run_symbol_pipeline` (which itself is
    defensive) is converted into a log line, never an unhandled task error
    that could kill the scheduler thread.
    """
    settings = get_settings()
    symbols = settings.supported_symbols
    if not symbols:
        return

    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def _bounded(sym: str) -> None:
        async with sem:
            await _run_symbol_pipeline(sym)

    results = await asyncio.gather(
        *(_bounded(s) for s in symbols), return_exceptions=True
    )
    for sym, res in zip(symbols, results, strict=False):
        if isinstance(res, BaseException):
            # _run_symbol_pipeline shouldn't raise, but defence-in-depth so
            # the scheduler thread cannot die from an exotic CancelledError
            # / asyncio.TimeoutError escape.
            logger.error(
                "pipeline_tick_uncaught",
                symbol=sym,
                error=f"{type(res).__name__}: {res}",
            )


def start_scheduler() -> AsyncIOScheduler:
    settings = get_settings()
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        _run_all_symbols,
        "interval",
        seconds=settings.compute_interval_seconds,
        id="compute_pipeline",
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now(UTC),
    )
    # End-of-day OI snapshot: refresh once per day at 22:30 UTC (~17:30 ET,
    # after US options markets close). Also run once on startup so a fresh
    # deployment isn't stuck waiting until tomorrow for OI data.
    scheduler.add_job(
        run_eod_oi_ingestion,
        CronTrigger(hour=22, minute=30, timezone="UTC"),
        id="eod_oi_daily",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        run_eod_oi_ingestion,
        "date",
        run_date=datetime.now(UTC),
        id="eod_oi_startup",
        max_instances=1,
    )
    scheduler.start()
    logger.info("scheduler_started", interval_seconds=settings.compute_interval_seconds)
    return scheduler


async def trigger_now() -> None:
    """Convenience hook for tests / startup."""
    await _run_all_symbols()
