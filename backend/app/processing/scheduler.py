"""APScheduler wiring for periodic metric recomputation."""

from __future__ import annotations

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


async def _run_all_symbols() -> None:
    settings = get_settings()
    for symbol in settings.supported_symbols:
        try:
            result = await run_pipeline_for_symbol(symbol)
        except Exception:  # noqa: BLE001
            logger.exception("pipeline_error", symbol=symbol)
            continue
        if result is not None:
            _state.record(symbol, result.ts, result.duration_ms)
        # Flow + alerts: best-effort, never block the chain pipeline.
        try:
            await run_flow_pipeline(symbol=symbol)
        except Exception:  # noqa: BLE001
            logger.exception("flow_pipeline_error", symbol=symbol)
        try:
            await run_alert_pipeline(symbol=symbol)
        except Exception:  # noqa: BLE001
            logger.exception("alert_pipeline_error", symbol=symbol)


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
