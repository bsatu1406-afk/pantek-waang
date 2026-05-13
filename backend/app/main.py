"""FastAPI application entrypoint."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

from app.api.deps import limiter
from app.api.endpoints import admin, data, health, inspector
from app.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.session import dispose_engine
from app.ingestion.bulk_writers import (
    get_flow_event_writer,
    get_futures_tick_writer,
    get_liquidity_snapshot_writer,
    get_options_trade_writer,
)
from app.ingestion.databento_globex import get_globex_live_ingester
from app.ingestion.databento_historical import run_historical_backfill
from app.ingestion.databento_live import get_live_ingester
from app.ingestion.writer import get_writer
from app.processing.scheduler import start_scheduler

logger = get_logger(__name__)


def _testing_mode() -> bool:
    return os.getenv("PYTEST_CURRENT_TEST") is not None or os.getenv("APP_TESTING") == "1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("startup", supported_symbols=settings.supported_symbols)

    background_tasks: list[asyncio.Task] = []
    scheduler = None

    if not _testing_mode():
        # Periodic flush of the in-memory writers (one per table).
        writer = get_writer()
        background_tasks.append(
            asyncio.create_task(writer.periodic_flush_loop(), name="writer_flush")
        )
        for w in (
            get_futures_tick_writer(),
            get_options_trade_writer(),
            get_flow_event_writer(),
            get_liquidity_snapshot_writer(),
        ):
            background_tasks.append(
                asyncio.create_task(
                    w.periodic_flush_loop(),
                    name=f"writer_flush_{w.model.__tablename__}",
                )
            )

        # Best-effort historical backfill (graceful no-op if API key missing).
        try:
            await run_historical_backfill()
        except Exception:  # noqa: BLE001
            logger.exception("historical_backfill_unhandled_error")

        # Live ingestion (graceful no-op if API key missing).
        try:
            get_live_ingester().start()
        except Exception:  # noqa: BLE001
            logger.exception("live_ingestion_start_failed")
        try:
            get_globex_live_ingester().start()
        except Exception:  # noqa: BLE001
            logger.exception("globex_live_start_failed")

        # 60s compute scheduler.
        try:
            scheduler = start_scheduler()
        except Exception:  # noqa: BLE001
            logger.exception("scheduler_start_failed")

    try:
        yield
    finally:
        logger.info("shutdown")
        if scheduler is not None:
            try:
                scheduler.shutdown(wait=False)
            except Exception:  # noqa: BLE001
                logger.exception("scheduler_shutdown_error")
        try:
            await get_live_ingester().stop()
        except Exception:  # noqa: BLE001
            logger.exception("live_ingester_stop_error")
        try:
            await get_globex_live_ingester().stop()
        except Exception:  # noqa: BLE001
            logger.exception("globex_ingester_stop_error")
        for t in background_tasks:
            t.cancel()
        for t in background_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        try:
            await get_writer().flush()
        except Exception:  # noqa: BLE001
            logger.exception("final_flush_error")
        for w in (
            get_futures_tick_writer(),
            get_options_trade_writer(),
            get_flow_event_writer(),
            get_liquidity_snapshot_writer(),
        ):
            try:
                await w.flush()
            except Exception:  # noqa: BLE001
                logger.exception("final_bulk_flush_error", table=w.model.__tablename__)
        await dispose_engine()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Options Flow Analytics Platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    # Note: we deliberately do NOT add SlowAPIMiddleware. It is based on
    # Starlette's BaseHTTPMiddleware which is incompatible with httpx
    # ASGITransport + anyio task groups. The @limiter.limit decorators on
    # individual routes still enforce limits; the middleware only adds extra
    # response headers we don't depend on.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(data.router)
    app.include_router(admin.router)
    app.include_router(inspector.router)
    return app


def _rate_limit_handler(request, exc: RateLimitExceeded):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )


app = create_app()
