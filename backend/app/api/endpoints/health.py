"""Public health endpoint."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

from app.config import get_settings
from app.processing.scheduler import get_pipeline_state

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    settings = get_settings()
    state = get_pipeline_state()
    return {
        "status": "ok",
        "now": datetime.now(UTC).isoformat(),
        "supported_symbols": settings.supported_symbols,
        "compute_interval_seconds": settings.compute_interval_seconds,
        "last_compute_per_symbol": {
            sym: ts.isoformat() if ts else None for sym, ts in state.last_run.items()
        },
    }
