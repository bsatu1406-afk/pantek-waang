"""Dead-letter queue (DLQ) writer for ingestion failures.

Any ingester that drops a record because it cannot be parsed, has invalid
fields, or repeatedly fails to write should call :func:`record_dlq` so
operators can inspect the offending payload via `/admin/inspector/dlq`.

The DLQ writer is intentionally **best-effort**: a failure to persist to
the DLQ table is logged but never re-raised, so DLQ failures cannot
themselves crash the ingester. We also cap the in-memory queue depth
at :attr:`Settings.ingestion_dlq_max_size` so a runaway feed can't OOM
the process before the periodic flush lands.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.dialects.postgresql import insert

from app.config import get_settings
from app.core.logging import get_logger
from app.db.models import DeadLetterEntry
from app.db.session import get_session_factory

logger = get_logger(__name__)


class DeadLetterQueue:
    """In-memory ring buffer that periodically drains to ``dead_letter_queue``.

    Designed to be a process-wide singleton — see :func:`get_dlq`.
    """

    def __init__(self, *, max_size: int | None = None) -> None:
        settings = get_settings()
        capacity = max_size or settings.ingestion_dlq_max_size
        self._buffer: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = asyncio.Lock()
        self._flush_interval_s = 5.0

    @property
    def pending(self) -> int:
        return len(self._buffer)

    async def add(
        self, *, source: str, reason: str, payload: dict | None = None
    ) -> None:
        async with self._lock:
            self._buffer.append(
                {
                    "ts": datetime.now(UTC),
                    "source": source,
                    "reason": reason,
                    "payload": payload,
                }
            )
        logger.warning(
            "dlq_recorded",
            source=source,
            reason=reason,
            pending=self.pending,
        )

    async def flush(self) -> int:
        async with self._lock:
            if not self._buffer:
                return 0
            batch = list(self._buffer)
            self._buffer.clear()

        try:
            factory = get_session_factory()
            async with factory() as session:
                stmt = insert(DeadLetterEntry).values(batch)
                await session.execute(stmt)
                await session.commit()
        except Exception:  # noqa: BLE001 — DLQ failure must never propagate
            logger.exception("dlq_flush_failed", rows=len(batch))
            # Best-effort: push the entries back so we try again next tick.
            async with self._lock:
                for entry in reversed(batch):
                    self._buffer.appendleft(entry)
            return 0
        return len(batch)

    async def periodic_flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self._flush_interval_s)
            try:
                await self.flush()
            except Exception:  # noqa: BLE001
                logger.exception("dlq_periodic_flush_error")


_dlq: DeadLetterQueue | None = None


def get_dlq() -> DeadLetterQueue:
    global _dlq
    if _dlq is None:
        _dlq = DeadLetterQueue()
    return _dlq


async def record_dlq(
    *, source: str, reason: str, payload: dict | None = None
) -> None:
    """Convenience wrapper used by ingesters / pipelines.

    Falls back to a log statement if a DLQ instance is somehow not yet
    constructed (defensive — should never happen in practice).
    """
    await get_dlq().add(source=source, reason=reason, payload=payload)


# Type alias for callers that prefer dependency injection.
DlqRecorder = Callable[..., Awaitable[None]]
