"""Generic batched writers for Phase-2 tables.

Mirrors :class:`app.ingestion.writer.OptionsChainWriter` but
parameterised by ORM model + conflict-key set so we can reuse the same
backpressure / batching machinery for ``futures_ticks``,
``options_trades``, ``flow_events``, and ``liquidity_snapshots``.

Each writer is created lazily on first use and shared as a process-wide
singleton.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import (
    FlowEvent,
    FuturesTick,
    LiquiditySnapshot,
    OptionsTrade,
)
from app.db.session import get_session_factory

logger = get_logger(__name__)


class BulkUpsertWriter:
    """Batched insert / upsert writer parameterised by ORM model.

    The ``conflict_keys`` set must match the primary key (or a unique
    index) on the table. Set to ``None`` to disable conflict handling
    (plain ``INSERT … ON CONFLICT DO NOTHING`` is used in that case).
    """

    def __init__(
        self,
        model: Any,
        *,
        conflict_keys: Sequence[str] | None,
        batch_size: int = 2000,
        flush_interval_s: float = 2.0,
        on_conflict: str = "update",
    ) -> None:
        self.model = model
        self.conflict_keys = list(conflict_keys) if conflict_keys else None
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_s
        self._on_conflict = on_conflict
        self._buffer: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._last_flush_ts: datetime = datetime.utcnow()

    async def add(self, row: dict[str, Any]) -> None:
        async with self._lock:
            self._buffer.append(row)
            should_flush = len(self._buffer) >= self._batch_size
        if should_flush:
            await self.flush()

    async def add_many(self, rows: Iterable[dict[str, Any]]) -> None:
        for r in rows:
            await self.add(r)

    async def flush(self) -> int:
        async with self._lock:
            if not self._buffer:
                return 0
            batch = self._buffer
            self._buffer = []
            self._last_flush_ts = datetime.utcnow()

        if self.conflict_keys:
            batch = self._dedupe_by(batch, self.conflict_keys)

        factory = get_session_factory()
        async with factory() as session:
            await self._do_insert(session, batch)
        return len(batch)

    async def _do_insert(self, session: AsyncSession, batch: list[dict]) -> None:
        if not batch:
            return
        stmt = insert(self.model).values(batch)
        if self.conflict_keys:
            update_cols = {
                c.name: stmt.excluded[c.name]
                for c in self.model.__table__.columns
                if c.name not in self.conflict_keys
            }
            if self._on_conflict == "update" and update_cols:
                stmt = stmt.on_conflict_do_update(
                    index_elements=self.conflict_keys, set_=update_cols
                )
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=self.conflict_keys)
        try:
            await session.execute(stmt)
            await session.commit()
        except Exception:  # noqa: BLE001
            await session.rollback()
            logger.exception(
                "bulk_writer_flush_failed",
                table=self.model.__tablename__,
                rows=len(batch),
            )

    @staticmethod
    def _dedupe_by(batch: list[dict], keys: Sequence[str]) -> list[dict]:
        deduped: dict[tuple, dict] = {}
        for row in batch:
            k = tuple(row.get(field) for field in keys)
            existing = deduped.get(k)
            if existing is None:
                deduped[k] = row
                continue
            for field, value in row.items():
                if value is not None:
                    existing[field] = value
        return list(deduped.values())

    async def periodic_flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self._flush_interval_s)
            try:
                await self.flush()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "bulk_writer_periodic_flush_error",
                    table=self.model.__tablename__,
                )


# ── Module-level singletons (lazy) ─────────────────────────────────────────

_futures_writer: BulkUpsertWriter | None = None
_options_trade_writer: BulkUpsertWriter | None = None
_flow_event_writer: BulkUpsertWriter | None = None
_liquidity_writer: BulkUpsertWriter | None = None


def get_futures_tick_writer() -> BulkUpsertWriter:
    global _futures_writer
    if _futures_writer is None:
        _futures_writer = BulkUpsertWriter(
            FuturesTick,
            conflict_keys=("ts", "symbol", "seq"),
        )
    return _futures_writer


def get_options_trade_writer() -> BulkUpsertWriter:
    global _options_trade_writer
    if _options_trade_writer is None:
        _options_trade_writer = BulkUpsertWriter(
            OptionsTrade,
            conflict_keys=("ts", "symbol", "expiration", "strike", "option_type", "seq"),
        )
    return _options_trade_writer


def get_flow_event_writer() -> BulkUpsertWriter:
    global _flow_event_writer
    if _flow_event_writer is None:
        _flow_event_writer = BulkUpsertWriter(
            FlowEvent,
            conflict_keys=None,  # event rows are append-only
        )
    return _flow_event_writer


def get_liquidity_snapshot_writer() -> BulkUpsertWriter:
    global _liquidity_writer
    if _liquidity_writer is None:
        _liquidity_writer = BulkUpsertWriter(
            LiquiditySnapshot,
            conflict_keys=("ts", "symbol"),
        )
    return _liquidity_writer
