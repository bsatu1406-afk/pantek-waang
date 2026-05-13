"""Agent 6 — Dead-letter queue & writer backpressure tests.

DB-touching tests are skipped in ``APP_TESTING=1`` mode (no Postgres in
CI). What we *can* verify without a database:

* ``DeadLetterQueue`` accepts entries up to its cap and silently drops
  oldest beyond it.
* ``OptionsChainWriter`` drops rows to DLQ once ``max_pending_rows`` is
  reached (we monkey-patch the DLQ recorder to count invocations).
* ``BulkUpsertWriter`` mirrors the same backpressure behaviour.
"""

from __future__ import annotations

import asyncio

import pytest

from app.ingestion import dlq as dlq_mod
from app.ingestion.bulk_writers import BulkUpsertWriter
from app.ingestion.writer import OptionsChainWriter


@pytest.mark.asyncio
async def test_dlq_buffer_caps_in_memory() -> None:
    """DLQ ring buffer should drop oldest entries beyond ``max_size``."""
    queue = dlq_mod.DeadLetterQueue(max_size=3)
    for i in range(5):
        await queue.add(source="opra_live", reason=f"r{i}", payload={"i": i})
    assert queue.pending == 3


@pytest.mark.asyncio
async def test_chain_writer_sheds_to_dlq_when_full(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict] = []

    async def fake_record(*, source: str, reason: str, payload: dict | None = None) -> None:
        captured.append({"source": source, "reason": reason, "payload": payload})

    monkeypatch.setattr("app.ingestion.writer.record_dlq", fake_record)

    # max_pending_rows = 2, batch_size much higher so we don't auto-flush.
    writer = OptionsChainWriter(
        batch_size=10_000, flush_interval_s=999.0, max_pending_rows=2
    )
    base = {
        "ts": None,
        "symbol": "SPXW",
        "expiration": None,
        "strike": 4500.0,
        "option_type": "C",
        "iv": 0.2,
    }
    for i in range(5):
        await writer.add({**base, "iv": 0.2 + i / 100})

    assert writer.pending == 2
    assert writer.shed_rows == 3
    assert len(captured) == 3
    assert all(e["reason"] == "backpressure_overflow" for e in captured)


@pytest.mark.asyncio
async def test_bulk_writer_sheds_to_dlq_when_full(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same backpressure semantics on the generic bulk writer."""
    captured: list[dict] = []

    async def fake_record(*, source: str, reason: str, payload: dict | None = None) -> None:
        captured.append({"source": source, "reason": reason})

    monkeypatch.setattr("app.ingestion.bulk_writers.record_dlq", fake_record)

    # Lightweight ORM stand-in: BulkUpsertWriter only reads ``__tablename__``
    # and ``__table__.columns`` on the flush path which we never trigger here.
    class _FakeModel:
        __tablename__ = "fake"

    writer = BulkUpsertWriter(
        _FakeModel,
        conflict_keys=("ts", "symbol"),
        batch_size=10_000,
        max_pending_rows=2,
        dlq_source="test",
    )
    for i in range(4):
        await writer.add({"ts": i, "symbol": "X"})

    assert writer.pending == 2
    assert writer.shed_rows == 2
    assert [e["source"] for e in captured] == ["test", "test"]


def test_dlq_module_singleton_is_stable() -> None:
    """:func:`get_dlq` returns the same instance every time."""
    a = dlq_mod.get_dlq()
    b = dlq_mod.get_dlq()
    assert a is b


@pytest.mark.asyncio
async def test_dlq_flush_swallows_db_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure in the underlying DB write should not surface to the caller
    and should put the entries back into the buffer for the next flush."""

    queue = dlq_mod.DeadLetterQueue(max_size=10)
    await queue.add(source="opra_live", reason="r1")
    await queue.add(source="opra_live", reason="r2")
    assert queue.pending == 2

    class _BoomSession:
        async def __aenter__(self) -> _BoomSession:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def execute(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("db down")

        async def commit(self) -> None:  # pragma: no cover — never reached
            return None

    def fake_factory() -> object:
        return _BoomSession

    # Patch the session factory so the flush hits the BoomSession.
    monkeypatch.setattr(dlq_mod, "get_session_factory", lambda: fake_factory())

    flushed = await queue.flush()
    assert flushed == 0
    assert queue.pending == 2  # entries re-queued
    # Sanity: a second flush attempt also returns 0 and keeps entries.
    flushed_again = await queue.flush()
    assert flushed_again == 0
    assert queue.pending == 2
    # Free the event loop for any pending tasks scheduled by add().
    await asyncio.sleep(0)
