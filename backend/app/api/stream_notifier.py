"""In-process pub/sub used by the streaming API (Agent 5).

The processing pipeline calls :func:`publish` at the end of every successful
chain-pipeline tick. WebSocket and SSE subscribers receive each published
payload via their own ``asyncio.Queue`` so a slow subscriber never blocks the
pipeline or other subscribers.

This module is deliberately dependency-free (only ``asyncio``) and contains no
HTTP / FastAPI logic — those concerns live in ``endpoints/stream.py``.

Backpressure policy: each subscriber queue is bounded. When a subscriber falls
behind we **drop the oldest** queued payload to make room for the newest — for
real-time market data, freshness beats completeness.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


# Per-subscriber queue depth. Sized to absorb a small burst (~30 s of 1 Hz
# updates) while still applying backpressure on truly stuck consumers.
DEFAULT_QUEUE_MAXSIZE: int = 32


class StreamNotifier:
    """In-process fan-out broker keyed by ``symbol``.

    Methods are coroutine-safe under a single event loop; instances are *not*
    safe to share across loops. The module-level singleton returned by
    :func:`get_stream_notifier` is bound to whichever loop owns it at first
    use.
    """

    def __init__(self, queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE) -> None:
        self._queue_maxsize = queue_maxsize
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)
        self._lock = asyncio.Lock()

    def subscribe(self, symbol: str) -> asyncio.Queue[dict[str, Any]]:
        """Register a new subscriber for ``symbol`` and return its queue.

        The returned queue is bounded; callers must consume promptly or
        accept ``publish`` dropping the oldest frame on overflow.
        """
        sym = symbol.upper()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._queue_maxsize)
        self._subscribers[sym].add(queue)
        logger.debug("stream_subscribe", symbol=sym, total=len(self._subscribers[sym]))
        return queue

    def unsubscribe(self, symbol: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Detach a subscriber. Safe to call multiple times for the same queue."""
        sym = symbol.upper()
        bucket = self._subscribers.get(sym)
        if bucket is None:
            return
        bucket.discard(queue)
        if not bucket:
            self._subscribers.pop(sym, None)
        logger.debug("stream_unsubscribe", symbol=sym, remaining=len(bucket))

    def subscriber_count(self, symbol: str | None = None) -> int:
        """Return the number of subscribers for ``symbol`` (or total)."""
        if symbol is None:
            return sum(len(s) for s in self._subscribers.values())
        return len(self._subscribers.get(symbol.upper(), set()))

    async def publish(self, symbol: str, payload: dict[str, Any]) -> int:
        """Broadcast ``payload`` to every subscriber of ``symbol``.

        Returns the number of subscribers that successfully received the
        payload. Slow subscribers have their oldest queued frame discarded so
        the freshest one can land — we never block the publisher.
        """
        sym = symbol.upper()
        bucket = self._subscribers.get(sym)
        if not bucket:
            return 0

        delivered = 0
        # Snapshot the set so concurrent unsubscribes during iteration are safe.
        for queue in list(bucket):
            try:
                queue.put_nowait(payload)
                delivered += 1
            except asyncio.QueueFull:
                # Drop oldest, then enqueue the latest. Best-effort: another
                # consumer may have drained between the get and the put.
                try:
                    _ = queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(payload)
                    delivered += 1
                except asyncio.QueueFull:
                    logger.warning(
                        "stream_publish_drop",
                        symbol=sym,
                        queue_maxsize=self._queue_maxsize,
                    )
        return delivered


_singleton: StreamNotifier | None = None


def get_stream_notifier() -> StreamNotifier:
    """Return the process-wide :class:`StreamNotifier` (lazy)."""
    global _singleton
    if _singleton is None:
        _singleton = StreamNotifier()
    return _singleton


def reset_stream_notifier_for_tests() -> None:
    """Drop the cached singleton so tests can start with a clean broker."""
    global _singleton
    _singleton = None
