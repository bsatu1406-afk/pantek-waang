"""Resolution and failover for Databento API key candidates.

The Rev 4 admin UI lets operators register multiple Databento API keys
(``databento_api_keys`` table). The ingester used to pick one from
``settings.opra_api_key`` / ``settings.globex_api_key`` and call it a
day; if Databento rejected the key (rate limit, expired sub, network
blip) the live stream would just stop.

This module wraps that with a small failover policy:

1. **Env first.** ``DATABENTO_API_KEY_OPRA`` / ``DATABENTO_API_KEY_GLOBEX``
   (or legacy ``DATABENTO_API_KEY``) are always tried first if non-empty.
2. **DB pool next.** Active rows for the requested dataset, sorted by
   ``priority`` ASC (lower is higher priority). ``BOTH`` rows are also
   eligible for either dataset.
3. **On error**, the caller is expected to call :func:`record_key_error`
   so the row's ``error_count`` and ``last_error_at`` are updated for
   the admin telemetry. The next call to :func:`iter_keys` will skip
   keys whose ``error_count`` has run away (>= ``MAX_ERRORS_BEFORE_SKIP``)
   *until* the cooldown expires.
4. **On success**, :func:`record_key_success` resets the counters so a
   transient failure doesn't permanently demote a healthy key.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.crypto import InvalidToken, decrypt_secret
from app.core.logging import get_logger
from app.db.models import DatabentoApiKey

logger = get_logger(__name__)


# When a key has logged this many errors in a row we skip it for the
# cooldown duration. Both values are intentionally small: the admin UI
# is the right place for a permanent demotion, this is just so a bad
# key doesn't stall the failover loop.
MAX_ERRORS_BEFORE_SKIP = 5
SKIP_COOLDOWN = timedelta(minutes=30)


@dataclass
class KeyCandidate:
    """One API key the ingester may try, in priority order."""

    label: str
    api_key: str
    source: str
    """``env`` | ``db``."""
    db_id: int | None = None
    """Row ID in ``databento_api_keys`` when ``source == 'db'`` — used so
    success/error feedback can update the right row."""


# ──────────────────────────────────────────────────────────────────────────


def _env_key_for(dataset: str) -> str | None:
    """Return the env-configured key for ``dataset`` or None."""
    settings = get_settings()
    dataset = dataset.upper()
    if dataset == "OPRA.PILLAR":
        return settings.opra_api_key or None
    if dataset == "GLBX.MDP3":
        return settings.globex_api_key or None
    return None


async def iter_keys(session: AsyncSession, dataset: str) -> list[KeyCandidate]:
    """Resolve every candidate key for ``dataset`` in priority order.

    Pure read — no side effects on the DB. Callers iterate the list
    until one connects successfully or the list is exhausted.
    """
    dataset = dataset.upper()
    if dataset not in {"OPRA.PILLAR", "GLBX.MDP3"}:
        raise ValueError(f"Unknown dataset: {dataset}")

    candidates: list[KeyCandidate] = []
    env_key = _env_key_for(dataset)
    if env_key:
        candidates.append(
            KeyCandidate(label=f"env:{dataset}", api_key=env_key, source="env")
        )

    now = datetime.now(UTC)
    stmt = (
        select(DatabentoApiKey)
        .where(
            DatabentoApiKey.is_active.is_(True),
            DatabentoApiKey.dataset.in_({dataset, "BOTH"}),
        )
        .order_by(DatabentoApiKey.priority.asc(), DatabentoApiKey.id.asc())
    )
    rows = (await session.execute(stmt)).scalars().all()
    for row in rows:
        # Cool-off long-broken keys without removing them from the pool.
        if row.error_count >= MAX_ERRORS_BEFORE_SKIP and row.last_error_at:
            if now - row.last_error_at < SKIP_COOLDOWN:
                logger.debug(
                    "key_pool.skip_cooldown",
                    label=row.label,
                    error_count=row.error_count,
                )
                continue
        try:
            plaintext = decrypt_secret(row.api_key_encrypted)
        except InvalidToken:
            logger.error(
                "key_pool.decrypt_failed",
                label=row.label,
                key_id=row.id,
            )
            continue
        candidates.append(
            KeyCandidate(
                label=row.label,
                api_key=plaintext,
                source="db",
                db_id=row.id,
            )
        )
    return candidates


async def record_key_success(session: AsyncSession, candidate: KeyCandidate) -> None:
    """Mark ``candidate`` as recently used and reset its error counter."""
    if candidate.source != "db" or candidate.db_id is None:
        return
    await session.execute(
        update(DatabentoApiKey)
        .where(DatabentoApiKey.id == candidate.db_id)
        .values(
            last_used_at=datetime.now(UTC),
            error_count=0,
            last_error_msg=None,
        )
    )
    await session.commit()


async def record_key_error(
    session: AsyncSession,
    candidate: KeyCandidate,
    *,
    error_msg: str,
) -> None:
    """Increment the error counter and store the most recent error message."""
    if candidate.source != "db" or candidate.db_id is None:
        return
    await session.execute(
        update(DatabentoApiKey)
        .where(DatabentoApiKey.id == candidate.db_id)
        .values(
            last_error_at=datetime.now(UTC),
            last_error_msg=error_msg[:1000],
            error_count=DatabentoApiKey.error_count + 1,
        )
    )
    await session.commit()


__all__ = [
    "KeyCandidate",
    "MAX_ERRORS_BEFORE_SKIP",
    "SKIP_COOLDOWN",
    "iter_keys",
    "record_key_error",
    "record_key_success",
]
