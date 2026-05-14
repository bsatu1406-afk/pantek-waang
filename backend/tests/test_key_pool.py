"""Unit tests for the Databento key-pool resolution + failover.

We mock the SQLAlchemy session because the logic under test is pure
selection / ordering / record-update reasoning; the DB-level integration
is exercised by the ``test_api_admin.py`` Postgres-backed tests.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.crypto import encrypt_secret
from app.ingestion import key_pool as kp_mod
from app.ingestion.key_pool import (
    MAX_ERRORS_BEFORE_SKIP,
    SKIP_COOLDOWN,
    KeyCandidate,
    iter_keys,
    record_key_error,
    record_key_success,
)


def _row(
    *,
    id_: int,
    label: str,
    dataset: str,
    plaintext: str,
    priority: int = 100,
    is_active: bool = True,
    error_count: int = 0,
    last_error_at: datetime | None = None,
):
    return SimpleNamespace(
        id=id_,
        label=label,
        dataset=dataset,
        api_key_encrypted=encrypt_secret(plaintext),
        priority=priority,
        is_active=is_active,
        error_count=error_count,
        last_error_at=last_error_at,
    )


def _fake_session_returning(rows: list) -> MagicMock:
    """Return a mock AsyncSession whose execute().scalars().all() == rows."""
    session = MagicMock()
    scalar_obj = MagicMock()
    scalar_obj.all.return_value = rows
    result_obj = MagicMock()
    result_obj.scalars.return_value = scalar_obj
    session.execute = AsyncMock(return_value=result_obj)
    return session


@pytest.mark.asyncio
async def test_env_key_listed_first(monkeypatch):
    """Env-configured key must appear at position 0 regardless of DB priority."""
    monkeypatch.setattr(
        kp_mod, "_env_key_for", lambda ds: "env-secret" if ds == "OPRA.PILLAR" else None
    )
    session = _fake_session_returning(
        [_row(id_=1, label="db-low", dataset="OPRA.PILLAR", plaintext="db-secret", priority=10)]
    )
    candidates = await iter_keys(session, "OPRA.PILLAR")
    assert len(candidates) == 2
    assert candidates[0].source == "env"
    assert candidates[0].api_key == "env-secret"
    assert candidates[1].source == "db"
    assert candidates[1].api_key == "db-secret"


@pytest.mark.asyncio
async def test_no_env_uses_db_only(monkeypatch):
    monkeypatch.setattr(kp_mod, "_env_key_for", lambda ds: None)
    rows = [
        _row(id_=1, label="primary", dataset="OPRA.PILLAR", plaintext="aaa", priority=10),
        _row(id_=2, label="backup", dataset="OPRA.PILLAR", plaintext="bbb", priority=20),
    ]
    session = _fake_session_returning(rows)
    candidates = await iter_keys(session, "OPRA.PILLAR")
    assert [c.api_key for c in candidates] == ["aaa", "bbb"]
    assert all(c.source == "db" for c in candidates)


@pytest.mark.asyncio
async def test_dataset_both_eligible_for_either_request(monkeypatch):
    """The ``BOTH`` dataset key must be returned for OPRA and for GLBX both."""
    monkeypatch.setattr(kp_mod, "_env_key_for", lambda ds: None)
    rows = [_row(id_=1, label="multi", dataset="BOTH", plaintext="multi-key", priority=50)]
    # The mock returns the same row irrespective of which dataset is queried.
    session = _fake_session_returning(rows)
    for ds in ("OPRA.PILLAR", "GLBX.MDP3"):
        candidates = await iter_keys(session, ds)
        assert len(candidates) == 1
        assert candidates[0].api_key == "multi-key"


@pytest.mark.asyncio
async def test_skip_when_error_count_high_within_cooldown(monkeypatch):
    monkeypatch.setattr(kp_mod, "_env_key_for", lambda ds: None)
    fresh_err = datetime.now(UTC) - timedelta(minutes=1)
    rows = [
        _row(
            id_=1,
            label="bad",
            dataset="OPRA.PILLAR",
            plaintext="x",
            error_count=MAX_ERRORS_BEFORE_SKIP,
            last_error_at=fresh_err,
        ),
        _row(id_=2, label="good", dataset="OPRA.PILLAR", plaintext="y"),
    ]
    session = _fake_session_returning(rows)
    candidates = await iter_keys(session, "OPRA.PILLAR")
    labels = [c.label for c in candidates]
    assert labels == ["good"]


@pytest.mark.asyncio
async def test_resume_after_cooldown(monkeypatch):
    """Once the cooldown expires, the previously-broken key reappears."""
    monkeypatch.setattr(kp_mod, "_env_key_for", lambda ds: None)
    old_err = datetime.now(UTC) - SKIP_COOLDOWN - timedelta(minutes=1)
    rows = [
        _row(
            id_=1,
            label="recovered",
            dataset="OPRA.PILLAR",
            plaintext="x",
            error_count=MAX_ERRORS_BEFORE_SKIP + 2,
            last_error_at=old_err,
        ),
    ]
    session = _fake_session_returning(rows)
    candidates = await iter_keys(session, "OPRA.PILLAR")
    assert [c.label for c in candidates] == ["recovered"]


@pytest.mark.asyncio
async def test_decrypt_failure_is_skipped(monkeypatch):
    """A row with a corrupt ciphertext should be skipped, not raise."""
    monkeypatch.setattr(kp_mod, "_env_key_for", lambda ds: None)
    bad = SimpleNamespace(
        id=1,
        label="rotten",
        dataset="OPRA.PILLAR",
        api_key_encrypted="!!!not-a-valid-fernet-token!!!",
        priority=10,
        is_active=True,
        error_count=0,
        last_error_at=None,
    )
    good = _row(id_=2, label="good", dataset="OPRA.PILLAR", plaintext="ok")
    session = _fake_session_returning([bad, good])
    candidates = await iter_keys(session, "OPRA.PILLAR")
    assert [c.label for c in candidates] == ["good"]


@pytest.mark.asyncio
async def test_record_key_success_noop_for_env_candidate():
    """env candidates have ``db_id is None`` — record_key_success skips."""
    session = MagicMock()
    session.execute = AsyncMock()
    await record_key_success(
        session, KeyCandidate(label="env:X", api_key="x", source="env")
    )
    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_key_error_noop_for_env_candidate():
    session = MagicMock()
    session.execute = AsyncMock()
    await record_key_error(
        session,
        KeyCandidate(label="env:X", api_key="x", source="env"),
        error_msg="boom",
    )
    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_iter_keys_rejects_unknown_dataset():
    session = MagicMock()
    with pytest.raises(ValueError):
        await iter_keys(session, "NASDAQ.OPRA")
