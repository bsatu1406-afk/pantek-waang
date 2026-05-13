"""End-to-end tests for API key auth middleware (Postgres required)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

pytestmark = pytest.mark.asyncio


async def _make_key(
    db_session, *, symbols: list[str], is_active: bool = True, expires_at=None
) -> tuple[str, ApiKey]:  # noqa: F821
    from app.core.security import display_prefix, generate_api_key, hash_api_key
    from app.db.models import ApiKey

    plaintext = generate_api_key()
    record = ApiKey(
        key_hash=hash_api_key(plaintext),
        key_prefix=display_prefix(plaintext),
        label="test-key",
        allowed_symbols=symbols,
        is_active=is_active,
        expires_at=expires_at,
        usage_count=0,
    )
    db_session.add(record)
    await db_session.commit()
    await db_session.refresh(record)
    return plaintext, record


async def test_missing_api_key_returns_401(app_client):
    resp = await app_client.get("/v1/SPXW/snapshot")
    assert resp.status_code == 401


async def test_invalid_api_key_returns_401(app_client):
    resp = await app_client.get(
        "/v1/SPXW/snapshot", headers={"X-API-Key": "ak_invalid_value"}
    )
    assert resp.status_code == 401


async def test_inactive_api_key_returns_403(app_client, db_session):
    plaintext, _ = await _make_key(db_session, symbols=["SPXW"], is_active=False)
    resp = await app_client.get(
        "/v1/SPXW/snapshot", headers={"X-API-Key": plaintext}
    )
    assert resp.status_code == 403


async def test_expired_api_key_returns_403(app_client, db_session):
    expired = datetime.now(UTC) - timedelta(days=1)
    plaintext, _ = await _make_key(db_session, symbols=["SPXW"], expires_at=expired)
    resp = await app_client.get(
        "/v1/SPXW/snapshot", headers={"X-API-Key": plaintext}
    )
    assert resp.status_code == 403


async def test_wrong_symbol_returns_403(app_client, db_session):
    plaintext, _ = await _make_key(db_session, symbols=["NDXP"])
    resp = await app_client.get(
        "/v1/SPXW/snapshot", headers={"X-API-Key": plaintext}
    )
    assert resp.status_code == 403


async def test_valid_api_key_allows_access(app_client, db_session):
    plaintext, record = await _make_key(db_session, symbols=["SPXW"])
    resp = await app_client.get(
        "/v1/SPXW/snapshot", headers={"X-API-Key": plaintext}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "SPXW"
    # Usage stats updated.
    await db_session.refresh(record)
    assert (record.usage_count or 0) >= 1
    assert record.last_used_at is not None


async def test_valid_key_envelope_includes_metadata(app_client, db_session):
    plaintext, _ = await _make_key(db_session, symbols=["SPXW"])
    resp = await app_client.get(
        "/v1/SPXW/snapshot", headers={"X-API-Key": plaintext}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "computed_at" in body
    assert "next_update_in_seconds" in body
    assert "data" in body
