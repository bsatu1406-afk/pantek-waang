"""Rev 3 — Agent 8 API hardening tests.

Covers:

* GEX ``mode=oi`` vs ``mode=volume`` wiring (correct metric type read).
* Strict ``mode`` enum validation on /v1/{symbol}/gex.
* Max-pain ``expiry`` filter: ``nearest``, ``all``, ``YYYY-MM-DD``, malformed.
* Typed response envelopes (``data`` matches the typed payload shape).
* Admin ``/admin/system/status`` shape (Rev 3 fields) + auth requirement.
* Admin ``/admin/inspector/dlq`` pagination + auth + validation.

DB-backed tests use the ``app_client`` + ``db_session`` fixtures defined in
``conftest.py`` (Postgres testcontainer or ``TEST_DATABASE_URL``).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

import pytest

pytestmark = pytest.mark.asyncio


# ── helpers ────────────────────────────────────────────────────────────────


async def _login(app_client) -> str:
    resp = await app_client.post(
        "/admin/login", json={"username": "admin", "password": "test-password"}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


async def _make_api_key(db_session, *, symbols: list[str]) -> str:
    from app.core.security import display_prefix, generate_api_key, hash_api_key
    from app.db.models import ApiKey

    plaintext = generate_api_key()
    record = ApiKey(
        key_hash=hash_api_key(plaintext),
        key_prefix=display_prefix(plaintext),
        label="hardening-test",
        allowed_symbols=symbols,
        is_active=True,
        usage_count=0,
    )
    db_session.add(record)
    await db_session.commit()
    await db_session.refresh(record)
    return plaintext


async def _insert_gex_metric(
    db_session,
    *,
    symbol: str,
    metric_type: str,
    net_total: float,
    ts: datetime | None = None,
) -> None:
    from app.db.models import ComputedMetric

    ts = ts or datetime.now(UTC)
    row = ComputedMetric(
        ts=ts,
        symbol=symbol,
        metric_type=metric_type,
        strike=0,
        expiration=date(2099, 1, 1),
        computed_at=ts,
        value=net_total,
        extra_json={
            "curve": [{"strike": 100.0, "value": net_total}],
            "top_positive": [{"strike": 110.0, "value": net_total}],
            "top_negative": [{"strike": 90.0, "value": -net_total}],
        },
    )
    db_session.add(row)
    await db_session.commit()


async def _insert_max_pain_rows(
    db_session,
    *,
    symbol: str,
    expirations: list[date],
) -> None:
    from app.db.models import ComputedMetric

    ts = datetime.now(UTC)
    for exp in expirations:
        db_session.add(
            ComputedMetric(
                ts=ts,
                symbol=symbol,
                metric_type="MAX_PAIN",
                strike=4200.0,
                expiration=exp,
                computed_at=ts,
                value=1000.0,
                extra_json={},
            )
        )
    await db_session.commit()


# ── 1. GEX mode wiring ──────────────────────────────────────────────────────


async def test_gex_mode_oi_reads_gex_net_total(app_client, db_session):
    api_key = await _make_api_key(db_session, symbols=["SPXW"])
    await _insert_gex_metric(
        db_session, symbol="SPXW", metric_type="GEX_NET_TOTAL", net_total=1234.5
    )
    await _insert_gex_metric(
        db_session, symbol="SPXW", metric_type="GEX_NET_TOTAL_VOL", net_total=9999.0
    )

    resp = await app_client.get(
        "/v1/SPXW/gex?mode=oi", headers={"X-API-Key": api_key}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["symbol"] == "SPXW"
    assert body["data"]["net_total"] == 1234.5


async def test_gex_mode_volume_reads_gex_net_total_vol(app_client, db_session):
    api_key = await _make_api_key(db_session, symbols=["NDXP"])
    await _insert_gex_metric(
        db_session, symbol="NDXP", metric_type="GEX_NET_TOTAL", net_total=10.0
    )
    await _insert_gex_metric(
        db_session, symbol="NDXP", metric_type="GEX_NET_TOTAL_VOL", net_total=77.7
    )

    resp = await app_client.get(
        "/v1/NDXP/gex?mode=volume", headers={"X-API-Key": api_key}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["data"]["net_total"] == 77.7
    # Response envelope honours the typed schema (curve / top_positive /
    # top_negative present even when no extra data).
    for key in ("curve", "top_positive", "top_negative"):
        assert key in body["data"]


async def test_gex_default_mode_is_oi(app_client, db_session):
    api_key = await _make_api_key(db_session, symbols=["SPXW"])
    await _insert_gex_metric(
        db_session, symbol="SPXW", metric_type="GEX_NET_TOTAL", net_total=42.0
    )
    await _insert_gex_metric(
        db_session, symbol="SPXW", metric_type="GEX_NET_TOTAL_VOL", net_total=999.0
    )

    resp = await app_client.get("/v1/SPXW/gex", headers={"X-API-Key": api_key})
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"]["net_total"] == 42.0


async def test_gex_garbage_mode_rejected_422(app_client, db_session):
    api_key = await _make_api_key(db_session, symbols=["SPXW"])
    resp = await app_client.get(
        "/v1/SPXW/gex?mode=garbage", headers={"X-API-Key": api_key}
    )
    assert resp.status_code == 422, resp.text


# ── 2. Max-pain expiry filter ───────────────────────────────────────────────


async def test_max_pain_nearest_returns_only_first_expiry(app_client, db_session):
    api_key = await _make_api_key(db_session, symbols=["SPXW"])
    today = date.today()
    await _insert_max_pain_rows(
        db_session,
        symbol="SPXW",
        expirations=[today, today + timedelta(days=7), today + timedelta(days=30)],
    )

    resp = await app_client.get(
        "/v1/SPXW/max-pain?expiry=nearest", headers={"X-API-Key": api_key}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["data"]["per_expiry"]) == 1
    assert body["data"]["per_expiry"][0]["expiration"] == today.isoformat()


async def test_max_pain_all_returns_every_expiry(app_client, db_session):
    api_key = await _make_api_key(db_session, symbols=["NDXP"])
    today = date.today()
    await _insert_max_pain_rows(
        db_session,
        symbol="NDXP",
        expirations=[today, today + timedelta(days=14), today + timedelta(days=45)],
    )

    resp = await app_client.get(
        "/v1/NDXP/max-pain?expiry=all", headers={"X-API-Key": api_key}
    )
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["data"]["per_expiry"]) == 3


async def test_max_pain_specific_iso_date_filters_correctly(app_client, db_session):
    api_key = await _make_api_key(db_session, symbols=["SPXW"])
    today = date.today()
    target = today + timedelta(days=21)
    await _insert_max_pain_rows(
        db_session,
        symbol="SPXW",
        expirations=[today, target, today + timedelta(days=60)],
    )

    resp = await app_client.get(
        f"/v1/SPXW/max-pain?expiry={target.isoformat()}",
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 200, resp.text
    rows = resp.json()["data"]["per_expiry"]
    assert len(rows) == 1
    assert rows[0]["expiration"] == target.isoformat()


async def test_max_pain_malformed_date_returns_400(app_client, db_session):
    api_key = await _make_api_key(db_session, symbols=["SPXW"])
    resp = await app_client.get(
        "/v1/SPXW/max-pain?expiry=2026-13-99", headers={"X-API-Key": api_key}
    )
    # Must not be a 500 — should surface a deliberate 400 (or 422 if FastAPI
    # converts it). The contract is "no 500s for malformed dates".
    assert resp.status_code == 400, resp.text
    assert resp.status_code != 500


async def test_max_pain_non_iso_garbage_returns_400(app_client, db_session):
    api_key = await _make_api_key(db_session, symbols=["SPXW"])
    resp = await app_client.get(
        "/v1/SPXW/max-pain?expiry=tomorrow", headers={"X-API-Key": api_key}
    )
    assert resp.status_code == 400, resp.text


# ── 3. Typed response envelopes ─────────────────────────────────────────────


async def test_walls_response_shape_is_typed(app_client, db_session):
    """``/v1/{symbol}/walls`` returns the typed ``WallsResponse`` shape."""
    api_key = await _make_api_key(db_session, symbols=["SPXW"])
    resp = await app_client.get(
        "/v1/SPXW/walls", headers={"X-API-Key": api_key}
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    # All four wall buckets are present (default to empty list).
    for key in ("call_wall_oi", "put_wall_oi", "call_wall_volume", "put_wall_volume"):
        assert key in data
        assert isinstance(data[key], list)


async def test_iv_response_shape_is_typed(app_client, db_session):
    """``/v1/{symbol}/iv`` returns the typed ``IvResponse`` shape."""
    api_key = await _make_api_key(db_session, symbols=["NDXP"])
    resp = await app_client.get(
        "/v1/NDXP/iv", headers={"X-API-Key": api_key}
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    # Typed shape: atm_iv (nullable float), skew (dict), surface (list).
    assert "atm_iv" in data
    assert "skew" in data
    assert "surface" in data
    assert isinstance(data["skew"], dict)
    assert isinstance(data["surface"], list)


# ── 4. Admin /system/status ─────────────────────────────────────────────────


async def test_system_status_requires_jwt(app_client):
    resp = await app_client.get("/admin/system/status")
    assert resp.status_code in (401, 403)


async def test_system_status_returns_rev3_fields(app_client):
    token = await _login(app_client)
    resp = await app_client.get(
        "/admin/system/status", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Pre-existing fields stay.
    for key in ("rows_per_symbol", "metric_rows_per_symbol", "active_api_keys"):
        assert key in body
    # New Rev 3 telemetry fields are present.
    for key in (
        "futures_lag_ms",
        "opra_lag_ms",
        "dlq_pending",
        "flow_events_last_hour",
        "last_pipeline_runs",
        "live_ingester",
    ):
        assert key in body, f"missing telemetry key: {key}"
    assert isinstance(body["dlq_pending"], int)
    assert isinstance(body["flow_events_last_hour"], int)
    assert isinstance(body["last_pipeline_runs"], list)
    assert isinstance(body["live_ingester"], dict)


# ── 5. Admin /inspector/dlq ─────────────────────────────────────────────────


async def _insert_dlq(db_session, n: int) -> None:
    from app.db.models import DeadLetterEntry

    base = datetime.now(UTC)
    for i in range(n):
        db_session.add(
            DeadLetterEntry(
                id=uuid4(),
                ts=base - timedelta(seconds=i),
                source="opra_live",
                reason=f"reason-{i}",
                payload={"i": i},
            )
        )
    await db_session.commit()


async def test_dlq_requires_jwt(app_client):
    resp = await app_client.get("/admin/inspector/dlq")
    assert resp.status_code in (401, 403)


async def test_dlq_pagination_honours_limit_offset(app_client, db_session):
    await _insert_dlq(db_session, n=12)
    token = await _login(app_client)
    headers = {"Authorization": f"Bearer {token}"}

    resp = await app_client.get(
        "/admin/inspector/dlq?limit=5&offset=0", headers=headers
    )
    assert resp.status_code == 200, resp.text
    page = resp.json()
    assert page["limit"] == 5
    assert page["offset"] == 0
    assert page["total"] >= 12
    assert len(page["items"]) == 5
    # Newest-first ordering — first item's ts is >= last item's ts.
    first_ts = page["items"][0]["ts"]
    last_ts = page["items"][-1]["ts"]
    assert first_ts >= last_ts

    # Second page should not overlap with the first.
    resp2 = await app_client.get(
        "/admin/inspector/dlq?limit=5&offset=5", headers=headers
    )
    assert resp2.status_code == 200
    page2 = resp2.json()
    ids_a = {item["id"] for item in page["items"]}
    ids_b = {item["id"] for item in page2["items"]}
    assert ids_a.isdisjoint(ids_b)


async def test_dlq_rejects_out_of_bound_limit(app_client):
    token = await _login(app_client)
    headers = {"Authorization": f"Bearer {token}"}

    # limit=0 rejected (gt=0)
    resp = await app_client.get("/admin/inspector/dlq?limit=0", headers=headers)
    assert resp.status_code == 422

    # limit < 0 rejected
    resp = await app_client.get("/admin/inspector/dlq?limit=-5", headers=headers)
    assert resp.status_code == 422

    # limit > 500 rejected (le=500)
    resp = await app_client.get("/admin/inspector/dlq?limit=501", headers=headers)
    assert resp.status_code == 422

    # Negative offset rejected (ge=0)
    resp = await app_client.get(
        "/admin/inspector/dlq?offset=-1", headers=headers
    )
    assert resp.status_code == 422


async def test_dlq_default_limit_is_50(app_client, db_session):
    await _insert_dlq(db_session, n=3)
    token = await _login(app_client)
    headers = {"Authorization": f"Bearer {token}"}

    resp = await app_client.get("/admin/inspector/dlq", headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["limit"] == 50
