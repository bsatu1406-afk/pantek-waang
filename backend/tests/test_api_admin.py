"""End-to-end tests for admin endpoints (Postgres required)."""

from __future__ import annotations


async def _login(app_client) -> str:
    resp = await app_client.post(
        "/admin/login", json={"username": "admin", "password": "test-password"}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


async def test_admin_login_success(app_client):
    token = await _login(app_client)
    assert token


async def test_admin_login_wrong_password(app_client):
    resp = await app_client.post(
        "/admin/login", json={"username": "admin", "password": "wrong"}
    )
    assert resp.status_code == 401


async def test_admin_endpoints_require_jwt(app_client):
    resp = await app_client.get("/admin/api-keys")
    assert resp.status_code in (401, 403)


async def test_create_list_update_delete_api_key(app_client):
    token = await _login(app_client)
    headers = {"Authorization": f"Bearer {token}"}

    # CREATE
    resp = await app_client.post(
        "/admin/api-keys",
        json={"label": "alpha", "allowed_symbols": ["spxw"]},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    plaintext = body["plaintext_key"]
    key_id = body["key"]["id"]
    assert plaintext.startswith("ak_")
    assert body["key"]["allowed_symbols"] == ["SPXW"]
    assert body["key"]["is_active"] is True

    # LIST
    resp = await app_client.get("/admin/api-keys", headers=headers)
    assert resp.status_code == 200
    keys = resp.json()
    assert any(k["id"] == key_id for k in keys)

    # UPDATE: deactivate + relabel
    resp = await app_client.patch(
        f"/admin/api-keys/{key_id}",
        json={"label": "alpha-v2", "is_active": False},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["label"] == "alpha-v2"
    assert resp.json()["is_active"] is False

    # USAGE endpoint
    resp = await app_client.get(f"/admin/api-keys/{key_id}/usage", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["usage_count"] == 0

    # DELETE
    resp = await app_client.delete(f"/admin/api-keys/{key_id}", headers=headers)
    assert resp.status_code == 204

    resp = await app_client.get(f"/admin/api-keys/{key_id}/usage", headers=headers)
    assert resp.status_code == 404


async def test_admin_system_status(app_client):
    token = await _login(app_client)
    headers = {"Authorization": f"Bearer {token}"}
    resp = await app_client.get("/admin/system/status", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "rows_per_symbol" in body
    assert "active_api_keys" in body
    assert "last_compute_per_symbol" in body


# ── Databento key pool (Rev 4) ──────────────────────────────────────────────


async def test_databento_key_pool_crud(app_client):
    token = await _login(app_client)
    headers = {"Authorization": f"Bearer {token}"}

    # LIST starts empty.
    resp = await app_client.get("/admin/databento-keys", headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json() == []

    # CREATE — OPRA.PILLAR primary.
    resp = await app_client.post(
        "/admin/databento-keys",
        json={
            "label": "Primary OPRA",
            "dataset": "opra.pillar",  # lowercase ok — normalizes
            "api_key": "db-superSecret-12345",
            "priority": 1,
        },
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["dataset"] == "OPRA.PILLAR"
    assert body["api_key_prefix"].startswith("db-")
    assert "superSecret" not in body["api_key_prefix"]
    key_id = body["id"]

    # CREATE — BOTH fallback.
    resp = await app_client.post(
        "/admin/databento-keys",
        json={
            "label": "Fallback Both",
            "dataset": "both",
            "api_key": "db-otherSecret-67890",
            "priority": 200,
        },
        headers=headers,
    )
    assert resp.status_code == 201

    # CREATE — rejected dataset
    resp = await app_client.post(
        "/admin/databento-keys",
        json={
            "label": "Bad",
            "dataset": "WHATEVER",
            "api_key": "db-x",
            "priority": 1,
        },
        headers=headers,
    )
    assert resp.status_code == 422

    # LIST returns both, ordered.
    resp = await app_client.get("/admin/databento-keys", headers=headers)
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2

    # PATCH priority
    resp = await app_client.patch(
        f"/admin/databento-keys/{key_id}",
        json={"priority": 999, "is_active": False},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["priority"] == 999
    assert resp.json()["is_active"] is False

    # TEST endpoint (decryption sanity check)
    resp = await app_client.post(
        f"/admin/databento-keys/{key_id}/test", headers=headers
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    # DELETE
    resp = await app_client.delete(
        f"/admin/databento-keys/{key_id}", headers=headers
    )
    assert resp.status_code == 204

    resp = await app_client.post(
        f"/admin/databento-keys/{key_id}/test", headers=headers
    )
    assert resp.status_code == 404


async def test_databento_key_requires_jwt(app_client):
    resp = await app_client.get("/admin/databento-keys")
    assert resp.status_code in (401, 403)
