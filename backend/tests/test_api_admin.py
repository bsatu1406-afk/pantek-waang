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
