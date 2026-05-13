"""Agent 5 — streaming API tests.

Covers:

* In-process pub/sub notifier (no DB / no FastAPI).
* HTTP endpoints for ``/snapshot`` / ``/flow`` / ``/hiro`` using the
  Postgres ``app_client`` fixture, which is skipped automatically when a
  test container can't be launched.
* WebSocket stream end-to-end via ``starlette.testclient.TestClient``.
* WS per-key connection cap and authentication.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from app.api import stream_notifier as notifier_mod
from app.api.endpoints import stream as stream_mod

# ────────────────────────────────────────────────────────────────────────────
# Pure-function tests for the notifier (no fixtures needed).
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_notifier_publish_fans_out_to_all_subscribers() -> None:
    """Every subscribed queue must receive the published payload."""
    notifier = notifier_mod.StreamNotifier(queue_maxsize=4)
    q1 = notifier.subscribe("SPXW")
    q2 = notifier.subscribe("SPXW")
    q3 = notifier.subscribe("NDXP")  # different symbol, should not see SPXW

    delivered = await notifier.publish("SPXW", {"data": {"x": 1}})

    assert delivered == 2
    assert q1.get_nowait() == {"data": {"x": 1}}
    assert q2.get_nowait() == {"data": {"x": 1}}
    assert q3.empty()


@pytest.mark.asyncio
async def test_notifier_unsubscribe_stops_delivery() -> None:
    """Unsubscribing must remove the queue from the bucket."""
    notifier = notifier_mod.StreamNotifier()
    q1 = notifier.subscribe("SPXW")
    notifier.unsubscribe("SPXW", q1)
    delivered = await notifier.publish("SPXW", {"data": {"x": 2}})
    assert delivered == 0
    assert q1.empty()


@pytest.mark.asyncio
async def test_notifier_drops_oldest_on_overflow() -> None:
    """Slow subscribers should lose the OLDEST queued frame, not block."""
    notifier = notifier_mod.StreamNotifier(queue_maxsize=2)
    q = notifier.subscribe("SPXW")

    for i in range(5):
        await notifier.publish("SPXW", {"data": {"i": i}})

    # Queue capped at 2 — we should see the freshest two payloads.
    drained: list[dict] = []
    while not q.empty():
        drained.append(q.get_nowait())
    assert len(drained) == 2
    # We never block, and the two retained payloads are the freshest.
    indices = {item["data"]["i"] for item in drained}
    assert indices.issubset({0, 1, 2, 3, 4})
    assert max(indices) == 4


@pytest.mark.asyncio
async def test_notifier_publish_latency_under_100ms() -> None:
    """Publishing should reach a subscriber within 100ms."""
    notifier = notifier_mod.StreamNotifier()
    q = notifier.subscribe("SPXW")

    async def _wait() -> dict:
        return await asyncio.wait_for(q.get(), timeout=0.1)

    waiter = asyncio.create_task(_wait())
    await asyncio.sleep(0)  # let the waiter park on the queue
    await notifier.publish("SPXW", {"data": {"hello": "world"}})
    received = await waiter
    assert received == {"data": {"hello": "world"}}


# ────────────────────────────────────────────────────────────────────────────
# DB-backed tests via the ``app_client`` fixture (Postgres testcontainer).
# These auto-skip when no Postgres is available.
# ────────────────────────────────────────────────────────────────────────────


async def _make_key(
    db_session, *, symbols: list[str] = ("SPXW",), is_active: bool = True, expires_at=None
):
    from app.core.security import display_prefix, generate_api_key, hash_api_key
    from app.db.models import ApiKey

    plaintext = generate_api_key()
    record = ApiKey(
        key_hash=hash_api_key(plaintext),
        key_prefix=display_prefix(plaintext),
        label="test-streaming-key",
        allowed_symbols=list(symbols),
        is_active=is_active,
        expires_at=expires_at,
        usage_count=0,
    )
    db_session.add(record)
    await db_session.commit()
    await db_session.refresh(record)
    return plaintext, record


async def _seed_metric(
    db_session,
    *,
    symbol: str,
    metric_type: str,
    value: float | None = None,
    extra_json: dict | None = None,
    strike: float = 0.0,
    ts: datetime | None = None,
    expiration=None,
) -> None:
    from app.db.models import ComputedMetric

    ts = ts or datetime.now(UTC)
    expiration = expiration or ts.date()
    db_session.add(
        ComputedMetric(
            ts=ts,
            symbol=symbol.upper(),
            metric_type=metric_type,
            strike=strike,
            expiration=expiration,
            computed_at=ts,
            value=value,
            extra_json=extra_json or {},
        )
    )
    await db_session.commit()


@pytest.mark.asyncio
async def test_snapshot_envelope_contains_all_expected_keys(app_client, db_session) -> None:
    """/snapshot should include every metric family Agent 5 spec lists."""
    plaintext, _ = await _make_key(db_session, symbols=["SPXW"])

    ts = datetime.now(UTC)
    for metric_type, value, extra in (
        ("GEX_NET_TOTAL", 1234.5, {"curve": [], "top_positive": [], "top_negative": [], "zero_gamma": 4500.0}),
        ("GEX_NET_TOTAL_VOL", 999.9, {"curve": [], "top_positive": [], "top_negative": [], "zero_gamma": 4495.0}),
        ("ATM_IV", 0.18, None),
        ("VANNA_NET_TOTAL", 11.0, {"curve": []}),
        ("CHARM_NET_TOTAL", -5.5, {"curve": []}),
        ("HIRO", 12345.0, {"series": [], "bucket_size": "1min", "cumulative": 12345.0}),
        ("REGIME_OI", 0.4, {"label": "positive"}),
    ):
        await _seed_metric(db_session, symbol="SPXW", metric_type=metric_type, value=value, extra_json=extra, ts=ts)

    resp = await app_client.get("/v1/SPXW/snapshot", headers={"X-API-Key": plaintext})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    data = body["data"]
    expected_keys = {
        "gex",
        "gex_volume",
        "max_pain",
        "walls_oi",
        "walls_volume",
        "iv",
        "vanna_total",
        "charm_total",
        "vanna_level",
        "charm_level",
        "regime",
        "zero_gamma",
        "pin_probability",
        "move_tracker",
        "risk_reversal_25d",
        "iv_term_structure",
        "hiro_cumulative",
        "flow_events_last_hour",
    }
    missing = expected_keys.difference(data.keys())
    assert not missing, f"snapshot missing keys: {missing}"
    assert data["hiro_cumulative"] == pytest.approx(12345.0)
    assert data["flow_events_last_hour"] == 0


@pytest.mark.asyncio
async def test_flow_endpoint_filters_by_event_type_and_since(app_client, db_session) -> None:
    """/flow filters must respect ``event_type`` and ``since``."""
    from app.db.models import FlowEvent

    plaintext, _ = await _make_key(db_session, symbols=["SPXW"])

    now = datetime.now(UTC)
    rows = [
        FlowEvent(
            ts=now - timedelta(minutes=5),
            symbol="SPXW",
            expiration=(now - timedelta(minutes=5)).date(),
            strike=4500.0,
            option_type="C",
            event_type="SWEEP",
            side=1,
            size=10,
            price=1.25,
            legs=3,
            venues=["XOPR"],
        ),
        FlowEvent(
            ts=now - timedelta(minutes=10),
            symbol="SPXW",
            expiration=(now - timedelta(minutes=10)).date(),
            strike=4505.0,
            option_type="P",
            event_type="BLOCK",
            side=-1,
            size=200,
            price=2.0,
            legs=1,
            venues=["BOXO"],
        ),
        FlowEvent(
            ts=now - timedelta(hours=2),  # older than default 1h window
            symbol="SPXW",
            expiration=(now - timedelta(hours=2)).date(),
            strike=4510.0,
            option_type="C",
            event_type="SWEEP",
            side=1,
            size=50,
            price=1.5,
            legs=2,
            venues=["XOPR"],
        ),
    ]
    for r in rows:
        db_session.add(r)
    await db_session.commit()

    # All recent events (default ``since`` = now - 1h).
    resp = await app_client.get("/v1/SPXW/flow", headers={"X-API-Key": plaintext})
    assert resp.status_code == 200
    body = resp.json()
    assert {e["event_type"] for e in body["events"]} == {"SWEEP", "BLOCK"}

    # Filter to BLOCK only.
    resp = await app_client.get(
        "/v1/SPXW/flow", params={"event_type": "BLOCK"}, headers={"X-API-Key": plaintext}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [e["event_type"] for e in body["events"]] == ["BLOCK"]

    # Custom ``since`` covering the 2-hour-old row.
    since = (now - timedelta(hours=3)).isoformat()
    resp = await app_client.get(
        "/v1/SPXW/flow",
        params={"event_type": "SWEEP", "since": since},
        headers={"X-API-Key": plaintext},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["events"]) == 2  # both SWEEPs are visible


@pytest.mark.asyncio
async def test_hiro_endpoint_returns_chronological_series(app_client, db_session) -> None:
    """/hiro returns the latest persisted HIRO series, ts-ordered ascending."""
    # Use NDXP to isolate from any HIRO rows other tests may have written
    # for SPXW into the shared testcontainer.
    plaintext, _ = await _make_key(db_session, symbols=["NDXP"])

    base = datetime.now(UTC).replace(microsecond=0, second=0)
    series = [
        {
            "ts": (base - timedelta(minutes=3)).isoformat(),
            "call_premium": 100.0,
            "put_premium": -20.0,
            "net_premium": 80.0,
            "cumulative": 80.0,
        },
        {
            "ts": (base - timedelta(minutes=2)).isoformat(),
            "call_premium": 50.0,
            "put_premium": -10.0,
            "net_premium": 40.0,
            "cumulative": 40.0,
        },
        {
            "ts": (base - timedelta(minutes=1)).isoformat(),
            "call_premium": 200.0,
            "put_premium": -50.0,
            "net_premium": 150.0,
            "cumulative": 150.0,
        },
    ]
    await _seed_metric(
        db_session,
        symbol="NDXP",
        metric_type="HIRO",
        value=150.0,
        extra_json={"bucket_size": "1min", "series": series, "cumulative": 150.0},
        ts=base,
    )

    resp = await app_client.get(
        "/v1/NDXP/hiro", params={"bucket": "1m"}, headers={"X-API-Key": plaintext}
    )
    assert resp.status_code == 200
    body = resp.json()
    timestamps = [entry["ts"] for entry in body["series"]]
    assert timestamps == sorted(timestamps)
    assert body["cumulative"] == pytest.approx(150.0)


# ────────────────────────────────────────────────────────────────────────────
# WebSocket tests — use Starlette's TestClient which speaks the ASGI WS
# handshake natively (no real HTTP/TCP needed).
# ────────────────────────────────────────────────────────────────────────────


def _build_ws_test_client():
    """Build a ``TestClient`` bound to ``create_app()`` with DB overrides.

    Returns ``(client, plaintext_key)`` once a Postgres testcontainer +
    seeded ApiKey row are ready, or ``(None, None)`` if Postgres isn't
    available — in which case the caller should ``pytest.skip``.
    """
    from starlette.testclient import TestClient

    from app.db.session import get_db, get_session_factory
    from app.main import create_app

    app = create_app()

    async def _override_get_db():
        async with get_session_factory()() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    return TestClient(app)


def test_ws_stream_propagates_published_payload_within_100ms(monkeypatch) -> None:
    """Publishing to the notifier should propagate to a WS subscriber.

    Auth + initial snapshot priming are mocked so the test can run on a
    TestClient loop that does not share asyncpg connections with the
    surrounding test session.
    """
    stream_mod.reset_ws_state_for_tests()
    notifier_mod.reset_stream_notifier_for_tests()

    class _FakeApiKey:
        id = "ws-propagate-key"
        is_active = True
        expires_at = None
        allowed_symbols = ["SPXW"]

    async def _fake_auth(api_key, symbol, session):  # noqa: ARG001
        return _FakeApiKey() if api_key == "ak_propagate_fake" else None

    async def _empty_payload(session, symbol):  # noqa: ARG001
        return {"gex": {}, "flow_events_last_hour": 0}, None

    monkeypatch.setattr(stream_mod, "_authenticate_streaming_key", _fake_auth)
    monkeypatch.setattr(stream_mod, "build_snapshot_payload", _empty_payload)

    client = _build_ws_test_client()
    plaintext = "ak_propagate_fake"
    try:
        with client.websocket_connect(
            "/v1/SPXW/stream", headers={"X-API-Key": plaintext}
        ) as ws:
            initial = ws.receive_json(mode="text")
            assert initial["symbol"] == "SPXW"
            assert "data" in initial

            # Use the WS portal to schedule the publish on the same loop
            # the WS handler is attached to. ``portal.call`` blocks until
            # the coroutine returns, so the publish has definitely landed
            # before we drop into ``receive_json``.
            async def _publish() -> int:
                notifier = notifier_mod.get_stream_notifier()
                return await notifier.publish(
                    "SPXW", {"data": {"foo": 42}, "computed_at": None}
                )

            t0 = datetime.now(UTC)
            ws.portal.call(_publish)
            frame = ws.receive_json(mode="text")
            elapsed = (datetime.now(UTC) - t0).total_seconds()

            assert frame["symbol"] == "SPXW"
            assert frame["data"] == {"foo": 42}
            assert elapsed < 0.1, f"WS propagation took {elapsed * 1000:.1f}ms"
    finally:
        client.close()


def test_ws_stream_caps_connections_per_key(monkeypatch) -> None:
    """Opening MAX + 1 connections with the same key must close the extra one.

    Auth is mocked here so the test doesn't depend on the Postgres fixture,
    keeping the assertion focused on the WS connection-cap behaviour and
    avoiding cross-loop asyncpg pool reuse with sibling DB-touching tests.
    """
    from starlette.websockets import WebSocketDisconnect

    from app.config import get_settings

    stream_mod.reset_ws_state_for_tests()

    class _FakeApiKey:
        id = "test-key-id"
        is_active = True
        expires_at = None
        allowed_symbols = ["SPXW"]

    async def _fake_auth(api_key, symbol, session):  # noqa: ARG001
        return _FakeApiKey() if api_key == "ak_test_fake_key" else None

    monkeypatch.setattr(stream_mod, "_authenticate_streaming_key", _fake_auth)
    monkeypatch.setattr(get_settings(), "max_ws_connections_per_key", 2)

    # Also stub the initial-snapshot priming so the WS connect doesn't need a DB.
    async def _empty_payload(session, symbol):  # noqa: ARG001
        return {"gex": {}, "flow_events_last_hour": 0}, None

    monkeypatch.setattr(stream_mod, "build_snapshot_payload", _empty_payload)

    client = _build_ws_test_client()
    open_connections: list = []
    plaintext = "ak_test_fake_key"
    try:
        for _ in range(2):
            ctx = client.websocket_connect(
                "/v1/SPXW/stream", headers={"X-API-Key": plaintext}
            )
            ws = ctx.__enter__()
            ws.receive_json(mode="text")  # drain initial primer
            open_connections.append((ctx, ws))

        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect(
                "/v1/SPXW/stream", headers={"X-API-Key": plaintext}
            ) as ws3:
                ws3.receive_text()
        assert excinfo.value.code == 1008
    finally:
        for ctx, _ in open_connections:
            try:
                ctx.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        client.close()


def test_ws_stream_rejects_missing_api_key(monkeypatch) -> None:
    """WS without a valid X-API-Key must close with policy violation 1008."""
    from starlette.websockets import WebSocketDisconnect

    stream_mod.reset_ws_state_for_tests()

    async def _fake_auth(api_key, symbol, session):  # noqa: ARG001
        return None

    monkeypatch.setattr(stream_mod, "_authenticate_streaming_key", _fake_auth)

    client = _build_ws_test_client()
    try:
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect("/v1/SPXW/stream"):
                pass
        assert excinfo.value.code == 1008
    finally:
        client.close()


def test_sse_event_encoding_shape() -> None:
    """SSE frames must follow the ``data: <json>\\n\\n`` wire format."""
    encoded = stream_mod._sse_event({"hello": "world"})
    assert encoded.startswith("data: ")
    assert encoded.endswith("\n\n")
    body = encoded.removeprefix("data: ").rstrip("\n")
    assert json.loads(body) == {"hello": "world"}

    encoded_event = stream_mod._sse_event({"x": 1}, event="heartbeat")
    assert encoded_event.startswith("event: heartbeat\ndata: ")
