"""Shared pytest fixtures.

The DB-backed tests (API + admin) require a Postgres instance. We use one of:

  1. ``TEST_DATABASE_URL`` env var if set — must be a postgresql+asyncpg URL.
  2. Otherwise, a Postgres testcontainer (when ``testcontainers`` is installed
     and a Docker daemon is reachable).
  3. If neither is available, DB-backed tests are skipped automatically.

Pure-function tests (processing, security primitives) do NOT depend on the DB
and always run.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

os.environ.setdefault("APP_TESTING", "1")
os.environ.setdefault("DATABENTO_API_KEY", "")
os.environ.setdefault("DATABENTO_API_KEY_OPRA", "")
os.environ.setdefault("DATABENTO_API_KEY_GLOBEX", "")
os.environ.setdefault("DISABLE_LIVE_INGESTION", "true")
os.environ.setdefault("DISABLE_HISTORICAL_BACKFILL", "true")
os.environ.setdefault("ADMIN_USERNAME", "admin")
os.environ.setdefault("ADMIN_PASSWORD", "test-password")
os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("SUPPORTED_SYMBOLS", "SPXW,NDXP")


def _get_postgres_url_from_env() -> str | None:
    return os.getenv("TEST_DATABASE_URL")


def _try_start_testcontainer() -> str | None:
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        return None
    try:
        container = PostgresContainer("postgres:15-alpine")
        container.start()
    except Exception:  # noqa: BLE001 -- Docker not available
        return None
    sync_url = container.get_connection_url()
    # PostgresContainer returns "postgresql+psycopg2://..." or "postgresql://..." depending on version.
    async_url = sync_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://").replace(
        "postgresql://", "postgresql+asyncpg://"
    )
    pytest._postgres_container = container  # type: ignore[attr-defined]  # keep alive
    return async_url


@pytest_asyncio.fixture(loop_scope="session", scope="session")
async def database_url() -> AsyncIterator[str | None]:
    url = _get_postgres_url_from_env() or _try_start_testcontainer()
    yield url
    container = getattr(pytest, "_postgres_container", None)
    if container is not None:
        try:
            container.stop()
        except Exception:  # noqa: BLE001
            pass


@pytest_asyncio.fixture(loop_scope="session", scope="session")
async def engine_for_tests(database_url: str | None):
    if database_url is None:
        pytest.skip("Postgres not available; set TEST_DATABASE_URL or install Docker.")
    os.environ["DATABASE_URL"] = database_url

    from app.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]

    from app.db import models  # noqa: F401 - register models
    from app.db.session import Base, dispose_engine, get_engine

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await dispose_engine()


@pytest_asyncio.fixture(loop_scope="session")
async def db_session(engine_for_tests):
    from app.db.session import get_session_factory
    factory = get_session_factory()
    async with factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture(loop_scope="session")
async def app_client(engine_for_tests):
    """HTTP client bound to the FastAPI app with DB sessions overridden."""
    import httpx

    from app.db.session import get_db
    from app.main import create_app

    app = create_app()

    async def _override_get_db():
        from app.db.session import get_session_factory
        async with get_session_factory()() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client



