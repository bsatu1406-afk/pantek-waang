"""Admin endpoints (JWT-protected)."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import authenticate_admin
from app.api.schemas import (
    AdminLoginRequest,
    AdminLoginResponse,
    ApiKeyCreate,
    ApiKeyCreateResponse,
    ApiKeySummary,
    ApiKeyUpdate,
    SystemStatus,
)
from app.config import get_settings
from app.core.security import (
    create_jwt_token,
    display_prefix,
    generate_api_key,
    hash_api_key,
    verify_password,
)
from app.db.models import ApiKey, ComputedMetric, OptionsChain
from app.db.session import get_db
from app.ingestion.writer import get_writer
from app.processing.scheduler import get_pipeline_state

router = APIRouter(prefix="/admin", tags=["admin"])


# ── Login ────────────────────────────────────────────────────────────────────

@router.post("/login", response_model=AdminLoginResponse)
async def admin_login(payload: AdminLoginRequest) -> AdminLoginResponse:
    settings = get_settings()
    if payload.username != settings.admin_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    # The admin password is bootstrapped from env. We accept either:
    #   1) a plaintext value matching ADMIN_PASSWORD, or
    #   2) a bcrypt-hashed value matching the admin password (for prod).
    is_hash = settings.admin_password.startswith("$2")
    valid = (
        verify_password(payload.password, settings.admin_password)
        if is_hash
        else payload.password == settings.admin_password
    )
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    token = create_jwt_token(settings.admin_username)
    return AdminLoginResponse(
        access_token=token, expires_in_seconds=settings.jwt_expire_minutes * 60
    )


# ── API key CRUD ────────────────────────────────────────────────────────────


def _to_summary(row: ApiKey) -> ApiKeySummary:
    return ApiKeySummary(
        id=row.id,
        key_prefix=row.key_prefix,
        label=row.label,
        allowed_symbols=list(row.allowed_symbols or []),
        created_at=row.created_at,
        expires_at=row.expires_at,
        is_active=row.is_active,
        last_used_at=row.last_used_at,
        usage_count=row.usage_count or 0,
    )


@router.get("/api-keys", response_model=list[ApiKeySummary])
async def list_api_keys(
    _admin: Annotated[str, Depends(authenticate_admin)],
    session: AsyncSession = Depends(get_db),
) -> list[ApiKeySummary]:
    rows = (await session.execute(select(ApiKey).order_by(ApiKey.created_at.desc()))).scalars().all()
    return [_to_summary(r) for r in rows]


@router.post("/api-keys", response_model=ApiKeyCreateResponse, status_code=201)
async def create_api_key(
    payload: ApiKeyCreate,
    _admin: Annotated[str, Depends(authenticate_admin)],
    session: AsyncSession = Depends(get_db),
) -> ApiKeyCreateResponse:
    plaintext = generate_api_key()
    record = ApiKey(
        key_hash=hash_api_key(plaintext),
        key_prefix=display_prefix(plaintext),
        label=payload.label,
        allowed_symbols=payload.allowed_symbols,
        expires_at=payload.expires_at,
        is_active=True,
        usage_count=0,
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    return ApiKeyCreateResponse(key=_to_summary(record), plaintext_key=plaintext)


@router.patch("/api-keys/{key_id}", response_model=ApiKeySummary)
async def update_api_key(
    key_id: UUID,
    payload: ApiKeyUpdate,
    _admin: Annotated[str, Depends(authenticate_admin)],
    session: AsyncSession = Depends(get_db),
) -> ApiKeySummary:
    row = await session.get(ApiKey, key_id)
    if row is None:
        raise HTTPException(status_code=404, detail="API key not found")
    if payload.label is not None:
        row.label = payload.label
    if payload.allowed_symbols is not None:
        row.allowed_symbols = payload.allowed_symbols
    if payload.expires_at is not None:
        row.expires_at = payload.expires_at
    if payload.is_active is not None:
        row.is_active = payload.is_active
    await session.commit()
    await session.refresh(row)
    return _to_summary(row)


@router.delete("/api-keys/{key_id}", status_code=204, response_class=Response)
async def delete_api_key(
    key_id: UUID,
    _admin: Annotated[str, Depends(authenticate_admin)],
    session: AsyncSession = Depends(get_db),
) -> Response:
    row = await session.get(ApiKey, key_id)
    if row is None:
        raise HTTPException(status_code=404, detail="API key not found")
    await session.delete(row)
    await session.commit()
    return Response(status_code=204)


@router.get("/api-keys/{key_id}/usage")
async def api_key_usage(
    key_id: UUID,
    _admin: Annotated[str, Depends(authenticate_admin)],
    session: AsyncSession = Depends(get_db),
) -> dict:
    row = await session.get(ApiKey, key_id)
    if row is None:
        raise HTTPException(status_code=404, detail="API key not found")
    return {
        "id": str(row.id),
        "label": row.label,
        "key_prefix": row.key_prefix,
        "usage_count": row.usage_count or 0,
        "last_used_at": row.last_used_at.isoformat() if row.last_used_at else None,
        "is_active": row.is_active,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
    }


# ── System status ────────────────────────────────────────────────────────────


@router.get("/system/status", response_model=SystemStatus)
async def system_status(
    _admin: Annotated[str, Depends(authenticate_admin)],
    session: AsyncSession = Depends(get_db),
) -> SystemStatus:
    settings = get_settings()
    state = get_pipeline_state()
    writer = get_writer()

    rows_per_symbol: dict[str, int] = {}
    metric_rows_per_symbol: dict[str, int] = {}
    for symbol in settings.supported_symbols:
        chain_count = (
            await session.execute(
                select(func.count())
                .select_from(OptionsChain)
                .where(OptionsChain.symbol == symbol)
            )
        ).scalar_one()
        metric_count = (
            await session.execute(
                select(func.count())
                .select_from(ComputedMetric)
                .where(ComputedMetric.symbol == symbol)
            )
        ).scalar_one()
        rows_per_symbol[symbol] = int(chain_count or 0)
        metric_rows_per_symbol[symbol] = int(metric_count or 0)

    active_keys = (
        await session.execute(
            select(func.count()).select_from(ApiKey).where(ApiKey.is_active.is_(True))
        )
    ).scalar_one()

    return SystemStatus(
        pipeline_running=bool(state.last_run),
        last_databento_event=writer.last_event_ts,
        last_compute_per_symbol={
            sym: state.last_run.get(sym) for sym in settings.supported_symbols
        },
        last_compute_duration_ms={
            sym: float(state.last_duration_ms.get(sym, 0.0))
            for sym in settings.supported_symbols
        },
        rows_per_symbol=rows_per_symbol,
        metric_rows_per_symbol=metric_rows_per_symbol,
        active_api_keys=int(active_keys or 0),
    )
