"""FastAPI dependencies: API-key auth, JWT admin auth, and rate limiting."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

import jwt
from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.security import decode_jwt_token, verify_api_key
from app.db.models import ApiKey
from app.db.session import get_db

# ── Rate limiter ─────────────────────────────────────────────────────────────


def _api_key_or_ip(request: Request) -> str:
    api_key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")
    if api_key:
        return f"key:{api_key[:11]}"
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=_api_key_or_ip)


def get_limiter() -> Limiter:
    return limiter


# ── API key auth ─────────────────────────────────────────────────────────────


async def authenticate_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    session: AsyncSession = Depends(get_db),
) -> ApiKey:
    """Validate the X-API-Key header and return the matching ApiKey row.

    Increments ``usage_count`` and updates ``last_used_at`` on success.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )

    prefix = x_api_key[:11]
    result = await session.execute(select(ApiKey).where(ApiKey.key_prefix == prefix))
    candidates = result.scalars().all()

    matched: ApiKey | None = None
    for candidate in candidates:
        if verify_api_key(x_api_key, candidate.key_hash):
            matched = candidate
            break

    if matched is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )

    if not matched.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="API key is inactive"
        )

    now = datetime.now(UTC)
    if matched.expires_at is not None:
        # Compare in UTC; row is stored as timezone-aware in PG.
        expires_at = matched.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if expires_at < now:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="API key expired"
            )

    # Update usage stats. We avoid a second roundtrip by updating in-session.
    matched.usage_count = (matched.usage_count or 0) + 1
    matched.last_used_at = now
    await session.commit()
    await session.refresh(matched)

    request.state.api_key = matched
    return matched


def require_symbol_access(symbol_param: str = "symbol"):
    """Factory: dependency that ensures the API key is allowed to access the path symbol."""

    async def _dep(
        request: Request,
        api_key: Annotated[ApiKey, Depends(authenticate_api_key)],
    ) -> ApiKey:
        symbol = request.path_params.get(symbol_param)
        if symbol is None:
            return api_key
        symbol_u = symbol.upper()
        if symbol_u not in [s.upper() for s in (api_key.allowed_symbols or [])]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key not authorized for symbol {symbol_u}",
            )
        return api_key

    return _dep


# ── Admin (JWT) auth ─────────────────────────────────────────────────────────

bearer_scheme = HTTPBearer(auto_error=False)


async def authenticate_admin(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> str:
    if creds is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token"
        )
    try:
        payload = decode_jwt_token(creds.credentials)
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        ) from exc

    sub = payload.get("sub")
    settings = get_settings()
    if sub != settings.admin_username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not an admin user"
        )
    return sub
