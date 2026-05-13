"""Security primitives: API key generation/hashing and JWT helpers."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from app.config import get_settings

API_KEY_PREFIX = "ak_"
API_KEY_RANDOM_BYTES = 24  # 32-char base64-urlsafe-no-pad
API_KEY_DISPLAY_PREFIX_LEN = 11  # "ak_" + 8 chars


def generate_api_key() -> str:
    """Return a fresh plaintext API key. Display prefix = first 11 chars."""
    token = secrets.token_urlsafe(API_KEY_RANDOM_BYTES)
    return f"{API_KEY_PREFIX}{token}"


def display_prefix(api_key: str) -> str:
    return api_key[:API_KEY_DISPLAY_PREFIX_LEN]


def hash_api_key(api_key: str) -> str:
    """Hash an API key with bcrypt. Returns the encoded hash as a UTF-8 string."""
    return bcrypt.hashpw(api_key.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_api_key(api_key: str, key_hash: str) -> bool:
    try:
        return bcrypt.checkpw(api_key.encode("utf-8"), key_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_jwt_token(subject: str, *, expires_minutes: int | None = None) -> str:
    settings = get_settings()
    now = datetime.now(UTC)
    exp = now + timedelta(minutes=expires_minutes or settings.jwt_expire_minutes)
    payload = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_jwt_token(token: str) -> dict:
    settings = get_settings()
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
