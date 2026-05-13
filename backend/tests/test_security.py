"""Tests for security primitives: API-key/password hashing + JWT tokens."""

from __future__ import annotations

import time

import pytest

from app.core.security import (
    create_jwt_token,
    decode_jwt_token,
    display_prefix,
    generate_api_key,
    hash_api_key,
    hash_password,
    verify_api_key,
    verify_password,
)


def test_api_key_round_trip():
    key = generate_api_key()
    assert key.startswith("ak_")
    h = hash_api_key(key)
    assert verify_api_key(key, h) is True
    assert verify_api_key("ak_wrong_key", h) is False


def test_api_key_display_prefix():
    key = generate_api_key()
    prefix = display_prefix(key)
    assert prefix.startswith("ak_")
    assert len(prefix) == 11


def test_password_round_trip():
    h = hash_password("hunter2")
    assert verify_password("hunter2", h)
    assert not verify_password("wrong", h)


def test_jwt_token_round_trip():
    token = create_jwt_token("admin", expires_minutes=5)
    payload = decode_jwt_token(token)
    assert payload["sub"] == "admin"
    assert "exp" in payload
    assert payload["exp"] > int(time.time())


def test_jwt_token_expired_raises():
    import jwt as pyjwt

    token = create_jwt_token("admin", expires_minutes=-1)  # already expired
    with pytest.raises(pyjwt.ExpiredSignatureError):
        decode_jwt_token(token)
