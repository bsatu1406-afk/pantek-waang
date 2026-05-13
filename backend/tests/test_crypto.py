"""Round-trip and tamper tests for app.core.crypto."""

from __future__ import annotations

import pytest
from cryptography.fernet import InvalidToken

from app.core.crypto import (
    decrypt_secret,
    encrypt_secret,
    mask_prefix,
    reset_crypto_cache,
)


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    reset_crypto_cache()
    yield
    reset_crypto_cache()


def test_encrypt_decrypt_roundtrip() -> None:
    plaintext = "db-1234567890abcdef-secret"
    token = encrypt_secret(plaintext)
    assert token != plaintext
    assert isinstance(token, str)
    assert decrypt_secret(token) == plaintext


def test_encrypt_produces_distinct_tokens_for_same_input() -> None:
    """Fernet randomizes the IV — same input must yield different tokens."""
    token1 = encrypt_secret("same-input")
    token2 = encrypt_secret("same-input")
    assert token1 != token2


def test_decrypt_rejects_tampered_token() -> None:
    plaintext = "hello"
    token = encrypt_secret(plaintext)
    tampered = token[:-2] + "AA"
    with pytest.raises(InvalidToken):
        decrypt_secret(tampered)


def test_decrypt_rejects_garbage() -> None:
    with pytest.raises(InvalidToken):
        decrypt_secret("not-a-token-at-all")


def test_mask_prefix_short_secret_returned_in_full() -> None:
    assert mask_prefix("abc", chars=8) == "abc"


def test_mask_prefix_long_secret_truncated() -> None:
    assert mask_prefix("db-12345678ZZZ", chars=8) == "db-12345"


def test_handles_unicode_and_empty() -> None:
    assert decrypt_secret(encrypt_secret("")) == ""
    assert decrypt_secret(encrypt_secret("rüssel")) == "rüssel"
