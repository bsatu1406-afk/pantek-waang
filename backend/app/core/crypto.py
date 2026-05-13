"""Symmetric encryption helper for sensitive at-rest secrets.

The only thing this module is used for today is the Databento API key
pool (``databento_api_keys.api_key_encrypted``). We do not roll our own
crypto — we wrap :class:`cryptography.fernet.Fernet`, which is the
audited AES-128-CBC + HMAC-SHA-256 scheme from the ``cryptography``
library.

The Fernet key itself is derived deterministically from the application
``JWT_SECRET`` so we don't introduce a second piece of operator-managed
secret material. The derivation uses HKDF-SHA-256 with a fixed
application-specific salt and ``info`` label so:

* knowing ``JWT_SECRET`` always reproduces the same Fernet key (so the
  encrypted blobs in the DB stay readable after a restart), and
* the derived key is not the same as ``JWT_SECRET`` itself — leaking
  the encryption key does *not* leak the JWT signing key.

**Rotation note.** Today there is exactly one active derivation. If we
ever rotate ``JWT_SECRET`` we'll need a re-encryption job: read each
encrypted blob with the old derived key, write it back with the new
one. There's no in-place support for that yet — opening the door now
would be premature.
"""

from __future__ import annotations

import base64
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from app.config import get_settings

# 16 bytes of fixed salt. Salt does not need to be secret — it just
# domain-separates this derivation from any other place the JWT secret
# might be used in HKDF.
_HKDF_SALT = b"pantek-waang.crypto.v1"
_HKDF_INFO = b"databento-api-key-encryption"


def _derive_fernet_key(jwt_secret: str) -> bytes:
    """Deterministically derive a 32-byte Fernet key from ``jwt_secret``.

    Returns a URL-safe base64 encoded 32-byte key, which is what
    :class:`Fernet` expects.
    """
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_HKDF_SALT,
        info=_HKDF_INFO,
    )
    raw = hkdf.derive(jwt_secret.encode("utf-8"))
    return base64.urlsafe_b64encode(raw)


@lru_cache(maxsize=1)
def _get_fernet() -> Fernet:
    """Cached Fernet instance derived from current settings."""
    settings = get_settings()
    secret = settings.jwt_secret or "dev-only-change-me"
    return Fernet(_derive_fernet_key(secret))


def reset_crypto_cache() -> None:
    """Test helper — force the next call to re-derive from settings."""
    _get_fernet.cache_clear()


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a string secret. Returns the ciphertext as a URL-safe str."""
    token = _get_fernet().encrypt(plaintext.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_secret(token: str) -> str:
    """Decrypt a token produced by :func:`encrypt_secret`.

    Raises :class:`cryptography.fernet.InvalidToken` on tampered or
    cross-secret blobs. We deliberately do not silently substitute an
    empty string — callers must handle the failure.
    """
    return _get_fernet().decrypt(token.encode("utf-8")).decode("utf-8")


def mask_prefix(plaintext: str, *, chars: int = 8) -> str:
    """Return the first ``chars`` characters of a secret for UI display.

    Anything shorter than ``chars`` is returned in full — the trade-off
    is acceptable because the caller is responsible for not passing in
    secrets shorter than the value they wanted to hide.
    """
    s = plaintext.strip()
    if len(s) <= chars:
        return s
    return s[:chars]


__all__ = [
    "InvalidToken",
    "decrypt_secret",
    "encrypt_secret",
    "mask_prefix",
    "reset_crypto_cache",
]
