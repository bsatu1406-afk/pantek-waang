"""Pydantic schemas for request/response payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

# ── Auth ────────────────────────────────────────────────────────────────────

class AdminLoginRequest(BaseModel):
    username: str
    password: str


class AdminLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int


# ── API key management ──────────────────────────────────────────────────────

class ApiKeyCreate(BaseModel):
    label: str = Field(min_length=1, max_length=200)
    allowed_symbols: list[str]
    expires_at: datetime | None = None

    @field_validator("allowed_symbols")
    @classmethod
    def _normalize_symbols(cls, v: list[str]) -> list[str]:
        return [s.strip().upper() for s in v if s.strip()]


class ApiKeyUpdate(BaseModel):
    label: str | None = Field(default=None, min_length=1, max_length=200)
    allowed_symbols: list[str] | None = None
    expires_at: datetime | None = None
    is_active: bool | None = None

    @field_validator("allowed_symbols")
    @classmethod
    def _normalize_symbols(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        return [s.strip().upper() for s in v if s.strip()]


class ApiKeySummary(BaseModel):
    id: UUID
    key_prefix: str
    label: str
    allowed_symbols: list[str]
    created_at: datetime
    expires_at: datetime | None
    is_active: bool
    last_used_at: datetime | None
    usage_count: int


class ApiKeyCreateResponse(BaseModel):
    key: ApiKeySummary
    plaintext_key: str = Field(
        description="Plaintext API key. Shown ONCE — store it securely."
    )


# ── Data endpoint envelopes ─────────────────────────────────────────────────

class DataEnvelope(BaseModel):
    symbol: str
    computed_at: datetime | None
    next_update_in_seconds: int
    data: dict[str, Any]


# ── System status ───────────────────────────────────────────────────────────

class SystemStatus(BaseModel):
    pipeline_running: bool
    last_databento_event: datetime | None
    last_compute_per_symbol: dict[str, datetime | None]
    last_compute_duration_ms: dict[str, float]
    rows_per_symbol: dict[str, int]
    metric_rows_per_symbol: dict[str, int]
    active_api_keys: int
