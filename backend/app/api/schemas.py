"""Pydantic schemas for request/response payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import TypeVar

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

# TypeVar with a default lets us keep ``DataEnvelope(...)`` working as a
# concrete dict envelope while ALSO supporting parametric ``DataEnvelope[T]``
# typed responses for the data endpoints.
T = TypeVar("T", default=Any)


class DataEnvelope(BaseModel, Generic[T]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbol: str
    computed_at: datetime | None
    next_update_in_seconds: int
    data: T


# ── Typed data payloads ─────────────────────────────────────────────────────

class GexResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    net_total: float
    curve: list[dict[str, Any]] = Field(default_factory=list)
    top_positive: list[dict[str, Any]] = Field(default_factory=list)
    top_negative: list[dict[str, Any]] = Field(default_factory=list)


class MaxPainExpiryEntry(BaseModel):
    expiration: str
    strike: float
    pain: float


class MaxPainAggregate(BaseModel):
    strike: float
    value: float


class MaxPainResponse(BaseModel):
    per_expiry: list[MaxPainExpiryEntry] = Field(default_factory=list)
    aggregate: MaxPainAggregate | None = None


class WallEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    rank: int
    strike: float
    value: float


class WallsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    call_wall_oi: list[WallEntry] = Field(default_factory=list)
    put_wall_oi: list[WallEntry] = Field(default_factory=list)
    call_wall_volume: list[WallEntry] = Field(default_factory=list)
    put_wall_volume: list[WallEntry] = Field(default_factory=list)


class IvResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    atm_iv: float | None = None
    skew: dict[str, float] = Field(default_factory=dict)
    surface: list[dict[str, Any]] = Field(default_factory=list)


# ── System status ───────────────────────────────────────────────────────────

class PipelineRunSummary(BaseModel):
    """Most recent pipeline run for one symbol."""

    symbol: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float = 0.0
    status: str = "unknown"
    rows_read: int = 0
    metric_rows_written: int = 0
    missing_metric_types: list[str] = Field(default_factory=list)
    error: str | None = None


class SystemStatus(BaseModel):
    """Operational telemetry returned by ``GET /admin/system/status``.

    Pre-existing fields (``rows_per_symbol``, ``active_api_keys``,
    ``last_compute_per_symbol`` ...) remain for backward compatibility;
    the Rev 3 fields below are additive.
    """

    pipeline_running: bool
    last_databento_event: datetime | None
    last_compute_per_symbol: dict[str, datetime | None]
    last_compute_duration_ms: dict[str, float]
    rows_per_symbol: dict[str, int]
    metric_rows_per_symbol: dict[str, int]
    active_api_keys: int

    # ── Rev 3 operational telemetry ─────────────────────────────────────────
    futures_lag_ms: float | None = None
    """`now() - max(futures_ticks.ts)` in milliseconds, or null if no rows."""
    opra_lag_ms: float | None = None
    """`now() - max(options_chain.ts)` in milliseconds, or null if no rows."""
    dlq_pending: int = 0
    """`dead_letter_queue` row count."""
    flow_events_last_hour: int = 0
    """Count of ``flow_events`` rows inserted in the last 1 hour."""
    last_pipeline_runs: list[PipelineRunSummary] = Field(default_factory=list)
    """Last row per symbol from ``pipeline_runs``."""
    live_ingester: dict[str, Any] = Field(default_factory=dict)
    """Diagnostics from :meth:`DatabentoLiveIngester.diagnostics`."""


# ── DLQ inspector ───────────────────────────────────────────────────────────

class DlqEntry(BaseModel):
    id: UUID
    ts: datetime
    source: str
    reason: str
    payload: dict[str, Any] | None = None


class DlqPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[DlqEntry] = Field(default_factory=list)


# ── Databento API key pool (Rev 4) ───────────────────────────────────────────


_DATASET_ALLOWED = {"OPRA.PILLAR", "GLBX.MDP3", "BOTH"}


class DatabentoKeyCreate(BaseModel):
    label: str = Field(min_length=1, max_length=200)
    dataset: str
    api_key: str = Field(min_length=8, max_length=512)
    priority: int = Field(default=100, ge=0, le=10_000)
    is_active: bool = True

    @field_validator("dataset")
    @classmethod
    def _normalize_dataset(cls, v: str) -> str:
        s = v.strip().upper()
        if s not in _DATASET_ALLOWED:
            raise ValueError(
                f"dataset must be one of {sorted(_DATASET_ALLOWED)}, got {v!r}"
            )
        return s


class DatabentoKeyUpdate(BaseModel):
    label: str | None = Field(default=None, min_length=1, max_length=200)
    priority: int | None = Field(default=None, ge=0, le=10_000)
    is_active: bool | None = None

    # We deliberately do NOT allow rotating the api_key/dataset via PATCH —
    # the operator should delete + re-create. This keeps the audit story
    # cleaner (one row = one secret).


class DatabentoKeySummary(BaseModel):
    id: int
    label: str
    dataset: str
    api_key_prefix: str
    priority: int
    is_active: bool
    last_used_at: datetime | None
    last_error_at: datetime | None
    last_error_msg: str | None
    error_count: int
    created_at: datetime


class DatabentoKeyTestResult(BaseModel):
    ok: bool
    message: str
    """Human-readable description of the test outcome."""
