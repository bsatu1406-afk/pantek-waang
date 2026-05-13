"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Databento ────────────────────────────────────────────────────────────
    # ``DATABENTO_API_KEY`` is the legacy single-key fallback used when the
    # dataset-specific keys below are not set. New deployments should set
    # ``DATABENTO_API_KEY_OPRA`` (OPRA Pillar — options) and
    # ``DATABENTO_API_KEY_GLOBEX`` (GLBX.MDP3 — CME futures) explicitly so each
    # ingester authenticates with the correct subscription.
    databento_api_key: str = Field(default="", alias="DATABENTO_API_KEY")
    databento_api_key_opra: str = Field(default="", alias="DATABENTO_API_KEY_OPRA")
    databento_api_key_globex: str = Field(default="", alias="DATABENTO_API_KEY_GLOBEX")

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://options:options@db:5432/options_db",
        alias="DATABASE_URL",
    )

    # ── Admin auth ───────────────────────────────────────────────────────────
    admin_username: str = Field(default="admin", alias="ADMIN_USERNAME")
    admin_password: str = Field(default="changeme", alias="ADMIN_PASSWORD")
    jwt_secret: str = Field(default="dev-only-change-me", alias="JWT_SECRET")
    jwt_expire_minutes: int = Field(default=480, alias="JWT_EXPIRE_MINUTES")
    jwt_algorithm: str = "HS256"

    # ── Options config ───────────────────────────────────────────────────────
    supported_symbols_raw: str = Field(default="SPXW,NDXP", alias="SUPPORTED_SYMBOLS")
    risk_free_rate: float = Field(default=0.05, alias="RISK_FREE_RATE")
    data_retention_days: int = Field(default=7, alias="DATA_RETENTION_DAYS")
    compute_interval_seconds: int = Field(default=60, alias="COMPUTE_INTERVAL_SECONDS")
    historical_backfill_days: int = Field(default=7, alias="HISTORICAL_BACKFILL_DAYS")

    # ── Ingestion behavior ───────────────────────────────────────────────────
    disable_live_ingestion: bool = Field(default=False, alias="DISABLE_LIVE_INGESTION")
    disable_historical_backfill: bool = Field(default=False, alias="DISABLE_HISTORICAL_BACKFILL")

    # ── Regime / processing thresholds ───────────────────────────────────────
    # Score threshold (absolute value) below which the regime is reported as
    # "neutral". Increase to add hysteresis around the zero-crossing and
    # prevent flickering when GEX_NET_TOTAL is small and noisy.
    gex_regime_threshold: float = Field(default=0.2, alias="GEX_REGIME_THRESHOLD")

    # ── Flow event detection thresholds (Agent 3) ────────────────────────────
    flow_sweep_min_premium: float = Field(
        default=50_000.0, alias="FLOW_SWEEP_MIN_PREMIUM"
    )
    """Minimum dollar premium (size × price × 100) for a multi-leg cluster
    to be flagged as a SWEEP. Sweeps are aggressive multi-venue prints."""

    flow_block_min_size: int = Field(default=100, alias="FLOW_BLOCK_MIN_SIZE")
    """Minimum single-print size (contracts) to be flagged as a BLOCK."""

    flow_uoa_vol_oi_ratio: float = Field(
        default=2.0, alias="FLOW_UOA_VOL_OI_RATIO"
    )
    """volume/OI ratio threshold for UOA classification when OI is known."""

    # ── Ingestion / DB write tuning (Agent 4 / 6) ────────────────────────────
    upsert_batch_size: int = Field(default=1000, alias="UPSERT_BATCH_SIZE")
    """Batch size used by ``BulkUpsertWriter`` / ``OptionsChainWriter``."""

    ingestion_max_pending_rows: int = Field(
        default=10_000, alias="INGESTION_MAX_PENDING_ROWS"
    )
    """Hard cap on rows in any single writer's pending buffer. Past this we
    log a WARNING and flush synchronously to apply backpressure."""

    ingestion_dlq_max_size: int = Field(
        default=1000, alias="INGESTION_DLQ_MAX_SIZE"
    )
    """Maximum dead-letter queue entries retained per ingester."""

    ingestion_registry_refresh_seconds: int = Field(
        default=4 * 60 * 60, alias="INGESTION_REGISTRY_REFRESH_SECONDS"
    )
    """How often the OPRA live ingester re-bootstraps its instrument registry
    to pick up new intraday contracts. Default 4 hours during RTH."""

    futures_feed_lag_warn_ms: int = Field(
        default=5_000, alias="FUTURES_FEED_LAG_WARN_MS"
    )
    """Log a WARNING when the freshest futures tick is older than this."""

    # ── Streaming API (Agent 5) ──────────────────────────────────────────────
    max_ws_connections_per_key: int = Field(
        default=5, alias="MAX_WS_CONNECTIONS_PER_KEY"
    )
    """Cap on simultaneous WebSocket connections per API key."""

    # ── Rev 4: RTH / 0DTE / spot resolver ────────────────────────────────────
    rth_open_time: str = Field(default="09:30", alias="RTH_OPEN_TIME")
    """RTH session open in America/New_York. Format ``HH:MM``."""

    rth_close_time: str = Field(default="16:15", alias="RTH_CLOSE_TIME")
    """RTH session close in America/New_York. SPX/NDX cash options stop
    trading at 16:00 ET; we keep a 15-minute buffer so the last pipeline
    tick still emits."""

    spot_parity_deviation_warn_pct: float = Field(
        default=0.5, alias="SPOT_PARITY_DEVIATION_WARN_PCT"
    )
    """Log a WARNING when the futures-basis spot vs. parity spot differ by
    more than this percent. Helps detect feed problems."""

    spot_stale_cache_max_age_seconds: float = Field(
        default=300.0, alias="SPOT_STALE_CACHE_MAX_AGE_SECONDS"
    )
    """Reject a stale-cache spot fallback older than this. Default 5 min."""

    spot_basis_ema_alpha: float = Field(
        default=0.1, alias="SPOT_BASIS_EMA_ALPHA"
    )
    """Smoothing factor (0–1) for the cash-minus-futures basis EMA."""

    atm_band_pct_0dte: float = Field(default=0.005, alias="ATM_BAND_PCT_0DTE")
    """Half-width of the ATM band used by 0DTE charm-rate computation.
    0.005 ⇒ ±0.5% of spot (so a 10-pt window at SPX ≈ 5000)."""

    # ── Misc ─────────────────────────────────────────────────────────────────
    rate_limit_per_minute: int = Field(default=120, alias="RATE_LIMIT_PER_MINUTE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("supported_symbols_raw")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @property
    def supported_symbols(self) -> list[str]:
        return [s.strip().upper() for s in self.supported_symbols_raw.split(",") if s.strip()]

    @property
    def opra_api_key(self) -> str:
        """API key used to authenticate against OPRA.PILLAR (options).

        Falls back to the legacy ``DATABENTO_API_KEY`` so existing single-key
        deployments keep working.
        """
        return self.databento_api_key_opra or self.databento_api_key

    @property
    def globex_api_key(self) -> str:
        """API key used to authenticate against GLBX.MDP3 (CME futures).

        Falls back to the legacy ``DATABENTO_API_KEY``.
        """
        return self.databento_api_key_globex or self.databento_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
