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
