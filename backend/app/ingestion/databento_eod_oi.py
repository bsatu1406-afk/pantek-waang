"""End-of-day Open Interest ingestion from Databento OPRA Pillar.

OPRA's ``statistics`` schema publishes ``open_interest`` records once per
trading day. This module pulls them at startup (and again daily, scheduled
by ``app.processing.scheduler``) and upserts them into ``eod_open_interest``.

The compute pipeline reads from this table to back-fill missing live OI when
producing GEX-by-OI and OI-walls — see ``processing.loader``.

This task is **best-effort**:
* If neither ``DATABENTO_API_KEY_OPRA`` nor ``DATABENTO_API_KEY`` is set,
  the function logs a warning and
  returns ``0``.
* If the Databento subscription does not include statistics on OPRA Pillar,
  the API will respond with a 422 / 404; we log the warning and return ``0``.
* Any other transient error is logged and swallowed — the rest of the
  application stays online.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from app.config import get_settings
from app.core.logging import get_logger
from app.db.models import EodOpenInterest
from app.db.session import get_session_factory

logger = get_logger(__name__)


DATASET = "OPRA.PILLAR"
PARENT_SUFFIX = ".OPT"

# Databento STAT_TYPE code for "open_interest" on OPRA Pillar.
# Documented at https://databento.com/docs/standards-and-conventions/common-fields-enums-types#stat-type.
STAT_TYPE_OPEN_INTEREST = 9


def _parent_symbol(underlying: str) -> str:
    return f"{underlying.upper()}{PARENT_SUFFIX}"


def _normalize_oi_row(underlying: str, raw: pd.Series) -> dict | None:
    """Best-effort conversion of a Databento statistics row into our shape."""
    expiry = raw.get("expiration") or raw.get("expiration_date")
    strike_raw = raw.get("strike_price")
    option_type = raw.get("instrument_class") or raw.get("option_type")
    quantity = raw.get("quantity")
    if (
        expiry is None
        or strike_raw is None
        or option_type is None
        or quantity is None
        or pd.isna(quantity)
    ):
        return None

    try:
        strike = float(strike_raw)
        if strike > 1e6:
            strike /= 1e9
    except (TypeError, ValueError):
        return None

    opt = str(option_type).upper()
    opt_char = "C" if opt in ("C", "CALL") else "P" if opt in ("P", "PUT") else None
    if opt_char is None:
        return None

    try:
        expiry_dt = pd.Timestamp(expiry).date()
    except Exception:  # noqa: BLE001
        return None

    try:
        oi_value = int(float(quantity))
    except (TypeError, ValueError):
        return None
    if oi_value < 0:
        return None

    today = datetime.now(UTC).date()
    return {
        "symbol": underlying.upper(),
        "expiration": expiry_dt,
        "strike": strike,
        "option_type": opt_char,
        "oi_date": today,
        "open_interest": oi_value,
        "updated_at": datetime.now(UTC),
    }


async def fetch_eod_oi_from_databento(
    underlying: str, *, lookback_days: int = 3
) -> list[dict]:
    """Fetch the latest EOD OI rows for ``underlying``. Returns possibly empty list."""
    settings = get_settings()
    if not settings.opra_api_key:
        logger.warning("eod_oi_skipped_no_api_key", symbol=underlying)
        return []

    try:
        import databento as db
    except ImportError:
        logger.warning("databento_import_failed_for_eod_oi")
        return []

    end = datetime.now(UTC) - timedelta(minutes=30)
    start = end - timedelta(days=lookback_days)

    parent = _parent_symbol(underlying)
    client = db.Historical(key=settings.opra_api_key)

    try:
        data = await asyncio.to_thread(
            client.timeseries.get_range,
            dataset=DATASET,
            schema="statistics",
            symbols=[parent],
            stype_in="parent",
            start=start,
            end=end,
        )
        df = await asyncio.to_thread(data.to_df)
    except Exception as exc:  # noqa: BLE001
        # Statistics may not be available on the user's subscription tier —
        # log and move on, the rest of the system continues to work.
        logger.warning(
            "eod_oi_fetch_failed",
            symbol=underlying,
            error=str(exc),
        )
        return []

    if df is None or df.empty:
        return []

    # Filter to only OI rows.
    if "stat_type" in df.columns:
        df = df[df["stat_type"] == STAT_TYPE_OPEN_INTEREST]
    if df.empty:
        return []

    out: list[dict] = []
    seen: set[tuple[str, date, float, str]] = set()
    # Iterate in descending timestamp so the freshest snapshot wins.
    sort_col = "ts_event" if "ts_event" in df.columns else df.columns[0]
    df = df.sort_values(sort_col, ascending=False)
    for _, raw in df.iterrows():
        normalized = _normalize_oi_row(underlying, raw)
        if normalized is None:
            continue
        key = (
            normalized["symbol"],
            normalized["expiration"],
            normalized["strike"],
            normalized["option_type"],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


async def upsert_eod_oi(rows: list[dict]) -> int:
    if not rows:
        return 0
    factory = get_session_factory()
    async with factory() as session:
        stmt = insert(EodOpenInterest).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "expiration", "strike", "option_type"],
            set_={
                "oi_date": stmt.excluded.oi_date,
                "open_interest": stmt.excluded.open_interest,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        await session.execute(stmt)
        await session.commit()
    return len(rows)


async def run_eod_oi_ingestion() -> int:
    """Pull EOD OI for every supported symbol. Returns total rows upserted."""
    settings = get_settings()
    total = 0
    for symbol in settings.supported_symbols:
        try:
            rows = await fetch_eod_oi_from_databento(symbol)
            inserted = await upsert_eod_oi(rows)
            total += inserted
            logger.info("eod_oi_ingested", symbol=symbol, rows=inserted)
        except Exception:  # noqa: BLE001
            logger.exception("eod_oi_ingestion_error", symbol=symbol)
    return total
