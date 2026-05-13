"""Historical backfill of options chain data from Databento.

Pulls the last ``HISTORICAL_BACKFILL_DAYS`` days of OPRA Pillar data for each
configured parent symbol and writes normalized rows into ``options_chain``.

Designed to fail gracefully when the Databento API key is missing or the
schemas are not available — startup proceeds and live ingestion is still attempted.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta

import pandas as pd

from app.config import get_settings
from app.core.logging import get_logger
from app.ingestion.writer import OptionsChainWriter, get_writer

logger = get_logger(__name__)


# Databento parent symbology suffix for OPRA options.
PARENT_SUFFIX = ".OPT"
DATASET = "OPRA.PILLAR"

_AVAILABLE_END_RE = re.compile(
    r"available up to '([0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9:.+\-]+)'"
)


def _parent_symbol(underlying: str) -> str:
    return f"{underlying.upper()}{PARENT_SUFFIX}"


def _parse_available_end(message: str) -> datetime | None:
    """Best-effort parse of ``available up to '<iso>'`` from a Databento 422 error."""
    m = _AVAILABLE_END_RE.search(message)
    if not m:
        return None
    raw = m.group(1).replace(" ", "T")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _normalize_definition_row(
    underlying: str, raw: pd.Series, ts: datetime
) -> dict | None:
    """Convert a Databento ``definition`` record into an options_chain row stub."""
    expiry = raw.get("expiration") or raw.get("expiration_date")
    strike_raw = raw.get("strike_price")
    option_type = raw.get("instrument_class") or raw.get("option_type")
    if expiry is None or strike_raw is None or option_type is None:
        return None
    try:
        # Databento often uses fixed-point integers scaled by 1e9 for prices.
        strike = float(strike_raw)
        if strike > 1e6:
            strike /= 1e9
    except (TypeError, ValueError):
        return None

    opt = str(option_type).upper()
    if opt in ("C", "CALL"):
        opt_char = "C"
    elif opt in ("P", "PUT"):
        opt_char = "P"
    else:
        return None

    try:
        expiry_dt = pd.Timestamp(expiry).date()
    except Exception:  # noqa: BLE001
        return None

    return {
        "ts": ts,
        "symbol": underlying.upper(),
        "expiration": expiry_dt,
        "strike": strike,
        "option_type": opt_char,
        "oi": None,
        "volume": None,
        "iv": None,
        "delta": None,
        "gamma": None,
        "last_price": None,
        "bid": None,
        "ask": None,
        "underlying_price": None,
    }


async def run_historical_backfill(writer: OptionsChainWriter | None = None) -> int:
    """Best-effort historical backfill. Returns total rows written across symbols."""
    settings = get_settings()
    if settings.disable_historical_backfill:
        logger.info("historical_backfill_disabled")
        return 0
    if not settings.opra_api_key:
        logger.warning("historical_backfill_skipped_no_api_key")
        return 0

    try:
        import databento as db
    except ImportError:
        logger.warning("databento_import_failed_for_backfill")
        return 0

    writer = writer or get_writer()
    # Use a generous buffer below "now" because Databento publishes historical
    # data with a ~15 minute lag. Without the buffer we routinely get
    # ``422 data_end_after_available_end``.
    end = datetime.now(UTC) - timedelta(minutes=30)
    start = end - timedelta(days=settings.historical_backfill_days)

    client = db.Historical(key=settings.opra_api_key)
    total_rows = 0
    for underlying in settings.supported_symbols:
        parent = _parent_symbol(underlying)
        df = None
        symbol_end = end
        # Retry once if Databento rejects the end as too recent: parse the
        # available_end from the error and replay with that as the cutoff.
        for retry in range(2):
            try:
                data = await asyncio.to_thread(
                    client.timeseries.get_range,
                    dataset=DATASET,
                    schema="definition",
                    symbols=[parent],
                    stype_in="parent",
                    start=start,
                    end=symbol_end,
                )
                df = await asyncio.to_thread(data.to_df)
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if retry == 0 and "data_end_after_available_end" in msg:
                    parsed = _parse_available_end(msg)
                    if parsed is not None and parsed > start:
                        logger.info(
                            "historical_backfill_retry_with_available_end",
                            symbol=underlying,
                            available_end=parsed.isoformat(),
                        )
                        symbol_end = parsed
                        continue
                logger.warning(
                    "historical_backfill_symbol_failed",
                    symbol=underlying,
                    error=msg,
                )
                df = None
                break
        if df is None:
            continue
        if df.empty:
            logger.info("historical_backfill_empty", symbol=underlying)
            continue

        ts = end
        for _, raw in df.iterrows():
            row = _normalize_definition_row(underlying, raw, ts)
            if row is None:
                continue
            await writer.add(row)
            total_rows += 1

    await writer.flush()
    logger.info("historical_backfill_complete", rows=total_rows)
    return total_rows
