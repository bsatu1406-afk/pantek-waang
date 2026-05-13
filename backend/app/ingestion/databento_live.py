"""Databento Live ingestion with reconnect / exponential backoff.

Subscribes to OPRA Pillar definition + cmbp-1 + trades + statistics for each
configured parent symbol. Maintains an in-memory contract registry keyed by
Databento's ``instrument_id`` so we can map live updates back to (symbol,
expiry, strike, type) when writing rows to TimescaleDB.

Notes on schema choice:
    OPRA.PILLAR's quote schema is ``cmbp-1`` (consolidated MBP-1) — the
    plain ``mbp-1`` name in some older docs / examples is **not** valid
    against this dataset and gets dropped at connect time. Records in
    ``cmbp-1`` carry ``levels[0].bid_px`` / ``levels[0].ask_px`` rather
    than the legacy ``bid_px_00`` / ``ask_px_00`` flat fields.

Failure modes handled:
  - Missing API key  -> log + return without crashing
  - ``databento`` package not installed  -> log + return
  - Stream disconnect -> exponential backoff up to ``MAX_RECONNECTS``
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from app.config import get_settings
from app.core.logging import get_logger
from app.ingestion.bulk_writers import (
    BulkUpsertWriter,
    get_options_trade_writer,
)
from app.ingestion.writer import OptionsChainWriter, get_writer

logger = get_logger(__name__)


def pd_to_date(value: Any):
    """Convert a Databento expiration value (date, datetime, ns int) to ``date``."""
    from datetime import date

    import pandas as pd

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return pd.Timestamp(value).date()
    except Exception:  # noqa: BLE001
        try:
            return datetime.fromtimestamp(int(value) / 1e9, tz=UTC).date()
        except (TypeError, ValueError):
            return None

DATASET = "OPRA.PILLAR"
PARENT_SUFFIX = ".OPT"
MAX_RECONNECTS = 5
INITIAL_BACKOFF_S = 2.0
DEFAULT_SCHEMAS = ("definition", "trades", "statistics", "cmbp-1")


def _parent(symbol: str) -> str:
    return f"{symbol.upper()}{PARENT_SUFFIX}"


def _scale_price(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    # Databento commonly uses int64 fixed-point with 1e9 scaling.
    if abs(f) > 1e6:
        f /= 1e9
    return f


def _coerce_str(value: Any) -> str | None:
    """Coerce gateway-supplied identifiers (often int) to a Text-friendly str.

    asyncpg refuses to insert an int into a ``Text`` column, which would
    silently fail an entire ``options_trades`` batch.
    """
    if value is None:
        return None
    return str(value)


class DatabentoLiveIngester:
    """Live OPRA Pillar ingester with an in-memory contract registry."""

    def __init__(
        self,
        writer: OptionsChainWriter | None = None,
        *,
        trade_writer: BulkUpsertWriter | None = None,
    ) -> None:
        self._settings = get_settings()
        self._writer = writer or get_writer()
        self._trade_writer = trade_writer or get_options_trade_writer()
        # instrument_id -> {symbol, expiration, strike, option_type}
        self._registry: dict[int, dict[str, Any]] = {}
        # instrument_id -> latest oi / volume / underlying_price etc.
        self._state: dict[int, dict[str, Any]] = {}
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        # Schemas mutated at runtime as the gateway tells us which are unsupported.
        self._schemas: list[str] = list(DEFAULT_SCHEMAS)
        # Telemetry: count records by type since the last log dump.
        self._record_counts: dict[str, int] = {}
        self._last_telemetry_at = 0.0
        self._telemetry_interval_s = 30.0
        # Persistent diagnostics (never cleared) so the admin Data Inspector
        # can surface them long after the periodic log dump.
        self._cumulative_record_counts: dict[str, int] = {}
        self._dropped_schemas: list[str] = []
        self._sample_record_attrs: dict[str, dict[str, Any]] = {}
        self._first_record_at: datetime | None = None
        self._last_record_at: datetime | None = None
        self._connection_attempts: int = 0
        self._last_error: str | None = None
        # Captured gateway frames (SystemMsg / ErrorMsg). Most recent first.
        self._system_messages: list[str] = []
        self._error_messages: list[str] = []

    # ── Diagnostics surface (read by /admin/inspector) ──────────────────────
    def diagnostics(self) -> dict[str, Any]:
        return {
            "registry_size": len(self._registry),
            "schemas_active": list(self._schemas),
            "schemas_dropped": list(self._dropped_schemas),
            "record_counts": dict(self._cumulative_record_counts),
            "sample_record_attrs": dict(self._sample_record_attrs),
            "first_record_at": (
                self._first_record_at.isoformat() if self._first_record_at else None
            ),
            "last_record_at": (
                self._last_record_at.isoformat() if self._last_record_at else None
            ),
            "connection_attempts": self._connection_attempts,
            "last_error": self._last_error,
            "system_messages": list(self._system_messages),
            "error_messages": list(self._error_messages),
            "supported_symbols": self._settings.supported_symbols,
        }

    # ── Public API ──────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run_with_reconnect(), name="databento_live")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    # ── Internals ───────────────────────────────────────────────────────────
    async def _run_with_reconnect(self) -> None:
        if self._settings.disable_live_ingestion:
            logger.info("live_ingestion_disabled")
            return
        if not self._settings.opra_api_key:
            logger.warning("live_ingestion_skipped_no_api_key")
            return

        try:
            import databento as db  # noqa: F401
        except ImportError:
            logger.warning("databento_import_failed_for_live")
            return

        # Bootstrap the contract registry. The Live ``definition`` schema only
        # emits records when contracts change, so without this we can't map any
        # incoming trade/MBP/statistics records back to (strike, expiry, type).
        await self._bootstrap_registry()

        backoff = INITIAL_BACKOFF_S
        attempt = 0
        while attempt < MAX_RECONNECTS:
            if self._stop.is_set():
                return
            attempt += 1
            self._connection_attempts = attempt
            try:
                logger.info(
                    "live_ingestion_connecting", attempt=attempt, schemas=self._schemas
                )
                await self._stream_once()
                # Clean exit (stream closed naturally) — try again with reset backoff.
                backoff = INITIAL_BACKOFF_S
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                self._last_error = msg[:500]
                dropped = self._drop_unsupported_schema(msg)
                if dropped:
                    self._dropped_schemas.append(dropped)
                    logger.warning(
                        "live_dropping_unsupported_schema",
                        dropped=dropped,
                        remaining=self._schemas,
                    )
                    if not self._schemas:
                        logger.error("live_ingestion_no_schemas_left")
                        return
                    # Don't burn an attempt for a config-time fix.
                    attempt -= 1
                    continue
                logger.exception("live_ingestion_stream_failed", error=msg)
                if attempt >= MAX_RECONNECTS:
                    logger.error("live_ingestion_giving_up", attempts=attempt)
                    return
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _bootstrap_registry(self) -> None:
        """Populate ``self._registry`` from the Historical definition schema.

        We pull the latest available trading day's definitions for each
        configured parent symbol and load them into the registry so live
        trade / statistics records can be mapped back to a contract.
        """
        try:
            import databento as db
        except ImportError:
            return

        from datetime import timedelta

        client = db.Historical(key=self._settings.opra_api_key)
        # Use a 30-min buffer below "now" to stay safely inside the published
        # data window (mirrors ``databento_historical.py``).
        end = datetime.now(UTC) - timedelta(minutes=30)
        start = end - timedelta(days=2)
        loaded = 0
        for underlying in self._settings.supported_symbols:
            parent = f"{underlying.upper()}{PARENT_SUFFIX}"
            try:
                data = await asyncio.to_thread(
                    client.timeseries.get_range,
                    dataset=DATASET,
                    schema="definition",
                    symbols=[parent],
                    stype_in="parent",
                    start=start,
                    end=end,
                )
                df = await asyncio.to_thread(data.to_df)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "live_registry_bootstrap_failed",
                    symbol=underlying,
                    error=str(exc),
                )
                continue
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                instrument_id = row.get("instrument_id")
                strike = _scale_price(row.get("strike_price"))
                expiry = row.get("expiration") or row.get("expiration_date")
                opt = row.get("instrument_class") or row.get("option_type")
                if instrument_id is None or strike is None or expiry is None or opt is None:
                    continue
                try:
                    instrument_id = int(instrument_id)
                except (TypeError, ValueError):
                    continue
                opt_u = str(opt).upper()
                if opt_u in ("C", "CALL"):
                    opt_char = "C"
                elif opt_u in ("P", "PUT"):
                    opt_char = "P"
                else:
                    continue
                try:
                    expiry_d = pd_to_date(expiry)
                except Exception:  # noqa: BLE001
                    continue
                if expiry_d is None:
                    continue
                self._registry[instrument_id] = {
                    "symbol": underlying.upper(),
                    "expiration": expiry_d,
                    "strike": strike,
                    "option_type": opt_char,
                }
                loaded += 1
        logger.info("live_registry_bootstrapped", contracts=loaded)

    def _drop_unsupported_schema(self, error_message: str) -> str | None:
        """If the error names a schema, drop it from the active list and return its name."""
        for schema in list(self._schemas):
            needle = f"{schema} schema not supported"
            if needle in error_message:
                self._schemas.remove(schema)
                return schema
        return None

    async def _stream_once(self) -> None:
        import databento as db

        client = db.Live(key=self._settings.opra_api_key)
        for symbol in self._settings.supported_symbols:
            parent = _parent(symbol)
            for schema in self._schemas:
                client.subscribe(
                    dataset=DATASET,
                    schema=schema,
                    symbols=[parent],
                    stype_in="parent",
                )

        async for record in client:
            if self._stop.is_set():
                break
            try:
                await self._handle_record(record)
            except Exception:  # noqa: BLE001
                logger.exception("record_processing_error")

    # ── Record handlers ─────────────────────────────────────────────────────
    async def _handle_record(self, record: Any) -> None:
        rtype = type(record).__name__
        self._record_counts[rtype] = self._record_counts.get(rtype, 0) + 1
        self._cumulative_record_counts[rtype] = (
            self._cumulative_record_counts.get(rtype, 0) + 1
        )
        if self._first_record_at is None:
            self._first_record_at = datetime.now(UTC)
        self._last_record_at = datetime.now(UTC)
        # Capture a snapshot of the first record of each type — invaluable for
        # diagnosing why fields like bid/ask aren't being parsed (e.g. Databento
        # SDK normalised the field name differently than expected).
        if rtype not in self._sample_record_attrs:
            attrs: dict[str, Any] = {}
            for name in dir(record):
                if name.startswith("_"):
                    continue
                try:
                    val = getattr(record, name)
                except Exception:  # noqa: BLE001
                    continue
                if callable(val):
                    continue
                # Stringify exotic types so the JSON encoder is happy.
                try:
                    if isinstance(val, str | int | float | bool | type(None)):
                        attrs[name] = val
                    elif isinstance(val, list | tuple):
                        attrs[name] = [str(x)[:200] for x in val[:3]]
                    else:
                        attrs[name] = str(val)[:200]
                except Exception:  # noqa: BLE001
                    attrs[name] = "<unserialisable>"
            self._sample_record_attrs[rtype] = attrs
        self._maybe_log_telemetry()

        if "Definition" in rtype:
            await self._handle_definition(record)
        elif "CMBP" in rtype or "Cmbp" in rtype or "Consolidated" in rtype:
            await self._handle_cmbp(record)
        elif "MBP" in rtype or "Mbp" in rtype:
            await self._handle_mbp(record)
        elif "Trade" in rtype:
            await self._handle_trade(record)
        elif "Stat" in rtype:
            await self._handle_statistics(record)
        elif "Error" in rtype:
            msg = (
                getattr(record, "err", None)
                or getattr(record, "msg", None)
                or str(record)
            )
            text = str(msg)[:300]
            self._last_error = text
            self._error_messages.append(text)
            self._error_messages = self._error_messages[-10:]
            logger.warning("live_gateway_error", msg=text)
        elif "System" in rtype:
            msg = getattr(record, "msg", None) or str(record)
            text = str(msg)[:300]
            self._system_messages.append(text)
            self._system_messages = self._system_messages[-10:]
            logger.info("live_gateway_system", msg=text)
        # Ignore other message types (symbol-mapping etc.).

    def _maybe_log_telemetry(self) -> None:
        loop = asyncio.get_event_loop()
        now = loop.time()
        if now - self._last_telemetry_at < self._telemetry_interval_s:
            return
        self._last_telemetry_at = now
        if not self._record_counts:
            return
        logger.info(
            "live_ingestion_telemetry",
            record_counts=dict(self._record_counts),
            registry_size=len(self._registry),
        )
        self._record_counts.clear()

    async def _handle_trade(self, record: Any) -> None:
        instrument_id = getattr(record, "instrument_id", None)
        contract = self._registry.get(instrument_id) if instrument_id else None
        if contract is None:
            # Diagnostic: log the first few unmatched ids so we can debug.
            self._unmatched_count = getattr(self, "_unmatched_count", 0) + 1
            if self._unmatched_count <= 5:
                logger.info(
                    "live_trade_unmatched_instrument",
                    instrument_id=instrument_id,
                    raw_symbol=getattr(record, "raw_symbol", None),
                    registry_sample=list(self._registry.keys())[:3],
                )
            return

        price = _scale_price(getattr(record, "price", None))
        size = getattr(record, "size", None)

        state = self._state.setdefault(instrument_id, {})
        if price is not None:
            state["last_price"] = price
        if size is not None:
            try:
                state["volume"] = (state.get("volume") or 0) + int(size)
            except (TypeError, ValueError):
                pass

        await self._emit_row(instrument_id, record)
        await self._emit_trade(instrument_id, record, price=price, size=size)

    async def _emit_trade(
        self,
        instrument_id: int,
        record: Any,
        *,
        price: float | None,
        size: Any,
    ) -> None:
        """Persist the trade-tape row used by Lee-Ready / HIRO / flow events."""
        if price is None or size is None:
            return
        try:
            size_int = int(size)
        except (TypeError, ValueError):
            return
        contract = self._registry[instrument_id]
        state = self._state.get(instrument_id, {})

        seq = getattr(record, "sequence", None) or getattr(record, "ts_event", 0)
        try:
            seq_int = int(seq)
        except (TypeError, ValueError):
            seq_int = 0

        # Quote-rule classifier inline (kept simple — full Lee-Ready with
        # tick fallback runs in the pipeline against the persisted rows).
        bid = state.get("bid")
        ask = state.get("ask")
        side: int | None = None
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            if price > mid:
                side = 1
            elif price < mid:
                side = -1
            else:
                side = 0

        signed_premium: float | None = None
        if side is not None and side != 0:
            # Dealer side is the opposite of customer.
            signed_premium = -side * size_int * price * 100.0

        await self._trade_writer.add({
            "ts": self._record_ts(record),
            "symbol": contract["symbol"],
            "expiration": contract["expiration"],
            "strike": contract["strike"],
            "option_type": contract["option_type"],
            "seq": seq_int,
            "price": price,
            "size": size_int,
            "bid": bid,
            "ask": ask,
            "exchange": _coerce_str(getattr(record, "publisher_id", None)),
            "side": side,
            "signed_premium": signed_premium,
        })

    async def _handle_definition(self, record: Any) -> None:
        instrument_id = getattr(record, "instrument_id", None)
        raw_symbol = getattr(record, "raw_symbol", None) or getattr(record, "symbol", None)
        expiration = getattr(record, "expiration", None) or getattr(record, "expiration_date", None)
        strike = _scale_price(getattr(record, "strike_price", None))
        instrument_class = getattr(record, "instrument_class", None) or getattr(
            record, "option_type", None
        )

        if instrument_id is None or expiration is None or strike is None or instrument_class is None:
            return

        underlying = self._underlying_for_raw_symbol(raw_symbol)
        if underlying is None:
            return

        opt = str(instrument_class).upper()
        if opt in ("C", "CALL"):
            opt_char = "C"
        elif opt in ("P", "PUT"):
            opt_char = "P"
        else:
            return

        try:
            from datetime import date

            if isinstance(expiration, datetime):
                expiry_date = expiration.date()
            elif isinstance(expiration, date):
                expiry_date = expiration
            else:
                expiry_date = datetime.fromtimestamp(int(expiration) / 1e9, tz=UTC).date()
        except Exception:  # noqa: BLE001
            return

        self._registry[instrument_id] = {
            "symbol": underlying,
            "expiration": expiry_date,
            "strike": strike,
            "option_type": opt_char,
        }

    def _underlying_for_raw_symbol(self, raw_symbol: str | None) -> str | None:
        if not raw_symbol:
            return None
        for sym in self._settings.supported_symbols:
            if raw_symbol.upper().startswith(sym.upper()):
                return sym.upper()
        return None

    async def _handle_mbp(self, record: Any) -> None:
        """Legacy ``mbp-1`` schema (flat ``bid_px_00`` / ``ask_px_00``).

        Kept for non-OPRA datasets that still use the older format. OPRA
        Pillar live records use ``cmbp-1`` and are dispatched to
        :meth:`_handle_cmbp` instead.
        """
        instrument_id = getattr(record, "instrument_id", None)
        contract = self._registry.get(instrument_id) if instrument_id else None
        if contract is None:
            return

        bid = _scale_price(getattr(record, "bid_px_00", None))
        ask = _scale_price(getattr(record, "ask_px_00", None))
        last_price = _scale_price(
            getattr(record, "last_price", None) or getattr(record, "price", None)
        )

        state = self._state.setdefault(instrument_id, {})
        if bid is not None:
            state["bid"] = bid
        if ask is not None:
            state["ask"] = ask
        if last_price is not None:
            state["last_price"] = last_price

        await self._emit_row(instrument_id, record)

    async def _handle_cmbp(self, record: Any) -> None:
        """Consolidated MBP-1 (OPRA Pillar quote schema).

        ``record.levels`` is a list of ``ConsolidatedBidAskPair`` entries with
        ``bid_px`` / ``ask_px`` / ``bid_sz`` / ``ask_sz`` fields. Top of book
        (level 0) is what we want for mid-price / spread metrics.
        """
        instrument_id = getattr(record, "instrument_id", None)
        contract = self._registry.get(instrument_id) if instrument_id else None
        if contract is None:
            return

        levels = getattr(record, "levels", None)
        bid: float | None = None
        ask: float | None = None
        if levels:
            try:
                top = levels[0]
            except (IndexError, TypeError):
                top = None
            if top is not None:
                bid = _scale_price(getattr(top, "bid_px", None))
                ask = _scale_price(getattr(top, "ask_px", None))

        # Fallback to legacy flat fields just in case the SDK normalises
        # differently in the future.
        if bid is None:
            bid = _scale_price(getattr(record, "bid_px_00", None))
        if ask is None:
            ask = _scale_price(getattr(record, "ask_px_00", None))

        state = self._state.setdefault(instrument_id, {})
        if bid is not None:
            state["bid"] = bid
        if ask is not None:
            state["ask"] = ask

        await self._emit_row(instrument_id, record)

    async def _handle_statistics(self, record: Any) -> None:
        instrument_id = getattr(record, "instrument_id", None)
        contract = self._registry.get(instrument_id) if instrument_id else None
        if contract is None:
            return

        stat_type = getattr(record, "stat_type", None)
        value = getattr(record, "quantity", None) or getattr(record, "value", None)

        state = self._state.setdefault(instrument_id, {})
        # Databento stat_type values: 9 -> open interest, 10 -> cumulative volume (approximate).
        if stat_type in (9, "open_interest", "OPEN_INTEREST"):
            try:
                state["oi"] = int(value)
            except (TypeError, ValueError):
                pass
        elif stat_type in (10, "cumulative_volume", "CUMULATIVE_VOLUME", "VOLUME"):
            try:
                state["volume"] = int(value)
            except (TypeError, ValueError):
                pass

        await self._emit_row(instrument_id, record)

    async def _emit_row(self, instrument_id: int, record: Any) -> None:
        contract = self._registry[instrument_id]
        state = self._state.get(instrument_id, {})
        ts = self._record_ts(record)
        row = {
            "ts": ts,
            "symbol": contract["symbol"],
            "expiration": contract["expiration"],
            "strike": contract["strike"],
            "option_type": contract["option_type"],
            "oi": state.get("oi"),
            "volume": state.get("volume"),
            "iv": state.get("iv"),
            "delta": state.get("delta"),
            "gamma": state.get("gamma"),
            "last_price": state.get("last_price"),
            "bid": state.get("bid"),
            "ask": state.get("ask"),
            "underlying_price": state.get("underlying_price"),
        }
        await self._writer.add(row)

    @staticmethod
    def _record_ts(record: Any) -> datetime:
        ts_event = getattr(record, "ts_event", None)
        if ts_event is None:
            return datetime.now(UTC)
        try:
            # Databento ts_event is nanoseconds since epoch.
            return datetime.fromtimestamp(int(ts_event) / 1e9, tz=UTC)
        except (TypeError, ValueError, OverflowError):
            return datetime.now(UTC)


_ingester: DatabentoLiveIngester | None = None


def get_live_ingester() -> DatabentoLiveIngester:
    global _ingester
    if _ingester is None:
        _ingester = DatabentoLiveIngester()
    return _ingester
