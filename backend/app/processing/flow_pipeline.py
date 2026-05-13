"""Flow-side post-processing: detect events and compute HIRO / basis from
the trade tapes and persist them.

Designed to run inside the same scheduler tick as the chain pipeline. We
load the last ``window_minutes`` minutes of options-trades from
``options_trades`` (Lee-Ready already pre-classified at ingest time) and
of futures ticks from ``futures_ticks``, then:

* Detect SWEEP / BLOCK / UOA on the options tape and persist into
  ``flow_events``.
* Compute HIRO over a configurable bucket and persist as
  ``computed_metrics`` rows of ``HIRO_*`` types.
* Compute SPX-ES basis from the latest futures tick + chain underlying
  price and persist as ``BASIS_SPX_ES``.
* Compute the daily ES volume profile and persist as ``VOLUME_PROFILE_ES``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.config import get_settings
from app.core.logging import get_logger
from app.db.models import (
    ComputedMetric,
    ContractAdv,
    FuturesTick,
    OptionsChain,
    OptionsTrade,
)
from app.db.session import get_session_factory
from app.ingestion.bulk_writers import get_flow_event_writer
from app.processing.basis import compute_basis
from app.processing.flow_events import FlowEventConfig, detect_flow_events
from app.processing.hiro import compute_hiro
from app.processing.volume_profile import compute_volume_profile

logger = get_logger(__name__)


SENTINEL_EXPIRY = pd.Timestamp("1970-01-01").date()


async def run_flow_pipeline(
    *,
    symbol: str,
    futures_symbols: Sequence[str] = ("ES", "NQ"),
    window_minutes: int = 60,
    hiro_bucket: str = "1min",
) -> dict:
    """Run flow-side analytics for ``symbol`` and return a summary dict.

    Returns ``{events: <int>, hiro_buckets: <int>, basis: <BasisSnapshot>,
    profile_bins: <int>}``.
    """
    factory = get_session_factory()
    now = datetime.now(UTC)
    window_start = now - timedelta(minutes=window_minutes)

    async with factory() as session:
        opt_trades = await _load_options_trades(session, symbol=symbol,
                                                start=window_start, end=now)
        fut_trades, fut_last = await _load_futures_trades(session,
                                                          symbols=futures_symbols,
                                                          start=window_start,
                                                          end=now)
        chain_underlying = await _load_chain_underlying(session, symbol=symbol)
        contract_adv = await _load_contract_adv(session, symbol=symbol)
        contract_oi = await _load_contract_oi(session, symbol=symbol)

    # ── Flow events ──────────────────────────────────────────────────────
    cfg = FlowEventConfig.from_settings(get_settings())
    events = detect_flow_events(
        opt_trades,
        contract_adv=contract_adv,
        contract_oi=contract_oi,
        config=cfg,
    )
    if events:
        writer = get_flow_event_writer()
        rows = [
            {
                "ts": pd.Timestamp(e["ts"]).to_pydatetime(),
                "symbol": e["symbol"],
                "expiration": pd.Timestamp(e["expiration"]).date(),
                "strike": float(e["strike"]),
                "option_type": e["option_type"],
                "event_type": e["event_type"],
                "side": int(e.get("side", 0)),
                "size": int(e.get("size", 0)),
                "price": e.get("price"),
                "legs": int(e.get("legs", 1)),
                "venues": list(e.get("venues") or []),
                "meta": e.get("meta"),
            }
            for e in events
        ]
        await writer.add_many(rows)

    # ── HIRO ─────────────────────────────────────────────────────────────
    hiro = compute_hiro(opt_trades, bucket=hiro_bucket)

    # ── Basis ────────────────────────────────────────────────────────────
    es_last = fut_last.get("ES") if fut_last else None
    basis = compute_basis(spot=chain_underlying, futures=es_last)

    # ── Volume profile (ES) ──────────────────────────────────────────────
    es_trades = (
        fut_trades[fut_trades["symbol_root"] == "ES"] if not fut_trades.empty else fut_trades
    )
    profile = compute_volume_profile(es_trades, bin_size=0.25)

    # ── Persist all derived metrics into computed_metrics ────────────────
    metric_rows: list[dict] = []
    if hiro.series:
        metric_rows.append(
            {
                "ts": now,
                "symbol": symbol,
                "metric_type": "HIRO",
                "strike": 0,
                "expiration": SENTINEL_EXPIRY,
                "computed_at": now,
                "value": hiro.cumulative,
                "extra_json": {
                    "bucket_size": hiro.bucket_size,
                    "series": hiro.series,
                    "cumulative": hiro.cumulative,
                },
            }
        )
    if basis.basis is not None:
        metric_rows.append(
            {
                "ts": now,
                "symbol": symbol,
                "metric_type": "BASIS_SPX_ES",
                "strike": 0,
                "expiration": SENTINEL_EXPIRY,
                "computed_at": now,
                "value": basis.basis,
                "extra_json": {
                    "spot": basis.spot,
                    "futures": basis.futures,
                    "basis": basis.basis,
                    "basis_pct": basis.basis_pct,
                },
            }
        )
    if profile.bins:
        metric_rows.append(
            {
                "ts": now,
                "symbol": "ES",
                "metric_type": "VOLUME_PROFILE_ES",
                "strike": 0,
                "expiration": SENTINEL_EXPIRY,
                "computed_at": now,
                "value": profile.poc,
                "extra_json": {
                    "bin_size": profile.bin_size,
                    "poc": profile.poc,
                    "vah": profile.vah,
                    "val": profile.val,
                    "total_volume": profile.total_volume,
                    "bins": profile.bins,
                },
            }
        )

    if metric_rows:
        async with factory() as session:
            stmt = insert(ComputedMetric).values(metric_rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["ts", "symbol", "metric_type", "strike", "expiration"],
                set_={
                    "computed_at": stmt.excluded.computed_at,
                    "value": stmt.excluded.value,
                    "extra_json": stmt.excluded.extra_json,
                },
            )
            await session.execute(stmt)
            await session.commit()

    return {
        "events": len(events),
        "hiro_buckets": len(hiro.series),
        "basis": basis.basis,
        "profile_bins": len(profile.bins),
    }


async def _load_options_trades(session, *, symbol: str, start, end) -> pd.DataFrame:
    stmt = (
        select(
            OptionsTrade.ts,
            OptionsTrade.symbol,
            OptionsTrade.expiration,
            OptionsTrade.strike,
            OptionsTrade.option_type,
            OptionsTrade.price,
            OptionsTrade.size,
            OptionsTrade.bid,
            OptionsTrade.ask,
            OptionsTrade.exchange,
            OptionsTrade.side,
        )
        .where(OptionsTrade.symbol == symbol)
        .where(OptionsTrade.ts >= start)
        .where(OptionsTrade.ts <= end)
    )
    res = await session.execute(stmt)
    rows = res.mappings().all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    for col in ("price", "size", "bid", "ask", "side", "strike"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


async def _load_futures_trades(session, *, symbols: Sequence[str], start, end):
    stmt = (
        select(
            FuturesTick.ts,
            FuturesTick.symbol,
            FuturesTick.price,
            FuturesTick.size,
            FuturesTick.aggressor,
        )
        .where(FuturesTick.ts >= start)
        .where(FuturesTick.ts <= end)
    )
    res = await session.execute(stmt)
    rows = res.mappings().all()
    if not rows:
        return pd.DataFrame(), {}
    df = pd.DataFrame([dict(r) for r in rows])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0).astype(int)
    df["side"] = df["aggressor"]
    # CME outrights are ``ESM6`` / ``NQH7`` — root is the first **two**
    # letters. ``[A-Z]+`` would greedily match ``ESH`` and drop every row
    # below; tighten the pattern so the filter actually keeps ES + NQ.
    df["symbol_root"] = df["symbol"].str.extract(r"^([A-Z]{2})").iloc[:, 0]
    df = df[df["symbol_root"].isin([s.upper() for s in symbols])]
    last: dict[str, float] = {}
    if not df.empty:
        recent = df.sort_values("ts").groupby("symbol_root").tail(1)
        for _, r in recent.iterrows():
            try:
                last[str(r["symbol_root"])] = float(r["price"])
            except (TypeError, ValueError):
                continue
    return df, last


async def _load_contract_adv(session, *, symbol: str) -> pd.DataFrame | None:
    """Fetch the trailing-ADV table for a single symbol.

    Returned columns: ``symbol``, ``expiration``, ``strike``,
    ``option_type``, ``avg_daily_volume`` — the exact shape expected by
    :func:`detect_flow_events`. ``None`` if no rows are available, which
    triggers the OI / absolute fallbacks downstream.
    """
    stmt = (
        select(
            ContractAdv.symbol,
            ContractAdv.expiration,
            ContractAdv.strike,
            ContractAdv.option_type,
            ContractAdv.avg_daily_volume,
        )
        .where(ContractAdv.symbol == symbol)
    )
    res = await session.execute(stmt)
    rows = res.mappings().all()
    if not rows:
        return None
    df = pd.DataFrame([dict(r) for r in rows])
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["avg_daily_volume"] = pd.to_numeric(df["avg_daily_volume"], errors="coerce")
    return df


async def _load_contract_oi(session, *, symbol: str) -> pd.DataFrame | None:
    """Fetch the most-recent open-interest snapshot per contract.

    Used as a secondary UOA fallback when no trailing-ADV row exists for
    a contract.
    """
    stmt = (
        select(
            OptionsChain.symbol,
            OptionsChain.expiration,
            OptionsChain.strike,
            OptionsChain.option_type,
            OptionsChain.oi,
        )
        .where(OptionsChain.symbol == symbol)
        .where(OptionsChain.oi.is_not(None))
        .order_by(OptionsChain.ts.desc())
    )
    res = await session.execute(stmt)
    rows = res.mappings().all()
    if not rows:
        return None
    df = pd.DataFrame([dict(r) for r in rows])
    df = df.drop_duplicates(
        subset=["symbol", "expiration", "strike", "option_type"], keep="first"
    )
    df = df.rename(columns={"oi": "open_interest"})
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")
    return df


async def _load_chain_underlying(session, *, symbol: str) -> float | None:
    stmt = (
        select(OptionsChain.underlying_price)
        .where(OptionsChain.symbol == symbol)
        .where(OptionsChain.underlying_price.is_not(None))
        .order_by(OptionsChain.ts.desc())
        .limit(1)
    )
    res = await session.execute(stmt)
    row = res.first()
    if row is None or row[0] is None:
        return None
    return float(row[0])
