"""HIRO time-series feed (Agent 5 — streaming API).

Exposes the cumulative signed-premium HIRO series persisted by
:mod:`app.processing.flow_pipeline`. Each pipeline tick rewrites the latest
``HIRO`` row with the full series for the trailing window (default 1 hour
of 1-minute buckets) in ``extra_json['series']``; this endpoint slices that
series by ``since`` and optionally re-aggregates into wider buckets.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, Path, Query, Request
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import limiter, require_symbol_access
from app.config import get_settings
from app.db.models import ComputedMetric
from app.db.session import get_db

router = APIRouter()


_SYMBOL_PATTERN = r"^[A-Za-z0-9_.-]+$"

_BUCKET_TO_PANDAS = {"1m": "1min", "5m": "5min", "15m": "15min"}


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        ts = pd.to_datetime(value, utc=True)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def _rebucket(series: list[dict[str, Any]], target: str) -> list[dict[str, Any]]:
    """Re-aggregate the persisted 1-minute series into ``target`` buckets."""
    if not series or target == "1min":
        return series
    rows: list[dict[str, Any]] = []
    for entry in series:
        ts = _parse_ts(entry.get("ts"))
        if ts is None:
            continue
        rows.append(
            {
                "ts": ts,
                "call_premium": float(entry.get("call_premium") or 0.0),
                "put_premium": float(entry.get("put_premium") or 0.0),
                "net_premium": float(entry.get("net_premium") or 0.0),
            }
        )
    if not rows:
        return []
    df = pd.DataFrame(rows).set_index("ts").sort_index()
    agg = df.resample(target).sum()
    agg["cumulative"] = agg["net_premium"].cumsum()
    return [
        {
            "ts": ts.isoformat(),
            "call_premium": float(row["call_premium"]),
            "put_premium": float(row["put_premium"]),
            "net_premium": float(row["net_premium"]),
            "cumulative": float(row["cumulative"]),
        }
        for ts, row in agg.iterrows()
    ]


@router.get("/v1/{symbol}/hiro")
@limiter.limit(lambda: f"{get_settings().rate_limit_per_minute}/minute")
async def get_hiro(
    request: Request,  # noqa: ARG001
    symbol: str = Path(..., min_length=1, max_length=20, pattern=_SYMBOL_PATTERN),
    bucket: str = Query("1m", pattern="^(1m|5m|15m)$"),
    since: datetime | None = Query(default=None),
    session: AsyncSession = Depends(get_db),
    _api_key=Depends(require_symbol_access()),
) -> dict[str, Any]:
    """Return the HIRO cumulative signed-premium series for ``symbol``.

    The persisted series uses 1-minute buckets; ``5m`` / ``15m`` are
    re-aggregated server-side. ``since`` defaults to the last 1 hour.
    """
    sym_u = symbol.upper()
    target_bucket = _BUCKET_TO_PANDAS[bucket]
    if since is None:
        since = datetime.now(UTC) - timedelta(hours=1)
    elif since.tzinfo is None:
        since = since.replace(tzinfo=UTC)

    latest_q = (
        select(ComputedMetric)
        .where(ComputedMetric.symbol == sym_u, ComputedMetric.metric_type == "HIRO")
        .order_by(desc(ComputedMetric.ts))
        .limit(1)
    )
    row = (await session.execute(latest_q)).scalar_one_or_none()
    if row is None:
        return {
            "symbol": sym_u,
            "bucket": bucket,
            "since": since.isoformat(),
            "cumulative": 0.0,
            "series": [],
        }

    raw_series = list((row.extra_json or {}).get("series") or [])
    # Filter by ``since`` (1-minute granularity). Items missing a parseable
    # timestamp are dropped — they cannot be ordered or filtered.
    filtered: list[dict[str, Any]] = []
    for entry in raw_series:
        ts = _parse_ts(entry.get("ts"))
        if ts is None or ts < since:
            continue
        filtered.append({**entry, "ts": ts.isoformat()})

    if target_bucket != "1min":
        filtered = _rebucket(filtered, target_bucket)
    else:
        filtered = sorted(filtered, key=lambda x: str(x.get("ts", "")))

    cumulative = filtered[-1]["cumulative"] if filtered else float(row.value or 0.0)
    return {
        "symbol": sym_u,
        "bucket": bucket,
        "since": since.isoformat(),
        "cumulative": float(cumulative),
        "series": filtered,
    }
