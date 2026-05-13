"""Detect notable option flow events: sweeps, blocks, and unusual activity (UOA).

These are heuristic, not strict definitions. Conventions used:

* **Sweep** — multiple prints of the *same* option contract from the *same*
  customer side, hitting *different exchanges*, all within a short time
  window. Sweeps are also required to clear a **total premium** gate
  (``size × price × 100`` summed across legs ≥
  ``Settings.flow_sweep_min_premium``). The hallmark of an aggressive
  multi-venue taker that wants size now and is willing to pay up.
* **Block** — a single print of size ≥ ``Settings.flow_block_min_size``.
  Often pre-negotiated upstairs and reported off-book.
* **UOA** — Unusual Options Activity. We classify a contract as UOA when
  today's traded volume is meaningfully larger than what we'd expect for
  it. We try three checks in order:

  1. **Trailing ADV** (``contract_adv``) — when a row exists for the
     contract, flag if ``today_volume ≥ avg_daily_volume × uoa_volume_multiplier``.
  2. **Volume / OI ratio** (``contract_oi``) — when ADV is unavailable
     but open-interest is known, flag if
     ``today_volume / open_interest ≥ Settings.flow_uoa_vol_oi_ratio``.
  3. **Absolute fallback** — if neither ADV nor OI is known, flag if
     ``today_volume ≥ uoa_min_absolute_volume``.

Idempotency
-----------
The detector drops **exact-duplicate** trade rows (same ``ts``, contract,
side, size, price, exchange) before processing. This makes the function
safe against replaying the same trade tape window (e.g. a re-tick from
the live feed): feeding the same trade twice does NOT yield duplicate
sweep/block/UOA events.

Function :func:`detect_flow_events` returns a list of structured event
dicts ready to be inserted into a ``flow_events`` table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

CONTRACT_MULTIPLIER = 100
_CONTRACT_KEYS = ["symbol", "expiration", "strike", "option_type"]


@dataclass
class FlowEventConfig:
    """Tunable thresholds for flow-event detection.

    Defaults mirror the production settings in :class:`app.config.Settings`.
    Use :meth:`from_settings` to construct one bound to live config values.
    """

    sweep_window_ms: int = 1000
    """Time window (ms) within which legs of a sweep must arrive."""

    sweep_min_legs: int = 3
    """Minimum leg count for a cluster to be flagged as a sweep."""

    sweep_min_premium: float = 50_000.0
    """Minimum cluster premium (size × price × 100, summed) in USD."""

    block_min_size: int = 100
    """Minimum single-print size (contracts) to flag as a BLOCK."""

    uoa_volume_multiplier: float = 5.0
    """Today's volume vs trailing ADV — flagged when ≥ multiplier × ADV."""

    uoa_min_absolute_volume: int = 5000
    """Absolute volume fallback when neither ADV nor OI is known."""

    uoa_vol_oi_ratio: float = 2.0
    """Today's volume / open-interest ratio — flagged when ≥ this."""

    @classmethod
    def from_settings(cls, settings: Any = None) -> FlowEventConfig:
        """Construct from a :class:`app.config.Settings` instance.

        Pulls the three tunables exposed via env vars and leaves the
        remaining knobs at their internal defaults. Importing
        :mod:`app.config` lazily keeps this module side-effect-free for
        tests that don't want to load .env.
        """
        if settings is None:
            from app.config import get_settings

            settings = get_settings()
        return cls(
            sweep_min_premium=float(settings.flow_sweep_min_premium),
            block_min_size=int(settings.flow_block_min_size),
            uoa_vol_oi_ratio=float(settings.flow_uoa_vol_oi_ratio),
        )


def detect_flow_events(
    trades: pd.DataFrame,
    *,
    contract_adv: pd.DataFrame | None = None,
    contract_oi: pd.DataFrame | None = None,
    config: FlowEventConfig | None = None,
) -> list[dict]:
    """Return a list of detected sweep / block / UOA events.

    ``trades`` must contain at minimum::

        ts, symbol, expiration, strike, option_type,
        price, size, side, exchange

    ``contract_adv`` (optional) is a DataFrame with one row per
    ``(symbol, expiration, strike, option_type)`` and column
    ``avg_daily_volume`` (rolling N-day average traded volume).

    ``contract_oi`` (optional) is a DataFrame with one row per
    ``(symbol, expiration, strike, option_type)`` and column
    ``open_interest`` — used as the secondary UOA gate when no ADV row
    exists for a contract.

    The returned dicts have:

        event_type    : "SWEEP" | "BLOCK" | "UOA"
        ts            : timestamp of the *first* leg / the trade
        symbol, expiration, strike, option_type
        side          : +1 / -1 (customer side; 0 for UOA which is volume-only)
        size          : aggregate contracts (legs summed for sweeps)
        price         : volume-weighted average price across legs
        legs          : leg count (1 for blocks/UOA)
        venues        : sorted list of exchanges (sweeps only)
        meta          : free-form payload (used downstream by the alert engine)
    """
    cfg = config or FlowEventConfig()
    if trades.empty:
        return []

    needed = {"ts", "symbol", "expiration", "strike", "option_type",
              "price", "size", "side"}
    missing = needed.difference(trades.columns)
    if missing:
        raise KeyError(f"detect_flow_events requires {needed}; missing {missing}")

    work = trades.copy()
    work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce")
    work = work.dropna(subset=["ts"])
    work["size"] = pd.to_numeric(work["size"], errors="coerce").fillna(0).astype(int)
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["side"] = pd.to_numeric(work["side"], errors="coerce").fillna(0).astype(int)
    work = work.sort_values("ts").reset_index(drop=True)

    # ── Idempotency: drop exact-duplicate trade rows ─────────────────────
    # OPRA can replay a print; the same upstream event seen twice should
    # not double-count toward sweeps, blocks, or UOA volume.
    dedup_cols = [
        c
        for c in (
            "ts", "symbol", "expiration", "strike", "option_type",
            "price", "size", "side", "exchange",
        )
        if c in work.columns
    ]
    work = work.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    events: list[dict] = []

    # ── Sweeps (multi-venue, same customer side, short window) ───────────
    if "exchange" in work.columns and cfg.sweep_min_legs > 1:
        events.extend(_detect_sweeps(work, cfg))

    # ── Blocks (single print >= threshold) ───────────────────────────────
    block_mask = work["size"] >= cfg.block_min_size
    for _, row in work[block_mask].iterrows():
        events.append({
            "event_type": "BLOCK",
            "ts": row["ts"].isoformat(),
            "symbol": row["symbol"],
            "expiration": _isoformat_date(row["expiration"]),
            "strike": float(row["strike"]),
            "option_type": str(row["option_type"]).upper(),
            "side": int(row["side"]),
            "size": int(row["size"]),
            "price": float(row["price"]) if pd.notna(row["price"]) else None,
            "legs": 1,
            "venues": (
                [row["exchange"]]
                if "exchange" in row.index and pd.notna(row.get("exchange"))
                else []
            ),
            "meta": {"threshold": cfg.block_min_size},
        })

    # ── UOA (today's volume vs ADV / OI / absolute fallback) ─────────────
    # Suppress UOA on contracts that already produced a sweep or block on
    # this batch — the spec defines UOA as a *residual* signal for
    # contracts that don't have an obvious aggressive print pattern.
    contracts_with_event = {
        (e["symbol"], e["expiration"], e["strike"], e["option_type"])
        for e in events
    }
    events.extend(
        _detect_uoa(work, contract_adv, contract_oi, cfg, contracts_with_event)
    )

    return events


def _detect_sweeps(trades: pd.DataFrame, cfg: FlowEventConfig) -> list[dict]:
    """Group adjacent same-side same-contract prints across multiple venues."""
    out: list[dict] = []
    for _, group in trades.groupby(_CONTRACT_KEYS, sort=False):
        if len(group) < cfg.sweep_min_legs:
            continue
        g = group.sort_values("ts").reset_index(drop=True)
        i = 0
        while i < len(g):
            j = i
            cluster_side = g.loc[i, "side"]
            if cluster_side == 0:
                i += 1
                continue
            t0 = g.loc[i, "ts"]
            window_ms = cfg.sweep_window_ms
            while (
                j + 1 < len(g)
                and g.loc[j + 1, "side"] == cluster_side
                and (g.loc[j + 1, "ts"] - t0).total_seconds() * 1000 <= window_ms
            ):
                j += 1
            cluster = g.iloc[i:j + 1]
            venues = sorted(
                {
                    str(v)
                    for v in cluster.get("exchange", pd.Series([]))
                    if pd.notna(v)
                }
            )
            size_total = int(cluster["size"].sum())
            premium_total = float(
                (cluster["price"].fillna(0) * cluster["size"]).sum() * CONTRACT_MULTIPLIER
            )
            if (
                len(cluster) >= cfg.sweep_min_legs
                and len(venues) >= 2
                and premium_total >= cfg.sweep_min_premium
            ):
                vwap = (
                    float(
                        (cluster["price"] * cluster["size"]).sum() / size_total
                    )
                    if size_total > 0
                    else None
                )
                out.append({
                    "event_type": "SWEEP",
                    "ts": t0.isoformat(),
                    "symbol": cluster.iloc[0]["symbol"],
                    "expiration": _isoformat_date(cluster.iloc[0]["expiration"]),
                    "strike": float(cluster.iloc[0]["strike"]),
                    "option_type": str(cluster.iloc[0]["option_type"]).upper(),
                    "side": int(cluster_side),
                    "size": size_total,
                    "price": vwap,
                    "legs": int(len(cluster)),
                    "venues": venues,
                    "meta": {
                        "window_ms": cfg.sweep_window_ms,
                        "premium": premium_total,
                        "premium_threshold": cfg.sweep_min_premium,
                    },
                })
                i = j + 1
            else:
                i += 1
    return out


def _detect_uoa(
    trades: pd.DataFrame,
    contract_adv: pd.DataFrame | None,
    contract_oi: pd.DataFrame | None,
    cfg: FlowEventConfig,
    contracts_with_event: set[tuple] | None = None,
) -> list[dict]:
    """Flag contracts whose today's total volume is well above ADV/OI."""
    daily = trades.groupby(_CONTRACT_KEYS, as_index=False)["size"].sum()
    daily = daily.rename(columns={"size": "today_volume"})
    if contract_adv is not None and not contract_adv.empty:
        daily = daily.merge(contract_adv, on=_CONTRACT_KEYS, how="left")
    else:
        daily["avg_daily_volume"] = pd.NA
    if contract_oi is not None and not contract_oi.empty:
        daily = daily.merge(contract_oi, on=_CONTRACT_KEYS, how="left")
    else:
        daily["open_interest"] = pd.NA

    contracts_with_event = contracts_with_event or set()
    out: list[dict] = []
    for _, row in daily.iterrows():
        contract_key = (
            row["symbol"],
            _isoformat_date(row["expiration"]),
            float(row["strike"]),
            str(row["option_type"]).upper(),
        )
        if contract_key in contracts_with_event:
            continue

        adv = row.get("avg_daily_volume")
        oi = row.get("open_interest")
        today_vol = int(row["today_volume"])
        is_uoa = False
        method: str | None = None
        threshold: float | None = None
        ratio: float | None = None

        if pd.notna(adv) and float(adv) > 0:
            threshold = float(adv) * cfg.uoa_volume_multiplier
            is_uoa = today_vol >= threshold
            method = "adv"
            ratio = today_vol / float(adv) if float(adv) > 0 else None
        elif pd.notna(oi) and float(oi) > 0:
            threshold = float(oi) * cfg.uoa_vol_oi_ratio
            is_uoa = today_vol >= threshold
            method = "vol_oi_ratio"
            ratio = today_vol / float(oi)
        else:
            threshold = float(cfg.uoa_min_absolute_volume)
            is_uoa = today_vol >= cfg.uoa_min_absolute_volume
            method = "absolute"

        if not is_uoa:
            continue

        # Use the LAST trade's timestamp on this contract as the event ts.
        contract_trades = trades[
            (trades["symbol"] == row["symbol"])
            & (trades["expiration"] == row["expiration"])
            & (trades["strike"] == row["strike"])
            & (trades["option_type"] == row["option_type"])
        ]
        last_ts = contract_trades["ts"].max()
        out.append({
            "event_type": "UOA",
            "ts": last_ts.isoformat(),
            "symbol": row["symbol"],
            "expiration": _isoformat_date(row["expiration"]),
            "strike": float(row["strike"]),
            "option_type": str(row["option_type"]).upper(),
            "side": 0,
            "size": today_vol,
            "price": None,
            "legs": int(len(contract_trades)),
            "venues": [],
            "meta": {
                "method": method,
                "today_volume": today_vol,
                "avg_daily_volume": (
                    float(adv) if pd.notna(adv) and adv is not None else None
                ),
                "open_interest": (
                    float(oi) if pd.notna(oi) and oi is not None else None
                ),
                "threshold": float(threshold),
                "ratio": ratio,
                "uoa_volume_multiplier": cfg.uoa_volume_multiplier,
                "uoa_vol_oi_ratio": cfg.uoa_vol_oi_ratio,
            },
        })
    return out


def _isoformat_date(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError):
        return str(value)
