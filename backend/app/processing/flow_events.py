"""Detect notable option flow events: sweeps, blocks, and unusual activity (UOA).

These are heuristic, not strict definitions. Conventions used:

* **Sweep** — multiple prints of the *same* option contract from the *same*
  customer side, hitting *different exchanges*, all within a short time
  window (default 100 ms). The hallmark of an aggressive multi-venue
  taker that wants size now and is willing to walk through liquidity.
* **Block** — a single print of size ≥ ``block_min_size`` (default 100
  contracts). Often pre-negotiated upstairs and reported off-book.
* **UOA** — Unusual Options Activity. Today's traded volume on a contract
  is ≥ ``uoa_volume_multiplier`` (default 5x) the contract's *trailing
  average daily volume*. We expect callers to supply the rolling average
  externally (intraday alerts compute it once per session); if missing
  we fall back to ``volume ≥ uoa_min_absolute_volume`` (default 5000).

Function ``detect_flow_events`` returns a list of structured event dicts
ready to be inserted into a ``flow_events`` table.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FlowEventConfig:
    sweep_window_ms: int = 100
    sweep_min_legs: int = 3
    block_min_size: int = 100
    uoa_volume_multiplier: float = 5.0
    uoa_min_absolute_volume: int = 5000


def detect_flow_events(
    trades: pd.DataFrame,
    *,
    contract_adv: pd.DataFrame | None = None,
    config: FlowEventConfig | None = None,
) -> list[dict]:
    """Return a list of detected sweep / block / UOA events.

    ``trades`` must contain at minimum::

        ts, symbol, expiration, strike, option_type,
        price, size, side, exchange

    ``contract_adv`` (optional) is a DataFrame with one row per
    ``(symbol, expiration, strike, option_type)`` and columns
    ``avg_daily_volume`` (rolling N-day average traded volume).

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
            "venues": [row["exchange"]] if "exchange" in row.index and pd.notna(row.get("exchange")) else [],
            "meta": {"threshold": cfg.block_min_size},
        })

    # ── UOA (today's volume vs trailing ADV) ─────────────────────────────
    events.extend(_detect_uoa(work, contract_adv, cfg))

    return events


def _detect_sweeps(trades: pd.DataFrame, cfg: FlowEventConfig) -> list[dict]:
    """Group adjacent same-side same-contract prints across multiple venues."""
    out: list[dict] = []
    keys = ["symbol", "expiration", "strike", "option_type"]
    for _, group in trades.groupby(keys, sort=False):
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
            if (
                len(cluster) >= cfg.sweep_min_legs
                and len(venues) >= 2
            ):
                size_total = int(cluster["size"].sum())
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
                    },
                })
                i = j + 1
            else:
                i += 1
    return out


def _detect_uoa(
    trades: pd.DataFrame,
    contract_adv: pd.DataFrame | None,
    cfg: FlowEventConfig,
) -> list[dict]:
    """Flag contracts whose today's total volume is well above ADV."""
    keys = ["symbol", "expiration", "strike", "option_type"]
    daily = trades.groupby(keys, as_index=False)["size"].sum()
    daily = daily.rename(columns={"size": "today_volume"})
    if contract_adv is not None and not contract_adv.empty:
        daily = daily.merge(contract_adv, on=keys, how="left")
    else:
        daily["avg_daily_volume"] = pd.NA

    out: list[dict] = []
    for _, row in daily.iterrows():
        adv = row.get("avg_daily_volume")
        today_vol = int(row["today_volume"])
        is_uoa = False
        threshold = None
        if pd.notna(adv) and adv > 0:
            threshold = float(adv) * cfg.uoa_volume_multiplier
            is_uoa = today_vol >= threshold
        else:
            threshold = cfg.uoa_min_absolute_volume
            is_uoa = today_vol >= cfg.uoa_min_absolute_volume

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
                "today_volume": today_vol,
                "avg_daily_volume": (
                    float(adv) if pd.notna(adv) and adv is not None else None
                ),
                "threshold": float(threshold),
                "uoa_volume_multiplier": cfg.uoa_volume_multiplier,
            },
        })
    return out


def _isoformat_date(value) -> str:  # type: ignore[no-untyped-def]
    if isinstance(value, str):
        return value
    try:
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError):
        return str(value)
