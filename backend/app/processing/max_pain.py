"""Classic max-pain calculation per expiration plus an aggregate over the nearest 5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CONTRACT_MULTIPLIER = 100


@dataclass
class MaxPainSummary:
    per_expiry: list[dict]
    aggregate_strike: float | None
    aggregate_value: float | None


def _expiry_max_pain(sub: pd.DataFrame) -> tuple[float | None, float | None, list[dict]]:
    """Return (strike, total dollar pain at strike, full pain curve)."""
    if sub.empty:
        return None, None, []

    strikes = np.sort(sub["strike"].dropna().unique())
    if len(strikes) == 0:
        return None, None, []

    calls = sub[sub["option_type"].str.upper() == "C"]
    puts = sub[sub["option_type"].str.upper() == "P"]

    # Dollar pain at expiration for each candidate underlying price S* = K_candidate.
    pain_curve: list[dict] = []
    best_strike = None
    best_value = None

    for s_star in strikes:
        call_loss = float(
            (np.maximum(s_star - calls["strike"].to_numpy(dtype=float), 0.0)
             * calls["oi"].fillna(0).to_numpy(dtype=float)).sum()
        )
        put_loss = float(
            (np.maximum(puts["strike"].to_numpy(dtype=float) - s_star, 0.0)
             * puts["oi"].fillna(0).to_numpy(dtype=float)).sum()
        )
        total = (call_loss + put_loss) * CONTRACT_MULTIPLIER
        pain_curve.append({"strike": float(s_star), "pain": total})
        if best_value is None or total < best_value:
            best_value = total
            best_strike = float(s_star)

    return best_strike, best_value, pain_curve


def compute_max_pain(df: pd.DataFrame, *, aggregate_n: int = 5) -> MaxPainSummary:
    """Compute max pain per expiration and an aggregate across the nearest ``aggregate_n``."""
    if df.empty:
        return MaxPainSummary(per_expiry=[], aggregate_strike=None, aggregate_value=None)

    per_expiry: list[dict] = []
    expiries_sorted = sorted(pd.to_datetime(df["expiration"].unique()))
    for expiry in expiries_sorted:
        sub = df[pd.to_datetime(df["expiration"]) == expiry]
        strike, value, curve = _expiry_max_pain(sub)
        per_expiry.append(
            {
                "expiration": str(pd.Timestamp(expiry).date()),
                "strike": strike,
                "pain": value,
                "curve": curve,
            }
        )

    nearest = [pd.Timestamp(e) for e in expiries_sorted[:aggregate_n]]
    sub = df[pd.to_datetime(df["expiration"]).isin(nearest)]
    agg_strike, agg_value, _ = _expiry_max_pain(sub)

    return MaxPainSummary(
        per_expiry=per_expiry,
        aggregate_strike=agg_strike,
        aggregate_value=agg_value,
    )
