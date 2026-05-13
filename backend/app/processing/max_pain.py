"""Classic max-pain calculation per expiration plus an aggregate over the nearest 5.

Max pain answers: at expiration, which underlying price ``K*`` minimises the
total dollar loss across all open contracts (calls + puts × OI)?

Behaviour:

* ``compute_max_pain(df)`` (default) — preserves the historical pipeline
  behaviour: one max-pain strike per distinct expiration, plus an
  ``aggregate_*`` pair folded across the nearest ``aggregate_n`` expirations.
* ``compute_max_pain(df, expiry=<value>)`` — restricts the calculation to a
  single expiration. Both the per-expiry list and the aggregate strike then
  reflect just that expiry, which mirrors the optional ``expiry`` query
  parameter on ``GET /v1/{symbol}/max-pain``.
* ``compute_max_pain(df, expiry=None, fold_all=True)`` — folds every
  expiration into a single OI distribution before solving. Useful for a
  global pin level when the caller doesn't care about expiration buckets.

NaN/inf safety: every numeric column is coerced to numeric and non-finite
values are dropped or zero-filled, so the resulting pain curve and strike
are guaranteed finite floats (or ``None`` when no contracts qualify).
"""

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


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing strike / option_type and coerce numerics to finite floats."""
    if df.empty or "strike" not in df.columns or "option_type" not in df.columns:
        return df.iloc[0:0]
    out = df.copy()
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out = out[np.isfinite(out["strike"])]
    if "oi" not in out.columns:
        out["oi"] = 0.0
    out["oi"] = pd.to_numeric(out["oi"], errors="coerce").fillna(0.0)
    out.loc[~np.isfinite(out["oi"]), "oi"] = 0.0
    return out


def _expiry_max_pain(sub: pd.DataFrame) -> tuple[float | None, float | None, list[dict]]:
    """Return (strike, total dollar pain at strike, full pain curve)."""
    if sub.empty:
        return None, None, []

    strikes = np.sort(sub["strike"].dropna().unique())
    strikes = strikes[np.isfinite(strikes)]
    if strikes.size == 0:
        return None, None, []

    is_call = sub["option_type"].astype(str).str.upper() == "C"
    calls = sub[is_call]
    puts = sub[~is_call]

    call_strikes = calls["strike"].to_numpy(dtype=float)
    call_oi = calls["oi"].to_numpy(dtype=float)
    put_strikes = puts["strike"].to_numpy(dtype=float)
    put_oi = puts["oi"].to_numpy(dtype=float)

    pain_curve: list[dict] = []
    best_strike: float | None = None
    best_value: float | None = None

    for s_star in strikes:
        call_loss = float(
            (np.maximum(s_star - call_strikes, 0.0) * call_oi).sum()
        )
        put_loss = float(
            (np.maximum(put_strikes - s_star, 0.0) * put_oi).sum()
        )
        total = (call_loss + put_loss) * CONTRACT_MULTIPLIER
        if not np.isfinite(total):
            total = 0.0
        pain_curve.append({"strike": float(s_star), "pain": total})
        if best_value is None or total < best_value:
            best_value = total
            best_strike = float(s_star)

    return best_strike, best_value, pain_curve


def compute_max_pain(
    df: pd.DataFrame,
    *,
    aggregate_n: int = 5,
    expiry: str | pd.Timestamp | None = None,
    fold_all: bool = False,
) -> MaxPainSummary:
    """Compute max pain per expiration and an aggregate across the nearest ``aggregate_n``.

    Args:
        df: Option chain with ``strike``, ``option_type``, ``oi``, and
            (when applicable) ``expiration`` columns.
        aggregate_n: Number of nearest expirations folded into the
            aggregate pain distribution. Ignored when ``fold_all`` is set
            or ``expiry`` is provided.
        expiry: Optional single expiration to filter to. When set, both the
            per-expiry list and the aggregate reflect that expiry only.
        fold_all: When True (and ``expiry`` is ``None``), every row is folded
            into a single distribution rather than bucketed per-expiry.

    Returns:
        ``MaxPainSummary`` with ``per_expiry`` rows, ``aggregate_strike`` and
        ``aggregate_value``. The strike/value are ``None`` when no contracts
        qualify.
    """
    df = _clean(df)
    if df.empty:
        return MaxPainSummary(per_expiry=[], aggregate_strike=None, aggregate_value=None)

    has_expiration = "expiration" in df.columns and not df["expiration"].isna().all()

    # Single-expiry filter takes precedence.
    if expiry is not None:
        if not has_expiration:
            return MaxPainSummary(per_expiry=[], aggregate_strike=None, aggregate_value=None)
        try:
            target = pd.Timestamp(expiry)
        except (TypeError, ValueError):
            return MaxPainSummary(per_expiry=[], aggregate_strike=None, aggregate_value=None)
        sub = df[pd.to_datetime(df["expiration"], errors="coerce") == target]
        strike, value, curve = _expiry_max_pain(sub)
        per_expiry: list[dict] = []
        if strike is not None:
            per_expiry = [
                {
                    "expiration": str(target.date()),
                    "strike": strike,
                    "pain": value,
                    "curve": curve,
                }
            ]
        return MaxPainSummary(
            per_expiry=per_expiry,
            aggregate_strike=strike,
            aggregate_value=value,
        )

    # Fold-all path: single distribution across every expiration.
    if fold_all or not has_expiration:
        strike, value, curve = _expiry_max_pain(df)
        per_expiry = (
            [{"expiration": "all", "strike": strike, "pain": value, "curve": curve}]
            if strike is not None
            else []
        )
        return MaxPainSummary(
            per_expiry=per_expiry,
            aggregate_strike=strike,
            aggregate_value=value,
        )

    # Default: per-expiry list + aggregate across the nearest ``aggregate_n``.
    per_expiry = []
    parsed_expiries = pd.to_datetime(df["expiration"].unique(), errors="coerce")
    parsed_expiries = parsed_expiries[~pd.isna(parsed_expiries)]
    expiries_sorted = sorted(parsed_expiries)
    for exp_ts in expiries_sorted:
        sub = df[pd.to_datetime(df["expiration"], errors="coerce") == exp_ts]
        strike, value, curve = _expiry_max_pain(sub)
        per_expiry.append(
            {
                "expiration": str(pd.Timestamp(exp_ts).date()),
                "strike": strike,
                "pain": value,
                "curve": curve,
            }
        )

    nearest = [pd.Timestamp(e) for e in expiries_sorted[:aggregate_n]]
    sub = df[pd.to_datetime(df["expiration"], errors="coerce").isin(nearest)]
    agg_strike, agg_value, _ = _expiry_max_pain(sub)

    return MaxPainSummary(
        per_expiry=per_expiry,
        aggregate_strike=agg_strike,
        aggregate_value=agg_value,
    )
