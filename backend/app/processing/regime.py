"""Regime score computation.

The "regime" answers a deceptively simple trader question: are dealers
currently positioned **long gamma (bullish, vol-suppressing)** or **short
gamma (bearish, vol-amplifying)** at a given underlying?

We expose two flavours so users can read both rest-state ("OI") and intraday
flow-state ("Volume") positioning:

* ``regime_oi``   — based on call/put walls weighted by open interest and the
                    GEX-by-OI net total.
* ``regime_vol``  — same, but weighted by today's traded volume.

The score is a number in roughly ``[-1, +1]``:
* ``score > +0.2``  → ``bullish``  (call dominance, supportive flow)
* ``score < -0.2``  → ``bearish``  (put dominance, downside flow)
* otherwise         → ``neutral``

The score is computed as a blend of two normalised signals:

1. **Wall dominance** — ``(Σcall_wall − Σput_wall) / (Σcall_wall + Σput_wall)``
2. **GEX sign**       — ``net_gex / max(|all_gex|, 1)`` clamped to ``[-1, 1]``

Final score = ``0.6 * wall_dominance + 0.4 * gex_sign`` (both already in
``[-1, 1]``) so a strong wall stack alone is enough to flip the regime even
when GEX hasn't been computed yet (for example before live OI lands).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app.processing.gex import GexSummary
from app.processing.walls import WallsSummary


@dataclass
class RegimeMode:
    score: float
    label: str
    call_wall_total: float
    put_wall_total: float
    net_gex: float


@dataclass
class RegimeSummary:
    oi: RegimeMode
    vol: RegimeMode

    def to_dict(self) -> dict:
        return {"oi": asdict(self.oi), "vol": asdict(self.vol)}


def _label_from_score(score: float, *, threshold: float = 0.2) -> str:
    if score > threshold:
        return "bullish"
    if score < -threshold:
        return "bearish"
    return "neutral"


def _wall_total(walls: dict | None, key: str) -> float:
    if not walls:
        return 0.0
    arr = walls.get(key) or []
    total = 0.0
    for entry in arr:
        try:
            total += float(entry.get("value") or 0.0)
        except (TypeError, ValueError):
            continue
    return total


def _wall_dominance(call_total: float, put_total: float) -> float:
    denom = call_total + put_total
    if denom <= 0:
        return 0.0
    return float((call_total - put_total) / denom)


def _gex_sign_score(gex: GexSummary | None) -> float:
    if gex is None or not gex.curve:
        return 0.0
    gross = sum(abs(row.get("net_gex") or 0.0) for row in gex.curve)
    if gross <= 0:
        return 0.0
    raw = float(gex.net_total / gross)
    if raw > 1.0:
        return 1.0
    if raw < -1.0:
        return -1.0
    return raw


def _mode(walls_payload: dict | None, gex: GexSummary | None) -> RegimeMode:
    call_total = _wall_total(walls_payload, "call_wall")
    put_total = _wall_total(walls_payload, "put_wall")
    wall_dom = _wall_dominance(call_total, put_total)
    gex_sign = _gex_sign_score(gex)
    score = 0.6 * wall_dom + 0.4 * gex_sign
    # Clamp final score to [-1, 1] (already true by construction, but defensive).
    score = max(-1.0, min(1.0, score))
    return RegimeMode(
        score=score,
        label=_label_from_score(score),
        call_wall_total=call_total,
        put_wall_total=put_total,
        net_gex=float(gex.net_total) if gex is not None else 0.0,
    )


def compute_regime(
    walls: WallsSummary,
    gex_oi: GexSummary,
    gex_vol: GexSummary,
) -> RegimeSummary:
    return RegimeSummary(
        oi=_mode(walls.by_oi, gex_oi),
        vol=_mode(walls.by_volume, gex_vol),
    )
