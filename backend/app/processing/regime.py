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

import math
from dataclasses import asdict, dataclass

from app.processing.gex import GexSummary
from app.processing.walls import WallsSummary

DEFAULT_REGIME_THRESHOLD = 0.2


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


def _label_from_score(score: float, *, threshold: float = DEFAULT_REGIME_THRESHOLD) -> str:
    """Map a raw regime score to a label using ``threshold`` as a deadband.

    The deadband around zero implements simple hysteresis so the regime
    label does not flip on small numerical noise: scores strictly inside
    ``[-threshold, +threshold]`` are reported as ``"neutral"``.
    """
    if not math.isfinite(score):
        return "neutral"
    th = abs(threshold) if math.isfinite(threshold) else DEFAULT_REGIME_THRESHOLD
    if score > th:
        return "bullish"
    if score < -th:
        return "bearish"
    return "neutral"


def _wall_total(walls: dict | None, key: str) -> float:
    if not walls:
        return 0.0
    arr = walls.get(key) or []
    total = 0.0
    for entry in arr:
        try:
            value = float(entry.get("value") or 0.0)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            total += value
    return total


def _wall_dominance(call_total: float, put_total: float) -> float:
    denom = call_total + put_total
    if denom <= 0 or not math.isfinite(denom):
        return 0.0
    raw = float((call_total - put_total) / denom)
    if not math.isfinite(raw):
        return 0.0
    # ``call_total`` and ``put_total`` are non-negative by construction
    # (gross weight totals), so ``raw`` is already in ``[-1, +1]``; this
    # clamp is defensive against malformed inputs.
    return max(-1.0, min(1.0, raw))


def _gex_sign_score(gex: GexSummary | None) -> float:
    if gex is None or not gex.curve:
        return 0.0
    gross = 0.0
    for row in gex.curve:
        try:
            v = float(row.get("net_gex") or 0.0)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            gross += abs(v)
    if gross <= 0:
        return 0.0
    try:
        net = float(gex.net_total)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(net):
        return 0.0
    raw = net / gross
    if not math.isfinite(raw):
        return 0.0
    if raw > 1.0:
        return 1.0
    if raw < -1.0:
        return -1.0
    return raw


def _mode(
    walls_payload: dict | None,
    gex: GexSummary | None,
    *,
    threshold: float,
) -> RegimeMode:
    call_total = _wall_total(walls_payload, "call_wall")
    put_total = _wall_total(walls_payload, "put_wall")
    wall_dom = _wall_dominance(call_total, put_total)
    gex_sign = _gex_sign_score(gex)
    score = 0.6 * wall_dom + 0.4 * gex_sign
    if not math.isfinite(score):
        score = 0.0
    # Clamp final score to [-1, 1] (already true by construction, but defensive).
    score = max(-1.0, min(1.0, score))
    net_gex = 0.0
    if gex is not None:
        try:
            candidate = float(gex.net_total)
        except (TypeError, ValueError):
            candidate = 0.0
        net_gex = candidate if math.isfinite(candidate) else 0.0
    return RegimeMode(
        score=score,
        label=_label_from_score(score, threshold=threshold),
        call_wall_total=call_total,
        put_wall_total=put_total,
        net_gex=net_gex,
    )


def _resolve_threshold(threshold: float | None) -> float:
    """Pick the hysteresis deadband, defaulting to ``Settings.gex_regime_threshold``."""
    if threshold is not None:
        if not math.isfinite(threshold):
            return DEFAULT_REGIME_THRESHOLD
        return abs(threshold)
    try:
        from app.config import get_settings

        settings_value = float(get_settings().gex_regime_threshold)
    except Exception:  # noqa: BLE001 - settings unavailable in some unit-test contexts
        return DEFAULT_REGIME_THRESHOLD
    if not math.isfinite(settings_value):
        return DEFAULT_REGIME_THRESHOLD
    return abs(settings_value)


def compute_regime(
    walls: WallsSummary,
    gex_oi: GexSummary,
    gex_vol: GexSummary,
    *,
    threshold: float | None = None,
) -> RegimeSummary:
    """Compute the OI/volume regime score + label.

    ``threshold`` overrides the hysteresis deadband. When ``None`` (the
    default used by the pipeline), the configured
    ``Settings.gex_regime_threshold`` is used.
    """
    th = _resolve_threshold(threshold)
    return RegimeSummary(
        oi=_mode(walls.by_oi, gex_oi, threshold=th),
        vol=_mode(walls.by_volume, gex_vol, threshold=th),
    )
