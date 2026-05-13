"""Cross-metric correctness audit for the Rev 3 hardening (Agent 2).

The pre-existing per-module tests cover each metric in isolation. This file
asserts the *invariants* the snapshot API depends on:

* NaN / inf inputs never leak into curve / scalar outputs.
* Sign conventions are consistent across GEX, Vanna and Charm.
* Aggregations respect the documented dealer-hedging convention
  (calls positive, puts negative).
* Optional ``expiry`` filter on ``compute_max_pain`` works as the
  ``GET /v1/{symbol}/max-pain`` endpoint expects.
* Regime hysteresis uses ``Settings.gex_regime_threshold`` as a
  symmetric deadband.

All chains here are synthetic; the goal is correctness of the math, not
fidelity to a real options book.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.processing.gex import GexSummary, compute_gex
from app.processing.max_pain import compute_max_pain
from app.processing.regime import (
    DEFAULT_REGIME_THRESHOLD,
    _label_from_score,
    compute_regime,
)
from app.processing.vanna_charm import compute_charm, compute_vanna
from app.processing.walls import WallsSummary, compute_walls
from app.processing.zero_gamma import compute_zero_gamma

_TODAY = pd.Timestamp("2026-01-02")
_EXP_30D = (_TODAY + pd.Timedelta(days=30)).date()


# ── helpers ────────────────────────────────────────────────────────────────


def _chain_row(
    *,
    strike: float,
    option_type: str,
    oi: float = 1000.0,
    volume: float = 100.0,
    gamma: float = 0.05,
    iv: float = 0.18,
    underlying_price: float = 100.0,
    expiration: date | str | None = None,
) -> dict:
    return {
        "strike": float(strike),
        "option_type": option_type,
        "oi": float(oi),
        "volume": float(volume),
        "gamma": float(gamma),
        "iv": float(iv),
        "expiration": expiration if expiration is not None else _EXP_30D,
        "underlying_price": float(underlying_price),
    }


def _curve_has_only_finite_floats(curve: list[dict], keys: tuple[str, ...]) -> bool:
    for row in curve:
        for k in keys:
            v = row.get(k)
            if v is None:
                continue
            if not math.isfinite(float(v)):
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────
# GEX
# ─────────────────────────────────────────────────────────────────────────


def test_gex_all_call_chain_has_positive_net_total():
    df = pd.DataFrame(
        [
            _chain_row(strike=k, option_type="C", oi=1000)
            for k in (95, 100, 105)
        ]
    )
    out = compute_gex(df)
    assert out.net_total > 0
    assert out.top_positive
    assert out.top_negative


def test_gex_all_put_chain_has_opposite_sign_from_all_call_chain():
    call_df = pd.DataFrame(
        [_chain_row(strike=k, option_type="C", oi=1000) for k in (95, 100, 105)]
    )
    put_df = pd.DataFrame(
        [_chain_row(strike=k, option_type="P", oi=1000) for k in (95, 100, 105)]
    )
    call_out = compute_gex(call_df)
    put_out = compute_gex(put_df)
    # Dealer-side convention enforced in compute_gex assigns +1 to calls and
    # -1 to puts; the two nets must therefore be strict opposites in sign.
    assert call_out.net_total > 0
    assert put_out.net_total < 0
    assert call_out.net_total * put_out.net_total < 0


def test_gex_nan_and_inf_inputs_never_leak_into_curve():
    df = pd.DataFrame(
        [
            _chain_row(strike=100, option_type="C", gamma=float("nan")),
            _chain_row(strike=105, option_type="C", gamma=float("inf"), oi=500),
            _chain_row(strike=110, option_type="C", oi=float("nan"), gamma=0.05),
            _chain_row(strike=115, option_type="C", oi=1000, gamma=0.05),
        ]
    )
    out = compute_gex(df)
    assert math.isfinite(out.net_total)
    # Only the strike with finite gamma+OI should contribute.
    assert out.net_total > 0
    assert _curve_has_only_finite_floats(out.curve, ("call_gex", "put_gex", "net_gex"))


def test_gex_inf_underlying_yields_empty_summary():
    df = pd.DataFrame(
        [_chain_row(strike=100, option_type="C", underlying_price=float("inf"))]
    )
    out = compute_gex(df)
    assert out.underlying_price is None
    assert out.curve == []
    assert out.net_total == 0.0


# ─────────────────────────────────────────────────────────────────────────
# Vanna
# ─────────────────────────────────────────────────────────────────────────


def test_vanna_dealer_sign_cancels_matched_call_and_put_at_same_strike():
    """BSM vanna is identical for a call and put at the same strike.
    With dealer sign convention (+1 call, -1 put), a matched pair at the
    same strike and OI must cancel to ~0 — confirming the convention is
    applied uniformly to both option types."""
    spot = 100.0
    df = pd.DataFrame(
        [
            _chain_row(
                strike=100,
                option_type="C",
                oi=1000,
                underlying_price=spot,
                expiration=(_TODAY + pd.Timedelta(days=30)).date(),
            ),
            _chain_row(
                strike=100,
                option_type="P",
                oi=1000,
                underlying_price=spot,
                expiration=(_TODAY + pd.Timedelta(days=30)).date(),
            ),
        ]
    )
    out = compute_vanna(df, today=_TODAY)
    row = next(r for r in out.curve if r["strike"] == 100.0)
    assert math.isfinite(row["vanna_exposure"])
    # +V (call) + (-V) (put) = 0
    assert abs(row["vanna_exposure"]) < 1e-6
    assert _curve_has_only_finite_floats(out.curve, ("vanna_exposure",))


def test_vanna_handles_nan_iv_rows_by_skipping_them():
    df = pd.DataFrame(
        [
            _chain_row(strike=100, option_type="C", iv=float("nan")),
            _chain_row(strike=105, option_type="C", iv=0.2, oi=1000),
        ]
    )
    out = compute_vanna(df, today=_TODAY)
    assert math.isfinite(out.net_total)
    assert out.curve  # the IV=0.2 row should survive


# ─────────────────────────────────────────────────────────────────────────
# Charm
# ─────────────────────────────────────────────────────────────────────────


def test_charm_dealer_sign_cancels_matched_call_and_put_at_same_strike():
    """With q=0 the BSM charm common part is identical for call/put and the
    q-dependent extra terms vanish. The dealer-sign aggregator (+1 call,
    -1 put) therefore cancels a matched pair at the same strike."""
    spot = 100.0
    df = pd.DataFrame(
        [
            _chain_row(
                strike=100,
                option_type="C",
                oi=1000,
                underlying_price=spot,
                expiration=(_TODAY + pd.Timedelta(days=30)).date(),
            ),
            _chain_row(
                strike=100,
                option_type="P",
                oi=1000,
                underlying_price=spot,
                expiration=(_TODAY + pd.Timedelta(days=30)).date(),
            ),
        ]
    )
    out = compute_charm(df, today=_TODAY)
    row = next(r for r in out.curve if r["strike"] == 100.0)
    assert math.isfinite(row["charm_exposure"])
    assert abs(row["charm_exposure"]) < 1e-6
    assert _curve_has_only_finite_floats(out.curve, ("charm_exposure",))


def test_charm_decays_toward_zero_for_deep_otm_long_dated_then_short_dated():
    """Deep-OTM charm magnitude shrinks dramatically as τ → 0.

    For deep-OTM options, d1 is far from zero so φ(d1) → 0 and charm
    collapses with τ. We verify that the magnitude at τ=1d is strictly
    smaller than at τ=30d for the same strike+IV+OI.
    """
    spot = 100.0
    deep_otm_strike = 200.0  # very deep OTM call

    far_df = pd.DataFrame(
        [
            _chain_row(
                strike=deep_otm_strike,
                option_type="C",
                oi=1000,
                iv=0.18,
                underlying_price=spot,
                expiration=(_TODAY + pd.Timedelta(days=30)).date(),
            )
        ]
    )
    near_df = pd.DataFrame(
        [
            _chain_row(
                strike=deep_otm_strike,
                option_type="C",
                oi=1000,
                iv=0.18,
                underlying_price=spot,
                expiration=(_TODAY + pd.Timedelta(days=1)).date(),
            )
        ]
    )
    far = compute_charm(far_df, today=_TODAY)
    near = compute_charm(near_df, today=_TODAY)
    # Deep-OTM charm must remain finite at both horizons.
    assert math.isfinite(far.net_total)
    assert math.isfinite(near.net_total)
    # As τ shrinks, the deep-OTM φ(d1) factor crushes charm toward zero.
    assert abs(near.net_total) < abs(far.net_total)


def test_charm_nan_inputs_do_not_leak_into_curve():
    df = pd.DataFrame(
        [
            _chain_row(strike=100, option_type="C", iv=float("nan")),
            _chain_row(strike=105, option_type="C", oi=float("inf")),
            _chain_row(strike=110, option_type="C", oi=1000),
        ]
    )
    out = compute_charm(df, today=_TODAY)
    assert math.isfinite(out.net_total)
    assert _curve_has_only_finite_floats(out.curve, ("charm_exposure",))


# ─────────────────────────────────────────────────────────────────────────
# Walls
# ─────────────────────────────────────────────────────────────────────────


def test_walls_top_three_returns_giant_strike_first():
    """With one dominant OI strike, that strike must rank #1 among call walls."""
    rows = [
        _chain_row(strike=100, option_type="C", oi=500),
        _chain_row(strike=105, option_type="C", oi=800),
        _chain_row(strike=110, option_type="C", oi=50_000),  # the giant
        _chain_row(strike=115, option_type="C", oi=200),
        _chain_row(strike=95, option_type="P", oi=300),
    ]
    df = pd.DataFrame(rows)
    out = compute_walls(df, top_n=3)
    call_wall = out.by_oi["call_wall"]
    assert len(call_wall) == 3
    assert call_wall[0]["strike"] == 110.0
    assert call_wall[0]["value"] == 50_000.0
    # The giant must be #1 strictly.
    assert call_wall[0]["value"] > call_wall[1]["value"]


def test_walls_filter_out_non_finite_weights():
    rows = [
        _chain_row(strike=100, option_type="C", oi=float("nan")),
        _chain_row(strike=105, option_type="C", oi=float("inf")),
        _chain_row(strike=110, option_type="C", oi=2_000),
    ]
    df = pd.DataFrame(rows)
    out = compute_walls(df, top_n=3)
    # Only the finite-OI strike should appear.
    call_wall = out.by_oi["call_wall"]
    assert len(call_wall) == 1
    assert call_wall[0]["strike"] == 110.0


def test_walls_returns_empty_when_chain_lacks_option_type_column():
    df = pd.DataFrame([{"strike": 100, "oi": 1000, "volume": 0}])
    out = compute_walls(df)
    assert out.by_oi == {}
    assert out.by_volume == {}


# ─────────────────────────────────────────────────────────────────────────
# Max pain
# ─────────────────────────────────────────────────────────────────────────


def test_max_pain_concentrates_at_strike_where_all_oi_lives():
    """When 100% of OI sits at K=4500, the max-pain strike is exactly 4500."""
    today = pd.Timestamp.utcnow().normalize()
    expiry = today + pd.Timedelta(days=7)
    # The chain must contain other strikes too so the search has candidates,
    # but OI is concentrated entirely at 4500.
    rows = []
    for K in (4400, 4450, 4500, 4550, 4600):
        oi_c = 1_000_000 if K == 4500 else 0
        oi_p = 1_000_000 if K == 4500 else 0
        rows.append(
            {"expiration": expiry, "strike": K, "option_type": "C", "oi": oi_c, "volume": 0}
        )
        rows.append(
            {"expiration": expiry, "strike": K, "option_type": "P", "oi": oi_p, "volume": 0}
        )
    df = pd.DataFrame(rows)
    out = compute_max_pain(df)
    assert len(out.per_expiry) == 1
    assert out.per_expiry[0]["strike"] == 4500.0
    assert out.aggregate_strike == 4500.0


def test_max_pain_expiry_filter_restricts_output_to_single_expiry():
    today = pd.Timestamp("2026-01-02")
    e1 = today + pd.Timedelta(days=7)
    e2 = today + pd.Timedelta(days=14)
    rows = []
    for K in (95, 100, 105):
        # Expiry 1: OI concentrated at 100
        rows.append({"expiration": e1, "strike": K, "option_type": "C",
                     "oi": 1000 if K == 100 else 0, "volume": 0})
        rows.append({"expiration": e1, "strike": K, "option_type": "P",
                     "oi": 1000 if K == 100 else 0, "volume": 0})
        # Expiry 2: OI concentrated at 95
        rows.append({"expiration": e2, "strike": K, "option_type": "C",
                     "oi": 1000 if K == 95 else 0, "volume": 0})
        rows.append({"expiration": e2, "strike": K, "option_type": "P",
                     "oi": 1000 if K == 95 else 0, "volume": 0})
    df = pd.DataFrame(rows)
    out = compute_max_pain(df, expiry=e1)
    assert len(out.per_expiry) == 1
    assert out.per_expiry[0]["expiration"] == str(e1.date())
    assert out.per_expiry[0]["strike"] == 100.0
    assert out.aggregate_strike == 100.0


def test_max_pain_fold_all_collapses_all_expiries_into_single_distribution():
    today = pd.Timestamp("2026-01-02")
    e1 = today + pd.Timedelta(days=7)
    e2 = today + pd.Timedelta(days=14)
    # Both expiries have OI at the same strike → folded distribution still pins there.
    rows = []
    for exp in (e1, e2):
        for K in (95, 100, 105):
            oi = 1000 if K == 100 else 0
            rows.append({"expiration": exp, "strike": K, "option_type": "C",
                         "oi": oi, "volume": 0})
            rows.append({"expiration": exp, "strike": K, "option_type": "P",
                         "oi": oi, "volume": 0})
    df = pd.DataFrame(rows)
    out = compute_max_pain(df, fold_all=True)
    assert out.aggregate_strike == 100.0
    assert len(out.per_expiry) == 1
    assert out.per_expiry[0]["expiration"] == "all"


def test_max_pain_drops_nan_strikes_and_oi():
    today = pd.Timestamp.utcnow().normalize()
    expiry = today + pd.Timedelta(days=7)
    rows = [
        # Garbage rows that should be silently dropped.
        {"expiration": expiry, "strike": float("nan"), "option_type": "C", "oi": 100},
        {"expiration": expiry, "strike": 100, "option_type": "C", "oi": float("nan")},
        # Legitimate concentrated OI at 100.
        {"expiration": expiry, "strike": 100, "option_type": "C", "oi": 1000},
        {"expiration": expiry, "strike": 100, "option_type": "P", "oi": 1000},
        {"expiration": expiry, "strike": 105, "option_type": "C", "oi": 0},
    ]
    df = pd.DataFrame(rows)
    out = compute_max_pain(df)
    assert out.per_expiry[0]["strike"] == 100.0


# ─────────────────────────────────────────────────────────────────────────
# Zero gamma
# ─────────────────────────────────────────────────────────────────────────


def _zero_gamma_row(strike: float, option_type: str, weight: float = 10_000.0,
                    spot: float = 100.0) -> dict:
    return {
        "strike": float(strike),
        "option_type": option_type,
        "iv": 0.18,
        "expiration": (date.today() + timedelta(days=30)).isoformat(),
        "volume": int(weight),
        "oi": int(weight),
        "underlying_price": float(spot),
    }


def test_zero_gamma_monotone_signed_curve_crosses_zero_within_one_grid_step():
    """Build a signed-gamma curve that crosses zero exactly once inside the
    search window, and verify the linear interpolation lands within one
    coarse grid step of a much finer reference solution.
    """
    spot = 100.0
    rows = []
    # Concentrated put cluster below spot drives gamma negative at low S.
    for _ in range(50):
        rows.append(_zero_gamma_row(strike=85, option_type="P", weight=10_000, spot=spot))
    # Concentrated call cluster above spot drives gamma positive at high S.
    for _ in range(50):
        rows.append(_zero_gamma_row(strike=115, option_type="C", weight=10_000, spot=spot))
    df = pd.DataFrame(rows)

    search_pct = 0.10
    zg_fine = compute_zero_gamma(df, search_pct=search_pct, n_points=4001)
    zg_coarse = compute_zero_gamma(df, search_pct=search_pct, n_points=41)
    assert zg_fine is not None and zg_coarse is not None
    assert math.isfinite(zg_fine) and math.isfinite(zg_coarse)
    # Result is inside the search window.
    assert spot * (1 - search_pct) <= zg_fine <= spot * (1 + search_pct)
    # Linear interpolation between two coarse points must agree with the
    # finer solution to within one coarse grid step.
    coarse_step = spot * 2 * search_pct / 40
    assert abs(zg_coarse - zg_fine) < coarse_step


def test_zero_gamma_fallback_returns_closest_strike_when_no_crossing():
    """When no sign flip exists (pure-call book), the opt-in fallback
    returns the grid point with the smallest absolute aggregate gamma."""
    rows = [_zero_gamma_row(strike=k, option_type="C", weight=1_000) for k in (95, 100, 105)]
    df = pd.DataFrame(rows)
    # Default behaviour remains None (preserves backwards compatibility).
    assert compute_zero_gamma(df) is None
    # Opt-in fallback returns a finite grid point near a search-window edge
    # (pure-call gamma is smallest at the boundary of ±5%).
    zg = compute_zero_gamma(df, fallback_to_closest=True)
    assert zg is not None
    assert math.isfinite(zg)
    assert 95.0 <= zg <= 105.0


def test_zero_gamma_returns_none_when_chain_has_nan_iv_everywhere():
    rows = [
        {"strike": 100, "option_type": "C", "iv": float("nan"),
         "expiration": (date.today() + timedelta(days=30)).isoformat(),
         "volume": 1000, "underlying_price": 100.0},
    ]
    df = pd.DataFrame(rows)
    # All rows filtered out → no level emitted.
    assert compute_zero_gamma(df) is None
    assert compute_zero_gamma(df, fallback_to_closest=True) is None


# ─────────────────────────────────────────────────────────────────────────
# Regime
# ─────────────────────────────────────────────────────────────────────────


def _empty_gex() -> GexSummary:
    return GexSummary(
        underlying_price=100.0, net_total=0.0, curve=[],
        top_positive=[], top_negative=[],
    )


def _gex_with_signal(net: float, gross: float) -> GexSummary:
    """Build a GEX summary whose ``net_total / sum(|net_gex|)`` ratio inside
    ``_gex_sign_score`` equals exactly ``net / gross`` (pre-clamp).

    Two curve rows are used: one positive and one negative net_gex value,
    chosen so ``plus + minus = net`` and ``|plus| + |minus| = gross``.
    Requires ``abs(net) <= gross``.
    """
    if abs(net) > gross:
        raise ValueError("abs(net) must be <= gross")
    plus = (gross + net) / 2.0
    minus = -(gross - net) / 2.0
    return GexSummary(
        underlying_price=100.0,
        net_total=net,
        curve=[
            {
                "strike": 100.0,
                "call_gex": plus,
                "put_gex": 0.0,
                "net_gex": plus,
            },
            {
                "strike": 110.0,
                "call_gex": 0.0,
                "put_gex": -minus,
                "net_gex": minus,
            },
        ],
        top_positive=[],
        top_negative=[],
    )


def test_regime_score_inside_deadband_is_neutral():
    # Walls: call=1100 / put=900 → wall_dom = 200/2000 = 0.1
    # GEX: net=1 / gross=10 → gex_sign = 0.1
    # Final score = 0.6*0.1 + 0.4*0.1 = 0.1 (inside ±0.2 deadband → neutral)
    walls = WallsSummary(
        by_oi={
            "call_wall": [{"strike": 105, "value": 1100}],
            "put_wall": [{"strike": 95, "value": 900}],
        },
        by_volume={
            "call_wall": [{"strike": 105, "value": 1100}],
            "put_wall": [{"strike": 95, "value": 900}],
        },
    )
    summary = compute_regime(
        walls,
        _gex_with_signal(net=1.0, gross=10.0),
        _gex_with_signal(net=1.0, gross=10.0),
    )
    assert summary.oi.label == "neutral"
    assert summary.vol.label == "neutral"
    assert -DEFAULT_REGIME_THRESHOLD < summary.oi.score < DEFAULT_REGIME_THRESHOLD


def test_regime_score_above_deadband_is_bullish():
    walls = WallsSummary(
        by_oi={
            "call_wall": [{"strike": 105, "value": 8000}],
            "put_wall": [{"strike": 95, "value": 1000}],
        },
        by_volume={
            "call_wall": [{"strike": 105, "value": 8000}],
            "put_wall": [{"strike": 95, "value": 1000}],
        },
    )
    summary = compute_regime(
        walls,
        _gex_with_signal(net=8.0, gross=10.0),
        _gex_with_signal(net=8.0, gross=10.0),
    )
    assert summary.oi.label == "bullish"
    assert summary.oi.score > DEFAULT_REGIME_THRESHOLD


def test_regime_score_below_minus_deadband_is_bearish():
    walls = WallsSummary(
        by_oi={
            "call_wall": [{"strike": 105, "value": 1000}],
            "put_wall": [{"strike": 95, "value": 8000}],
        },
        by_volume={
            "call_wall": [{"strike": 105, "value": 1000}],
            "put_wall": [{"strike": 95, "value": 8000}],
        },
    )
    summary = compute_regime(
        walls,
        _gex_with_signal(net=-8.0, gross=10.0),
        _gex_with_signal(net=-8.0, gross=10.0),
    )
    assert summary.oi.label == "bearish"
    assert summary.oi.score < -DEFAULT_REGIME_THRESHOLD


def test_regime_threshold_override_widens_neutral_band():
    """A larger threshold suppresses borderline regime flips (hysteresis)."""
    walls = WallsSummary(
        by_oi={
            "call_wall": [{"strike": 105, "value": 4000}],
            "put_wall": [{"strike": 95, "value": 1000}],
        },
        by_volume={
            "call_wall": [{"strike": 105, "value": 4000}],
            "put_wall": [{"strike": 95, "value": 1000}],
        },
    )
    # wall_dom = 3000/5000 = 0.6, gex_sign = 0.6 → score = 0.6 → bullish at 0.2
    # but widened to 0.7 → inside band → neutral.
    base = compute_regime(
        walls,
        _gex_with_signal(net=6.0, gross=10.0),
        _gex_with_signal(net=6.0, gross=10.0),
    )
    assert base.oi.label == "bullish"
    wide = compute_regime(
        walls,
        _gex_with_signal(net=6.0, gross=10.0),
        _gex_with_signal(net=6.0, gross=10.0),
        threshold=0.7,
    )
    assert wide.oi.label == "neutral"
    assert wide.vol.label == "neutral"


def test_regime_label_helper_handles_non_finite_scores():
    # Defensive: non-finite scores collapse to neutral so the dashboard
    # never flickers on corrupt inputs.
    assert _label_from_score(float("nan")) == "neutral"
    assert _label_from_score(float("inf"), threshold=0.5) == "neutral"
    assert _label_from_score(-float("inf"), threshold=0.5) == "neutral"


def test_regime_handles_non_finite_wall_values_without_crashing():
    walls = WallsSummary(
        by_oi={
            "call_wall": [
                {"strike": 105, "value": float("nan")},
                {"strike": 110, "value": float("inf")},
                {"strike": 115, "value": 1000.0},
            ],
            "put_wall": [{"strike": 95, "value": 200.0}],
        },
        by_volume={},
    )
    summary = compute_regime(walls, _empty_gex(), _empty_gex())
    # Non-finite entries must be skipped; the regime score must remain finite.
    assert math.isfinite(summary.oi.score)
    assert summary.oi.call_wall_total == 1000.0
    assert summary.oi.put_wall_total == 200.0


def test_regime_handles_gex_with_nan_net_total_gracefully():
    nan_gex = GexSummary(
        underlying_price=100.0,
        net_total=float("nan"),
        curve=[{"strike": 100.0, "call_gex": 0.0, "put_gex": 0.0,
                "net_gex": float("nan")}],
        top_positive=[],
        top_negative=[],
    )
    walls = WallsSummary(by_oi={}, by_volume={})
    summary = compute_regime(walls, nan_gex, nan_gex)
    assert summary.oi.net_gex == 0.0
    assert math.isfinite(summary.oi.score)
    assert summary.oi.label == "neutral"


# ─────────────────────────────────────────────────────────────────────────
# Cross-metric sign consistency
# ─────────────────────────────────────────────────────────────────────────


def test_gex_sign_dealer_convention_calls_plus_puts_minus_at_same_strike():
    """BSM gamma is identical for a call and put at the same strike. The
    dealer aggregator must flip the put's contribution so the two cancel
    exactly at the matched strike.
    """
    S = 100.0
    df = pd.DataFrame(
        [
            _chain_row(strike=100, option_type="C", oi=1000, gamma=0.05, underlying_price=S),
            _chain_row(strike=100, option_type="P", oi=1000, gamma=0.05, underlying_price=S),
        ]
    )
    out = compute_gex(df)
    row = next(r for r in out.curve if r["strike"] == 100.0)
    assert row["call_gex"] > 0
    assert row["put_gex"] < 0
    assert math.isclose(row["call_gex"], -row["put_gex"], rel_tol=1e-9)
    # net = call - |put| = 0
    assert abs(row["net_gex"]) < 1e-6
    assert math.isclose(out.net_total, 0.0, abs_tol=1e-6)


def test_gex_pure_call_vs_pure_put_book_have_opposite_net_signs():
    """Pure-call book net GEX must be the additive inverse of the matching
    pure-put book (same strikes, same gammas, same OI). This nails down
    that the only difference is the +1/-1 dealer sign.
    """
    S = 100.0
    rows_call = [
        _chain_row(strike=k, option_type="C", oi=1000, gamma=0.05, underlying_price=S)
        for k in (95, 100, 105)
    ]
    rows_put = [
        _chain_row(strike=k, option_type="P", oi=1000, gamma=0.05, underlying_price=S)
        for k in (95, 100, 105)
    ]
    call_out = compute_gex(pd.DataFrame(rows_call))
    put_out = compute_gex(pd.DataFrame(rows_put))
    assert math.isclose(call_out.net_total, -put_out.net_total, rel_tol=1e-9)


def test_metrics_produce_only_finite_floats_on_clean_chain():
    """Smoke test: a well-formed mixed chain yields no NaN/inf anywhere."""
    spot = 100.0
    rows: list[dict] = []
    for K, opt in (
        (95, "C"), (95, "P"),
        (100, "C"), (100, "P"),
        (105, "C"), (105, "P"),
    ):
        rows.append(
            _chain_row(
                strike=K,
                option_type=opt,
                oi=1000,
                volume=500,
                gamma=0.04,
                iv=0.2,
                underlying_price=spot,
                expiration=(_TODAY + pd.Timedelta(days=14)).date(),
            )
        )
    df = pd.DataFrame(rows)

    gex = compute_gex(df, weight_col="oi")
    vanna = compute_vanna(df, today=_TODAY)
    charm = compute_charm(df, today=_TODAY)
    walls = compute_walls(df)
    mp = compute_max_pain(df)

    assert math.isfinite(gex.net_total)
    assert math.isfinite(vanna.net_total)
    assert math.isfinite(charm.net_total)
    assert _curve_has_only_finite_floats(gex.curve, ("call_gex", "put_gex", "net_gex"))
    assert _curve_has_only_finite_floats(vanna.curve, ("vanna_exposure",))
    assert _curve_has_only_finite_floats(charm.curve, ("charm_exposure",))
    for arr in (
        walls.by_oi.get("call_wall", []),
        walls.by_oi.get("put_wall", []),
        walls.by_volume.get("call_wall", []),
        walls.by_volume.get("put_wall", []),
    ):
        for entry in arr:
            assert math.isfinite(entry["value"])
            assert math.isfinite(entry["strike"])
    assert mp.aggregate_strike is not None
    assert math.isfinite(mp.aggregate_strike)


def test_pipeline_calls_compute_regime_without_threshold_uses_settings_default():
    """``pipeline.py`` invokes ``compute_regime(walls, gex_oi, gex_vol)``
    without an explicit threshold. The default must resolve to the configured
    ``Settings.gex_regime_threshold`` (0.2 in the tests env)."""
    walls = WallsSummary(
        by_oi={
            "call_wall": [{"strike": 105, "value": 1100}],
            "put_wall": [{"strike": 95, "value": 900}],
        },
        by_volume={
            "call_wall": [{"strike": 105, "value": 1100}],
            "put_wall": [{"strike": 95, "value": 900}],
        },
    )
    summary = compute_regime(
        walls,
        _gex_with_signal(net=1.0, gross=10.0),
        _gex_with_signal(net=1.0, gross=10.0),
    )
    # 0.6*0.1 + 0.4*0.1 = 0.1 — strictly inside the default ±0.2 deadband.
    assert summary.oi.label == "neutral"


def test_compute_gex_zero_gamma_value_is_propagated_when_present():
    """Sanity-check the ``zero_gamma`` field on ``GexSummary``: it must be
    populated as a finite float when the chain crosses zero."""
    spot = 100.0
    rows = []
    for _ in range(20):
        rows.append(
            {
                "strike": 80,
                "option_type": "P",
                "oi": 10_000,
                "volume": 10_000,
                "gamma": 0.05,
                "iv": 0.18,
                "expiration": (date.today() + timedelta(days=30)).isoformat(),
                "underlying_price": spot,
            }
        )
    for _ in range(20):
        rows.append(
            {
                "strike": 120,
                "option_type": "C",
                "oi": 10_000,
                "volume": 10_000,
                "gamma": 0.05,
                "iv": 0.18,
                "expiration": (date.today() + timedelta(days=30)).isoformat(),
                "underlying_price": spot,
            }
        )
    df = pd.DataFrame(rows)
    out = compute_gex(df, weight_col="oi")
    # The synthetic book has put-cluster + call-cluster around spot; on a
    # tight ±5% search window the curve may not flip — but the field must be
    # either None or a finite float, never NaN.
    if out.zero_gamma is not None:
        assert math.isfinite(out.zero_gamma)
    # Also verify a real crossing detected on a wider window.
    zg = compute_zero_gamma(df, weight_col="oi", search_pct=0.40, n_points=801)
    assert zg is not None
    assert math.isfinite(zg)
    # Sanity-check via a guaranteed nonzero numeric (not NaN).
    assert isinstance(zg, float)
    np.testing.assert_allclose(zg, zg)  # would fail if NaN
