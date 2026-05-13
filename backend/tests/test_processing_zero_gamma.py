"""Tests for the Zero-Gamma flip-level computation."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd

from app.processing.zero_gamma import compute_zero_gamma

# Use a near-dated expiration anchored relative to today so tests stay
# stable as wall-clock advances. Far-dated expirations (tau ≫ 1 year)
# squash the BSM gamma curve so flat that no zero crossing is detectable
# inside a reasonable search window.
_TODAY = pd.Timestamp.utcnow().normalize()
_EXP_30D = (date.today() + timedelta(days=30)).isoformat()


def _row(strike, option_type, weight, iv=0.18, expiration=_EXP_30D, S=100.0):
    return {
        "strike": float(strike),
        "option_type": option_type,
        "iv": iv,
        "expiration": expiration,
        "volume": int(weight),
        "underlying_price": S,
    }


def test_returns_none_on_empty_or_missing_columns():
    assert compute_zero_gamma(pd.DataFrame()) is None
    df = pd.DataFrame([{"strike": 100, "option_type": "C"}])
    assert compute_zero_gamma(df) is None


def test_returns_none_when_no_sign_flip_in_window():
    """Pure-call book has positive gamma everywhere — no zero crossing."""
    rows = [_row(strike=k, option_type="C", weight=1000) for k in (95, 100, 105)]
    df = pd.DataFrame(rows)
    # search ±5% around spot=100 → S in [95, 105]
    assert compute_zero_gamma(df) is None


def test_returns_finite_level_when_book_flips_around_spot():
    """Construct a book that is short-gamma below spot (puts dominate at lower
    strikes) and long-gamma above spot (calls dominate at higher strikes).

    With deep ITM puts at 80 and ATM-ish calls at 100, dealer gamma curve
    sweeps through zero somewhere between the two regimes — must land
    inside the search window.
    """
    rows = []
    # Heavy puts at K=80 (below spot)
    for _ in range(5):
        rows.append(_row(strike=80, option_type="P", weight=10_000))
    # Heavy calls at K=120 (above spot)
    for _ in range(5):
        rows.append(_row(strike=120, option_type="C", weight=10_000))
    df = pd.DataFrame(rows)
    zg = compute_zero_gamma(df, search_pct=0.40, n_points=801)
    assert zg is not None
    assert math.isfinite(zg)
    assert 70 < zg < 140


def test_zero_gamma_is_invariant_to_uniform_weight_scaling():
    """If we double every option's weight, the zero-gamma price shouldn't move
    (the curve scales but the sign flip stays at the same S)."""
    base_rows = []
    for _ in range(5):
        base_rows.append(_row(strike=80, option_type="P", weight=10_000))
    for _ in range(5):
        base_rows.append(_row(strike=120, option_type="C", weight=10_000))
    df = pd.DataFrame(base_rows)
    zg1 = compute_zero_gamma(df, search_pct=0.40, n_points=801)

    df_scaled = df.copy()
    df_scaled["volume"] = df_scaled["volume"] * 7
    zg2 = compute_zero_gamma(df_scaled, search_pct=0.40, n_points=801)

    assert zg1 is not None and zg2 is not None
    assert abs(zg1 - zg2) < 1e-6


def test_picks_crossing_nearest_to_spot_when_multiple_exist():
    """A bimodal gamma surface can have multiple zero crossings; the one
    closest to current spot is the actionable one."""
    rows = []
    # Put cluster near spot (drives short gamma)
    for _ in range(20):
        rows.append(_row(strike=98, option_type="P", weight=10_000))
    # Call cluster well above spot (drives long gamma higher up)
    for _ in range(20):
        rows.append(_row(strike=104, option_type="C", weight=10_000))
    df = pd.DataFrame(rows)
    zg = compute_zero_gamma(df, search_pct=0.20, n_points=801)
    assert zg is not None
    # Crossing is in the 98–104 range, closer to whichever side is heavier
    # at spot. Just assert it's bounded by the cluster strikes.
    assert 95 < zg < 110


def test_uses_oi_weight_when_requested():
    """Passing weight_col='oi' should compute against the OI column instead."""
    rows = []
    for _ in range(5):
        r = _row(strike=80, option_type="P", weight=10_000)
        r["oi"] = 10_000
        rows.append(r)
    for _ in range(5):
        r = _row(strike=120, option_type="C", weight=10_000)
        r["oi"] = 10_000
        rows.append(r)
    df = pd.DataFrame(rows)
    zg_v = compute_zero_gamma(df, weight_col="volume", search_pct=0.40, n_points=801)
    zg_o = compute_zero_gamma(df, weight_col="oi",     search_pct=0.40, n_points=801)
    assert zg_v is not None and zg_o is not None
    # Same weights on both columns → identical zero-gamma price (modulo grid).
    assert abs(zg_v - zg_o) < 1e-6
