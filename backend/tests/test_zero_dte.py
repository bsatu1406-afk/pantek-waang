"""Rev 4 — 0DTE engine tests."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from app.processing.zero_dte import (
    compute_back_month_summary,
    compute_charm_decay_rate,
    compute_flip_speed,
    compute_zero_dte_summary,
    split_by_expiry,
)


def _chain(today: date, *, n_strikes: int = 7, spot: float = 5_000.0) -> pd.DataFrame:
    """A minimal chain with 0DTE rows + 1 back-month row per strike."""
    rows = []
    for kind, exp in (("0dte", today), ("back", today + timedelta(days=30))):
        for i in range(-n_strikes // 2 + 1, n_strikes // 2 + 1):
            strike = spot + 5 * i
            for opt_type, sign in (("C", +1.0), ("P", -1.0)):
                rows.append(
                    {
                        "strike": strike,
                        "expiration": pd.Timestamp(exp),
                        "option_type": opt_type,
                        "gamma": 0.002,
                        "delta": 0.5 * sign,
                        "iv": 0.20,
                        "oi": 1_000 if kind == "0dte" else 5_000,
                        "volume": 100,
                        "charm": 12.0,
                        "underlying_price": spot,
                        "vanna": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_split_by_expiry_basic() -> None:
    today = date(2026, 1, 15)
    df = _chain(today)
    zero, back = split_by_expiry(df, today=today)
    # We seeded 7 strikes × 2 option types = 14 per cohort.
    assert len(zero) == 14
    assert len(back) == 14
    # Every row on the 0DTE side has expiration == today
    assert all(pd.Timestamp(r).date() == today for r in zero["expiration"])


def test_split_by_expiry_empty_input() -> None:
    df = pd.DataFrame()
    zero, back = split_by_expiry(df, today=date(2026, 1, 15))
    assert zero.empty
    assert back.empty


def test_split_by_expiry_no_0dte_today() -> None:
    today = date(2026, 1, 15)
    df = _chain(today)
    other_day = date(2026, 1, 16)
    zero, back = split_by_expiry(df, today=other_day)
    assert zero.empty
    assert len(back) == 28


# ──────────────────────────────────────────────────────────────────────────


def test_compute_charm_decay_rate_atm_only() -> None:
    """Only ATM rows contribute to the rate."""
    today = date(2026, 1, 15)
    df = _chain(today)
    zero, _ = split_by_expiry(df, today=today)
    rate = compute_charm_decay_rate(zero, atm_band_pct=0.005)
    # Each ATM row has charm=12 ⇒ |12| / (365·24) ≈ 0.00137.
    assert rate > 0.0
    assert rate < 0.01


def test_compute_charm_decay_rate_empty() -> None:
    rate = compute_charm_decay_rate(pd.DataFrame())
    assert rate == 0.0


def test_compute_charm_decay_rate_no_atm_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "strike": 4_500.0,
                "underlying_price": 5_000.0,
                "charm": 12.0,
            }
        ]
    )
    rate = compute_charm_decay_rate(df, atm_band_pct=0.005)
    assert rate == 0.0


# ──────────────────────────────────────────────────────────────────────────


def test_compute_flip_speed_first_tick_is_zero() -> None:
    speed = compute_flip_speed(
        net_gex_now=1_000.0, net_gex_prev=None, elapsed_seconds=15.0
    )
    assert speed == 0.0


def test_compute_flip_speed_positive() -> None:
    speed = compute_flip_speed(
        net_gex_now=1_000.0, net_gex_prev=500.0, elapsed_seconds=10.0
    )
    assert speed == pytest.approx(50.0)


def test_compute_flip_speed_rejects_pathological_dt() -> None:
    assert compute_flip_speed(
        net_gex_now=1.0, net_gex_prev=0.0, elapsed_seconds=0.1
    ) == 0.0
    assert compute_flip_speed(
        net_gex_now=1.0, net_gex_prev=0.0, elapsed_seconds=float("nan")
    ) == 0.0


def test_compute_flip_speed_abs_value() -> None:
    speed = compute_flip_speed(
        net_gex_now=-100.0, net_gex_prev=200.0, elapsed_seconds=10.0
    )
    assert speed == pytest.approx(30.0)


# ──────────────────────────────────────────────────────────────────────────


def test_compute_zero_dte_summary_no_today_rows_writes_zero_placeholder() -> None:
    today = date(2026, 1, 15)
    df = _chain(today)
    summary = compute_zero_dte_summary(
        df, risk_free_rate=0.05, today=date(2026, 1, 14)
    )
    assert summary.has_0dte is False
    assert summary.gex_oi.net_total == 0.0
    assert summary.gex_vol.net_total == 0.0
    assert summary.charm.net_total == 0.0
    assert summary.charm_decay_rate == 0.0
    assert summary.flip_speed == 0.0


def test_compute_zero_dte_summary_has_rows_today() -> None:
    today = date(2026, 1, 15)
    df = _chain(today)
    summary = compute_zero_dte_summary(
        df, risk_free_rate=0.05, today=today
    )
    assert summary.has_0dte is True
    # gex curve must contain something
    assert len(summary.gex_oi.curve) > 0
    # decay rate computed on ATM-only rows
    assert summary.charm_decay_rate > 0.0


def test_compute_zero_dte_summary_flip_speed_with_prev() -> None:
    today = date(2026, 1, 15)
    df = _chain(today)
    summary = compute_zero_dte_summary(
        df,
        risk_free_rate=0.05,
        today=today,
        prev_net_gex=0.0,
        prev_ts_seconds=1000.0,
        now_ts_seconds=1015.0,
    )
    assert summary.flip_speed >= 0.0
    # With a non-zero net_gex and a fresh prev=0, flip_speed should equal
    # |net_gex_now| / 15.
    if summary.gex_oi.net_total != 0.0:
        expected = abs(summary.gex_oi.net_total) / 15.0
        assert summary.flip_speed == pytest.approx(expected, rel=1e-6)


def test_compute_back_month_summary_excludes_today() -> None:
    today = date(2026, 1, 15)
    df = _chain(today)
    bm = compute_back_month_summary(df, risk_free_rate=0.05, today=today)
    # Back month gex should be non-empty (we seeded 30-day expiry rows).
    assert len(bm.gex_oi.curve) > 0
