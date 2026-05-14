"""Tests for the Rev 4 RTH session lifecycle helpers."""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from app.processing.session import (
    SECONDS_PER_YEAR,
    is_expiration_day,
    is_rth_now,
    minutes_to_close,
    next_business_day,
    session_close_today,
    session_open_today,
    session_snapshot,
    time_to_expiry_0dte_years,
)

ET = ZoneInfo("America/New_York")


# Pick a known Monday (2026-05-18) and verify every helper.
MONDAY_11_00 = datetime(2026, 5, 18, 11, 0, tzinfo=ET)
MONDAY_09_29 = datetime(2026, 5, 18, 9, 29, tzinfo=ET)
MONDAY_09_30 = datetime(2026, 5, 18, 9, 30, tzinfo=ET)
MONDAY_16_15 = datetime(2026, 5, 18, 16, 15, tzinfo=ET)
MONDAY_17_00 = datetime(2026, 5, 18, 17, 0, tzinfo=ET)
SATURDAY_11_00 = datetime(2026, 5, 16, 11, 0, tzinfo=ET)
MEMORIAL_DAY = datetime(2026, 5, 25, 11, 0, tzinfo=ET)  # Mon holiday


# ──────────────────────────────────────────────────────────────────────────


def test_is_rth_now_inside_window() -> None:
    assert is_rth_now(now=MONDAY_11_00) is True


def test_is_rth_now_before_open() -> None:
    assert is_rth_now(now=MONDAY_09_29) is False


def test_is_rth_now_at_open_boundary_inclusive() -> None:
    assert is_rth_now(now=MONDAY_09_30) is True


def test_is_rth_now_at_close_boundary_inclusive() -> None:
    assert is_rth_now(now=MONDAY_16_15) is True


def test_is_rth_now_after_close() -> None:
    assert is_rth_now(now=MONDAY_17_00) is False


def test_is_rth_now_weekend() -> None:
    assert is_rth_now(now=SATURDAY_11_00) is False


def test_is_rth_now_holiday() -> None:
    # Memorial Day 2026 (May 25) is a Monday US federal holiday.
    assert is_rth_now(now=MEMORIAL_DAY) is False


def test_session_open_close_today_returns_aware_datetimes() -> None:
    open_dt = session_open_today(now=MONDAY_11_00)
    close_dt = session_close_today(now=MONDAY_11_00)
    assert open_dt.tzinfo is not None
    assert close_dt.tzinfo is not None
    assert open_dt.time() == time(9, 30)
    assert close_dt.time() == time(16, 15)


def test_minutes_to_close_basic() -> None:
    # 11:00 to 16:15 = 5h 15m = 315 minutes.
    assert minutes_to_close(now=MONDAY_11_00) == pytest.approx(315.0, abs=0.01)


def test_minutes_to_close_negative_after_close() -> None:
    assert minutes_to_close(now=MONDAY_17_00) < 0


# ──────────────────────────────────────────────────────────────────────────
# time_to_expiry_0dte_years
# ──────────────────────────────────────────────────────────────────────────


def test_time_to_expiry_0dte_at_open_equals_session_length() -> None:
    """At open the τ equals the entire session length expressed in years."""
    tau = time_to_expiry_0dte_years(now=MONDAY_09_30)
    expected_seconds = (16 * 3600 + 15 * 60) - (9 * 3600 + 30 * 60)
    assert tau == pytest.approx(expected_seconds / SECONDS_PER_YEAR, abs=1e-9)


def test_time_to_expiry_0dte_at_close_is_zero() -> None:
    assert time_to_expiry_0dte_years(now=MONDAY_16_15) == 0.0


def test_time_to_expiry_0dte_after_close_is_zero() -> None:
    assert time_to_expiry_0dte_years(now=MONDAY_17_00) == 0.0


def test_time_to_expiry_0dte_weekend_is_zero() -> None:
    assert time_to_expiry_0dte_years(now=SATURDAY_11_00) == 0.0


def test_time_to_expiry_0dte_before_open_is_full_session() -> None:
    """A snapshot taken at 08:00 ET should still return a full-session τ."""
    pre_open = datetime(2026, 5, 18, 8, 0, tzinfo=ET)
    full = time_to_expiry_0dte_years(now=pre_open)
    expected_seconds = (16 * 3600 + 15 * 60) - (9 * 3600 + 30 * 60)
    assert full == pytest.approx(expected_seconds / SECONDS_PER_YEAR, abs=1e-9)


def test_time_to_expiry_0dte_monotonic_throughout_session() -> None:
    """τ must monotonically decrease as the session progresses."""
    samples = [
        time_to_expiry_0dte_years(now=datetime(2026, 5, 18, h, 0, tzinfo=ET))
        for h in range(10, 17)
    ]
    for prev, cur in zip(samples, samples[1:], strict=False):
        assert cur < prev


# ──────────────────────────────────────────────────────────────────────────
# is_expiration_day
# ──────────────────────────────────────────────────────────────────────────


def test_is_expiration_day_monday_spxw() -> None:
    assert is_expiration_day("SPXW", today=date(2026, 5, 18)) is True


def test_is_expiration_day_tuesday_returns_false() -> None:
    # Mon/Wed/Fri schedule by default.
    assert is_expiration_day("SPXW", today=date(2026, 5, 19)) is False


def test_is_expiration_day_holiday_returns_false() -> None:
    assert is_expiration_day("SPXW", today=date(2026, 5, 25)) is False


def test_is_expiration_day_unknown_symbol_returns_false() -> None:
    assert is_expiration_day("AAPL", today=date(2026, 5, 18)) is False


def test_is_expiration_day_ndxp_friday() -> None:
    assert is_expiration_day("NDXP", today=date(2026, 5, 22)) is True


# ──────────────────────────────────────────────────────────────────────────


def test_next_business_day_skips_weekend() -> None:
    # Friday → next is Monday.
    fri = date(2026, 5, 22)
    assert next_business_day(fri) == date(2026, 5, 25) or next_business_day(fri) == date(2026, 5, 26)


def test_session_snapshot_includes_required_fields() -> None:
    snap = session_snapshot(now=MONDAY_11_00, symbol="SPXW")
    for key in (
        "is_rth",
        "minutes_to_close",
        "time_to_expiry_0dte_years",
        "session_open",
        "session_close",
        "now_eastern",
        "is_expiration_day",
    ):
        assert key in snap
    assert snap["is_rth"] is True
    assert snap["is_expiration_day"] is True
