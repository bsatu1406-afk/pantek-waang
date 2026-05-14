"""Central source of truth for trading-session state.

Rev 4 — Agent 1: every other module that needs to know "are we in RTH
right now?" or "what is τ for a 0DTE contract right now?" imports from
this module. No one else computes RTH boundaries — that keeps the policy
in one place and removes the risk that the scheduler, the metrics, and
the API drift out of sync.

The default session is the US equity-index options session:

* **Open**:  09:30 America/New_York (inclusive)
* **Close**: 16:15 America/New_York (inclusive)

Options on SPX / NDX actually stop trading at 16:00 ET; the 15-minute
buffer is intentional so we still emit one or two "post-bell" frames so
end-of-day metrics (charm rate at τ → 0, final HIRO bucket) land cleanly
before the scheduler tells everyone the session is over.

Configuration via env vars (see :class:`app.config.Settings`):

* ``RTH_OPEN_TIME``  — ``HH:MM`` (default ``09:30``)
* ``RTH_CLOSE_TIME`` — ``HH:MM`` (default ``16:15``)

Holidays are sourced from the ``holidays`` package (pure-Python, no
native deps) using the ``NYSE`` calendar where available, falling back
to the US federal calendar. Half-day early-close dates **are** treated
as full holidays for now — the metrics for a half day are typically
noise on a normal trading day so we conservatively skip computation
rather than ship partial figures.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from typing import Final
from zoneinfo import ZoneInfo

import holidays  # type: ignore[import-untyped]

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


RTH_TZ: Final[ZoneInfo] = ZoneInfo("America/New_York")

# Days that SPXW / NDXP traditionally settle on. The CBOE has expanded
# this universe over the years; the M/W/F default matches the long-running
# weekly cycle. Tuesday + Thursday expirations are detected automatically
# when the symbol's listed contracts include them, but we don't *force*
# expiration-day analytics on those days unless a 0DTE chain actually exists.
_DEFAULT_EXPIRY_WEEKDAYS: Final[frozenset[int]] = frozenset({0, 2, 4})  # Mon, Wed, Fri


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _parse_hhmm(value: str, default: time) -> time:
    """Parse ``HH:MM`` into a ``time``; fall back to ``default`` on error."""
    try:
        hh, mm = value.split(":", 1)
        return time(int(hh), int(mm))
    except Exception:  # noqa: BLE001
        logger.warning(
            "session.bad_rth_time_format",
            value=value,
            fallback=default.strftime("%H:%M"),
        )
        return default


def _rth_window() -> tuple[time, time]:
    """Return the configured (open, close) times as :class:`datetime.time` pairs."""
    settings = get_settings()
    raw_open = getattr(settings, "rth_open_time", "09:30")
    raw_close = getattr(settings, "rth_close_time", "16:15")
    return (
        _parse_hhmm(raw_open, time(9, 30)),
        _parse_hhmm(raw_close, time(16, 15)),
    )


@lru_cache(maxsize=1)
def _us_market_holidays() -> holidays.HolidayBase:
    """Return a cached US market holiday calendar.

    We prefer the dedicated NYSE / CBOE calendars when the ``holidays``
    package version available exposes them; otherwise we fall back to the
    federal US calendar which is a strict superset for full-day closures.
    """
    nyse = getattr(holidays, "NYSE", None)
    if callable(nyse):
        try:
            return nyse(years=range(date.today().year - 1, date.today().year + 3))
        except Exception:  # noqa: BLE001
            pass
    return holidays.UnitedStates(  # type: ignore[no-any-return]
        years=range(date.today().year - 1, date.today().year + 3)
    )


def _is_business_day(d: date) -> bool:
    if d.weekday() >= 5:  # Sat / Sun
        return False
    return d not in _us_market_holidays()


def _now_eastern() -> datetime:
    """Current wall-clock in America/New_York (DST-aware)."""
    return datetime.now(tz=RTH_TZ)


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────


def is_rth_now(*, now: datetime | None = None) -> bool:
    """True if the current wall-clock time is inside RTH on a US business day.

    The optional ``now`` argument is for tests — passing a tz-aware datetime
    overrides the system clock. Naive datetimes are interpreted as already
    being in :data:`RTH_TZ`.
    """
    moment = now if now is not None else _now_eastern()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=RTH_TZ)
    else:
        moment = moment.astimezone(RTH_TZ)

    if not _is_business_day(moment.date()):
        return False

    open_t, close_t = _rth_window()
    current = moment.time()
    return open_t <= current <= close_t


def session_open_today(*, now: datetime | None = None) -> datetime:
    """Today's RTH open as a tz-aware ``datetime`` in America/New_York."""
    moment = now if now is not None else _now_eastern()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=RTH_TZ)
    else:
        moment = moment.astimezone(RTH_TZ)
    open_t, _ = _rth_window()
    return moment.replace(hour=open_t.hour, minute=open_t.minute, second=0, microsecond=0)


def session_close_today(*, now: datetime | None = None) -> datetime:
    """Today's RTH close as a tz-aware ``datetime`` in America/New_York."""
    moment = now if now is not None else _now_eastern()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=RTH_TZ)
    else:
        moment = moment.astimezone(RTH_TZ)
    _, close_t = _rth_window()
    return moment.replace(hour=close_t.hour, minute=close_t.minute, second=0, microsecond=0)


def minutes_to_close(*, now: datetime | None = None) -> float:
    """Minutes remaining until session close. Negative if already after close."""
    moment = now if now is not None else _now_eastern()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=RTH_TZ)
    else:
        moment = moment.astimezone(RTH_TZ)
    close = session_close_today(now=moment)
    return (close - moment).total_seconds() / 60.0


# 1 year ≈ 365.25 days × 24 h × 60 min × 60 s = 31_557_600 s. We use this
# convention throughout the Greek-pricing layer so τ for a 0DTE contract
# (intraday seconds-to-close) is expressed in the same year-fraction unit
# the BSM module already speaks.
SECONDS_PER_YEAR: Final[float] = 365.25 * 24 * 60 * 60


def time_to_expiry_0dte_years(*, now: datetime | None = None) -> float:
    """τ (years) for an option that expires at today's session close.

    * During RTH → ``(close - now) / SECONDS_PER_YEAR`` (always > 0)
    * After close → ``0.0`` (expired)
    * Before open → ``(close - open) / SECONDS_PER_YEAR`` (full session)
    * On weekends / holidays → ``0.0`` (no 0DTE today)
    """
    moment = now if now is not None else _now_eastern()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=RTH_TZ)
    else:
        moment = moment.astimezone(RTH_TZ)

    if not _is_business_day(moment.date()):
        return 0.0

    open_dt = session_open_today(now=moment)
    close_dt = session_close_today(now=moment)

    if moment >= close_dt:
        return 0.0
    if moment < open_dt:
        return max(0.0, (close_dt - open_dt).total_seconds() / SECONDS_PER_YEAR)
    return max(0.0, (close_dt - moment).total_seconds() / SECONDS_PER_YEAR)


def is_expiration_day(symbol: str, *, today: date | None = None) -> bool:
    """True if today is a regularly-scheduled expiration day for ``symbol``.

    SPXW / NDXP both list Mon/Wed/Fri weekly expirations on top of the
    standard monthly. We deliberately keep the rule conservative — if
    the calendar says today is a holiday, no 0DTE today; otherwise the
    weekday must be in :data:`_DEFAULT_EXPIRY_WEEKDAYS`.
    """
    today_d = today if today is not None else _now_eastern().date()
    if not _is_business_day(today_d):
        return False
    if symbol.upper() not in {"SPX", "SPXW", "NDX", "NDXP"}:
        # Other symbols not in scope for Rev 4; return False rather than
        # raise so callers can ask about any symbol without a try/except.
        return False
    return today_d.weekday() in _DEFAULT_EXPIRY_WEEKDAYS


def next_business_day(from_day: date | None = None) -> date:
    """Return the next US business day strictly after ``from_day``."""
    cur = (from_day or _now_eastern().date()) + timedelta(days=1)
    while not _is_business_day(cur):
        cur = cur + timedelta(days=1)
    return cur


def session_snapshot(*, now: datetime | None = None, symbol: str | None = None) -> dict[str, object]:
    """Build the ``session_state`` dict embedded in WebSocket frames.

    A single dict so the API surface stays stable; consumers (frontend
    Live dashboard, plugins) pick the fields they need.
    """
    moment = now if now is not None else _now_eastern()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=RTH_TZ)
    else:
        moment = moment.astimezone(RTH_TZ)

    payload: dict[str, object] = {
        "is_rth": is_rth_now(now=moment),
        "minutes_to_close": round(minutes_to_close(now=moment), 3),
        "time_to_expiry_0dte_years": time_to_expiry_0dte_years(now=moment),
        "session_open": session_open_today(now=moment).isoformat(),
        "session_close": session_close_today(now=moment).isoformat(),
        "now_eastern": moment.isoformat(),
    }
    if symbol is not None:
        payload["is_expiration_day"] = is_expiration_day(symbol, today=moment.date())
    return payload
