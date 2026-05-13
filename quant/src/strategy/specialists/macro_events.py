"""Agent #2 — Macro & Event Specialist.

Multi-year US event calendar (2018-2025) + event-conditioned opportunistic
setups that emit Signals only when an event is in the relevant window.

Public API:
    EVENTS                 -- list[Event] (full calendar)
    EVENT_LOCKOUT_DATES    -- frozenset[date] (HIGH-impact full-day blacklist)
    is_event_day(d)        -> tuple[bool, str]
    event_distance(d)      -> dict[str, int]   (trading-day distance per kind)
    is_session_blocked(d)  -> bool
    MacroEventParams       -- dataclass
    generate_signals(bars, params) -> list[Signal]

The three opportunistic setups are:

    1. event_am_range_fade    -- post-CPI/PPI/NFP/ISM/GDP morning fade.
       After a high-impact pre-market release, the first 90 minutes of RTH
       tends to extend then mean-revert. We fade the 9:30-11:00 ET range
       direction at 11:00 ET with a tight stop on the range extreme.

    2. pre_fomc_drift_long    -- pre-FOMC announcement drift.
       Empirically, ES drifts up into the 2:00 PM ET FOMC announcement on
       most non-recession FOMC days. We enter long at 12:00 PM ET and exit
       at 1:50 PM ET (10 minutes before the announcement) so we never hold
       through the event itself.

    3. post_fomc_reaction_fade -- post-FOMC initial reaction fade.
       The initial 30-min reaction (2:00-2:30 PM ET) often gets retraced
       during the press conference (2:30-3:30 PM ET). We fade the direction
       at 2:30 PM ET, with stop at the 2:00-2:30 reaction extreme.

All entry times are converted ET -> UTC respecting daylight saving via
zoneinfo. Signals are emitted at the bar whose timestamp matches the
target ET wall-clock minute.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from functools import cache
from zoneinfo import ZoneInfo

import polars as pl

from src.common.types import Side, Signal
from src.strategy.specialists._interface import SpecialistParams

SPECIALIST_ID = "macro_events"

ET = ZoneInfo("America/New_York")
UTC = UTC


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """A single high-impact macro/calendar event."""

    d: date
    kind: str          # "FOMC", "FOMC_MINUTES", "CPI", "PPI", "NFP", "GDP",
                       # "ISM", "POWELL", "OPEX", "QUAD_WITCH", "HOLIDAY",
                       # "EARLY_CLOSE"
    impact: str        # "HIGH" | "MEDIUM" | "LOW"
    et_time: time | None = None
    source: str = ""


# ---------------------------------------------------------------------------
# Hard-coded calendars
# ---------------------------------------------------------------------------

# FOMC announcement dates (day 2 of each meeting). Sources: federalreserve.gov
# meeting calendars 2018-2025. Includes the 2020-03-03 + 2020-03-15 emergency
# inter-meeting rate cuts because they move ES violently.
FOMC_DATES: tuple[date, ...] = (
    date(2018, 1, 31), date(2018, 3, 21), date(2018, 5, 2),  date(2018, 6, 13),
    date(2018, 8, 1),  date(2018, 9, 26), date(2018, 11, 8), date(2018, 12, 19),
    date(2019, 1, 30), date(2019, 3, 20), date(2019, 5, 1),  date(2019, 6, 19),
    date(2019, 7, 31), date(2019, 9, 18), date(2019, 10, 30), date(2019, 12, 11),
    date(2020, 1, 29), date(2020, 3, 3),  date(2020, 3, 15), date(2020, 4, 29),
    date(2020, 6, 10), date(2020, 7, 29), date(2020, 9, 16), date(2020, 11, 5),
    date(2020, 12, 16),
    date(2021, 1, 27), date(2021, 3, 17), date(2021, 4, 28), date(2021, 6, 16),
    date(2021, 7, 28), date(2021, 9, 22), date(2021, 11, 3), date(2021, 12, 15),
    date(2022, 1, 26), date(2022, 3, 16), date(2022, 5, 4),  date(2022, 6, 15),
    date(2022, 7, 27), date(2022, 9, 21), date(2022, 11, 2), date(2022, 12, 14),
    date(2023, 2, 1),  date(2023, 3, 22), date(2023, 5, 3),  date(2023, 6, 14),
    date(2023, 7, 26), date(2023, 9, 20), date(2023, 11, 1), date(2023, 12, 13),
    date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1),  date(2024, 6, 12),
    date(2024, 7, 31), date(2024, 9, 18), date(2024, 11, 7), date(2024, 12, 18),
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),  date(2025, 6, 18),
    date(2025, 7, 30), date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 10),
)

# FOMC minutes release dates. Released ~3 weeks after each meeting at 2:00 PM ET.
# Source: federalreserve.gov "Minutes of the FOMC".
FOMC_MINUTES_DATES: tuple[date, ...] = (
    date(2018, 2, 21), date(2018, 4, 11), date(2018, 5, 23), date(2018, 7, 5),
    date(2018, 8, 22), date(2018, 10, 17), date(2018, 11, 29), date(2019, 1, 9),
    date(2019, 2, 20), date(2019, 4, 10), date(2019, 5, 22), date(2019, 7, 10),
    date(2019, 8, 21), date(2019, 10, 9), date(2019, 11, 20), date(2020, 1, 3),
    date(2020, 2, 19), date(2020, 4, 8),  date(2020, 5, 20), date(2020, 7, 1),
    date(2020, 8, 19), date(2020, 10, 7), date(2020, 11, 25), date(2021, 1, 6),
    date(2021, 2, 17), date(2021, 4, 7),  date(2021, 5, 19), date(2021, 7, 7),
    date(2021, 8, 18), date(2021, 10, 13), date(2021, 11, 24), date(2022, 1, 5),
    date(2022, 2, 16), date(2022, 4, 6),  date(2022, 5, 25), date(2022, 7, 6),
    date(2022, 8, 17), date(2022, 10, 12), date(2022, 11, 23), date(2023, 1, 4),
    date(2023, 2, 22), date(2023, 4, 12), date(2023, 5, 24), date(2023, 7, 5),
    date(2023, 8, 16), date(2023, 10, 11), date(2023, 11, 21), date(2024, 1, 3),
    date(2024, 2, 21), date(2024, 4, 10), date(2024, 5, 22), date(2024, 7, 3),
    date(2024, 8, 21), date(2024, 10, 9), date(2024, 11, 26), date(2025, 1, 8),
    date(2025, 2, 19), date(2025, 4, 9),  date(2025, 5, 28), date(2025, 7, 9),
    date(2025, 8, 20), date(2025, 10, 8), date(2025, 11, 19),
)

# US CPI release dates (BLS). 8:30 AM ET.
CPI_DATES: tuple[date, ...] = (
    date(2018, 1, 12), date(2018, 2, 14), date(2018, 3, 13), date(2018, 4, 11),
    date(2018, 5, 10), date(2018, 6, 12), date(2018, 7, 12), date(2018, 8, 10),
    date(2018, 9, 13), date(2018, 10, 11), date(2018, 11, 14), date(2018, 12, 12),
    date(2019, 1, 11), date(2019, 2, 13), date(2019, 3, 12), date(2019, 4, 10),
    date(2019, 5, 10), date(2019, 6, 12), date(2019, 7, 11), date(2019, 8, 13),
    date(2019, 9, 12), date(2019, 10, 10), date(2019, 11, 13), date(2019, 12, 11),
    date(2020, 1, 14), date(2020, 2, 13), date(2020, 3, 11), date(2020, 4, 10),
    date(2020, 5, 12), date(2020, 6, 10), date(2020, 7, 14), date(2020, 8, 12),
    date(2020, 9, 11), date(2020, 10, 13), date(2020, 11, 12), date(2020, 12, 10),
    date(2021, 1, 13), date(2021, 2, 10), date(2021, 3, 10), date(2021, 4, 13),
    date(2021, 5, 12), date(2021, 6, 10), date(2021, 7, 13), date(2021, 8, 11),
    date(2021, 9, 14), date(2021, 10, 13), date(2021, 11, 10), date(2021, 12, 10),
    date(2022, 1, 12), date(2022, 2, 10), date(2022, 3, 10), date(2022, 4, 12),
    date(2022, 5, 11), date(2022, 6, 10), date(2022, 7, 13), date(2022, 8, 10),
    date(2022, 9, 13), date(2022, 10, 13), date(2022, 11, 10), date(2022, 12, 13),
    date(2023, 1, 12), date(2023, 2, 14), date(2023, 3, 14), date(2023, 4, 12),
    date(2023, 5, 10), date(2023, 6, 13), date(2023, 7, 12), date(2023, 8, 10),
    date(2023, 9, 13), date(2023, 10, 12), date(2023, 11, 14), date(2023, 12, 12),
    date(2024, 1, 11), date(2024, 2, 13), date(2024, 3, 12), date(2024, 4, 10),
    date(2024, 5, 15), date(2024, 6, 12), date(2024, 7, 11), date(2024, 8, 14),
    date(2024, 9, 11), date(2024, 10, 10), date(2024, 11, 13), date(2024, 12, 11),
    date(2025, 1, 15), date(2025, 2, 12), date(2025, 3, 12), date(2025, 4, 10),
    date(2025, 5, 13), date(2025, 6, 11), date(2025, 7, 15), date(2025, 8, 12),
    date(2025, 9, 11), date(2025, 10, 24), date(2025, 11, 13), date(2025, 12, 10),
)

# US PPI release dates (BLS). 8:30 AM ET. Usually 1 day before/after CPI.
PPI_DATES: tuple[date, ...] = (
    date(2018, 1, 11), date(2018, 2, 15), date(2018, 3, 14), date(2018, 4, 10),
    date(2018, 5, 9),  date(2018, 6, 13), date(2018, 7, 11), date(2018, 8, 9),
    date(2018, 9, 12), date(2018, 10, 10), date(2018, 11, 9), date(2018, 12, 11),
    date(2019, 1, 15), date(2019, 2, 14), date(2019, 3, 13), date(2019, 4, 11),
    date(2019, 5, 9),  date(2019, 6, 11), date(2019, 7, 12), date(2019, 8, 9),
    date(2019, 9, 11), date(2019, 10, 8), date(2019, 11, 14), date(2019, 12, 12),
    date(2020, 1, 15), date(2020, 2, 6),  date(2020, 3, 12), date(2020, 4, 9),
    date(2020, 5, 8),  date(2020, 6, 11), date(2020, 7, 10), date(2020, 8, 11),
    date(2020, 9, 10), date(2020, 10, 14), date(2020, 11, 13), date(2020, 12, 11),
    date(2021, 1, 15), date(2021, 2, 17), date(2021, 3, 12), date(2021, 4, 9),
    date(2021, 5, 13), date(2021, 6, 15), date(2021, 7, 14), date(2021, 8, 12),
    date(2021, 9, 10), date(2021, 10, 14), date(2021, 11, 9), date(2021, 12, 14),
    date(2022, 1, 13), date(2022, 2, 15), date(2022, 3, 15), date(2022, 4, 13),
    date(2022, 5, 12), date(2022, 6, 14), date(2022, 7, 14), date(2022, 8, 11),
    date(2022, 9, 14), date(2022, 10, 12), date(2022, 11, 15), date(2022, 12, 9),
    date(2023, 1, 18), date(2023, 2, 16), date(2023, 3, 15), date(2023, 4, 13),
    date(2023, 5, 11), date(2023, 6, 14), date(2023, 7, 13), date(2023, 8, 11),
    date(2023, 9, 14), date(2023, 10, 11), date(2023, 11, 15), date(2023, 12, 13),
    date(2024, 1, 12), date(2024, 2, 16), date(2024, 3, 14), date(2024, 4, 11),
    date(2024, 5, 14), date(2024, 6, 13), date(2024, 7, 12), date(2024, 8, 13),
    date(2024, 9, 12), date(2024, 10, 11), date(2024, 11, 14), date(2024, 12, 12),
    date(2025, 1, 14), date(2025, 2, 13), date(2025, 3, 13), date(2025, 4, 11),
    date(2025, 5, 15), date(2025, 6, 12), date(2025, 7, 16), date(2025, 8, 14),
    date(2025, 9, 10), date(2025, 10, 16), date(2025, 11, 14), date(2025, 12, 11),
)

# US ISM Manufacturing PMI release dates. 10:00 AM ET, first business day of month.
ISM_MFG_DATES: tuple[date, ...] = (
    date(2018, 1, 2),  date(2018, 2, 1),  date(2018, 3, 1),  date(2018, 4, 2),
    date(2018, 5, 1),  date(2018, 6, 1),  date(2018, 7, 2),  date(2018, 8, 1),
    date(2018, 9, 4),  date(2018, 10, 1), date(2018, 11, 1), date(2018, 12, 3),
    date(2019, 1, 3),  date(2019, 2, 1),  date(2019, 3, 1),  date(2019, 4, 1),
    date(2019, 5, 1),  date(2019, 6, 3),  date(2019, 7, 1),  date(2019, 8, 1),
    date(2019, 9, 3),  date(2019, 10, 1), date(2019, 11, 1), date(2019, 12, 2),
    date(2020, 1, 3),  date(2020, 2, 3),  date(2020, 3, 2),  date(2020, 4, 1),
    date(2020, 5, 1),  date(2020, 6, 1),  date(2020, 7, 1),  date(2020, 8, 3),
    date(2020, 9, 1),  date(2020, 10, 1), date(2020, 11, 2), date(2020, 12, 1),
    date(2021, 1, 5),  date(2021, 2, 1),  date(2021, 3, 1),  date(2021, 4, 1),
    date(2021, 5, 3),  date(2021, 6, 1),  date(2021, 7, 1),  date(2021, 8, 2),
    date(2021, 9, 1),  date(2021, 10, 1), date(2021, 11, 1), date(2021, 12, 1),
    date(2022, 1, 4),  date(2022, 2, 1),  date(2022, 3, 1),  date(2022, 4, 1),
    date(2022, 5, 2),  date(2022, 6, 1),  date(2022, 7, 1),  date(2022, 8, 1),
    date(2022, 9, 1),  date(2022, 10, 3), date(2022, 11, 1), date(2022, 12, 1),
    date(2023, 1, 4),  date(2023, 2, 1),  date(2023, 3, 1),  date(2023, 4, 3),
    date(2023, 5, 1),  date(2023, 6, 1),  date(2023, 7, 3),  date(2023, 8, 1),
    date(2023, 9, 1),  date(2023, 10, 2), date(2023, 11, 1), date(2023, 12, 1),
    date(2024, 1, 3),  date(2024, 2, 1),  date(2024, 3, 1),  date(2024, 4, 1),
    date(2024, 5, 1),  date(2024, 6, 3),  date(2024, 7, 1),  date(2024, 8, 1),
    date(2024, 9, 3),  date(2024, 10, 1), date(2024, 11, 1), date(2024, 12, 2),
    date(2025, 1, 3),  date(2025, 2, 3),  date(2025, 3, 3),  date(2025, 4, 1),
    date(2025, 5, 1),  date(2025, 6, 2),  date(2025, 7, 1),  date(2025, 8, 1),
    date(2025, 9, 2),  date(2025, 10, 1), date(2025, 11, 3), date(2025, 12, 1),
)

# GDP advance / 2nd / 3rd estimate release dates (BEA). 8:30 AM ET.
GDP_DATES: tuple[date, ...] = (
    date(2018, 1, 26), date(2018, 2, 28), date(2018, 3, 28),
    date(2018, 4, 27), date(2018, 5, 30), date(2018, 6, 28),
    date(2018, 7, 27), date(2018, 8, 29), date(2018, 9, 27),
    date(2018, 10, 26), date(2018, 11, 28), date(2018, 12, 21),
    date(2019, 2, 28), date(2019, 3, 28),
    date(2019, 4, 26), date(2019, 5, 30), date(2019, 6, 27),
    date(2019, 7, 26), date(2019, 8, 29), date(2019, 9, 26),
    date(2019, 10, 30), date(2019, 11, 27), date(2019, 12, 20),
    date(2020, 1, 30), date(2020, 2, 27), date(2020, 3, 26),
    date(2020, 4, 29), date(2020, 5, 28), date(2020, 6, 25),
    date(2020, 7, 30), date(2020, 8, 27), date(2020, 9, 30),
    date(2020, 10, 29), date(2020, 11, 25), date(2020, 12, 22),
    date(2021, 1, 28), date(2021, 2, 25), date(2021, 3, 25),
    date(2021, 4, 29), date(2021, 5, 27), date(2021, 6, 24),
    date(2021, 7, 29), date(2021, 8, 26), date(2021, 9, 30),
    date(2021, 10, 28), date(2021, 11, 24), date(2021, 12, 22),
    date(2022, 1, 27), date(2022, 2, 24), date(2022, 3, 30),
    date(2022, 4, 28), date(2022, 5, 26), date(2022, 6, 29),
    date(2022, 7, 28), date(2022, 8, 25), date(2022, 9, 29),
    date(2022, 10, 27), date(2022, 11, 30), date(2022, 12, 22),
    date(2023, 1, 26), date(2023, 2, 23), date(2023, 3, 30),
    date(2023, 4, 27), date(2023, 5, 25), date(2023, 6, 29),
    date(2023, 7, 27), date(2023, 8, 30), date(2023, 9, 28),
    date(2023, 10, 26), date(2023, 11, 29), date(2023, 12, 21),
    date(2024, 1, 25), date(2024, 2, 28), date(2024, 3, 28),
    date(2024, 4, 25), date(2024, 5, 30), date(2024, 6, 27),
    date(2024, 7, 25), date(2024, 8, 29), date(2024, 9, 26),
    date(2024, 10, 30), date(2024, 11, 27), date(2024, 12, 19),
    date(2025, 1, 30), date(2025, 2, 27), date(2025, 3, 27),
    date(2025, 4, 30), date(2025, 5, 29), date(2025, 6, 26),
    date(2025, 7, 30), date(2025, 8, 28), date(2025, 9, 25),
    date(2025, 10, 30), date(2025, 11, 26), date(2025, 12, 18),
)

# NYSE/CME full-day holidays 2018-2025 (US equity-index futures halt at 16:00 ET
# the day before, no RTH next day).
US_HOLIDAYS: tuple[date, ...] = (
    date(2018, 1, 1),  date(2018, 1, 15), date(2018, 2, 19), date(2018, 3, 30),
    date(2018, 5, 28), date(2018, 7, 4),  date(2018, 9, 3),  date(2018, 11, 22),
    date(2018, 12, 5), date(2018, 12, 25),
    date(2019, 1, 1),  date(2019, 1, 21), date(2019, 2, 18), date(2019, 4, 19),
    date(2019, 5, 27), date(2019, 7, 4),  date(2019, 9, 2),  date(2019, 11, 28),
    date(2019, 12, 25),
    date(2020, 1, 1),  date(2020, 1, 20), date(2020, 2, 17), date(2020, 4, 10),
    date(2020, 5, 25), date(2020, 7, 3),  date(2020, 9, 7),  date(2020, 11, 26),
    date(2020, 12, 25),
    date(2021, 1, 1),  date(2021, 1, 18), date(2021, 2, 15), date(2021, 4, 2),
    date(2021, 5, 31), date(2021, 7, 5),  date(2021, 9, 6),  date(2021, 11, 25),
    date(2021, 12, 24),
    date(2022, 1, 17), date(2022, 2, 21), date(2022, 4, 15), date(2022, 5, 30),
    date(2022, 6, 20), date(2022, 7, 4),  date(2022, 9, 5),  date(2022, 11, 24),
    date(2022, 12, 26),
    date(2023, 1, 2),  date(2023, 1, 16), date(2023, 2, 20), date(2023, 4, 7),
    date(2023, 5, 29), date(2023, 6, 19), date(2023, 7, 4),  date(2023, 9, 4),
    date(2023, 11, 23), date(2023, 12, 25),
    date(2024, 1, 1),  date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29),
    date(2024, 5, 27), date(2024, 6, 19), date(2024, 7, 4),  date(2024, 9, 2),
    date(2024, 11, 28), date(2024, 12, 25),
    date(2025, 1, 1),  date(2025, 1, 9),  date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4),
    date(2025, 9, 1),  date(2025, 11, 27), date(2025, 12, 25),
)

# NYSE early-close days (close at 13:00 ET). Day before Independence Day,
# Black Friday, Christmas Eve (when not a full holiday).
US_EARLY_CLOSE_DAYS: tuple[date, ...] = (
    date(2018, 7, 3),  date(2018, 11, 23), date(2018, 12, 24),
    date(2019, 7, 3),  date(2019, 11, 29), date(2019, 12, 24),
    date(2020, 11, 27), date(2020, 12, 24),
    date(2021, 11, 26),
    date(2022, 11, 25),
    date(2023, 7, 3),  date(2023, 11, 24),
    date(2024, 7, 3),  date(2024, 11, 29), date(2024, 12, 24),
    date(2025, 7, 3),  date(2025, 11, 28), date(2025, 12, 24),
)


# ---------------------------------------------------------------------------
# Calendar helpers (pure date arithmetic)
# ---------------------------------------------------------------------------


def _first_friday(y: int, m: int) -> date:
    d = date(y, m, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def _nth_weekday(y: int, m: int, weekday: int, n: int) -> date:
    """n-th `weekday` (Mon=0..Sun=6) of (y, m)."""
    d = date(y, m, 1)
    first = d + timedelta(days=(weekday - d.weekday()) % 7)
    return first + timedelta(days=7 * (n - 1))


def _nfp_dates(y_start: int, y_end: int) -> list[date]:
    out: list[date] = []
    for y in range(y_start, y_end + 1):
        for m in range(1, 13):
            d = _first_friday(y, m)
            # If first Friday is also Jul 4 / Jan 1 holiday, NFP shifts to Thu.
            if d == date(2020, 9, 4) or d == date(2021, 7, 2):
                pass
            if d in (date(2025, 7, 4),):
                d = date(2025, 7, 3)
            out.append(d)
    return out


def _opex_dates(y_start: int, y_end: int) -> list[date]:
    return [_nth_weekday(y, m, 4, 3) for y in range(y_start, y_end + 1)
            for m in range(1, 13)]


def _quad_witching_dates(y_start: int, y_end: int) -> list[date]:
    return [_nth_weekday(y, m, 4, 3) for y in range(y_start, y_end + 1)
            for m in (3, 6, 9, 12)]


# ---------------------------------------------------------------------------
# Event registry
# ---------------------------------------------------------------------------


def _build_events() -> list[Event]:
    events: list[Event] = []
    for d in FOMC_DATES:
        events.append(Event(d, "FOMC", "HIGH", time(14, 0), "federalreserve.gov"))
    for d in FOMC_MINUTES_DATES:
        events.append(Event(d, "FOMC_MINUTES", "MEDIUM", time(14, 0),
                            "federalreserve.gov"))
    for d in CPI_DATES:
        events.append(Event(d, "CPI", "HIGH", time(8, 30), "bls.gov/cpi"))
    for d in PPI_DATES:
        events.append(Event(d, "PPI", "MEDIUM", time(8, 30), "bls.gov/ppi"))
    for d in _nfp_dates(2018, 2025):
        events.append(Event(d, "NFP", "HIGH", time(8, 30), "bls.gov/empsit"))
    for d in GDP_DATES:
        events.append(Event(d, "GDP", "HIGH", time(8, 30), "bea.gov"))
    for d in ISM_MFG_DATES:
        events.append(Event(d, "ISM", "MEDIUM", time(10, 0), "ismworld.org"))
    for d in _opex_dates(2018, 2025):
        kind = "QUAD_WITCH" if d.month in (3, 6, 9, 12) else "OPEX"
        events.append(Event(d, kind, "MEDIUM", time(16, 0), "cboe.com"))
    for d in US_HOLIDAYS:
        events.append(Event(d, "HOLIDAY", "HIGH", None, "nyse.com"))
    for d in US_EARLY_CLOSE_DAYS:
        events.append(Event(d, "EARLY_CLOSE", "MEDIUM", time(13, 0), "nyse.com"))
    return events


EVENTS: list[Event] = _build_events()


# Full-day blacklist: any HIGH-impact event => blocked for non-event-specific
# specialists. (Our own setups bypass this — they explicitly trade event days.)
EVENT_LOCKOUT_DATES: frozenset[date] = frozenset(
    e.d for e in EVENTS if e.impact == "HIGH"
)


@cache
def _events_by_date() -> dict[date, list[Event]]:
    out: dict[date, list[Event]] = {}
    for e in EVENTS:
        out.setdefault(e.d, []).append(e)
    return out


@cache
def _dates_by_kind() -> dict[str, list[date]]:
    out: dict[str, list[date]] = {}
    for e in EVENTS:
        out.setdefault(e.kind, []).append(e.d)
    for k in out:
        out[k].sort()
    return out


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_event_day(d: date) -> tuple[bool, str]:
    """Return (True, highest-impact-kind) if `d` carries a high-impact event."""
    items = _events_by_date().get(d, [])
    if not items:
        return (False, "")
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    items_sorted = sorted(items, key=lambda e: order.get(e.impact, 9))
    top = items_sorted[0]
    if top.impact != "HIGH":
        return (False, top.kind)
    return (True, top.kind)


def is_session_blocked(d: date) -> bool:
    """True iff `d` is a HIGH-impact event day (e.g. FOMC, CPI, NFP, holiday)."""
    return d in EVENT_LOCKOUT_DATES


def _trading_days_distance(a: date, b: date) -> int:
    """Signed trading-day distance from `a` to `b` (excludes weekends + holidays).

    Positive => b is in the future relative to a. Negative => b is in the past.
    """
    if a == b:
        return 0
    lo, hi = (a, b) if a < b else (b, a)
    cur = lo
    n = 0
    holidays = set(US_HOLIDAYS)
    while cur < hi:
        cur += timedelta(days=1)
        if cur.weekday() >= 5 or cur in holidays:
            continue
        n += 1
    return n if a < b else -n


def event_distance(d: date) -> dict[str, int]:
    """Trading-day distance from `d` to the nearest event of each kind.

    Distance is negative if the nearest event is in the past, positive if in
    the future, zero if today.

    Returned kinds: fomc, fomc_minutes, cpi, ppi, nfp, gdp, ism, opex,
    quad_witch.
    """
    out: dict[str, int] = {}
    kind_map = {
        "FOMC": "fomc",
        "FOMC_MINUTES": "fomc_minutes",
        "CPI": "cpi",
        "PPI": "ppi",
        "NFP": "nfp",
        "GDP": "gdp",
        "ISM": "ism",
        "OPEX": "opex",
        "QUAD_WITCH": "quad_witch",
    }
    by_kind = _dates_by_kind()
    for raw_kind, short in kind_map.items():
        dates_for_kind = by_kind.get(raw_kind, [])
        if not dates_for_kind:
            continue
        # nearest by absolute calendar distance, then signed by direction
        nearest = min(dates_for_kind, key=lambda x: abs((x - d).days))
        out[short] = _trading_days_distance(d, nearest)
    return out


# ---------------------------------------------------------------------------
# Specialist params + signal generation
# ---------------------------------------------------------------------------


@dataclass
class MacroEventParams(SpecialistParams):
    specialist_id: str = SPECIALIST_ID

    # Setup 1: post-event AM range fade (CPI/PPI/NFP/ISM/GDP days)
    am_fade_entry_et: time = time(11, 0)         # entry bar (ET wall-clock)
    am_fade_min_range_pts: float = 4.0           # need at least this 9:30-11:00 range
    am_fade_max_range_pts: float = 30.0          # but not totally blown-out (no-trade)
    am_fade_stop_buffer_pts: float = 2.0         # stop = range extreme +/- buffer
    am_fade_target_pct_of_range: float = 0.5     # target = entry +/- (range * pct)
    am_fade_confidence: float = 0.55

    # Setup 2: pre-FOMC drift long (FOMC days)
    pre_fomc_entry_et: time = time(12, 0)
    pre_fomc_stop_pts: float = 4.0
    pre_fomc_target_pts: float = 6.0
    pre_fomc_min_pre_drift_pts: float = -8.0     # only buy if not already up >+8
    pre_fomc_max_pre_drift_pts: float = 6.0
    pre_fomc_confidence: float = 0.60

    # Setup 3: post-FOMC initial-reaction fade
    post_fomc_entry_et: time = time(14, 30)
    post_fomc_min_reaction_pts: float = 5.0
    post_fomc_stop_buffer_pts: float = 2.0
    post_fomc_target_pct: float = 0.6            # fade target = 60% retrace of reaction
    post_fomc_confidence: float = 0.55

    # Universal
    max_signals_per_day: int = 3


# ---------------------------------------------------------------------------
# Internal helpers for generate_signals
# ---------------------------------------------------------------------------


def _et_to_utc(d: date, t: time) -> datetime:
    """Convert ET wall-clock (d, t) to a UTC tz-aware datetime."""
    return datetime.combine(d, t, tzinfo=ET).astimezone(UTC)


def _bar_at_or_after(bars: pl.DataFrame, ts: datetime) -> dict | None:
    """Return first bar row whose ts is >= `ts`, else None."""
    df = bars.filter(pl.col("ts") >= ts).head(1)
    if df.is_empty():
        return None
    return df.to_dicts()[0]


def _range_between(
    bars: pl.DataFrame, start_ts: datetime, end_ts: datetime,
) -> tuple[float, float, float, float] | None:
    """Return (range_open, range_high, range_low, range_close) over [start, end)."""
    df = bars.filter((pl.col("ts") >= start_ts) & (pl.col("ts") < end_ts))
    if df.is_empty():
        return None
    rng_open = float(df["open"][0])
    rng_close = float(df["close"][-1])
    rng_high = float(df["high"].max())
    rng_low = float(df["low"].min())
    return rng_open, rng_high, rng_low, rng_close


def _session_date_from_bars(bars: pl.DataFrame) -> date | None:
    """Best-effort extract the session_date scalar from a single-day bars frame."""
    if "session_date" in bars.columns and bars.height > 0:
        v = bars["session_date"][0]
        if isinstance(v, date):
            return v
        if isinstance(v, datetime):
            return v.date()
    # fallback: derive from first ts in ET
    if bars.height == 0:
        return None
    first_ts = bars["ts"][0]
    if isinstance(first_ts, datetime):
        return first_ts.astimezone(ET).date()
    return None


# ---------------------------------------------------------------------------
# Setups
# ---------------------------------------------------------------------------


def _setup_am_range_fade(
    bars: pl.DataFrame, sd: date, params: MacroEventParams, kind: str,
) -> Signal | None:
    """Fade the 9:30-11:00 ET range direction on a morning-release event day."""
    rng_start = _et_to_utc(sd, time(9, 30))
    rng_end = _et_to_utc(sd, params.am_fade_entry_et)
    rng = _range_between(bars, rng_start, rng_end)
    if rng is None:
        return None
    rng_open, rng_high, rng_low, rng_close = rng

    pts_move = rng_close - rng_open
    pts_range = rng_high - rng_low
    if pts_range < params.am_fade_min_range_pts:
        return None
    if pts_range > params.am_fade_max_range_pts:
        return None  # likely a tape-bomb session, fade is unsafe

    entry_row = _bar_at_or_after(bars, rng_end)
    if entry_row is None:
        return None
    entry_ts: datetime = entry_row["ts"]
    entry_px = float(entry_row["close"])

    if pts_move > 0:
        # market rallied into 11:00; fade short toward the range midpoint
        side = Side.SHORT
        stop = rng_high + params.am_fade_stop_buffer_pts
        target = entry_px - pts_range * params.am_fade_target_pct_of_range
        setup_id = f"event_am_range_fade_short_{kind.lower()}"
    elif pts_move < 0:
        side = Side.LONG
        stop = rng_low - params.am_fade_stop_buffer_pts
        target = entry_px + pts_range * params.am_fade_target_pct_of_range
        setup_id = f"event_am_range_fade_long_{kind.lower()}"
    else:
        return None

    return Signal(
        timestamp=entry_ts,
        side=side,
        confidence=params.am_fade_confidence,
        specialist=SPECIALIST_ID,
        setup_id=setup_id,
        entry_price=entry_px,
        stop_price=float(stop),
        target_price=float(target),
        metadata={
            "event_kind": kind,
            "range_open": rng_open,
            "range_high": rng_high,
            "range_low": rng_low,
            "range_close": rng_close,
            "range_move_pts": float(pts_move),
            "range_pts": float(pts_range),
        },
    )


def _setup_pre_fomc_drift(
    bars: pl.DataFrame, sd: date, params: MacroEventParams,
) -> Signal | None:
    """Long ES at 12:00 ET on FOMC day, exit before the 14:00 ET announcement."""
    entry_ts_target = _et_to_utc(sd, params.pre_fomc_entry_et)
    entry_row = _bar_at_or_after(bars, entry_ts_target)
    if entry_row is None:
        return None
    open_ts = _et_to_utc(sd, time(9, 30))
    open_row = _bar_at_or_after(bars, open_ts)
    if open_row is None:
        return None

    entry_px = float(entry_row["close"])
    open_px = float(open_row["open"])
    pre_drift = entry_px - open_px
    if pre_drift < params.pre_fomc_min_pre_drift_pts:
        return None  # already crashed pre-event; trend likely down
    if pre_drift > params.pre_fomc_max_pre_drift_pts:
        return None  # already rallied; less edge

    stop = entry_px - params.pre_fomc_stop_pts
    target = entry_px + params.pre_fomc_target_pts
    return Signal(
        timestamp=entry_row["ts"],
        side=Side.LONG,
        confidence=params.pre_fomc_confidence,
        specialist=SPECIALIST_ID,
        setup_id="pre_fomc_drift_long",
        entry_price=entry_px,
        stop_price=float(stop),
        target_price=float(target),
        metadata={
            "event_kind": "FOMC",
            "pre_drift_pts": float(pre_drift),
            "exit_by_et": "13:50",
        },
    )


def _setup_post_fomc_fade(
    bars: pl.DataFrame, sd: date, params: MacroEventParams,
) -> Signal | None:
    """Fade the initial 14:00-14:30 ET FOMC reaction at 14:30 ET."""
    react_start = _et_to_utc(sd, time(14, 0))
    react_end = _et_to_utc(sd, params.post_fomc_entry_et)
    rng = _range_between(bars, react_start, react_end)
    if rng is None:
        return None
    r_open, r_high, r_low, r_close = rng
    reaction = r_close - r_open
    if abs(reaction) < params.post_fomc_min_reaction_pts:
        return None

    entry_row = _bar_at_or_after(bars, react_end)
    if entry_row is None:
        return None
    entry_px = float(entry_row["close"])

    if reaction > 0:
        side = Side.SHORT
        stop = r_high + params.post_fomc_stop_buffer_pts
        target = entry_px - abs(reaction) * params.post_fomc_target_pct
        setup_id = "post_fomc_reaction_fade_short"
    else:
        side = Side.LONG
        stop = r_low - params.post_fomc_stop_buffer_pts
        target = entry_px + abs(reaction) * params.post_fomc_target_pct
        setup_id = "post_fomc_reaction_fade_long"

    return Signal(
        timestamp=entry_row["ts"],
        side=side,
        confidence=params.post_fomc_confidence,
        specialist=SPECIALIST_ID,
        setup_id=setup_id,
        entry_price=entry_px,
        stop_price=float(stop),
        target_price=float(target),
        metadata={
            "event_kind": "FOMC",
            "reaction_open": r_open,
            "reaction_high": r_high,
            "reaction_low": r_low,
            "reaction_close": r_close,
            "reaction_pts": float(reaction),
        },
    )


# ---------------------------------------------------------------------------
# Public signal generator
# ---------------------------------------------------------------------------


def generate_signals(
    bars: pl.DataFrame, params: MacroEventParams,
) -> list[Signal]:
    """Emit event-conditioned opportunistic signals for a single session day.

    The caller is expected to hand us ONE session day's bars (RTH only,
    1-second). We pick the session_date, look up which events fall on that
    day, and run the corresponding setup(s).
    """
    if bars.is_empty():
        return []
    sd = _session_date_from_bars(bars)
    if sd is None:
        return []

    items = _events_by_date().get(sd, [])
    if not items:
        return []
    kinds = {e.kind for e in items}

    # Skip full-close holidays — no RTH to trade.
    if "HOLIDAY" in kinds:
        return []

    signals: list[Signal] = []

    # Setup 1: morning-release fade for CPI/PPI/NFP/GDP/ISM days.
    morning_event_kinds = {"CPI", "PPI", "NFP", "GDP", "ISM"}
    am_kinds = kinds & morning_event_kinds
    if am_kinds:
        primary = "CPI" if "CPI" in am_kinds else (
            "NFP" if "NFP" in am_kinds else sorted(am_kinds)[0]
        )
        sig = _setup_am_range_fade(bars, sd, params, primary)
        if sig is not None:
            signals.append(sig)

    # Setup 2 + 3: FOMC day.
    if "FOMC" in kinds:
        pre = _setup_pre_fomc_drift(bars, sd, params)
        if pre is not None:
            signals.append(pre)
        post = _setup_post_fomc_fade(bars, sd, params)
        if post is not None:
            signals.append(post)

    # Cap per-day signals (deterministic order = chronological by ts).
    signals.sort(key=lambda s: s.timestamp)
    return signals[: params.max_signals_per_day]


__all__ = [
    "EVENTS",
    "EVENT_LOCKOUT_DATES",
    "SPECIALIST_ID",
    "Event",
    "MacroEventParams",
    "event_distance",
    "generate_signals",
    "is_event_day",
    "is_session_blocked",
]
