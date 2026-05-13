"""Pin the v4 prop firm rules with engine unit tests.

These tests intentionally drive the engine through synthetic equity paths
without using any real market data; they validate that:

* Daily lock fires exactly at -$250 from prev-day EOD high.
* Bucket decrements by $250 per lock day; 4 locks => breach.
* Eval pass resets equity to $25k and bucket to $1k.
* Withdrawal cycle switches from 50% to 80% AFTER cycle 3.
* Withdrawal compound endapan example from briefing Section 2 reproduces
  exactly the numbers in the briefing table.
* Funded daily $500 profit cap locks the day positively WITHOUT consuming
  the bucket.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from src.common.types import Side
from src.simulation.engine import (
    ACCOUNT_COST,
    AccountPhase,
    BUCKET_DECREMENT,
    BUCKET_START,
    DAILY_LOCK_DELTA,
    DAILY_PROFIT_CAP_FUNDED,
    EVAL_PASS_EQUITY,
    STARTING_EQUITY,
    PropFirmEngine,
    WD_CYCLE_THRESHOLD,
    WD_PCT_EARLY,
    WD_PCT_LATE,
    WD_PCT_SWITCH_AFTER,
)

UTC = timezone.utc


def _dt(d: date, h: int = 9, m: int = 30) -> datetime:
    return datetime(d.year, d.month, d.day, h, m, tzinfo=UTC)


def _force_funded(eng: PropFirmEngine):
    """Convenience: open one account and push it straight into funded by
    bumping equity above the eval target, then closing the session."""
    a = eng.open_account(date(2024, 5, 1))
    d = date(2024, 5, 1)
    eng.on_session_open(d, _dt(d))
    a.equity = EVAL_PASS_EQUITY + 0.01
    eng.on_session_close(d, _dt(d, 16, 0))
    assert a.phase == AccountPhase.ACTIVE_FUNDED
    assert a.equity == STARTING_EQUITY  # reset
    assert a.total_bucket == BUCKET_START  # reset
    return a


# ---------------------------------------------------------------------------
# Daily lock + bucket
# ---------------------------------------------------------------------------


def test_daily_lock_fires_at_exact_minus_250() -> None:
    eng = PropFirmEngine()
    a = _force_funded(eng)
    # Day 2: ref = day 1's EOD high (== 25000). Drift equity down to 24750
    # via mark_to_market; it should fire daily lock at exactly $250 drop.
    d2 = date(2024, 5, 2)
    eng.on_session_open(d2, _dt(d2))
    assert a.eod_high_prev_day == STARTING_EQUITY
    # No position; equity is constant at 25000 — set day_open and walk to -250.
    a.equity = STARTING_EQUITY - 250.0
    eng.mark_to_market(_dt(d2, 10, 0), price=5000.0)
    assert a.locked_today is True
    assert a.daily_locks == 1
    assert a.total_bucket == BUCKET_START - BUCKET_DECREMENT


def test_lock_does_not_fire_at_minus_249_99() -> None:
    eng = PropFirmEngine()
    a = _force_funded(eng)
    d2 = date(2024, 5, 2)
    eng.on_session_open(d2, _dt(d2))
    a.equity = STARTING_EQUITY - 249.99
    eng.mark_to_market(_dt(d2, 10, 0), price=5000.0)
    assert a.locked_today is False
    assert a.total_bucket == BUCKET_START


def test_bucket_decrements_to_zero_in_four_locks_then_breach() -> None:
    eng = PropFirmEngine()
    a = _force_funded(eng)
    for i in range(4):
        d = date(2024, 5, 2 + i)
        eng.on_session_open(d, _dt(d))
        # synthesise -$250 drop
        a.equity = a.eod_high_prev_day - 250.0
        eng.mark_to_market(_dt(d, 10, 0), price=5000.0)
        if i < 3:
            assert a.phase == AccountPhase.ACTIVE_FUNDED, f"premature breach at lock {i+1}"
            assert a.total_bucket == BUCKET_START - (i + 1) * BUCKET_DECREMENT
        eng.on_session_close(d, _dt(d, 16, 0))
    assert a.phase == AccountPhase.PASSED_EVAL_BREACHED_FUNDED
    assert a.total_bucket == 0
    assert a.daily_locks == 4


def test_profitable_day_does_not_consume_bucket() -> None:
    eng = PropFirmEngine()
    a = _force_funded(eng)
    d2 = date(2024, 5, 2)
    eng.on_session_open(d2, _dt(d2))
    a.equity = STARTING_EQUITY + 100  # profit day
    eng.mark_to_market(_dt(d2, 10, 0), price=5000.0)
    eng.on_session_close(d2, _dt(d2, 16, 0))
    assert a.total_bucket == BUCKET_START
    assert a.daily_locks == 0
    assert a.winning_days == 1


# ---------------------------------------------------------------------------
# Funded $500 profit cap
# ---------------------------------------------------------------------------


def test_funded_profit_cap_locks_without_bucket_hit() -> None:
    eng = PropFirmEngine()
    a = _force_funded(eng)
    d2 = date(2024, 5, 2)
    eng.on_session_open(d2, _dt(d2))
    a.equity = STARTING_EQUITY + 500  # exactly cap
    eng.mark_to_market(_dt(d2, 10, 0), price=5000.0)
    assert a.profit_capped_today is True
    assert a.locked_today is False
    assert a.total_bucket == BUCKET_START  # NO bucket consumption


# ---------------------------------------------------------------------------
# Eval -> funded transition
# ---------------------------------------------------------------------------


def test_pass_eval_resets_equity_and_bucket() -> None:
    eng = PropFirmEngine()
    a = eng.open_account(date(2024, 5, 1))
    d = date(2024, 5, 1)
    eng.on_session_open(d, _dt(d))
    # Lock once during eval to consume some bucket
    a.equity = STARTING_EQUITY - 250.0
    eng.mark_to_market(_dt(d, 10, 0), price=5000.0)
    assert a.phase == AccountPhase.ACTIVE_EVAL
    assert a.total_bucket == BUCKET_START - BUCKET_DECREMENT
    # End the day still in eval (no pass)
    eng.on_session_close(d, _dt(d, 16, 0))
    # New day, push to eval target
    d2 = date(2024, 5, 2)
    eng.on_session_open(d2, _dt(d2))
    a.equity = EVAL_PASS_EQUITY + 0.5
    eng.on_session_close(d2, _dt(d2, 16, 0))
    assert a.phase == AccountPhase.ACTIVE_FUNDED
    assert a.equity == STARTING_EQUITY
    assert a.total_bucket == BUCKET_START
    assert a.cycles_done == 0


# ---------------------------------------------------------------------------
# Withdrawal cycle: 50% for cycles 1..3, 80% for cycles 4+
# ---------------------------------------------------------------------------


def _winning_day(eng: PropFirmEngine, a, d: date, profit: float) -> None:
    eng.on_session_open(d, _dt(d))
    a.equity += profit
    eng.on_session_close(d, _dt(d, 16, 0))


def test_withdrawal_pct_switches_to_80_after_cycle_3() -> None:
    eng = PropFirmEngine()
    a = _force_funded(eng)
    # 4 cycles, each preceded by exactly 5 winning days @ $100
    day = date(2024, 5, 2)
    for cycle in range(4):
        for _ in range(WD_CYCLE_THRESHOLD):
            _winning_day(eng, a, day, 100.0)
            day += timedelta(days=1)
        # After the 5th winning day in the cycle, the engine should have just
        # executed a withdrawal at session close.
        assert a.cycles_done == cycle + 1, f"cycle count off after cycle {cycle+1}"
        if cycle < WD_PCT_SWITCH_AFTER:
            assert a.cycles_50pct == cycle + 1
            assert a.cycles_80pct == 0
            assert a.withdrawals[-1].pct == WD_PCT_EARLY
        else:
            assert a.cycles_50pct == WD_PCT_SWITCH_AFTER
            assert a.cycles_80pct == cycle + 1 - WD_PCT_SWITCH_AFTER
            assert a.withdrawals[-1].pct == WD_PCT_LATE


def test_winning_day_counter_resets_to_zero_after_withdrawal() -> None:
    eng = PropFirmEngine()
    a = _force_funded(eng)
    day = date(2024, 5, 2)
    for _ in range(WD_CYCLE_THRESHOLD):
        _winning_day(eng, a, day, 100.0)
        day += timedelta(days=1)
    assert a.cycles_done == 1
    assert a.winning_days == 0


def test_losing_day_does_not_increment_winning_counter() -> None:
    eng = PropFirmEngine()
    a = _force_funded(eng)
    day = date(2024, 5, 2)
    _winning_day(eng, a, day, 100.0)
    day += timedelta(days=1)
    _winning_day(eng, a, day, -50.0)
    assert a.winning_days == 1
    assert a.n_losing_days_total == 1


def test_no_withdrawal_when_no_profit_vs_baseline() -> None:
    eng = PropFirmEngine()
    a = _force_funded(eng)
    day = date(2024, 5, 2)
    # Five winning days that net to zero vs baseline (oscillate)
    for delta in [+50, -50, +50, -50, +50]:
        _winning_day(eng, a, day, delta)
        day += timedelta(days=1)
    # winning_days only ticks on positive-EOD days, so we need a cleaner setup:
    # five small positive days, then push equity back to baseline before close.
    # Here equity > 25k after net +50, so a WD should happen but tiny.
    # Just assert: when we engineer profit_base <= 0, no WD.
    eng2 = PropFirmEngine()
    a2 = _force_funded(eng2)
    day = date(2024, 5, 2)
    for _ in range(WD_CYCLE_THRESHOLD):
        eng2.on_session_open(day, _dt(day))
        a2.equity = STARTING_EQUITY + 0.01  # winning day, but trivial
        eng2.on_session_close(day, _dt(day, 16, 0))
        day += timedelta(days=1)
    # Now force equity to be == baseline at WD attempt:
    # We can't easily do that without rewinding; just verify the rule directly:
    a2.equity = STARTING_EQUITY  # no profit
    a2.winning_days = WD_CYCLE_THRESHOLD
    wd = eng2._maybe_withdraw(a2, _dt(day, 16, 0))
    assert wd is None


# ---------------------------------------------------------------------------
# Briefing Section 2 worked example: compound endapan profit
# ---------------------------------------------------------------------------


def test_briefing_section2_table_reproduction() -> None:
    """Reproduce the exact table in the briefing Section 2.

    | Cycle | new profit since last WD | profit base | %   | $ wd     | equity after | residual |
    |   1   |  1,500                   |  1,500      | 50  |    750   | 25,750       |  750     |
    |   2   |  2,000                   |  2,750      | 50  |  1,375   | 26,375       |  1,375   |
    |   3   |  1,500                   |  2,875      | 50  |  1,437.5 | 26,437.50    |  1,437.5 |
    |   4   |  2,000                   |  3,437.5    | 80  |  2,750   | 25,687.50    |  687.5   |
    |   5   |  1,800                   |  2,487.5    | 80  |  1,990   | 25,497.50    |  497.5   |

    NOTE: The briefing's printed table lists cycle 5 equity_after as
    ``$25,697.50`` but the arithmetic is wrong:
        equity_before_wd = 25,687.50 + 1,800 = 27,487.50
        wd = 2,487.50 * 0.80 = 1,990
        equity_after = 27,487.50 - 1,990 = 25,497.50
    The residual ``$497.50`` printed in the briefing only matches
    ``$25,497.50`` so we treat that as the source of truth.
    """
    eng = PropFirmEngine()
    a = _force_funded(eng)
    day = date(2024, 5, 2)

    def _wd_cycle(new_profit: float, expected_pct: float, expected_amount: float, expected_equity_after: float) -> None:
        nonlocal day
        # Engineer 5 winning days totaling new_profit.
        for i in range(WD_CYCLE_THRESHOLD):
            eng.on_session_open(day, _dt(day))
            delta = new_profit / WD_CYCLE_THRESHOLD
            a.equity += delta
            eng.on_session_close(day, _dt(day, 16, 0))
            day += timedelta(days=1)
        wd = a.withdrawals[-1]
        assert wd.pct == expected_pct
        assert abs(wd.amount - expected_amount) < 0.01, (wd.amount, expected_amount)
        assert abs(a.equity - expected_equity_after) < 0.01, (a.equity, expected_equity_after)

    _wd_cycle(1500.0, 0.50,   750.0, 25_750.0)
    _wd_cycle(2000.0, 0.50, 1_375.0, 26_375.0)
    _wd_cycle(1500.0, 0.50, 1_437.5, 26_437.5)
    _wd_cycle(2000.0, 0.80, 2_750.0, 25_687.5)
    _wd_cycle(1800.0, 0.80, 1_990.0, 25_497.5)  # see docstring re: briefing typo


# ---------------------------------------------------------------------------
# Position lifecycle smoke test
# ---------------------------------------------------------------------------


def test_close_position_realizes_pnl() -> None:
    eng = PropFirmEngine()
    a = _force_funded(eng)
    d = date(2024, 5, 2)
    eng.on_session_open(d, _dt(d))
    assert eng.open_position(a, _dt(d, 10), Side.LONG, qty=1, entry_price=5000.0,
                              stop_price=4990.0, specialist="test")
    eng.close_position(a, _dt(d, 10, 30), price=5002.0, reason="tp")
    assert a.position is None
    # 1 MES * 2 pts * $5/pt = $10 pnl
    assert abs(a.trades[-1].pnl - 10.0) < 1e-6


def test_eod_flat_forces_close() -> None:
    eng = PropFirmEngine()
    a = _force_funded(eng)
    d = date(2024, 5, 2)
    eng.on_session_open(d, _dt(d))
    eng.open_position(a, _dt(d, 15), Side.LONG, qty=1, entry_price=5000.0,
                      stop_price=4990.0, specialist="test")
    eng.on_session_close(d, _dt(d, 16, 0))
    assert a.position is None
    assert any(t.reason == "eod_flat" for t in a.trades)


def test_reinvest_buys_new_accounts() -> None:
    eng = PropFirmEngine(max_active=3)
    eng.open_account(date(2024, 5, 1))
    eng.external_cash = 250.0
    bought = eng.reinvest_into_new_accounts(date(2024, 5, 2))
    assert bought == 2  # $250 / $80 = 3, but capped at 2 (max_active=3 - already 1)
    assert eng.n_active() == 3
