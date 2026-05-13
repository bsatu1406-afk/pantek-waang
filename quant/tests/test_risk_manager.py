"""Unit tests for Agent #10 — Risk Manager.

These tests pin every hard rule from ``briefings/agent_10_risk_manager.md``
and exercise the multi-account allocator + the Monte Carlo breach
estimator. They use only synthetic ``Account`` / ``Signal`` instances and
do NOT touch the engine's internals — the engine state is read-only here.
"""
from __future__ import annotations

from datetime import UTC, date, datetime

from src.common.types import Side, Signal
from src.simulation.engine import (
    BUCKET_START,
    DAILY_LOCK_DELTA,
    DAILY_PROFIT_CAP_FUNDED,
    STARTING_EQUITY,
    Account,
    AccountPhase,
    Position,
)
from src.strategy.specialists.risk_manager import (
    RiskParams,
    allocate_signal,
    calculate_size,
    monte_carlo_breach_prob,
    should_take_signal,
)

UTC = UTC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _funded_acct(
    aid: str = "A001",
    equity: float = STARTING_EQUITY,
    bucket: float = BUCKET_START,
    cycles: int = 0,
    eod_high_prev: float | None = None,
    locked: bool = False,
    capped: bool = False,
) -> Account:
    return Account(
        id=aid,
        buy_date=date(2024, 1, 1),
        phase=AccountPhase.ACTIVE_FUNDED,
        equity=equity,
        total_bucket=bucket,
        cycles_done=cycles,
        locked_today=locked,
        profit_capped_today=capped,
        eod_high_prev_day=eod_high_prev if eod_high_prev is not None else equity,
    )


def _eval_acct(aid: str = "A001", **kw) -> Account:
    a = _funded_acct(aid=aid, **kw)
    a.phase = AccountPhase.ACTIVE_EVAL
    return a


def _sig(
    et_hour: int = 10,
    et_minute: int = 0,
    side: Side = Side.LONG,
    confidence: float = 0.7,
    entry: float = 5000.0,
) -> Signal:
    # 10:00 ET in winter ≈ 15:00 UTC. We construct from UTC = ET + 5h
    # (results don't depend on exact DST; the cutoff test uses ET-aware time).
    ts = datetime(2024, 1, 15, et_hour + 5, et_minute, tzinfo=UTC)
    stop = entry - 5.0 if side == Side.LONG else entry + 5.0
    return Signal(
        timestamp=ts,
        side=side,
        confidence=confidence,
        specialist="test",
        setup_id="t",
        entry_price=entry,
        stop_price=stop,
    )


# ---------------------------------------------------------------------------
# Hard-rule vetos
# ---------------------------------------------------------------------------


def test_inactive_phase_veto() -> None:
    a = _funded_acct()
    a.phase = AccountPhase.BREACHED_EVAL
    allow, reason = should_take_signal(a, _sig(), {})
    assert not allow
    assert reason == "phase_inactive"


def test_locked_today_veto() -> None:
    a = _funded_acct(locked=True)
    allow, reason = should_take_signal(a, _sig(), {})
    assert not allow
    assert reason == "locked_today"


def test_profit_capped_today_veto() -> None:
    a = _funded_acct(capped=True)
    allow, reason = should_take_signal(a, _sig(), {})
    assert not allow
    assert reason == "profit_capped_today"


def test_position_open_veto() -> None:
    a = _funded_acct()
    a.position = Position(
        side=Side.LONG, qty=1, entry_price=5000.0,
        open_ts=datetime(2024, 1, 15, 14, 30, tzinfo=UTC),
        specialist="x", stop_price=4995.0,
    )
    allow, reason = should_take_signal(a, _sig(), {})
    assert not allow
    assert reason == "position_open"


def test_low_confidence_veto() -> None:
    # Default min_confidence is 0.45 — anything below should be vetoed.
    a = _funded_acct()
    allow, reason = should_take_signal(a, _sig(confidence=0.35), {})
    assert not allow
    assert reason == "low_confidence"


def test_thin_floor_cushion_veto() -> None:
    # equity is right at the floor: cushion = 0 < min_floor_distance (60)
    a = _funded_acct(equity=STARTING_EQUITY - DAILY_LOCK_DELTA, eod_high_prev=STARTING_EQUITY)
    allow, reason = should_take_signal(a, _sig(), {})
    assert not allow
    assert reason == "thin_floor_cushion"


def test_after_cutoff_veto() -> None:
    # 15:45 ET is within the avoid_close window (default 30 min before 16:00 ET)
    a = _funded_acct()
    sig = _sig(et_hour=15, et_minute=45)
    allow, reason = should_take_signal(a, sig, {})
    assert not allow
    assert reason == "after_cutoff"


def test_after_cutoff_boundary_allowed() -> None:
    # 15:29 ET should still pass — exactly one minute before the 15:30 cutoff
    a = _funded_acct()
    sig = _sig(et_hour=15, et_minute=29)
    allow, _ = should_take_signal(a, sig, {})
    assert allow


def test_near_profit_cap_veto_funded() -> None:
    # equity is within $50 of the +$500 daily cap
    eod_high = STARTING_EQUITY
    near_cap = eod_high + DAILY_PROFIT_CAP_FUNDED - 25.0  # $25 from cap
    a = _funded_acct(equity=near_cap, eod_high_prev=eod_high)
    allow, reason = should_take_signal(a, _sig(), {})
    assert not allow
    assert reason == "near_profit_cap"


def test_eval_account_ignores_profit_cap() -> None:
    # eval accounts have no $500 daily cap — they must NOT be vetoed on that rule
    eod_high = STARTING_EQUITY
    near_cap = eod_high + DAILY_PROFIT_CAP_FUNDED - 25.0
    a = _eval_acct(equity=near_cap, eod_high_prev=eod_high)
    allow, reason = should_take_signal(a, _sig(), {})
    assert allow, f"eval account spuriously vetoed: {reason}"


def test_low_bucket_low_confidence_veto() -> None:
    # bucket at $250 = one decrement from breach; confidence <= 0.8 ⇒ veto
    a = _funded_acct(bucket=250.0)
    allow, reason = should_take_signal(a, _sig(confidence=0.75), {})
    assert not allow
    assert reason == "low_bucket_low_conf"


def test_low_bucket_high_confidence_forces_qty_one() -> None:
    a = _funded_acct(bucket=250.0)
    params = RiskParams(max_qty_per_signal=3, low_bucket_max_qty=1)
    qty = calculate_size(a, _sig(confidence=0.95), {}, params)
    assert qty == 1


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_calculate_size_default_qty() -> None:
    a = _funded_acct()
    params = RiskParams(max_qty_per_signal=2)
    assert calculate_size(a, _sig(confidence=0.7), {}, params) == 2


def test_calculate_size_returns_zero_when_vetoed() -> None:
    a = _funded_acct(locked=True)
    assert calculate_size(a, _sig(), {}) == 0


def test_high_confidence_capped_by_high_conf_max_qty() -> None:
    a = _funded_acct()
    params = RiskParams(max_qty_per_signal=5, high_conf_max_qty=2, high_conf_threshold=0.85)
    qty = calculate_size(a, _sig(confidence=0.9), {}, params)
    assert qty == 2


# ---------------------------------------------------------------------------
# Multi-account allocation
# ---------------------------------------------------------------------------


def test_allocate_prefers_larger_cushion_then_lower_cycles() -> None:
    # a_big: larger floor-cushion (equity well above floor) ⇒ should be first.
    # Use eval accounts so the funded $500 daily cap doesn't interfere with
    # the ordering test (we only want to exercise floor-cushion + cycles).
    a_big = _eval_acct(aid="BIG", equity=25_400.0, eod_high_prev=25_000.0, cycles=0)
    a_mid = _eval_acct(aid="MID", equity=25_200.0, eod_high_prev=25_000.0, cycles=2)
    a_small = _eval_acct(aid="SML", equity=25_100.0, eod_high_prev=25_000.0, cycles=4)
    allocations = allocate_signal([a_small, a_mid, a_big], _sig(), {})
    assert [acct.id for acct, _ in allocations] == ["BIG", "MID", "SML"]
    assert all(qty == 1 for _, qty in allocations)


def test_allocate_excludes_vetoed_accounts() -> None:
    a_live = _funded_acct("LIVE")
    a_locked = _funded_acct("LOCK", locked=True)
    a_breached = _funded_acct("BRC")
    a_breached.phase = AccountPhase.PASSED_EVAL_BREACHED_FUNDED
    allocations = allocate_signal([a_live, a_locked, a_breached], _sig(), {})
    assert [acct.id for acct, _ in allocations] == ["LIVE"]


def test_allocate_empty_when_all_vetoed() -> None:
    a = _funded_acct(locked=True)
    assert allocate_signal([a], _sig()) == []


def test_allocate_respects_top_n_limit() -> None:
    accts = [
        _eval_acct(aid=f"A{i:02d}", equity=25_400.0 - i * 10, eod_high_prev=25_000.0)
        for i in range(5)
    ]
    params = RiskParams(allocate_top_n=2)
    allocations = allocate_signal(accts, _sig(), {}, params)
    assert len(allocations) == 2


# ---------------------------------------------------------------------------
# Monte Carlo breach probability
# ---------------------------------------------------------------------------


def test_mc_zero_when_no_samples() -> None:
    a = _funded_acct()
    assert monte_carlo_breach_prob(a, [], n_iters=100) == 0.0


def test_mc_breach_prob_high_for_negative_only_pnl() -> None:
    """Pinning: an account fed only catastrophic losses must breach almost certainly."""
    a = _funded_acct()
    # Every draw is a -$260 day → triggers daily lock + bucket decrement.
    samples = [-260.0] * 10
    p = monte_carlo_breach_prob(a, samples, n_iters=200, n_days=60, seed=7)
    assert p > 0.95


def test_mc_breach_prob_low_for_profitable_pnl() -> None:
    a = _funded_acct()
    # All positive PnL — no locks → never breach.
    samples = [50.0, 100.0, 75.0]
    p = monte_carlo_breach_prob(a, samples, n_iters=500, n_days=60, seed=11)
    assert p == 0.0


def test_mc_breach_prob_monotone_in_bucket() -> None:
    """Same loss distribution ⇒ breach prob must decrease as bucket grows."""
    samples = [-300.0, 50.0, 50.0]  # ~1/3 of days are locks
    a_low = _funded_acct(bucket=500.0)
    a_high = _funded_acct(bucket=1000.0)
    p_low = monte_carlo_breach_prob(a_low, samples, n_iters=400, n_days=60, seed=21)
    p_high = monte_carlo_breach_prob(a_high, samples, n_iters=400, n_days=60, seed=21)
    assert p_high <= p_low


def test_mc_inactive_account_returns_terminal_prob() -> None:
    a = _funded_acct()
    a.phase = AccountPhase.PASSED_EVAL_BREACHED_FUNDED
    # Already breached — return 1.0 (terminal); samples ignored.
    p = monte_carlo_breach_prob(a, [10.0, -10.0], n_iters=100)
    assert p == 1.0


# ---------------------------------------------------------------------------
# generate_signals (no-op)
# ---------------------------------------------------------------------------


def test_generate_signals_returns_empty() -> None:
    import polars as pl

    from src.strategy.specialists.risk_manager import generate_signals

    df = pl.DataFrame({"ts": [], "open": [], "high": [], "low": [], "close": [], "volume": []})
    sigs = generate_signals(df, RiskParams())
    assert sigs == []
