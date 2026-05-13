"""Agent #10 — Risk Manager & Multi-Account Optimizer.

The Risk Manager is **not** a signal-generating specialist. It sits on top of
the prop-firm engine + the per-account state and acts as a veto / sizing /
allocation layer for signals produced by the other 9 specialists.

Public API (see ``briefings/agent_10_risk_manager.md``):

* ``calculate_size(account, signal, ensemble_state) -> int``
* ``should_take_signal(account, signal, ensemble_state) -> tuple[bool, str]``
* ``allocate_signal(accounts, signal) -> list[tuple[Account, int]]``
* ``monte_carlo_breach_prob(account, daily_pnl_samples, n_iters) -> float``

A no-op ``generate_signals`` is exposed so the standalone audit harness
``src.backtest.standalone`` can import the module without failing — the
harness will simply observe ``trades=0`` and mark the verdict ``MARGINAL``.
The real verdict for this specialist is **breach reduction**, measured by the
A/B comparative simulation in :func:`run_ab_simulation`.

PINNED: this module never mutates :class:`Account` state nor the engine's
constants. It only reads.
"""
from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import polars as pl

from src.common.types import Signal
from src.simulation.engine import (
    BUCKET_DECREMENT,
    DAILY_LOCK_DELTA,
    DAILY_PROFIT_CAP_FUNDED,
    Account,
    AccountPhase,
)
from src.strategy.specialists._interface import SpecialistParams

_ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


@dataclass
class RiskParams(SpecialistParams):
    """Risk Manager configuration.

    All thresholds are conservative defaults sourced from the v4 briefing.
    """

    specialist_id: str = "risk_manager"

    # sizing
    max_qty_per_signal: int = 1                # MES contracts per signal per account
    high_conf_max_qty: int = 1                 # max qty when confidence > high_conf_threshold
    high_conf_threshold: float = 0.85          # confidence above which high_conf_max_qty applies

    # cushion-from-floor veto
    min_floor_distance: float = 60.0           # $ from daily-lock floor to allow new entries

    # bucket-cushion veto (one decrement away from breach)
    low_bucket_threshold: float = 250.0        # bucket <= this is treated as one-lock-away
    low_bucket_min_confidence: float = 0.8     # below this confidence, veto entirely when low bucket
    low_bucket_max_qty: int = 1                # forced qty when low bucket but high confidence

    # intraday cutoff (US/Eastern)
    avoid_close_minutes: int = 30              # do not enter signals within this many minutes of 16:00 ET
    funded_profit_lock_cooldown_minutes: int = 0  # reserved for future use

    # funded daily profit-cap cushion: don't enter if intraday PnL is already
    # ``profit_cap_buffer`` away from triggering the +$500 positive lock.
    profit_cap_buffer: float = 50.0

    # confidence floor — signals below this are always vetoed. Defaults
    # to 0.45 based on the A/B comparative simulation in
    # ``scripts/ab_risk_manager.py``; lower values let too many catastrophic
    # tail trades through.
    min_confidence: float = 0.45

    # multi-account allocation
    allocate_top_n: int = 8                    # how many accounts to allocate one signal to


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_active(account: Account) -> bool:
    return account.phase in (AccountPhase.ACTIVE_EVAL, AccountPhase.ACTIVE_FUNDED)


def _floor_cushion(account: Account) -> float:
    """Dollar distance between current equity and the daily-lock floor.

    Positive = cushion, negative = already past the floor (engine would lock
    on the next mark-to-market).
    """
    floor = account.eod_high_prev_day - DAILY_LOCK_DELTA
    return account.equity - floor


def _profit_cap_distance(account: Account) -> float:
    """For funded accounts, dollars remaining until the +$500 daily cap.

    For eval accounts (no cap), returns ``float('inf')``.
    """
    if account.phase != AccountPhase.ACTIVE_FUNDED:
        return float("inf")
    cap_level = account.eod_high_prev_day + DAILY_PROFIT_CAP_FUNDED
    return cap_level - account.equity


def _is_after_cutoff(signal_ts, avoid_close_minutes: int) -> bool:
    """True if signal fires within ``avoid_close_minutes`` of the 16:00 ET close.

    Signal timestamps are stored as tz-aware UTC datetimes; we convert to
    US/Eastern to compare against the RTH close.
    """
    if signal_ts.tzinfo is None:
        return False
    et = signal_ts.astimezone(_ET)
    cutoff_minutes_from_open = 6 * 60 + 30 - avoid_close_minutes  # 16:00 ET - cutoff
    minutes_since_open = (et.hour * 60 + et.minute) - (9 * 60 + 30)
    return minutes_since_open >= cutoff_minutes_from_open


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def should_take_signal(
    account: Account,
    signal: Signal,
    ensemble_state: dict | None = None,
    params: RiskParams | None = None,
) -> tuple[bool, str]:
    """Hard-rule veto layer.

    Returns ``(allow, reason)``. ``reason`` is a short verb-phrase suitable
    for logging / event tracing.
    """
    params = params or RiskParams()

    # 1) Phase must be active
    if not _is_active(account):
        return False, "phase_inactive"

    # 2) Daily flags
    if account.locked_today:
        return False, "locked_today"
    if account.profit_capped_today:
        return False, "profit_capped_today"

    # 3) Position already open — engine enforces ≤1 anyway, but pre-veto
    if account.position is not None:
        return False, "position_open"

    # 4) Confidence floor
    if signal.confidence < params.min_confidence:
        return False, "low_confidence"

    # 5) Intraday cutoff (15:30 ET by default)
    if _is_after_cutoff(signal.timestamp, params.avoid_close_minutes):
        return False, "after_cutoff"

    # 6) Floor cushion — keep ``min_floor_distance`` between equity and lock floor
    cushion = _floor_cushion(account)
    if cushion < params.min_floor_distance:
        return False, "thin_floor_cushion"

    # 7) Profit-cap proximity (funded only): don't open if already close to cap
    if account.phase == AccountPhase.ACTIVE_FUNDED:
        pc_dist = _profit_cap_distance(account)
        if pc_dist < params.profit_cap_buffer:
            return False, "near_profit_cap"

    # 8) Bucket cushion: one decrement from breach
    if account.bucket_remaining <= params.low_bucket_threshold:
        if signal.confidence <= params.low_bucket_min_confidence:
            return False, "low_bucket_low_conf"
        # high-confidence escape: allow but force qty=1 (handled in calculate_size)

    return True, "ok"


def calculate_size(
    account: Account,
    signal: Signal,
    ensemble_state: dict | None = None,
    params: RiskParams | None = None,
) -> int:
    """Return the integer MES quantity to trade for this account+signal.

    Returns 0 if the signal should be vetoed.
    """
    params = params or RiskParams()
    allow, _reason = should_take_signal(account, signal, ensemble_state, params)
    if not allow:
        return 0

    # Low-bucket high-confidence override: force qty=1
    if account.bucket_remaining <= params.low_bucket_threshold:
        return min(params.low_bucket_max_qty, params.max_qty_per_signal)

    # High-confidence sizing (cap by both knobs to be safe)
    if signal.confidence >= params.high_conf_threshold:
        return min(params.high_conf_max_qty, params.max_qty_per_signal)

    return params.max_qty_per_signal


def allocate_signal(
    accounts: Iterable[Account],
    signal: Signal,
    ensemble_state: dict | None = None,
    params: RiskParams | None = None,
) -> list[tuple[Account, int]]:
    """Decide which accounts take this signal and at what qty.

    Preference order (descending):
      1. Larger floor-cushion (more headroom before the daily lock).
      2. Lower ``cycles_done`` (so we build endapan before unlocking 80% mode).
      3. Higher ``bucket_remaining`` (further from breach).

    Only the top ``params.allocate_top_n`` accounts (after vetoing) are
    allocated. Returns an empty list if no account passes the veto.
    """
    params = params or RiskParams()
    candidates: list[tuple[Account, int]] = []
    for acct in accounts:
        qty = calculate_size(acct, signal, ensemble_state, params)
        if qty > 0:
            candidates.append((acct, qty))

    candidates.sort(
        key=lambda pair: (
            -_floor_cushion(pair[0]),
            pair[0].cycles_done,
            -pair[0].bucket_remaining,
        )
    )
    return candidates[: params.allocate_top_n]


# ---------------------------------------------------------------------------
# Monte Carlo breach probability
# ---------------------------------------------------------------------------


def monte_carlo_breach_prob(
    account: Account,
    daily_pnl_samples: list[float],
    n_iters: int = 5000,
    n_days: int = 60,
    seed: int | None = None,
) -> float:
    """Estimate P(bucket → 0) over the next ``n_days`` trading days.

    Bootstrap from the supplied empirical ``daily_pnl_samples`` (per-account,
    per-day net PnL). Each draw is treated as one trading day. The engine's
    lock rule is approximated as follows:

      * On day ``i``, compute ``floor = eod_high_prev_day - DAILY_LOCK_DELTA``.
      * If a draw ``pnl_i`` would put equity at or below the floor (i.e.
        ``equity + pnl_i <= floor``), it counts as a lock day:
            - ``equity := floor``
            - ``bucket -= 250``
      * Otherwise: ``equity += pnl_i``.
      * If ``bucket <= 0`` at any point ⇒ breach.

    ``eod_high_prev_day`` is updated to ``max(prev, day_end_equity)``.

    Returns a probability in ``[0, 1]``. Returns 0.0 if there are no samples
    (caller can't infer risk without data).
    """
    if not daily_pnl_samples:
        return 0.0
    if not _is_active(account):
        # Inactive accounts cannot breach further — treat as terminal.
        return 1.0 if account.phase == AccountPhase.PASSED_EVAL_BREACHED_FUNDED else 0.0

    rng = random.Random(seed)
    samples = list(daily_pnl_samples)
    breaches = 0

    for _ in range(n_iters):
        equity = account.equity
        bucket = account.total_bucket
        eod_high = account.eod_high_prev_day

        breached = False
        for _day in range(n_days):
            if bucket <= 0:
                breached = True
                break
            pnl = rng.choice(samples)
            floor = eod_high - DAILY_LOCK_DELTA
            if equity + pnl <= floor:
                equity = floor
                bucket -= BUCKET_DECREMENT
                if bucket <= 0:
                    breached = True
                    break
            else:
                equity += pnl
            eod_high = max(eod_high, equity)

        if breached:
            breaches += 1

    return breaches / n_iters


# ---------------------------------------------------------------------------
# Specialist protocol no-op
# ---------------------------------------------------------------------------


def generate_signals(bars: pl.DataFrame, params: RiskParams | None = None) -> list[Signal]:
    """Risk Manager is not signal-generating; returns no signals.

    The standalone audit harness imports this for protocol uniformity. The
    real verdict for this specialist is breach-reduction (see
    :func:`run_ab_simulation`).
    """
    _ = (bars, params)  # silence unused-arg lint
    return []


__all__ = [
    "RiskParams",
    "allocate_signal",
    "calculate_size",
    "generate_signals",
    "monte_carlo_breach_prob",
    "should_take_signal",
]
