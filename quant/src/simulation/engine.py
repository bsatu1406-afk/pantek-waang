"""Prop firm simulation engine for the v4 briefing.

PINNED RULES (do not modify without a briefing update):

* Account purchase cost: $80
* Starting equity: $25,000
* Eval pass target: equity >= $26,500  (+$1,500)
* RTH only: 09:30 - 16:00 ET, no overnight hold

* DAILY LOCK ($250):
    - Reference: EOD high of the previous trading day (high-water dynamic)
    - If intraday equity <= ref - 250 at any time during RTH:
        * Force-flat all positions @ current bid/ask
        * acct.locked_today = True (no more entries that day)
        * acct.daily_locks += 1
        * acct.total_bucket -= 250 (decremental, NEVER replenishes)
    - If acct.total_bucket <= 0 after the decrement: BREACH (account halts)

* DAILY $500 PROFIT CAP (funded only): when intraday equity >= ref + 500,
    lock the day POSITIVELY (no more entries; bucket NOT decremented).

* WITHDRAWAL CYCLE (modified):
    - Trigger: 5 winning days (EOD equity > opening equity), non-consecutive
    - Cycle 1-3 -> withdraw 50% of (equity - 25000)
    - Cycle 4+  -> withdraw 80% of (equity - 25000)
    - profit base is *vs the funded baseline $25,000* so endapan profit
      compounds naturally between cycles.

* EVAL -> FUNDED transition: equity resets to $25,000 and bucket to $1,000.

* Max active accounts: 8.

Engine is intentionally bar-driven, not tick-driven: the caller hands the
engine an ordered iterable of `BarUpdate`s with intraday extremes so the
engine can detect daily lock / cap exactly. Trades are matched at mid+slippage
in this MVP — specialists may pass `entry_price/exit_price` to override.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time
from enum import Enum
from typing import Iterable, Optional

from src.common.types import Side, Trade

log = logging.getLogger(__name__)


# -- Pinned constants (sourced from briefing v4) -----------------------------

ACCOUNT_COST = 80.0
STARTING_EQUITY = 25_000.0
EVAL_PASS_EQUITY = 26_500.0           # +$1,500
DAILY_LOCK_DELTA = 250.0              # equity must drop this much below ref
DAILY_PROFIT_CAP_FUNDED = 500.0
BUCKET_START = 1_000.0
BUCKET_DECREMENT = 250.0              # per daily-lock day
WD_CYCLE_THRESHOLD = 5                # winning days needed
WD_PCT_EARLY = 0.50                   # cycles 1..3
WD_PCT_LATE = 0.80                    # cycles 4+
WD_PCT_SWITCH_AFTER = 3               # after this many cycles, switch to LATE
MAX_ACTIVE_ACCOUNTS = 8

# MES tick economics (1 contract = $5/point, 0.25 tick = $1.25)
MES_TICK_SIZE = 0.25
MES_TICK_VALUE = 1.25
MES_PER_POINT = 5.0


class AccountPhase(str, Enum):
    """Lifecycle phase. Externally we map to the briefing's status labels."""

    ACTIVE_EVAL = "active_eval"
    ACTIVE_FUNDED = "active_funded"
    PASSED_EVAL_BREACHED_FUNDED = "passed_eval_breached_funded"
    BREACHED_EVAL = "breached_eval"
    RETIRED_PROFITABLE = "retired_profitable"


# -- Dataclasses -------------------------------------------------------------


@dataclass
class Position:
    side: Side
    qty: int
    entry_price: float
    open_ts: datetime
    specialist: str
    stop_price: float
    target_price: Optional[float] = None
    signal_id: Optional[str] = None


@dataclass
class Withdrawal:
    account_id: str
    ts: datetime
    amount: float
    pct: float
    profit_base: float                # equity - 25000 right before WD


@dataclass
class Account:
    """One simulated prop firm account."""

    id: str
    buy_date: date
    phase: AccountPhase = AccountPhase.ACTIVE_EVAL
    equity: float = STARTING_EQUITY
    total_bucket: float = BUCKET_START
    daily_locks: int = 0
    cycles_done: int = 0
    cycles_50pct: int = 0
    cycles_80pct: int = 0
    winning_days: int = 0
    locked_today: bool = False
    profit_capped_today: bool = False

    # daily bookkeeping
    eod_high_prev_day: float = STARTING_EQUITY  # high-water-mark of prev day's EOD
    day_open_equity: float = STARTING_EQUITY    # equity at this RTH session open
    intraday_high: float = STARTING_EQUITY      # running max equity intraday
    intraday_low: float = STARTING_EQUITY       # running min equity intraday

    # event traces
    pass_eval_date: Optional[date] = None
    breach_date: Optional[date] = None
    last_session_date: Optional[date] = None

    # aggregates for report
    total_withdrawn: float = 0.0
    withdrawals: list[Withdrawal] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    best_day_pnl: float = 0.0
    worst_day_pnl: float = 0.0
    n_winning_days_total: int = 0
    n_losing_days_total: int = 0

    # open positions (single-position model in this MVP; engine enforces ≤1)
    position: Optional[Position] = None

    @property
    def alive(self) -> bool:
        return self.phase in (AccountPhase.ACTIVE_EVAL, AccountPhase.ACTIVE_FUNDED)

    @property
    def in_funded(self) -> bool:
        return self.phase == AccountPhase.ACTIVE_FUNDED

    @property
    def bucket_remaining(self) -> float:
        return max(self.total_bucket, 0.0)


# -- Engine ------------------------------------------------------------------


class PropFirmEngine:
    """Drives a fleet of accounts through trading bars.

    The engine itself does not generate signals — it consumes them and applies
    the prop-firm rules. Specialists pass signals to `submit_signal`; the engine
    decides whether to take (size, veto by daily lock, etc.) and tracks PnL.

    Caller is expected to drive the engine bar-by-bar in chronological order:

        eng = PropFirmEngine()
        eng.open_account(buy_date=date(2024,5,1))
        for bar in bars:
            eng.on_session_open(bar)                # at 09:30 ET
            for s in signals_in_this_bar(bar):
                eng.submit_signal(s, bar)
            eng.mark_to_market(bar)
            ... at 16:00 ET:
            eng.on_session_close(bar)
    """

    def __init__(self, max_active: int = MAX_ACTIVE_ACCOUNTS) -> None:
        self.max_active = max_active
        self.accounts: list[Account] = []
        self.external_cash: float = 0.0       # cumulative withdrawn cash
        self.purchase_cost: float = 0.0
        self.all_withdrawals: list[Withdrawal] = []
        # diagnostics
        self.events: list[dict] = []

    # ---- account lifecycle -------------------------------------------------

    def n_active(self) -> int:
        return sum(1 for a in self.accounts if a.alive)

    def can_buy_account(self) -> bool:
        return self.n_active() < self.max_active

    def open_account(self, buy_date: date, account_id: Optional[str] = None) -> Account:
        if not self.can_buy_account():
            raise RuntimeError("max active accounts reached")
        aid = account_id or f"A{len(self.accounts)+1:03d}"
        acct = Account(id=aid, buy_date=buy_date)
        self.accounts.append(acct)
        self.purchase_cost += ACCOUNT_COST
        self._event(buy_date, aid, "OPEN_ACCOUNT", {"phase": acct.phase.value})
        return acct

    # ---- session boundaries ------------------------------------------------

    def on_session_open(self, session_date: date, dt: datetime) -> None:
        """Call at 09:30 ET each RTH day. Resets daily flags + reference."""
        for a in self.accounts:
            if not a.alive:
                continue
            a.locked_today = False
            a.profit_capped_today = False
            a.day_open_equity = a.equity
            a.intraday_high = a.equity
            a.intraday_low = a.equity
            a.last_session_date = session_date

    def on_session_close(self, session_date: date, dt: datetime) -> None:
        """Call at 16:00 ET each RTH day. Forces flat, ticks winning_days, attempts WD."""
        for a in self.accounts:
            if not a.alive:
                continue
            if a.position is not None:
                self._close_position(a, dt, price=a.position.entry_price, reason="eod_flat")
            # winning-day check (EOD vs opening)
            day_pnl = a.equity - a.day_open_equity
            if day_pnl > 0:
                a.winning_days += 1
                a.n_winning_days_total += 1
            elif day_pnl < 0:
                a.n_losing_days_total += 1
            a.best_day_pnl = max(a.best_day_pnl, day_pnl)
            a.worst_day_pnl = min(a.worst_day_pnl, day_pnl)
            # update reference for next day's daily lock
            a.eod_high_prev_day = max(a.day_open_equity, a.intraday_high, a.equity)
            # try eval pass / withdrawal
            self._maybe_pass_eval(a, session_date, dt)
            self._maybe_withdraw(a, dt)

    # ---- intra-bar updates -------------------------------------------------

    def mark_to_market(self, dt: datetime, price: float) -> None:
        """Mark all open positions to current price and apply equity-floor checks.

        `price` is the last trade price; engine assumes 1c MES sizing in this MVP.
        """
        for a in self.accounts:
            if not a.alive:
                continue
            self._mark_account(a, dt, price)

    def _mark_account(self, a: Account, dt: datetime, price: float) -> None:
        # 1) update equity from open position
        unrealized = 0.0
        if a.position is not None:
            sign = 1 if a.position.side == Side.LONG else -1
            unrealized = sign * (price - a.position.entry_price) * a.position.qty * MES_PER_POINT
        cur_equity = a.equity + unrealized
        a.intraday_high = max(a.intraday_high, cur_equity)
        a.intraday_low = min(a.intraday_low, cur_equity)

        # 2) daily lock check (RTH)
        floor = a.eod_high_prev_day - DAILY_LOCK_DELTA
        if (not a.locked_today) and cur_equity <= floor:
            # force flat @ assumed exit price; engine assumption: exit @ price s.t.
            # realized equity == floor (i.e. -$250 from ref).
            if a.position is not None:
                # Compute the exit price that yields realized PnL = floor - a.equity.
                target_pnl = floor - a.equity
                sign = 1 if a.position.side == Side.LONG else -1
                px = a.position.entry_price + target_pnl / (sign * a.position.qty * MES_PER_POINT)
                self._close_position(a, dt, price=px, reason="daily_lock")
            a.equity = floor
            a.intraday_low = min(a.intraday_low, a.equity)
            a.locked_today = True
            a.daily_locks += 1
            a.total_bucket -= BUCKET_DECREMENT
            self._event(dt.date(), a.id, "DAILY_LOCK", {
                "bucket_after": a.total_bucket,
                "equity": a.equity,
                "ref_eod_high": a.eod_high_prev_day,
            })
            if a.total_bucket <= 0:
                self._breach(a, dt, reason="bucket_zero")

        # 3) funded daily profit cap
        if a.in_funded and (not a.profit_capped_today) and cur_equity >= a.eod_high_prev_day + DAILY_PROFIT_CAP_FUNDED:
            # close current position at the cap level
            cap_level = a.eod_high_prev_day + DAILY_PROFIT_CAP_FUNDED
            if a.position is not None:
                target_pnl = cap_level - a.equity
                sign = 1 if a.position.side == Side.LONG else -1
                px = a.position.entry_price + target_pnl / (sign * a.position.qty * MES_PER_POINT)
                self._close_position(a, dt, price=px, reason="profit_cap")
            a.equity = cap_level
            a.profit_capped_today = True
            self._event(dt.date(), a.id, "PROFIT_CAP", {"equity": a.equity})

    # ---- order management --------------------------------------------------

    def open_position(
        self,
        a: Account,
        dt: datetime,
        side: Side,
        qty: int,
        entry_price: float,
        stop_price: float,
        specialist: str,
        target_price: Optional[float] = None,
        signal_id: Optional[str] = None,
    ) -> bool:
        if not a.alive or a.locked_today or a.profit_capped_today:
            return False
        if a.position is not None:
            return False  # one position per account in MVP
        a.position = Position(
            side=side, qty=qty, entry_price=entry_price, open_ts=dt,
            specialist=specialist, stop_price=stop_price, target_price=target_price,
            signal_id=signal_id,
        )
        return True

    def close_position(self, a: Account, dt: datetime, price: float, reason: str) -> Optional[Trade]:
        return self._close_position(a, dt, price, reason)

    def _close_position(self, a: Account, dt: datetime, price: float, reason: str) -> Optional[Trade]:
        if a.position is None:
            return None
        p = a.position
        sign = 1 if p.side == Side.LONG else -1
        pnl = sign * (price - p.entry_price) * p.qty * MES_PER_POINT
        a.equity += pnl
        trade = Trade(
            account_id=a.id, open_ts=p.open_ts, close_ts=dt, side=p.side, qty=p.qty,
            entry_price=p.entry_price, exit_price=price, pnl=pnl, reason=reason,
            specialist=p.specialist, signal_id=p.signal_id,
        )
        a.trades.append(trade)
        a.position = None
        return trade

    # ---- eval / withdrawal -------------------------------------------------

    def _maybe_pass_eval(self, a: Account, session_date: date, dt: datetime) -> None:
        if a.phase != AccountPhase.ACTIVE_EVAL:
            return
        if a.equity >= EVAL_PASS_EQUITY:
            a.phase = AccountPhase.ACTIVE_FUNDED
            a.pass_eval_date = session_date
            # reset equity and bucket per briefing
            a.equity = STARTING_EQUITY
            a.total_bucket = BUCKET_START
            a.eod_high_prev_day = STARTING_EQUITY
            a.winning_days = 0
            a.cycles_done = 0
            a.cycles_50pct = 0
            a.cycles_80pct = 0
            self._event(session_date, a.id, "PASS_EVAL", {"equity_after_reset": a.equity})

    def _maybe_withdraw(self, a: Account, dt: datetime) -> Optional[Withdrawal]:
        if not a.in_funded:
            return None
        if a.winning_days < WD_CYCLE_THRESHOLD:
            return None
        profit_base = a.equity - STARTING_EQUITY
        if profit_base <= 0:
            return None
        pct = WD_PCT_EARLY if a.cycles_done < WD_PCT_SWITCH_AFTER else WD_PCT_LATE
        amount = profit_base * pct
        a.equity -= amount
        a.winning_days = 0
        a.cycles_done += 1
        if pct == WD_PCT_EARLY:
            a.cycles_50pct += 1
        else:
            a.cycles_80pct += 1
        a.total_withdrawn += amount
        wd = Withdrawal(account_id=a.id, ts=dt, amount=amount, pct=pct, profit_base=profit_base)
        a.withdrawals.append(wd)
        self.all_withdrawals.append(wd)
        self.external_cash += amount
        self._event(dt.date(), a.id, "WITHDRAW", {
            "amount": amount, "pct": pct, "profit_base": profit_base,
            "equity_after": a.equity, "cycle": a.cycles_done,
        })
        return wd

    # ---- breach ------------------------------------------------------------

    def _breach(self, a: Account, dt: datetime, reason: str) -> None:
        if a.in_funded:
            a.phase = AccountPhase.PASSED_EVAL_BREACHED_FUNDED
        else:
            a.phase = AccountPhase.BREACHED_EVAL
        a.breach_date = dt.date()
        self._event(dt.date(), a.id, "BREACH", {"reason": reason, "phase": a.phase.value})

    # ---- helpers -----------------------------------------------------------

    def reinvest_into_new_accounts(self, today: date) -> int:
        """Spend external_cash on new $80 accounts up to the active cap."""
        bought = 0
        while self.external_cash >= ACCOUNT_COST and self.can_buy_account():
            self.external_cash -= ACCOUNT_COST
            self.purchase_cost += ACCOUNT_COST
            self.open_account(today)
            bought += 1
        return bought

    def _event(self, d: date, acct_id: str, kind: str, data: dict) -> None:
        self.events.append({"date": d.isoformat(), "account": acct_id, "event": kind, **data})

    # ---- snapshots for reporting ------------------------------------------

    def ecosystem_value(self) -> float:
        equity = sum(a.equity for a in self.accounts if a.alive)
        return self.external_cash + equity - self.purchase_cost

    def snapshot(self) -> dict:
        return {
            "n_accounts_bought": len(self.accounts),
            "n_active": self.n_active(),
            "external_cash": self.external_cash,
            "purchase_cost": self.purchase_cost,
            "ecosystem_value": self.ecosystem_value(),
            "n_withdrawals": len(self.all_withdrawals),
            "total_withdrawn": sum(w.amount for w in self.all_withdrawals),
        }
