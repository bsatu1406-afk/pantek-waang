"""A/B comparative simulation: prop-firm engine WITH vs WITHOUT the Risk
Manager veto layer.

The simulation drives a multi-account fleet through a synthetic stream of
signals + trade outcomes. The trade-outcome distribution is intentionally
stressed (some catastrophic-loss days, modest wins) so the daily-lock
mechanism activates and we can observe the veto layer's effect on
breach rate + total withdrawal.

Run:
    PYTHONPATH=. python scripts/ab_risk_manager.py [--seed 42] [--n-days 90]
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from src.common.types import Side, Signal
from src.simulation.engine import (
    AccountPhase,
    PropFirmEngine,
)
from src.strategy.specialists.risk_manager import (
    RiskParams,
    allocate_signal,
)

UTC = UTC


# ---------------------------------------------------------------------------
# Synthetic signal stream
# ---------------------------------------------------------------------------


@dataclass
class SyntheticTrade:
    """A signal + the realised exit price we will drive the engine with.

    We pre-compute exit prices so the A/B comparison sees IDENTICAL market
    outcomes for both arms — the only thing that changes is whether the
    veto layer accepts the signal.
    """

    signal: Signal
    exit_price: float            # engine receives this on the post-entry MTM
    pnl_points: float            # signed points for diagnostics


def _bday_after(d: date, n: int) -> date:
    """Return the n-th business day after ``d`` (Mon-Fri, ignores holidays)."""
    cur = d
    added = 0
    while added < n:
        cur = cur + timedelta(days=1)
        if cur.weekday() < 5:
            added += 1
    return cur


def _session_open_utc(d: date) -> datetime:
    """Approx 09:30 ET in UTC. Use 14:30 UTC year-round (winter EST); the
    engine doesn't care about DST in this synthetic harness."""
    return datetime(d.year, d.month, d.day, 14, 30, tzinfo=UTC)


def _session_close_utc(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 21, 0, tzinfo=UTC)


def _build_signal_stream(
    n_days: int,
    signals_per_day: tuple[int, int],
    seed: int,
) -> list[list[SyntheticTrade]]:
    """Return one list of SyntheticTrade per simulated day.

    Trade outcomes are drawn from a heavy-left-tail distribution so that
    daily-lock events occur regularly in the no-veto baseline.
    """
    rng = random.Random(seed)
    start = date(2024, 1, 2)
    out: list[list[SyntheticTrade]] = []

    # Outcome pools. Points on ES; MES is 1/10 size so 1 pt = $5 on MES.
    # The $250 daily-lock threshold corresponds to ~50 pt single-trade loss.
    # Confidence is correlated with outcome quality below: a confidence of
    # 0.8 means ~80% chance the trade comes from ``good_outcomes``, 20%
    # from ``bad_outcomes`` (which contains the catastrophic tail). This
    # mirrors the real ensemble where ML-ranked confidence is a meaningful
    # predictor — and gives the veto's ``min_confidence`` knob real value.
    good_outcomes = [
        8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0,
        9.0, 11.0, 13.0, 15.0, 8.0, 10.0, 12.0, 14.0, 16.0,
        -3.0, -5.0, -4.0,  # small adverse moves can still happen in "good" trades
    ]
    bad_outcomes = [
        -3.0, -5.0, -7.0, -8.0, -10.0, -4.0,
        -15.0, -18.0, -20.0,
        -55.0,                                # ~10% of bad outcomes are catastrophic
        2.0, 4.0,                             # occasional luck even on bad setups
    ]

    for day_idx in range(n_days):
        d = _bday_after(start, day_idx)
        n_sigs = rng.randint(*signals_per_day)
        day_trades: list[SyntheticTrade] = []
        for _ in range(n_sigs):
            # Spread signal times across 09:30 - 15:45 ET. ~10% of signals
            # land after 15:30 ET so the cutoff rule is exercised.
            minutes_after_open = rng.randint(0, 6 * 60 + 15)
            sig_ts = _session_open_utc(d) + timedelta(minutes=minutes_after_open)
            side = Side.LONG if rng.random() < 0.5 else Side.SHORT
            confidence = rng.uniform(0.3, 0.98)
            entry = 5000.0 + rng.uniform(-20.0, 20.0)
            stop_pts = 5.0
            stop = entry - stop_pts if side == Side.LONG else entry + stop_pts
            # Confidence ↔ outcome: high-confidence ⇒ usually draws from
            # good_outcomes; low-confidence ⇒ usually draws from bad_outcomes.
            pool = good_outcomes if rng.random() < confidence else bad_outcomes
            pts = rng.choice(pool)
            sign = 1 if side == Side.LONG else -1
            exit_px = entry + sign * pts
            sig = Signal(
                timestamp=sig_ts,
                side=side,
                confidence=confidence,
                specialist="synthetic",
                setup_id="synth",
                entry_price=entry,
                stop_price=stop,
            )
            day_trades.append(SyntheticTrade(signal=sig, exit_price=exit_px, pnl_points=pts))
        day_trades.sort(key=lambda t: t.signal.timestamp)
        out.append(day_trades)
    return out


# ---------------------------------------------------------------------------
# Single-arm simulation
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    label: str
    n_accounts: int
    n_breached: int
    breach_rate: float
    total_withdrawn: float
    ecosystem_value: float
    n_trades_taken: int
    n_signals_seen: int
    n_vetoed: int
    n_daily_locks: int


def _open_initial_accounts(eng: PropFirmEngine, n: int, day: date) -> None:
    for _ in range(n):
        if eng.can_buy_account():
            eng.open_account(day)


def _run_one_arm(
    label: str,
    stream: list[list[SyntheticTrade]],
    use_veto: bool,
    n_initial_accounts: int = 4,
    params: RiskParams | None = None,
) -> ArmResult:
    params = params or RiskParams()
    eng = PropFirmEngine()

    # Start with a fleet; the engine reinvests cash automatically at EOD.
    if not stream or not stream[0]:
        raise RuntimeError("empty synthetic stream")
    seed_day = stream[0][0].signal.timestamp.date()
    _open_initial_accounts(eng, n_initial_accounts, seed_day)

    n_signals_seen = 0
    n_vetoed = 0
    n_trades_taken = 0
    n_daily_locks = 0

    for day_trades in stream:
        if not day_trades:
            continue
        d = day_trades[0].signal.timestamp.date()
        eng.on_session_open(d, _session_open_utc(d))

        # Snapshot daily_locks count to compute per-day lock delta
        locks_before = sum(a.daily_locks for a in eng.accounts)

        for trade in day_trades:
            n_signals_seen += 1
            sig = trade.signal

            # Determine which accounts take this signal.
            alive = [a for a in eng.accounts if a.alive]
            if use_veto:
                allocations = allocate_signal(alive, sig, {}, params)
                if not allocations:
                    n_vetoed += 1
                # Account-specific veto reasons are already in allocate_signal.
            else:
                # Baseline: every alive account takes the signal at qty=1, no
                # veto layer whatsoever. Engine still enforces locked_today /
                # profit_capped_today / position_open, so this is the
                # legitimate "no risk manager" arm.
                allocations = [(a, 1) for a in alive]

            for acct, qty in allocations:
                if qty <= 0:
                    continue
                # Engine itself will reject if locked / position open — that
                # is the engine's invariant, not the veto.
                opened = eng.open_position(
                    acct, sig.timestamp, sig.side, qty=qty,
                    entry_price=sig.entry_price,
                    stop_price=sig.stop_price,
                    target_price=None,
                    specialist=sig.specialist,
                    signal_id=sig.setup_id,
                )
                if not opened:
                    continue
                n_trades_taken += 1
                # Drive a single MTM at the worst-case adverse price (so the
                # engine can fire daily_lock if the move is catastrophic),
                # then close at the realised exit.
                adverse_px = (
                    sig.entry_price + trade.pnl_points
                    if trade.pnl_points < 0
                    else sig.entry_price + (1.0 if sig.side == Side.LONG else -1.0)
                )
                eng.mark_to_market(sig.timestamp + timedelta(seconds=30), adverse_px)
                if acct.position is not None:
                    eng.close_position(
                        acct, sig.timestamp + timedelta(seconds=60),
                        trade.exit_price, reason="synthetic_exit",
                    )

        # Close session.
        eng.on_session_close(d, _session_close_utc(d))
        locks_after = sum(a.daily_locks for a in eng.accounts)
        n_daily_locks += (locks_after - locks_before)
        eng.reinvest_into_new_accounts(d)

    n_breached = sum(
        1 for a in eng.accounts
        if a.phase in (AccountPhase.BREACHED_EVAL, AccountPhase.PASSED_EVAL_BREACHED_FUNDED)
    )
    snap = eng.snapshot()
    return ArmResult(
        label=label,
        n_accounts=len(eng.accounts),
        n_breached=n_breached,
        breach_rate=n_breached / max(len(eng.accounts), 1),
        total_withdrawn=snap["total_withdrawn"],
        ecosystem_value=snap["ecosystem_value"],
        n_trades_taken=n_trades_taken,
        n_signals_seen=n_signals_seen,
        n_vetoed=n_vetoed,
        n_daily_locks=n_daily_locks,
    )


# ---------------------------------------------------------------------------
# A/B harness
# ---------------------------------------------------------------------------


def run_ab_simulation(
    n_days: int = 90,
    signals_per_day: tuple[int, int] = (3, 6),
    seed: int = 42,
    n_initial_accounts: int = 4,
    params: RiskParams | None = None,
) -> dict:
    stream = _build_signal_stream(n_days, signals_per_day, seed)

    baseline = _run_one_arm(
        "WITHOUT veto", stream, use_veto=False,
        n_initial_accounts=n_initial_accounts, params=params,
    )
    veto = _run_one_arm(
        "WITH    veto", stream, use_veto=True,
        n_initial_accounts=n_initial_accounts, params=params,
    )

    delta_breach = baseline.breach_rate - veto.breach_rate
    breach_drop_pct = (
        (baseline.breach_rate - veto.breach_rate) / baseline.breach_rate * 100
        if baseline.breach_rate > 0
        else 0.0
    )
    delta_withdrawal = veto.total_withdrawn - baseline.total_withdrawn
    wd_pct_kept = (
        veto.total_withdrawn / baseline.total_withdrawn * 100
        if baseline.total_withdrawn > 0
        else 100.0
    )

    # Verdict per brief:
    #   POSITIVE: breach rate drops >= 30%, withdrawal >= 90% of baseline.
    #   MARGINAL: 10-30% breach drop OR withdrawal < 90% of baseline.
    #   NEGATIVE: < 10% breach drop OR > 25% withdrawal loss.
    if breach_drop_pct >= 30.0 and wd_pct_kept >= 90.0:
        verdict = "POSITIVE_EDGE"
    elif breach_drop_pct < 10.0 or wd_pct_kept < 75.0:
        verdict = "NEGATIVE"
    else:
        verdict = "MARGINAL"

    return {
        "baseline": baseline,
        "veto": veto,
        "delta_breach_rate": delta_breach,
        "breach_drop_pct": breach_drop_pct,
        "delta_withdrawal": delta_withdrawal,
        "withdrawal_pct_kept": wd_pct_kept,
        "verdict": verdict,
    }


def _fmt_arm(r: ArmResult) -> str:
    return (
        f"  {r.label}: "
        f"accts={r.n_accounts} breached={r.n_breached} ({r.breach_rate*100:.1f}%) "
        f"locks={r.n_daily_locks} "
        f"trades={r.n_trades_taken}/{r.n_signals_seen} "
        f"vetoed={r.n_vetoed} "
        f"withdrawn=${r.total_withdrawn:,.2f} "
        f"ecosystem=${r.ecosystem_value:,.2f}"
    )


def print_report(out: dict) -> None:
    print("=" * 78)
    print("A/B simulation — Risk Manager veto layer (Agent #10)")
    print("=" * 78)
    print(_fmt_arm(out["baseline"]))
    print(_fmt_arm(out["veto"]))
    print("-" * 78)
    print(
        f"  delta breach rate: {out['delta_breach_rate']*100:+.2f} pp  "
        f"({out['breach_drop_pct']:+.1f}% drop)"
    )
    print(
        f"  delta withdrawal : ${out['delta_withdrawal']:+,.2f}  "
        f"(kept {out['withdrawal_pct_kept']:.1f}% of baseline)"
    )
    print(f"  verdict          : {out['verdict']}")
    print("=" * 78)


def run_ab_multi_seed(
    seeds: list[int],
    n_days: int,
    n_initial_accounts: int,
    params: RiskParams | None = None,
) -> dict:
    """Average the A/B comparison across multiple seeds for a stable verdict."""
    runs = []
    for s in seeds:
        runs.append(
            run_ab_simulation(
                n_days=n_days,
                seed=s,
                n_initial_accounts=n_initial_accounts,
                params=params,
            )
        )
    avg_baseline_br = sum(r["baseline"].breach_rate for r in runs) / len(runs)
    avg_veto_br = sum(r["veto"].breach_rate for r in runs) / len(runs)
    avg_baseline_wd = sum(r["baseline"].total_withdrawn for r in runs) / len(runs)
    avg_veto_wd = sum(r["veto"].total_withdrawn for r in runs) / len(runs)

    breach_drop_pct = (
        (avg_baseline_br - avg_veto_br) / avg_baseline_br * 100
        if avg_baseline_br > 0
        else 0.0
    )
    wd_pct_kept = (
        avg_veto_wd / avg_baseline_wd * 100 if avg_baseline_wd > 0 else 100.0
    )
    if breach_drop_pct >= 30.0 and wd_pct_kept >= 90.0:
        verdict = "POSITIVE_EDGE"
    elif breach_drop_pct < 10.0 or wd_pct_kept < 75.0:
        verdict = "NEGATIVE"
    else:
        verdict = "MARGINAL"
    return {
        "n_seeds": len(seeds),
        "avg_baseline_breach_rate": avg_baseline_br,
        "avg_veto_breach_rate": avg_veto_br,
        "avg_baseline_withdrawn": avg_baseline_wd,
        "avg_veto_withdrawn": avg_veto_wd,
        "breach_drop_pct": breach_drop_pct,
        "withdrawal_pct_kept": wd_pct_kept,
        "verdict": verdict,
        "per_seed": runs,
    }


def print_multi_seed_report(out: dict) -> None:
    print("=" * 78)
    print(f"A/B simulation — Risk Manager (Agent #10) — averaged over {out['n_seeds']} seeds")
    print("=" * 78)
    print(
        f"  WITHOUT veto: avg breach_rate={out['avg_baseline_breach_rate']*100:6.2f}%   "
        f"avg withdrawn=${out['avg_baseline_withdrawn']:,.2f}"
    )
    print(
        f"  WITH    veto: avg breach_rate={out['avg_veto_breach_rate']*100:6.2f}%   "
        f"avg withdrawn=${out['avg_veto_withdrawn']:,.2f}"
    )
    print("-" * 78)
    print(
        f"  breach drop : {out['breach_drop_pct']:+6.2f}%   "
        f"withdrawal kept: {out['withdrawal_pct_kept']:6.2f}% of baseline"
    )
    print(f"  verdict     : {out['verdict']}")
    print("=" * 78)
    print()
    print("Per-seed detail:")
    for run in out["per_seed"]:
        b = run["baseline"]
        v = run["veto"]
        print(
            f"  seed run: WITHOUT br={b.breach_rate*100:5.1f}% wd=${b.total_withdrawn:>10,.2f} "
            f"| WITH br={v.breach_rate*100:5.1f}% wd=${v.total_withdrawn:>10,.2f} "
            f"=> {run['verdict']}"
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 42, 99, 137, 211, 314, 500])
    p.add_argument("--n-days", type=int, default=180)
    p.add_argument("--initial-accounts", type=int, default=4)
    p.add_argument("--single-seed", type=int, default=None,
                   help="If set, run a single seed and print detailed report")
    args = p.parse_args()
    if args.single_seed is not None:
        out = run_ab_simulation(
            n_days=args.n_days,
            seed=args.single_seed,
            n_initial_accounts=args.initial_accounts,
        )
        print_report(out)
    else:
        out = run_ab_multi_seed(
            seeds=args.seeds,
            n_days=args.n_days,
            n_initial_accounts=args.initial_accounts,
        )
        print_multi_seed_report(out)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
