"""End-to-end ensemble backtest runner.

Walks a calendar of RTH sessions, asks the ensemble for signals, feeds them
into the prop firm engine, and writes per-account + monthly reports.

Usage:
    PYTHONPATH=. python -m src.backtest.runner \\
        --start 2020-05-13 --end 2025-05-13 \\
        --specialists vwap_extended momentum cvd_signals
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional

import polars as pl

from src.common.types import Side
from src.data.loader import BarLoader
from src.simulation.engine import (
    ACCOUNT_COST,
    AccountPhase,
    PropFirmEngine,
    STARTING_EQUITY,
)
from src.strategy.ensemble import Ensemble, EnsembleConfig

log = logging.getLogger(__name__)

ET = timezone(timedelta(hours=-5))  # we just use ts already in UTC for engine boundaries
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_backtest(
    start: date,
    end: date,
    specialists: list[str],
    use_ml: bool = False,
    use_risk: bool = False,
    max_active: int = 8,
    initial_accounts: int = 1,
    initial_capital: float = 0.0,
) -> dict:
    """Run an end-to-end backtest.

    Parameters
    ----------
    initial_accounts : how many accounts to buy on day 0 (trader's initial
        fleet). Each costs ACCOUNT_COST and is tracked in `purchase_cost`.
    initial_capital : starting external_cash for the trader (in addition
        to the accounts they buy upfront). This lets the simulation honor
        a realistic starting bankroll so a few early breaches do not end
        the run before any withdrawals can spawn replacements.
    """
    cfg = EnsembleConfig(
        enabled_specialists=specialists,
        use_ml_ranker=use_ml,
        use_risk_manager=use_risk,
    )
    ens = Ensemble(cfg)
    eng = PropFirmEngine(max_active=max_active)
    # Seed first account(s)
    loader = BarLoader()
    available_dates = [d for d in loader.available_session_dates() if start <= d <= end]
    if not available_dates:
        log.error("no bars available in [%s, %s]", start, end)
        return {}
    if initial_capital > 0:
        eng.external_cash = initial_capital
    for i in range(max(1, initial_accounts)):
        eng.open_account(available_dates[0])

    for sd in available_dates:
        bars = loader.load_session(sd)
        if bars is None or bars.is_empty():
            continue
        # session boundaries (polars datetime returns python datetime directly)
        first_ts = bars.head(1)["ts"][0]
        last_ts = bars.tail(1)["ts"][0]
        eng.on_session_open(sd, first_ts)
        sigs = ens.generate_for_session(sd, bars)
        sigs_by_ts = sorted(sigs, key=lambda s: s.timestamp)
        # Bar-driven loop — 1s granularity.
        bars_rows = bars.iter_rows(named=True)
        sig_idx = 0
        n_sigs = len(sigs_by_ts)
        for row in bars_rows:
            ts = row["ts"]
            price = (row["high"] + row["low"]) / 2  # mid as MTM
            # Open new signals up to this bar
            while sig_idx < n_sigs and sigs_by_ts[sig_idx].timestamp <= ts:
                s = sigs_by_ts[sig_idx]
                for acct in eng.accounts:
                    if not acct.alive:
                        continue
                    ok, _ = ens.veto_for_account(acct, s, state={"day": sd})
                    if not ok:
                        continue
                    eng.open_position(
                        acct, ts, s.side, qty=1,
                        entry_price=s.entry_price,
                        stop_price=s.stop_price,
                        target_price=s.target_price,
                        specialist=s.specialist,
                        signal_id=s.setup_id,
                    )
                sig_idx += 1
            # MTM + lock/cap checks
            eng.mark_to_market(ts, price)
            # Per-account stop / target hits on open positions
            for acct in eng.accounts:
                if acct.position is None or not acct.alive:
                    continue
                p = acct.position
                if p.side == Side.LONG:
                    if row["low"] <= p.stop_price:
                        eng.close_position(acct, ts, p.stop_price, "stop")
                    elif p.target_price is not None and row["high"] >= p.target_price:
                        eng.close_position(acct, ts, p.target_price, "tp")
                else:
                    if row["high"] >= p.stop_price:
                        eng.close_position(acct, ts, p.stop_price, "stop")
                    elif p.target_price is not None and row["low"] <= p.target_price:
                        eng.close_position(acct, ts, p.target_price, "tp")
        eng.on_session_close(sd, last_ts)
        # Reinvest withdrawn cash at EOD
        eng.reinvest_into_new_accounts(sd)

    snap = eng.snapshot()
    return {"snapshot": snap, "engine": eng}


def write_per_account_csv(eng: PropFirmEngine, path: Path) -> None:
    rows = []
    for a in eng.accounts:
        n_trades = len(a.trades)
        wins = sum(1 for t in a.trades if t.pnl > 0)
        wr = wins / n_trades if n_trades else 0.0
        cum_pnl = sum(t.pnl for t in a.trades)
        rows.append({
            "id": a.id,
            "buy_date": a.buy_date.isoformat(),
            "final_phase": a.phase.value,
            "pass_eval_date": a.pass_eval_date.isoformat() if a.pass_eval_date else "",
            "breach_date": a.breach_date.isoformat() if a.breach_date else "",
            "bucket_remaining": round(a.bucket_remaining, 2),
            "n_daily_locks": a.daily_locks,
            "n_wd_cycles": a.cycles_done,
            "n_wd_50pct": a.cycles_50pct,
            "n_wd_80pct": a.cycles_80pct,
            "total_withdrawn": round(a.total_withdrawn, 2),
            "final_equity": round(a.equity, 2),
            "cum_pnl": round(cum_pnl, 2),
            "winning_days_total": a.n_winning_days_total,
            "losing_days_total": a.n_losing_days_total,
            "n_trades": n_trades,
            "wr": round(wr, 4),
            "best_day_pnl": round(a.best_day_pnl, 2),
            "worst_day_pnl": round(a.worst_day_pnl, 2),
        })
    if not rows:
        path.write_text("")
        return
    df = pl.DataFrame(rows)
    df.write_csv(path)


def write_monthly_cashflow_csv(eng: PropFirmEngine, path: Path) -> None:
    # Group withdrawals by month, also count trades + breaches per month.
    if not eng.events:
        path.write_text("")
        return
    months: dict[str, dict] = {}
    def _bucket(d: str) -> str:
        return d[:7]
    for ev in eng.events:
        m = _bucket(ev["date"])
        rec = months.setdefault(m, {
            "month": m, "n_buys": 0, "n_breaches": 0, "wd_50": 0, "wd_80": 0,
            "withdrawn": 0.0,
        })
        if ev["event"] == "OPEN_ACCOUNT":
            rec["n_buys"] += 1
        elif ev["event"] == "BREACH":
            rec["n_breaches"] += 1
        elif ev["event"] == "WITHDRAW":
            if abs(ev["pct"] - 0.50) < 1e-6:
                rec["wd_50"] += 1
            else:
                rec["wd_80"] += 1
            rec["withdrawn"] += ev["amount"]
    rows = list(months.values())
    rows.sort(key=lambda r: r["month"])
    # active accounts at month end and ecosystem net are derived; skip for now.
    pl.DataFrame(rows).write_csv(path)


def write_summary_md(eng: PropFirmEngine, path: Path) -> None:
    snap = eng.snapshot()
    total_wd = snap["total_withdrawn"]
    n_wd = snap["n_withdrawals"]
    purch = snap["purchase_cost"]
    out = []
    out.append("# Backtest Summary\n")
    out.append("## Headline numbers")
    out.append("")
    out.append(f"- Accounts bought: **{snap['n_accounts_bought']}**")
    out.append(f"- Active at end: **{snap['n_active']}**")
    out.append(f"- Purchase cost: **${purch:,.2f}**")
    out.append(f"- External cash withdrawn: **${snap['external_cash']:,.2f}**")
    out.append(f"- Total withdrawals (raw $): **${total_wd:,.2f}** across **{n_wd}** cycles")
    out.append(f"- Ecosystem value (cash + active equity - purchase): **${snap['ecosystem_value']:,.2f}**")
    out.append(f"- Trader net cashflow: **${snap['external_cash'] - purch:,.2f}**")
    out.append("")
    out.append("See `per_account_detail.csv` and `monthly_cashflow.csv` for breakdowns.")
    path.write_text("\n".join(out))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2020-05-13")
    p.add_argument("--end", default="2025-05-13")
    p.add_argument("--specialists", nargs="+", default=[])
    p.add_argument("--ml", action="store_true")
    p.add_argument("--risk", action="store_true")
    p.add_argument("--max-active", type=int, default=8)
    p.add_argument("--initial-accounts", type=int, default=1,
                   help="How many accounts to buy on day 0 (default 1).")
    p.add_argument("--initial-capital", type=float, default=0.0,
                   help="Starting external_cash for the trader (default 0).")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.specialists:
        log.error("no specialists enabled — aborting")
        return 2
    res = run_backtest(
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        args.specialists,
        use_ml=args.ml,
        use_risk=args.risk,
        max_active=args.max_active,
        initial_accounts=args.initial_accounts,
        initial_capital=args.initial_capital,
    )
    eng = res["engine"]
    write_per_account_csv(eng, REPORTS_DIR / "per_account_detail.csv")
    write_monthly_cashflow_csv(eng, REPORTS_DIR / "monthly_cashflow.csv")
    write_summary_md(eng, REPORTS_DIR / "backtest_summary.md")
    (REPORTS_DIR / "events.json").write_text(json.dumps(eng.events, indent=2))
    print(json.dumps(res["snapshot"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
