"""Walk-forward validation of the ensemble strategy.

This script uses Agent #11's `run_walkforward` harness to roll the
ensemble through the historical data with strict purging between
train/test windows.

The strategy_fn is intentionally a no-op for this MVP (specialists are
not parameter-trained yet — they emit signals based on PINNED params).
What the harness validates here is the *temporal stability* of the
ensemble: do the per-fold WR/PF/net_pnl metrics hold up out-of-sample?

Usage:

    PYTHONPATH=. python scripts/run_walk_forward.py \
        --start 2020-05-13 --end 2025-05-13 \
        --specialists baseline_short_vwap vwap_extended volume_profile \
                      momentum macro_events \
        --train-months 12 --test-months 3 --step-months 3

The output goes to `reports/walk_forward.json` + `reports/walk_forward.md`.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from src.backtest.walk_forward import run_walkforward
from src.common.types import Side
from src.data.loader import BarLoader
from src.simulation.engine import MES_PER_POINT, PropFirmEngine
from src.strategy.ensemble import Ensemble, EnsembleConfig

log = logging.getLogger(__name__)


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _ensemble_backtest(params: dict, bars_by_date: dict):
    """backtest_fn signature for run_walkforward.

    Runs the ensemble + a fresh PropFirmEngine over the given slice and
    returns a metrics dict.
    """
    specialists = params.get("specialists", [])
    use_risk = params.get("use_risk", True)
    cfg = EnsembleConfig(
        enabled_specialists=specialists,
        use_ml_ranker=False,
        use_risk_manager=use_risk,
    )
    ens = Ensemble(cfg)
    eng = PropFirmEngine(max_active_accounts=8)
    eng.open_account(min(bars_by_date), "AWF")
    n_trades = 0
    n_wins = 0
    gross_win = 0.0
    gross_loss = 0.0
    net = 0.0
    n_breach = 0
    for sd in sorted(bars_by_date):
        bars = bars_by_date[sd]
        if bars is None or bars.is_empty():
            continue
        first_ts = bars.head(1)["ts"][0]
        last_ts = bars.tail(1)["ts"][0]
        eng.on_session_open(sd, first_ts)
        sigs = ens.generate_for_session(sd, bars)
        sigs_sorted = sorted(sigs, key=lambda s: s.timestamp)
        sig_idx = 0
        for row in bars.iter_rows(named=True):
            ts = row["ts"]
            price = (row["high"] + row["low"]) / 2
            while sig_idx < len(sigs_sorted) and sigs_sorted[sig_idx].timestamp <= ts:
                s = sigs_sorted[sig_idx]
                sig_idx += 1
                for acct in list(eng.accounts):
                    if not acct.alive or acct.position is not None:
                        continue
                    if acct.locked_today or acct.profit_capped_today:
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
                    break
            for acct in eng.accounts:
                if acct.position is None:
                    continue
                p = acct.position
                hit_target = (
                    p.target_price is not None
                    and (
                        (p.side == Side.LONG and row["high"] >= p.target_price)
                        or (p.side == Side.SHORT and row["low"] <= p.target_price)
                    )
                )
                hit_stop = (
                    p.stop_price is not None
                    and (
                        (p.side == Side.LONG and row["low"] <= p.stop_price)
                        or (p.side == Side.SHORT and row["high"] >= p.stop_price)
                    )
                )
                if hit_stop:
                    pnl = eng.close_position(acct, ts, p.stop_price, reason="stop")
                elif hit_target:
                    pnl = eng.close_position(acct, ts, p.target_price, reason="target")
                else:
                    pnl = None
                if pnl is not None:
                    n_trades += 1
                    if pnl > 0:
                        n_wins += 1
                        gross_win += pnl
                    else:
                        gross_loss += -pnl
                    net += pnl
            eng.mark_to_market(ts, price)
        eng.on_session_close(sd, last_ts)
        for a in eng.accounts:
            if "BREACHED" in a.phase.value.upper():
                pass  # counted at end
    n_breach = sum(1 for a in eng.accounts if not a.alive and "BREACH" in a.phase.value.upper())
    wr = (n_wins / n_trades) if n_trades > 0 else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    n_months = max(1, (max(bars_by_date) - min(bars_by_date)).days / 30.0)
    return {
        "n_trades": n_trades,
        "n_wins": n_wins,
        "wr": round(wr, 4),
        "pf": round(pf, 4) if pf != float("inf") else 999.0,
        "net_pnl": round(net, 2),
        "breach": int(n_breach > 0),
        "n_withdrawals": eng.snapshot()["n_withdrawals"],
        "total_withdrawn": round(eng.snapshot()["total_withdrawn"], 2),
        "n_months": round(n_months, 2),
    }


def _no_train(params: dict | None, bars_by_date: dict) -> dict:
    """Strategy fn — no parameter learning in MVP. Pass params through."""
    return dict(params or {})


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--specialists", nargs="+", required=True)
    p.add_argument("--train-months", type=int, default=12)
    p.add_argument("--test-months", type=int, default=3)
    p.add_argument("--step-months", type=int, default=3)
    p.add_argument("--purge-days", type=int, default=1)
    p.add_argument("--no-risk", action="store_true", help="disable risk manager veto")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    loader = BarLoader()
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    log.info("loading bars_by_date over %s..%s", start, end)
    bars_by_date = {}
    for sd in loader.available_session_dates():
        if sd < start or sd > end:
            continue
        df = loader.load_session(sd)
        if df is None or df.is_empty():
            continue
        bars_by_date[sd] = df
    log.info("loaded %d session days", len(bars_by_date))

    initial_params = {
        "specialists": args.specialists,
        "use_risk": not args.no_risk,
    }
    report = run_walkforward(
        _no_train,
        _ensemble_backtest,
        bars_by_date,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        purge_days=args.purge_days,
        initial_params=initial_params,
    )
    out_json = Path("reports/walk_forward.json")
    out_md = Path("reports/walk_forward.md")
    report.to_json(out_json)
    report.to_markdown(out_md)
    log.info("walk-forward -> %s + %s (%d folds)", out_json, out_md, len(report.folds))
    print(f"folds={len(report.folds)} aggregate={report.aggregate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
