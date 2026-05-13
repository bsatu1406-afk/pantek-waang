"""Standalone-style audit for ML Ranker on SYNTHETIC bars.

The official `python -m src.backtest.standalone` harness yields 0 sessions
when no Parquet files exist on disk. This script replicates the harness
math (1-contract MES; stop/target/EOD exits) against synthetic bars so we
can produce a numeric audit output for the structured report. The
Orchestrator will re-run the official harness on the real 5-year corpus
after merge.

Usage:
    PYTHONPATH=. python scripts/audit_ml_ranker_synthetic.py [--days 60]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from scripts.train_ml_ranker import _make_session_bars
from src.backtest.standalone import _aggregate, _simulate_trade, _verdict
from src.strategy.specialists import ml_ranker as ml

log = logging.getLogger("audit_ml_ranker_synth")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--start", default="2024-01-02")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--out", default="reports/ml_ranker_synthetic_audit.json")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    params = ml.MLRankerParams()
    base = 4500.0
    cur = date.fromisoformat(args.start)
    sessions = 0
    results = []
    while sessions < args.days:
        if cur.weekday() >= 5:
            cur += timedelta(days=1)
            continue
        bars = _make_session_bars(cur, rng=rng, base_price=base)
        sigs = ml.generate_signals(bars, params)
        for s in sigs:
            results.append(_simulate_trade(s, bars))
        sessions += 1
        cur += timedelta(days=1)
        base += rng.normal(0.0, 1.0)

    agg = _aggregate(results)
    verdict = _verdict(agg["net_pnl"], agg["wr"], agg["pf"], agg["trades"])
    log.info("synthetic audit: sessions=%d trades=%d wr=%.1f%% pf=%.2f net=$%.2f long_wr=%.1f%% short_wr=%.1f%% verdict=%s",
             sessions, agg["trades"], agg["wr"] * 100, agg["pf"], agg["net_pnl"],
             agg["long_wr"] * 100, agg["short_wr"] * 100, verdict.value)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "specialist": "ml_ranker",
        "data_source": "synthetic",
        "sessions": sessions,
        "agg": agg,
        "verdict": verdict.value,
        "note": (
            "Synthetic OU bars; ml_ranker emits candidate signals from VWAP/"
            "momentum rules and scores them with the trained checkpoint. "
            "Not a substitute for the Orchestrator's real-data audit."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
