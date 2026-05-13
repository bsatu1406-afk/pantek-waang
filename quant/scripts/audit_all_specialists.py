"""Run the standalone audit on EVERY specialist module discovered in
`src/strategy/specialists/`, and emit a `reports/specialist_audit.csv` row
per specialist.

Skips modules whose name starts with '_' (private/interface modules).
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from datetime import date
from pathlib import Path

from src.backtest import standalone, report_generator

REPO_ROOT = Path(__file__).resolve().parents[1]
SPECIALISTS_DIR = REPO_ROOT / "src" / "strategy" / "specialists"

log = logging.getLogger(__name__)


def discover() -> list[str]:
    return sorted(
        p.stem for p in SPECIALISTS_DIR.glob("*.py")
        if not p.stem.startswith("_") and p.stem != "__init__"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dev-period", default="2020-01:2023-12")
    p.add_argument("--holdout-period", default="2024-01:2024-12")
    p.add_argument("--specialists", nargs="*", default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dev = standalone._parse_period(args.dev_period)
    ho = standalone._parse_period(args.holdout_period)
    specialists = args.specialists or discover()

    rows: list[dict] = []
    for name in specialists:
        log.info("auditing %s", name)
        try:
            out = standalone.run(name, dev, ho)
        except Exception as e:
            log.exception("specialist %s crashed: %s", name, e)
            rows.append({
                "specialist": name, "dev_period": args.dev_period,
                "dev_trades": 0, "dev_wr": 0, "dev_pf": 0, "dev_net": 0,
                "dev_verdict": "CRASH",
                "holdout_period": args.holdout_period,
                "holdout_trades": 0, "holdout_wr": 0, "holdout_pf": 0,
                "holdout_net": 0, "holdout_verdict": "CRASH",
                "branch": "", "notes": str(e)[:200],
            })
            continue
        d = out["dev"]
        h = out.get("holdout", {})
        rows.append({
            "specialist": name,
            "dev_period": args.dev_period,
            "dev_trades": d["trades"], "dev_wr": round(d["wr"], 4),
            "dev_pf": round(d["pf"], 4) if d["pf"] != float("inf") else "inf",
            "dev_net": round(d["net_pnl"], 2),
            "dev_verdict": out.get("dev_verdict", "?"),
            "holdout_period": args.holdout_period,
            "holdout_trades": h.get("trades", 0),
            "holdout_wr": round(h.get("wr", 0), 4),
            "holdout_pf": round(h.get("pf", 0), 4) if h.get("pf", 0) != float("inf") else "inf",
            "holdout_net": round(h.get("net_pnl", 0), 2),
            "holdout_verdict": out.get("holdout_verdict", "?"),
            "branch": "", "notes": "",
        })
        log.info("%s: dev=%s holdout=%s", name, out.get("dev_verdict"), out.get("holdout_verdict", "n/a"))
    out_path = report_generator.emit_specialist_audit(rows)
    log.info("audit table -> %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
