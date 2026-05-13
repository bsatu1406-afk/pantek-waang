"""Final report generator (Section 6 deliverables A-G + monthly cashflow).

Reads the engine state from a backtest run and emits:

- A. `reports/backtest_summary.md`       — headline narrative
- B. `reports/per_account_detail.csv`    — one row per account ever opened
- C. `reports/monthly_cashflow.csv`      — per-month rollup
- D. `reports/specialist_audit.csv`      — per-specialist verdict table
- E. `reports/walk_forward.json`         — fold metrics (from Agent #11)
- F. `reports/events.json`               — full engine event log
- G. `reports/engine_rules.md`           — copy of docs/ENGINE_RULES.md
                                            (for completeness)

This module is called from `runner.py` at the end of a backtest. It is also
runnable standalone:

    PYTHONPATH=. python -m src.backtest.report_generator
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
DOCS_DIR = REPO_ROOT / "docs"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def emit_engine_rules_copy() -> Path:
    src = DOCS_DIR / "ENGINE_RULES.md"
    dst = REPORTS_DIR / "engine_rules.md"
    if src.exists():
        shutil.copy(src, dst)
    return dst


def emit_specialist_audit(audit_rows: list[dict]) -> Path:
    path = REPORTS_DIR / "specialist_audit.csv"
    if not audit_rows:
        path.write_text(
            "specialist,dev_period,dev_trades,dev_wr,dev_pf,dev_net,dev_verdict,"
            "holdout_period,holdout_trades,holdout_wr,holdout_pf,holdout_net,"
            "holdout_verdict,branch,notes\n"
        )
        return path
    pl.DataFrame(audit_rows).write_csv(path)
    return path


def emit_summary_narrative(snap: dict, additional: dict | None = None) -> Path:
    """Produce a human-readable narrative of the backtest result.

    `snap` is the dict returned by `PropFirmEngine.snapshot()`.
    `additional` may include {"period": "2020-05..2025-05", "specialists": [...],
                              "verdicts": [...]}.
    """
    lines: list[str] = []
    lines.append("# Orderflow Prop v4 — Backtest Summary\n")
    if additional:
        if "period" in additional:
            lines.append(f"**Backtest period:** {additional['period']}\n")
        if "specialists" in additional:
            lines.append("**Active specialists:**")
            for s in additional["specialists"]:
                v = additional.get("verdicts", {}).get(s, "?")
                lines.append(f"- `{s}` ({v})")
            lines.append("")

    lines.append("## Aggregate result")
    lines.append("")
    lines.append(f"- Accounts ever opened: **{snap['n_accounts_bought']}**")
    lines.append(f"- Active at end: **{snap['n_active']}**")
    lines.append(f"- Account purchases cost: **${snap['purchase_cost']:,.2f}**")
    lines.append(f"- External cash withdrawn: **${snap['external_cash']:,.2f}**")
    lines.append(
        f"- Total withdrawals (raw $): **${snap['total_withdrawn']:,.2f}** "
        f"in **{snap['n_withdrawals']}** cycles"
    )
    lines.append(
        f"- Ecosystem value (cash + active equity − purchases): "
        f"**${snap['ecosystem_value']:,.2f}**"
    )
    net = snap['external_cash'] - snap['purchase_cost']
    lines.append(f"- **Trader net cash:** ${net:,.2f}\n")

    lines.append("## Files in this folder")
    lines.append("")
    for name, desc in [
        ("per_account_detail.csv", "One row per account; final state + cumulative pnl."),
        ("monthly_cashflow.csv", "Monthly account buys / breaches / withdrawals."),
        ("specialist_audit.csv", "Per-specialist standalone 1c MES verdict + holdout."),
        ("walk_forward.json", "Per-fold metrics + aggregate (mean/std WR/PF)."),
        ("events.json", "Full engine event log (account opens, breaches, withdrawals)."),
        ("engine_rules.md", "Copy of ENGINE_RULES.md — the pinned v4 rules."),
    ]:
        lines.append(f"- `{name}` — {desc}")
    lines.append("")

    path = REPORTS_DIR / "backtest_summary.md"
    path.write_text("\n".join(lines))
    return path
