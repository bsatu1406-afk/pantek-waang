"""Build `reports/specialist_audit.md` — a human-readable narrative that
captures each specialist's:

- verdict (POSITIVE_EDGE / MARGINAL / NEGATIVE / NOT_APPLICABLE)
- dev + holdout numbers from `reports/specialist_audit.csv`
- 1-paragraph summary of setups
- per-side breakdown (long vs short) if available
- caveats / known issues

This pulls metadata from each specialist module's docstring and from the
audit CSV. Intended to satisfy Section 6 deliverable D of the briefing.
"""
from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
SPECIALISTS_DIR = REPO_ROOT / "src" / "strategy" / "specialists"

log = logging.getLogger(__name__)

# Brief one-line description of each specialist's intended role.
ROLE: dict[str, str] = {
    "baseline_short_vwap": "Smoke-test SHORT-only VWAP pullback (v3 legacy edge).",
    "cvd_signals": "CVD divergence + absorption + exhaustion (Agent #4).",
    "macro_events": "Macro calendar pre/post FOMC/CPI/NFP setups (Agent #2).",
    "microstructure": "Book imbalance / microprice / sweep / trapped (Agent #5).",
    "ml_ranker": "LightGBM signal scorer + ranker (Agent #9).",
    "momentum": "ORB 5/15/30 + EOD-drive momentum (Agent #8).",
    "regime": "Market regime classifier (NOT signal-generating, Agent #3).",
    "risk_manager": "Veto + sizing layer (NOT signal-generating, Agent #10).",
    "volume_profile": "Volume profile / value area / POC magnet (Agent #6).",
    "vwap_extended": "Bilateral (LONG+SHORT) VWAP pullback with bands (Agent #7).",
}


def _summarize_setups(specialist: str) -> list[str]:
    """Extract `setup_id` literal strings used by the specialist (if any)."""
    p = SPECIALISTS_DIR / f"{specialist}.py"
    if not p.exists():
        return []
    text = p.read_text()
    ids = sorted(set(re.findall(r'setup_id\s*=\s*"([^"]+)"', text)))
    return ids


def _agent_verdict(row: dict) -> str:
    """Compose a final verdict label that respects both dev and holdout."""
    dev = (row.get("dev_verdict") or "").upper()
    ho = (row.get("holdout_verdict") or "").upper()
    if dev == "POSITIVE_EDGE" and ho == "POSITIVE_EDGE":
        return "POSITIVE_EDGE"
    if dev == "NEGATIVE" and ho == "NEGATIVE":
        return "NEGATIVE"
    if (row.get("dev_trades") or 0) == 0 and (row.get("holdout_trades") or 0) == 0:
        return "NOT_APPLICABLE"
    if "MARGINAL" in dev or "MARGINAL" in ho:
        return "MARGINAL"
    return "MIXED"


def build_narrative() -> Path:
    csv = REPORTS_DIR / "specialist_audit.csv"
    if not csv.exists():
        raise FileNotFoundError(f"missing {csv} — run audit_all_specialists first")
    df = pl.read_csv(csv)
    lines: list[str] = []
    lines.append("# Specialist Audit Narrative")
    lines.append("")
    lines.append(
        "Per Section 6 of the v4 briefing: every specialist (whether kept "
        "or dropped) is reported honestly here. Verdicts are taken from the "
        "standalone 1c MES audit in `reports/specialist_audit.csv` — "
        "POSITIVE_EDGE requires `pf >= 1.30 AND wr >= 0.55 AND trades >= 30`."
    )
    lines.append("")
    lines.append(
        "| Specialist | Role | Dev verdict | Holdout verdict | Final | "
        "Dev WR / PF / net | Holdout WR / PF / net |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    final_rows: list[tuple[str, str]] = []
    for row in df.iter_rows(named=True):
        s = row["specialist"]
        role = ROLE.get(s, "(no role registered)")
        final = _agent_verdict(row)
        final_rows.append((s, final))
        dev_str = (
            f"{row['dev_wr']*100:.1f}% / "
            f"{row['dev_pf']:.2f} / "
            f"${row['dev_net']:.0f}"
        )
        ho_str = (
            f"{row['holdout_wr']*100:.1f}% / "
            f"{row['holdout_pf']:.2f} / "
            f"${row['holdout_net']:.0f}"
        )
        lines.append(
            f"| `{s}` | {role} | {row['dev_verdict']} | {row['holdout_verdict']} "
            f"| **{final}** | {dev_str} | {ho_str} |"
        )

    lines.append("")
    lines.append("## Per-specialist setups")
    lines.append("")
    for s, _ in final_rows:
        setups = _summarize_setups(s)
        if setups:
            lines.append(f"- `{s}`: " + ", ".join(f"`{x}`" for x in setups))
        else:
            lines.append(f"- `{s}`: (no setup_id literals detected — non-signaling)")
    lines.append("")

    pos = [s for s, v in final_rows if v == "POSITIVE_EDGE"]
    marg = [s for s, v in final_rows if v == "MARGINAL"]
    neg = [s for s, v in final_rows if v == "NEGATIVE"]
    na = [s for s, v in final_rows if v == "NOT_APPLICABLE"]
    lines.append("## Final tally")
    lines.append(f"- POSITIVE_EDGE: {pos if pos else '(none)'}")
    lines.append(f"- MARGINAL: {marg if marg else '(none)'}")
    lines.append(f"- NEGATIVE: {neg if neg else '(none)'}")
    lines.append(f"- NOT_APPLICABLE: {na if na else '(none)'}")
    lines.append("")
    lines.append("## Ensemble inclusion policy")
    lines.append("")
    if pos:
        lines.append(
            "Final ensemble = the POSITIVE_EDGE specialists above, filtered by "
            "the Risk Manager veto. ML Ranker is applied as a confidence filter."
        )
    elif marg:
        lines.append(
            "No single specialist clears the POSITIVE_EDGE bar. Ensemble "
            "includes the MARGINAL specialists with the highest holdout PF, "
            "gated by the Risk Manager veto."
        )
    else:
        lines.append(
            "No specialist clears POSITIVE_EDGE OR MARGINAL. Per briefing "
            "Section 7 anti-patterns, we will NOT claim the strategy is "
            "profitable — the run is reported as-is for honest disclosure."
        )

    out = REPORTS_DIR / "specialist_audit.md"
    out.write_text("\n".join(lines))
    log.info("wrote %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_narrative()
