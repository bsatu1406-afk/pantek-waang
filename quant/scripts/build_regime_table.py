"""Build the regime classification table from processed RTH bars and print a
year-over-year heatmap + stability verdict.

Usage::

    PYTHONPATH=. python scripts/build_regime_table.py
    PYTHONPATH=. python scripts/build_regime_table.py --start 2020-05-13 --end 2025-05-13

The script reads bars via ``BarLoader`` (the parent-provided loader at
``src/data/loader.py``), runs ``classify_regimes``, persists the table to
``data/processed/regime/regime_table.parquet``, and prints diagnostics.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl

from src.data.loader import BarLoader
from src.strategy.specialists.regime import (
    REGIME_TABLE_PATH,
    Regime,
    classify_regimes,
    persist_regime_table,
    stability_stddev_pp,
    yearly_distribution,
)


def _verdict_from_stddev(stddev_pp: float) -> str:
    if stddev_pp < 8.0:
        return "POSITIVE_EDGE"
    if stddev_pp < 15.0:
        return "MARGINAL"
    return "NEGATIVE"


def _print_heatmap(table: pl.DataFrame) -> None:
    dist = yearly_distribution(table)
    if dist.is_empty():
        print("(no rows to summarize)")
        return
    years = sorted(set(dist["year"].to_list()))
    regimes = [r.value for r in Regime]
    print()
    header = f"{'regime':<16}" + "".join(f"{yr:>8}" for yr in years)
    print(header)
    print("-" * len(header))
    for r in regimes:
        line = f"{r:<16}"
        for yr in years:
            sel = dist.filter(
                (pl.col("year") == yr) & (pl.col("regime") == r)
            )
            share = sel["share"][0] if sel.height else 0.0
            line += f"{share * 100:>7.1f}%"
        print(line)
    # also print totals
    totals_line = f"{'(n sessions)':<16}"
    for yr in years:
        n = dist.filter(pl.col("year") == yr)["total"]
        n_val = int(n[0]) if n.len() else 0
        totals_line += f"{n_val:>8}"
    print(totals_line)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=None, help="ISO YYYY-MM-DD; default = all available")
    p.add_argument("--end", default=None, help="ISO YYYY-MM-DD; default = all available")
    p.add_argument("--out", default=str(REGIME_TABLE_PATH))
    args = p.parse_args()

    loader = BarLoader()
    available = loader.available_session_dates()
    if not available:
        print("[fatal] no processed bars found under data/processed/bars_1s/")
        print("        run scripts/download_data.py + python -m src.data.processor first")
        return 2
    start = date.fromisoformat(args.start) if args.start else available[0]
    end = date.fromisoformat(args.end) if args.end else available[-1]

    print(f"[load] bars_1s from {start} .. {end} ({len(available)} sessions on disk)")
    bars = loader.load_range(start, end)
    print(f"[load] got {bars.height:,} 1s bars covering "
          f"{bars['session_date'].n_unique() if not bars.is_empty() else 0} session days")

    if bars.is_empty():
        print("[fatal] no bars in the requested window")
        return 2

    table = classify_regimes(bars)
    out_path = persist_regime_table(table, Path(args.out))
    print(f"[write] {out_path}  ({table.height} rows)")

    # Overall distribution
    overall = (
        table.group_by("regime").agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / table.height).alias("share"))
        .sort("share", descending=True)
    )
    print("\noverall regime distribution:")
    for row in overall.iter_rows(named=True):
        print(f"  {row['regime']:<16} {row['n']:>5}  {row['share'] * 100:5.1f}%")

    # Year-over-year heatmap
    print("\nyear-over-year regime distribution heatmap (% of sessions):")
    _print_heatmap(table)

    # Stability verdict
    stddev_pp = stability_stddev_pp(table)
    verdict = _verdict_from_stddev(stddev_pp)
    print(f"\nstability std-dev (pp): {stddev_pp:.2f}")
    print(f"regime_verdict: {verdict}")

    if table["session_date"].n_unique() < 200:
        print("\n[caveat] fewer than 200 session days classified — the year-over-year")
        print("         stability metric is only meaningful with multi-year data.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
