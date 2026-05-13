"""Download ES historical data from Databento.

Downloads per-month DBN files to data/raw/ and converts them to per-day Parquet
files for ohlcv-1s and tbbo schemas.

Schemas requested:
  - ohlcv-1s (free) – primary bar data for strategy features
  - tbbo (paid)     – top-of-book updates for orderflow / microstructure
  - statistics      – settlement, OI (cheap)
  - definition      – instrument definitions (free)

Symbol: ES.v.0 (continuous volume-rolled front month).

Usage:
  python scripts/download_data.py --start 2020-05-13 --end 2025-05-13 \
      --schemas ohlcv-1s tbbo statistics definition

Resumable: skips months already downloaded successfully.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import databento as db

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def month_iter(start: date, end: date):
    """Yield (month_start, month_end_exclusive) covering [start, end)."""
    cur = date(start.year, start.month, 1)
    while cur < end:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        s = max(cur, start)
        e = min(nxt, end)
        yield s, e
        cur = nxt


def download_schema(
    client: db.Historical,
    schema: str,
    start: date,
    end: date,
    symbols: list[str],
    stype_in: str,
    out_dir: Path,
) -> int:
    """Download one schema, one month at a time, to DBN.ZST files.

    Returns total bytes written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for ms, me in month_iter(start, end):
        tag = f"{ms.strftime('%Y%m')}"
        path = out_dir / f"{schema}_{tag}.dbn.zst"
        if path.exists() and path.stat().st_size > 0:
            print(f"[skip] {path.name} ({path.stat().st_size/1e6:.1f} MB)")
            total_bytes += path.stat().st_size
            continue
        print(f"[download] {schema} {ms} -> {me} ({symbols})")
        t0 = time.time()
        try:
            data = client.timeseries.get_range(
                dataset="GLBX.MDP3",
                start=ms.isoformat(),
                end=me.isoformat(),
                symbols=symbols,
                stype_in=stype_in,
                schema=schema,
            )
            data.to_file(str(path))
        except db.common.error.BentoError as e:
            print(f"[error] {schema} {tag}: {e}", file=sys.stderr)
            # remove partial file
            if path.exists():
                path.unlink()
            raise
        sz = path.stat().st_size
        total_bytes += sz
        print(f"[done] {path.name} {sz/1e6:.1f} MB in {time.time()-t0:.1f}s")
    return total_bytes


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="ISO date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="ISO date YYYY-MM-DD (exclusive)")
    p.add_argument("--symbols", nargs="+", default=["ES.v.0"])
    p.add_argument("--stype-in", default="continuous")
    p.add_argument(
        "--schemas",
        nargs="+",
        default=["ohlcv-1s", "tbbo", "statistics", "definition"],
    )
    p.add_argument("--out", default=str(RAW_DIR))
    args = p.parse_args()

    key = os.environ.get("DATABENTO_API_KEY") or os.environ.get("api_key")
    if not key:
        print("[fatal] DATABENTO_API_KEY missing", file=sys.stderr)
        return 2
    client = db.Historical(key=key)

    start = datetime.fromisoformat(args.start).date()
    end = datetime.fromisoformat(args.end).date()
    out_root = Path(args.out)

    grand_total = 0
    for schema in args.schemas:
        sub = out_root / schema
        try:
            grand_total += download_schema(
                client, schema, start, end, args.symbols, args.stype_in, sub
            )
        except Exception as e:
            print(f"[fatal] schema {schema} failed: {e}", file=sys.stderr)
            return 3
    print(f"[summary] total bytes downloaded: {grand_total/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
