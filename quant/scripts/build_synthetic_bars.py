"""Generate a small synthetic ES bar fixture for local sanity audits.

This is **not** real market data — it's a deterministic Geometric Brownian
Motion overlaid with an opening-range structure, a slight intra-day drift,
and occasional volatility-contraction segments. Use ONLY for plumbing /
smoke tests; any verdict produced from this is purely a wiring check.

Outputs Parquet files compatible with `src.data.loader.BarLoader`:

    data/processed/bars_1s/<YYYY>/<MM>/bars_1s_<YYYY-MM-DD>.parquet
"""
from __future__ import annotations

import argparse
import math
import random
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "processed" / "bars_1s"


def _rth_seconds_utc(sd: date) -> list[datetime]:
    out: list[datetime] = []
    cur = datetime(sd.year, sd.month, sd.day, 9, 30, 0, tzinfo=ET)
    end = datetime(sd.year, sd.month, sd.day, 16, 0, 0, tzinfo=ET)
    while cur < end:
        out.append(cur.astimezone(UTC))
        cur += timedelta(seconds=1)
    return out


def _gen_day(sd: date, seed: int, base_px: float = 5000.0) -> pl.DataFrame:
    """Generate one synthetic RTH session of 1-s bars."""
    rng = random.Random(seed)
    seconds = _rth_seconds_utc(sd)
    n = len(seconds)
    # Choose a regime per day: 0 = up-trend, 1 = down-trend, 2 = chop
    regime = seed % 3

    drift_per_s = {0: +0.0006, 1: -0.0006, 2: 0.0}[regime]
    sigma_per_s = 0.07  # roughly tick-scale per second

    # Optional contraction segment somewhere mid-session
    contract_start = rng.randint(60 * 60, 60 * 200)   # m=60..200
    contract_end = contract_start + 60 * 15           # 15-min contraction

    px = base_px + rng.uniform(-3.0, 3.0)
    closes: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    opens: list[float] = []
    vols: list[int] = []

    prev_close = px
    for i in range(n):
        in_contract = contract_start <= i < contract_end
        sigma = sigma_per_s * (0.2 if in_contract else 1.0)
        drift = drift_per_s if not in_contract else 0.0
        # increment
        step = drift + rng.gauss(0.0, sigma)
        px = px + step
        # OR-friendly: bias first 5 min to oscillate (no drift, wider sigma)
        if i < 60 * 5:
            px = base_px + 1.5 * math.sin(i / 30.0) + rng.gauss(0.0, 0.25)
        # 1-tick noise around close → high/low
        bar_high = px + abs(rng.gauss(0.0, 0.18)) + 0.25
        bar_low = px - abs(rng.gauss(0.0, 0.18)) - 0.25
        bar_open = prev_close
        bar_close = px
        if bar_high < max(bar_open, bar_close):
            bar_high = max(bar_open, bar_close) + 0.25
        if bar_low > min(bar_open, bar_close):
            bar_low = min(bar_open, bar_close) - 0.25
        opens.append(round(bar_open / 0.25) * 0.25)
        highs.append(round(bar_high / 0.25) * 0.25)
        lows.append(round(bar_low / 0.25) * 0.25)
        closes.append(round(bar_close / 0.25) * 0.25)
        vols.append(rng.randint(40, 220))
        prev_close = bar_close

    return pl.DataFrame(
        {
            "ts": seconds,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
            "session_date": [sd] * n,
        }
    ).with_columns(
        pl.col("ts").cast(pl.Datetime("us", "UTC")),
        pl.col("session_date").cast(pl.Date),
    )


def _is_business_day(d: date) -> bool:
    return d.weekday() < 5


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-01-02")
    p.add_argument("--end", default="2024-01-31")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cur = start
    seed = args.seed
    n_written = 0
    while cur <= end:
        if not _is_business_day(cur):
            cur += timedelta(days=1)
            continue
        df = _gen_day(cur, seed=seed)
        seed += 1
        sub = OUT_DIR / f"{cur.year:04d}" / f"{cur.month:02d}"
        sub.mkdir(parents=True, exist_ok=True)
        out = sub / f"bars_1s_{cur.isoformat()}.parquet"
        df.write_parquet(out)
        print(f"wrote {out} ({df.height} rows)")
        n_written += 1
        cur += timedelta(days=1)
    print(f"done: wrote {n_written} session days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
