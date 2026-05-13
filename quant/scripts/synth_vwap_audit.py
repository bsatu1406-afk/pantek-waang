"""Generate synthetic ES OHLCV-1s sessions and run the standalone audit on
`vwap_extended`.

This is a wiring-only smoke test for child-agent VMs that don't have the
parent's Databento-downloaded data. It does NOT prove edge — synthetic data
has no real microstructure — but it verifies:

  1. `BarLoader.iter_sessions` correctly picks up our parquet files.
  2. `vwap_extended.generate_signals` runs end-to-end against the loader.
  3. The audit harness can simulate trades and emit per-side WR/PF.

Run:
    PYTHONPATH=. python scripts/synth_vwap_audit.py
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = REPO_ROOT / "data" / "processed" / "bars_1s"

UTC = timezone.utc


def make_session(sd: date, seed: int) -> pl.DataFrame:
    """Create one synthetic RTH session: 23,400 bars (6.5h * 3600s).

    Price walks as a mean-reverting OU around a slow drift. We inject 1-2
    extension legs followed by sharp pullbacks to give the VWAP specialist
    both LONG and SHORT setup opportunities.
    """
    rng = np.random.default_rng(seed)
    n = 23_400  # 6.5h * 60min * 60s
    start = datetime(sd.year, sd.month, sd.day, 14, 30, tzinfo=UTC)  # 09:30 ET ≈ 14:30 UTC

    # OU process around 5000
    mu = 5000.0
    theta = 0.001
    sigma = 0.05
    x = np.empty(n)
    x[0] = mu
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) + sigma * rng.standard_normal()

    # Inject two extension+pullback legs (one up, one down) to seed setups
    seg = n // 4
    # Up extension at hour 1.5
    a = seg
    for i in range(120):
        x[a + i] += 5.0 * (i / 120)
    for i in range(60):
        x[a + 120 + i] += 5.0 - 6.5 * (i / 60)  # sharp pullback
    # Down extension at hour 4
    b = 3 * seg
    for i in range(120):
        x[b + i] -= 5.0 * (i / 120)
    for i in range(60):
        x[b + 120 + i] -= 5.0 - 6.5 * (i / 60)

    closes = x
    opens = np.concatenate(([closes[0]], closes[:-1]))
    highs = np.maximum(opens, closes) + 0.25 + 0.25 * rng.random(n)
    lows = np.minimum(opens, closes) - 0.25 - 0.25 * rng.random(n)
    vols = (100 + rng.integers(0, 200, n)).astype(np.int64)
    ts = [start + timedelta(seconds=int(i)) for i in range(n)]

    return pl.DataFrame(
        {
            "ts": ts,
            "session_date": [sd] * n,
            "open": opens.tolist(),
            "high": highs.tolist(),
            "low": lows.tolist(),
            "close": closes.tolist(),
            "volume": vols.tolist(),
        }
    ).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC")))


def write_sessions(start: date, n_days: int) -> list[date]:
    written: list[date] = []
    sd = start
    seed = 0
    while len(written) < n_days:
        # Skip weekends
        if sd.weekday() < 5:
            df = make_session(sd, seed=seed)
            out_dir = PROC_DIR / f"{sd.year:04d}" / f"{sd.month:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"bars_1s_{sd.isoformat()}.parquet"
            df.write_parquet(out)
            written.append(sd)
            seed += 1
        sd = sd + timedelta(days=1)
    return written


def main() -> int:
    # Generate 10 synthetic sessions starting 2024-01-02
    days = write_sessions(date(2024, 1, 2), n_days=10)
    print(f"wrote {len(days)} synthetic sessions: {days[0]} .. {days[-1]}")

    # Now run the standalone audit programmatically
    from src.backtest.standalone import _print_report, run

    out = run("vwap_extended", (days[0], days[-1]))
    _print_report(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
