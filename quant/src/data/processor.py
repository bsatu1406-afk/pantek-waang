"""Convert raw Databento DBN.zst files into per-day Parquet bars.

Output schema (per session date, RTH-only filtered):

    bars_1s.parquet               # 1-second OHLCV bars
        ts: datetime (UTC, tz-aware)
        session_date: date (US/Eastern)
        open, high, low, close: float64
        volume: int64
        trades: int64

    tbbo_quotes.parquet           # one row per TBBO update (top-of-book + trade)
        ts: datetime
        session_date: date
        bid_px: float64
        bid_sz: int64
        ask_px: float64
        ask_sz: int64
        price: float64            # trade price (NaN if no trade in this tick)
        size: int64               # trade size
        side: str ('B','S','N')   # buy/sell aggressor / no trade
        action: str ('A','C','T') # add/cancel/trade

    cvd_1s.parquet                # 1-second CVD aggregates (derived)
        ts: datetime
        session_date: date
        delta: int64              # signed volume this second
        cvd: int64                # cumulative session delta (resets at 09:30 ET)
        buy_vol, sell_vol: int64
        n_trades: int64

Files land in `data/processed/<YYYY>/<MM>/<schema>_<DD>.parquet`.

Driver `process_month(month)` walks all DBN files for a given month and
spits out per-session-date Parquet files in one pass.
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import databento as db
import polars as pl
import zoneinfo

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROC_DIR = REPO_ROOT / "data" / "processed"

ET = zoneinfo.ZoneInfo("America/New_York")
RTH_START = time(9, 30)
RTH_END = time(16, 0)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _rth_mask(ts_utc_us: pl.Expr) -> pl.Expr:
    """Polars expression returning a Bool mask of rows whose ts is in RTH ET."""
    # ts is tz-naive UTC microseconds; cast to ET via convert_time_zone
    et = ts_utc_us.dt.convert_time_zone("America/New_York")
    return (
        (et.dt.time() >= RTH_START)
        & (et.dt.time() < RTH_END)
        & (et.dt.weekday() <= 5)
    )


def _session_date(ts_utc_us: pl.Expr) -> pl.Expr:
    return ts_utc_us.dt.convert_time_zone("America/New_York").dt.date()


def _read_dbn_to_polars(path: Path) -> pl.DataFrame:
    """Read a single DBN.zst into Polars via the databento client.

    We avoid the pandas dependency by going through `to_df()`, which the
    client always returns; then convert to polars. For very large TBBO files
    this is the bottleneck — we shard by month to keep memory bounded.
    """
    store = db.DBNStore.from_file(str(path))
    pdf = store.to_df()
    if pdf.empty:
        return pl.DataFrame()
    pdf = pdf.reset_index()
    # ts_event is the canonical event timestamp; sometimes named differently
    if "ts_event" not in pdf.columns and "ts_recv" in pdf.columns:
        pdf["ts_event"] = pdf["ts_recv"]
    df = pl.from_pandas(pdf)
    return df


# -- per-schema processors ---------------------------------------------------


def process_ohlcv_1s(month_tag: str) -> int:
    path = RAW_DIR / "ohlcv-1s" / f"ohlcv-1s_{month_tag}.dbn.zst"
    if not path.exists():
        log.info("skip ohlcv-1s %s (no file)", month_tag)
        return 0
    log.info("ohlcv-1s %s loading...", month_tag)
    df = _read_dbn_to_polars(path)
    if df.is_empty():
        return 0
    # columns: ts_event, open, high, low, close, volume, symbol, instrument_id
    df = df.rename({"ts_event": "ts"}) if "ts_event" in df.columns else df
    if df.schema["ts"] != pl.Datetime("ns", "UTC"):
        df = df.with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC")))
    # Filter to RTH (Mon-Fri 09:30-16:00 ET)
    df = df.with_columns(
        _session_date(pl.col("ts")).alias("session_date"),
    )
    df = df.filter(_rth_mask(pl.col("ts")))
    if df.is_empty():
        log.warning("ohlcv-1s %s: no RTH rows", month_tag)
        return 0
    # Cast prices to float64, volume to int
    for c in ("open", "high", "low", "close"):
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Float64))
    if "volume" in df.columns:
        df = df.with_columns(pl.col("volume").cast(pl.Int64))
    # Trim to expected cols
    keep = ["ts", "session_date", "open", "high", "low", "close", "volume"]
    df = df.select([c for c in keep if c in df.columns])
    # Write per-session-date Parquet
    n_days = 0
    for sd, sub in df.partition_by("session_date", as_dict=True).items():
        # `sd` is a tuple in newer polars; coerce
        if isinstance(sd, tuple):
            sd = sd[0]
        out_dir = PROC_DIR / "bars_1s" / f"{sd.year:04d}" / f"{sd.month:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"bars_1s_{sd.isoformat()}.parquet"
        sub.write_parquet(out, compression="zstd")
        n_days += 1
    log.info("ohlcv-1s %s -> %d session days", month_tag, n_days)
    return n_days


def process_tbbo(month_tag: str) -> int:
    path = RAW_DIR / "tbbo" / f"tbbo_{month_tag}.dbn.zst"
    if not path.exists():
        log.info("skip tbbo %s (no file)", month_tag)
        return 0
    log.info("tbbo %s loading...", month_tag)
    df = _read_dbn_to_polars(path)
    if df.is_empty():
        return 0

    # TBBO columns of interest: ts_event, action, side, price, size,
    # bid_px_00, bid_sz_00, ask_px_00, ask_sz_00 (top-of-book level 0).
    df = df.rename({"ts_event": "ts"}) if "ts_event" in df.columns else df
    if df.schema["ts"] != pl.Datetime("us", "UTC"):
        df = df.with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC")))
    df = df.with_columns(_session_date(pl.col("ts")).alias("session_date"))
    df = df.filter(_rth_mask(pl.col("ts")))
    if df.is_empty():
        return 0
    # Pick standard column names
    col_map = {
        "bid_px_00": "bid_px", "ask_px_00": "ask_px",
        "bid_sz_00": "bid_sz", "ask_sz_00": "ask_sz",
    }
    df = df.rename({k: v for k, v in col_map.items() if k in df.columns})
    keep = [
        "ts", "session_date",
        "bid_px", "bid_sz", "ask_px", "ask_sz",
        "price", "size", "side", "action",
    ]
    df = df.select([c for c in keep if c in df.columns])
    # Cast numerics
    for c in ("bid_px", "ask_px", "price"):
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Float64))
    for c in ("bid_sz", "ask_sz", "size"):
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Int64))
    n_days = 0
    for sd, sub in df.partition_by("session_date", as_dict=True).items():
        if isinstance(sd, tuple):
            sd = sd[0]
        out_dir = PROC_DIR / "tbbo" / f"{sd.year:04d}" / f"{sd.month:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"tbbo_{sd.isoformat()}.parquet"
        sub.write_parquet(out, compression="zstd")
        # Also compute 1s CVD aggregate
        try:
            agg = _aggregate_cvd_1s(sub)
            if not agg.is_empty():
                cvd_dir = PROC_DIR / "cvd_1s" / f"{sd.year:04d}" / f"{sd.month:02d}"
                cvd_dir.mkdir(parents=True, exist_ok=True)
                agg.write_parquet(cvd_dir / f"cvd_1s_{sd.isoformat()}.parquet", compression="zstd")
        except Exception as e:
            log.warning("cvd-1s %s: %s", sd, e)
        n_days += 1
    log.info("tbbo %s -> %d session days", month_tag, n_days)
    gc.collect()
    return n_days


def _aggregate_cvd_1s(quotes: pl.DataFrame) -> pl.DataFrame:
    """From TBBO rows produce 1s CVD aggregate (delta = buy_vol - sell_vol).

    Trade aggressor side comes from the `side` field per Databento spec:
        'B' = buy-side aggression, 'S' = sell-side aggression, 'N' = none.
    """
    if "side" not in quotes.columns or "size" not in quotes.columns:
        return pl.DataFrame()
    q = quotes.filter(pl.col("action") == "T") if "action" in quotes.columns else quotes
    q = q.with_columns([
        pl.col("ts").dt.truncate("1s").alias("ts_1s"),
        pl.when(pl.col("side") == "B").then(pl.col("size")).otherwise(0).alias("buy_vol"),
        pl.when(pl.col("side") == "S").then(pl.col("size")).otherwise(0).alias("sell_vol"),
    ])
    g = q.group_by(["session_date", "ts_1s"]).agg([
        pl.col("buy_vol").sum().alias("buy_vol"),
        pl.col("sell_vol").sum().alias("sell_vol"),
        pl.len().alias("n_trades"),
    ]).sort(["session_date", "ts_1s"]).rename({"ts_1s": "ts"})
    g = g.with_columns((pl.col("buy_vol") - pl.col("sell_vol")).alias("delta"))
    g = g.with_columns(pl.col("delta").cum_sum().over("session_date").alias("cvd"))
    return g


# -- driver -----------------------------------------------------------------


def list_months(start: date, end: date) -> list[str]:
    """Yield 'YYYYMM' month tags between start and end (inclusive of start month)."""
    out: list[str] = []
    cur = date(start.year, start.month, 1)
    while cur < end:
        out.append(cur.strftime("%Y%m"))
        cur = date(cur.year + cur.month // 12, (cur.month % 12) + 1, 1)
    return out


def process_month(month_tag: str) -> dict:
    return {
        "month": month_tag,
        "ohlcv_1s_days": process_ohlcv_1s(month_tag),
        "tbbo_days": process_tbbo(month_tag),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2020-05-13")
    p.add_argument("--end", default="2025-05-13")
    p.add_argument("--months", nargs="*", help="explicit month tags (e.g. 202205)")
    args = p.parse_args()
    if args.months:
        months = args.months
    else:
        months = list_months(date.fromisoformat(args.start), date.fromisoformat(args.end))
    results: list[dict] = []
    for m in months:
        try:
            results.append(process_month(m))
        except Exception as e:
            log.error("month %s failed: %s", m, e)
            results.append({"month": m, "error": str(e)})
    for r in results:
        log.info("%s", r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
