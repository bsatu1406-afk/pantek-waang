"""Train + checkpoint the ML Ranker on synthetic bars + synthetic upstream signals.

Used during development when real Databento parquet files aren't on disk yet.
The Orchestrator will rerun a full retrain against real upstream signals once
Agents #4 (CVD), #7 (VWAP), #8 (Momentum) have landed.

Usage:
    PYTHONPATH=. python scripts/train_ml_ranker.py [--days 200] [--seed 7] \
        [--out models/ml_ranker.joblib]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from src.common.types import Side, Signal
from src.strategy.specialists import ml_ranker as ml

log = logging.getLogger("train_ml_ranker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _make_session_bars(sd: date, *, rng: np.random.Generator, base_price: float) -> pl.DataFrame:
    """Synthesise one RTH session of ES-like 1-second OHLCV bars.

    Uses a mean-reverting + drift Ornstein-Uhlenbeck process so that hits of
    +/- ATR levels are non-trivial (target_price/stop_price labels have signal).
    """
    n_secs = 6 * 60 * 60 + 30 * 60  # 09:30..16:00 ET
    start_utc = datetime.combine(sd, time(14, 30), tzinfo=UTC)  # ET 09:30 ~ UTC 14:30 (DST-agnostic for synth)
    ts = [start_utc + timedelta(seconds=i) for i in range(n_secs)]
    # OU returns with mean 0, vol ~ 0.02 points/sec, plus intraday drift
    sigma = 0.05
    theta = 0.01
    x = 0.0
    closes = np.zeros(n_secs)
    for i in range(n_secs):
        # daily drift + OU mean-reversion to 0
        x = x - theta * x + rng.normal(0.0, sigma)
        closes[i] = base_price + x + 0.0  # base price is the level
        base_price = base_price + rng.normal(0.0, 0.005)  # tiny random walk on level
    highs = closes + rng.uniform(0.05, 0.4, size=n_secs)
    lows = closes - rng.uniform(0.05, 0.4, size=n_secs)
    opens = np.concatenate(([closes[0]], closes[:-1]))
    volume = rng.integers(20, 200, size=n_secs)
    return pl.DataFrame(
        {
            "ts": ts,
            "session_date": [sd] * n_secs,
            "open": opens.astype(float),
            "high": highs.astype(float),
            "low": lows.astype(float),
            "close": closes.astype(float),
            "volume": volume.astype(int),
        }
    ).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))


_SPECIALISTS = [
    ("cvd", "cvd_divergence_long", Side.LONG),
    ("cvd", "cvd_divergence_short", Side.SHORT),
    ("vwap_ext", "vwap_pullback_short", Side.SHORT),
    ("vwap_ext", "vwap_reversion_long", Side.LONG),
    ("momentum", "momentum_breakout_long", Side.LONG),
    ("momentum", "momentum_breakout_short", Side.SHORT),
]


def _synth_upstream_signals(
    bars: pl.DataFrame,
    *,
    rng: np.random.Generator,
    per_day: int = 20,
) -> list[Signal]:
    """Mock upstream specialists' signals: random pick of bar rows + side.

    The trick to making this trainable: bias the signal generation so that
    each specialist has a slightly different win rate distribution that the
    model can pick up on via the (specialist, setup_id) features and the
    co-occurring rolling features.
    """
    n = bars.height
    if n < 1000:
        return []
    out: list[Signal] = []
    enriched = ml._enrich_bars(bars).to_pandas() if False else ml._enrich_bars(bars)  # noqa
    # Cheap: sample row indices uniformly inside the trading day, but avoid
    # the first 15 min and last 15 min to leave room for fwd window.
    valid_idx = list(range(60 * 15, n - 60 * 15))
    rng.shuffle(valid_idx)
    picks = valid_idx[: per_day]
    cls = bars["close"].to_numpy()
    hi = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    ts_col = bars["ts"].to_list()
    for i in picks:
        spec_name, setup, side = _SPECIALISTS[rng.integers(0, len(_SPECIALISTS))]
        # crude ATR: range over last 20 bars
        j0 = max(0, i - 20)
        atr20 = float(np.mean(hi[j0 : i + 1] - lo[j0 : i + 1])) or 0.5
        entry = float(cls[i])
        sgn = 1.0 if side == Side.LONG else -1.0
        # Each specialist has a slightly different stop/target structure
        rr = {
            "cvd": 1.2,
            "vwap_ext": 1.5,
            "momentum": 1.8,
        }[spec_name] + rng.normal(0.0, 0.2)
        stop = entry - sgn * atr20 * 1.0
        target = entry + sgn * atr20 * max(0.8, rr)
        # Inject a known weak edge:
        # - cvd_divergence_long has ~55% WR by construction
        # - vwap_pullback_short has ~58% WR
        # - momentum_breakout_long has ~52% WR
        # by tweaking confidence accordingly so the model can learn ordering.
        conf = {
            "cvd": 0.55 + rng.normal(0.0, 0.05),
            "vwap_ext": 0.6 + rng.normal(0.0, 0.05),
            "momentum": 0.5 + rng.normal(0.0, 0.07),
        }[spec_name]
        out.append(
            Signal(
                timestamp=ts_col[i],
                side=side,
                confidence=float(np.clip(conf, 0.0, 1.0)),
                specialist=spec_name,
                setup_id=setup,
                entry_price=entry,
                stop_price=float(stop),
                target_price=float(target),
                metadata={"atr_20": atr20, "synthetic": True},
            )
        )
    out.sort(key=lambda s: s.timestamp)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=150)
    p.add_argument("--per-day", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=str(ml.DEFAULT_MODEL_PATH))
    p.add_argument("--cv-only", action="store_true", help="Skip final fit, only run walk-forward CV")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    params = ml.MLRankerParams(model_path=args.out)

    # Build a corpus: many synthetic sessions starting 2020-01 onward (dev period).
    all_feats: list[pl.DataFrame] = []
    all_labels: list[pl.DataFrame] = []
    sig_ts_all: list[datetime] = []
    base = 4000.0
    cur = date(2020, 1, 2)
    sessions_built = 0
    while sessions_built < args.days:
        if cur.weekday() >= 5:
            cur += timedelta(days=1)
            continue
        bars = _make_session_bars(cur, rng=rng, base_price=base)
        sigs = _synth_upstream_signals(bars, rng=rng, per_day=args.per_day)
        if sigs:
            feats = ml.build_features(sigs, bars)
            labels = ml.compute_labels(sigs, bars, params)
            # Sort both by sig_idx
            feats = feats.sort("sig_idx")
            labels = labels.sort("sig_idx")
            all_feats.append(feats)
            all_labels.append(labels.select(["sig_idx", "y"]))
            sig_ts_all.extend([s.timestamp for s in sigs])
            sessions_built += 1
        cur += timedelta(days=1)
        base += rng.normal(0.0, 0.5)
        if cur.year >= 2024:  # keep dev period 2020..2023 only
            break

    if not all_feats:
        log.error("no features produced — aborting")
        return 2

    feats = pl.concat(all_feats, how="vertical_relaxed").drop("sig_idx")
    # Re-index globally
    feats = feats.with_row_index("sig_idx").with_columns(pl.col("sig_idx").cast(pl.Int64))
    labels = pl.concat(all_labels, how="vertical_relaxed").drop("sig_idx")
    labels = labels.with_row_index("sig_idx").with_columns(pl.col("sig_idx").cast(pl.Int64))
    sig_ts_series = pl.Series("ts", sig_ts_all).cast(pl.Datetime("ns", "UTC"))

    log.info("built %d feature rows across %d sessions", feats.height, sessions_built)
    log.info("label class balance: y=1 -> %.1f%%", 100.0 * float(labels["y"].mean()))

    # Walk-forward CV
    cv = ml.walk_forward_cv(feats, labels, sig_ts_series, params)
    summary = ml.summarise_cv(cv)
    log.info("walk-forward CV:")
    for r in cv:
        log.info(
            "  fold %d  n_train=%d n_test=%d  AUC=%.3f  baseline_wr=%.3f top30_wr=%.3f  lift=%+.1fpp",
            r["fold"], r["n_train"], r["n_test"], r["auc"],
            r["baseline_wr"], r["top30_wr"], r["lift_pp"],
        )
    log.info("CV summary: %s", summary)

    if args.cv_only:
        return 0

    # Final fit on the full dev corpus
    model = ml.fit_model(feats.drop("sig_idx"), labels["y"], params)
    out_path = ml.save_checkpoint(model, params, summary, args.out)
    log.info("saved checkpoint -> %s", out_path)
    # Side-car JSON summary for the orchestrator
    sidecar = Path(out_path).with_suffix(".meta.json")
    sidecar.write_text(json.dumps({"cv": cv, "summary": summary,
                                   "n_train_rows": feats.height,
                                   "class_balance": float(labels["y"].mean()),
                                   "feature_columns": list(ml.FEATURE_COLUMNS)}, indent=2, default=str))
    log.info("saved CV sidecar -> %s", sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
