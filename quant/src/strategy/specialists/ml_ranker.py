"""Agent #9 — Machine Learning Ranker (LightGBM).

Scores each upstream signal by its probability of hitting target before stop
within a fixed forward window. Used by the ensemble runner as (a) a filter
(keep top-K per day) and (b) a tiebreaker when signals overlap.

Public API
----------
- ``MLRankerParams``                       — dataclass with hyperparams.
- ``build_features(signals, bars)``        — past-only feature matrix.
- ``fit_model(features, labels, params)``  — trains a LightGBM booster.
- ``score_signals(signals, bars, model)``  — returns Signals with
                                              ``metadata['ml_score']`` set and
                                              ``.confidence`` replaced by the
                                              calibrated score.
- ``generate_signals(bars, params)``       — audit/runner-compatible entrypoint.
                                              Emits a small set of candidate
                                              signals from ``bars`` and scores
                                              them with the persisted booster
                                              (if one exists on disk).
- ``walk_forward_cv(features, labels, params)`` — purged walk-forward AUC.
- ``save_checkpoint`` / ``load_checkpoint`` — joblib persistence.

No-lookahead guarantee
----------------------
Features use only bars with ``ts <= signal.ts``; labels use only bars with
``signal.ts < ts <= signal.ts + forward_window_seconds``. The two passes
share NO state. ``walk_forward_cv`` enforces a purge gap >= forward_window
between train end and test start.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.common.types import Side, Signal
from src.strategy.specialists._interface import SpecialistParams

try:
    import lightgbm as lgb
except ImportError as e:  # pragma: no cover
    raise RuntimeError("lightgbm is required for ml_ranker") from e

try:
    import joblib
except ImportError as e:  # pragma: no cover
    raise RuntimeError("joblib is required for ml_ranker") from e

from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "ml_ranker.joblib"

# Categorical inputs ---------------------------------------------------------

# Known upstream specialist ids — extended whenever a new specialist lands.
_SPECIALIST_VOCAB: tuple[str, ...] = (
    "unknown",
    "regime",
    "cvd",
    "microstructure",
    "volume_profile",
    "vwap_ext",
    "vwap_extended",
    "baseline_short_vwap",
    "momentum",
    "macro_event",
    "ml_ranker_candidate",
)

# Known setup ids; signals carrying setup_id outside the vocab map to "other".
_SETUP_VOCAB: tuple[str, ...] = (
    "other",
    "vwap_pullback_short",
    "vwap_pullback_long",
    "vwap_reversion_long",
    "vwap_reversion_short",
    "cvd_divergence_long",
    "cvd_divergence_short",
    "momentum_breakout_long",
    "momentum_breakout_short",
    "ml_ranker_vwap_long",
    "ml_ranker_vwap_short",
    "ml_ranker_momo_long",
    "ml_ranker_momo_short",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "side_long",
    "hour_of_session",
    "minute_of_session",
    "rv_60s",
    "rv_300s",
    "rv_900s",
    "atr_5",
    "atr_20",
    "atr_ratio",
    "vol_zscore_60s",
    "ret_60s",
    "ret_300s",
    "cum_ret_session",
    "vwap_dist_atr",
    "cvd_slope_60s",
    "cvd_slope_300s",
    "stop_distance_atr",
    "target_distance_atr",
    "rr_ratio",
    "specialist_code",
    "setup_code",
    "confidence_in",
)

_CATEGORICAL_COLUMNS: tuple[str, ...] = ("specialist_code", "setup_code")


# Params ---------------------------------------------------------------------


@dataclass
class MLRankerParams(SpecialistParams):
    """Hyperparameters for the ML ranker."""

    specialist_id: str = "ml_ranker"
    forward_window_seconds: int = 600
    target_atr_mult: float = 1.0
    stop_atr_mult: float = 1.0
    atr_window: int = 20
    train_months: int = 9
    test_months: int = 3
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_data_in_leaf: int = 20
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    random_state: int = 42

    # Walk-forward / CV
    n_walk_forward_splits: int = 4
    purge_seconds: int = 600
    embargo_seconds: int = 60

    # generate_signals candidate emission
    candidate_cooldown_seconds: int = 120
    candidate_vwap_band_sigma: float = 1.5
    candidate_momo_lookback: int = 60
    candidate_max_per_day: int = 40
    score_threshold: float = 0.50
    top_k_per_day: int = 8
    confidence_blend_alpha: float = 0.7  # final_conf = a*ml + (1-a)*orig_conf

    # Persistence
    model_path: str = str(DEFAULT_MODEL_PATH)
    metadata: dict = field(default_factory=dict)


# Helpers --------------------------------------------------------------------


def _code(value: str, vocab: tuple[str, ...]) -> int:
    """Map a string to its index in ``vocab``; unknown → index 0."""
    try:
        return vocab.index(value)
    except ValueError:
        return 0


def _signals_to_df(signals: list[Signal]) -> pl.DataFrame:
    if not signals:
        return pl.DataFrame(
            schema={
                "sig_idx": pl.Int64,
                "ts": pl.Datetime("ns", "UTC"),
                "side": pl.Utf8,
                "confidence_in": pl.Float64,
                "specialist": pl.Utf8,
                "setup_id": pl.Utf8,
                "entry_price": pl.Float64,
                "stop_price": pl.Float64,
                "target_price": pl.Float64,
            }
        )
    rows = []
    for i, s in enumerate(signals):
        ts = s.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        rows.append(
            {
                "sig_idx": i,
                "ts": ts,
                "side": s.side.value,
                "confidence_in": float(s.confidence),
                "specialist": str(s.specialist),
                "setup_id": str(s.setup_id),
                "entry_price": float(s.entry_price),
                "stop_price": float(s.stop_price),
                "target_price": float(s.target_price)
                if s.target_price is not None
                else float("nan"),
            }
        )
    df = pl.DataFrame(rows)
    # Coerce ts to ns precision UTC
    df = df.with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))
    return df.sort("ts")


def _enrich_bars(bars: pl.DataFrame) -> pl.DataFrame:
    """Compute past-only rolling indicators on bars.

    Every column produced here is causal (uses only rows up to and including
    the current row).
    """
    if bars.is_empty():
        return bars
    df = bars.sort("ts")
    # Ensure required columns exist
    needed = {"ts", "open", "high", "low", "close", "volume"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"bars missing columns: {missing}")

    df = df.with_columns(
        [
            pl.col("close").pct_change().alias("ret_1s"),
            (pl.col("high") - pl.col("low")).alias("range_1s"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("ret_1s").rolling_std(window_size=60, min_samples=10).alias("rv_60s"),
            pl.col("ret_1s").rolling_std(window_size=300, min_samples=30).alias("rv_300s"),
            pl.col("ret_1s").rolling_std(window_size=900, min_samples=60).alias("rv_900s"),
            pl.col("range_1s").rolling_mean(window_size=5, min_samples=2).alias("atr_5"),
            pl.col("range_1s").rolling_mean(window_size=20, min_samples=5).alias("atr_20"),
            pl.col("volume").rolling_mean(window_size=60, min_samples=10).alias("vol_ma_60"),
            pl.col("volume").rolling_std(window_size=60, min_samples=10).alias("vol_std_60"),
            pl.col("close").rolling_mean(window_size=60, min_samples=10).alias("close_ma_60"),
            (pl.col("close") - pl.col("close").shift(60)).alias("ret_60s"),
            (pl.col("close") - pl.col("close").shift(300)).alias("ret_300s"),
        ]
    )
    df = df.with_columns(
        [
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("typical"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("typical") * pl.col("volume")).cum_sum().over("session_date").alias("ctpv")
            if "session_date" in df.columns
            else (pl.col("typical") * pl.col("volume")).cum_sum().alias("ctpv"),
            pl.col("volume").cum_sum().over("session_date").alias("cv")
            if "session_date" in df.columns
            else pl.col("volume").cum_sum().alias("cv"),
            pl.col("close").first().over("session_date").alias("session_open")
            if "session_date" in df.columns
            else pl.col("close").first().alias("session_open"),
            pl.col("ts").first().over("session_date").alias("session_start_ts")
            if "session_date" in df.columns
            else pl.col("ts").first().alias("session_start_ts"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("ctpv") / pl.col("cv").clip(lower_bound=1)).alias("vwap"),
            ((pl.col("ts") - pl.col("session_start_ts")).dt.total_seconds() / 60.0)
            .alias("minutes_since_open"),
            (pl.col("close") / pl.col("session_open") - 1.0).alias("cum_ret_session"),
        ]
    )
    df = df.with_columns(
        [
            ((pl.col("close") - pl.col("vwap")) / pl.col("atr_20").fill_null(1.0).clip(lower_bound=0.1))
            .alias("vwap_dist_atr"),
            ((pl.col("volume") - pl.col("vol_ma_60")) / pl.col("vol_std_60").fill_null(1.0).clip(lower_bound=0.1))
            .alias("vol_zscore_60s"),
        ]
    )
    return df


def _attach_cvd_features(bars_enriched: pl.DataFrame, cvd: pl.DataFrame | None) -> pl.DataFrame:
    """Attach CVD slopes via join_asof if a CVD frame is available."""
    if cvd is None or cvd.is_empty():
        return bars_enriched.with_columns(
            [
                pl.lit(0.0).alias("cvd_slope_60s"),
                pl.lit(0.0).alias("cvd_slope_300s"),
            ]
        )
    c = cvd.sort("ts").with_columns(
        [
            (pl.col("cvd") - pl.col("cvd").shift(60)).alias("cvd_slope_60s"),
            (pl.col("cvd") - pl.col("cvd").shift(300)).alias("cvd_slope_300s"),
        ]
    ).select(["ts", "cvd_slope_60s", "cvd_slope_300s"])
    return bars_enriched.sort("ts").join_asof(c, on="ts", strategy="backward")


# Feature build --------------------------------------------------------------


def build_features(
    signals: list[Signal],
    bars: pl.DataFrame,
    *,
    cvd: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Past-only feature matrix indexed by ``(sig_idx, ts, specialist, setup_id)``.

    Parameters
    ----------
    signals
        Upstream signals to score.
    bars
        Contiguous 1-second OHLCV bars (one or many session days).
    cvd
        Optional per-second CVD frame (columns: ``ts, cvd``). When absent,
        CVD slope features default to 0.
    """
    sig_df = _signals_to_df(signals)
    if sig_df.is_empty() or bars.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in FEATURE_COLUMNS} | {"sig_idx": pl.Int64})

    enriched = _enrich_bars(bars)
    enriched = _attach_cvd_features(enriched, cvd)
    # Past-only as-of join: pull the most recent bar row at or before each signal's ts.
    keep = [
        "ts",
        "rv_60s",
        "rv_300s",
        "rv_900s",
        "atr_5",
        "atr_20",
        "ret_60s",
        "ret_300s",
        "cum_ret_session",
        "vwap_dist_atr",
        "vol_zscore_60s",
        "cvd_slope_60s",
        "cvd_slope_300s",
        "minutes_since_open",
    ]
    feats = sig_df.sort("ts").join_asof(
        enriched.select(keep).sort("ts"),
        on="ts",
        strategy="backward",
    )
    feats = feats.with_columns(
        [
            (pl.col("side") == "long").cast(pl.Int8).alias("side_long"),
            (pl.col("minutes_since_open") / 60.0).floor().alias("hour_of_session"),
            (pl.col("minutes_since_open") % 60.0).alias("minute_of_session"),
            pl.col("atr_5").alias("atr_5"),
            pl.col("atr_20").alias("atr_20"),
            (pl.col("atr_5") / pl.col("atr_20").clip(lower_bound=0.05)).alias("atr_ratio"),
            (
                (pl.col("entry_price") - pl.col("stop_price")).abs()
                / pl.col("atr_20").clip(lower_bound=0.05)
            ).alias("stop_distance_atr"),
            (
                (pl.col("target_price") - pl.col("entry_price")).abs()
                / pl.col("atr_20").clip(lower_bound=0.05)
            ).alias("target_distance_atr"),
            (
                (pl.col("target_price") - pl.col("entry_price")).abs()
                / (pl.col("entry_price") - pl.col("stop_price")).abs().clip(lower_bound=1e-6)
            ).alias("rr_ratio"),
            pl.col("specialist")
            .map_elements(lambda v: _code(v, _SPECIALIST_VOCAB), return_dtype=pl.Int32)
            .alias("specialist_code"),
            pl.col("setup_id")
            .map_elements(lambda v: _code(v, _SETUP_VOCAB), return_dtype=pl.Int32)
            .alias("setup_code"),
        ]
    )
    # Drop unused
    out_cols = ["sig_idx", *FEATURE_COLUMNS]
    return feats.select(out_cols)


# Labels ---------------------------------------------------------------------


def compute_labels(
    signals: list[Signal],
    bars: pl.DataFrame,
    params: MLRankerParams,
) -> pl.DataFrame:
    """Forward-only label generation.

    For each signal, label = 1 iff ``target_price`` is hit (in the side's
    favourable direction) strictly BEFORE ``stop_price`` within
    ``forward_window_seconds`` after the signal timestamp.

    Returns a DataFrame with columns ``sig_idx, y, hit_reason, hit_ts``.
    Uses only bars with ``ts > signal.ts``.
    """
    if not signals or bars.is_empty():
        return pl.DataFrame(
            schema={
                "sig_idx": pl.Int64,
                "y": pl.Int8,
                "hit_reason": pl.Utf8,
                "hit_ts": pl.Datetime("ns", "UTC"),
            }
        )
    sig_df = _signals_to_df(signals).with_columns(
        (pl.col("ts") + pl.duration(seconds=params.forward_window_seconds)).alias("forward_end")
    )
    # Resolve target_price NaN -> entry +/- target_atr_mult * atr_20 (computed on past bars).
    enriched = _enrich_bars(bars).select(["ts", "atr_20"]).sort("ts")
    sig_df = sig_df.sort("ts").join_asof(enriched, on="ts", strategy="backward")
    sig_df = sig_df.with_columns(
        pl.when(pl.col("target_price").is_nan() | pl.col("target_price").is_null())
        .then(
            pl.when(pl.col("side") == "long")
            .then(pl.col("entry_price") + params.target_atr_mult * pl.col("atr_20").fill_null(0.5))
            .otherwise(pl.col("entry_price") - params.target_atr_mult * pl.col("atr_20").fill_null(0.5))
        )
        .otherwise(pl.col("target_price"))
        .alias("target_price_eff")
    )
    bars_simple = bars.sort("ts").select(["ts", "high", "low"])

    joined = sig_df.join_where(
        bars_simple,
        pl.col("ts_right") > pl.col("ts"),
        pl.col("ts_right") <= pl.col("forward_end"),
    )
    if joined.is_empty():
        return pl.DataFrame(
            {
                "sig_idx": sig_df["sig_idx"],
                "y": pl.zeros(sig_df.height, dtype=pl.Int8, eager=True),
                "hit_reason": ["no_data"] * sig_df.height,
                "hit_ts": [None] * sig_df.height,
            }
        )
    joined = joined.with_columns(
        [
            (
                pl.when(pl.col("side") == "long")
                .then(pl.col("low") <= pl.col("stop_price"))
                .otherwise(pl.col("high") >= pl.col("stop_price"))
            ).alias("stop_hit"),
            (
                pl.when(pl.col("side") == "long")
                .then(pl.col("high") >= pl.col("target_price_eff"))
                .otherwise(pl.col("low") <= pl.col("target_price_eff"))
            ).alias("tp_hit"),
        ]
    )
    firsts = joined.group_by("sig_idx").agg(
        [
            pl.col("ts_right").filter(pl.col("stop_hit")).min().alias("stop_ts"),
            pl.col("ts_right").filter(pl.col("tp_hit")).min().alias("tp_ts"),
        ]
    )
    firsts = firsts.with_columns(
        pl.when(pl.col("tp_ts").is_null() & pl.col("stop_ts").is_null())
        .then(pl.lit("none"))
        .when(pl.col("tp_ts").is_null())
        .then(pl.lit("stop"))
        .when(pl.col("stop_ts").is_null())
        .then(pl.lit("tp"))
        .when(pl.col("tp_ts") <= pl.col("stop_ts"))
        .then(pl.lit("tp"))
        .otherwise(pl.lit("stop"))
        .alias("hit_reason"),
    )
    firsts = firsts.with_columns(
        [
            (pl.col("hit_reason") == "tp").cast(pl.Int8).alias("y"),
            pl.when(pl.col("hit_reason") == "tp")
            .then(pl.col("tp_ts"))
            .when(pl.col("hit_reason") == "stop")
            .then(pl.col("stop_ts"))
            .otherwise(None)
            .alias("hit_ts"),
        ]
    )
    # Fill missing sig_idx (signals with no joined rows) as y=0/hit_reason="none"
    all_idx = sig_df.select(["sig_idx"])
    out = all_idx.join(
        firsts.select(["sig_idx", "y", "hit_reason", "hit_ts"]),
        on="sig_idx",
        how="left",
    ).with_columns(
        [
            pl.col("y").fill_null(0).cast(pl.Int8),
            pl.col("hit_reason").fill_null("none"),
        ]
    )
    return out


# Model fitting --------------------------------------------------------------


def fit_model(
    features: pl.DataFrame,
    labels: pl.Series | pl.DataFrame | np.ndarray,
    params: MLRankerParams | None = None,
) -> lgb.LGBMClassifier:
    """Fit a LightGBM binary classifier."""
    if params is None:
        params = MLRankerParams()
    if isinstance(labels, pl.DataFrame):
        labels = labels["y"].to_numpy()
    elif isinstance(labels, pl.Series):
        labels = labels.to_numpy()
    X = features.select(list(FEATURE_COLUMNS)).fill_nan(0.0).fill_null(0.0).to_numpy()
    y = np.asarray(labels).astype(int)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"feature/label length mismatch: {X.shape[0]} vs {y.shape[0]}")
    if len(np.unique(y)) < 2:
        # Degenerate: still fit (returns a trivial model), but warn.
        log.warning("fit_model: only one class present in labels; model will be trivial")
    model = lgb.LGBMClassifier(
        n_estimators=params.n_estimators,
        learning_rate=params.learning_rate,
        num_leaves=params.num_leaves,
        max_depth=params.max_depth,
        min_data_in_leaf=params.min_data_in_leaf,
        feature_fraction=params.feature_fraction,
        bagging_fraction=params.bagging_fraction,
        bagging_freq=params.bagging_freq,
        objective="binary",
        random_state=params.random_state,
        n_jobs=-1,
        verbose=-1,
    )
    cat_idx = [FEATURE_COLUMNS.index(c) for c in _CATEGORICAL_COLUMNS]
    model.fit(X, y, feature_name=list(FEATURE_COLUMNS), categorical_feature=cat_idx)
    return model


# Walk-forward CV ------------------------------------------------------------


def walk_forward_cv(
    features: pl.DataFrame,
    labels: pl.DataFrame,
    signal_ts: pl.Series,
    params: MLRankerParams,
) -> list[dict[str, Any]]:
    """Purged walk-forward CV.

    Splits the sample into ``n_walk_forward_splits`` chronological folds.
    For each fold:
      - test window = next chunk after train window
      - PURGE: drop training rows whose forward_end overlaps test start
      - EMBARGO: gap of ``embargo_seconds`` between train_end and test_start
    Returns one dict per fold with AUC and lift over baseline.
    """
    n = features.height
    if n < 50:
        return []
    # Sort by ts
    ts_np = signal_ts.cast(pl.Datetime("ns", "UTC")).cast(pl.Int64).to_numpy()
    order = np.argsort(ts_np)
    ts_sorted = ts_np[order]
    y_full = labels["y"].to_numpy()[order]
    X_full = (
        features.select(list(FEATURE_COLUMNS))
        .fill_nan(0.0)
        .fill_null(0.0)
        .to_numpy()
    )[order]

    purge_ns = int((params.purge_seconds + params.embargo_seconds) * 1e9)
    n_splits = max(2, params.n_walk_forward_splits)
    # Build chronological fold edges (equal-count splits)
    fold_size = n // (n_splits + 1)
    if fold_size < 20:
        n_splits = max(2, n // 60)
        fold_size = max(20, n // (n_splits + 1))
    results: list[dict[str, Any]] = []
    cat_idx = [FEATURE_COLUMNS.index(c) for c in _CATEGORICAL_COLUMNS]
    for k in range(1, n_splits + 1):
        train_end_i = k * fold_size
        test_start_i = train_end_i
        test_end_i = min(n, test_start_i + fold_size)
        if test_end_i - test_start_i < 10 or train_end_i < 30:
            continue
        test_start_ts = ts_sorted[test_start_i]
        # Purge: keep training rows whose ts + purge_ns < test_start_ts
        train_mask = ts_sorted[:train_end_i] + purge_ns < test_start_ts
        X_tr = X_full[:train_end_i][train_mask]
        y_tr = y_full[:train_end_i][train_mask]
        X_te = X_full[test_start_i:test_end_i]
        y_te = y_full[test_start_i:test_end_i]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            results.append(
                {"fold": k, "n_train": len(y_tr), "n_test": len(y_te), "auc": float("nan"),
                 "baseline_wr": float(np.mean(y_te)) if len(y_te) else 0.0,
                 "top30_wr": float("nan"), "lift_pp": float("nan")}
            )
            continue
        model = lgb.LGBMClassifier(
            n_estimators=params.n_estimators,
            learning_rate=params.learning_rate,
            num_leaves=params.num_leaves,
            max_depth=params.max_depth,
            min_data_in_leaf=params.min_data_in_leaf,
            feature_fraction=params.feature_fraction,
            bagging_fraction=params.bagging_fraction,
            bagging_freq=params.bagging_freq,
            objective="binary",
            random_state=params.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_tr,
            y_tr,
            feature_name=list(FEATURE_COLUMNS),
            categorical_feature=cat_idx,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
            )
            proba = model.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_te, proba))
        baseline_wr = float(np.mean(y_te))
        # Top-30% by score
        cutoff = np.quantile(proba, 0.70)
        top_mask = proba >= cutoff
        top_wr = float(np.mean(y_te[top_mask])) if top_mask.any() else float("nan")
        lift_pp = (top_wr - baseline_wr) * 100 if np.isfinite(top_wr) else float("nan")
        results.append(
            {
                "fold": k,
                "n_train": len(y_tr),
                "n_test": len(y_te),
                "auc": auc,
                "baseline_wr": baseline_wr,
                "top30_wr": top_wr,
                "lift_pp": lift_pp,
            }
        )
    return results


def summarise_cv(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {"n_folds": 0, "mean_auc": float("nan"), "mean_lift_pp": float("nan")}
    aucs = [r["auc"] for r in results if np.isfinite(r["auc"])]
    lifts = [r["lift_pp"] for r in results if np.isfinite(r["lift_pp"])]
    return {
        "n_folds": len(results),
        "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "mean_lift_pp": float(np.mean(lifts)) if lifts else float("nan"),
        "min_auc": float(np.min(aucs)) if aucs else float("nan"),
        "max_auc": float(np.max(aucs)) if aucs else float("nan"),
    }


# Persistence ----------------------------------------------------------------


def save_checkpoint(
    model: lgb.LGBMClassifier,
    params: MLRankerParams,
    cv_summary: dict[str, float] | None = None,
    path: str | Path | None = None,
) -> Path:
    """Persist booster + feature columns + params to joblib."""
    p = Path(path or params.model_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "categorical_columns": list(_CATEGORICAL_COLUMNS),
        "specialist_vocab": list(_SPECIALIST_VOCAB),
        "setup_vocab": list(_SETUP_VOCAB),
        "params": asdict(params),
        "cv_summary": cv_summary or {},
    }
    joblib.dump(payload, p)
    return p


def load_checkpoint(path: str | Path | None = None) -> dict[str, Any] | None:
    p = Path(path or DEFAULT_MODEL_PATH)
    if not p.exists():
        return None
    return joblib.load(p)


# Scoring --------------------------------------------------------------------


def score_signals(
    signals: list[Signal],
    bars: pl.DataFrame,
    model: lgb.LGBMClassifier | dict[str, Any] | None,
    *,
    cvd: pl.DataFrame | None = None,
    params: MLRankerParams | None = None,
) -> list[Signal]:
    """Return new Signal objects with ``metadata['ml_score']`` and re-rated confidence.

    Parameters
    ----------
    signals
        Upstream signals (won't be mutated).
    bars
        Bars covering all signals' timestamps (one session day or many).
    model
        Either an LGBM classifier, a loaded checkpoint dict, or None (fallback
        to ``confidence_in``).
    cvd
        Optional per-second CVD frame.
    """
    if params is None:
        params = MLRankerParams()
    if not signals:
        return []
    feats = build_features(signals, bars, cvd=cvd)
    if feats.is_empty():
        return list(signals)
    X = feats.select(list(FEATURE_COLUMNS)).fill_nan(0.0).fill_null(0.0).to_numpy()
    sig_idx = feats["sig_idx"].to_list()

    if isinstance(model, dict):
        clf = model.get("model")
    else:
        clf = model
    if clf is None:
        scores = {i: float(s.confidence) for i, s in enumerate(signals)}
    else:
        # Suppress benign sklearn feature-name warning when X is numpy.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
            )
            proba = clf.predict_proba(X)[:, 1]
        scores = {int(i): float(p) for i, p in zip(sig_idx, proba, strict=False)}

    out: list[Signal] = []
    a = float(params.confidence_blend_alpha)
    for i, s in enumerate(signals):
        ml = scores.get(i, float(s.confidence))
        # Calibrated blend
        new_conf = max(0.0, min(1.0, a * ml + (1.0 - a) * float(s.confidence)))
        meta = dict(s.metadata)
        meta["ml_score"] = ml
        meta["orig_confidence"] = float(s.confidence)
        out.append(
            Signal(
                timestamp=s.timestamp,
                side=s.side,
                confidence=new_conf,
                specialist=s.specialist,
                setup_id=s.setup_id,
                entry_price=s.entry_price,
                stop_price=s.stop_price,
                target_price=s.target_price,
                metadata=meta,
            )
        )
    return out


# Candidate-signal emission (for the standalone audit harness) ---------------


def _emit_candidate_signals(
    bars: pl.DataFrame,
    params: MLRankerParams,
) -> list[Signal]:
    """Emit a small, rule-based set of candidate signals to score.

    The ML Ranker normally consumes upstream specialists' output. During the
    standalone audit (and when no upstream is wired) we synthesise candidates
    from simple VWAP-extension and momentum-breakout rules so the harness has
    something to evaluate end-to-end.
    """
    if bars.is_empty():
        return []
    df = _enrich_bars(bars)
    df = df.with_columns(
        [
            pl.col("close").rolling_max(window_size=params.candidate_momo_lookback, min_samples=10)
            .shift(1)
            .alias("momo_hi"),
            pl.col("close").rolling_min(window_size=params.candidate_momo_lookback, min_samples=10)
            .shift(1)
            .alias("momo_lo"),
            pl.col("close").rolling_std(window_size=300, min_samples=30).alias("close_std_300"),
        ]
    )
    df = df.with_columns(
        [
            (
                pl.col("close")
                > pl.col("vwap") + params.candidate_vwap_band_sigma * pl.col("close_std_300")
            ).alias("vwap_short"),
            (
                pl.col("close")
                < pl.col("vwap") - params.candidate_vwap_band_sigma * pl.col("close_std_300")
            ).alias("vwap_long"),
            (pl.col("close") > pl.col("momo_hi")).alias("momo_long"),
            (pl.col("close") < pl.col("momo_lo")).alias("momo_short"),
        ]
    )
    rows = df.iter_rows(named=True)
    out: list[Signal] = []
    last_emit_ts: datetime | None = None
    cooldown = timedelta(seconds=params.candidate_cooldown_seconds)
    for row in rows:
        if row.get("minutes_since_open") is None or row["minutes_since_open"] < 15:
            continue
        if row.get("atr_20") is None or not np.isfinite(row["atr_20"]) or row["atr_20"] <= 0:
            continue
        if last_emit_ts is not None and row["ts"] - last_emit_ts < cooldown:
            continue
        side: Side | None = None
        setup_id = "other"
        if row.get("vwap_short"):
            side, setup_id = Side.SHORT, "ml_ranker_vwap_short"
        elif row.get("vwap_long"):
            side, setup_id = Side.LONG, "ml_ranker_vwap_long"
        elif row.get("momo_long"):
            side, setup_id = Side.LONG, "ml_ranker_momo_long"
        elif row.get("momo_short"):
            side, setup_id = Side.SHORT, "ml_ranker_momo_short"
        if side is None:
            continue
        atr = float(row["atr_20"])
        entry = float(row["close"])
        sgn = 1.0 if side == Side.LONG else -1.0
        stop = entry - sgn * atr * params.stop_atr_mult
        target = entry + sgn * atr * params.target_atr_mult
        # Hard-stop sanity: must be on the opposite side of entry
        if (side == Side.LONG and stop >= entry) or (side == Side.SHORT and stop <= entry):
            continue
        out.append(
            Signal(
                timestamp=row["ts"],
                side=side,
                confidence=0.5,
                specialist="ml_ranker_candidate",
                setup_id=setup_id,
                entry_price=entry,
                stop_price=stop,
                target_price=target,
                metadata={"atr_20": atr, "vwap_dist_atr": float(row.get("vwap_dist_atr") or 0.0)},
            )
        )
        last_emit_ts = row["ts"]
        if len(out) >= params.candidate_max_per_day:
            break
    return out


def generate_signals(
    bars: pl.DataFrame,
    params: MLRankerParams,
    *,
    upstream_signals: list[Signal] | None = None,
    cvd: pl.DataFrame | None = None,
    model: lgb.LGBMClassifier | dict[str, Any] | None = None,
) -> list[Signal]:
    """Audit-compatible entrypoint.

    If ``upstream_signals`` is provided, score them. Otherwise, synthesise
    candidate signals from ``bars`` and score those (the standalone audit
    path). Returns up to ``top_k_per_day`` signals above ``score_threshold``,
    ordered by signal timestamp.
    """
    if params is None:
        params = MLRankerParams()
    candidates = upstream_signals or _emit_candidate_signals(bars, params)
    if not candidates:
        return []
    ckpt = model if model is not None else load_checkpoint(params.model_path)
    scored = score_signals(candidates, bars, ckpt, cvd=cvd, params=params)
    # Filter by score threshold, then keep top-K per session_date
    filtered = [s for s in scored if s.metadata.get("ml_score", s.confidence) >= params.score_threshold]
    if not filtered:
        return []
    # Group by session date (UTC date here is fine; harness iterates per session day).
    by_day: dict[date, list[Signal]] = {}
    for s in filtered:
        d = s.timestamp.date()
        by_day.setdefault(d, []).append(s)
    out: list[Signal] = []
    for _d, ss in by_day.items():
        ss.sort(key=lambda x: x.metadata.get("ml_score", x.confidence), reverse=True)
        out.extend(ss[: params.top_k_per_day])
    out.sort(key=lambda x: x.timestamp)
    return out


__all__ = [
    "FEATURE_COLUMNS",
    "MLRankerParams",
    "build_features",
    "compute_labels",
    "fit_model",
    "generate_signals",
    "load_checkpoint",
    "save_checkpoint",
    "score_signals",
    "summarise_cv",
    "walk_forward_cv",
]
