"""Unit tests for Agent #9 — ML Ranker."""
from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import numpy as np
import polars as pl

from src.common.types import Side, Signal
from src.strategy.specialists import ml_ranker as ml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_bars(sd: date = date(2022, 6, 15), *, seed: int = 1) -> pl.DataFrame:
    """6-hour RTH session of synthetic 1-second OHLCV bars."""
    rng = np.random.default_rng(seed)
    n = 6 * 60 * 60  # shorter than full RTH for test speed
    base = 4200.0
    sigma = 0.08
    theta = 0.012
    x = 0.0
    closes = np.zeros(n)
    for i in range(n):
        x = x - theta * x + rng.normal(0.0, sigma)
        closes[i] = base + x
    highs = closes + rng.uniform(0.05, 0.3, size=n)
    lows = closes - rng.uniform(0.05, 0.3, size=n)
    opens = np.concatenate(([closes[0]], closes[:-1]))
    ts = [datetime.combine(sd, time(14, 30), tzinfo=UTC) + timedelta(seconds=i) for i in range(n)]
    df = pl.DataFrame(
        {
            "ts": ts,
            "session_date": [sd] * n,
            "open": opens.astype(float),
            "high": highs.astype(float),
            "low": lows.astype(float),
            "close": closes.astype(float),
            "volume": rng.integers(20, 200, size=n).astype(int),
        }
    ).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))
    return df


def _mk_signal(ts: datetime, side: Side, entry: float, atr: float = 1.0, *,
               specialist: str = "cvd", setup_id: str = "cvd_divergence_long",
               confidence: float = 0.6) -> Signal:
    sgn = 1.0 if side == Side.LONG else -1.0
    return Signal(
        timestamp=ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts,
        side=side,
        confidence=confidence,
        specialist=specialist,
        setup_id=setup_id,
        entry_price=entry,
        stop_price=entry - sgn * atr,
        target_price=entry + sgn * atr * 1.5,
        metadata={"atr_20": atr},
    )


# ---------------------------------------------------------------------------
# Test 1: MLRankerParams defaults + signature contract
# ---------------------------------------------------------------------------


def test_params_defaults_and_signature():
    p = ml.MLRankerParams()
    assert p.specialist_id == "ml_ranker"
    assert p.forward_window_seconds == 600
    assert p.n_estimators >= 100
    # Module exports the public API expected by the brief.
    for fn_name in ("build_features", "fit_model", "score_signals", "generate_signals"):
        assert hasattr(ml, fn_name), f"missing function {fn_name}"


# ---------------------------------------------------------------------------
# Test 2: build_features produces past-only feature matrix with required cols
# ---------------------------------------------------------------------------


def test_build_features_columns_and_no_future_leak():
    bars = _make_bars()
    # Pick a signal in the middle of the session.
    mid_ts = bars["ts"][1000]
    sig = _mk_signal(mid_ts, Side.LONG, entry=float(bars["close"][1000]))
    feats = ml.build_features([sig], bars)
    assert feats.height == 1
    for c in ml.FEATURE_COLUMNS:
        assert c in feats.columns, f"missing feature column {c}"
    # Mutating the future-half of bars MUST NOT change the past-only features.
    bars_future_mutated = bars.with_columns(
        pl.when(pl.col("ts") > mid_ts).then(pl.col("close") * 100.0).otherwise(pl.col("close")).alias("close"),
        pl.when(pl.col("ts") > mid_ts).then(pl.col("high") * 100.0).otherwise(pl.col("high")).alias("high"),
        pl.when(pl.col("ts") > mid_ts).then(pl.col("low") * 100.0).otherwise(pl.col("low")).alias("low"),
    )
    feats2 = ml.build_features([sig], bars_future_mutated)
    # All feature columns should be identical (future mutation must not bleed back).
    for c in ml.FEATURE_COLUMNS:
        v1 = feats[c][0]
        v2 = feats2[c][0]
        if v1 is None and v2 is None:
            continue
        if isinstance(v1, float) and (np.isnan(v1) and np.isnan(v2)):
            continue
        assert v1 == v2, f"feature {c} leaked future data: {v1} vs {v2}"


# ---------------------------------------------------------------------------
# Test 3: compute_labels — no-lookahead and correctness on a known scenario
# ---------------------------------------------------------------------------


def test_compute_labels_no_lookahead_and_correctness():
    bars = _make_bars(seed=2)
    mid_idx = 1500
    mid_ts = bars["ts"][mid_idx]
    entry = float(bars["close"][mid_idx])
    # Construct a SHORT signal with target ABOVE entry (so cannot be hit from
    # below — must be a stop) — explicitly malformed entry for the label to
    # certainly be 0. (target is above entry => requires LOW going below
    # target, which for SHORT means lo <= target. We flip semantics: stop
    # above entry, target below entry.)
    sig_long = _mk_signal(mid_ts, Side.LONG, entry=entry, atr=0.5)
    # PAST mutation must NOT change labels (labels use only future window).
    params = ml.MLRankerParams(forward_window_seconds=600)
    lbl1 = ml.compute_labels([sig_long], bars, params)
    bars_past_mutated = bars.with_columns(
        pl.when(pl.col("ts") < mid_ts).then(pl.col("close") + 50.0).otherwise(pl.col("close")).alias("close"),
        pl.when(pl.col("ts") < mid_ts).then(pl.col("high") + 50.0).otherwise(pl.col("high")).alias("high"),
        pl.when(pl.col("ts") < mid_ts).then(pl.col("low") + 50.0).otherwise(pl.col("low")).alias("low"),
    )
    lbl2 = ml.compute_labels([sig_long], bars_past_mutated, params)
    assert int(lbl1["y"][0]) == int(lbl2["y"][0]), "label leaked past data"

    # Constructed case: take a moment where price is about to spike up.
    # Find a future bar with a +1 point move within 60s and place a tight LONG.
    closes = bars["close"].to_numpy()
    hi = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    target_idx = None
    for i in range(0, len(closes) - 200):
        if hi[i + 1 : i + 60].max() - closes[i] > 0.6 and lo[i + 1 : i + 60].min() - closes[i] > -0.5:
            target_idx = i
            break
    if target_idx is not None:
        ts_i = bars["ts"][target_idx]
        entry = float(closes[target_idx])
        sig = Signal(
            timestamp=ts_i, side=Side.LONG, confidence=0.5,
            specialist="cvd", setup_id="cvd_divergence_long",
            entry_price=entry,
            stop_price=entry - 0.5,
            target_price=entry + 0.4,
            metadata={},
        )
        lbl = ml.compute_labels([sig], bars, params)
        assert int(lbl["y"][0]) == 1, "expected target-hit label"


# ---------------------------------------------------------------------------
# Test 4: end-to-end fit_model + score_signals on synthetic corpus
# ---------------------------------------------------------------------------


def test_fit_and_score_end_to_end():
    rng = np.random.default_rng(11)
    feats_list = []
    labels_list = []
    sig_ts: list[datetime] = []
    params = ml.MLRankerParams(forward_window_seconds=600, n_estimators=80)
    for d in range(8):
        bars = _make_bars(sd=date(2022, 6, 15) + timedelta(days=d), seed=10 + d)
        # 12 random LONGs + 12 random SHORTs in mid-session
        sigs = []
        n = bars.height
        for _ in range(24):
            i = int(rng.integers(900, n - 900))
            ts_i = bars["ts"][i]
            entry = float(bars["close"][i])
            side = Side.LONG if rng.random() < 0.5 else Side.SHORT
            sigs.append(_mk_signal(ts_i, side, entry, atr=0.5,
                                   specialist=("cvd" if rng.random() < 0.5 else "vwap_ext"),
                                   setup_id=("cvd_divergence_long" if side == Side.LONG else "vwap_pullback_short")))
        f = ml.build_features(sigs, bars).sort("sig_idx")
        lbls = ml.compute_labels(sigs, bars, params).sort("sig_idx")
        feats_list.append(f)
        labels_list.append(lbls.select(["sig_idx", "y"]))
        sig_ts.extend([s.timestamp for s in sigs])

    feats = pl.concat(feats_list).drop("sig_idx")
    labels = pl.concat(labels_list).drop("sig_idx")
    model = ml.fit_model(feats, labels["y"], params)
    # Score the same corpus' signals on the FIRST day's bars (in-sample, just smoke).
    bars0 = _make_bars(sd=date(2022, 6, 15), seed=10)
    sigs0 = [_mk_signal(bars0["ts"][1000 + 60 * k], Side.LONG, float(bars0["close"][1000 + 60 * k]),
                        atr=0.5) for k in range(5)]
    scored = ml.score_signals(sigs0, bars0, model, params=params)
    assert len(scored) == 5
    for s in scored:
        assert 0.0 <= s.confidence <= 1.0
        assert "ml_score" in s.metadata
        assert 0.0 <= s.metadata["ml_score"] <= 1.0


# ---------------------------------------------------------------------------
# Test 5: walk_forward_cv enforces purge / chronology
# ---------------------------------------------------------------------------


def test_walk_forward_purge_and_returns_metrics():
    rng = np.random.default_rng(3)
    feats_list = []
    labels_list = []
    sig_ts: list[datetime] = []
    params = ml.MLRankerParams(forward_window_seconds=120, n_estimators=60,
                               n_walk_forward_splits=3, purge_seconds=120, embargo_seconds=30)
    for d in range(10):
        bars = _make_bars(sd=date(2022, 7, 1) + timedelta(days=d), seed=20 + d)
        n = bars.height
        sigs = []
        for _ in range(30):
            i = int(rng.integers(900, n - 300))
            ts_i = bars["ts"][i]
            entry = float(bars["close"][i])
            side = Side.LONG if rng.random() < 0.5 else Side.SHORT
            sigs.append(_mk_signal(ts_i, side, entry, atr=0.4))
        f = ml.build_features(sigs, bars).sort("sig_idx")
        lbls = ml.compute_labels(sigs, bars, params).sort("sig_idx")
        feats_list.append(f)
        labels_list.append(lbls.select(["sig_idx", "y"]))
        sig_ts.extend([s.timestamp for s in sigs])

    feats = pl.concat(feats_list).drop("sig_idx")
    feats = feats.with_row_index("sig_idx").with_columns(pl.col("sig_idx").cast(pl.Int64))
    labels = pl.concat(labels_list).drop("sig_idx")
    labels = labels.with_row_index("sig_idx").with_columns(pl.col("sig_idx").cast(pl.Int64))
    ts_series = pl.Series("ts", sig_ts).cast(pl.Datetime("ns", "UTC"))
    cv = ml.walk_forward_cv(feats, labels, ts_series, params)
    assert len(cv) >= 1
    for r in cv:
        assert {"fold", "n_train", "n_test", "auc", "baseline_wr", "top30_wr", "lift_pp"} <= set(r.keys())
        # Train and test are non-empty after purging.
        assert r["n_train"] > 0
        assert r["n_test"] > 0
    summary = ml.summarise_cv(cv)
    assert summary["n_folds"] >= 1


# ---------------------------------------------------------------------------
# Test 6: save / load checkpoint round-trips
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip(tmp_path):
    bars = _make_bars()
    sigs = [_mk_signal(bars["ts"][1000 + 60 * k], Side.LONG, float(bars["close"][1000 + 60 * k]), atr=0.5)
            for k in range(20)]
    params = ml.MLRankerParams(n_estimators=60, forward_window_seconds=120)
    feats = ml.build_features(sigs, bars).sort("sig_idx")
    labels = ml.compute_labels(sigs, bars, params).sort("sig_idx").select(["sig_idx", "y"])
    model = ml.fit_model(feats.drop("sig_idx"), labels["y"], params)
    path = tmp_path / "ml_ranker_test.joblib"
    ml.save_checkpoint(model, params, {"mean_auc": 0.6}, path=path)
    payload = ml.load_checkpoint(path)
    assert payload is not None
    assert payload["feature_columns"] == list(ml.FEATURE_COLUMNS)
    scored = ml.score_signals(sigs, bars, payload, params=params)
    assert all("ml_score" in s.metadata for s in scored)


# ---------------------------------------------------------------------------
# Test 7: generate_signals works end-to-end (audit-harness contract)
# ---------------------------------------------------------------------------


def test_generate_signals_audit_contract():
    bars = _make_bars(seed=5)
    params = ml.MLRankerParams(
        score_threshold=0.0,  # accept everything for the smoke test
        top_k_per_day=20,
    )
    # Without a checkpoint on disk, generate_signals should fall back to
    # passing through scores using the original confidence values.
    sigs = ml.generate_signals(bars, params, model=None)
    # generate_signals should return Signal objects sorted by timestamp.
    assert isinstance(sigs, list)
    if sigs:
        ts_list = [s.timestamp for s in sigs]
        assert ts_list == sorted(ts_list)
        for s in sigs:
            assert isinstance(s, Signal)
            assert s.specialist in ("ml_ranker_candidate",)
            # Hard stop must be on the opposite side of entry.
            if s.side == Side.LONG:
                assert s.stop_price < s.entry_price
            else:
                assert s.stop_price > s.entry_price


# ---------------------------------------------------------------------------
# Test 8: build_features tolerates empty signals / empty bars without crashing
# ---------------------------------------------------------------------------


def test_empty_inputs_are_safe():
    bars = _make_bars()
    empty_feats = ml.build_features([], bars)
    assert empty_feats.is_empty()
    empty_labels = ml.compute_labels([], bars, ml.MLRankerParams())
    assert empty_labels.is_empty()
    sig = _mk_signal(bars["ts"][1000], Side.LONG, float(bars["close"][1000]))
    no_bar_feats = ml.build_features([sig], pl.DataFrame())
    assert no_bar_feats.is_empty()
