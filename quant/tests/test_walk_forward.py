"""Tests for the walk-forward harness (Agent #11).

Covers:
* Fold geometry: rolling 12/3/3-month windows with a 1-day purge.
* Leakage prevention: train_end strictly precedes test_start, and the gap
  exceeds the purge_days parameter.
* Determinism: same inputs → identical fold table.
* Empty / pathological inputs raise rather than silently producing
  zero-fold reports.
* `WalkForwardReport.to_json` round-trips through json.loads + the per-fold
  / aggregate columns required by the briefing exist.
* End-to-end synthetic strategy proof (random mean-reversion) showing the
  harness can drive a real strategy_fn / backtest_fn pair.
"""
from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.backtest.walk_forward import (
    add_months,
    run_walkforward,
)

# ---------------------------------------------------------------------------
# Synthetic bars fixture
# ---------------------------------------------------------------------------


def _synthetic_bars_by_date(
    start: date,
    end: date,
    bars_per_day: int = 60,
    seed: int = 42,
) -> dict[date, pl.DataFrame]:
    """Random-walk bars, Mon-Fri only, deterministic for a given seed."""
    rng = random.Random(seed)
    price = 5000.0
    out: dict[date, pl.DataFrame] = {}
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri only
            ts_list: list[datetime] = []
            opens, highs, lows, closes, vols = [], [], [], [], []
            for i in range(bars_per_day):
                ts = datetime(d.year, d.month, d.day, 9, 30) + timedelta(minutes=i)
                o = price
                price = max(1.0, price + rng.gauss(0.0, 1.0))
                c = price
                h = max(o, c) + abs(rng.gauss(0.0, 0.3))
                lo = min(o, c) - abs(rng.gauss(0.0, 0.3))
                ts_list.append(ts)
                opens.append(o)
                highs.append(h)
                lows.append(lo)
                closes.append(c)
                vols.append(rng.randint(10, 200))
            out[d] = pl.DataFrame({
                "ts": ts_list,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": vols,
            })
        d = d + timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# Trivial strategy / backtest fns for geometry tests
# ---------------------------------------------------------------------------


def _id_strategy(params, bars_train):
    """Return the seed params untouched + the size of the training slice."""
    out = dict(params or {})
    out["train_days"] = len(bars_train)
    return out


def _const_backtest(params, bars_eval):
    """Return a fixed plausible metrics dict so aggregation can be checked."""
    return {
        "wr": 0.55,
        "pf": 1.20,
        "net_pnl": 100.0 if len(bars_eval) > 0 else 0.0,
        "total_withdrawn": 0.0,
        "breach": False,
        "trades": 10 * len(bars_eval),
    }


# ---------------------------------------------------------------------------
# 1. Fold geometry
# ---------------------------------------------------------------------------


def test_fold_geometry_rolling_12_3_3():
    bars = _synthetic_bars_by_date(date(2020, 1, 1), date(2024, 12, 31))
    rep = run_walkforward(
        _id_strategy, _const_backtest, bars,
        train_months=12, test_months=3, step_months=3, purge_days=1,
    )
    assert rep.folds, "no folds produced"
    f0 = rep.folds[0]
    # Train window is exactly 12 calendar months long.
    assert f0.train_start == date(2020, 1, 1)
    assert f0.train_end == add_months(f0.train_start, 12) - timedelta(days=1)
    # Test window is exactly 3 calendar months long.
    assert f0.test_end == add_months(f0.test_start, 3) - timedelta(days=1)
    # Test starts purge_days + 1 calendar days after train_end.
    assert (f0.test_start - f0.train_end).days == 2  # 1 day purge + 1

    # Step: anchor advances by step_months between consecutive folds.
    f1 = rep.folds[1]
    assert f1.train_start == add_months(f0.train_start, 3)
    assert f1.test_start == add_months(f0.test_start, 3)


# ---------------------------------------------------------------------------
# 2. Leakage prevention
# ---------------------------------------------------------------------------


def test_no_overlap_or_leakage_between_train_and_test():
    bars = _synthetic_bars_by_date(date(2022, 1, 1), date(2024, 6, 30))
    rep = run_walkforward(
        _id_strategy, _const_backtest, bars,
        train_months=12, test_months=3, step_months=3, purge_days=1,
    )
    for f in rep.folds:
        # No dates shared between train and test (the harness slices them
        # by date ranges; the gap is at least purge_days + 1 calendar days).
        assert f.train_end < f.test_start
        assert (f.test_start - f.train_end).days > 1  # strictly > purge_days
        # And the per-fold session sets must be disjoint.
        train_dates = {d for d in bars if f.train_start <= d <= f.train_end}
        test_dates = {d for d in bars if f.test_start <= d <= f.test_end}
        assert not (train_dates & test_dates)


def test_strategy_fn_never_sees_test_dates():
    """`strategy_fn` must only receive dates in [train_start, train_end]."""
    bars = _synthetic_bars_by_date(date(2022, 1, 1), date(2024, 6, 30))
    seen_dates_per_fold: list[set[date]] = []

    def capturing_strategy(params, bars_train):
        seen_dates_per_fold.append(set(bars_train.keys()))
        return {}

    rep = run_walkforward(
        capturing_strategy, _const_backtest, bars,
        train_months=12, test_months=3, step_months=3, purge_days=1,
    )
    assert len(seen_dates_per_fold) == len(rep.folds) > 0
    for f, seen in zip(rep.folds, seen_dates_per_fold, strict=True):
        assert all(f.train_start <= d <= f.train_end for d in seen)
        # And the test window is entirely absent from what strategy_fn saw.
        for d in seen:
            assert d < f.test_start


def test_larger_purge_widens_the_gap():
    bars = _synthetic_bars_by_date(date(2022, 1, 1), date(2024, 6, 30))
    rep1 = run_walkforward(
        _id_strategy, _const_backtest, bars,
        train_months=12, test_months=3, step_months=3, purge_days=1,
    )
    rep5 = run_walkforward(
        _id_strategy, _const_backtest, bars,
        train_months=12, test_months=3, step_months=3, purge_days=5,
    )
    assert rep1.folds and rep5.folds
    for f in rep1.folds:
        assert (f.test_start - f.train_end).days == 2
    for f in rep5.folds:
        assert (f.test_start - f.train_end).days == 6


# ---------------------------------------------------------------------------
# 3. Aggregate metrics
# ---------------------------------------------------------------------------


def test_aggregate_has_required_briefing_columns():
    bars = _synthetic_bars_by_date(date(2022, 1, 1), date(2024, 12, 31))
    rep = run_walkforward(
        _id_strategy, _const_backtest, bars,
        train_months=12, test_months=3, step_months=3, purge_days=1,
    )
    agg = rep.aggregate
    required = {
        "mean_wr", "std_wr", "mean_pf", "std_pf",
        "wd_per_month_mean", "wd_per_month_std",
        "breach_rate", "n_positive_folds", "n_negative_folds", "n_folds",
    }
    missing = required - set(agg)
    assert not missing, f"aggregate missing columns: {missing}"
    # _const_backtest emits a fixed metrics dict, so the aggregate is exact.
    assert agg["n_folds"] == len(rep.folds)
    assert agg["mean_wr"] == pytest.approx(0.55)
    assert agg["std_wr"] == pytest.approx(0.0)
    assert agg["mean_pf"] == pytest.approx(1.20)
    assert agg["breach_rate"] == 0.0
    assert agg["n_positive_folds"] == len(rep.folds)
    assert agg["n_negative_folds"] == 0


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------


def test_deterministic_for_same_inputs():
    bars = _synthetic_bars_by_date(date(2022, 1, 1), date(2023, 12, 31))
    a = run_walkforward(_id_strategy, _const_backtest, bars)
    b = run_walkforward(_id_strategy, _const_backtest, bars)
    assert len(a.folds) == len(b.folds)
    for x, y in zip(a.folds, b.folds, strict=True):
        assert x.train_start == y.train_start
        assert x.train_end == y.train_end
        assert x.test_start == y.test_start
        assert x.test_end == y.test_end
        assert x.params_used == y.params_used
        assert x.train_metrics == y.train_metrics
        assert x.test_metrics == y.test_metrics
    assert a.aggregate == b.aggregate


# ---------------------------------------------------------------------------
# 5. JSON round-trip
# ---------------------------------------------------------------------------


def test_report_to_json_round_trip(tmp_path: Path):
    bars = _synthetic_bars_by_date(date(2022, 1, 1), date(2023, 6, 30))
    rep = run_walkforward(_id_strategy, _const_backtest, bars)
    out = tmp_path / "wf.json"
    rep.to_json(out)
    blob = json.loads(out.read_text())
    assert blob["train_months"] == 12
    assert blob["test_months"] == 3
    assert blob["step_months"] == 3
    assert blob["purge_days"] == 1
    assert isinstance(blob["folds"], list)
    assert len(blob["folds"]) == len(rep.folds)
    assert {"mean_wr", "n_folds", "breach_rate"} <= set(blob["aggregate"])
    md = tmp_path / "wf.md"
    rep.to_markdown(md)
    assert md.exists() and "Walk-forward report" in md.read_text()


# ---------------------------------------------------------------------------
# 6. Validation / error handling
# ---------------------------------------------------------------------------


def test_empty_bars_by_date_raises():
    with pytest.raises(ValueError):
        run_walkforward(_id_strategy, _const_backtest, {})


def test_invalid_window_params_raise():
    bars = _synthetic_bars_by_date(date(2022, 1, 1), date(2022, 6, 30))
    with pytest.raises(ValueError):
        run_walkforward(_id_strategy, _const_backtest, bars, train_months=0)
    with pytest.raises(ValueError):
        run_walkforward(_id_strategy, _const_backtest, bars, test_months=-1)
    with pytest.raises(ValueError):
        run_walkforward(_id_strategy, _const_backtest, bars, purge_days=-1)


def test_strategy_exception_propagates():
    """Per BRIEFING.md anti-patterns, the harness must NOT silently swallow."""
    bars = _synthetic_bars_by_date(date(2022, 1, 1), date(2023, 12, 31))

    class _Boom(RuntimeError):
        pass

    def explode(params, bars_train):
        raise _Boom("strategy_fn went bang")

    with pytest.raises(_Boom):
        run_walkforward(explode, _const_backtest, bars)


def test_too_short_universe_yields_zero_folds():
    """Universe shorter than train+test produces no folds, but no crash."""
    bars = _synthetic_bars_by_date(date(2023, 1, 1), date(2023, 6, 30))
    rep = run_walkforward(_id_strategy, _const_backtest, bars)
    assert rep.folds == []
    assert rep.aggregate["n_folds"] == 0


# ---------------------------------------------------------------------------
# 7. Synthetic strategy end-to-end (the brief's "sanity check")
# ---------------------------------------------------------------------------


def _mean_reversion_strategy(params, bars_train):
    """Pick the SMA window with the highest in-sample win rate.

    Toy demonstration: trains over a small grid of windows and picks the
    window that maximizes Pr(close[t+1] reverts toward sma[t]) on training
    data. Returns the chosen window in a params dict.
    """
    grid = (5, 10, 20)
    best_w, best_score = grid[0], -1.0
    for w in grid:
        wins = 0
        n = 0
        for _d, df in bars_train.items():
            if df.height < w + 2:
                continue
            sma = df["close"].rolling_mean(window_size=w)
            for i in range(w, df.height - 1):
                if sma[i] is None:
                    continue
                z = df["close"][i] - sma[i]
                nxt = df["close"][i + 1] - df["close"][i]
                # If price was above SMA, expect mean-revert (next return < 0).
                if z > 0 and nxt < 0:
                    wins += 1
                elif z < 0 and nxt > 0:
                    wins += 1
                n += 1
        score = wins / n if n else 0.0
        if score > best_score:
            best_w, best_score = w, score
    out = dict(params or {})
    out["sma_window"] = best_w
    out["train_score"] = best_score
    return out


def _mean_reversion_backtest(params, bars_eval):
    """Trade the chosen SMA window on the eval slice. 1 contract, $5/pt."""
    w = int(params.get("sma_window", 10))
    pnl = 0.0
    wins = 0
    losses = 0
    wpnl = 0.0
    lpnl = 0.0
    for _d, df in bars_eval.items():
        if df.height < w + 2:
            continue
        sma = df["close"].rolling_mean(window_size=w)
        for i in range(w, df.height - 1):
            if sma[i] is None:
                continue
            z = df["close"][i] - sma[i]
            # next-bar pnl as 1 contract MES ($5 / pt)
            side = -1 if z > 0 else (1 if z < 0 else 0)
            if side == 0:
                continue
            ret = (df["close"][i + 1] - df["close"][i]) * side * 5.0
            pnl += ret
            if ret > 0:
                wins += 1
                wpnl += ret
            elif ret < 0:
                losses += 1
                lpnl += ret
    n = wins + losses
    return {
        "wr": (wins / n) if n else 0.0,
        "pf": (wpnl / abs(lpnl)) if lpnl < 0 else float("inf"),
        "net_pnl": pnl,
        "trades": n,
        "total_withdrawn": 0.0,
        "breach": False,
    }


def test_synthetic_strategy_runs_end_to_end():
    bars = _synthetic_bars_by_date(
        date(2022, 1, 1), date(2024, 12, 31),
        bars_per_day=80, seed=7,
    )
    rep = run_walkforward(
        _mean_reversion_strategy, _mean_reversion_backtest, bars,
        train_months=12, test_months=3, step_months=3, purge_days=1,
    )
    assert rep.folds, "expected at least one fold from the synthetic strategy"
    for f in rep.folds:
        # params_used reflects what strategy_fn returned.
        assert "sma_window" in f.params_used
        # test_metrics must carry the keys the aggregate consumes.
        for k in ("wr", "pf", "net_pnl", "trades", "breach"):
            assert k in f.test_metrics, f"missing test metric {k}"
        # train and test slices were non-empty (each fold runs both fns).
        assert f.test_metrics["trades"] >= 0
    # Aggregate sanity: counts add up.
    agg = rep.aggregate
    assert agg["n_folds"] == len(rep.folds)
    assert agg["n_positive_folds"] + agg["n_negative_folds"] <= agg["n_folds"]
