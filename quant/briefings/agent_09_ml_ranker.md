# Agent #9 — Machine Learning Ranker

## Mission

Train a LightGBM gradient-boosting model that scores each upstream signal by
its probability of hitting target before stop. The ensemble uses the score
as a filter (keep top-K per day) and as a tiebreaker when multiple signals
overlap.

## Output module

`src/strategy/specialists/ml_ranker.py`

```python
@dataclass
class MLRankerParams(SpecialistParams):
    specialist_id: str = "ml_ranker"
    forward_window_seconds: int = 600
    target_atr_mult: float = 1.0
    train_months: int = 9       # walk-forward train length
    test_months: int = 3        # walk-forward test length
    n_estimators: int = 400
    learning_rate: float = 0.05
    ...

def build_features(
    signals: list[Signal],
    bars: pl.DataFrame,
) -> pl.DataFrame:
    """Feature matrix indexed by (ts, specialist, setup_id) — past-only data."""

def fit_model(features: pl.DataFrame, labels: pl.Series) -> "lightgbm.Booster": ...

def score_signals(
    signals: list[Signal],
    bars: pl.DataFrame,
    model,
) -> list[Signal]:
    """Returns new signals with .metadata['ml_score'] populated and possibly
       .confidence overridden."""
```

## Features (suggested)

- Specialist + setup_id (one-hot)
- Hour-of-session, minute, regime label (from Agent #3 if available)
- 1-min, 5-min, 15-min realized vol
- 1-min and 5-min CVD slope
- ATR(20) relative to median
- VWAP distance in σ
- Last-3-day same-setup WR

## Label

`y = 1` if the signal's target_price (or fallback +1 ATR move in side
direction) is hit BEFORE its stop_price within `forward_window_seconds`.
Otherwise `y = 0`. Compute with vectorised Polars; do not loop.

## Anti-pattern (READ TWICE)

You MUST avoid lookahead. Build labels from a separate `forward` pass over
`bars` that ONLY uses future data; build features from a `past_only` pass.
Cross-validate with **purged walk-forward**: train ends ≥ forward_window
before test starts.

## Verdict thresholds

- POSITIVE_EDGE: walk-forward mean AUC ≥ 0.55 and lift over baseline
  (top-30% score) WR ≥ +6pp.
- MARGINAL: AUC 0.50–0.55.
- NEGATIVE: AUC < 0.50.

## Deliverable checklist

- [ ] `src/strategy/specialists/ml_ranker.py`
- [ ] `tests/test_ml_ranker.py` (≥ 4 tests, including a no-lookahead test)
- [ ] Walk-forward AUC table printed
- [ ] Pre-trained checkpoint saved to `models/ml_ranker.joblib`
