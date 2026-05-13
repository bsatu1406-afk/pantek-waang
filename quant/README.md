# orderflow-prop — ES Orderflow Strategy + Prop Firm Simulation (v4)

This subproject builds, backtests, and validates an orderflow-based futures
strategy for ES (E-mini S&P 500) and simulates trading it across a fleet of
simulated prop firm accounts with the modified withdrawal cycle defined in the
v4 briefing.

It is implemented as a **10-agent trading desk**: an Orchestrator (this codebase)
plus nine specialist child sessions that each own one module under
`src/strategy/specialists/`.

## Layout

```
quant/
├── pyproject.toml
├── data/
│   ├── raw/           # DBN.zst files from Databento (per schema/per month)
│   └── processed/     # Parquet per trading day, normalized
├── src/
│   ├── common/        # shared types, time utils
│   ├── data/          # data loaders + processors (DBN -> Parquet bars)
│   ├── simulation/    # prop-firm engine (accounts, drawdown, withdrawals)
│   ├── strategy/
│   │   ├── specialists/   # agent #2..#8 signal generators
│   │   └── ensemble.py    # signal router + ML rank + risk veto
│   ├── backtest/      # walk-forward + 1c MES standalone audit runner
│   └── report/        # per-account, monthly cashflow, specialist audit
├── tests/             # pytest unit tests (engine rules are pinned)
├── scripts/           # one-shot scripts (download_data.py, etc.)
└── reports/           # generated reports (markdown / json / csv)
```

## Engine rules (PINNED)

Simulation engine implements the v4 briefing literally. **Do not modify** these
constants without a corresponding briefing update:

| Param | Value |
|---|---|
| Account purchase | $80 |
| Starting equity | $25,000 |
| Eval pass target | $26,500 (+$1,500) |
| Daily lock floor | -$250 below prev-day EOD high |
| Daily $500 max profit cap (funded) | locks day positively, no bucket hit |
| Total drawdown bucket | $1,000 → decrements $250 per daily-lock day |
| Bucket = 0 | account BREACHED |
| Funded reset on eval pass | equity $25,000, bucket $1,000 |
| Withdrawal trigger | every 5 non-consecutive winning days |
| Withdrawal % | 50% for cycles 1-3, 80% for cycle 4+ |
| Withdrawal base | `current_equity - 25000` (compounds endapan) |
| RTH | 09:30–16:00 ET, no overnight |
| Max active accounts | 8 |

## Quick start

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .[dev]

# Download 5y of data (free ohlcv-1s + paid TBBO):
DATABENTO_API_KEY=... python scripts/download_data.py \
    --start 2020-05-13 --end 2025-05-13 \
    --schemas ohlcv-1s definition statistics tbbo

# Preprocess raw DBN -> per-day Parquet:
python -m src.data.processor

# Run engine unit tests (pins the v4 rules):
pytest tests/test_engine.py -v

# Run a single specialist standalone backtest (1c MES audit):
python -m src.backtest.standalone --specialist regime --period 2024-01:2024-12

# Run full ensemble + walk-forward:
python -m src.backtest.runner --config configs/walk_forward.yaml
```

## Reports

After `runner` completes, look in `reports/`:

- `per_account_detail.csv` — one row per account ever bought
- `backtest_report.md` — narrative summary (sections A–G)
- `walk_forward.json` — per-fold metrics
- `specialist_audit.md` — honest per-specialist verdict

See [`PR description`](#) for the rendered headline numbers.
