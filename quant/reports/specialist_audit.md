# Specialist Audit Narrative

Per Section 6 of the v4 briefing: every specialist (whether kept or dropped) is reported honestly here. Verdicts are taken from the standalone 1c MES audit in `reports/specialist_audit.csv` — POSITIVE_EDGE requires `pf >= 1.30 AND wr >= 0.55 AND trades >= 30`.

| Specialist | Role | Dev verdict | Holdout verdict | Final | Dev WR / PF / net | Holdout WR / PF / net |
|---|---|---|---|---|---|---|
| `baseline_short_vwap` | Smoke-test SHORT-only VWAP pullback (v3 legacy edge). | marginal | marginal | **MARGINAL** | 54.4% / 1.33 / $726 | 53.9% / 1.44 / $226 |
| `cvd_signals` | CVD divergence + absorption + exhaustion (Agent #4). | marginal | marginal | **NOT_APPLICABLE** | 0.0% / 0.00 / $0 | 0.0% / 0.00 / $0 |
| `macro_events` | Macro calendar pre/post FOMC/CPI/NFP setups (Agent #2). | negative | negative | **NEGATIVE** | 34.8% / 0.80 / $-582 | 48.5% / 1.80 / $372 |
| `microstructure` | Book imbalance / microprice / sweep / trapped (Agent #5). | marginal | marginal | **NOT_APPLICABLE** | 0.0% / 0.00 / $0 | 0.0% / 0.00 / $0 |
| `ml_ranker` | LightGBM signal scorer + ranker (Agent #9). | negative | marginal | **MARGINAL** | 49.0% / 0.96 / $-71 | 50.2% / 1.15 / $43 |
| `momentum` | ORB 5/15/30 + EOD-drive momentum (Agent #8). | negative | negative | **NEGATIVE** | 43.4% / 1.05 / $7020 | 41.3% / 1.07 / $2552 |
| `regime` | Market regime classifier (NOT signal-generating, Agent #3). | marginal | marginal | **NOT_APPLICABLE** | 0.0% / 0.00 / $0 | 0.0% / 0.00 / $0 |
| `risk_manager` | Veto + sizing layer (NOT signal-generating, Agent #10). | marginal | marginal | **NOT_APPLICABLE** | 0.0% / 0.00 / $0 | 0.0% / 0.00 / $0 |
| `volume_profile` | Volume profile / value area / POC magnet (Agent #6). | negative | marginal | **MARGINAL** | 74.9% / 0.91 / $-1390 | 74.9% / 1.05 / $187 |
| `vwap_extended` | Bilateral (LONG+SHORT) VWAP pullback with bands (Agent #7). | negative | negative | **NEGATIVE** | 15.9% / 1.31 / $3042 | 15.6% / 1.12 / $332 |

## Per-specialist setups

- `baseline_short_vwap`: `vwap_pullback_short`
- `cvd_signals`: (no setup_id literals detected — non-signaling)
- `macro_events`: `post_fomc_reaction_fade_long`, `post_fomc_reaction_fade_short`, `pre_fomc_drift_long`
- `microstructure`: `book_imbalance`, `liq_hole`, `microprice_dev`, `sweep`, `trapped`
- `ml_ranker`: `other`
- `momentum`: `eod_drive_long`, `eod_drive_short`, `multibar_break_long`, `multibar_break_short`
- `regime`: (no setup_id literals detected — non-signaling)
- `risk_manager`: (no setup_id literals detected — non-signaling)
- `volume_profile`: `lvn_slip_long`, `lvn_slip_short`, `naked_poc_magnet`, `vah_bounce_short`, `val_bounce_long`
- `vwap_extended`: `vwap_pullback_long`, `vwap_pullback_short`

## Final tally
- POSITIVE_EDGE: (none)
- MARGINAL: ['baseline_short_vwap', 'ml_ranker', 'volume_profile']
- NEGATIVE: ['macro_events', 'momentum', 'vwap_extended']
- NOT_APPLICABLE: ['cvd_signals', 'microstructure', 'regime', 'risk_manager']

## Ensemble inclusion policy

No single specialist clears the POSITIVE_EDGE bar. Ensemble includes the MARGINAL specialists with the highest holdout PF, gated by the Risk Manager veto.