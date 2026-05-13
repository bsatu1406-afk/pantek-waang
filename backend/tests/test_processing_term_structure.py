"""Term-structure + 25Δ risk-reversal tests."""

from __future__ import annotations

import pandas as pd

from app.processing.term_structure import compute_term_structure


def _chain():
    today = pd.Timestamp("2026-01-02")
    rows = []
    expirations = [today + pd.Timedelta(days=d) for d in (7, 30)]
    # symmetric synthetic chain around spot 5800
    for exp in expirations:
        for strike in (5700, 5800, 5900):
            for opt, delta_call in (("C", 1), ("P", -1)):
                # crude IVs to give the test a deterministic skew
                iv = 0.20 + (strike - 5800) * 0.0005 * delta_call
                # synthetic delta values targeting 25Δ at 5700 (puts) / 5900 (calls)
                if opt == "C":
                    if strike == 5700:
                        delta = 0.75
                    elif strike == 5800:
                        delta = 0.50
                    else:
                        delta = 0.25
                else:
                    if strike == 5700:
                        delta = -0.25
                    elif strike == 5800:
                        delta = -0.50
                    else:
                        delta = -0.75
                rows.append({
                    "strike": strike,
                    "expiration": exp.date(),
                    "option_type": opt,
                    "iv": iv,
                    "delta": delta,
                    "underlying_price": 5800.0,
                })
    return pd.DataFrame(rows)


def test_term_structure_returns_one_entry_per_expiration():
    out = compute_term_structure(_chain(), today=pd.Timestamp("2026-01-02"))
    assert len(out) == 2
    # Sorted by ascending DTE.
    assert out[0]["days_to_expiry"] <= out[1]["days_to_expiry"]


def test_atm_iv_is_finite_and_close_to_synthetic_value():
    out = compute_term_structure(_chain(), today=pd.Timestamp("2026-01-02"))
    for entry in out:
        assert entry["atm_iv"] is not None
        assert 0.18 <= entry["atm_iv"] <= 0.22


def test_risk_reversal_25d_present():
    out = compute_term_structure(_chain(), today=pd.Timestamp("2026-01-02"))
    for entry in out:
        assert entry["risk_reversal_25d"] is not None


def test_empty_input_returns_empty_list():
    assert compute_term_structure(pd.DataFrame()) == []
