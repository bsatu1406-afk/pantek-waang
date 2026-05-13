"""Flow-event detector tests."""

from __future__ import annotations

import pandas as pd

from app.processing.flow_events import FlowEventConfig, detect_flow_events


def _row(**kw):
    base = dict(
        ts=pd.Timestamp("2026-01-02T14:30:00Z"),
        symbol="SPXW",
        expiration=pd.Timestamp("2026-01-02").date(),
        strike=5800.0,
        option_type="C",
        price=1.00,
        size=10,
        side=1,
        exchange="CBOE",
    )
    base.update(kw)
    return base


def test_block_flagged_at_threshold():
    trades = pd.DataFrame([
        _row(size=99),
        _row(size=100),
        _row(size=200),
    ])
    cfg = FlowEventConfig(block_min_size=100,
                          uoa_min_absolute_volume=10_000_000,  # disable UOA
                          uoa_volume_multiplier=1e9)
    events = detect_flow_events(trades, config=cfg)
    blocks = [e for e in events if e["event_type"] == "BLOCK"]
    assert len(blocks) == 2
    assert {b["size"] for b in blocks} == {100, 200}


def test_sweep_requires_multi_venue_same_side():
    trades = pd.DataFrame([
        _row(ts=pd.Timestamp("2026-01-02T14:30:00.000Z"), size=20, exchange="CBOE", price=1.00),
        _row(ts=pd.Timestamp("2026-01-02T14:30:00.020Z"), size=20, exchange="ARCA", price=1.01),
        _row(ts=pd.Timestamp("2026-01-02T14:30:00.080Z"), size=20, exchange="ISE", price=1.02),
    ])
    # Disable the premium gate to isolate the multi-venue requirement.
    cfg = FlowEventConfig(sweep_window_ms=200, sweep_min_legs=3,
                          sweep_min_premium=0.0,
                          block_min_size=10_000, uoa_min_absolute_volume=10_000_000,
                          uoa_volume_multiplier=1e9)
    events = detect_flow_events(trades, config=cfg)
    sweeps = [e for e in events if e["event_type"] == "SWEEP"]
    assert len(sweeps) == 1
    assert sweeps[0]["size"] == 60
    assert sweeps[0]["legs"] == 3
    assert set(sweeps[0]["venues"]) == {"CBOE", "ARCA", "ISE"}


def test_sweep_not_flagged_outside_window():
    trades = pd.DataFrame([
        _row(ts=pd.Timestamp("2026-01-02T14:30:00.000Z"), exchange="CBOE"),
        _row(ts=pd.Timestamp("2026-01-02T14:31:00.000Z"), exchange="ARCA"),
        _row(ts=pd.Timestamp("2026-01-02T14:32:00.000Z"), exchange="ISE"),
    ])
    cfg = FlowEventConfig(sweep_window_ms=200, sweep_min_legs=3,
                          sweep_min_premium=0.0,
                          block_min_size=10_000, uoa_min_absolute_volume=10_000_000,
                          uoa_volume_multiplier=1e9)
    events = detect_flow_events(trades, config=cfg)
    sweeps = [e for e in events if e["event_type"] == "SWEEP"]
    assert sweeps == []


def test_uoa_uses_absolute_threshold_when_no_adv():
    trades = pd.DataFrame([
        _row(size=2000),
        _row(size=4000),  # total 6000 on this contract
    ])
    cfg = FlowEventConfig(uoa_min_absolute_volume=5000,
                          uoa_volume_multiplier=1e9,
                          block_min_size=10_000)
    events = detect_flow_events(trades, config=cfg)
    uoas = [e for e in events if e["event_type"] == "UOA"]
    assert len(uoas) == 1
    assert uoas[0]["size"] == 6000


def test_uoa_uses_adv_when_provided():
    trades = pd.DataFrame([_row(size=300)])  # today_volume = 300
    adv = pd.DataFrame([{
        "symbol": "SPXW",
        "expiration": pd.Timestamp("2026-01-02").date(),
        "strike": 5800.0,
        "option_type": "C",
        "avg_daily_volume": 50.0,
    }])
    cfg = FlowEventConfig(
        uoa_volume_multiplier=5.0,           # 50 × 5 = 250 threshold
        uoa_min_absolute_volume=10_000_000,  # disable absolute fallback
        block_min_size=10_000,
    )
    events = detect_flow_events(trades, contract_adv=adv, config=cfg)
    uoas = [e for e in events if e["event_type"] == "UOA"]
    assert len(uoas) == 1
    assert uoas[0]["meta"]["avg_daily_volume"] == 50.0


def test_empty_input_returns_empty_list():
    assert detect_flow_events(pd.DataFrame()) == []
