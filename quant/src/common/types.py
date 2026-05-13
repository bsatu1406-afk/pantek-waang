"""Shared dataclasses + enums used across specialists, engine, and reports."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class Phase(str, Enum):
    EVAL = "eval"
    FUNDED = "funded"
    PASSED_EVAL_BREACHED_FUNDED = "passed_eval_breached_funded"
    BREACHED_EVAL = "breached_eval"
    ACTIVE_EVAL = "active_eval"
    ACTIVE_FUNDED = "active_funded"
    RETIRED_PROFITABLE = "retired_profitable"


class EdgeLabel(str, Enum):
    POSITIVE_EDGE = "positive_edge"
    MARGINAL = "marginal"
    NEGATIVE = "negative"


@dataclass(frozen=True)
class Signal:
    """A trading signal emitted by a specialist.

    All specialists MUST emit Signals via this exact contract.
    """

    timestamp: datetime          # bar timestamp (UTC, tz-aware) when signal fires
    side: Side                   # long or short
    confidence: float            # 0.0..1.0 — specialist's self-rated confidence
    specialist: str              # short id, e.g. "cvd", "regime", "vwap_ext"
    setup_id: str                # specialist-internal id of the setup pattern
    entry_price: float           # suggested entry (mid or aggressive)
    stop_price: float            # hard stop in price terms
    target_price: Optional[float] = None  # optional TP; None = managed exit
    metadata: dict = field(default_factory=dict)  # diagnostic blob


@dataclass
class Trade:
    """A realized round-trip."""

    account_id: str
    open_ts: datetime
    close_ts: datetime
    side: Side
    qty: int                     # MES contracts (positive int, 1c = 1)
    entry_price: float
    exit_price: float
    pnl: float                   # $ — already accounts for tick value & qty
    reason: str                  # "tp", "stop", "eod_flat", "daily_lock", etc.
    specialist: str
    signal_id: Optional[str] = None
