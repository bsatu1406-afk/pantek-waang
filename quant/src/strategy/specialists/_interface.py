"""The single, mandatory specialist interface contract.

All specialist modules (`regime.py`, `cvd_signals.py`, `microstructure.py`,
`volume_profile.py`, `vwap_extended.py`, `momentum.py`, ...) MUST expose
`generate_signals(bars, params) -> list[Signal]` matching this protocol.

The standalone audit runner introspects modules by attribute name and will
hard-fail if the contract is broken.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import polars as pl

from src.common.types import Signal


@dataclass
class SpecialistParams:
    """Generic params bag. Each specialist subclasses this with concrete fields."""

    specialist_id: str = "unknown"


@runtime_checkable
class SpecialistProtocol(Protocol):
    """The function signature every specialist module must implement."""

    def generate_signals(self, bars: pl.DataFrame, params: SpecialistParams) -> list[Signal]:
        ...


# In practice we use module-level functions, not classes, so the runner checks
# for the *name* `generate_signals` on the imported module.
__all__ = ["SpecialistParams", "SpecialistProtocol"]
