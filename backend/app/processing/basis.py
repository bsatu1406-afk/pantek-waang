"""SPX cash − ES futures basis tracker.

The basis is the dollar premium that the front-month ES futures contract
trades over (or under) SPX cash, driven by the cost of carry and dealer
positioning. Sudden basis dislocations often precede broader risk events.

This module is a simple stateless transformer: pass the latest spot
indication (e.g. SPX cash quote) and the futures last price, get back
``basis = futures - spot`` plus a few derived numbers we cache for the
website.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BasisSnapshot:
    spot: float | None
    futures: float | None
    basis: float | None
    basis_pct: float | None        # basis / spot
    """Decimal (e.g. 0.0012 = 12 bps), or None if spot is unavailable."""


def compute_basis(*, spot: float | None, futures: float | None) -> BasisSnapshot:
    if spot is None or futures is None:
        return BasisSnapshot(spot=spot, futures=futures, basis=None, basis_pct=None)
    if not (spot > 0 and futures > 0):
        return BasisSnapshot(spot=spot, futures=futures, basis=None, basis_pct=None)
    diff = float(futures) - float(spot)
    return BasisSnapshot(
        spot=float(spot),
        futures=float(futures),
        basis=diff,
        basis_pct=diff / float(spot),
    )
