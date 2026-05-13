"""Data loaders for processed Parquet bars. Specialist code uses ONLY this API.

A specialist must NEVER reach into `data/raw/` or call the Databento client
directly — that would invalidate the data caching contract. Always go through
`BarLoader.load_session(date)` etc.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterator, Optional

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = REPO_ROOT / "data" / "processed"


def _path_for(prefix: str, sd: date) -> Path:
    return (
        PROC_DIR
        / prefix
        / f"{sd.year:04d}"
        / f"{sd.month:02d}"
        / f"{prefix}_{sd.isoformat()}.parquet"
    )


class BarLoader:
    """Loads per-session-date Parquet files for the v4 backtest universe."""

    def available_session_dates(self, prefix: str = "bars_1s") -> list[date]:
        out: list[date] = []
        root = PROC_DIR / prefix
        if not root.exists():
            return out
        for f in sorted(root.rglob(f"{prefix}_*.parquet")):
            try:
                sd_str = f.stem.replace(f"{prefix}_", "")
                out.append(date.fromisoformat(sd_str))
            except ValueError:
                continue
        return out

    def load_session(self, sd: date, prefix: str = "bars_1s") -> Optional[pl.DataFrame]:
        p = _path_for(prefix, sd)
        if not p.exists():
            return None
        return pl.read_parquet(p)

    def load_range(
        self,
        start: date,
        end: date,
        prefix: str = "bars_1s",
    ) -> pl.DataFrame:
        """Concatenate sessions in [start, end] inclusive."""
        frames: list[pl.DataFrame] = []
        for sd in self.available_session_dates(prefix):
            if sd < start or sd > end:
                continue
            df = self.load_session(sd, prefix)
            if df is not None and not df.is_empty():
                frames.append(df)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="vertical_relaxed")

    def iter_sessions(
        self,
        start: date,
        end: date,
        prefix: str = "bars_1s",
    ) -> Iterator[tuple[date, pl.DataFrame]]:
        for sd in self.available_session_dates(prefix):
            if sd < start or sd > end:
                continue
            df = self.load_session(sd, prefix)
            if df is not None and not df.is_empty():
                yield sd, df


# Convenience aliases for specialists
def load_bars(sd: date) -> Optional[pl.DataFrame]:
    return BarLoader().load_session(sd, "bars_1s")


def load_tbbo(sd: date) -> Optional[pl.DataFrame]:
    return BarLoader().load_session(sd, "tbbo")


def load_cvd_1s(sd: date) -> Optional[pl.DataFrame]:
    return BarLoader().load_session(sd, "cvd_1s")
