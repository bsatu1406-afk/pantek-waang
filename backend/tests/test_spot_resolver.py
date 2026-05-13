"""Tests for the Rev 4 futures-first spot resolver."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

import app.processing.spot as spot_mod
from app.processing.spot import (
    SpotResult,
    get_front_month_contract,
    reset_basis_cache,
    resolve_spot,
    spot_result_to_payload,
    synthesize_underlying_price,
)

# ──────────────────────────────────────────────────────────────────────────
# Fake AsyncSession + chain / futures builders
# ──────────────────────────────────────────────────────────────────────────


class FakeSession:
    """Minimal async session that returns canned futures rows."""

    def __init__(self, futures_rows: list[dict] | None) -> None:
        self._rows = futures_rows or []

    async def execute(self, _stmt, _params):  # noqa: ARG002
        rows = self._rows

        class _Mappings:
            def all(self_inner):
                return [dict(r) for r in rows]

        class _Result:
            def mappings(self_inner):
                return _Mappings()

        return _Result()


def _parity_chain(spot: float, *, expiry_days: int = 7) -> pd.DataFrame:
    """Build an ATM call/put pair so parity recovers ``spot``."""
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    expiry = today + pd.Timedelta(days=expiry_days)
    # At spot ≈ K, T tiny, parity → C − P ≈ S − K · e^{−rT}.
    # Use intrinsic-flat options: C = max(S-K, 0)+0.5, P = max(K-S, 0)+0.5.
    rows = []
    for strike in [spot - 5, spot, spot + 5]:
        c_intr = max(spot - strike, 0.0)
        p_intr = max(strike - spot, 0.0)
        rows.append({
            "strike": strike, "expiration": expiry, "option_type": "C",
            "bid": c_intr + 0.4, "ask": c_intr + 0.6,
        })
        rows.append({
            "strike": strike, "expiration": expiry, "option_type": "P",
            "bid": p_intr + 0.4, "ask": p_intr + 0.6,
        })
    return pd.DataFrame(rows)


# Pick a contract that is guaranteed to be in the future for the entire
# foreseeable test horizon — Z8 = Dec 2028 (>= 2 years out from any
# reasonable test run). Tests that need to test expired contracts pass
# their own override.
def _futures_rows(*, contract: str = "ESZ8", price: float = 5_000.0,
                   volume: int = 1000) -> list[dict]:
    now = datetime.now(UTC)
    return [
        {"contract_symbol": contract, "price": price, "ts": now, "volume": volume}
    ]


@pytest.fixture(autouse=True)
def _isolate_cache():
    reset_basis_cache()
    yield
    reset_basis_cache()


# ──────────────────────────────────────────────────────────────────────────
# Front-month selection
# ──────────────────────────────────────────────────────────────────────────


def test_get_front_month_picks_nearest_unexpired_quarterly() -> None:
    today = pd.Timestamp("2026-04-01")
    futures_df = pd.DataFrame(
        {
            "contract_symbol": ["ESH6", "ESM6", "ESU6"],
            "volume": [100, 50, 25],
        }
    )
    # March (H) expired before April; June (M) is the front month.
    contract = get_front_month_contract("SPXW", futures_df, today=today)
    assert contract == "ESM6"


def test_get_front_month_returns_none_for_unknown_symbol() -> None:
    assert get_front_month_contract("AAPL", pd.DataFrame()) is None


def test_get_front_month_skips_spreads_and_garbage() -> None:
    today = pd.Timestamp("2026-04-01")
    futures_df = pd.DataFrame(
        {"contract_symbol": ["ESH7-ESM7", "ESM7", "GARBAGE"]}
    )
    assert get_front_month_contract("SPXW", futures_df, today=today) == "ESM7"


# ──────────────────────────────────────────────────────────────────────────
# Parity fallback (Rev 3 path, still exercised)
# ──────────────────────────────────────────────────────────────────────────


def test_synthesize_underlying_price_recovers_spot_within_one_pct() -> None:
    chain = _parity_chain(spot=5000.0)
    est = synthesize_underlying_price(chain, risk_free_rate=0.05)
    assert est is not None
    assert abs(est - 5000.0) < 50.0  # one-percent band


def test_synthesize_underlying_price_handles_empty_chain() -> None:
    assert synthesize_underlying_price(pd.DataFrame()) is None


# ──────────────────────────────────────────────────────────────────────────
# resolve_spot — priority chain
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_spot_primary_futures_basis() -> None:
    """With both futures and parity available, source must be futures_basis."""
    # Build chain where parity recovers ≈5000 (anchored at strike=5000 with
    # tiny T) and put futures slightly below; basis = parity − futures.
    chain = _parity_chain(spot=5_000.0)
    fut_rows = _futures_rows(price=4_990.0)
    session = FakeSession(fut_rows)

    result = await resolve_spot("SPXW", chain, session)  # type: ignore[arg-type]
    assert result is not None
    assert isinstance(result, SpotResult)
    assert result.source == "futures_basis"
    assert result.futures_price == 4_990.0
    # Parity ≈ 5000 (within discount tolerance), futures = 4990, so basis ~ +5–10.
    assert result.basis is not None
    assert 0.0 < result.basis < 20.0
    # cash = futures + basis must be > futures
    assert result.price > result.futures_price


@pytest.mark.asyncio
async def test_resolve_spot_parity_fallback_when_no_futures() -> None:
    chain = _parity_chain(spot=4_980.0)
    session = FakeSession(futures_rows=[])

    result = await resolve_spot("SPXW", chain, session)  # type: ignore[arg-type]
    assert result is not None
    assert result.source == "parity"
    assert result.futures_price is None
    assert result.basis is None
    assert abs(result.price - 4_980.0) < 50.0


@pytest.mark.asyncio
async def test_resolve_spot_returns_none_when_nothing_available() -> None:
    session = FakeSession(futures_rows=[])
    result = await resolve_spot("SPXW", pd.DataFrame(), session)  # type: ignore[arg-type]
    assert result is None


@pytest.mark.asyncio
async def test_resolve_spot_stale_cache_when_chain_empty_after_priming() -> None:
    """After a fresh result is cached, an empty subsequent call may reuse it."""
    # Prime with a working call.
    chain = _parity_chain(spot=5_000.0)
    session = FakeSession(_futures_rows(price=5_000.0))
    first = await resolve_spot("SPXW", chain, session)  # type: ignore[arg-type]
    assert first is not None and first.source == "futures_basis"

    # Now call with no chain and no futures.
    empty_session = FakeSession(futures_rows=[])
    second = await resolve_spot("SPXW", pd.DataFrame(), empty_session)  # type: ignore[arg-type]
    assert second is not None
    assert second.source == "stale_cache"
    assert second.price == pytest.approx(first.price, abs=1e-6)


@pytest.mark.asyncio
async def test_resolve_spot_stale_cache_rejected_when_too_old() -> None:
    """Stale cache must not be used if older than the configured limit."""
    chain = _parity_chain(spot=5_000.0)
    session = FakeSession(_futures_rows(price=5_000.0))
    first = await resolve_spot("SPXW", chain, session)  # type: ignore[arg-type]
    assert first is not None

    # Forcibly age the cache.
    cached_price, _ = spot_mod._last_spot_cache["SPXW"]
    spot_mod._last_spot_cache["SPXW"] = (
        cached_price,
        datetime.now(UTC) - timedelta(hours=1),
    )

    empty_session = FakeSession(futures_rows=[])
    second = await resolve_spot("SPXW", pd.DataFrame(), empty_session)  # type: ignore[arg-type]
    assert second is None


# ──────────────────────────────────────────────────────────────────────────
# EMA smoothing
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_basis_ema_smoothing_pulls_toward_new_observation() -> None:
    """A second basis observation must move the EMA toward the new value."""
    # First tick: parity ≈5000, futures=4990, basis_instant ≈ +10.
    chain1 = _parity_chain(spot=5_000.0)
    sess1 = FakeSession(_futures_rows(price=4_990.0))
    r1 = await resolve_spot("SPXW", chain1, sess1)  # type: ignore[arg-type]
    assert r1 is not None and r1.basis is not None
    first_basis = r1.basis

    # Second tick: parity ≈4990, futures=4990, basis_instant ≈ 0.
    # With α=0.1 the EMA should move from +10 toward 0 by ~1.
    chain2 = _parity_chain(spot=4_990.0)
    sess2 = FakeSession(_futures_rows(price=4_990.0))
    r2 = await resolve_spot("SPXW", chain2, sess2)  # type: ignore[arg-type]
    assert r2 is not None and r2.basis is not None
    # Moved toward 0 (away from first_basis).
    assert r2.basis < first_basis


# ──────────────────────────────────────────────────────────────────────────
# Parity divergence WARNING
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parity_divergence_warning_logged(caplog: pytest.LogCaptureFixture, monkeypatch) -> None:
    """A large parity / primary disagreement must emit a structured WARNING.

    To force a divergence we (a) prime the basis cache to a stale value,
    then (b) hit ``resolve_spot`` with a parity value that wildly
    disagrees with futures + stale_basis.
    """
    # Force a very tight warn threshold so any divergence trips it.
    monkeypatch.setattr(
        spot_mod, "get_settings",
        lambda: SimpleNamespace(
            spot_basis_ema_alpha=0.1,
            spot_stale_cache_max_age_seconds=300,
            spot_parity_deviation_warn_pct=0.0001,
            risk_free_rate=0.05,
        ),
    )
    # Prime basis cache to a "wrong" value.
    spot_mod._basis_cache["SPXW"] = spot_mod._BasisEntry(
        value=-50.0, updated_at=datetime.now(UTC)
    )
    chain = _parity_chain(spot=5_000.0)  # parity ≈ 5000
    session = FakeSession(_futures_rows(price=5_000.0))  # primary ≈ 5000 + (-50) = 4950

    captured: list[tuple[str, dict]] = []

    def _capture(event, **kwargs):
        captured.append((event, kwargs))

    monkeypatch.setattr(spot_mod.logger, "warning", _capture)
    with caplog.at_level(logging.WARNING):
        result = await resolve_spot("SPXW", chain, session)  # type: ignore[arg-type]
    assert result is not None
    events = [name for name, _ in captured]
    assert any("parity_divergence" in e for e in events), (
        f"captured events: {events}"
    )


# ──────────────────────────────────────────────────────────────────────────
# payload serializer
# ──────────────────────────────────────────────────────────────────────────


def test_spot_result_to_payload_none_returns_nulls() -> None:
    payload = spot_result_to_payload(None)
    assert payload["price"] is None
    assert payload["source"] is None


def test_spot_result_to_payload_round_trip() -> None:
    r = SpotResult(
        price=5_000.123456,
        source="futures_basis",
        futures_price=4_998.5,
        basis=1.6,
        basis_age_seconds=12.345678,
        parity_price=5_002.0,
        parity_deviation_pct=0.038,
    )
    payload = spot_result_to_payload(r)
    assert payload["source"] == "futures_basis"
    assert payload["price"] == round(5_000.123456, 6)
    assert payload["basis_age_seconds"] == round(12.345678, 3)
