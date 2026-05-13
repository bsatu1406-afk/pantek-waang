"""Alert pipeline: load active rules, evaluate against the latest snapshot,
persist firings into ``alert_events``.

Runs once per scheduler tick after the chain + flow pipelines have
finished. Each rule is matched against:

* the most recent ``computed_metrics`` rows for the rule's symbol
  (collapsed into a nested dict keyed by ``metric_type``), and
* (optionally) the previous tick's payload for ``cross_above`` /
  ``cross_below`` operators.

The cooldown logic prevents the same rule from spamming events more
often than ``cooldown_seconds``.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from app.core.logging import get_logger
from app.db.models import AlertEvent, AlertRule, ComputedMetric
from app.db.session import get_session_factory
from app.processing.alerts import evaluate

logger = get_logger(__name__)


_LAST_PAYLOAD: dict[str, dict] = {}
"""In-memory cache of the previous evaluation payload per symbol — used
by the ``cross_above`` / ``cross_below`` operators."""


async def run_alert_pipeline(*, symbol: str) -> int:
    """Evaluate every enabled rule against the latest snapshot for ``symbol``.

    Returns the number of new events persisted.
    """
    factory = get_session_factory()
    now = datetime.now(UTC)

    async with factory() as session:
        rules_stmt = (
            select(AlertRule)
            .where(AlertRule.enabled.is_(True))
            .where(AlertRule.symbol == symbol)
        )
        result = await session.execute(rules_stmt)
        rules = list(result.scalars().all())
        if not rules:
            return 0

        payload = await _build_payload(session, symbol)

    previous = _LAST_PAYLOAD.get(symbol)
    fired_event_rows: list[dict] = []
    fired_rule_ids: list = []
    for rule in rules:
        if rule.last_fired_at is not None:
            cooldown = timedelta(seconds=int(rule.cooldown_seconds or 0))
            if now - rule.last_fired_at < cooldown:
                continue
        try:
            outcome = evaluate(rule.rule, payload=payload, previous=previous)
        except Exception:  # noqa: BLE001
            logger.exception("alert_evaluation_error", rule=str(rule.id))
            continue
        if not outcome.fired:
            continue
        fired_event_rows.append({
            "rule_id": rule.id,
            "ts": now,
            "symbol": symbol,
            "matched": list(outcome.matched),
            "payload": _shrink_payload(payload),
        })
        fired_rule_ids.append(rule.id)

    _LAST_PAYLOAD[symbol] = payload

    if not fired_event_rows:
        return 0

    async with factory() as session:
        ins = insert(AlertEvent).values(fired_event_rows)
        await session.execute(ins)
        await session.execute(
            update(AlertRule)
            .where(AlertRule.id.in_(fired_rule_ids))
            .values(last_fired_at=now)
        )
        await session.commit()

    return len(fired_event_rows)


async def _build_payload(session, symbol: str) -> dict:
    """Latest value of every metric_type for ``symbol`` in the last 10 min."""
    cutoff = datetime.now(UTC) - timedelta(minutes=10)
    stmt = (
        select(
            ComputedMetric.metric_type,
            ComputedMetric.value,
            ComputedMetric.extra_json,
            ComputedMetric.ts,
        )
        .where(ComputedMetric.symbol == symbol)
        .where(ComputedMetric.ts >= cutoff)
        .order_by(ComputedMetric.ts.desc())
    )
    res = await session.execute(stmt)
    rows = res.all()
    payload: dict[str, dict] = defaultdict(dict)
    seen: set[str] = set()
    for metric_type, value, extra_json, ts in rows:
        if metric_type in seen:
            continue
        seen.add(metric_type)
        entry: dict = {"value": float(value) if value is not None else None,
                       "ts": ts.isoformat()}
        if isinstance(extra_json, dict):
            entry.update(extra_json)
        payload[metric_type] = dict(entry)
    return dict(payload)


def _shrink_payload(payload: dict) -> dict:
    """Trim large list fields before persisting payload alongside the alert
    event so we don't blow up the JSONB column on every fire."""
    out: dict = {}
    for k, v in payload.items():
        if isinstance(v, dict):
            entry = {kk: vv for kk, vv in v.items()
                     if not isinstance(vv, list) or len(vv) <= 10}
            out[k] = entry
        else:
            out[k] = v
    return out
