"""Alert rule framework — declarative trigger evaluator.

Rules are stored in the ``alert_rules`` table as small JSON expressions.
At pipeline time we feed each rule a snapshot of the current metric set
and emit ``alert_events`` rows for every rule that fires. The website
later surfaces those events to subscribers (Discord / Telegram / push).

Supported expression grammar (intentionally tiny):

    {
      "field": "<dotted.path.into.payload>",
      "op":    "<gt|lt|ge|le|eq|ne|abs_gt|abs_lt|cross_above|cross_below>",
      "value": <constant>,
      "lookback": <int seconds, optional, only for cross_above/cross_below>
    }

Composable via boolean trees:

    {"all": [<rule>, <rule>, ...]}        # every clause must be true
    {"any": [<rule>, <rule>, ...]}        # at least one clause true

For ``cross_above`` / ``cross_below`` the evaluator needs a ``previous``
payload (the last evaluated snapshot) to detect the boundary crossing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AlertEvaluationResult:
    fired: bool
    matched: list[str]
    """Names of clauses that contributed to the firing."""


_OPS_NUMERIC = {
    "gt":  lambda a, b: a > b,
    "lt":  lambda a, b: a < b,
    "ge":  lambda a, b: a >= b,
    "le":  lambda a, b: a <= b,
    "eq":  lambda a, b: a == b,
    "ne":  lambda a, b: a != b,
    "abs_gt": lambda a, b: abs(a) > b,
    "abs_lt": lambda a, b: abs(a) < b,
}


def evaluate(
    rule: dict[str, Any],
    *,
    payload: dict[str, Any],
    previous: dict[str, Any] | None = None,
) -> AlertEvaluationResult:
    """Recursively evaluate ``rule`` against the current ``payload``."""
    matched: list[str] = []

    def walk(node: dict[str, Any]) -> bool:
        if not isinstance(node, dict):
            return False
        if "all" in node and isinstance(node["all"], list):
            return all(walk(child) for child in node["all"])
        if "any" in node and isinstance(node["any"], list):
            return any(walk(child) for child in node["any"])
        op = node.get("op")
        field = node.get("field")
        if not isinstance(field, str) or not isinstance(op, str):
            return False
        current_val = _resolve(payload, field)
        constant = node.get("value")
        if op in _OPS_NUMERIC:
            if current_val is None or constant is None:
                return False
            try:
                ok = _OPS_NUMERIC[op](float(current_val), float(constant))
            except (TypeError, ValueError):
                return False
            if ok:
                matched.append(f"{field} {op} {constant}")
            return ok
        if op in {"cross_above", "cross_below"}:
            if previous is None:
                return False
            prev_val = _resolve(previous, field)
            if current_val is None or prev_val is None or constant is None:
                return False
            try:
                cur = float(current_val)
                prev = float(prev_val)
                threshold = float(constant)
            except (TypeError, ValueError):
                return False
            if op == "cross_above":
                ok = prev <= threshold < cur
            else:
                ok = prev >= threshold > cur
            if ok:
                matched.append(f"{field} {op} {constant}")
            return ok
        return False

    fired = walk(rule)
    return AlertEvaluationResult(fired=fired, matched=matched)


def _resolve(payload: dict[str, Any], path: str) -> Any:
    cur: Any = payload
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur
