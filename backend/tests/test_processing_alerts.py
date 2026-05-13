"""Alert-rule evaluator tests."""

from __future__ import annotations

from app.processing.alerts import evaluate


def test_simple_gt_fires():
    rule = {"field": "GEX_NET_TOTAL.value", "op": "gt", "value": 1_000_000}
    res = evaluate(rule, payload={"GEX_NET_TOTAL": {"value": 5_000_000}})
    assert res.fired is True
    assert "GEX_NET_TOTAL.value gt 1000000" in res.matched[0]


def test_lt_does_not_fire_when_equal():
    rule = {"field": "x.v", "op": "lt", "value": 5}
    res = evaluate(rule, payload={"x": {"v": 5}})
    assert res.fired is False


def test_abs_gt():
    rule = {"field": "x.v", "op": "abs_gt", "value": 100}
    assert evaluate(rule, payload={"x": {"v": -150}}).fired is True
    assert evaluate(rule, payload={"x": {"v": -50}}).fired is False


def test_all_combinator():
    rule = {
        "all": [
            {"field": "g.value", "op": "gt", "value": 0},
            {"field": "r.value", "op": "ge", "value": 0.5},
        ]
    }
    payload = {"g": {"value": 1}, "r": {"value": 0.6}}
    assert evaluate(rule, payload=payload).fired is True
    payload = {"g": {"value": 1}, "r": {"value": 0.4}}
    assert evaluate(rule, payload=payload).fired is False


def test_any_combinator():
    rule = {
        "any": [
            {"field": "a", "op": "gt", "value": 0},
            {"field": "b", "op": "gt", "value": 0},
        ]
    }
    assert evaluate(rule, payload={"a": -1, "b": 1}).fired is True
    assert evaluate(rule, payload={"a": -1, "b": -1}).fired is False


def test_cross_above_uses_previous():
    rule = {"field": "x", "op": "cross_above", "value": 10}
    res = evaluate(rule, payload={"x": 12}, previous={"x": 8})
    assert res.fired is True
    res = evaluate(rule, payload={"x": 12}, previous={"x": 11})
    assert res.fired is False  # already above last tick


def test_missing_field_does_not_fire():
    rule = {"field": "missing.path", "op": "gt", "value": 0}
    res = evaluate(rule, payload={})
    assert res.fired is False
