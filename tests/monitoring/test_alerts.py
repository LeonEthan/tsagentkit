"""Tests for tsagentkit.monitoring.alerts – AlertCondition, AlertManager, factory functions."""

from __future__ import annotations

import pytest

from tsagentkit.monitoring.alerts import (
    Alert,
    AlertCondition,
    AlertManager,
    create_default_coverage_alerts,
    create_default_drift_alerts,
)

# ---------------------------------------------------------------------------
# AlertCondition.evaluate
# ---------------------------------------------------------------------------


class TestAlertConditionEvaluate:
    def _cond(self, operator: str, threshold: float = 0.5) -> AlertCondition:
        return AlertCondition(
            name="test",
            metric="m",
            operator=operator,
            threshold=threshold,
        )

    def test_lt_true(self) -> None:
        assert self._cond("lt", 0.5).evaluate(0.3) is True

    def test_lt_false(self) -> None:
        assert self._cond("lt", 0.5).evaluate(0.7) is False

    def test_gt_true(self) -> None:
        assert self._cond("gt", 0.5).evaluate(0.8) is True

    def test_gt_false(self) -> None:
        assert self._cond("gt", 0.5).evaluate(0.2) is False

    def test_eq_true(self) -> None:
        assert self._cond("eq", 1.0).evaluate(1.0) is True

    def test_eq_false(self) -> None:
        assert self._cond("eq", 1.0).evaluate(1.1) is False

    def test_ne_true(self) -> None:
        assert self._cond("ne", 1.0).evaluate(2.0) is True

    def test_ne_false(self) -> None:
        assert self._cond("ne", 1.0).evaluate(1.0) is False

    def test_lte_true_equal(self) -> None:
        assert self._cond("lte", 0.5).evaluate(0.5) is True

    def test_lte_true_less(self) -> None:
        assert self._cond("lte", 0.5).evaluate(0.3) is True

    def test_lte_false(self) -> None:
        assert self._cond("lte", 0.5).evaluate(0.7) is False

    def test_gte_true_equal(self) -> None:
        assert self._cond("gte", 0.5).evaluate(0.5) is True

    def test_gte_true_greater(self) -> None:
        assert self._cond("gte", 0.5).evaluate(0.8) is True

    def test_gte_false(self) -> None:
        assert self._cond("gte", 0.5).evaluate(0.3) is False

    def test_unknown_operator_returns_false(self) -> None:
        assert self._cond("unknown", 0.5).evaluate(0.5) is False


# ---------------------------------------------------------------------------
# AlertCondition.format_message
# ---------------------------------------------------------------------------


class TestAlertConditionFormatMessage:
    def test_default_message(self) -> None:
        cond = AlertCondition(name="test_alert", metric="coverage", operator="lt", threshold=0.8)
        msg = cond.format_message(0.6)
        assert "test_alert" in msg
        assert "coverage" in msg
        assert "below" in msg

    def test_custom_message(self) -> None:
        cond = AlertCondition(
            name="custom",
            metric="m",
            operator="gt",
            threshold=1.0,
            message="Value {value} exceeded {threshold}",
        )
        msg = cond.format_message(2.0)
        assert "2.0" in msg
        assert "1.0" in msg

    def test_custom_message_with_context(self) -> None:
        cond = AlertCondition(
            name="ctx",
            metric="m",
            operator="gt",
            threshold=1.0,
            message="Series {series_id} value {value}",
        )
        msg = cond.format_message(3.0, context={"series_id": "ABC"})
        assert "ABC" in msg
        assert "3.0" in msg


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------


class TestAlertManager:
    def test_add_and_check(self) -> None:
        mgr = AlertManager()
        mgr.add_condition(
            AlertCondition(
                name="low",
                metric="coverage",
                operator="lt",
                threshold=0.8,
            )
        )
        alerts = mgr.check_metrics({"coverage": 0.6})
        assert len(alerts) == 1
        assert isinstance(alerts[0], Alert)
        assert alerts[0].condition.name == "low"

    def test_no_trigger(self) -> None:
        mgr = AlertManager()
        mgr.add_condition(
            AlertCondition(
                name="low",
                metric="coverage",
                operator="lt",
                threshold=0.5,
            )
        )
        alerts = mgr.check_metrics({"coverage": 0.9})
        assert len(alerts) == 0

    def test_metric_not_present_skipped(self) -> None:
        mgr = AlertManager()
        mgr.add_condition(
            AlertCondition(
                name="drift",
                metric="drift_score",
                operator="gt",
                threshold=0.1,
            )
        )
        alerts = mgr.check_metrics({"coverage": 0.9})
        assert len(alerts) == 0

    def test_remove_condition(self) -> None:
        mgr = AlertManager()
        mgr.add_condition(AlertCondition(name="a", metric="m", operator="gt", threshold=0.5))
        mgr.add_condition(AlertCondition(name="b", metric="m", operator="lt", threshold=0.5))
        assert mgr.remove_condition("a") is True
        assert len(mgr.conditions) == 1
        assert mgr.conditions[0].name == "b"

    def test_remove_nonexistent(self) -> None:
        mgr = AlertManager()
        assert mgr.remove_condition("nope") is False

    def test_get_alert_history(self) -> None:
        mgr = AlertManager()
        mgr.add_condition(
            AlertCondition(
                name="warn",
                metric="m",
                operator="gt",
                threshold=0.5,
                severity="warning",
            )
        )
        mgr.add_condition(
            AlertCondition(
                name="crit",
                metric="m",
                operator="gt",
                threshold=0.3,
                severity="critical",
            )
        )
        mgr.check_metrics({"m": 0.8})
        all_history = mgr.get_alert_history()
        assert len(all_history) == 2
        warnings = mgr.get_alert_history(severity="warning")
        assert len(warnings) == 1
        assert warnings[0].condition.name == "warn"

    def test_clear_history(self) -> None:
        mgr = AlertManager()
        mgr.add_condition(AlertCondition(name="x", metric="m", operator="gt", threshold=0.0))
        mgr.check_metrics({"m": 1.0})
        assert len(mgr.get_alert_history()) == 1
        mgr.clear_history()
        assert len(mgr.get_alert_history()) == 0

    def test_alert_to_dict(self) -> None:
        mgr = AlertManager()
        mgr.add_condition(
            AlertCondition(
                name="test",
                metric="score",
                operator="gt",
                threshold=0.5,
            )
        )
        alerts = mgr.check_metrics({"score": 0.9})
        d = alerts[0].to_dict()
        assert d["name"] == "test"
        assert d["metric"] == "score"
        assert d["value"] == 0.9
        assert "timestamp" in d
        assert "message" in d


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestFactoryFunctions:
    def test_create_default_coverage_alerts(self) -> None:
        conditions = create_default_coverage_alerts()
        assert len(conditions) == 3  # default: [0.5, 0.8, 0.95]
        names = [c.name for c in conditions]
        assert "low_coverage_50%" in names
        assert "low_coverage_80%" in names
        assert "low_coverage_95%" in names

    def test_create_default_coverage_alerts_custom_levels(self) -> None:
        conditions = create_default_coverage_alerts(coverage_levels=[0.9], tolerance=0.1)
        assert len(conditions) == 1
        assert conditions[0].threshold == pytest.approx(0.8)

    def test_create_default_drift_alerts(self) -> None:
        conditions = create_default_drift_alerts()
        assert len(conditions) == 1
        assert conditions[0].name == "drift_detected"
        assert conditions[0].operator == "gt"

    def test_create_default_drift_alerts_custom_threshold(self) -> None:
        conditions = create_default_drift_alerts(drift_threshold=0.1)
        assert conditions[0].threshold == 0.1
