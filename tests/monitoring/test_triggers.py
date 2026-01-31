"""Tests for retrain triggers."""

from datetime import datetime, timedelta, timezone

import pytest

from tsagentkit.monitoring.report import DriftReport, StabilityReport, TriggerResult
from tsagentkit.monitoring.triggers import (
    RetrainTrigger,
    TriggerEvaluator,
    TriggerType,
)


class TestRetrainTrigger:
    """Test RetrainTrigger dataclass."""

    def test_drift_trigger_defaults(self):
        """Test drift trigger with default threshold."""
        trigger = RetrainTrigger(TriggerType.DRIFT)
        assert trigger.threshold == 0.2
        assert trigger.enabled

    def test_performance_trigger_defaults(self):
        """Test performance trigger with default threshold."""
        trigger = RetrainTrigger(TriggerType.PERFORMANCE, metric_name="wape")
        assert trigger.threshold == 0.1

    def test_custom_threshold(self):
        """Test trigger with custom threshold."""
        trigger = RetrainTrigger(TriggerType.DRIFT, threshold=0.15)
        assert trigger.threshold == 0.15

    def test_disabled_trigger(self):
        """Test disabled trigger."""
        trigger = RetrainTrigger(TriggerType.DRIFT, enabled=False)
        assert not trigger.enabled


class TestTriggerEvaluatorInit:
    """Test TriggerEvaluator initialization."""

    def test_filters_disabled_triggers(self):
        """Test that disabled triggers are filtered out."""
        triggers = [
            RetrainTrigger(TriggerType.DRIFT, enabled=True),
            RetrainTrigger(TriggerType.SCHEDULE, enabled=False),
        ]
        evaluator = TriggerEvaluator(triggers)
        assert len(evaluator.triggers) == 1


class TestDriftTriggerEvaluation:
    """Test drift trigger evaluation."""

    def test_drift_trigger_fires(self):
        """Test that drift trigger fires when threshold exceeded."""
        trigger = RetrainTrigger(TriggerType.DRIFT, threshold=0.2)
        evaluator = TriggerEvaluator([trigger])

        drift_report = DriftReport(
            drift_detected=True,
            feature_drifts={},
            overall_drift_score=0.25,
            threshold_used=0.2,
        )

        results = evaluator.evaluate(drift_report=drift_report)

        assert len(results) == 1
        assert results[0].fired
        assert "exceeded" in results[0].reason

    def test_drift_trigger_no_fire(self):
        """Test that drift trigger doesn't fire when below threshold."""
        trigger = RetrainTrigger(TriggerType.DRIFT, threshold=0.2)
        evaluator = TriggerEvaluator([trigger])

        drift_report = DriftReport(
            drift_detected=False,
            feature_drifts={},
            overall_drift_score=0.1,
            threshold_used=0.2,
        )

        results = evaluator.evaluate(drift_report=drift_report)

        assert not results[0].fired
        assert "below" in results[0].reason

    def test_drift_trigger_no_report(self):
        """Test drift trigger with no report."""
        trigger = RetrainTrigger(TriggerType.DRIFT)
        evaluator = TriggerEvaluator([trigger])

        results = evaluator.evaluate()

        assert not results[0].fired
        assert "No drift report" in results[0].reason


class TestScheduleTriggerEvaluation:
    """Test schedule trigger evaluation."""

    def test_schedule_trigger_fires(self):
        """Test that schedule trigger fires when interval elapsed."""
        trigger = RetrainTrigger(
            TriggerType.SCHEDULE,
            schedule_interval="1d",
        )
        evaluator = TriggerEvaluator([trigger])

        # Last trained 2 days ago
        last_train = datetime.now(timezone.utc) - timedelta(days=2)

        results = evaluator.evaluate(last_train_time=last_train)

        assert results[0].fired
        assert "elapsed" in results[0].reason

    def test_schedule_trigger_no_fire(self):
        """Test that schedule trigger doesn't fire when interval not elapsed."""
        trigger = RetrainTrigger(
            TriggerType.SCHEDULE,
            schedule_interval="7d",
        )
        evaluator = TriggerEvaluator([trigger])

        # Last trained just now
        last_train = datetime.now(timezone.utc)

        results = evaluator.evaluate(last_train_time=last_train)

        assert not results[0].fired
        assert "not yet elapsed" in results[0].reason

    def test_schedule_trigger_no_last_train(self):
        """Test schedule trigger with no last_train_time."""
        trigger = RetrainTrigger(TriggerType.SCHEDULE, schedule_interval="1d")
        evaluator = TriggerEvaluator([trigger])

        results = evaluator.evaluate()

        assert not results[0].fired
        assert "No last_train_time" in results[0].reason

    def test_invalid_interval(self):
        """Test schedule trigger with invalid interval."""
        trigger = RetrainTrigger(
            TriggerType.SCHEDULE,
            schedule_interval="invalid",
        )
        evaluator = TriggerEvaluator([trigger])

        last_train = datetime.now(timezone.utc) - timedelta(days=2)
        results = evaluator.evaluate(last_train_time=last_train)

        assert not results[0].fired
        assert "Invalid" in results[0].reason


class TestPerformanceTriggerEvaluation:
    """Test performance trigger evaluation."""

    def test_performance_trigger_fires(self):
        """Test that performance trigger fires when metric exceeds threshold."""
        trigger = RetrainTrigger(
            TriggerType.PERFORMANCE,
            metric_name="wape",
            threshold=0.15,
        )
        evaluator = TriggerEvaluator([trigger])

        metrics = {"wape": 0.20}
        results = evaluator.evaluate(current_metrics=metrics)

        assert results[0].fired
        assert "exceeded" in results[0].reason

    def test_performance_trigger_no_fire(self):
        """Test that performance trigger doesn't fire when metric within threshold."""
        trigger = RetrainTrigger(
            TriggerType.PERFORMANCE,
            metric_name="wape",
            threshold=0.15,
        )
        evaluator = TriggerEvaluator([trigger])

        metrics = {"wape": 0.10}
        results = evaluator.evaluate(current_metrics=metrics)

        assert not results[0].fired
        assert "within" in results[0].reason

    def test_performance_trigger_metric_not_found(self):
        """Test performance trigger when metric not in current_metrics."""
        trigger = RetrainTrigger(
            TriggerType.PERFORMANCE,
            metric_name="wape",
        )
        evaluator = TriggerEvaluator([trigger])

        metrics = {"smape": 0.20}
        results = evaluator.evaluate(current_metrics=metrics)

        assert not results[0].fired
        assert "not found" in results[0].reason

    def test_performance_trigger_no_metrics(self):
        """Test performance trigger with no metrics provided."""
        trigger = RetrainTrigger(TriggerType.PERFORMANCE, metric_name="wape")
        evaluator = TriggerEvaluator([trigger])

        results = evaluator.evaluate()

        assert not results[0].fired
        assert "No current metrics" in results[0].reason


class TestManualTrigger:
    """Test manual trigger evaluation."""

    def test_manual_trigger_never_fires(self):
        """Test that manual trigger never fires automatically."""
        trigger = RetrainTrigger(TriggerType.MANUAL)
        evaluator = TriggerEvaluator([trigger])

        results = evaluator.evaluate()

        assert not results[0].fired
        assert "not activated" in results[0].reason


class TestShouldRetrain:
    """Test should_retrain convenience method."""

    def test_should_retrain_true(self):
        """Test should_retrain returns True when trigger fires."""
        trigger = RetrainTrigger(TriggerType.DRIFT, threshold=0.2)
        evaluator = TriggerEvaluator([trigger])

        drift_report = DriftReport(
            drift_detected=True,
            feature_drifts={},
            overall_drift_score=0.25,
            threshold_used=0.2,
        )

        assert evaluator.should_retrain(drift_report=drift_report)

    def test_should_retrain_false(self):
        """Test should_retrain returns False when no triggers fire."""
        trigger = RetrainTrigger(TriggerType.DRIFT, threshold=0.2)
        evaluator = TriggerEvaluator([trigger])

        drift_report = DriftReport(
            drift_detected=False,
            feature_drifts={},
            overall_drift_score=0.1,
            threshold_used=0.2,
        )

        assert not evaluator.should_retrain(drift_report=drift_report)


class TestGetFiredTriggers:
    """Test get_fired_triggers method."""

    def test_gets_only_fired_triggers(self):
        """Test that only fired triggers are returned."""
        triggers = [
            RetrainTrigger(TriggerType.DRIFT, threshold=0.2),
            RetrainTrigger(TriggerType.SCHEDULE, schedule_interval="7d"),
        ]
        evaluator = TriggerEvaluator(triggers)

        drift_report = DriftReport(
            drift_detected=True,
            feature_drifts={},
            overall_drift_score=0.25,
            threshold_used=0.2,
        )

        fired = evaluator.get_fired_triggers(drift_report=drift_report)

        assert len(fired) == 1
        assert fired[0].trigger_type == TriggerType.DRIFT.value


class TestIntervalParsing:
    """Test interval string parsing."""

    def test_parse_days(self):
        """Test parsing day intervals."""
        evaluator = TriggerEvaluator([])
        interval = evaluator._parse_interval("7d")
        assert interval == timedelta(days=7)

    def test_parse_hours(self):
        """Test parsing hour intervals."""
        evaluator = TriggerEvaluator([])
        interval = evaluator._parse_interval("24h")
        assert interval == timedelta(hours=24)

    def test_parse_minutes(self):
        """Test parsing minute intervals."""
        evaluator = TriggerEvaluator([])
        interval = evaluator._parse_interval("30m")
        assert interval == timedelta(minutes=30)

    def test_parse_weeks(self):
        """Test parsing week intervals."""
        evaluator = TriggerEvaluator([])
        interval = evaluator._parse_interval("2w")
        assert interval == timedelta(weeks=2)

    def test_parse_invalid(self):
        """Test parsing invalid interval."""
        evaluator = TriggerEvaluator([])
        interval = evaluator._parse_interval("xyz")
        assert interval is None

    def test_parse_empty(self):
        """Test parsing empty interval."""
        evaluator = TriggerEvaluator([])
        interval = evaluator._parse_interval("")
        assert interval is None
