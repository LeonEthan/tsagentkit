"""Retrain triggers for model monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING

from tsagentkit.monitoring.report import TriggerResult

if TYPE_CHECKING:
    from tsagentkit.monitoring.report import DriftReport, StabilityReport


class TriggerType(Enum):
    """Types of retrain triggers.

    - DRIFT: Data drift detected (PSI/KS above threshold)
    - SCHEDULE: Time-based trigger (e.g., daily, weekly)
    - PERFORMANCE: Metric degradation (accuracy drop)
    - MANUAL: Explicit manual trigger
    """

    DRIFT = "drift"
    SCHEDULE = "schedule"
    PERFORMANCE = "performance"
    MANUAL = "manual"


@dataclass
class RetrainTrigger:
    """Configuration for a retrain trigger.

    Attributes:
        trigger_type: Type of trigger
        threshold: Threshold value for trigger activation
        metric_name: For PERFORMANCE triggers, the metric to monitor
        schedule_interval: For SCHEDULE triggers, the interval (e.g., "1d", "7d")
        enabled: Whether this trigger is active

    Example:
        >>> drift_trigger = RetrainTrigger(
        ...     trigger_type=TriggerType.DRIFT,
        ...     threshold=0.2,
        ... )
        >>> schedule_trigger = RetrainTrigger(
        ...     trigger_type=TriggerType.SCHEDULE,
        ...     schedule_interval="7d",
        ... )
        >>> perf_trigger = RetrainTrigger(
        ...     trigger_type=TriggerType.PERFORMANCE,
        ...     metric_name="wape",
        ...     threshold=0.15,
        ... )
    """

    trigger_type: TriggerType
    threshold: float | None = None
    metric_name: str | None = None
    schedule_interval: str | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate trigger configuration."""
        if self.trigger_type == TriggerType.DRIFT and self.threshold is None:
            self.threshold = 0.2  # Default PSI threshold
        elif self.trigger_type == TriggerType.PERFORMANCE and self.threshold is None:
            self.threshold = 0.1  # Default 10% degradation


class TriggerEvaluator:
    """Evaluate retrain triggers based on monitoring data.

    Example:
        >>> triggers = [
        ...     RetrainTrigger(TriggerType.DRIFT, threshold=0.2),
        ...     RetrainTrigger(TriggerType.SCHEDULE, schedule_interval="7d"),
        ... ]
        >>> evaluator = TriggerEvaluator(triggers)
        >>>
        >>> # Evaluate with monitoring data
        >>> results = evaluator.evaluate(
        ...     drift_report=drift_report,
        ...     last_train_time=datetime(2024, 1, 1),
        ... )
        >>>
        >>> if evaluator.should_retrain(...):
        ...     print("Retraining needed!")
    """

    def __init__(self, triggers: list[RetrainTrigger]):
        """Initialize with trigger configurations.

        Args:
            triggers: List of trigger configurations to evaluate
        """
        self.triggers = [t for t in triggers if t.enabled]

    def evaluate(
        self,
        drift_report: DriftReport | None = None,
        stability_report: StabilityReport | None = None,
        current_metrics: dict[str, float] | None = None,
        last_train_time: datetime | None = None,
    ) -> list[TriggerResult]:
        """Evaluate all triggers and return those that fired.

        Args:
            drift_report: Latest drift detection results
            stability_report: Latest stability metrics
            current_metrics: Current model performance metrics
            last_train_time: When model was last trained

        Returns:
            List of TriggerResult for all evaluated triggers
        """
        results = []

        for trigger in self.triggers:
            result = self._evaluate_single_trigger(
                trigger,
                drift_report,
                stability_report,
                current_metrics,
                last_train_time,
            )
            results.append(result)

        return results

    def _evaluate_single_trigger(
        self,
        trigger: RetrainTrigger,
        drift_report: DriftReport | None,
        stability_report: StabilityReport | None,
        current_metrics: dict[str, float] | None,
        last_train_time: datetime | None,
    ) -> TriggerResult:
        """Evaluate a single trigger.

        Args:
            trigger: Trigger configuration
            drift_report: Drift detection results
            stability_report: Stability metrics
            current_metrics: Current performance metrics
            last_train_time: Last training timestamp

        Returns:
            TriggerResult with evaluation outcome
        """
        if trigger.trigger_type == TriggerType.DRIFT:
            return self._evaluate_drift_trigger(trigger, drift_report)
        elif trigger.trigger_type == TriggerType.SCHEDULE:
            return self._evaluate_schedule_trigger(trigger, last_train_time)
        elif trigger.trigger_type == TriggerType.PERFORMANCE:
            return self._evaluate_performance_trigger(trigger, current_metrics)
        elif trigger.trigger_type == TriggerType.MANUAL:
            return self._evaluate_manual_trigger(trigger)
        else:
            return TriggerResult(
                trigger_type=trigger.trigger_type.value,
                fired=False,
                reason=f"Unknown trigger type: {trigger.trigger_type}",
            )

    def _evaluate_drift_trigger(
        self,
        trigger: RetrainTrigger,
        drift_report: DriftReport | None,
    ) -> TriggerResult:
        """Evaluate drift-based trigger."""
        if drift_report is None:
            return TriggerResult(
                trigger_type=TriggerType.DRIFT.value,
                fired=False,
                reason="No drift report provided",
            )

        threshold = trigger.threshold or 0.2

        if drift_report.overall_drift_score > threshold:
            return TriggerResult(
                trigger_type=TriggerType.DRIFT.value,
                fired=True,
                reason=(
                    f"PSI drift score {drift_report.overall_drift_score:.3f} "
                    f"exceeded threshold {threshold}"
                ),
                metadata={
                    "psi_score": drift_report.overall_drift_score,
                    "threshold": threshold,
                    "drifting_features": drift_report.get_drifting_features(),
                },
            )
        else:
            return TriggerResult(
                trigger_type=TriggerType.DRIFT.value,
                fired=False,
                reason=(
                    f"PSI drift score {drift_report.overall_drift_score:.3f} "
                    f"below threshold {threshold}"
                ),
                metadata={"psi_score": drift_report.overall_drift_score},
            )

    def _evaluate_schedule_trigger(
        self,
        trigger: RetrainTrigger,
        last_train_time: datetime | None,
    ) -> TriggerResult:
        """Evaluate schedule-based trigger."""
        if last_train_time is None:
            return TriggerResult(
                trigger_type=TriggerType.SCHEDULE.value,
                fired=False,
                reason="No last_train_time provided",
            )

        if trigger.schedule_interval is None:
            return TriggerResult(
                trigger_type=TriggerType.SCHEDULE.value,
                fired=False,
                reason="No schedule_interval configured",
            )

        # Parse interval (e.g., "7d", "1h")
        interval = self._parse_interval(trigger.schedule_interval)
        if interval is None:
            return TriggerResult(
                trigger_type=TriggerType.SCHEDULE.value,
                fired=False,
                reason=f"Invalid schedule_interval: {trigger.schedule_interval}",
            )

        # Ensure last_train_time is timezone-aware
        if last_train_time.tzinfo is None:
            last_train_time = last_train_time.replace(tzinfo=UTC)

        next_train_time = last_train_time + interval
        now = datetime.now(UTC)

        if now >= next_train_time:
            return TriggerResult(
                trigger_type=TriggerType.SCHEDULE.value,
                fired=True,
                reason=(
                    f"Schedule interval {trigger.schedule_interval} elapsed "
                    f"since last training at {last_train_time.isoformat()}"
                ),
                metadata={
                    "last_train_time": last_train_time.isoformat(),
                    "next_train_time": next_train_time.isoformat(),
                    "interval": trigger.schedule_interval,
                },
            )
        else:
            return TriggerResult(
                trigger_type=TriggerType.SCHEDULE.value,
                fired=False,
                reason=(
                    f"Schedule interval {trigger.schedule_interval} not yet elapsed"
                ),
                metadata={
                    "next_train_time": next_train_time.isoformat(),
                },
            )

    def _evaluate_performance_trigger(
        self,
        trigger: RetrainTrigger,
        current_metrics: dict[str, float] | None,
    ) -> TriggerResult:
        """Evaluate performance-based trigger."""
        if current_metrics is None:
            return TriggerResult(
                trigger_type=TriggerType.PERFORMANCE.value,
                fired=False,
                reason="No current metrics provided",
            )

        if trigger.metric_name is None:
            return TriggerResult(
                trigger_type=TriggerType.PERFORMANCE.value,
                fired=False,
                reason="No metric_name configured",
            )

        if trigger.metric_name not in current_metrics:
            return TriggerResult(
                trigger_type=TriggerType.PERFORMANCE.value,
                fired=False,
                reason=f"Metric '{trigger.metric_name}' not found in current metrics",
            )

        metric_value = current_metrics[trigger.metric_name]
        threshold = trigger.threshold or 0.1

        # For error metrics like MAPE, higher is worse
        if metric_value > threshold:
            return TriggerResult(
                trigger_type=TriggerType.PERFORMANCE.value,
                fired=True,
                reason=(
                    f"Metric '{trigger.metric_name}' value {metric_value:.4f} "
                    f"exceeded threshold {threshold}"
                ),
                metadata={
                    "metric_name": trigger.metric_name,
                    "metric_value": metric_value,
                    "threshold": threshold,
                },
            )
        else:
            return TriggerResult(
                trigger_type=TriggerType.PERFORMANCE.value,
                fired=False,
                reason=(
                    f"Metric '{trigger.metric_name}' value {metric_value:.4f} "
                    f"within threshold {threshold}"
                ),
                metadata={
                    "metric_name": trigger.metric_name,
                    "metric_value": metric_value,
                },
            )

    def _evaluate_manual_trigger(
        self,
        trigger: RetrainTrigger,
    ) -> TriggerResult:
        """Evaluate manual trigger.

        Manual triggers are always "no-op" unless explicitly set to fire.
        In practice, a manual trigger would be checked via an external signal.
        """
        return TriggerResult(
            trigger_type=TriggerType.MANUAL.value,
            fired=False,
            reason="Manual trigger not activated (use manual retrain API)",
        )

    def _parse_interval(self, interval_str: str) -> timedelta | None:
        """Parse interval string into timedelta.

        Args:
            interval_str: Interval string like "7d", "1h", "30m"

        Returns:
            timedelta or None if invalid
        """
        if not interval_str:
            return None

        try:
            # Extract number and unit
            num = int("".join(filter(str.isdigit, interval_str)))
            unit = "".join(filter(str.isalpha, interval_str)).lower()

            if unit in ("d", "day", "days"):
                return timedelta(days=num)
            elif unit in ("h", "hour", "hours"):
                return timedelta(hours=num)
            elif unit in ("m", "min", "minute", "minutes"):
                return timedelta(minutes=num)
            elif unit in ("w", "week", "weeks"):
                return timedelta(weeks=num)
            else:
                return None
        except (ValueError, TypeError):
            return None

    def should_retrain(
        self,
        drift_report: DriftReport | None = None,
        stability_report: StabilityReport | None = None,
        current_metrics: dict[str, float] | None = None,
        last_train_time: datetime | None = None,
    ) -> bool:
        """Check if any trigger indicates retraining is needed.

        Args:
            drift_report: Drift detection results
            stability_report: Stability metrics
            current_metrics: Current performance metrics
            last_train_time: Last training timestamp

        Returns:
            True if any trigger fired, False otherwise
        """
        results = self.evaluate(
            drift_report,
            stability_report,
            current_metrics,
            last_train_time,
        )
        return any(r.fired for r in results)

    def get_fired_triggers(
        self,
        drift_report: DriftReport | None = None,
        stability_report: StabilityReport | None = None,
        current_metrics: dict[str, float] | None = None,
        last_train_time: datetime | None = None,
    ) -> list[TriggerResult]:
        """Get only the triggers that fired.

        Args:
            drift_report: Drift detection results
            stability_report: Stability metrics
            current_metrics: Current performance metrics
            last_train_time: Last training timestamp

        Returns:
            List of TriggerResult for triggers that fired
        """
        results = self.evaluate(
            drift_report,
            stability_report,
            current_metrics,
            last_train_time,
        )
        return [r for r in results if r.fired]
