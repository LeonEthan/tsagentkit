"""Alert conditions and triggering for monitoring.

Provides configurable alert conditions for coverage, drift,
and model performance monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable


@dataclass(frozen=True)
class AlertCondition:
    """Alert condition configuration.

    Defines when an alert should be triggered based on a metric
    and threshold.

    Attributes:
        name: Alert name/identifier
        metric: Metric to monitor (e.g., "coverage", "drift_score")
        operator: Comparison operator ("lt", "gt", "eq", "ne")
        threshold: Threshold value for triggering
        severity: Alert severity ("info", "warning", "critical")
        message: Optional custom alert message
    """

    name: str
    metric: str
    operator: str  # "lt", "gt", "eq", "ne"
    threshold: float
    severity: str = "warning"
    message: str | None = None

    def evaluate(self, value: float) -> bool:
        """Evaluate if condition is met.

        Args:
            value: Current metric value

        Returns:
            True if alert should trigger
        """
        if self.operator == "lt":
            return value < self.threshold
        if self.operator == "gt":
            return value > self.threshold
        if self.operator == "eq":
            return value == self.threshold
        if self.operator == "ne":
            return value != self.threshold
        if self.operator == "lte":
            return value <= self.threshold
        if self.operator == "gte":
            return value >= self.threshold
        return False

    def format_message(self, value: float, context: dict[str, Any] | None = None) -> str:
        """Format alert message with current value.

        Args:
            value: Current metric value
            context: Additional context for message formatting

        Returns:
            Formatted alert message
        """
        if self.message:
            ctx = context or {}
            return self.message.format(
                name=self.name,
                metric=self.metric,
                value=value,
                threshold=self.threshold,
                **ctx,
            )

        op_str = {
            "lt": "below",
            "gt": "above",
            "eq": "equal to",
            "ne": "not equal to",
            "lte": "at or below",
            "gte": "at or above",
        }.get(self.operator, self.operator)

        return f"Alert '{self.name}': {self.metric} ({value:.4f}) is {op_str} threshold ({self.threshold:.4f})"


@dataclass
class Alert:
    """Triggered alert instance.

    Represents an alert that has been triggered with full context.

    Attributes:
        condition: The alert condition that triggered
        value: The metric value that triggered the alert
        timestamp: ISO 8601 timestamp of when alert triggered
        context: Additional context about the alert
    """

    condition: AlertCondition
    value: float
    timestamp: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.condition.name,
            "metric": self.condition.metric,
            "severity": self.condition.severity,
            "value": self.value,
            "threshold": self.condition.threshold,
            "operator": self.condition.operator,
            "timestamp": self.timestamp,
            "message": self.condition.format_message(self.value, self.context),
            "context": self.context,
        }


class AlertManager:
    """Manager for alert conditions and triggering.

    Provides a centralized way to define alert conditions and
check them against current metrics.

    Example:
        >>> manager = AlertManager()
        >>> manager.add_condition(AlertCondition(
        ...     name="low_coverage",
        ...     metric="coverage_80",
        ...     operator="lt",
        ...     threshold=0.75,
        ...     severity="critical",
        ... ))
        >>> alerts = manager.check_metrics({"coverage_80": 0.70})
        >>> print(len(alerts))
        1
    """

    def __init__(self) -> None:
        """Initialize alert manager."""
        self.conditions: list[AlertCondition] = []
        self._alert_history: list[Alert] = []

    def add_condition(self, condition: AlertCondition) -> None:
        """Add an alert condition.

        Args:
            condition: Alert condition to add
        """
        self.conditions.append(condition)

    def remove_condition(self, name: str) -> bool:
        """Remove an alert condition by name.

        Args:
            name: Name of condition to remove

        Returns:
            True if condition was found and removed
        """
        for i, cond in enumerate(self.conditions):
            if cond.name == name:
                self.conditions.pop(i)
                return True
        return False

    def check_metrics(
        self,
        metrics: dict[str, float],
        context: dict[str, Any] | None = None,
    ) -> list[Alert]:
        """Check all conditions against current metrics.

        Args:
            metrics: Dictionary of metric names to values
            context: Additional context for alert messages

        Returns:
            List of triggered alerts
        """
        from datetime import datetime, timezone

        triggered: list[Alert] = []

        for condition in self.conditions:
            if condition.metric not in metrics:
                continue

            value = metrics[condition.metric]
            if condition.evaluate(value):
                alert = Alert(
                    condition=condition,
                    value=value,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    context=context or {},
                )
                triggered.append(alert)
                self._alert_history.append(alert)

        return triggered

    def check_coverage(
        self,
        coverage_checks: list[Any],
        context: dict[str, Any] | None = None,
    ) -> list[Alert]:
        """Check coverage results against conditions.

        Args:
            coverage_checks: List of CoverageCheck objects
            context: Additional context

        Returns:
            List of triggered alerts
        """
        metrics: dict[str, float] = {}
        for check in coverage_checks:
            if hasattr(check, "is_acceptable") and hasattr(check, "actual_coverage"):
                metric_name = f"coverage_{check.expected_coverage:.0%}"
                metrics[metric_name] = check.actual_coverage

        return self.check_metrics(metrics, context)

    def get_alert_history(
        self,
        severity: str | None = None,
    ) -> list[Alert]:
        """Get history of triggered alerts.

        Args:
            severity: Filter by severity (optional)

        Returns:
            List of historical alerts
        """
        if severity is None:
            return self._alert_history.copy()
        return [a for a in self._alert_history if a.condition.severity == severity]

    def clear_history(self) -> None:
        """Clear alert history."""
        self._alert_history.clear()


def create_default_coverage_alerts(
    coverage_levels: list[float] | None = None,
    tolerance: float = 0.05,
) -> list[AlertCondition]:
    """Create default alert conditions for coverage monitoring.

    Args:
        coverage_levels: Coverage levels to monitor (default: [0.5, 0.8, 0.95])
        tolerance: Tolerance for coverage deviation

    Returns:
        List of alert conditions
    """
    levels = coverage_levels or [0.5, 0.8, 0.95]
    conditions: list[AlertCondition] = []

    for level in levels:
        conditions.append(
            AlertCondition(
                name=f"low_coverage_{level:.0%}",
                metric=f"coverage_{level:.0%}",
                operator="lt",
                threshold=level - tolerance,
                severity="warning" if level < 0.9 else "critical",
                message=f"Coverage for {level:.0%} interval is below acceptable threshold",
            )
        )

    return conditions


def create_default_drift_alerts(
    drift_threshold: float = 0.05,
) -> list[AlertCondition]:
    """Create default alert conditions for drift monitoring.

    Args:
        drift_threshold: Drift score threshold

    Returns:
        List of alert conditions
    """
    return [
        AlertCondition(
            name="drift_detected",
            metric="drift_score",
            operator="gt",
            threshold=drift_threshold,
            severity="warning",
            message="Data drift detected above threshold",
        ),
    ]
