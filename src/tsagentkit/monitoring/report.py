"""Report dataclasses for monitoring results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


@dataclass(frozen=True)
class FeatureDriftResult:
    """Drift result for a single feature.

    Attributes:
        feature_name: Name of the feature analyzed
        metric: Drift metric used ("psi" or "ks")
        statistic: Drift statistic value
        p_value: P-value for statistical test (KS only)
        drift_detected: Whether drift was detected for this feature
        reference_distribution: Summary of reference distribution
        current_distribution: Summary of current distribution

    Example:
        >>> result = FeatureDriftResult(
        ...     feature_name="sales",
        ...     metric="psi",
        ...     statistic=0.25,
        ...     p_value=None,
        ...     drift_detected=True,
        ...     reference_distribution={"mean": 100.0, "std": 15.0},
        ...     current_distribution={"mean": 120.0, "std": 20.0},
        ... )
        >>> print(result)
        FeatureDriftResult(sales, psi=0.250, drift=True)
    """

    feature_name: str
    metric: str
    statistic: float
    p_value: float | None
    drift_detected: bool
    reference_distribution: dict
    current_distribution: dict

    def __repr__(self) -> str:
        return (
            f"FeatureDriftResult({self.feature_name}, "
            f"{self.metric}={self.statistic:.3f}, "
            f"drift={self.drift_detected})"
        )


@dataclass(frozen=True)
class DriftReport:
    """Report from drift detection analysis.

    Attributes:
        drift_detected: Whether any drift was detected
        feature_drifts: Dict mapping feature names to drift results
        overall_drift_score: Aggregated drift score across all features
        threshold_used: Threshold used for drift detection
        reference_timestamp: Timestamp of reference data
        current_timestamp: Timestamp of current data

    Example:
        >>> report = DriftReport(
        ...     drift_detected=True,
        ...     feature_drifts={"sales": feature_result},
        ...     overall_drift_score=0.25,
        ...     threshold_used=0.2,
        ... )
        >>> print(report.summary())
        Drift detected in 1/1 features. Overall score: 0.250
    """

    drift_detected: bool
    feature_drifts: dict[str, FeatureDriftResult]
    overall_drift_score: float
    threshold_used: float
    reference_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    current_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def summary(self) -> str:
        """Generate a human-readable summary of the drift report."""
        n_drifting = sum(1 for r in self.feature_drifts.values() if r.drift_detected)
        n_total = len(self.feature_drifts)
        return (
            f"Drift detected in {n_drifting}/{n_total} features. "
            f"Overall score: {self.overall_drift_score:.3f} "
            f"(threshold: {self.threshold_used})"
        )

    def get_drifting_features(self) -> list[str]:
        """Return list of feature names with detected drift."""
        return [
            name for name, result in self.feature_drifts.items()
            if result.drift_detected
        ]


@dataclass(frozen=True)
class CalibrationReport:
    """Report on quantile calibration.

    Attributes:
        target_quantiles: List of target quantile levels
        empirical_coverage: Dict mapping quantile to empirical coverage
        calibration_errors: Dict mapping quantile to calibration error
        well_calibrated: Whether all quantiles are well-calibrated
        tolerance: Tolerance used for calibration check

    Example:
        >>> report = CalibrationReport(
        ...     target_quantiles=[0.1, 0.5, 0.9],
        ...     empirical_coverage={0.1: 0.08, 0.5: 0.52, 0.9: 0.91},
        ...     calibration_errors={0.1: 0.02, 0.5: 0.02, 0.9: 0.01},
        ...     well_calibrated=True,
        ...     tolerance=0.05,
        ... )
    """

    target_quantiles: list[float]
    empirical_coverage: dict[float, float]
    calibration_errors: dict[float, float]
    well_calibrated: bool
    tolerance: float

    def summary(self) -> str:
        """Generate a human-readable summary."""
        errors_str = ", ".join(
            f"q={q:.2f}: err={e:.3f}"
            for q, e in self.calibration_errors.items()
        )
        status = "well-calibrated" if self.well_calibrated else "poorly-calibrated"
        return f"Calibration ({status}): {errors_str}"


@dataclass(frozen=True)
class StabilityReport:
    """Report on prediction stability.

    Attributes:
        jitter_metrics: Dict mapping series_id to jitter metric
        overall_jitter: Aggregate jitter across all series
        jitter_threshold: Threshold used for jitter evaluation
        high_jitter_series: List of series with high jitter
        coverage_report: Optional calibration report for quantiles

    Example:
        >>> report = StabilityReport(
        ...     jitter_metrics={"A": 0.05, "B": 0.15},
        ...     overall_jitter=0.10,
        ...     jitter_threshold=0.10,
        ...     high_jitter_series=["B"],
        ... )
        >>> print(report.is_stable)
        False
    """

    jitter_metrics: dict[str, float]
    overall_jitter: float
    jitter_threshold: float
    high_jitter_series: list[str]
    coverage_report: CalibrationReport | None = None

    @property
    def is_stable(self) -> bool:
        """Whether predictions are considered stable."""
        return len(self.high_jitter_series) == 0

    def summary(self) -> str:
        """Generate a human-readable summary."""
        status = "stable" if self.is_stable else "unstable"
        n_high = len(self.high_jitter_series)
        return (
            f"Stability ({status}): overall_jitter={self.overall_jitter:.3f}, "
            f"{n_high} series exceed threshold"
        )


@dataclass(frozen=True)
class TriggerResult:
    """Result of a retrain trigger evaluation.

    Attributes:
        trigger_type: Type of trigger that was evaluated
        fired: Whether the trigger fired
        reason: Human-readable reason for trigger firing or not
        timestamp: When the trigger was evaluated
        metadata: Additional trigger-specific metadata

    Example:
        >>> result = TriggerResult(
        ...     trigger_type=TriggerType.DRIFT,
        ...     fired=True,
        ...     reason="PSI drift score 0.25 exceeded threshold 0.2",
        ...     metadata={"psi_score": 0.25, "threshold": 0.2},
        ... )
    """

    trigger_type: str
    fired: bool
    reason: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "FIRED" if self.fired else "no-op"
        return f"TriggerResult({self.trigger_type}, {status})"
