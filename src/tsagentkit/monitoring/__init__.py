"""Monitoring module for drift detection, coverage, and model stability.

Provides utilities for detecting data drift, monitoring prediction stability,
checking quantile coverage, and triggering alerts when necessary.
"""

from __future__ import annotations

from tsagentkit.monitoring.alerts import (
    Alert,
    AlertCondition,
    AlertManager,
    create_default_coverage_alerts,
    create_default_drift_alerts,
)
from tsagentkit.monitoring.coverage import CoverageCheck, CoverageMonitor
from tsagentkit.monitoring.drift import DriftDetector
from tsagentkit.monitoring.report import (
    CalibrationReport,
    DriftReport,
    FeatureDriftResult,
    StabilityReport,
    TriggerResult,
)
from tsagentkit.monitoring.stability import StabilityMonitor
from tsagentkit.monitoring.triggers import RetrainTrigger, TriggerEvaluator, TriggerType

__all__ = [
    # Coverage monitoring
    "CoverageMonitor",
    "CoverageCheck",
    # Alerts
    "AlertManager",
    "AlertCondition",
    "Alert",
    "create_default_coverage_alerts",
    "create_default_drift_alerts",
    # Drift detection
    "DriftDetector",
    "DriftReport",
    "FeatureDriftResult",
    # Stability monitoring
    "StabilityMonitor",
    "StabilityReport",
    "CalibrationReport",
    # Retrain triggers
    "TriggerEvaluator",
    "RetrainTrigger",
    "TriggerType",
    "TriggerResult",
]
