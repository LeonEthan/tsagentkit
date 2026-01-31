"""Monitoring module for drift detection and model stability.

Provides utilities for detecting data drift, monitoring prediction stability,
and triggering retraining when necessary.
"""

from __future__ import annotations

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
