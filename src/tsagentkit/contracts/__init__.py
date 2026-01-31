"""Contracts module for tsagentkit.

Provides data validation, task specifications, and result structures
for the forecasting pipeline.
"""

from .errors import (
    EBacktestInsufficientData,
    EBacktestInvalidWindow,
    EContractDuplicateKey,
    EContractInvalidFrequency,
    EContractInvalidType,
    EContractMissingColumn,
    EContractUnsorted,
    ECovariateLeakage,
    EFallbackExhausted,
    EModelFitFailed,
    EModelPredictFailed,
    EQACriticalIssue,
    EQALeakageDetected,
    ESplitRandomForbidden,
    ETaskSpecIncompatible,
    ETaskSpecInvalid,
    TSAgentKitError,
    get_error_class,
)
from .results import (
    ForecastResult,
    ModelArtifact,
    Provenance,
    RunArtifact,
    ValidationReport,
)
from .schema import validate_contract
from .task_spec import TaskSpec

__all__ = [
    # Errors
    "TSAgentKitError",
    "EContractMissingColumn",
    "EContractInvalidType",
    "EContractDuplicateKey",
    "EContractUnsorted",
    "EContractInvalidFrequency",
    "ESplitRandomForbidden",
    "ECovariateLeakage",
    "EModelFitFailed",
    "EModelPredictFailed",
    "EFallbackExhausted",
    "EQACriticalIssue",
    "EQALeakageDetected",
    "ETaskSpecInvalid",
    "ETaskSpecIncompatible",
    "EBacktestInsufficientData",
    "EBacktestInvalidWindow",
    "get_error_class",
    # Results
    "ForecastResult",
    "ModelArtifact",
    "Provenance",
    "RunArtifact",
    "ValidationReport",
    # Schema
    "validate_contract",
    # Task Spec
    "TaskSpec",
]
