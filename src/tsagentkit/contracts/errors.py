"""Error codes and exceptions for tsagentkit.

Defines structured error codes used throughout the pipeline.
Aligned to docs/PRD.md Section 6.1 with compatibility aliases.
"""

from typing import Any


class TSAgentKitError(Exception):
    """Base exception for all tsagentkit errors.

    Attributes:
        error_code: Unique error code string for programmatic handling
        message: Human-readable error message
        context: Additional context data for debugging
    """

    error_code: str = "E_UNKNOWN"

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            return f"[{self.error_code}] {self.message} (context: {self.context})"
        return f"[{self.error_code}] {self.message}"


# ---------------------------
# Contract Errors
# ---------------------------

class EContractInvalid(TSAgentKitError):
    """Input schema/contract invalid."""
    error_code = "E_CONTRACT_INVALID"


class EContractMissingColumn(TSAgentKitError):
    """Input data is missing required columns."""
    error_code = "E_CONTRACT_MISSING_COLUMN"


class EContractInvalidType(TSAgentKitError):
    """Column has invalid data type."""
    error_code = "E_CONTRACT_INVALID_TYPE"


class EContractDuplicateKey(TSAgentKitError):
    """Duplicate (unique_id, ds) pairs found in data."""
    error_code = "E_CONTRACT_DUPLICATE_KEY"


class EFreqInferFail(TSAgentKitError):
    """Frequency cannot be inferred/validated."""
    error_code = "E_FREQ_INFER_FAIL"


class EDSNotMonotonic(TSAgentKitError):
    """Time index not monotonic per series."""
    error_code = "E_DS_NOT_MONOTONIC"


class ESplitRandomForbidden(TSAgentKitError):
    """Random train/test splits are strictly forbidden."""
    error_code = "E_SPLIT_RANDOM_FORBIDDEN"


class EContractUnsorted(EDSNotMonotonic):
    """Data is not sorted by (unique_id, ds)."""
    error_code = "E_DS_NOT_MONOTONIC"


# ---------------------------
# QA Errors
# ---------------------------

class EQAMinHistory(TSAgentKitError):
    """Series history too short."""
    error_code = "E_QA_MIN_HISTORY"


class EQARepairPeeksFuture(TSAgentKitError):
    """Repair strategy violates PIT safety."""
    error_code = "E_QA_REPAIR_PEEKS_FUTURE"


class EQACriticalIssue(TSAgentKitError):
    """Critical data quality issue detected in strict mode."""
    error_code = "E_QA_CRITICAL_ISSUE"


class EQALeakageDetected(TSAgentKitError):
    """Data leakage detected in strict mode."""
    error_code = "E_QA_LEAKAGE_DETECTED"


# ---------------------------
# Covariate Errors
# ---------------------------

class ECovariateLeakage(TSAgentKitError):
    """Past/observed covariate leaks into future."""
    error_code = "E_COVARIATE_LEAKAGE"


class ECovariateIncompleteKnown(TSAgentKitError):
    """Future-known covariate missing in horizon."""
    error_code = "E_COVARIATE_INCOMPLETE_KNOWN"


class ECovariateStaticInvalid(TSAgentKitError):
    """Static covariate invalid cardinality."""
    error_code = "E_COVARIATE_STATIC_INVALID"


# ---------------------------
# Model Errors
# ---------------------------

class EModelFitFailed(TSAgentKitError):
    """Model fitting failed."""
    error_code = "E_MODEL_FIT_FAIL"


class EModelPredictFailed(TSAgentKitError):
    """Model prediction failed."""
    error_code = "E_MODEL_PREDICT_FAIL"


class EModelLoadFailed(TSAgentKitError):
    """Model loading failed."""
    error_code = "E_MODEL_LOAD_FAILED"


class EAdapterNotAvailable(TSAgentKitError):
    """TSFM adapter not available."""
    error_code = "E_ADAPTER_NOT_AVAILABLE"


class EFallbackExhausted(TSAgentKitError):
    """All models in the fallback ladder failed."""
    error_code = "E_FALLBACK_EXHAUSTED"


class EOOM(TSAgentKitError):
    """Out-of-memory during fit/predict."""
    error_code = "E_OOM"


# ---------------------------
# Task Spec Errors
# ---------------------------

class ETaskSpecInvalid(TSAgentKitError):
    """Task specification is invalid or incomplete."""
    error_code = "E_TASK_SPEC_INVALID"


class ETaskSpecIncompatible(TSAgentKitError):
    """Task spec is incompatible with data."""
    error_code = "E_TASK_SPEC_INCOMPATIBLE"


# ---------------------------
# Backtest Errors
# ---------------------------

class EBacktestFail(TSAgentKitError):
    """Backtest execution failed."""
    error_code = "E_BACKTEST_FAIL"


class EBacktestInsufficientData(TSAgentKitError):
    """Not enough data for requested backtest windows."""
    error_code = "E_BACKTEST_INSUFFICIENT_DATA"


class EBacktestInvalidWindow(TSAgentKitError):
    """Invalid backtest window configuration."""
    error_code = "E_BACKTEST_INVALID_WINDOW"


# ---------------------------
# Calibration / Anomaly Errors
# ---------------------------

class ECalibrationFail(TSAgentKitError):
    """Calibration failed."""
    error_code = "E_CALIBRATION_FAIL"


class EAnomalyFail(TSAgentKitError):
    """Anomaly detection failed."""
    error_code = "E_ANOMALY_FAIL"


# Registry for lookup
ERROR_REGISTRY: dict[str, type[TSAgentKitError]] = {
    "E_CONTRACT_INVALID": EContractInvalid,
    "E_CONTRACT_MISSING_COLUMN": EContractMissingColumn,
    "E_CONTRACT_INVALID_TYPE": EContractInvalidType,
    "E_CONTRACT_DUPLICATE_KEY": EContractDuplicateKey,
    "E_FREQ_INFER_FAIL": EFreqInferFail,
    "E_DS_NOT_MONOTONIC": EDSNotMonotonic,
    "E_SPLIT_RANDOM_FORBIDDEN": ESplitRandomForbidden,
    "E_CONTRACT_UNSORTED": EContractUnsorted,
    "E_QA_MIN_HISTORY": EQAMinHistory,
    "E_QA_REPAIR_PEEKS_FUTURE": EQARepairPeeksFuture,
    "E_QA_CRITICAL_ISSUE": EQACriticalIssue,
    "E_QA_LEAKAGE_DETECTED": EQALeakageDetected,
    "E_COVARIATE_LEAKAGE": ECovariateLeakage,
    "E_COVARIATE_INCOMPLETE_KNOWN": ECovariateIncompleteKnown,
    "E_COVARIATE_STATIC_INVALID": ECovariateStaticInvalid,
    "E_MODEL_FIT_FAIL": EModelFitFailed,
    "E_MODEL_PREDICT_FAIL": EModelPredictFailed,
    "E_MODEL_LOAD_FAILED": EModelLoadFailed,
    "E_ADAPTER_NOT_AVAILABLE": EAdapterNotAvailable,
    "E_FALLBACK_EXHAUSTED": EFallbackExhausted,
    "E_OOM": EOOM,
    "E_TASK_SPEC_INVALID": ETaskSpecInvalid,
    "E_TASK_SPEC_INCOMPATIBLE": ETaskSpecIncompatible,
    "E_BACKTEST_FAIL": EBacktestFail,
    "E_BACKTEST_INSUFFICIENT_DATA": EBacktestInsufficientData,
    "E_BACKTEST_INVALID_WINDOW": EBacktestInvalidWindow,
    "E_CALIBRATION_FAIL": ECalibrationFail,
    "E_ANOMALY_FAIL": EAnomalyFail,
}


def get_error_class(error_code: str) -> type[TSAgentKitError]:
    """Get error class by error code."""
    return ERROR_REGISTRY.get(error_code, TSAgentKitError)


__all__ = [
    "TSAgentKitError",
    "EContractInvalid",
    "EContractMissingColumn",
    "EContractInvalidType",
    "EContractDuplicateKey",
    "EFreqInferFail",
    "EDSNotMonotonic",
    "ESplitRandomForbidden",
    "EContractUnsorted",
    "EQAMinHistory",
    "EQARepairPeeksFuture",
    "EQACriticalIssue",
    "EQALeakageDetected",
    "ECovariateLeakage",
    "ECovariateIncompleteKnown",
    "ECovariateStaticInvalid",
    "EModelFitFailed",
    "EModelPredictFailed",
    "EModelLoadFailed",
    "EAdapterNotAvailable",
    "EFallbackExhausted",
    "EOOM",
    "ETaskSpecInvalid",
    "ETaskSpecIncompatible",
    "EBacktestFail",
    "EBacktestInsufficientData",
    "EBacktestInvalidWindow",
    "ECalibrationFail",
    "EAnomalyFail",
    "get_error_class",
]
