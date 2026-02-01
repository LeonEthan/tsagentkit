"""Error codes and exceptions for tsagentkit.

This module defines all structured error codes used throughout the pipeline
for consistent error handling and debugging.
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


# Contract Errors

class EContractMissingColumn(TSAgentKitError):
    """Input data is missing required columns.

    Required columns: 'unique_id' (str), 'ds' (datetime), 'y' (numeric)
    """
    error_code = "E_CONTRACT_MISSING_COLUMN"


class EContractInvalidType(TSAgentKitError):
    """Column has invalid data type."""
    error_code = "E_CONTRACT_INVALID_TYPE"


class EContractDuplicateKey(TSAgentKitError):
    """Duplicate (unique_id, ds) pairs found in data."""
    error_code = "E_CONTRACT_DUPLICATE_KEY"


class EContractUnsorted(TSAgentKitError):
    """Data is not sorted by (unique_id, ds)."""
    error_code = "E_CONTRACT_UNSORTED"


class EContractInvalidFrequency(TSAgentKitError):
    """Could not infer or invalid frequency."""
    error_code = "E_CONTRACT_INVALID_FREQUENCY"


# Guardrail Errors

class ESplitRandomForbidden(TSAgentKitError):
    """Random train/test splits are strictly forbidden.

    Time series data must use temporal splits only.
    """
    error_code = "E_SPLIT_RANDOM_FORBIDDEN"


class ECovariateLeakage(TSAgentKitError):
    """Future covariate leakage detected.

    Observed covariates cannot be used for future horizons.
    """
    error_code = "E_COVARIATE_LEAKAGE"


# Model Errors

class EModelFitFailed(TSAgentKitError):
    """Model training failed.

    This error triggers the fallback ladder.
    """
    error_code = "E_MODEL_FIT_FAILED"


class EModelPredictFailed(TSAgentKitError):
    """Model prediction failed."""
    error_code = "E_MODEL_PREDICT_FAILED"


class EModelLoadFailed(TSAgentKitError):
    """Model loading failed.

    Typically occurs when TSFM adapter cannot load the underlying model.
    """
    error_code = "E_MODEL_LOAD_FAILED"


class EAdapterNotAvailable(TSAgentKitError):
    """TSFM adapter not available.

    The required package for the TSFM adapter is not installed.
    """
    error_code = "E_ADAPTER_NOT_AVAILABLE"


class EFallbackExhausted(TSAgentKitError):
    """All models in the fallback ladder failed."""
    error_code = "E_FALLBACK_EXHAUSTED"


# QA Errors

class EQACriticalIssue(TSAgentKitError):
    """Critical data quality issue detected in strict mode."""
    error_code = "E_QA_CRITICAL_ISSUE"


class EQALeakageDetected(TSAgentKitError):
    """Data leakage detected in strict mode."""
    error_code = "E_QA_LEAKAGE_DETECTED"


# Task Spec Errors

class ETaskSpecInvalid(TSAgentKitError):
    """Task specification is invalid or incomplete."""
    error_code = "E_TASK_SPEC_INVALID"


class ETaskSpecIncompatible(TSAgentKitError):
    """Task spec is incompatible with data (e.g., horizon too long)."""
    error_code = "E_TASK_SPEC_INCOMPATIBLE"


# Backtest Errors

class EBacktestInsufficientData(TSAgentKitError):
    """Not enough data for requested backtest windows."""
    error_code = "E_BACKTEST_INSUFFICIENT_DATA"


class EBacktestInvalidWindow(TSAgentKitError):
    """Invalid backtest window configuration."""
    error_code = "E_BACKTEST_INVALID_WINDOW"


# Registry for lookup
ERROR_REGISTRY: dict[str, type[TSAgentKitError]] = {
    "E_CONTRACT_MISSING_COLUMN": EContractMissingColumn,
    "E_CONTRACT_INVALID_TYPE": EContractInvalidType,
    "E_CONTRACT_DUPLICATE_KEY": EContractDuplicateKey,
    "E_CONTRACT_UNSORTED": EContractUnsorted,
    "E_CONTRACT_INVALID_FREQUENCY": EContractInvalidFrequency,
    "E_SPLIT_RANDOM_FORBIDDEN": ESplitRandomForbidden,
    "E_COVARIATE_LEAKAGE": ECovariateLeakage,
    "E_MODEL_FIT_FAILED": EModelFitFailed,
    "E_MODEL_PREDICT_FAILED": EModelPredictFailed,
    "E_MODEL_LOAD_FAILED": EModelLoadFailed,
    "E_ADAPTER_NOT_AVAILABLE": EAdapterNotAvailable,
    "E_FALLBACK_EXHAUSTED": EFallbackExhausted,
    "E_QA_CRITICAL_ISSUE": EQACriticalIssue,
    "E_QA_LEAKAGE_DETECTED": EQALeakageDetected,
    "E_TASK_SPEC_INVALID": ETaskSpecInvalid,
    "E_TASK_SPEC_INCOMPATIBLE": ETaskSpecIncompatible,
    "E_BACKTEST_INSUFFICIENT_DATA": EBacktestInsufficientData,
    "E_BACKTEST_INVALID_WINDOW": EBacktestInvalidWindow,
}


def get_error_class(error_code: str) -> type[TSAgentKitError]:
    """Get error class by error code.

    Args:
        error_code: The error code string

    Returns:
        The corresponding error class, or TSAgentKitError if not found
    """
    return ERROR_REGISTRY.get(error_code, TSAgentKitError)
