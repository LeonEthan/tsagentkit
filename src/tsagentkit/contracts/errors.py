"""Error codes and exceptions for tsagentkit.

Defines structured error codes used throughout the pipeline.
Aligned to docs/PRD.md Section 6.1 with compatibility aliases.
"""

# ruff: noqa: N818

from typing import Any


class TSAgentKitError(Exception):
    """Base exception for all tsagentkit errors.

    Attributes:
        error_code: Unique error code string for programmatic handling
        message: Human-readable error message
        context: Additional context data for debugging
        fix_hint: Actionable hint for resolving the error (agent-friendly)
    """

    error_code: str = "E_UNKNOWN"
    fix_hint: str = ""

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        fix_hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}
        if fix_hint is not None:
            self.fix_hint = fix_hint

    def __str__(self) -> str:
        parts = [f"[{self.error_code}] {self.message}"]
        if self.context:
            parts.append(f"(context: {self.context})")
        if self.fix_hint:
            parts.append(f"[hint: {self.fix_hint}]")
        return " ".join(parts)

    def to_agent_dict(self) -> dict[str, Any]:
        """Return a structured dict suitable for agent consumption.

        Returns:
            Dictionary with error_code, message, fix_hint, and context.
        """
        return {
            "error_code": self.error_code,
            "message": self.message,
            "fix_hint": self.fix_hint,
            "context": self.context,
        }


# ---------------------------
# Contract Errors
# ---------------------------


class EContractInvalid(TSAgentKitError):
    """Input schema/contract invalid."""

    error_code = "E_CONTRACT_INVALID"


class EContractMissingColumn(TSAgentKitError):
    """Input data is missing required columns."""

    error_code = "E_CONTRACT_MISSING_COLUMN"
    fix_hint = "Ensure DataFrame contains required columns ('unique_id', 'ds', 'y'). Use df.rename(columns={...}) to map."


class EContractInvalidType(TSAgentKitError):
    """Column has invalid data type."""

    error_code = "E_CONTRACT_INVALID_TYPE"


class EContractDuplicateKey(TSAgentKitError):
    """Duplicate (unique_id, ds) pairs found in data."""

    error_code = "E_CONTRACT_DUPLICATE_KEY"
    fix_hint = "Remove duplicates: df = df.drop_duplicates(subset=['unique_id', 'ds'], keep='last')"


class EFreqInferFail(TSAgentKitError):
    """Frequency cannot be inferred/validated."""

    error_code = "E_FREQ_INFER_FAIL"
    fix_hint = (
        "Specify freq explicitly in TaskSpec(freq='D'), or ensure regular time intervals in data."
    )


class EDSNotMonotonic(TSAgentKitError):
    """Time index not monotonic per series."""

    error_code = "E_DS_NOT_MONOTONIC"
    fix_hint = (
        "Sort your DataFrame: df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)"
    )


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
    fix_hint = "Provide more historical data or lower backtest.min_train_size in TaskSpec."


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
    fix_hint = "Mark past-only covariates with role='past', or use align_covariates() for automatic alignment."


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


class EModelNotLoaded(TSAgentKitError):
    """Model operation attempted before explicit load_model()."""

    error_code = "E_MODEL_NOT_LOADED"
    fix_hint = "Preload adapter in ModelPool or call adapter.load_model() during init stage."


class EAdapterNotAvailable(TSAgentKitError):
    """TSFM adapter not available."""

    error_code = "E_ADAPTER_NOT_AVAILABLE"


class ETSFMRequiredUnavailable(TSAgentKitError):
    """TSFM is required by policy but no required adapter is available."""

    error_code = "E_TSFM_REQUIRED_UNAVAILABLE"
    fix_hint = "Install TSFM adapters: pip install tsagentkit[tsfm], or set tsfm_policy={'mode': 'preferred'} to allow fallback."


class EFallbackExhausted(TSAgentKitError):
    """All models in the fallback ladder failed."""

    error_code = "E_FALLBACK_EXHAUSTED"
    fix_hint = "Verify data has enough observations (>=2 per series), or relax router_thresholds."


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
# Artifact Lifecycle Errors
# ---------------------------


class EArtifactSchemaIncompatible(TSAgentKitError):
    """Serialized artifact schema/type is incompatible."""

    error_code = "E_ARTIFACT_SCHEMA_INCOMPATIBLE"


class EArtifactLoadFailed(TSAgentKitError):
    """Serialized artifact cannot be loaded safely."""

    error_code = "E_ARTIFACT_LOAD_FAILED"


# ---------------------------
# Backtest Errors
# ---------------------------


class EBacktestFail(TSAgentKitError):
    """Backtest execution failed."""

    error_code = "E_BACKTEST_FAIL"


class EBacktestInsufficientData(TSAgentKitError):
    """Not enough data for requested backtest windows."""

    error_code = "E_BACKTEST_INSUFFICIENT_DATA"
    fix_hint = "Reduce backtest.n_windows or backtest.h, or provide more historical data."


class EBacktestInvalidWindow(TSAgentKitError):
    """Invalid backtest window configuration."""

    error_code = "E_BACKTEST_INVALID_WINDOW"


# ---------------------------
# Calibration / Anomaly Errors
# ---------------------------


class ECalibrationFail(TSAgentKitError):
    """Calibration failed."""

    error_code = "E_CALIBRATION_FAIL"
    fix_hint = (
        "Ensure cross-validation output contains 'y' and 'yhat' columns with sufficient data."
    )


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
    "E_MODEL_NOT_LOADED": EModelNotLoaded,
    "E_ADAPTER_NOT_AVAILABLE": EAdapterNotAvailable,
    "E_TSFM_REQUIRED_UNAVAILABLE": ETSFMRequiredUnavailable,
    "E_FALLBACK_EXHAUSTED": EFallbackExhausted,
    "E_OOM": EOOM,
    "E_TASK_SPEC_INVALID": ETaskSpecInvalid,
    "E_TASK_SPEC_INCOMPATIBLE": ETaskSpecIncompatible,
    "E_ARTIFACT_SCHEMA_INCOMPATIBLE": EArtifactSchemaIncompatible,
    "E_ARTIFACT_LOAD_FAILED": EArtifactLoadFailed,
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
    "EModelNotLoaded",
    "EAdapterNotAvailable",
    "ETSFMRequiredUnavailable",
    "EFallbackExhausted",
    "EOOM",
    "ETaskSpecInvalid",
    "ETaskSpecIncompatible",
    "EArtifactSchemaIncompatible",
    "EArtifactLoadFailed",
    "EBacktestFail",
    "EBacktestInsufficientData",
    "EBacktestInvalidWindow",
    "ECalibrationFail",
    "EAnomalyFail",
    "get_error_class",
]
