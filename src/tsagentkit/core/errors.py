"""Core error types for tsagentkit.

Minimal error hierarchy with 4 core types covering 99% of cases.
Each error includes a fix hint for rapid resolution.
"""

from __future__ import annotations

from typing import Any


class TSAgentKitError(Exception):
    """Base exception with rich context."""

    code: str = "E_UNKNOWN"
    hint: str = ""

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        fix_hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}
        if fix_hint:
            self.hint = fix_hint

    def __str__(self) -> str:
        parts = [f"[{self.code}] {self.message}"]
        if self.context:
            parts.append(f"(context: {self.context})")
        if self.hint:
            parts.append(f"[hint: {self.hint}]")
        return " ".join(parts)

    @property
    def error_code(self) -> str:
        """Backward compatibility alias for code."""
        return self.code

    @property
    def fix_hint(self) -> str:
        """Backward compatibility alias for hint."""
        return self.hint


class EContract(TSAgentKitError):
    """Input data invalid (wrong columns, types, etc.)"""

    code = "E_CONTRACT"
    error_code = "E_CONTRACT"  # type: ignore[misc]
    hint = "Check data format: DataFrame must have [unique_id, ds, y] columns"


class ENoTSFM(TSAgentKitError):
    """No TSFM adapters available."""

    code = "E_NO_TSFM"
    error_code = "E_NO_TSFM"  # type: ignore[misc]
    hint = "Install TSFM adapters (chronos, moirai, timesfm) or set tsfm_mode='preferred'"


class EInsufficient(TSAgentKitError):
    """Not enough TSFMs succeeded."""

    code = "E_INSUFFICIENT"
    error_code = "E_INSUFFICIENT"  # type: ignore[misc]
    hint = "Check model compatibility with data frequency and length"


class ETemporal(TSAgentKitError):
    """Temporal integrity violation."""

    code = "E_TEMPORAL"
    error_code = "E_TEMPORAL"  # type: ignore[misc]
    hint = "Data must be sorted by ds. No future dates in covariates."


# Legacy subclasses for backward compatibility (with distinct error codes)
class EContractViolation(TSAgentKitError):
    """Legacy: Input data violates contract requirements."""
    code = "E_CONTRACT_VIOLATION"
    error_code = "E_CONTRACT_VIOLATION"  # type: ignore[misc]
    hint = "Check data format: DataFrame must have [unique_id, ds, y] columns"


class EDataQuality(TSAgentKitError):
    """Legacy: Data quality issue detected."""
    code = "E_DATA_QUALITY"
    error_code = "E_DATA_QUALITY"  # type: ignore[misc]
    hint = "Run diagnose() to identify specific quality issues"


class EModelFailed(TSAgentKitError):
    """Legacy: Model fitting or prediction failed."""
    code = "E_MODEL_FAILED"
    error_code = "E_MODEL_FAILED"  # type: ignore[misc]
    hint = "Check model compatibility with data frequency and length"


class ETSFMRequired(TSAgentKitError):
    """Legacy: TSFM required but unavailable."""
    code = "E_TSFM_REQUIRED"
    error_code = "E_TSFM_REQUIRED"  # type: ignore[misc]
    hint = "Install TSFM adapters (chronos, moirai, timesfm) or set tsfm_mode='preferred'"


# Additional legacy aliases (for test compatibility)
EContractMissingColumn = EContractViolation
EContractInvalidType = EContractViolation
EContractDuplicateKey = EContractViolation
EFreqInferFail = EContractViolation
EDSNotMonotonic = EContractViolation
ESplitRandomForbidden = EContractViolation
EContractUnsorted = EContractViolation
EQAMinHistory = EDataQuality
EQACriticalIssue = EDataQuality
ECovariateLeakage = EDataQuality
EQARepairPeeksFuture = EDataQuality
EQALeakageDetected = EDataQuality
EModelFitFailed = EModelFailed
EModelPredictFailed = EModelFailed
EFallbackExhausted = EModelFailed
EAdapterNotAvailable = EModelFailed
EModelNotLoaded = EModelFailed
EModelLoadFailed = EModelFailed
ETSFMRequiredUnavailable = ETSFMRequired
ETaskSpecInvalid = EContractViolation
ETaskSpecIncompatible = EContractViolation
EArtifactSchemaIncompatible = EContractViolation
EArtifactLoadFailed = EContractViolation
EBacktestFail = EModelFailed
EBacktestInsufficientData = EDataQuality
EBacktestInvalidWindow = EContractViolation
ECalibrationFail = EModelFailed
EAnomalyFail = EModelFailed
EOOM = EModelFailed
ECovariateIncompleteKnown = EDataQuality
ECovariateStaticInvalid = EDataQuality

# Error registry for lookup (backward compatibility)
ERROR_REGISTRY: dict[str, type[TSAgentKitError]] = {
    "E_CONTRACT_VIOLATION": EContractViolation,
    "E_DATA_QUALITY": EDataQuality,
    "E_MODEL_FAILED": EModelFailed,
    "E_TSFM_REQUIRED": ETSFMRequired,
    "E_CONTRACT": EContract,
    "E_NO_TSFM": ENoTSFM,
    "E_INSUFFICIENT": EInsufficient,
    "E_TEMPORAL": ETemporal,
}


def get_error_class(error_code: str) -> type[TSAgentKitError]:
    """Get error class by code."""
    return ERROR_REGISTRY.get(error_code, TSAgentKitError)


__all__ = [
    "TSAgentKitError",
    "EContract",
    "ENoTSFM",
    "EInsufficient",
    "ETemporal",
    # Legacy aliases
    "EContractViolation",
    "EDataQuality",
    "EModelFailed",
    "ETSFMRequired",
    # Error registry
    "ERROR_REGISTRY",
    "get_error_class",
    # Additional legacy aliases
    "EContractMissingColumn",
    "EContractInvalidType",
    "EContractDuplicateKey",
    "EFreqInferFail",
    "EDSNotMonotonic",
    "ESplitRandomForbidden",
    "EContractUnsorted",
    "EQAMinHistory",
    "EQACriticalIssue",
    "ECovariateLeakage",
    "EQARepairPeeksFuture",
    "EQALeakageDetected",
    "EModelFitFailed",
    "EModelPredictFailed",
    "EFallbackExhausted",
    "EAdapterNotAvailable",
    "EModelNotLoaded",
    "EModelLoadFailed",
    "ETSFMRequiredUnavailable",
    "ETaskSpecInvalid",
    "ETaskSpecIncompatible",
    "EArtifactSchemaIncompatible",
    "EArtifactLoadFailed",
    "EBacktestFail",
    "EBacktestInsufficientData",
    "EBacktestInvalidWindow",
    "ECalibrationFail",
    "EAnomalyFail",
    "EOOM",
    "ECovariateIncompleteKnown",
    "ECovariateStaticInvalid",
]
