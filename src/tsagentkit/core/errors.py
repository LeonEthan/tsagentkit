"""Core error types with rich context.

Consolidates 30+ specific error classes into 5 core error types
with rich context for debugging.
"""

from __future__ import annotations

from typing import Any


class TSAgentKitError(Exception):
    """Base exception with rich context.

    All errors in tsagentkit use this class with specific error_code
    values instead of creating many subclasses.
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
        if fix_hint:
            self.fix_hint = fix_hint

    def __str__(self) -> str:
        parts = [f"[{self.error_code}] {self.message}"]
        if self.context:
            parts.append(f"(context: {self.context})")
        if self.fix_hint:
            parts.append(f"[hint: {self.fix_hint}]")
        return " ".join(parts)


class EContractViolation(TSAgentKitError):
    """Input data violates contract requirements."""

    error_code = "E_CONTRACT_VIOLATION"
    fix_hint = "Check data format: DataFrame must have [unique_id, ds, y] columns"


class EDataQuality(TSAgentKitError):
    """Data quality issue detected."""

    error_code = "E_DATA_QUALITY"
    fix_hint = "Run diagnose() to identify specific quality issues"


class EModelFailed(TSAgentKitError):
    """Model fitting or prediction failed."""

    error_code = "E_MODEL_FAILED"
    fix_hint = "Check model compatibility with data frequency and length"


class ETSFMRequired(TSAgentKitError):
    """TSFM required but unavailable."""

    error_code = "E_TSFM_REQUIRED"
    fix_hint = "Install TSFM adapters (chronos, moirai, timesfm) or set tsfm_mode='preferred'"


# Legacy aliases for backward compatibility during transition
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

# Additional legacy error aliases
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

# Error registry for lookup
ERROR_REGISTRY: dict[str, type[TSAgentKitError]] = {
    "E_CONTRACT_VIOLATION": EContractViolation,
    "E_DATA_QUALITY": EDataQuality,
    "E_MODEL_FAILED": EModelFailed,
    "E_TSFM_REQUIRED": ETSFMRequired,
}


def get_error_class(error_code: str) -> type[TSAgentKitError]:
    """Get error class by code."""
    return ERROR_REGISTRY.get(error_code, TSAgentKitError)
