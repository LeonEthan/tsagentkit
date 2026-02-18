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


class EContract(TSAgentKitError):  # noqa: N818
    """Input data invalid (wrong columns, types, etc.)"""

    code = "E_CONTRACT"
    error_code = "E_CONTRACT"  # type: ignore[misc]
    hint = "Check data format: DataFrame must have [unique_id, ds, y] columns"


class ENoTSFM(TSAgentKitError):  # noqa: N818
    """No TSFM models registered (internal invariant violation)."""

    code = "E_NO_TSFM"
    error_code = "E_NO_TSFM"  # type: ignore[misc]
    hint = "TSFM registry invariant violated. Ensure default TSFM specs exist in models.registry.REGISTRY."


class EInsufficient(TSAgentKitError):  # noqa: N818
    """Not enough TSFMs succeeded."""

    code = "E_INSUFFICIENT"
    error_code = "E_INSUFFICIENT"  # type: ignore[misc]
    hint = "Check model compatibility with data frequency and length"


class ETemporal(TSAgentKitError):  # noqa: N818
    """Temporal integrity violation."""

    code = "E_TEMPORAL"
    error_code = "E_TEMPORAL"  # type: ignore[misc]
    hint = "Data must be sorted by ds. No future dates in covariates."


__all__ = [
    "TSAgentKitError",
    "EContract",
    "ENoTSFM",
    "EInsufficient",
    "ETemporal",
]
