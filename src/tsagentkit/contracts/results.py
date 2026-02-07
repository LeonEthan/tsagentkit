"""Forecast result structures.

Defines the data structures for forecast outputs including provenance tracking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

_QUANTILE_PATTERN = re.compile(r"^q[_]?([0-9]+(?:\.[0-9]+)?)$")


def _parse_quantile_column(col: str) -> float | None:
    match = _QUANTILE_PATTERN.match(col)
    if not match:
        return None
    value = float(match.group(1))
    if value > 1:
        value = value / 100.0
    if not 0 < value < 1:
        return None
    return value


def _is_datetime_like(series: Any) -> bool:
    try:
        import pandas as pd  # Optional import for accurate dtype checks.

        return bool(pd.api.types.is_datetime64_any_dtype(series))
    except Exception:
        dtype = getattr(series, "dtype", None)
        kind = getattr(dtype, "kind", None)
        return kind == "M"


@dataclass(frozen=True)
class ForecastFrame:
    """Forecast frame in long format.

    Expected columns: unique_id, ds, model, yhat (+ intervals/quantiles).
    """

    df: Any


@dataclass(frozen=True)
class CVFrame:
    """Cross-validation frame in long format.

    Expected columns: unique_id, ds, cutoff, model, y, yhat (+ intervals/quantiles).
    """

    df: Any


@dataclass(frozen=True)
class Provenance:
    """Provenance information for a forecast run.

    Provides full traceability of the forecasting pipeline including
    data signatures, model configurations, and execution metadata.

    Attributes:
        run_id: Unique identifier for this run (UUID)
        timestamp: ISO 8601 timestamp of execution
        data_signature: Hash of input data
        task_signature: Hash of task specification
        plan_signature: Hash of execution plan
        model_signature: Hash of model configuration
        qa_repairs: List of data repairs applied
        fallbacks_triggered: List of fallback events
        metadata: Additional execution metadata
    """

    run_id: str
    timestamp: str
    data_signature: str
    task_signature: str
    plan_signature: str
    model_signature: str
    qa_repairs: list[dict[str, Any]] = field(default_factory=list)
    fallbacks_triggered: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "data_signature": self.data_signature,
            "task_signature": self.task_signature,
            "plan_signature": self.plan_signature,
            "model_signature": self.model_signature,
            "qa_repairs": self.qa_repairs,
            "fallbacks_triggered": self.fallbacks_triggered,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Provenance:
        """Create from dictionary."""
        return cls(**data)


@dataclass(frozen=True)
class ForecastResult:
    """Result of a forecast operation.

    Contains the forecast values with optional quantiles and full
    provenance information for reproducibility.

    Attributes:
        df: DataFrame with columns [unique_id, ds, model, yhat] + quantile columns
        provenance: Full provenance information
        model_name: Name of the model that produced this forecast
        horizon: Forecast horizon
    """

    df: Any
    provenance: Provenance
    model_name: str
    horizon: int

    def __post_init__(self) -> None:
        """Validate the dataframe structure."""
        required_cols = {"unique_id", "ds", "model", "yhat"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"ForecastResult df missing columns: {missing}")

        # Validate types
        if not _is_datetime_like(self.df["ds"]):
            raise ValueError("Column 'ds' must be datetime")

    def get_quantile_columns(self) -> list[str]:
        """Get list of quantile column names.

        Returns:
            List of column names starting with 'q' (quantile columns)
        """
        return [c for c in self.df.columns if _parse_quantile_column(c) is not None]

    def get_series(self, unique_id: str) -> pd.DataFrame:
        """Get forecast for a specific series.

        Args:
            unique_id: The series identifier

        Returns:
            DataFrame with forecast for the specified series
        """
        return self.df[self.df["unique_id"] == unique_id].copy()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Note: DataFrame is converted to records format.
        """
        return {
            "df": self.df.to_dict("records"),
            "provenance": self.provenance.to_dict(),
            "model_name": self.model_name,
            "horizon": self.horizon,
        }


@dataclass(frozen=True)
class ValidationReport:
    """Report from data validation.

    Contains the results of validating input data against the
    required schema and constraints.

    Attributes:
        valid: Whether validation passed
        errors: List of validation errors (if any)
        warnings: List of validation warnings
        stats: Statistics about the data
    """

    valid: bool
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def raise_if_errors(self) -> None:
        """Raise the first error if any exist."""
        from .errors import get_error_class

        if self.errors:
            err = self.errors[0]
            error_code = err.get("code", "E_CONTRACT_MISSING_COLUMN")
            message = err.get("message", "Validation failed")
            context = err.get("context", {})

            error_class = get_error_class(error_code)
            raise error_class(message, context)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


@dataclass(frozen=True)
class ModelArtifact:
    """Container for a fitted model.

    Stores the fitted model along with its configuration and metadata
    for later prediction and provenance tracking.

    Attributes:
        model: The fitted model (type depends on implementation)
        model_name: Name of the model
        config: Model configuration dictionary
        signature: Hash of model configuration
        fit_timestamp: ISO 8601 timestamp of fitting
        metadata: Additional model metadata
    """

    model: Any
    model_name: str
    config: dict[str, Any] = field(default_factory=dict)
    signature: str = ""
    fit_timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compute signature if not provided."""
        if not self.signature:
            import hashlib
            import json

            # Create deterministic signature from config
            config_str = json.dumps(self.config, sort_keys=True, separators=(",", ":"))
            object.__setattr__(  # Bypass frozen
                self,
                "signature",
                hashlib.sha256(config_str.encode()).hexdigest()[:16],
            )


@dataclass(frozen=True)
class RepairReport:
    """Detailed repair report for audit trail.

    Provides comprehensive information about data repairs applied
    during QA, ensuring full traceability and PIT safety verification.

    Attributes:
        repair_type: Type of repair ("missing_values", "winsorize", "median_filter")
        column: Column that was repaired
        count: Number of values repaired
        method: Method used for repair
        scope: Scope of repair ("observed_history", "future")
        before_sample: Sample statistics before repair (optional)
        after_sample: Sample statistics after repair (optional)
        time_range: Time range of repair as (start, end) ISO strings (optional)
        pit_safe: Whether repair is PIT-safe
        validation_passed: Whether validation passed
    """

    repair_type: str
    column: str
    count: int
    method: str
    scope: str = "observed_history"

    # PIT safety information
    before_sample: dict[str, Any] | None = None
    after_sample: dict[str, Any] | None = None
    time_range: tuple[str, str] | None = None

    # Validation
    pit_safe: bool = True
    validation_passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "repair_type": self.repair_type,
            "column": self.column,
            "count": self.count,
            "method": self.method,
            "scope": self.scope,
            "pit_safe": self.pit_safe,
            "validation_passed": self.validation_passed,
            "before_sample": self.before_sample,
            "after_sample": self.after_sample,
            "time_range": self.time_range,
        }


@dataclass(frozen=True)
class RunArtifact:
    """Complete artifact from a forecasting run.

    The comprehensive output of the forecasting pipeline containing
    all results, reports, and provenance information.

    Attributes:
        forecast: The forecast result
        plan: Execution plan that was used
        backtest_report: Backtest results (if performed)
        qa_report: QA report (if available)
        model_artifact: The fitted model artifact
        provenance: Full provenance information
        metadata: Additional run metadata
    """

    forecast: ForecastResult
    plan: dict[str, Any] | None = None
    task_spec: dict[str, Any] | None = None
    plan_spec: dict[str, Any] | None = None
    validation_report: dict[str, Any] | None = None
    backtest_report: dict[str, Any] | None = None
    qa_report: dict[str, Any] | None = None
    model_artifact: ModelArtifact | None = None
    provenance: Provenance | None = None
    calibration_artifact: dict[str, Any] | None = None
    anomaly_report: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        from tsagentkit.contracts.artifact_payloads import (
            run_artifact_payload_from_dict,
        )

        payload = {
            "forecast": self.forecast.to_dict() if self.forecast else None,
            "plan": self.plan,
            "task_spec": self.task_spec,
            "plan_spec": self.plan_spec,
            "validation_report": self.validation_report,
            "backtest_report": self.backtest_report,
            "qa_report": self.qa_report,
            "model_artifact": {
                "model_name": self.model_artifact.model_name,
                "signature": self.model_artifact.signature,
                "fit_timestamp": self.model_artifact.fit_timestamp,
            } if self.model_artifact else None,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "calibration_artifact": self.calibration_artifact,
            "anomaly_report": self.anomaly_report,
            "metadata": self.metadata,
        }
        return run_artifact_payload_from_dict(payload).model_dump()

    def summary(self) -> str:
        """Generate a human-readable summary."""
        model_name = self.forecast.model_name if self.forecast else "N/A"
        forecast_rows = len(self.forecast.df) if self.forecast else 0

        plan_desc = "N/A"
        if isinstance(self.plan, dict):
            candidates = self.plan.get("candidate_models")
            if candidates:
                chain = "->".join(candidates)
                plan_desc = f"Plan({chain})"
            else:
                primary = self.plan.get("primary_model")
                fallback = self.plan.get("fallback_chain", [])
                if primary:
                    chain = "->".join([primary] + list(fallback)) if fallback else primary
                    plan_desc = f"Plan({chain})"
                else:
                    plan_desc = str(self.plan.get("signature") or self.plan)
        else:
            plan_desc = str(self.plan)

        lines = [
            "Run Artifact Summary",
            "=" * 40,
            f"Model: {model_name}",
            f"Plan: {plan_desc}",
            f"Forecast rows: {forecast_rows}",
        ]

        if self.backtest_report:
            n_windows = self.backtest_report.get("n_windows")
            if n_windows is not None:
                lines.append(f"Backtest windows: {n_windows}")
            metrics = self.backtest_report.get("aggregate_metrics", {})
            if metrics:
                lines.append("Aggregate Metrics:")
                for name, value in sorted(metrics.items()):
                    lines.append(f"  {name}: {value:.4f}")

        if self.provenance:
            lines.append("\nProvenance:")
            lines.append(f"  Data signature: {self.provenance.data_signature}")
            lines.append(f"  Timestamp: {self.provenance.timestamp}")

        return "\n".join(lines)


__all__ = [
    "CVFrame",
    "ForecastFrame",
    "ForecastResult",
    "ModelArtifact",
    "Provenance",
    "RepairReport",
    "RunArtifact",
    "ValidationReport",
]
