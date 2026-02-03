"""Provenance tracking for forecasting runs.

Provides utilities for tracking data lineage and reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pandas as pd

if TYPE_CHECKING:
    pass

    from tsagentkit.contracts import TaskSpec
    from tsagentkit.contracts import Provenance
    from tsagentkit.router import PlanSpec


def compute_data_signature(df: pd.DataFrame) -> str:
    """Compute a hash signature for a DataFrame.

    Args:
        df: DataFrame to hash

    Returns:
        SHA-256 hash string (truncated to 16 chars)
    """
    # Use sorted values to ensure consistent hashing
    cols = sorted(df.columns)
    data_str = ""

    for col in cols:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            # Convert datetimes to ISO format strings
            values = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
        else:
            values = df[col].astype(str).tolist()
        data_str += f"{col}:{','.join(values)};"

    return hashlib.sha256(data_str.encode()).hexdigest()[:16]


def compute_config_signature(config: dict[str, Any]) -> str:
    """Compute a hash signature for a configuration dict.

    Args:
        config: Configuration dictionary

    Returns:
        SHA-256 hash string (truncated to 16 chars)
    """
    json_str = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def create_provenance(
    data: pd.DataFrame,
    task_spec: TaskSpec,
    plan: PlanSpec,
    model_config: dict[str, Any] | None = None,
    qa_repairs: list[dict[str, Any]] | None = None,
    fallbacks_triggered: list[dict[str, Any]] | None = None,
    feature_matrix: Any | None = None,
    drift_report: Any | None = None,
    column_map: dict[str, str] | None = None,
    original_panel_contract: dict[str, Any] | None = None,
) -> "Provenance":
    """Create a provenance record for a forecasting run.

    Args:
        data: Input data
        task_spec: Task specification
        plan: Execution plan
        model_config: Model configuration
        qa_repairs: List of QA repairs applied
        fallbacks_triggered: List of fallback events
        feature_matrix: Optional FeatureMatrix for feature signature (v0.2)
        drift_report: Optional DriftReport for drift info (v0.2)

    Returns:
        Provenance object with signatures and metadata
    """
    from datetime import datetime, timezone

    from tsagentkit.contracts import Provenance

    metadata: dict[str, Any] = {}

    # v0.2: Add feature signature if available
    if feature_matrix is not None:
        metadata["feature_signature"] = feature_matrix.signature
        metadata["feature_config_hash"] = feature_matrix.config_hash
        metadata["n_features"] = len(feature_matrix.feature_cols)

    # v0.2: Add drift info if available
    if drift_report is not None:
        metadata["drift_detected"] = drift_report.drift_detected
        metadata["drift_score"] = drift_report.overall_drift_score
        metadata["drift_threshold"] = drift_report.threshold_used
        if drift_report.drift_detected:
            metadata["drifting_features"] = drift_report.get_drifting_features()

    if column_map:
        metadata["column_map"] = column_map
    if original_panel_contract:
        metadata["original_panel_contract"] = original_panel_contract

    from tsagentkit.router import compute_plan_signature

    return Provenance(
        run_id=str(uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        data_signature=compute_data_signature(data),
        task_signature=task_spec.model_hash(),
        plan_signature=compute_plan_signature(plan),
        model_signature=compute_config_signature(model_config or {}),
        qa_repairs=qa_repairs or [],
        fallbacks_triggered=fallbacks_triggered or [],
        metadata=metadata,
    )


def log_event(
    step_name: str,
    status: str,
    duration_ms: float,
    error_code: str | None = None,
    artifacts_generated: list[str] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Log a structured event.

    Creates an event dictionary with all required fields per PRD section 6.2:
    - step_name: Pipeline step name
    - status: Execution status
    - duration_ms: Execution duration
    - error_code: Error code if applicable
    - artifacts_generated: List of generated artifacts
    - timestamp: ISO 8601 timestamp
    - context: Additional context

    Args:
        step_name: Name of the pipeline step
        status: Status (e.g., "success", "failed")
        duration_ms: Duration in milliseconds
        error_code: Error code if failed
        artifacts_generated: List of artifact names generated
        context: Additional context dictionary

    Returns:
        Event dictionary with all structured logging fields
    """
    from datetime import datetime, timezone

    event = {
        "step_name": step_name,
        "status": status,
        "duration_ms": round(duration_ms, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error_code": error_code,
        "artifacts_generated": artifacts_generated or [],
    }

    if context:
        event["context"] = context

    return event


def format_event_json(event: dict[str, Any]) -> str:
    """Format an event as JSON string for structured logging.

    Args:
        event: Event dictionary from log_event()

    Returns:
        JSON string representation
    """
    return json.dumps(event, sort_keys=True, separators=(",", ":"))


class StructuredLogger:
    """Structured logger for tsagentkit pipeline events.

    Provides consistent JSON-formatted logging with all required fields
    per PRD section 6.2 (Observability & Error Codes).

    Example:
        >>> logger = StructuredLogger()
        >>> logger.start_step("fit")
        >>> # ... do work ...
        >>> event = logger.end_step("fit", status="success")
        >>> print(logger.to_json())

    Attributes:
        events: List of logged events
    """

    def __init__(self) -> None:
        """Initialize the structured logger."""
        self.events: list[dict[str, Any]] = []
        self._start_times: dict[str, float] = {}

    def start_step(self, step_name: str) -> None:
        """Record the start time for a step.

        Args:
            step_name: Name of the step
        """
        import time

        self._start_times[step_name] = time.time()

    def end_step(
        self,
        step_name: str,
        status: str = "success",
        error_code: str | None = None,
        artifacts_generated: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """End a step and log the event.

        Args:
            step_name: Name of the step
            status: Execution status
            error_code: Error code if failed
            artifacts_generated: List of artifacts generated
            context: Additional context

        Returns:
            The logged event
        """
        import time

        start_time = self._start_times.get(step_name)
        if start_time is not None:
            duration_ms = (time.time() - start_time) * 1000
            del self._start_times[step_name]
        else:
            duration_ms = 0.0

        event = log_event(
            step_name=step_name,
            status=status,
            duration_ms=duration_ms,
            error_code=error_code,
            artifacts_generated=artifacts_generated,
            context=context,
        )

        self.events.append(event)
        return event

    def log(
        self,
        step_name: str,
        status: str,
        duration_ms: float,
        error_code: str | None = None,
        artifacts_generated: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Log an event directly.

        Args:
            step_name: Name of the step
            status: Execution status
            duration_ms: Duration in milliseconds
            error_code: Error code if failed
            artifacts_generated: List of artifacts generated
            context: Additional context

        Returns:
            The logged event
        """
        event = log_event(
            step_name=step_name,
            status=status,
            duration_ms=duration_ms,
            error_code=error_code,
            artifacts_generated=artifacts_generated,
            context=context,
        )

        self.events.append(event)
        return event

    def to_json(self) -> str:
        """Export all events as JSON.

        Returns:
            JSON array string
        """
        return json.dumps(self.events, sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> list[dict[str, Any]]:
        """Export all events as list of dictionaries.

        Returns:
            List of event dictionaries
        """
        return self.events.copy()

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of logged events.

        Returns:
            Summary dictionary with counts and timing
        """
        total_duration = sum(e.get("duration_ms", 0) for e in self.events)
        success_count = sum(1 for e in self.events if e.get("status") == "success")
        failed_count = sum(1 for e in self.events if e.get("status") == "failed")

        return {
            "total_events": len(self.events),
            "success_count": success_count,
            "failed_count": failed_count,
            "total_duration_ms": round(total_duration, 3),
            "steps": [e.get("step_name") for e in self.events],
        }
