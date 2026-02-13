"""Provenance tracking for forecasting runs.

Provides utilities for tracking data lineage and reproducibility.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

import pandas as pd

if TYPE_CHECKING:
    from tsagentkit.contracts import Provenance, TaskSpec
    from tsagentkit.router import PlanSpec


from datetime import UTC

from tsagentkit.utils import compute_config_signature, compute_data_signature

EventPayload = dict[str, object]


class FeatureMatrixLike(Protocol):
    """Minimal feature matrix shape used for provenance metadata."""

    signature: str
    config_hash: str
    feature_cols: Sequence[str]


class DriftReportLike(Protocol):
    """Minimal drift report shape used for provenance metadata."""

    drift_detected: bool
    overall_drift_score: float
    threshold_used: float

    def get_drifting_features(self) -> list[str]: ...


class RouteDecisionLike(Protocol):
    """Minimal route decision shape used for provenance metadata."""

    buckets: object
    reasons: object
    stats: object


def _serialize_repair(repair: object) -> EventPayload:
    """Normalize repair entries to dictionaries for deterministic payloads."""
    if isinstance(repair, Mapping):
        return dict(repair)
    to_dict = getattr(repair, "to_dict", None)
    if callable(to_dict):
        serialized = to_dict()
        if isinstance(serialized, Mapping):
            return dict(serialized)
    return {"value": repair}


def create_provenance(
    data: pd.DataFrame,
    task_spec: TaskSpec,
    plan: PlanSpec,
    model_config: Mapping[str, object] | None = None,
    qa_repairs: list[object] | None = None,
    fallbacks_triggered: list[EventPayload] | None = None,
    feature_matrix: FeatureMatrixLike | None = None,
    drift_report: DriftReportLike | None = None,
    column_map: dict[str, str] | None = None,
    original_panel_contract: Mapping[str, object] | None = None,
    route_decision: RouteDecisionLike | None = None,
) -> Provenance:
    """Create a provenance record for a forecasting run.

    Args:
        data: Input data
        task_spec: Task specification
        plan: Execution plan
        model_config: Model configuration
        qa_repairs: List of QA repairs applied (RepairReport objects)
        fallbacks_triggered: List of fallback events
        feature_matrix: Optional FeatureMatrix for feature signature (v0.2)
        drift_report: Optional DriftReport for drift info (v0.2)
        route_decision: Optional RouteDecision for routing audit trail (v1.0)

    Returns:
        Provenance object with signatures and metadata
    """
    from datetime import datetime

    from tsagentkit.contracts import Provenance

    metadata: EventPayload = {}

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
        metadata["original_panel_contract"] = dict(original_panel_contract)

    # v1.0: Add route decision for audit trail
    if route_decision is not None:
        metadata["route_decision"] = {
            "buckets": route_decision.buckets,
            "reasons": route_decision.reasons,
            "stats": route_decision.stats,
        }

    from tsagentkit.router import compute_plan_signature

    # Convert RepairReport objects to dicts for serialization
    repairs_serialized: list[EventPayload] = []
    for repair in (qa_repairs or []):
        repairs_serialized.append(_serialize_repair(repair))

    return Provenance(
        run_id=str(uuid4()),
        timestamp=datetime.now(UTC).isoformat(),
        data_signature=compute_data_signature(data),
        task_signature=task_spec.model_hash(),
        plan_signature=compute_plan_signature(plan),
        model_signature=compute_config_signature(dict(model_config or {})),
        qa_repairs=repairs_serialized,
        fallbacks_triggered=[dict(event) for event in (fallbacks_triggered or [])],
        metadata=metadata,
    )


def log_event(
    step_name: str,
    status: str,
    duration_ms: float,
    error_code: str | None = None,
    artifacts_generated: list[str] | None = None,
    context: Mapping[str, object] | None = None,
) -> EventPayload:
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
    from datetime import datetime

    event: EventPayload = {
        "step_name": step_name,
        "status": status,
        "duration_ms": round(duration_ms, 3),
        "timestamp": datetime.now(UTC).isoformat(),
        "error_code": error_code,
        "artifacts_generated": artifacts_generated or [],
    }

    if context:
        event["context"] = dict(context)

    return event


def format_event_json(event: Mapping[str, object]) -> str:
    """Format an event as JSON string for structured logging.

    Args:
        event: Event dictionary from log_event()

    Returns:
        JSON string representation
    """
    return json.dumps(dict(event), sort_keys=True, separators=(",", ":"))


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
        self.events: list[EventPayload] = []
        self._start_times: dict[str, float] = {}

    def _record_event(
        self,
        *,
        step_name: str,
        status: str,
        duration_ms: float,
        error_code: str | None = None,
        artifacts_generated: list[str] | None = None,
        context: Mapping[str, object] | None = None,
    ) -> EventPayload:
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
        context: Mapping[str, object] | None = None,
    ) -> EventPayload:
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

        return self._record_event(
            step_name=step_name,
            status=status,
            duration_ms=duration_ms,
            error_code=error_code,
            artifacts_generated=artifacts_generated,
            context=context,
        )

    def log(
        self,
        step_name: str,
        status: str,
        duration_ms: float,
        error_code: str | None = None,
        artifacts_generated: list[str] | None = None,
        context: Mapping[str, object] | None = None,
    ) -> EventPayload:
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
        return self._record_event(
            step_name=step_name,
            status=status,
            duration_ms=duration_ms,
            error_code=error_code,
            artifacts_generated=artifacts_generated,
            context=context,
        )

    def to_json(self) -> str:
        """Export all events as JSON.

        Returns:
            JSON array string
        """
        return json.dumps(self.events, sort_keys=True, separators=(",", ":"))

    def get_events(self) -> list[EventPayload]:
        """Get all logged events.

        Returns:
            List of event dictionaries (copy)
        """
        return self.events.copy()

    def to_dict(self) -> list[EventPayload]:
        """Export all events as list of dictionaries.

        Returns:
            List of event dictionaries
        """
        return self.events.copy()

    def get_summary(self) -> EventPayload:
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
