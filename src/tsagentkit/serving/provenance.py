"""Provenance tracking for forecasting runs.

Provides utilities for tracking data lineage and reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    pass

    from tsagentkit.contracts import TaskSpec
    from tsagentkit.router import Plan


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
    plan: Plan,
    model_config: dict[str, Any] | None = None,
    qa_repairs: list[dict[str, Any]] | None = None,
    fallbacks_triggered: list[dict[str, Any]] | None = None,
    feature_matrix: Any | None = None,
    drift_report: Any | None = None,
) -> dict[str, Any]:
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
        Provenance dictionary with signatures and metadata
    """
    from datetime import datetime, timezone

    provenance: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_signature": compute_data_signature(data),
        "task_signature": task_spec.model_hash(),
        "plan_signature": plan.signature,
        "model_signature": compute_config_signature(model_config or {}),
        "qa_repairs": qa_repairs or [],
        "fallbacks_triggered": fallbacks_triggered or [],
    }

    # v0.2: Add feature signature if available
    if feature_matrix is not None:
        provenance["feature_signature"] = feature_matrix.signature
        provenance["feature_config_hash"] = feature_matrix.config_hash
        provenance["n_features"] = len(feature_matrix.feature_cols)

    # v0.2: Add drift info if available
    if drift_report is not None:
        provenance["drift_detected"] = drift_report.drift_detected
        provenance["drift_score"] = drift_report.overall_drift_score
        provenance["drift_threshold"] = drift_report.threshold_used
        if drift_report.drift_detected:
            provenance["drifting_features"] = drift_report.get_drifting_features()

    return provenance


def log_event(
    step_name: str,
    status: str,
    duration_ms: float,
    error_code: str | None = None,
    artifacts_generated: list[str] | None = None,
) -> dict[str, Any]:
    """Log a structured event.

    Args:
        step_name: Name of the pipeline step
        status: Status (e.g., "success", "failed")
        duration_ms: Duration in milliseconds
        error_code: Error code if failed
        artifacts_generated: List of artifact names generated

    Returns:
        Event dictionary
    """
    return {
        "step_name": step_name,
        "status": status,
        "duration_ms": duration_ms,
        "error_code": error_code,
        "artifacts_generated": artifacts_generated or [],
    }
