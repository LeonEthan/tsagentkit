"""RunArtifact lifecycle helpers for save/load/compatibility/replay."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tsagentkit.contracts import (
    EArtifactLoadFailed,
    EArtifactSchemaIncompatible,
    ForecastResult,
    Provenance,
    RunArtifact,
    validate_run_artifact_compatibility,
)


def _json_default(value: Any) -> Any:
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_run_artifact(
    artifact: RunArtifact,
    path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """Persist RunArtifact payload as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = artifact.to_dict()
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, sort_keys=True, default=_json_default)
    return output_path


def load_run_artifact(path: str | Path) -> RunArtifact:
    """Load and validate RunArtifact from JSON payload."""
    input_path = Path(path)
    try:
        with input_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        raise EArtifactLoadFailed(
            "Failed to read RunArtifact payload from disk.",
            context={"path": str(input_path), "error": str(exc)},
        ) from exc

    if not isinstance(payload, dict):
        raise EArtifactLoadFailed(
            "RunArtifact payload must be a JSON object.",
            context={"path": str(input_path), "type": type(payload).__name__},
        )

    return RunArtifact.from_dict(payload)


def validate_run_artifact_for_serving(
    artifact: RunArtifact | dict[str, Any],
    *,
    expected_task_signature: str | None = None,
    expected_plan_signature: str | None = None,
    supported_schema_versions: set[int] | None = None,
) -> dict[str, Any]:
    """Validate compatibility and optional signature gates for serving."""
    payload = artifact.to_dict() if isinstance(artifact, RunArtifact) else dict(artifact)
    normalized = validate_run_artifact_compatibility(
        payload,
        supported_schema_versions=supported_schema_versions,
    )

    provenance = normalized.provenance
    if not isinstance(provenance, dict):
        forecast_provenance = (
            normalized.forecast.get("provenance")
            if isinstance(normalized.forecast, dict)
            else None
        )
        provenance = forecast_provenance if isinstance(forecast_provenance, dict) else None
    if not isinstance(provenance, dict):
        raise EArtifactSchemaIncompatible("RunArtifact is missing provenance payload.")

    task_signature = provenance.get("task_signature")
    if expected_task_signature is not None and task_signature != expected_task_signature:
        raise EArtifactSchemaIncompatible(
            "RunArtifact task signature does not match expected serving signature.",
            context={
                "expected_task_signature": expected_task_signature,
                "actual_task_signature": task_signature,
            },
        )

    plan_signature = provenance.get("plan_signature")
    if expected_plan_signature is not None and plan_signature != expected_plan_signature:
        raise EArtifactSchemaIncompatible(
            "RunArtifact plan signature does not match expected serving signature.",
            context={
                "expected_plan_signature": expected_plan_signature,
                "actual_plan_signature": plan_signature,
            },
        )

    return {
        "artifact_type": normalized.artifact_type,
        "artifact_schema_version": normalized.artifact_schema_version,
        "tsagentkit_version": normalized.tsagentkit_version,
        "lifecycle_stage": normalized.lifecycle_stage,
        "run_id": provenance.get("run_id"),
        "task_signature": task_signature,
        "plan_signature": plan_signature,
    }


def replay_forecast_from_artifact(
    artifact: RunArtifact | dict[str, Any],
) -> ForecastResult:
    """Reconstruct a forecast object from a serialized or loaded artifact."""
    run_artifact = artifact if isinstance(artifact, RunArtifact) else RunArtifact.from_dict(artifact)
    source = run_artifact.provenance or run_artifact.forecast.provenance

    replay_metadata = dict(source.metadata)
    replay_metadata["replayed_from_artifact"] = True
    replay_metadata["source_run_id"] = source.run_id
    replayed_provenance = Provenance(
        run_id=source.run_id,
        timestamp=source.timestamp,
        data_signature=source.data_signature,
        task_signature=source.task_signature,
        plan_signature=source.plan_signature,
        model_signature=source.model_signature,
        qa_repairs=list(source.qa_repairs),
        fallbacks_triggered=list(source.fallbacks_triggered),
        metadata=replay_metadata,
    )

    return ForecastResult(
        df=run_artifact.forecast.df.copy(),
        provenance=replayed_provenance,
        model_name=run_artifact.forecast.model_name,
        horizon=run_artifact.forecast.horizon,
    )


__all__ = [
    "save_run_artifact",
    "load_run_artifact",
    "validate_run_artifact_for_serving",
    "replay_forecast_from_artifact",
]
