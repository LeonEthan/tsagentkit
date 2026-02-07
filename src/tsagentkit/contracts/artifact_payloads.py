"""Pydantic payload models for serialized run artifacts.

These models define JSON-friendly artifact boundaries while allowing
internal runtime code to continue using dataclasses and DataFrames.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .errors import EArtifactSchemaIncompatible

class _PayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


RUN_ARTIFACT_TYPE = "tsagentkit.run_artifact"
RUN_ARTIFACT_SCHEMA_VERSION = 1
SUPPORTED_RUN_ARTIFACT_SCHEMA_VERSIONS = frozenset({RUN_ARTIFACT_SCHEMA_VERSION})


class CalibrationArtifactPayload(_PayloadModel):
    """Serializable payload for calibration artifacts."""

    method: Literal["none", "conformal"]
    level: int
    by: Literal["unique_id", "global"] = "unique_id"
    deltas: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnomalyReportPayload(_PayloadModel):
    """Serializable payload for anomaly reports."""

    method: str
    level: int
    score: str
    summary: dict[str, Any] = Field(default_factory=dict)
    frame: list[dict[str, Any]] = Field(default_factory=list)


class RunArtifactPayload(_PayloadModel):
    """Serializable payload for run artifacts."""

    artifact_type: Literal["tsagentkit.run_artifact"] = RUN_ARTIFACT_TYPE
    artifact_schema_version: int = Field(RUN_ARTIFACT_SCHEMA_VERSION, ge=1)
    tsagentkit_version: str | None = None
    lifecycle_stage: Literal["train", "serve", "train_serve"] = "train_serve"
    forecast: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    task_spec: dict[str, Any] | None = None
    plan_spec: dict[str, Any] | None = None
    validation_report: dict[str, Any] | None = None
    backtest_report: dict[str, Any] | None = None
    qa_report: dict[str, Any] | None = None
    model_artifact: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    calibration_artifact: CalibrationArtifactPayload | None = None
    anomaly_report: AnomalyReportPayload | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()  # type: ignore[no-any-return]
    if hasattr(value, "to_dict"):
        return value.to_dict()  # type: ignore[no-any-return]
    return None


def _frame_to_records(frame: Any) -> list[dict[str, Any]]:
    if frame is None:
        return []
    if isinstance(frame, list):
        return frame
    if hasattr(frame, "to_dict"):
        try:
            return frame.to_dict("records")  # type: ignore[no-any-return]
        except TypeError:
            pass
    raise TypeError("Anomaly frame must be a DataFrame-like object or list[dict].")


def calibration_payload_from_any(value: Any | None) -> CalibrationArtifactPayload | None:
    if value is None:
        return None
    if isinstance(value, CalibrationArtifactPayload):
        return value

    mapping = _as_mapping(value)
    if mapping is None:
        try:
            method = value.method
            level = value.level
            by = value.by
        except AttributeError as exc:
            raise TypeError(
                "Calibration artifact must define method, level, and by."
            ) from exc
        mapping = {
            "method": method,
            "level": level,
            "by": by,
            "deltas": value.__dict__.get("deltas", {}),
            "metadata": value.__dict__.get("metadata", {}),
        }

    return CalibrationArtifactPayload.model_validate(mapping)


def anomaly_payload_from_any(value: Any | None) -> AnomalyReportPayload | None:
    if value is None:
        return None
    if isinstance(value, AnomalyReportPayload):
        return value

    mapping = _as_mapping(value)
    if mapping is None:
        try:
            method = value.method
            level = value.level
            score = value.score
        except AttributeError as exc:
            raise TypeError(
                "Anomaly report must define method, level, and score."
            ) from exc
        frame = value.__dict__.get("frame")
        mapping = {
            "method": method,
            "level": level,
            "score": score,
            "summary": value.__dict__.get("summary", {}),
            "frame": _frame_to_records(frame),
        }
    else:
        mapping = dict(mapping)
        mapping["frame"] = _frame_to_records(mapping.get("frame"))

    return AnomalyReportPayload.model_validate(mapping)


def calibration_payload_dict(value: Any | None) -> dict[str, Any] | None:
    payload = calibration_payload_from_any(value)
    return payload.model_dump() if payload is not None else None


def anomaly_payload_dict(value: Any | None) -> dict[str, Any] | None:
    payload = anomaly_payload_from_any(value)
    return payload.model_dump() if payload is not None else None


def run_artifact_payload_from_dict(data: dict[str, Any]) -> RunArtifactPayload:
    normalized = dict(data)
    normalized["calibration_artifact"] = calibration_payload_from_any(
        normalized.get("calibration_artifact")
    )
    normalized["anomaly_report"] = anomaly_payload_from_any(normalized.get("anomaly_report"))
    return RunArtifactPayload.model_validate(normalized)


def validate_run_artifact_compatibility(
    data: dict[str, Any],
    *,
    supported_schema_versions: set[int] | None = None,
    expected_artifact_type: str = RUN_ARTIFACT_TYPE,
) -> RunArtifactPayload:
    """Validate schema/type compatibility for serialized run artifacts."""
    payload = run_artifact_payload_from_dict(data)

    supported = (
        supported_schema_versions
        if supported_schema_versions is not None
        else set(SUPPORTED_RUN_ARTIFACT_SCHEMA_VERSIONS)
    )
    if payload.artifact_schema_version not in supported:
        raise EArtifactSchemaIncompatible(
            "RunArtifact schema version is not supported.",
            context={
                "artifact_schema_version": payload.artifact_schema_version,
                "supported_schema_versions": sorted(supported),
            },
        )
    if payload.artifact_type != expected_artifact_type:
        raise EArtifactSchemaIncompatible(
            "RunArtifact type is incompatible.",
            context={
                "artifact_type": payload.artifact_type,
                "expected_artifact_type": expected_artifact_type,
            },
        )

    return payload


__all__ = [
    "RUN_ARTIFACT_TYPE",
    "RUN_ARTIFACT_SCHEMA_VERSION",
    "SUPPORTED_RUN_ARTIFACT_SCHEMA_VERSIONS",
    "CalibrationArtifactPayload",
    "AnomalyReportPayload",
    "RunArtifactPayload",
    "calibration_payload_from_any",
    "anomaly_payload_from_any",
    "calibration_payload_dict",
    "anomaly_payload_dict",
    "run_artifact_payload_from_dict",
    "validate_run_artifact_compatibility",
]
