"""Run artifact packaging.

Bundles all outputs from a forecasting run into a comprehensive artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from tsagentkit.contracts import (
    RUN_ARTIFACT_SCHEMA_VERSION,
    RUN_ARTIFACT_TYPE,
    RunArtifact,
    anomaly_payload_dict,
    calibration_payload_dict,
)

if TYPE_CHECKING:
    from tsagentkit.backtest import BacktestReport
    from tsagentkit.contracts import ForecastResult, ModelArtifact, Provenance
    from tsagentkit.qa import QAReport
    from tsagentkit.router import PlanSpec

PayloadDict = dict[str, object]


def _coerce_payload_dict(value: object, *, fallback_key: str) -> PayloadDict:
    """Convert common model-like objects to dictionary payloads."""
    if isinstance(value, Mapping):
        return dict(value)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        serialized = to_dict()
        if isinstance(serialized, Mapping):
            return dict(serialized)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        serialized = model_dump()
        if isinstance(serialized, Mapping):
            return dict(serialized)

    return {fallback_key: str(value)}


def package_run(
    forecast: ForecastResult,
    plan: PlanSpec | Mapping[str, object],
    task_spec: object | None = None,
    plan_spec: Mapping[str, object] | None = None,
    validation_report: Mapping[str, object] | None = None,
    backtest_report: BacktestReport | None = None,
    qa_report: QAReport | None = None,
    model_artifact: ModelArtifact | None = None,
    provenance: Provenance | None = None,
    calibration_artifact: object | None = None,
    anomaly_report: object | None = None,
    metadata: Mapping[str, object] | None = None,
    lifecycle_stage: str = "train_serve",
    degradation_events: list[PayloadDict] | None = None,
) -> RunArtifact:
    """Package all run outputs into a comprehensive artifact.

    Args:
        forecast: ForecastResult with predictions + provenance
        plan: Execution plan (PlanSpec or dict)
        backtest_report: Optional backtest results
        qa_report: Optional QA report
        model_artifact: Optional fitted model
        provenance: Optional provenance information (overrides forecast provenance)
        calibration_artifact: Optional calibration artifact
        anomaly_report: Optional anomaly report
        metadata: Optional metadata

    Returns:
        RunArtifact containing all run outputs
    """
    plan_dict = _coerce_payload_dict(plan, fallback_key="plan")
    resolved_plan_spec = dict(plan_spec) if plan_spec is not None else dict(plan_dict)
    backtest_dict = backtest_report.to_dict() if backtest_report else None
    qa_dict = qa_report.to_dict() if qa_report and hasattr(qa_report, "to_dict") else None
    calibration_dict = calibration_payload_dict(calibration_artifact)
    anomaly_dict = anomaly_payload_dict(anomaly_report)
    try:
        package_version = version("tsagentkit")
    except PackageNotFoundError:
        package_version = None

    return RunArtifact(
        forecast=forecast,
        plan=plan_dict,
        task_spec=task_spec,
        plan_spec=resolved_plan_spec,
        validation_report=dict(validation_report) if validation_report else None,
        backtest_report=backtest_dict,
        qa_report=qa_dict,
        model_artifact=model_artifact,
        provenance=provenance or forecast.provenance,
        calibration_artifact=calibration_dict,
        anomaly_report=anomaly_dict,
        degradation_events=[dict(event) for event in (degradation_events or [])],
        metadata=dict(metadata) if metadata else {},
        artifact_type=RUN_ARTIFACT_TYPE,
        artifact_schema_version=RUN_ARTIFACT_SCHEMA_VERSION,
        tsagentkit_version=package_version,
        lifecycle_stage=lifecycle_stage,
    )
