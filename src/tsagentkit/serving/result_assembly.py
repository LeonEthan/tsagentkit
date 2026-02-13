"""Helpers for assembling serving outputs and run artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tsagentkit.contracts import ForecastResult
from tsagentkit.utils.compat import safe_model_dump

from .packaging import package_run
from .provenance import create_provenance

if TYPE_CHECKING:
    import pandas as pd

    from tsagentkit.contracts import DryRunResult
    from tsagentkit.qa import QAReport


def build_dry_run_result(
    validation: Any,
    qa_report: QAReport | None,
    plan: Any,
    route_decision: Any,
    task_spec: Any,
) -> DryRunResult:
    """Build DryRunResult payload from pipeline state."""
    from tsagentkit.contracts.results import DryRunResult

    return DryRunResult(
        validation=validation.to_dict() if validation else {},
        qa_report={
            "issues": list(qa_report.issues) if qa_report else [],
            "repairs": [safe_model_dump(r, r) for r in qa_report.repairs] if qa_report else [],
            "leakage_detected": qa_report.leakage_detected if qa_report else False,
        },
        plan=safe_model_dump(plan, {"plan": str(plan)}),
        route_decision=safe_model_dump(route_decision, {"route_decision": str(route_decision)}),
        task_spec_used=safe_model_dump(task_spec, {}),
    )


def resolve_model_identity(
    selection_map: dict[str, str] | None,
    model_artifact: Any,
    model_artifacts: dict[str, Any] | None,
) -> tuple[str, Any]:
    """Resolve model name string and primary artifact for final packaging."""
    if selection_map is not None and model_artifacts:
        model_names = sorted(set(selection_map.values()))
        model_name = f"per_series({','.join(model_names)})"
        primary_artifact = next(iter(model_artifacts.values()))
        return model_name, primary_artifact

    model_name = model_artifact.model_name if model_artifact else "unknown"
    return model_name, model_artifact


def assemble_run_artifact(
    *,
    data: pd.DataFrame,
    task_spec: Any,
    plan: Any,
    forecast_df: pd.DataFrame,
    validation: Any,
    backtest_report: Any,
    qa_report: QAReport | None,
    model_artifact: Any,
    model_artifacts: dict[str, Any] | None,
    selection_map: dict[str, str] | None,
    qa_repairs: list[dict[str, Any]],
    fallbacks_triggered: list[dict[str, Any]],
    feature_matrix: Any,
    drift_report: Any,
    column_map: dict[str, str] | None,
    original_panel_contract: Any,
    route_decision: Any,
    calibration_artifact: Any,
    anomaly_report: Any,
    degradation_events: list[dict[str, Any]],
    mode: str,
    total_duration_ms: float,
    events: list[dict[str, Any]],
) -> Any:
    """Assemble final RunArtifact from pipeline state."""
    model_name, primary_artifact = resolve_model_identity(
        selection_map=selection_map,
        model_artifact=model_artifact,
        model_artifacts=model_artifacts,
    )

    provenance = create_provenance(
        data=data,
        task_spec=task_spec,
        plan=plan,
        model_config=safe_model_dump(plan),
        qa_repairs=qa_repairs,
        fallbacks_triggered=fallbacks_triggered,
        feature_matrix=feature_matrix,
        drift_report=drift_report,
        column_map=column_map,
        original_panel_contract=safe_model_dump(original_panel_contract),
        route_decision=route_decision,
    )

    forecast_result = ForecastResult(
        df=forecast_df,
        provenance=provenance,
        model_name=model_name,
        horizon=task_spec.horizon,
    )

    return package_run(
        forecast=forecast_result,
        plan=plan,
        task_spec=safe_model_dump(task_spec),
        validation_report=validation.to_dict() if validation else None,
        backtest_report=backtest_report,
        qa_report=qa_report,
        model_artifact=primary_artifact,
        provenance=provenance,
        calibration_artifact=calibration_artifact,
        anomaly_report=anomaly_report,
        degradation_events=degradation_events,
        metadata={
            "mode": mode,
            "total_duration_ms": total_duration_ms,
            "events": events,
            "per_series_selection": selection_map is not None,
        },
    )

