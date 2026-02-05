"""Run artifact packaging.

Bundles all outputs from a forecasting run into a comprehensive artifact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tsagentkit.contracts import RunArtifact

if TYPE_CHECKING:
    from tsagentkit.backtest import BacktestReport
    from tsagentkit.contracts import ForecastResult, ModelArtifact, Provenance
    from tsagentkit.qa import QAReport
    from tsagentkit.router import PlanSpec


def package_run(
    forecast: ForecastResult,
    plan: PlanSpec | dict[str, Any],
    task_spec: Any | None = None,
    plan_spec: dict[str, Any] | None = None,
    validation_report: dict[str, Any] | None = None,
    backtest_report: BacktestReport | None = None,
    qa_report: QAReport | None = None,
    model_artifact: ModelArtifact | None = None,
    provenance: Provenance | None = None,
    calibration_artifact: dict[str, Any] | None = None,
    anomaly_report: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
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
    if hasattr(plan, "to_dict"):
        plan_dict = plan.to_dict()
    elif hasattr(plan, "model_dump"):
        plan_dict = plan.model_dump()
    else:
        plan_dict = plan
    if plan_spec is None:
        plan_spec = plan_dict
    backtest_dict = backtest_report.to_dict() if backtest_report else None
    qa_dict = qa_report.to_dict() if qa_report and hasattr(qa_report, "to_dict") else None

    return RunArtifact(
        forecast=forecast,
        plan=plan_dict,
        task_spec=task_spec,
        plan_spec=plan_spec,
        validation_report=validation_report,
        backtest_report=backtest_dict,
        qa_report=qa_dict,
        model_artifact=model_artifact,
        provenance=provenance or forecast.provenance,
        calibration_artifact=calibration_artifact,
        anomaly_report=anomaly_report,
        metadata=metadata or {},
    )
