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
    from tsagentkit.router import Plan


def package_run(
    forecast: "ForecastResult",
    plan: "Plan" | dict[str, Any],
    backtest_report: "BacktestReport" | None = None,
    qa_report: "QAReport" | None = None,
    model_artifact: "ModelArtifact" | None = None,
    provenance: "Provenance" | None = None,
    metadata: dict[str, Any] | None = None,
) -> RunArtifact:
    """Package all run outputs into a comprehensive artifact.

    Args:
        forecast: ForecastResult with predictions + provenance
        plan: Execution plan (Plan or dict)
        backtest_report: Optional backtest results
        qa_report: Optional QA report
        model_artifact: Optional fitted model
        provenance: Optional provenance information (overrides forecast provenance)
        metadata: Optional metadata

    Returns:
        RunArtifact containing all run outputs
    """
    plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else plan
    backtest_dict = backtest_report.to_dict() if backtest_report else None
    qa_dict = qa_report.to_dict() if qa_report and hasattr(qa_report, "to_dict") else None

    return RunArtifact(
        forecast=forecast,
        plan=plan_dict,
        backtest_report=backtest_dict,
        qa_report=qa_dict,
        model_artifact=model_artifact,
        provenance=provenance or forecast.provenance,
        metadata=metadata or {},
    )
