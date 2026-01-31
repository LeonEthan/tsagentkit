"""Run artifact packaging.

Bundles all outputs from a forecasting run into a comprehensive artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pandas as pd

    from tsagentkit.backtest import BacktestReport
    from tsagentkit.contracts import ForecastResult, ModelArtifact, Provenance
    from tsagentkit.qa import QAReport
    from tsagentkit.router import Plan


@dataclass(frozen=True)
class RunArtifact:
    """Complete artifact from a forecasting run.

    The comprehensive output of the forecasting pipeline containing
    all results, reports, and provenance information.

    Attributes:
        forecast: The forecast result DataFrame
        plan: Execution plan that was used
        model_name: Name of the model that produced the forecast
        backtest_report: Backtest results (if performed)
        qa_report: QA report (if available)
        model_artifact: The fitted model artifact
        provenance: Full provenance information
        metadata: Additional run metadata
    """

    forecast: pd.DataFrame
    plan: Plan
    model_name: str
    backtest_report: BacktestReport | None = None
    qa_report: QAReport | None = None
    model_artifact: ModelArtifact | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the artifact
        """
        return {
            "forecast": self.forecast.to_dict("records"),
            "plan": self.plan.to_dict(),
            "model_name": self.model_name,
            "backtest_report": (
                self.backtest_report.to_dict() if self.backtest_report else None
            ),
            "qa_report": None,  # QAReport serialization if needed
            "model_artifact": {
                "model_name": self.model_artifact.model_name,
                "signature": self.model_artifact.signature,
                "fit_timestamp": self.model_artifact.fit_timestamp,
            } if self.model_artifact else None,
            "provenance": self.provenance,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Generate a human-readable summary.

        Returns:
            Summary string
        """
        lines = [
            "Run Artifact Summary",
            "=" * 40,
            f"Model: {self.model_name}",
            f"Plan: {self.plan.to_signature()}",
            f"Forecast rows: {len(self.forecast)}",
        ]

        if self.backtest_report:
            lines.append(f"Backtest windows: {self.backtest_report.n_windows}")
            lines.append("Aggregate Metrics:")
            for name, value in sorted(self.backtest_report.aggregate_metrics.items()):
                lines.append(f"  {name}: {value:.4f}")

        if self.provenance:
            lines.append(f"\nProvenance:")
            lines.append(f"  Data signature: {self.provenance.get('data_signature', 'N/A')}")
            lines.append(f"  Timestamp: {self.provenance.get('timestamp', 'N/A')}")

        return "\n".join(lines)


def package_run(
    forecast: pd.DataFrame,
    plan: Plan,
    model_name: str,
    backtest_report: BacktestReport | None = None,
    qa_report: QAReport | None = None,
    model_artifact: ModelArtifact | None = None,
    provenance: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> RunArtifact:
    """Package all run outputs into a comprehensive artifact.

    Args:
        forecast: Forecast DataFrame
        plan: Execution plan
        model_name: Name of model used
        backtest_report: Optional backtest results
        qa_report: Optional QA report
        model_artifact: Optional fitted model
        provenance: Optional provenance information
        metadata: Optional metadata

    Returns:
        RunArtifact containing all run outputs
    """
    return RunArtifact(
        forecast=forecast,
        plan=plan,
        model_name=model_name,
        backtest_report=backtest_report,
        qa_report=qa_report,
        model_artifact=model_artifact,
        provenance=provenance or {},
        metadata=metadata or {},
    )
