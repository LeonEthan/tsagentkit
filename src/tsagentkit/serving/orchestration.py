"""Main forecasting orchestration.

Provides the unified entry point run_forecast() for executing
the complete forecasting pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from tsagentkit.contracts import (
    AnomalySpec,
    CalibratorSpec,
    TaskSpec,
)
from tsagentkit.covariates import CovariateBundle

from .pipeline import (
    ForecastPipeline,
    MonitoringConfig,
    _step_fit,
    _step_predict,
)

# Re-export make_plan for backward compatibility with tests
from tsagentkit.router import make_plan

if TYPE_CHECKING:
    from tsagentkit.contracts import DryRunResult, RunArtifact
    from tsagentkit.features import FeatureConfig
    from tsagentkit.hierarchy import HierarchyStructure


@dataclass
class TSAgentSession:
    """Class-backed orchestration session.

    Provides a session abstraction for running forecasting pipelines
    with consistent configuration across multiple runs.
    """

    mode: Literal["quick", "standard", "strict"] = "standard"
    task_spec_defaults: dict[str, Any] | None = None
    model_pool: Any | None = None

    def run(
        self,
        data: pd.DataFrame,
        task_spec: TaskSpec,
        covariates: CovariateBundle | None = None,
        mode: Literal["quick", "standard", "strict"] | None = None,
        fit_func: Any | None = None,
        predict_func: Any | None = None,
        monitoring_config: MonitoringConfig | None = None,
        reference_data: pd.DataFrame | None = None,
        repair_strategy: dict[str, Any] | None = None,
        hierarchy: HierarchyStructure | None = None,
        feature_config: FeatureConfig | None = None,
        calibrator_spec: CalibratorSpec | None = None,
        anomaly_spec: AnomalySpec | None = None,
        reconciliation_method: str = "bottom_up",
        dry_run: bool = False,
    ) -> RunArtifact | DryRunResult:
        """Run forecasting pipeline within this session."""
        effective_mode = mode or self.mode

        pipeline = ForecastPipeline(
            data=data,
            task_spec=task_spec,
            covariates=covariates,
            mode=effective_mode,
            fit_func=fit_func,
            predict_func=predict_func,
            monitoring_config=monitoring_config,
            reference_data=reference_data,
            repair_strategy=repair_strategy,
            hierarchy=hierarchy,
            feature_config=feature_config,
            calibrator_spec=calibrator_spec,
            anomaly_spec=anomaly_spec,
            reconciliation_method=reconciliation_method,
            dry_run=dry_run,
        )
        return pipeline.run()

    def fit(
        self,
        dataset: Any,
        plan: Any,
        fit_func: Any | None = None,
        on_fallback: Any | None = None,
        covariates: Any | None = None,
    ) -> Any:
        """Fit step entrypoint for session consumers."""
        return _step_fit(
            dataset=dataset,
            plan=plan,
            fit_func=fit_func,
            on_fallback=on_fallback,
            covariates=covariates,
        )

    def predict(
        self,
        artifact: Any,
        dataset: Any,
        task_spec: TaskSpec,
        predict_func: Any | None = None,
        plan: Any | None = None,
        covariates: Any | None = None,
        reconciliation_method: str = "bottom_up",
    ) -> pd.DataFrame:
        """Predict step entrypoint for session consumers."""
        return _step_predict(
            artifact=artifact,
            dataset=dataset,
            task_spec=task_spec,
            predict_func=predict_func,
            plan=plan,
            covariates=covariates,
            reconciliation_method=reconciliation_method,
        )


def run_forecast(
    data: pd.DataFrame,
    task_spec: TaskSpec,
    covariates: CovariateBundle | None = None,
    mode: Literal["quick", "standard", "strict"] = "standard",
    fit_func: Any | None = None,
    predict_func: Any | None = None,
    monitoring_config: MonitoringConfig | None = None,
    reference_data: pd.DataFrame | None = None,
    repair_strategy: dict[str, Any] | None = None,
    hierarchy: HierarchyStructure | None = None,
    feature_config: FeatureConfig | None = None,
    calibrator_spec: CalibratorSpec | None = None,
    anomaly_spec: AnomalySpec | None = None,
    reconciliation_method: str = "bottom_up",
    dry_run: bool = False,
    session: TSAgentSession | None = None,
) -> RunArtifact | DryRunResult:
    """Execute the complete forecasting pipeline.

    This is the main entry point for tsagentkit. It orchestrates the
    entire workflow: validation -> QA -> dataset creation -> planning ->
    (backtest for standard/strict) -> fit -> predict -> package.

    Args:
        data: Input DataFrame with columns [unique_id, ds, y]
        task_spec: Task specification with horizon, freq, etc.
        covariates: Optional covariate bundle (bundle mode)
        mode: Execution mode:
            - "quick": Skip backtest, fit on all data
            - "standard": Full pipeline with backtest (default)
            - "strict": Fail on any QA issue (no auto-repair)
        fit_func: Optional custom model fit function (fit(dataset, plan))
        predict_func: Optional custom model predict function (predict(dataset, artifact, spec))
        monitoring_config: Optional monitoring configuration
        reference_data: Optional reference data for drift detection
        repair_strategy: Optional QA repair configuration (overrides TaskSpec)
        hierarchy: Optional hierarchy structure for reconciliation
        feature_config: Optional feature configuration for feature engineering
        calibrator_spec: Optional calibration specification
        anomaly_spec: Optional anomaly detection specification
        reconciliation_method: Reconciliation method for hierarchical forecasts.
            One of "bottom_up", "top_down", "middle_out", "ols", "wls",
            "min_trace". Defaults to "bottom_up".
        dry_run: If True, execute only validate → QA → make_plan and
            return a ``DryRunResult`` without fitting or predicting.
            Defaults to False.
        session: Optional TSAgentSession for session-based execution.

    Returns:
        RunArtifact with forecast, metrics, and provenance; or
        DryRunResult if ``dry_run=True``.

    Raises:
        EContractMissingColumn: If required columns missing
        EContractInvalidType: If columns have wrong types
        ESplitRandomForbidden: If data is not temporally ordered
        EFallbackExhausted: If all models fail
    """
    active_session = session or TSAgentSession(mode=mode)
    return active_session.run(
        data=data,
        task_spec=task_spec,
        covariates=covariates,
        mode=mode,
        fit_func=fit_func,
        predict_func=predict_func,
        monitoring_config=monitoring_config,
        reference_data=reference_data,
        repair_strategy=repair_strategy,
        hierarchy=hierarchy,
        feature_config=feature_config,
        calibrator_spec=calibrator_spec,
        anomaly_spec=anomaly_spec,
        reconciliation_method=reconciliation_method,
        dry_run=dry_run,
    )


# Import step functions for backward compatibility
from .pipeline import (
    _fit_predict_with_fallback,
    _step_fit,
    _step_predict,
)

# Re-export for backward compatibility
__all__ = [
    "run_forecast",
    "TSAgentSession",
    "MonitoringConfig",
    "_fit_predict_with_fallback",
    "_step_fit",
    "_step_predict",
]
