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
        """Fit step entrypoint for session consumers.

        Uses the ForecastPipeline._fit_single method internally.
        """
        from tsagentkit.models import fit as default_fit
        from tsagentkit.utils.compat import call_with_optional_kwargs

        effective_fit_func = fit_func or default_fit
        kwargs = {"covariates": covariates} if covariates is not None else {}

        if effective_fit_func is default_fit:
            return effective_fit_func(dataset, plan, on_fallback=on_fallback, **kwargs)
        return call_with_optional_kwargs(effective_fit_func, dataset, plan, **kwargs)

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
        """Predict step entrypoint for session consumers.

        Uses the models.predict with reconciliation if hierarchical.
        """
        from tsagentkit.models import predict as default_predict
        from tsagentkit.utils.compat import call_with_optional_kwargs
        from tsagentkit.utils import normalize_quantile_columns
        from tsagentkit.contracts import ForecastResult

        effective_predict_func = predict_func or default_predict
        kwargs = {"covariates": covariates} if covariates is not None else {}

        forecast = call_with_optional_kwargs(
            effective_predict_func, dataset, artifact, task_spec, **kwargs
        )

        if isinstance(forecast, ForecastResult):
            forecast = forecast.df

        # Add model column if missing
        if "model" not in forecast.columns:
            model_name = getattr(artifact, "model_name", None)
            if model_name is None and hasattr(artifact, "metadata"):
                model_name = artifact.metadata.get("model_name") if artifact.metadata else None
            forecast = forecast.copy()
            forecast["model"] = model_name or "model"

        # Apply reconciliation if hierarchical
        if plan and dataset.is_hierarchical() and dataset.hierarchy:
            from tsagentkit.hierarchy import ReconciliationMethod, reconcile_forecasts

            method = ReconciliationMethod.from_string(reconciliation_method)
            forecast = reconcile_forecasts(
                base_forecasts=forecast,
                structure=dataset.hierarchy,
                method=method,
            )

        forecast = normalize_quantile_columns(forecast)
        if {"unique_id", "ds"}.issubset(forecast.columns):
            forecast = forecast.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        return forecast


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


# Re-export for backward compatibility
__all__ = [
    "run_forecast",
    "TSAgentSession",
    "MonitoringConfig",
]
