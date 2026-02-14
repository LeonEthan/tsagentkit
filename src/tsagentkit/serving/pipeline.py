"""Forecasting pipeline with step-wise execution.

Provides a clean, extensible pipeline architecture that replaces the
monolithic _run_forecast_impl function with focused step methods.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import pandas as pd

from tsagentkit.backtest import multi_model_backtest
from tsagentkit.contracts import (
    AnomalySpec,
    CalibratorSpec,
    ECovariateIncompleteKnown,
    ECovariateLeakage,
    ECovariateStaticInvalid,
    ETaskSpecInvalid,
    PanelContract,
    TaskSpec,
    TSAgentKitError,
    ValidationReport,
    validate_contract,
)
from tsagentkit.covariates import AlignedDataset, CovariateBundle, align_covariates
from tsagentkit.qa import QAReport, run_qa
from tsagentkit.router import make_plan
from tsagentkit.router.fallback import should_trigger_fallback
from tsagentkit.series import TSDataset
from tsagentkit.time import infer_freq
from tsagentkit.utils import drop_future_rows

from .execution import (
    FitFunc,
    PredictFunc,
    fit_per_series_models,
    fit_predict_with_fallback,
    fit_single_model,
    predict_per_series_models,
    predict_single_model,
)
from .forecast_postprocess import (
    normalize_and_sort_forecast,
    postprocess_forecast,
    resolve_model_name,
)
from .provenance import log_event
from .refinement import calibrate_forecast, detect_data_drift, detect_forecast_anomalies
from .result_assembly import assemble_run_artifact, build_dry_run_result

if TYPE_CHECKING:
    from tsagentkit.anomaly import AnomalyReport
    from tsagentkit.backtest import MultiModelBacktestReport
    from tsagentkit.calibration import CalibratorArtifact
    from tsagentkit.contracts import DryRunResult, PlanSpec, RouteDecision, RunArtifact
    from tsagentkit.features import FeatureConfig, FeatureMatrix
    from tsagentkit.hierarchy import HierarchyStructure
    from tsagentkit.monitoring import DriftReport


@dataclass
class MonitoringConfig:
    """Configuration for monitoring during forecasting."""

    enabled: bool = False
    drift_method: Literal["psi", "ks"] = "psi"
    drift_threshold: float | None = None
    check_stability: bool = False
    jitter_threshold: float = 0.1


@dataclass
class PipelineState:
    """Mutable state container for pipeline execution.

    This separates mutable state from the pipeline configuration,
    making the pipeline itself easier to reason about.
    """

    # Input data (mutated through steps)
    data: pd.DataFrame
    task_spec: TaskSpec

    # Step outputs
    validation: ValidationReport | None = None
    qa_report: QAReport | None = None
    covariate_error: Exception | None = None
    aligned_dataset: AlignedDataset | None = None
    dataset: TSDataset | None = None
    feature_matrix: FeatureMatrix | None = None
    plan: PlanSpec | None = None
    route_decision: RouteDecision | None = None
    backtest_report: MultiModelBacktestReport | None = None
    selection_map: dict[str, str] | None = None
    model_artifact: object | None = None
    model_artifacts: dict[str, object] | None = None
    forecast_df: pd.DataFrame | None = None
    calibration_artifact: CalibratorArtifact | None = None
    anomaly_report: AnomalyReport | None = None
    drift_report: DriftReport | None = None

    # Metadata
    column_map: dict[str, str] | None = None
    original_panel_contract: PanelContract | None = None
    qa_repairs: list[dict[str, object]] = field(default_factory=list)


class ForecastPipeline:
    """Execute the complete forecasting pipeline as discrete steps.

    Each step is a focused method that performs one operation,
    logs events, and handles errors appropriately. State is maintained
    in the PipelineState dataclass rather than local variables.

    Example:
        >>> pipeline = ForecastPipeline(data, task_spec)
        >>> result = pipeline.run()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        task_spec: TaskSpec,
        covariates: CovariateBundle | None = None,
        mode: Literal["quick", "standard", "strict"] = "standard",
        fit_func: FitFunc | None = None,
        predict_func: PredictFunc | None = None,
        monitoring_config: MonitoringConfig | None = None,
        reference_data: pd.DataFrame | None = None,
        repair_strategy: dict[str, object] | None = None,
        hierarchy: HierarchyStructure | None = None,
        feature_config: FeatureConfig | None = None,
        calibrator_spec: CalibratorSpec | None = None,
        anomaly_spec: AnomalySpec | None = None,
        reconciliation_method: str = "bottom_up",
        dry_run: bool = False,
    ):
        # Configuration (immutable during execution)
        self.mode = mode
        self.fit_func = fit_func
        self.predict_func = predict_func
        self.monitoring_config = monitoring_config or MonitoringConfig()
        self.reference_data = reference_data
        self.repair_strategy = repair_strategy
        self.hierarchy = hierarchy
        self.feature_config = feature_config
        self.calibrator_spec = calibrator_spec
        self.anomaly_spec = anomaly_spec
        self.reconciliation_method = reconciliation_method
        self.dry_run = dry_run
        self.covariates = covariates

        # State (mutable during execution)
        self.state = PipelineState(
            data=data.copy(),
            task_spec=task_spec,
            original_panel_contract=task_spec.panel_contract,
        )

        # Event tracking
        self.events: list[dict[str, object]] = []
        self.fallbacks_triggered: list[dict[str, object]] = []
        self.degradation_events: list[dict[str, object]] = []
        self.start_time = time.time()

    def run(self) -> RunArtifact | DryRunResult:
        """Execute the complete pipeline."""
        return self._execute_pipeline()

    def _execute_pipeline(self) -> RunArtifact | DryRunResult:
        """Internal pipeline execution with all steps."""
        # Phase 1: Data Preparation
        self._step_validate()
        self._step_infer_frequency()
        self._step_qa()
        self._step_align_covariates()
        self._step_drop_future_rows()
        self._step_build_dataset()
        self._step_feature_engineering()

        # Phase 2: Planning
        self._step_make_plan()

        # Dry run returns early
        if self.dry_run:
            return self._build_dry_run_result()

        # Phase 3: Model Selection & Training
        self._step_backtest()
        self._step_fit()

        # Phase 4: Prediction & Refinement
        self._step_predict()
        self._step_calibration()
        self._step_anomaly_detection()
        self._step_drift_detection()

        # Phase 5: Packaging
        return self._step_package()

    def _step_validate(self) -> None:
        """Validate input data against contract."""
        step_start = time.time()

        report, normalized = validate_contract(
            self.state.data,
            panel_contract=self.state.task_spec.panel_contract,
            apply_aggregation=True,
            return_data=True,
        )
        self.state.validation = report
        self.state.data = normalized

        self._log_event("validate", "success" if report.valid else "failed", step_start)

        if not report.valid:
            report.raise_if_errors()

        # Normalize panel columns
        self.state.data, self.state.column_map = TSDataset._normalize_panel_columns(
            self.state.data,
            self.state.task_spec.panel_contract,
        )
        if self.state.column_map:
            self.state.task_spec = self.state.task_spec.model_copy(
                update={"panel_contract": PanelContract()}
            )

    def _step_infer_frequency(self) -> None:
        """Infer frequency if not explicitly provided."""
        if self.state.task_spec.freq:
            return

        if not self.state.task_spec.infer_freq:
            raise ETaskSpecInvalid(
                "TaskSpec.freq is required when infer_freq=False.",
                context={"freq": self.state.task_spec.freq},
            )

        step_start = time.time()
        try:
            inferred = infer_freq(
                self.state.data,
                id_col=self.state.task_spec.panel_contract.unique_id_col,
                ds_col=self.state.task_spec.panel_contract.ds_col,
            )
            self.state.task_spec = self.state.task_spec.model_copy(update={"freq": inferred})
            self._log_event("infer_freq", "success", step_start, {"freq": inferred})
        except Exception:
            self._log_event("infer_freq", "failed", step_start)
            raise

    def _step_qa(self) -> None:
        """Run quality assurance checks."""
        step_start = time.time()

        try:
            report = run_qa(
                self.state.data,
                self.state.task_spec,
                self.mode,
                apply_repairs=self.mode != "strict",
                repair_strategy=self.repair_strategy,
            )
            self.state.qa_report = report
            self.state.qa_repairs = report.repairs
            self._log_event("qa", "success", step_start)
        except (ECovariateLeakage, ECovariateIncompleteKnown, ECovariateStaticInvalid) as e:
            self._handle_covariate_error(e, step_start, "qa", rerun_qa=True)
        except Exception:
            self._log_event("qa", "failed", step_start)
            raise

    def _handle_covariate_error(
        self,
        error: Exception,
        step_start: float,
        step_name: str,
        *,
        rerun_qa: bool = False,
    ) -> None:
        """Handle covariate-related errors with graceful degradation.

        Args:
            error: The covariate error that occurred
            step_start: Timestamp when the step started
            step_name: Name of the step ("qa" or "align_covariates")
            rerun_qa: Whether to re-run QA without covariate checks (QA step only)
        """
        self.state.covariate_error = error

        # Log failure with error code
        self._log_event(step_name, "failed", step_start, error_code=type(error).__name__)

        # In strict mode, always raise
        if self.mode == "strict":
            raise error

        # For QA step with rerun enabled, re-run without covariate checks
        if rerun_qa and step_name == "qa":
            report = run_qa(
                self.state.data,
                self.state.task_spec,
                self.mode,
                apply_repairs=self.mode != "strict",
                repair_strategy=self.repair_strategy,
                skip_covariate_checks=True,
            )

            # Augment report with covariate issue
            issues = list(report.issues)
            issues.append({
                "type": "covariate_guardrail",
                "error": str(error),
                "severity": "critical",
                "action": "dropped_covariates",
            })
            self.state.qa_report = QAReport(
                issues=issues,
                repairs=report.repairs,
                leakage_detected=isinstance(error, ECovariateLeakage),
            )
            self.state.qa_repairs = report.repairs
            self._log_event("qa", "success", step_start, {"covariates_dropped": True})

        # Record degradation event
        self.degradation_events.append({
            "type": "covariates_dropped",
            "step": step_name,
            "error_type": type(error).__name__,
            "error": str(error),
            "action": "dropped_covariates",
        })

    def _step_align_covariates(self) -> None:
        """Align covariates with the target data."""
        if self.state.covariate_error:
            self._skip_step("align_covariates", {"covariates_dropped": True})
            return

        step_start = time.time()
        try:
            self.state.aligned_dataset = align_covariates(
                self.state.data.copy(),
                self.state.task_spec,
                covariates=self.covariates,
            )
            self._log_event(
                "align_covariates", "success", step_start, artifacts=["aligned_covariates"]
            )
        except (ECovariateLeakage, ECovariateIncompleteKnown, ECovariateStaticInvalid) as e:
            self._handle_covariate_error(e, step_start, "align_covariates")
        except Exception:
            self._log_event("align_covariates", "failed", step_start)
            if self.mode == "strict":
                raise

    def _step_drop_future_rows(self) -> None:
        """Drop rows where target is null (future data)."""
        step_start = time.time()

        self.state.data, drop_info = drop_future_rows(
            self.state.data,
            id_col=self.state.task_spec.panel_contract.unique_id_col,
            ds_col=self.state.task_spec.panel_contract.ds_col,
            y_col=self.state.task_spec.panel_contract.y_col,
        )

        if drop_info:
            self.state.qa_repairs.append(drop_info)
            self._log_event(
                "drop_future_rows", "success", step_start, artifacts=["clean_data"]
            )

    def _step_build_dataset(self) -> None:
        """Build TSDataset from prepared data."""
        step_start = time.time()

        dataset = TSDataset.from_dataframe(self.state.data, self.state.task_spec, validate=False)

        if self.hierarchy is not None:
            dataset = dataset.with_hierarchy(self.hierarchy)

        if self.state.aligned_dataset is not None:
            dataset = self._attach_covariates(dataset)

        self.state.dataset = dataset
        self._log_event("build_dataset", "success", step_start)

    def _attach_covariates(self, dataset: TSDataset) -> TSDataset:
        """Attach aligned covariates to dataset."""
        spec = self.state.task_spec.panel_contract
        aligned = self.state.aligned_dataset

        # Rebuild aligned dataset with normalized panel
        aligned_dataset = AlignedDataset(
            panel=self.state.data[[spec.unique_id_col, spec.ds_col, spec.y_col]].copy(),
            static_x=aligned.static_x,
            past_x=aligned.past_x,
            future_x=aligned.future_x,
            covariate_spec=aligned.covariate_spec,
            future_index=aligned.future_index,
        )

        return dataset.with_covariates(
            aligned_dataset,
            panel_with_covariates=self.state.data.copy(),
            covariate_bundle=self.covariates,
        )

    def _step_feature_engineering(self) -> None:
        """Apply feature engineering if configured."""
        if self.feature_config is None:
            return

        step_start = time.time()
        try:
            from tsagentkit.features import FeatureFactory

            factory = FeatureFactory(self.feature_config)
            self.state.feature_matrix = factory.create_features(self.state.dataset)
            self._log_event(
                "feature_engineering",
                "success",
                step_start,
                artifacts=["feature_matrix"],
                context={
                    "n_features": len(self.state.feature_matrix.feature_cols),
                    "feature_hash": self.state.feature_matrix.config_hash,
                },
            )
        except Exception:
            self._log_event("feature_engineering", "failed", step_start)
            if self.mode == "strict":
                raise

    def _step_make_plan(self) -> None:
        """Create execution plan and route decision."""
        step_start = time.time()

        self.state.plan, self.state.route_decision = make_plan(
            self.state.dataset,
            self.state.task_spec,
            self.state.qa_report,
        )

        self._log_event(
            "make_plan",
            "success",
            step_start,
            artifacts=["plan", "route_decision"],
            context={
                "buckets": self.state.route_decision.buckets,
                "reasons": self.state.route_decision.reasons,
            },
        )

        # Check if covariates should be dropped per plan
        if self.state.covariate_error is not None:
            if not self.state.plan.allow_drop_covariates:
                raise self.state.covariate_error
            self.fallbacks_triggered.append({
                "type": "covariates_dropped",
                "error": str(self.state.covariate_error),
            })

    def _build_dry_run_result(self) -> DryRunResult:
        """Build DryRunResult for dry run mode."""
        return build_dry_run_result(
            validation=self.state.validation,
            qa_report=self.state.qa_report,
            plan=self.state.plan,
            route_decision=self.state.route_decision,
            task_spec=self.state.task_spec,
        )

    def _step_backtest(self) -> None:
        """Run backtest for model selection (standard/strict mode)."""
        if self.mode not in ("standard", "strict"):
            return

        step_start = time.time()
        try:
            from tsagentkit.models import fit as default_fit
            from tsagentkit.models import predict as default_predict

            fit_func = self.fit_func or default_fit
            predict_func = self.predict_func or default_predict
            backtest_cfg = self.state.task_spec.backtest

            multi_report = multi_model_backtest(
                dataset=self.state.dataset,
                spec=self.state.task_spec,
                plan=self.state.plan,
                selection_metric=backtest_cfg.selection_metric,
                n_windows=backtest_cfg.n_windows,
                min_train_size=backtest_cfg.min_train_size,
                step_size=backtest_cfg.step if backtest_cfg.step is not None else self.state.task_spec.h,
                fit_func=fit_func,
                predict_func=predict_func,
            )
            self.state.backtest_report = multi_report
            self.state.selection_map = multi_report.selection_map

            self._log_event(
                "backtest",
                "success",
                step_start,
                artifacts=["backtest_report", "selection_map"],
                context={
                    "n_candidates": len(multi_report.candidate_models),
                    "model_distribution": multi_report.get_model_distribution(),
                },
            )
        except Exception:
            self._log_event("backtest", "failed", step_start)
            if self.mode == "strict":
                raise

    def _step_fit(self) -> None:
        """Fit model(s) on the dataset."""
        step_start = time.time()

        if self.state.selection_map is not None:
            self._fit_per_series(step_start)
        else:
            self._fit_single(step_start)

    def _fit_per_series(self, step_start: float) -> None:
        """Fit models grouped by winning model selection.

        Groups bottom-level series by their selected winning model from backtest,
        then fits each model once on its assigned series subset. This ensures
        efficient training while respecting per-series model selection results.
        """
        self.state.model_artifacts = fit_per_series_models(
            dataset=self.state.dataset,
            plan=self.state.plan,
            selection_map=self.state.selection_map,
            fit_func=self.fit_func,
            on_fallback=self._on_fallback,
        )
        self._log_event(
            "fit_per_series",
            "success",
            step_start,
            artifacts=["model_artifacts"],
            context={
                "n_models": len(self.state.model_artifacts),
                "models": list(self.state.model_artifacts.keys()),
            },
        )

    def _fit_single(self, step_start: float) -> None:
        """Fit single model for all series."""
        self.state.model_artifact = fit_single_model(
            dataset=self.state.dataset,
            plan=self.state.plan,
            fit_func=self.fit_func,
            on_fallback=self._on_fallback,
            covariates=self.state.aligned_dataset,
        )

        self._log_event("fit", "success", step_start, artifacts=["model_artifact"])

    def _on_fallback(self, from_model: str, to_model: str, error: Exception) -> None:
        """Callback when model fallback occurs."""
        self.fallbacks_triggered.append({
            "from": from_model,
            "to": to_model,
            "error_code": self._error_code_from_exception(error),
            "error": str(error),
        })

    def _step_predict(self) -> None:
        """Generate forecasts from fitted model(s)."""
        step_start = time.time()

        try:
            if self.state.selection_map is not None:
                self._predict_per_series(step_start)
            else:
                self._predict_single(step_start)
        except Exception as e:
            self._handle_predict_error(e, step_start)

    def _predict_per_series(self, step_start: float) -> None:
        """Predict using per-series models with hierarchical reconciliation."""
        self.state.forecast_df = predict_per_series_models(
            dataset=self.state.dataset,
            artifacts=self.state.model_artifacts,
            selection_map=self.state.selection_map,
            task_spec=self.state.task_spec,
            predict_func=self.predict_func,
        )
        # Apply hierarchical reconciliation if needed
        self.state.forecast_df = postprocess_forecast(
            self.state.forecast_df,
            model_name="per_series",
            dataset=self.state.dataset,
            plan=self.state.plan,
            reconciliation_method=self.reconciliation_method,
        )
        self.state.forecast_df = self._normalize_forecast(self.state.forecast_df)
        self._log_event("predict_per_series", "success", step_start, artifacts=["forecast"])

    def _predict_single(self, step_start: float) -> None:
        """Predict using single model."""
        self.state.forecast_df = predict_single_model(
            self.state.dataset,
            self.state.model_artifact,
            self.state.task_spec,
            self.predict_func,
            self.state.aligned_dataset,
        )
        self.state.forecast_df = postprocess_forecast(
            self.state.forecast_df,
            model_name=resolve_model_name(self.state.model_artifact),
            dataset=self.state.dataset,
            plan=self.state.plan,
            reconciliation_method=self.reconciliation_method,
        )

        self._log_event("predict", "success", step_start, artifacts=["forecast"])

    def _handle_predict_error(self, error: Exception, step_start: float) -> None:
        """Handle prediction errors with fallback."""
        self._log_event("predict", "failed", step_start, error_code=self._error_code_from_exception(error))

        if not should_trigger_fallback(error) or self.state.selection_map is not None:
            raise error

        # Attempt fallback
        try:
            self._fit_predict_fallback(error, step_start)
        except Exception as fallback_error:
            self._log_event(
                "predict_fallback",
                "failed",
                step_start,
                error_code=self._error_code_from_exception(fallback_error),
            )
            raise

    def _fit_predict_fallback(self, initial_error: Exception, step_start: float) -> None:
        """Fit and predict with fallback to remaining candidates."""
        start_after = self.state.model_artifact.model_name if self.state.model_artifact else None

        artifact, forecast = fit_predict_with_fallback(
            dataset=self.state.dataset,
            plan=self.state.plan,
            task_spec=self.state.task_spec,
            fit_func=self.fit_func,
            predict_func=self.predict_func,
            covariates=self.state.aligned_dataset,
            start_after=start_after,
            initial_error=initial_error,
            on_fallback=self._on_fallback,
            reconciliation_method=self.reconciliation_method,
        )

        self.state.model_artifact = artifact
        self.state.forecast_df = self._normalize_forecast(forecast)
        self._log_event("predict_fallback", "success", step_start, artifacts=["forecast"])

    def _normalize_forecast(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """Normalize and sort forecast dataframe."""
        return normalize_and_sort_forecast(forecast)

    def _step_calibration(self) -> None:
        """Apply calibration if configured."""
        if self.calibrator_spec is None:
            return

        step_start = time.time()
        try:
            cv_frame = getattr(self.state.backtest_report, "cv_frame", None)
            self.state.forecast_df, self.state.calibration_artifact = calibrate_forecast(
                forecast=self.state.forecast_df,
                cv_frame=cv_frame,
                calibrator_spec=self.calibrator_spec,
            )
            self._log_event(
                "calibration", "success", step_start, artifacts=["calibration_artifact"]
            )
        except Exception:
            self._log_event("calibration", "failed", step_start)
            if self.mode == "strict":
                raise

    def _step_anomaly_detection(self) -> None:
        """Run anomaly detection if configured."""
        if self.anomaly_spec is None:
            return

        step_start = time.time()
        try:
            self.state.anomaly_report = detect_forecast_anomalies(
                forecast=self.state.forecast_df,
                historical_data=self.state.data,
                panel_contract=self.state.task_spec.panel_contract,
                anomaly_spec=self.anomaly_spec,
                calibration_artifact=self.state.calibration_artifact,
                strict=(self.mode == "strict"),
            )
            if self.state.anomaly_report is not None:
                self._log_event(
                    "anomaly_detection", "success", step_start, artifacts=["anomaly_report"]
                )
        except Exception:
            self._log_event("anomaly_detection", "failed", step_start)
            if self.mode == "strict":
                raise

    def _step_drift_detection(self) -> None:
        """Run drift detection if configured."""
        if not (self.monitoring_config.enabled and self.reference_data is not None):
            return

        step_start = time.time()
        try:
            self.state.drift_report = detect_data_drift(
                reference_data=self.reference_data,
                current_data=self.state.data,
                method=self.monitoring_config.drift_method,
                threshold=self.monitoring_config.drift_threshold,
            )
            self._log_event(
                "drift_detection", "success", step_start, artifacts=["drift_report"]
            )
        except Exception:
            self._log_event("drift_detection", "failed", step_start)
            # Drift detection failures are non-critical

    def _step_package(self) -> RunArtifact:
        """Package all results into RunArtifact."""
        return assemble_run_artifact(
            data=self.state.data,
            task_spec=self.state.task_spec,
            plan=self.state.plan,
            forecast_df=self.state.forecast_df,
            validation=self.state.validation,
            backtest_report=self.state.backtest_report,
            qa_report=self.state.qa_report,
            model_artifact=self.state.model_artifact,
            model_artifacts=self.state.model_artifacts,
            selection_map=self.state.selection_map,
            qa_repairs=self.state.qa_repairs,
            fallbacks_triggered=self.fallbacks_triggered,
            feature_matrix=self.state.feature_matrix,
            drift_report=self.state.drift_report,
            column_map=self.state.column_map,
            original_panel_contract=self.state.original_panel_contract,
            route_decision=self.state.route_decision,
            calibration_artifact=self.state.calibration_artifact,
            anomaly_report=self.state.anomaly_report,
            degradation_events=self.degradation_events,
            mode=self.mode,
            total_duration_ms=(time.time() - self.start_time) * 1000,
            events=self.events,
        )

    def _log_event(
        self,
        step_name: str,
        status: str,
        step_start: float,
        artifacts: list[str] | None = None,
        context: dict[str, object] | None = None,
        error_code: str | None = None,
    ) -> None:
        """Log a pipeline event."""
        duration_ms = (time.time() - step_start) * 1000
        event = log_event(
            step_name=step_name,
            status=status,
            duration_ms=duration_ms,
            artifacts_generated=artifacts,
            context=context,
            error_code=error_code,
        )
        self.events.append(event)

    def _skip_step(self, step_name: str, context: dict[str, object] | None = None) -> None:
        """Log a skipped step."""
        self.events.append(
            log_event(
                step_name=step_name,
                status="skipped",
                duration_ms=0,
                context=context,
            )
        )

    @staticmethod
    def _error_code_from_exception(error: Exception) -> str:
        """Extract error code from exception."""
        if isinstance(error, TSAgentKitError):
            return error.error_code
        return type(error).__name__


def _run_forecast_impl(
    data: pd.DataFrame,
    task_spec: TaskSpec,
    covariates: CovariateBundle | None = None,
    mode: Literal["quick", "standard", "strict"] = "standard",
    fit_func: FitFunc | None = None,
    predict_func: PredictFunc | None = None,
    monitoring_config: MonitoringConfig | None = None,
    reference_data: pd.DataFrame | None = None,
    repair_strategy: dict[str, object] | None = None,
    hierarchy: HierarchyStructure | None = None,
    feature_config: FeatureConfig | None = None,
    calibrator_spec: CalibratorSpec | None = None,
    anomaly_spec: AnomalySpec | None = None,
    reconciliation_method: str = "bottom_up",
    dry_run: bool = False,
) -> RunArtifact | DryRunResult:
    """Execute the complete forecasting pipeline using the refactored Pipeline class.

    This function is a thin wrapper around ForecastPipeline.run() that maintains
    backward compatibility with the existing API.
    """
    pipeline = ForecastPipeline(
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
    return pipeline.run()
