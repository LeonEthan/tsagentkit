"""Main forecasting orchestration.

Provides the unified entry point run_forecast() for executing
the complete forecasting pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from tsagentkit.backtest import rolling_backtest
from tsagentkit.contracts import (
    AnomalySpec,
    CalibratorSpec,
    EAnomalyFail,
    ECalibrationFail,
    ECovariateIncompleteKnown,
    ECovariateLeakage,
    ECovariateStaticInvalid,
    EFallbackExhausted,
    EQACriticalIssue,
    ETaskSpecInvalid,
    ForecastResult,
    PanelContract,
    TaskSpec,
    ValidationReport,
    validate_contract,
)
from tsagentkit.covariates import AlignedDataset, CovariateBundle, align_covariates
from tsagentkit.qa import QAReport, run_qa
from tsagentkit.router import make_plan
from tsagentkit.series import TSDataset
from tsagentkit.time import infer_freq
from tsagentkit.utils import drop_future_rows, normalize_quantile_columns

from .packaging import package_run
from .provenance import create_provenance, log_event

if TYPE_CHECKING:
    from tsagentkit.contracts import RunArtifact
    from tsagentkit.features import FeatureConfig, FeatureMatrix
    from tsagentkit.hierarchy import HierarchyStructure


@dataclass
class MonitoringConfig:
    """Configuration for monitoring during forecasting.

    Attributes:
        enabled: Whether monitoring is enabled
        drift_method: Drift detection method ("psi" or "ks")
        drift_threshold: Threshold for drift detection
        check_stability: Whether to compute stability metrics
        jitter_threshold: Threshold for jitter warnings

    Example:
        >>> config = MonitoringConfig(
        ...     enabled=True,
        ...     drift_method="psi",
        ...     drift_threshold=0.2,
        ... )
    """

    enabled: bool = False
    drift_method: Literal["psi", "ks"] = "psi"
    drift_threshold: float | None = None
    check_stability: bool = False
    jitter_threshold: float = 0.1


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
) -> RunArtifact:
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
        monitoring_config: Optional monitoring configuration (v0.2)
        reference_data: Optional reference data for drift detection (v0.2)
        repair_strategy: Optional QA repair configuration (overrides TaskSpec)
        hierarchy: Optional hierarchy structure for reconciliation
        feature_config: Optional feature configuration for feature engineering (v1.0)
        calibrator_spec: Optional calibration specification
        anomaly_spec: Optional anomaly detection specification

    Returns:
        RunArtifact with forecast, metrics, and provenance

    Raises:
        EContractMissingColumn: If required columns missing
        EContractInvalidType: If columns have wrong types
        ESplitRandomForbidden: If data is not temporally ordered
        EFallbackExhausted: If all models fail
    """
    events: list[dict[str, Any]] = []
    qa_repairs: list[dict[str, Any]] = []
    fallbacks_triggered: list[dict[str, Any]] = []
    start_time = time.time()
    column_map: dict[str, str] | None = None
    original_panel_contract = task_spec.panel_contract

    # Step 1: Validate
    data = data.copy()
    step_start = time.time()
    validation, data = _step_validate(data, task_spec)
    events.append(
        log_event(
            step_name="validate",
            status="success" if validation.valid else "failed",
            duration_ms=(time.time() - step_start) * 1000,
            error_code=_get_error_code(validation) if not validation.valid else None,
        )
    )

    if not validation.valid:
        validation.raise_if_errors()

    # Normalize panel columns to canonical names if needed
    data, column_map = TSDataset._normalize_panel_columns(
        data,
        task_spec.panel_contract,
    )
    if column_map:
        task_spec = task_spec.model_copy(update={"panel_contract": PanelContract()})

    # Infer frequency if missing
    if not task_spec.freq:
        if not task_spec.infer_freq:
            raise ETaskSpecInvalid(
                "TaskSpec.freq is required when infer_freq=False.",
                context={"freq": task_spec.freq},
            )
        step_start = time.time()
        try:
            inferred = infer_freq(
                data,
                id_col=task_spec.panel_contract.unique_id_col,
                ds_col=task_spec.panel_contract.ds_col,
            )
            task_spec = task_spec.model_copy(update={"freq": inferred})
            events.append(
                log_event(
                    step_name="infer_freq",
                    status="success",
                    duration_ms=(time.time() - step_start) * 1000,
                    context={"freq": inferred},
                )
            )
        except Exception as e:
            events.append(
                log_event(
                    step_name="infer_freq",
                    status="failed",
                    duration_ms=(time.time() - step_start) * 1000,
                    error_code=type(e).__name__,
                )
            )
            raise

    # Step 2: QA
    step_start = time.time()
    effective_repair_strategy = repair_strategy
    covariate_error: Exception | None = None
    try:
        qa_report = _step_qa(
            data,
            task_spec,
            mode,
            apply_repairs=mode != "strict",
            repair_strategy=effective_repair_strategy,
        )
        qa_repairs = qa_report.repairs
        events.append(
            log_event(
                step_name="qa",
                status="success",
                duration_ms=(time.time() - step_start) * 1000,
            )
        )
    except (ECovariateLeakage, ECovariateIncompleteKnown, ECovariateStaticInvalid) as e:
        covariate_error = e
        if mode == "strict":
            events.append(
                log_event(
                    step_name="qa",
                    status="failed",
                    duration_ms=(time.time() - step_start) * 1000,
                    error_code=type(e).__name__,
                )
            )
            raise
        qa_report = _step_qa(
            data,
            task_spec,
            mode,
            apply_repairs=mode != "strict",
            repair_strategy=effective_repair_strategy,
            skip_covariate_checks=True,
        )
        qa_repairs = qa_report.repairs
        issues = list(qa_report.issues)
        issues.append(
            {
                "type": "covariate_guardrail",
                "error": str(e),
                "severity": "critical",
                "action": "dropped_covariates",
            }
        )
        qa_report = QAReport(
            issues=issues,
            repairs=qa_report.repairs,
            leakage_detected=isinstance(e, ECovariateLeakage),
        )
        events.append(
            log_event(
                step_name="qa",
                status="success",
                duration_ms=(time.time() - step_start) * 1000,
                context={"covariates_dropped": True},
            )
        )
    except Exception as e:
        events.append(
            log_event(
                step_name="qa",
                status="failed",
                duration_ms=(time.time() - step_start) * 1000,
                error_code=type(e).__name__,
            )
        )
        raise

    # Step 2b: Align covariates before dropping future rows (preserve single-table future-known)
    step_start = time.time()
    aligned_dataset: AlignedDataset | None = None
    panel_with_covariates = data.copy()
    if covariate_error is None:
        try:
            aligned_dataset = align_covariates(
                panel_with_covariates,
                task_spec,
                covariates=covariates,
            )
            events.append(
                log_event(
                    step_name="align_covariates",
                    status="success",
                    duration_ms=(time.time() - step_start) * 1000,
                    artifacts_generated=["aligned_covariates"],
                )
            )
        except (ECovariateLeakage, ECovariateIncompleteKnown, ECovariateStaticInvalid) as e:
            covariate_error = e
            events.append(
                log_event(
                    step_name="align_covariates",
                    status="failed",
                    duration_ms=(time.time() - step_start) * 1000,
                    error_code=type(e).__name__,
                )
            )
            if mode == "strict":
                raise
        except Exception as e:
            events.append(
                log_event(
                    step_name="align_covariates",
                    status="failed",
                    duration_ms=(time.time() - step_start) * 1000,
                    error_code=type(e).__name__,
                )
            )
            if mode == "strict":
                raise
    else:
        events.append(
            log_event(
                step_name="align_covariates",
                status="skipped",
                duration_ms=(time.time() - step_start) * 1000,
                context={"covariates_dropped": True},
            )
        )

    # Step 2c: Drop future rows (y is null beyond last observed per series)
    step_start = time.time()
    data, drop_info = drop_future_rows(
        data,
        id_col=task_spec.panel_contract.unique_id_col,
        ds_col=task_spec.panel_contract.ds_col,
        y_col=task_spec.panel_contract.y_col,
    )
    if drop_info:
        qa_repairs.append(drop_info)
        events.append(
            log_event(
                step_name="drop_future_rows",
                status="success",
                duration_ms=(time.time() - step_start) * 1000,
                artifacts_generated=["clean_data"],
            )
        )

    # Step 3: Build Dataset
    step_start = time.time()
    dataset = TSDataset.from_dataframe(data, task_spec, validate=False)
    if hierarchy is not None:
        dataset = dataset.with_hierarchy(hierarchy)
    if aligned_dataset is not None:
        uid_col = task_spec.panel_contract.unique_id_col
        ds_col = task_spec.panel_contract.ds_col
        y_col = task_spec.panel_contract.y_col
        aligned_dataset = AlignedDataset(
            panel=data[[uid_col, ds_col, y_col]].copy(),
            static_x=aligned_dataset.static_x,
            past_x=aligned_dataset.past_x,
            future_x=aligned_dataset.future_x,
            covariate_spec=aligned_dataset.covariate_spec,
            future_index=aligned_dataset.future_index,
        )
        dataset = dataset.with_covariates(
            aligned_dataset,
            panel_with_covariates=panel_with_covariates,
            covariate_bundle=covariates,
        )
    events.append(
        log_event(
            step_name="build_dataset",
            status="success",
            duration_ms=(time.time() - step_start) * 1000,
        )
    )

    # Step 3b: Feature Engineering (v1.0)
    feature_matrix: FeatureMatrix | None = None
    if feature_config is not None:
        step_start = time.time()
        try:
            from tsagentkit.features import FeatureFactory

            factory = FeatureFactory(feature_config)
            feature_matrix = factory.create_features(dataset)
            events.append(
                log_event(
                    step_name="feature_engineering",
                    status="success",
                    duration_ms=(time.time() - step_start) * 1000,
                    artifacts_generated=["feature_matrix"],
                    context={
                        "n_features": len(feature_matrix.feature_cols),
                        "feature_hash": feature_matrix.config_hash,
                    },
                )
            )
        except Exception as e:
            events.append(
                log_event(
                    step_name="feature_engineering",
                    status="failed",
                    duration_ms=(time.time() - step_start) * 1000,
                    error_code=type(e).__name__,
                )
            )
            if mode == "strict":
                raise

    # Step 4: Make Plan
    step_start = time.time()
    plan, route_decision = make_plan(dataset, task_spec, qa_report)
    events.append(
        log_event(
            step_name="make_plan",
            status="success",
            duration_ms=(time.time() - step_start) * 1000,
            artifacts_generated=["plan", "route_decision"],
            context={
                "buckets": route_decision.buckets,
                "reasons": route_decision.reasons,
            },
        )
    )

    if covariate_error is not None:
        if not plan.allow_drop_covariates:
            raise covariate_error
        fallbacks_triggered.append(
            {
                "type": "covariates_dropped",
                "error": str(covariate_error),
            }
        )

    # Step 5: Backtest (if standard or strict mode)
    backtest_report = None
    if mode in ("standard", "strict"):
        step_start = time.time()
        try:
            from tsagentkit.models import fit as default_fit
            from tsagentkit.models import predict as default_predict

            if fit_func is None:
                fit_func = default_fit
            if predict_func is None:
                predict_func = default_predict

            backtest_cfg = task_spec.backtest
            n_windows = backtest_cfg.n_windows
            min_train_size = backtest_cfg.min_train_size
            # Preserve historical default behavior (step=horizon) unless
            # the caller explicitly sets backtest.step.
            step_size = (
                backtest_cfg.step
                if "step" in backtest_cfg.model_fields_set
                else None
            )

            backtest_report = rolling_backtest(
                dataset=dataset,
                spec=task_spec,
                plan=plan,
                fit_func=fit_func,
                predict_func=predict_func,
                n_windows=n_windows,
                step_size=step_size,
                min_train_size=min_train_size,
                route_decision=route_decision,
            )
            events.append(
                log_event(
                    step_name="rolling_backtest",
                    status="success",
                    duration_ms=(time.time() - step_start) * 1000,
                    artifacts_generated=["backtest_report"],
                )
            )
        except Exception as e:
            events.append(
                log_event(
                    step_name="rolling_backtest",
                    status="failed",
                    duration_ms=(time.time() - step_start) * 1000,
                    error_code=type(e).__name__,
                )
            )
            if mode == "strict":
                raise

    # Step 6: Fit Model
    step_start = time.time()

    def on_fallback(from_model: str, to_model: str, error: Exception) -> None:
        fallbacks_triggered.append(
            {
                "from": from_model,
                "to": to_model,
                "error": str(error),
            }
        )

    model_artifact = _step_fit(
        dataset=dataset,
        plan=plan,
        fit_func=fit_func,
        on_fallback=on_fallback,
        covariates=aligned_dataset,
    )

    events.append(
        log_event(
            step_name="fit",
            status="success",
            duration_ms=(time.time() - step_start) * 1000,
            artifacts_generated=["model_artifact"],
        )
    )

    # Step 7: Predict
    step_start = time.time()
    try:
        forecast_df = _step_predict(
            artifact=model_artifact,
            dataset=dataset,
            task_spec=task_spec,
            predict_func=predict_func,
            plan=plan,
            covariates=aligned_dataset,
        )
        events.append(
            log_event(
                step_name="predict",
                status="success",
                duration_ms=(time.time() - step_start) * 1000,
                artifacts_generated=["forecast"],
            )
        )
    except Exception as e:
        events.append(
            log_event(
                step_name="predict",
                status="failed",
                duration_ms=(time.time() - step_start) * 1000,
                error_code=type(e).__name__,
            )
        )
        try:
            model_artifact, forecast_df = _fit_predict_with_fallback(
                dataset=dataset,
                plan=plan,
                task_spec=task_spec,
                fit_func=fit_func,
                predict_func=predict_func,
                covariates=aligned_dataset,
                start_after=model_artifact.model_name,
                initial_error=e,
                on_fallback=on_fallback,
            )
            events.append(
                log_event(
                    step_name="predict_fallback",
                    status="success",
                    duration_ms=(time.time() - step_start) * 1000,
                    artifacts_generated=["forecast"],
                )
            )
        except Exception as fallback_error:
            events.append(
                log_event(
                    step_name="predict_fallback",
                    status="failed",
                    duration_ms=(time.time() - step_start) * 1000,
                    error_code=type(fallback_error).__name__,
                )
            )
            raise

    # Step 8: Calibration (optional)
    calibration_artifact = None
    if calibrator_spec is not None:
        step_start = time.time()
        try:
            from tsagentkit.calibration import apply_calibrator, fit_calibrator

            if backtest_report is None or backtest_report.cv_frame is None:
                raise ECalibrationFail(
                    "Calibration requires CV residuals from backtest.",
                    context={"mode": mode},
                )

            cv_frame = backtest_report.cv_frame
            if hasattr(cv_frame, "df"):
                cv_frame = cv_frame.df

            calibration_artifact = fit_calibrator(
                cv_frame,
                method=calibrator_spec.method,
                level=calibrator_spec.level,
                by=calibrator_spec.by,
            )
            forecast_df = apply_calibrator(forecast_df, calibration_artifact)
            events.append(
                log_event(
                    step_name="calibration",
                    status="success",
                    duration_ms=(time.time() - step_start) * 1000,
                    artifacts_generated=["calibration_artifact"],
                )
            )
        except Exception as e:
            events.append(
                log_event(
                    step_name="calibration",
                    status="failed",
                    duration_ms=(time.time() - step_start) * 1000,
                    error_code=type(e).__name__,
                )
            )
            if mode == "strict":
                raise

    # Step 9: Anomaly Detection (optional)
    anomaly_report = None
    if anomaly_spec is not None:
        step_start = time.time()
        try:
            from tsagentkit.anomaly import detect_anomalies

            uid_col = task_spec.panel_contract.unique_id_col
            ds_col = task_spec.panel_contract.ds_col
            y_col = task_spec.panel_contract.y_col

            actuals = data[[uid_col, ds_col, y_col]].copy()
            merged = forecast_df.merge(
                actuals,
                on=[uid_col, ds_col],
                how="left",
            )
            if merged[y_col].notna().any():
                anomaly_report = detect_anomalies(
                    merged,
                    method=anomaly_spec.method,
                    level=anomaly_spec.level,
                    score=anomaly_spec.score,
                    calibrator=calibration_artifact,
                    strict=(mode == "strict"),
                )
                events.append(
                    log_event(
                        step_name="anomaly_detection",
                        status="success",
                        duration_ms=(time.time() - step_start) * 1000,
                        artifacts_generated=["anomaly_report"],
                    )
                )
            elif mode == "strict":
                raise EAnomalyFail(
                    "No actuals available for anomaly detection.",
                    context={"mode": mode},
                )
        except Exception as e:
            events.append(
                log_event(
                    step_name="anomaly_detection",
                    status="failed",
                    duration_ms=(time.time() - step_start) * 1000,
                    error_code=type(e).__name__,
                )
            )
            if mode == "strict":
                raise

    # Step 10: Drift Detection (v0.2)
    drift_report = None
    if monitoring_config and monitoring_config.enabled and reference_data is not None:
        step_start = time.time()
        try:
            from tsagentkit.monitoring import DriftDetector

            detector = DriftDetector(
                method=monitoring_config.drift_method,
                threshold=monitoring_config.drift_threshold,
            )
            drift_report = detector.detect(
                reference_data=reference_data,
                current_data=data,
            )
            events.append(
                log_event(
                    step_name="drift_detection",
                    status="success",
                    duration_ms=(time.time() - step_start) * 1000,
                    artifacts_generated=["drift_report"],
                )
            )
        except Exception as e:
            events.append(
                log_event(
                    step_name="drift_detection",
                    status="failed",
                    duration_ms=(time.time() - step_start) * 1000,
                    error_code=type(e).__name__,
                )
            )

    # Step 11: Create Provenance
    provenance = create_provenance(
        data=data,
        task_spec=task_spec,
        plan=plan,
        model_config=plan.model_dump() if hasattr(plan, "model_dump") else None,
        qa_repairs=qa_repairs,
        fallbacks_triggered=fallbacks_triggered,
        feature_matrix=feature_matrix,
        drift_report=drift_report,
        column_map=column_map,
        original_panel_contract=(
            original_panel_contract.model_dump()
            if hasattr(original_panel_contract, "model_dump")
            else None
        ),
        route_decision=route_decision,
    )

    # Step 12: Package
    forecast_result = ForecastResult(
        df=forecast_df,
        provenance=provenance,
        model_name=model_artifact.model_name,
        horizon=task_spec.horizon,
    )
    artifact = package_run(
        forecast=forecast_result,
        plan=plan,
        task_spec=task_spec.model_dump() if hasattr(task_spec, "model_dump") else None,
        validation_report=validation.to_dict() if validation else None,
        backtest_report=backtest_report,
        qa_report=qa_report,
        model_artifact=model_artifact,
        provenance=provenance,
        calibration_artifact=calibration_artifact,
        anomaly_report=anomaly_report,
        metadata={
            "mode": mode,
            "total_duration_ms": (time.time() - start_time) * 1000,
            "events": events,
        },
    )

    return artifact


def _step_validate(
    data: pd.DataFrame,
    task_spec: TaskSpec,
) -> tuple[ValidationReport, pd.DataFrame]:
    """Execute validation step."""
    report, normalized = validate_contract(
        data,
        panel_contract=task_spec.panel_contract,
        apply_aggregation=True,
        return_data=True,
    )
    return report, normalized


def _step_qa(
    data: pd.DataFrame,
    task_spec: TaskSpec,
    mode: Literal["quick", "standard", "strict"],
    apply_repairs: bool = False,
    repair_strategy: dict[str, Any] | None = None,
    skip_covariate_checks: bool = False,
) -> QAReport:
    """Execute QA step.

    For v0.1, this is a minimal implementation.
    """
    report = run_qa(
        data,
        task_spec,
        mode,
        apply_repairs=apply_repairs,
        repair_strategy=repair_strategy,
        skip_covariate_checks=skip_covariate_checks,
    )

    if mode == "strict":
        if report.leakage_detected:
            raise ECovariateLeakage("Covariate leakage detected")
        if report.has_critical_issues():
            raise EQACriticalIssue("Critical QA issues detected")

    return report


def _step_fit(
    dataset: TSDataset,
    plan: Any,
    fit_func: Any | None,
    on_fallback: Any | None = None,
    covariates: AlignedDataset | None = None,
) -> Any:
    """Execute fit step with fallback."""
    from tsagentkit.models import fit as default_fit

    if fit_func is None:
        # Use default fit function
        fit_func = default_fit
    kwargs = {"covariates": covariates} if covariates is not None else {}
    if fit_func is default_fit:
        return fit_func(dataset, plan, on_fallback=on_fallback, **kwargs)
    return _call_with_optional_kwargs(fit_func, dataset, plan, **kwargs)


def _step_predict(
    artifact: Any,
    dataset: TSDataset,
    task_spec: TaskSpec,
    predict_func: Any | None,
    plan: Any | None = None,
    covariates: AlignedDataset | None = None,
) -> pd.DataFrame:
    """Execute predict step."""
    if predict_func is None:
        # Use default predict function
        from tsagentkit.models import predict as default_predict

        predict_func = default_predict

    kwargs = {"covariates": covariates} if covariates is not None else {}
    forecast = _call_with_optional_kwargs(predict_func, dataset, artifact, task_spec, **kwargs)
    if isinstance(forecast, ForecastResult):
        forecast = forecast.df

    if "model" not in forecast.columns:
        model_name = getattr(artifact, "model_name", None)
        if model_name is None and hasattr(artifact, "metadata"):
            model_name = artifact.metadata.get("model_name") if artifact.metadata else None
        forecast = forecast.copy()
        forecast["model"] = model_name or "model"

    # Apply reconciliation if hierarchical
    if plan and dataset.is_hierarchical() and dataset.hierarchy:
        from tsagentkit.hierarchy import ReconciliationMethod, reconcile_forecasts

        method_str = "bottom_up"
        method_map = {
            "bottom_up": ReconciliationMethod.BOTTOM_UP,
            "top_down": ReconciliationMethod.TOP_DOWN,
            "middle_out": ReconciliationMethod.MIDDLE_OUT,
            "ols": ReconciliationMethod.OLS,
            "wls": ReconciliationMethod.WLS,
            "min_trace": ReconciliationMethod.MIN_TRACE,
        }
        method = method_map.get(method_str, ReconciliationMethod.BOTTOM_UP)

        forecast = reconcile_forecasts(
            base_forecasts=forecast,
            structure=dataset.hierarchy,
            method=method,
        )

    forecast = normalize_quantile_columns(forecast)
    if {"unique_id", "ds"}.issubset(forecast.columns):
        forecast = forecast.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    return forecast


def _fit_predict_with_fallback(
    dataset: TSDataset,
    plan: Any,
    task_spec: TaskSpec,
    fit_func: Any | None,
    predict_func: Any | None,
    covariates: AlignedDataset | None = None,
    start_after: str | None = None,
    initial_error: Exception | None = None,
    on_fallback: Any | None = None,
) -> tuple[Any, pd.DataFrame]:
    """Fit and predict with fallback across remaining candidates."""
    from tsagentkit.models import fit as default_fit
    from tsagentkit.models import predict as default_predict

    fit_callable = fit_func or default_fit
    predict_callable = predict_func or default_predict

    candidates = list(getattr(plan, "candidate_models", []) or [])
    start_idx = 0
    if start_after in candidates:
        start_idx = candidates.index(start_after) + 1
    remaining = candidates[start_idx:]

    last_error: Exception | None = None

    if start_after and remaining and on_fallback and initial_error is not None:
        on_fallback(start_after, remaining[0], initial_error)

    for i, model_name in enumerate(remaining):
        plan_for_model = plan
        if hasattr(plan, "model_copy"):
            plan_for_model = plan.model_copy(update={"candidate_models": [model_name]})

        try:
            artifact = _call_with_optional_kwargs(
                fit_callable,
                dataset,
                plan_for_model,
                covariates=covariates,
            )
        except Exception as e:
            last_error = e
            if on_fallback and i < len(remaining) - 1:
                on_fallback(model_name, remaining[i + 1], e)
            continue

        try:
            forecast = _step_predict(
                artifact=artifact,
                dataset=dataset,
                task_spec=task_spec,
                predict_func=predict_callable,
                plan=plan,
                covariates=covariates,
            )
            return artifact, forecast
        except Exception as e:
            last_error = e
            if on_fallback and i < len(remaining) - 1:
                on_fallback(model_name, remaining[i + 1], e)
            continue

    raise EFallbackExhausted(
        f"All models failed during predict fallback. Last error: {last_error}",
        context={
            "models_attempted": remaining,
            "last_error": str(last_error),
        },
    )


def _get_error_code(validation: ValidationReport) -> str | None:
    """Extract error code from validation report."""
    if validation.errors:
        return validation.errors[0].get("code")
    return None


def _call_with_optional_kwargs(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a function with only supported keyword arguments."""
    if not kwargs:
        return func(*args)

    try:
        import inspect

        params = inspect.signature(func).parameters
        accepted = {k: v for k, v in kwargs.items() if k in params}
        return func(*args, **accepted)
    except Exception:
        # Fall back to direct call if signature inspection fails
        return func(*args, **kwargs)
