"""Main forecasting orchestration.

Provides the unified entry point run_forecast() for executing
the complete forecasting pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from tsagentkit.backtest import rolling_backtest
from tsagentkit.contracts import (
    EModelFitFailed,
    TaskSpec,
    ValidationReport,
    validate_contract,
)
from tsagentkit.qa import QAReport
from tsagentkit.router import FallbackLadder, execute_with_fallback, make_plan
from tsagentkit.series import TSDataset

from .packaging import RunArtifact, package_run
from .provenance import create_provenance, log_event


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
    mode: Literal["quick", "standard", "strict"] = "standard",
    fit_func: Any | None = None,
    predict_func: Any | None = None,
    monitoring_config: MonitoringConfig | None = None,
    reference_data: pd.DataFrame | None = None,
) -> RunArtifact:
    """Execute the complete forecasting pipeline.

    This is the main entry point for tsagentkit. It orchestrates the
    entire workflow: validation -> QA -> dataset creation -> planning ->
    (backtest for standard/strict) -> fit -> predict -> package.

    Args:
        data: Input DataFrame with columns [unique_id, ds, y]
        task_spec: Task specification with horizon, freq, etc.
        mode: Execution mode:
            - "quick": Skip backtest, fit on all data
            - "standard": Full pipeline with backtest (default)
            - "strict": Fail on any QA issue (no auto-repair)
        fit_func: Optional custom model fit function
        predict_func: Optional custom model predict function
        monitoring_config: Optional monitoring configuration (v0.2)
        reference_data: Optional reference data for drift detection (v0.2)

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

    # Step 1: Validate
    step_start = time.time()
    validation = _step_validate(data)
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

    # Step 2: QA
    step_start = time.time()
    qa_report = _step_qa(data, task_spec, mode)
    events.append(
        log_event(
            step_name="qa",
            status="success",
            duration_ms=(time.time() - step_start) * 1000,
        )
    )

    # Step 3: Build Dataset
    step_start = time.time()
    dataset = TSDataset.from_dataframe(data, task_spec, validate=False)
    events.append(
        log_event(
            step_name="build_dataset",
            status="success",
            duration_ms=(time.time() - step_start) * 1000,
        )
    )

    # Step 4: Make Plan
    step_start = time.time()
    plan = make_plan(dataset, task_spec, qa_report)
    events.append(
        log_event(
            step_name="make_plan",
            status="success",
            duration_ms=(time.time() - step_start) * 1000,
            artifacts_generated=["plan"],
        )
    )

    # Step 5: Backtest (if standard or strict mode)
    backtest_report = None
    if mode in ("standard", "strict") and fit_func and predict_func:
        step_start = time.time()
        try:
            backtest_report = rolling_backtest(
                dataset=dataset,
                spec=task_spec,
                plan=plan,
                fit_func=fit_func,
                predict_func=predict_func,
                n_windows=3,
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

    model_artifact, model_name = _step_fit(
        dataset=dataset,
        plan=plan,
        fit_func=fit_func,
        on_fallback=on_fallback,
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
    forecast_df = _step_predict(
        model=model_artifact,
        dataset=dataset,
        horizon=task_spec.horizon,
        predict_func=predict_func,
    )
    events.append(
        log_event(
            step_name="predict",
            status="success",
            duration_ms=(time.time() - step_start) * 1000,
            artifacts_generated=["forecast"],
        )
    )

    # Step 8: Drift Detection (v0.2)
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

    # Step 9: Create Provenance
    provenance = create_provenance(
        data=data,
        task_spec=task_spec,
        plan=plan,
        model_config=plan.config,
        qa_repairs=qa_repairs,
        fallbacks_triggered=fallbacks_triggered,
        drift_report=drift_report,
    )

    # Step 10: Package
    artifact = package_run(
        forecast=forecast_df,
        plan=plan,
        model_name=model_name,
        backtest_report=backtest_report,
        qa_report=qa_report,
        model_artifact=model_artifact,
        provenance=provenance,
        metadata={
            "mode": mode,
            "total_duration_ms": (time.time() - start_time) * 1000,
            "events": events,
        },
    )

    return artifact


def _step_validate(data: pd.DataFrame) -> ValidationReport:
    """Execute validation step."""
    return validate_contract(data)


def _step_qa(
    data: pd.DataFrame,
    task_spec: TaskSpec,
    mode: Literal["quick", "standard", "strict"],
) -> QAReport:
    """Execute QA step.

    For v0.1, this is a minimal implementation.
    """
    # v0.1: Minimal QA - just check for basic issues
    issues = []
    repairs = []

    # Check for missing values
    missing_count = data["y"].isna().sum()
    if missing_count > 0:
        issues.append(
            {
                "type": "missing_values",
                "column": "y",
                "count": int(missing_count),
                "severity": "warning" if mode != "strict" else "critical",
            }
        )

    return QAReport(
        issues=issues,
        repairs=repairs,
        leakage_detected=False,
    )


def _step_fit(
    dataset: TSDataset,
    plan: Any,
    fit_func: Any | None,
    on_fallback: Any | None = None,
) -> tuple[Any, str]:
    """Execute fit step with fallback."""
    if fit_func is None:
        # Use default fit function
        from tsagentkit.models import fit as default_fit

        fit_func = default_fit

    return execute_with_fallback(
        fit_func=fit_func,
        dataset=dataset,
        plan=plan,
        on_fallback=on_fallback,
    )


def _step_predict(
    model: Any,
    dataset: TSDataset,
    horizon: int,
    predict_func: Any | None,
) -> pd.DataFrame:
    """Execute predict step."""
    if predict_func is None:
        # Use default predict function
        from tsagentkit.models import predict as default_predict

        predict_func = default_predict

    return predict_func(model, dataset, horizon)


def _get_error_code(validation: ValidationReport) -> str | None:
    """Extract error code from validation report."""
    if validation.errors:
        return validation.errors[0].get("code")
    return None
