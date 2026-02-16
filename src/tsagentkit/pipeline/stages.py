"""Pipeline stage definitions.

Each stage is a pure function that transforms input to output.
Stages are composed in the runner to form the complete pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tsagentkit.core.config import ForecastConfig
from tsagentkit.core.data import CovariateSet, TSDataset
from tsagentkit.core.errors import EContract, EInsufficient
from tsagentkit.core.results import ForecastResult, RunResult


@dataclass(frozen=True)
class PipelineStage:
    """A single pipeline stage.

    Each stage is a named function with clear input/output contracts.
    """

    name: str
    run: Callable[..., Any]


# =============================================================================
# Stage Implementations
# =============================================================================


def validate_stage(df: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    """Validate input data and normalize column names.

    Replaces validate_contract() with simplified validation.
    """
    # Check required columns
    required = [config.id_col, config.time_col, config.target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise EContract(
            f"Missing required columns: {missing}",
            context={"required": required, "found": list(df.columns)},
        )

    # Check for empty DataFrame
    if len(df) == 0:
        raise EContract("DataFrame is empty")

    # Check for nulls in key columns
    for col in required:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise EContract(
                f"Column '{col}' has {null_count} null values",
                fix_hint="Remove or fill null values in key columns",
            )

    # Normalize column names to standard
    df = df.copy()
    if config.id_col != "unique_id":
        df = df.rename(columns={config.id_col: "unique_id"})
    if config.time_col != "ds":
        df = df.rename(columns={config.time_col: "ds"})
    if config.target_col != "y":
        df = df.rename(columns={config.target_col: "y"})

    return df


def qa_stage(df: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    """Run quality assurance checks.

    Simplified from run_qa() - focuses on critical issues only.
    """
    issues = []

    # Check series lengths
    series_lengths = df.groupby("unique_id").size()
    short_series = series_lengths[series_lengths < config.min_train_size]
    if len(short_series) > 0:
        issues.append(f"{len(short_series)} series shorter than min_train_size")

    # Check for duplicate keys
    dups = df.duplicated(subset=["unique_id", "ds"], keep=False).sum()
    if dups > 0:
        raise EContract(
            f"Found {dups} duplicate (unique_id, ds) pairs",
            fix_hint="Remove duplicates: df = df.drop_duplicates(subset=['unique_id', 'ds'], keep='last')",
        )

    # Check temporal ordering
    for uid, group in df.groupby("unique_id"):
        if not group["ds"].is_monotonic_increasing:
            raise EContract(
                f"Series '{uid}' is not sorted by time",
                fix_hint="Sort data: df = df.sort_values(['unique_id', 'ds'])",
            )

    # In strict mode, any issue is an error
    if config.mode == "strict" and issues:
        raise EContract(
            "Data quality issues found in strict mode",
            context={"issues": issues},
        )

    return df


def build_dataset_stage(
    df: pd.DataFrame, config: ForecastConfig, covariates: CovariateSet | None = None
) -> TSDataset:
    """Build TSDataset from validated DataFrame.

    Simplified from build_dataset() - removes complex hierarchy/covariate attachment.
    """
    return TSDataset.from_dataframe(df, config, covariates)


def plan_stage(dataset: TSDataset, config: ForecastConfig) -> tuple[TSDataset, Any]:
    """Create ensemble execution plan.

    Returns Plan with all TSFM and statistical models for ensemble.
    """
    from tsagentkit.router import build_plan

    plan = build_plan(
        dataset=dataset,
        tsfm_mode=config.tsfm_mode,
        allow_fallback=config.allow_fallback,
        ensemble_method=config.ensemble_method,
        require_all_tsfm=config.require_all_tsfm,
    )

    return dataset, plan


def backtest_stage(
    dataset: TSDataset, plan: Any, config: ForecastConfig
) -> tuple[TSDataset, Any, dict[str, float] | None]:
    """Run backtest for model selection (optional).

    Skipped if n_backtest_windows is 0 or mode is 'quick'.
    """
    if config.n_backtest_windows == 0 or config.mode == "quick":
        return dataset, plan, None

    # Simplified backtest - just return plan without selection
    # Full backtest logic would be imported from backtest module
    return dataset, plan, None


def _fit_and_predict_single(
    dataset: TSDataset,
    candidate: Any,
    h: int,
    quantiles: list[float] | None = None,
) -> pd.DataFrame | None:
    """Fit and predict for a single model candidate.

    Returns None if the model fails.
    """
    from tsagentkit.models.protocol import fit, predict
    from tsagentkit.models.registry import get_spec

    try:
        spec = get_spec(candidate.adapter_name or candidate.name)
        artifact = fit(spec, dataset)
        forecast_df = predict(spec, artifact, dataset, h)
        return forecast_df
    except Exception:
        return None


def _compute_ensemble(
    predictions: list[pd.DataFrame],
    method: str,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Compute ensemble forecast from multiple model predictions.

    Args:
        predictions: List of forecast DataFrames
        method: 'median' or 'mean'
        quantiles: List of quantile levels to ensemble

    Returns:
        Ensemble forecast DataFrame
    """
    from tsagentkit.models.ensemble import ensemble_with_quantiles

    if not predictions:
        raise EInsufficient("No predictions to ensemble")

    if len(predictions) == 1:
        return predictions[0]

    return ensemble_with_quantiles(
        predictions,
        method=method,  # type: ignore[arg-type]
        quantiles=quantiles,
    )


def ensemble_stage(
    dataset: TSDataset, plan: Any, config: ForecastConfig
) -> tuple[ForecastResult, list[dict[str, str]]]:
    """Fit all models in ensemble and return aggregated forecast.

    Args:
        dataset: Time-series dataset
        plan: Ensemble Plan with all models
        config: Forecast configuration

    Returns:
        ForecastResult with ensemble forecast, plus list of model errors
    """
    from tsagentkit.router import Plan

    if not isinstance(plan, Plan):
        raise ValueError(f"Expected Plan, got {type(plan)}")

    predictions: list[pd.DataFrame] = []
    model_errors: list[dict[str, str]] = []

    # Fit and predict all models
    for candidate in plan.all_models():
        forecast_df = _fit_and_predict_single(dataset, candidate, config.h, quantiles=config.quantiles)

        if forecast_df is not None:
            predictions.append(forecast_df)
        else:
            model_errors.append({
                "model": candidate.name,
                "error": "Failed to fit or predict",
            })
            # If required TSFM failed, check if we should abort
            if candidate.is_tsfm and plan.require_all_tsfm:
                raise EInsufficient(
                    f"Required TSFM model '{candidate.name}' failed",
                    context={"errors": model_errors},
                )

    # Check minimum models requirement
    if len(predictions) < plan.min_models_for_ensemble:
        raise EInsufficient(
            f"Insufficient models succeeded: {len(predictions)} < {plan.min_models_for_ensemble}",
            context={"errors": model_errors},
        )

    # Compute ensemble
    ensemble_df = _compute_ensemble(predictions, plan.ensemble_method, quantiles=config.quantiles)

    result = ForecastResult(
        df=ensemble_df,
        model_name=f"ensemble_{plan.ensemble_method}",
        config=config,
    )

    return result, model_errors


def package_stage(
    forecast: ForecastResult,
    model_errors: list[dict[str, str]],
    config: ForecastConfig,
    duration_ms: float,
) -> RunResult:
    """Package results into RunResult."""
    return RunResult(
        forecast=forecast,
        duration_ms=duration_ms,
        model_used=forecast.model_name,
        model_errors=model_errors,
    )


# =============================================================================
# Stage Registry
# =============================================================================

STAGES = [
    PipelineStage("validate", validate_stage),
    PipelineStage("qa", qa_stage),
    PipelineStage("build", build_dataset_stage),
    PipelineStage("plan", plan_stage),
    PipelineStage("backtest", backtest_stage),
    PipelineStage("ensemble", ensemble_stage),
    PipelineStage("package", package_stage),
]
