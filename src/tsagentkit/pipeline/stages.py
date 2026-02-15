"""Pipeline stage definitions.

Each stage is a pure function that transforms input to output.
Stages are composed in the runner to form the complete pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from tsagentkit.core.config import ForecastConfig
from tsagentkit.core.data import CovariateSet, TSDataset
from tsagentkit.core.errors import EContractViolation, EDataQuality, EModelFailed
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
        raise EContractViolation(
            f"Missing required columns: {missing}",
            context={"required": required, "found": list(df.columns)},
        )

    # Check for empty DataFrame
    if len(df) == 0:
        raise EContractViolation("DataFrame is empty")

    # Check for nulls in key columns
    for col in required:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise EContractViolation(
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
        raise EDataQuality(
            f"Found {dups} duplicate (unique_id, ds) pairs",
            fix_hint="Remove duplicates: df = df.drop_duplicates(subset=['unique_id', 'ds'], keep='last')",
        )

    # Check temporal ordering
    for uid, group in df.groupby("unique_id"):
        if not group["ds"].is_monotonic_increasing:
            raise EDataQuality(
                f"Series '{uid}' is not sorted by time",
                fix_hint="Sort data: df = df.sort_values(['unique_id', 'ds'])",
            )

    # In strict mode, any issue is an error
    if config.mode == "strict" and issues:
        raise EDataQuality(
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


def plan_stage(dataset: TSDataset, config: ForecastConfig) -> tuple[TSDataset, list[str]]:
    """Create execution plan (simplified model selection).

    Returns list of candidate models in priority order.
    """
    from tsagentkit.router import inspect_tsfm_adapters

    candidates = []

    # Check TSFM availability
    available_tsfm = inspect_tsfm_adapters()
    tsfm_available = bool(available_tsfm)

    if tsfm_available and config.tsfm_mode != "disabled":
        # Add TSFM models first
        for name in available_tsfm:
            candidates.append(f"tsfm-{name}")

    if not tsfm_available and config.tsfm_mode == "required":
        raise EModelFailed(
            "TSFM required but no adapters available",
            fix_hint="Install TSFM adapters or set tsfm_mode='preferred'",
        )

    # Add statistical baselines as fallbacks
    if config.tsfm_mode != "required" or not tsfm_available:
        # Determine appropriate baselines from data characteristics
        candidates.extend(["SeasonalNaive", "HistoricAverage", "Naive"])

    return dataset, candidates


def backtest_stage(
    dataset: TSDataset, candidates: list[str], config: ForecastConfig
) -> tuple[TSDataset, list[str], dict[str, float] | None]:
    """Run backtest for model selection (optional).

    Skipped if n_backtest_windows is 0 or mode is 'quick'.
    """
    if config.n_backtest_windows == 0 or config.mode == "quick":
        return dataset, candidates, None

    # Simplified backtest - just return candidates without selection
    # Full backtest logic would be imported from backtest module
    return dataset, candidates, None


def fit_predict_stage(
    dataset: TSDataset, candidates: list[str], config: ForecastConfig
) -> tuple[ForecastResult, list[dict[str, str]]]:
    """Fit models and generate forecasts with fallback.

    Simplified from the complex fit/predict/fallback chain.
    """
    from tsagentkit.models import fit, predict, fit_tsfm, predict_tsfm

    fallbacks = []
    last_error = None

    for i, model_name in enumerate(candidates):
        try:
            # Check if TSFM adapter
            if model_name.startswith("tsfm-"):
                adapter_name = model_name.replace("tsfm-", "")
                artifact = fit_tsfm(dataset, adapter_name)
                forecast_df = predict_tsfm(dataset, artifact, config.h)
            else:
                # Use standard models module
                artifact = fit(dataset, model_name)
                forecast_df = predict(dataset, artifact, config.h)

            result = ForecastResult(
                df=forecast_df,
                model_name=model_name,
                config=config,
            )
            return result, fallbacks

        except Exception as e:
            last_error = e
            if not config.allow_fallback:
                raise

            if i < len(candidates) - 1:
                fallbacks.append({
                    "from": model_name,
                    "to": candidates[i + 1],
                    "error": str(e)[:100],
                })
            continue

    # All candidates failed
    raise EModelFailed(
        f"All models failed. Last error: {last_error}",
        context={"models_attempted": candidates},
    )


def package_stage(
    forecast: ForecastResult,
    fallbacks: list[dict[str, str]],
    config: ForecastConfig,
    duration_ms: float,
) -> RunResult:
    """Package results into RunResult."""
    return RunResult(
        forecast=forecast,
        duration_ms=duration_ms,
        model_used=forecast.model_name,
        fallbacks=fallbacks,
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
    PipelineStage("fit_predict", fit_predict_stage),
    PipelineStage("package", package_stage),
]
