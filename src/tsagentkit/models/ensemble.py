"""Ensemble utilities for combining multiple model forecasts.

Provides functions for fitting multiple TSFM models and creating
median ensemble predictions for quick mode.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from tsagentkit.contracts import EModelFitFailed, EModelPredictFailed, ModelArtifact
from tsagentkit.utils import normalize_quantile_columns, parse_quantile_column

if TYPE_CHECKING:
    from collections.abc import Callable

    from tsagentkit.contracts import TaskSpec
    from tsagentkit.router import PlanSpec
    from tsagentkit.series import TSDataset

    FitFunc = Callable[..., ModelArtifact]
    PredictFunc = Callable[..., Any]

logger = logging.getLogger(__name__)


def _is_tsfm_model(model_name: str) -> bool:
    """Check if model name represents a TSFM model."""
    return model_name.lower().startswith("tsfm-")


def fit_all_tsfm_models(
    dataset: TSDataset,
    plan: PlanSpec,
    fit_func: FitFunc | None = None,
    on_fallback: Any | None = None,
) -> dict[str, ModelArtifact]:
    """Fit all TSFM models from the plan (for quick mode ensemble).

    Attempts to fit each TSFM candidate model. Failed models are skipped
    with a warning. If no models succeed, raises an error.

    Args:
        dataset: The time series dataset to fit on
        plan: The execution plan containing candidate models
        fit_func: Optional custom fit function
        on_fallback: Optional callback for fallback events

    Returns:
        Dictionary mapping model names to their fitted artifacts

    Raises:
        EModelFitFailed: If no TSFM models could be fitted
    """
    from tsagentkit.models import fit as default_fit

    fit_callable = fit_func or default_fit

    # Filter to only TSFM models
    tsfm_models = [m for m in plan.candidate_models if _is_tsfm_model(m)]

    if not tsfm_models:
        raise EModelFitFailed(
            "No TSFM models found in plan for ensemble fitting",
            context={"candidate_models": plan.candidate_models},
        )

    artifacts: dict[str, ModelArtifact] = {}

    for model_name in tsfm_models:
        try:
            # Create a single-model plan for this candidate
            from tsagentkit.contracts import PlanSpec as PlanSpecClass

            single_model_plan = PlanSpecClass(
                plan_name=f"ensemble_{model_name}",
                candidate_models=[model_name],
                use_static=plan.use_static,
                use_past=plan.use_past,
                use_future_known=plan.use_future_known,
                min_train_size=plan.min_train_size,
                max_train_size=plan.max_train_size,
                interval_mode=plan.interval_mode,
                levels=plan.levels,
                quantiles=plan.quantiles,
                allow_drop_covariates=plan.allow_drop_covariates,
                allow_baseline=plan.allow_baseline,
            )

            from tsagentkit.utils.compat import call_with_optional_kwargs

            artifact = call_with_optional_kwargs(
                fit_callable,
                dataset=dataset,
                plan=single_model_plan,
                on_fallback=on_fallback,
            )

            if isinstance(artifact, ModelArtifact):
                artifacts[model_name] = artifact
            else:
                # Handle case where fit returns something else
                logger.warning("Fit for %s did not return ModelArtifact, skipping", model_name)

        except Exception as e:
            logger.warning("Failed to fit TSFM model %s: %s", model_name, str(e))
            if on_fallback:
                on_fallback(model_name, "failed", e)
            continue

    if not artifacts:
        raise EModelFitFailed(
            "No TSFM models could be fitted for ensemble",
            context={
                "attempted_models": tsfm_models,
                "plan_candidate_models": plan.candidate_models,
            },
        )

    return artifacts


def predict_ensemble_median(
    dataset: TSDataset,
    artifacts: dict[str, ModelArtifact],
    spec: TaskSpec,
    predict_func: PredictFunc | None = None,
) -> pd.DataFrame:
    """Generate median ensemble forecast from multiple models.

    Collects predictions from all fitted models and computes the median
    of yhat values. Quantile columns are taken from the first successful
    model (not ensembled).

    Args:
        dataset: The time series dataset to predict on
        artifacts: Dictionary of fitted model artifacts
        spec: Task specification with forecast parameters
        predict_func: Optional custom predict function

    Returns:
        DataFrame with ensemble forecast

    Raises:
        EModelPredictFailed: If all model predictions fail
    """
    from tsagentkit.models import predict as default_predict

    predict_callable = predict_func or default_predict

    forecasts: list[pd.DataFrame] = []
    first_forecast_df: pd.DataFrame | None = None

    for model_name, artifact in artifacts.items():
        try:
            result = predict_callable(
                dataset=dataset,
                artifact=artifact,
                spec=spec,
            )

            # Handle ForecastResult or DataFrame
            forecast_df = result.df if hasattr(result, "df") else result

            # Type assertion for mypy
            assert isinstance(forecast_df, pd.DataFrame), f"Expected DataFrame, got {type(forecast_df)}"

            # Ensure model column exists
            if "model" not in forecast_df.columns:
                forecast_df = forecast_df.copy()
                forecast_df["model"] = model_name

            forecasts.append(forecast_df)

            # Keep first forecast for quantile reference
            if first_forecast_df is None:
                first_forecast_df = forecast_df

        except Exception as e:
            logger.warning("Failed to predict with model %s: %s", model_name, str(e))
            continue

    if not forecasts:
        raise EModelPredictFailed(
            "All model predictions failed for ensemble",
            context={"n_models": len(artifacts)},
        )

    # Compute median ensemble
    return compute_median_ensemble(forecasts, first_forecast_df)


def compute_median_ensemble(
    forecasts: list[pd.DataFrame],
    first_forecast: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute median across model forecasts.

    Only ensembles the yhat (point forecast) column. Quantile columns
    are taken from the first forecast (if provided).

    Args:
        forecasts: List of forecast DataFrames from different models
        first_forecast: First successful forecast for quantile reference

    Returns:
        DataFrame with median ensemble forecast
    """
    if not forecasts:
        raise ValueError("No forecasts provided for ensemble")

    # Combine all forecasts
    combined = pd.concat(forecasts, ignore_index=True)

    # Verify required columns
    required_cols = {"unique_id", "ds", "yhat"}
    missing = required_cols - set(combined.columns)
    if missing:
        raise EModelPredictFailed(
            f"Forecast missing required columns for ensemble: {missing}",
            context={"available_columns": list(combined.columns)},
        )

    # Compute median of yhat grouped by unique_id and ds
    ensemble = combined.groupby(["unique_id", "ds"], as_index=False).agg({"yhat": "median"})
    ensemble["model"] = "median_ensemble"

    # Get quantile columns from first forecast (not ensembled)
    if first_forecast is not None:
        quantile_cols = [c for c in first_forecast.columns if parse_quantile_column(c) is not None]

        if quantile_cols:
            # Select quantile columns from first forecast
            quantile_df = first_forecast[["unique_id", "ds"] + quantile_cols].drop_duplicates(
                subset=["unique_id", "ds"]
            )

            # Merge with ensemble
            ensemble = ensemble.merge(
                quantile_df,
                on=["unique_id", "ds"],
                how="left",
            )

    # Normalize quantile column names
    ensemble = normalize_quantile_columns(ensemble)

    return ensemble


__all__ = [
    "fit_all_tsfm_models",
    "predict_ensemble_median",
    "compute_median_ensemble",
    "_is_tsfm_model",
]
