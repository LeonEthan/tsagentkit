"""Per-series model fitting and prediction.

This module provides functionality to fit and predict with different models
for different series, based on per-series model selection from multi-model
backtest results.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd

from tsagentkit.contracts import ForecastResult, ModelArtifact

if TYPE_CHECKING:
    from tsagentkit.contracts import TaskSpec
    from tsagentkit.router import PlanSpec
    from tsagentkit.series import TSDataset


def fit_per_series(
    dataset: TSDataset,
    plan: PlanSpec,
    selection_map: dict[str, str],
    fit_func: Callable[[TSDataset, PlanSpec], ModelArtifact] | None = None,
    on_fallback: Callable[[str, str, Exception], None] | None = None,
) -> dict[str, ModelArtifact]:
    """Fit the selected model for each series.

    Groups series by their assigned model and fits each model on its
    assigned subset of series.

    Args:
        dataset: TSDataset with all series
        plan: Execution plan with model configuration
        selection_map: Dictionary mapping unique_id to best model name
        fit_func: Optional custom fit function (defaults to models.fit)
        on_fallback: Optional callback for fallback events

    Returns:
        Dictionary mapping model name to fitted ModelArtifact
    """
    from tsagentkit.models import fit as default_fit

    fit_callable = fit_func or default_fit

    # Group series by selected model
    model_to_series: dict[str, list[str]] = defaultdict(list)
    for uid, model_name in selection_map.items():
        model_to_series[model_name].append(uid)

    artifacts: dict[str, ModelArtifact] = {}

    # Fit each model on its assigned series
    for model_name, series_ids in model_to_series.items():
        # Filter dataset to relevant series
        subset = dataset.filter_series(series_ids)

        # Create a single-model plan
        model_plan = plan.model_copy(update={"candidate_models": [model_name]})

        try:
            artifact = fit_callable(
                subset,
                model_plan,
                on_fallback=on_fallback,
            )
            artifacts[model_name] = artifact
        except Exception:
            # If a model fails, we still want to continue with others
            # The orchestration layer will handle fallback logic
            continue

    return artifacts


def _call_predict_with_kwargs(
    func: Callable[..., Any],
    dataset: "TSDataset",
    artifact: ModelArtifact,
    spec: "TaskSpec",
) -> Any:
    """Call predict function with kwargs, handling different signatures."""
    import inspect

    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
    except (ValueError, TypeError):
        # Fall back to positional args if introspection fails
        return func(dataset, artifact, spec)

    # Build kwargs based on common parameter names
    kwargs: dict[str, Any] = {}
    for param in params:
        if param in ("dataset", "data"):
            kwargs[param] = dataset
        elif param in ("artifact", "model"):
            kwargs[param] = artifact
        elif param in ("spec", "task_spec"):
            kwargs[param] = spec

    if kwargs:
        return func(**kwargs)
    # Fall back to positional if no recognized params
    return func(dataset, artifact, spec)


def predict_per_series(
    dataset: TSDataset,
    artifacts: dict[str, ModelArtifact],
    selection_map: dict[str, str],
    spec: TaskSpec,
    predict_func: Callable[[TSDataset, ModelArtifact, TaskSpec], ForecastResult]
    | None = None,
) -> pd.DataFrame:
    """Predict using the fitted model for each series.

    Groups series by their assigned model and generates predictions using
    the appropriate fitted model for each group.

    Args:
        dataset: TSDataset with all series
        artifacts: Dictionary mapping model name to fitted ModelArtifact
        selection_map: Dictionary mapping unique_id to best model name
        spec: Task specification
        predict_func: Optional custom predict function (defaults to models.predict)

    Returns:
        DataFrame with forecasts for all series
    """
    from tsagentkit.models import predict as default_predict

    predict_callable = predict_func or default_predict

    # Group series by model
    model_to_series: dict[str, list[str]] = defaultdict(list)
    for uid, model_name in selection_map.items():
        model_to_series[model_name].append(uid)

    forecasts: list[pd.DataFrame] = []

    # Predict for each model's assigned series
    for model_name, series_ids in model_to_series.items():
        if model_name not in artifacts:
            # Model fitting failed, skip these series
            continue

        artifact = artifacts[model_name]
        subset = dataset.filter_series(series_ids)

        try:
            # Use kwargs to handle different predict function signatures gracefully
            result = _call_predict_with_kwargs(predict_callable, subset, artifact, spec)
            forecast_df = result.df.copy() if isinstance(result, ForecastResult) else result.copy()

            # Ensure model column is set
            if "model" not in forecast_df.columns:
                forecast_df["model"] = model_name
            else:
                forecast_df = forecast_df.copy()
                forecast_df["model"] = model_name

            forecasts.append(forecast_df)
        except Exception:
            # If prediction fails for a model, skip those series
            continue

    if not forecasts:
        raise ValueError("All model predictions failed")

    return pd.concat(forecasts, ignore_index=True)


def fit_predict_per_series(
    dataset: TSDataset,
    plan: PlanSpec,
    selection_map: dict[str, str],
    spec: TaskSpec,
    fit_func: Callable[[TSDataset, PlanSpec], ModelArtifact] | None = None,
    predict_func: Callable[[TSDataset, ModelArtifact, TaskSpec], ForecastResult]
    | None = None,
    on_fallback: Callable[[str, str, Exception], None] | None = None,
) -> tuple[dict[str, ModelArtifact], pd.DataFrame]:
    """Fit and predict with per-series model selection in one call.

    Convenience function that combines fit_per_series and predict_per_series.

    Args:
        dataset: TSDataset with all series
        plan: Execution plan with model configuration
        selection_map: Dictionary mapping unique_id to best model name
        spec: Task specification
        fit_func: Optional custom fit function
        predict_func: Optional custom predict function
        on_fallback: Optional callback for fallback events

    Returns:
        Tuple of (artifacts dict, forecast DataFrame)
    """
    artifacts = fit_per_series(
        dataset=dataset,
        plan=plan,
        selection_map=selection_map,
        fit_func=fit_func,
        on_fallback=on_fallback,
    )

    forecast_df = predict_per_series(
        dataset=dataset,
        artifacts=artifacts,
        selection_map=selection_map,
        spec=spec,
        predict_func=predict_func,
    )

    return artifacts, forecast_df


__all__ = [
    "fit_per_series",
    "predict_per_series",
    "fit_predict_per_series",
]
