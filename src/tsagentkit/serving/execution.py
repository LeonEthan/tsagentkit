"""Model execution helpers for serving pipelines."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tsagentkit.contracts import ForecastResult


def fit_per_series_models(
    dataset: Any,
    plan: Any,
    selection_map: dict[str, str],
    fit_func: Any | None,
    on_fallback: Any | None,
) -> dict[str, Any]:
    """Fit per-series artifacts using the selected model per series."""
    from tsagentkit.models import fit_per_series

    return fit_per_series(
        dataset=dataset,
        plan=plan,
        selection_map=selection_map,
        fit_func=fit_func,
        on_fallback=on_fallback,
    )


def fit_single_model(
    dataset: Any,
    plan: Any,
    fit_func: Any | None,
    on_fallback: Any | None,
    covariates: Any | None,
) -> Any:
    """Fit a single model artifact for all series."""
    from tsagentkit.models import fit as default_fit
    from tsagentkit.utils.compat import call_with_optional_kwargs

    fit_callable = fit_func or default_fit
    kwargs = {"covariates": covariates}

    if fit_callable is default_fit:
        return fit_callable(dataset, plan, on_fallback=on_fallback, **kwargs)

    return call_with_optional_kwargs(fit_callable, dataset, plan, **kwargs)


def predict_per_series_models(
    dataset: Any,
    artifacts: dict[str, Any],
    selection_map: dict[str, str],
    task_spec: Any,
    predict_func: Any | None,
) -> pd.DataFrame:
    """Generate per-series forecasts from per-series artifacts."""
    from tsagentkit.models import predict_per_series

    return predict_per_series(
        dataset=dataset,
        artifacts=artifacts,
        selection_map=selection_map,
        spec=task_spec,
        predict_func=predict_func,
    )


def predict_single_model(
    dataset: Any,
    model_artifact: Any,
    task_spec: Any,
    predict_func: Any | None,
    covariates: Any | None,
) -> pd.DataFrame:
    """Generate forecast from a single model artifact."""
    from tsagentkit.models import predict as default_predict
    from tsagentkit.utils.compat import call_with_optional_kwargs

    predict_callable = predict_func or default_predict
    kwargs = {"covariates": covariates}

    forecast = call_with_optional_kwargs(
        predict_callable,
        dataset,
        model_artifact,
        task_spec,
        **kwargs,
    )
    if isinstance(forecast, ForecastResult):
        return forecast.df
    return forecast


def fit_predict_with_fallback(
    dataset: Any,
    plan: Any,
    task_spec: Any,
    fit_func: Any | None,
    predict_func: Any | None,
    covariates: Any | None,
    start_after: str | None,
    initial_error: Exception,
    on_fallback: Any | None,
    reconciliation_method: str,
) -> tuple[Any, pd.DataFrame]:
    """Fit and predict using fallback ladder from router policy."""
    from tsagentkit.router.fallback import fit_predict_with_fallback as _fit_predict_with_fallback

    return _fit_predict_with_fallback(
        dataset=dataset,
        plan=plan,
        task_spec=task_spec,
        fit_func=fit_func,
        predict_func=predict_func,
        covariates=covariates,
        start_after=start_after,
        initial_error=initial_error,
        on_fallback=on_fallback,
        reconciliation_method=reconciliation_method,
    )

