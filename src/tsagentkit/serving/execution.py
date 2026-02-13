"""Model execution helpers for serving pipelines."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import pandas as pd

from tsagentkit.contracts import ForecastResult

FitFunc: TypeAlias = Callable[..., object]
PredictFunc: TypeAlias = Callable[..., object]
FallbackHandler: TypeAlias = Callable[[str, str, Exception], None]


def fit_per_series_models(
    dataset: object,
    plan: object,
    selection_map: dict[str, str],
    fit_func: FitFunc | None,
    on_fallback: FallbackHandler | None,
) -> dict[str, object]:
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
    dataset: object,
    plan: object,
    fit_func: FitFunc | None,
    on_fallback: FallbackHandler | None,
    covariates: object | None,
) -> object:
    """Fit a single model artifact for all series."""
    from tsagentkit.models import fit as default_fit
    from tsagentkit.utils.compat import call_with_optional_kwargs

    fit_callable = fit_func or default_fit
    kwargs = {"covariates": covariates}

    if fit_callable is default_fit:
        return fit_callable(dataset, plan, on_fallback=on_fallback, **kwargs)

    return call_with_optional_kwargs(fit_callable, dataset, plan, **kwargs)


def predict_per_series_models(
    dataset: object,
    artifacts: dict[str, object],
    selection_map: dict[str, str],
    task_spec: object,
    predict_func: PredictFunc | None,
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
    dataset: object,
    model_artifact: object,
    task_spec: object,
    predict_func: PredictFunc | None,
    covariates: object | None,
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
    dataset: object,
    plan: object,
    task_spec: object,
    fit_func: FitFunc | None,
    predict_func: PredictFunc | None,
    covariates: object | None,
    start_after: str | None,
    initial_error: Exception,
    on_fallback: FallbackHandler | None,
    reconciliation_method: str,
) -> tuple[object, pd.DataFrame]:
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


__all__ = [
    "FallbackHandler",
    "FitFunc",
    "PredictFunc",
    "fit_per_series_models",
    "fit_predict_with_fallback",
    "fit_single_model",
    "predict_per_series_models",
    "predict_single_model",
]
