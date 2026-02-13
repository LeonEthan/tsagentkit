"""Fallback ladder implementation.

Provides automatic model degradation when primary models fail.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import pandas as pd

from tsagentkit.contracts.errors import EFallbackExhausted, TSAgentKitError
from tsagentkit.contracts.results import ForecastResult
from tsagentkit.router.plan import PlanSpec, get_candidate_models
from tsagentkit.utils import normalize_quantile_columns

if TYPE_CHECKING:
    from tsagentkit.series import TSDataset

T = TypeVar("T")

FALLBACK_TRIGGER_ERROR_CODES = frozenset(
    {
        "E_MODEL_FIT_FAIL",
        "E_MODEL_PREDICT_FAIL",
        "E_MODEL_LOAD_FAILED",
        "E_ADAPTER_NOT_AVAILABLE",
        "E_OOM",
    }
)


def should_trigger_fallback(error: Exception) -> bool:
    """Return whether a failure should trigger the next model fallback.

    Fallback is only allowed for explicit, model-runtime-style failures.
    Unexpected exceptions should surface immediately.
    """
    if isinstance(error, MemoryError):
        return True
    if isinstance(error, TSAgentKitError):
        return error.error_code in FALLBACK_TRIGGER_ERROR_CODES
    return False


def execute_with_fallback(
    fit_func: Callable[[str, TSDataset], T],
    dataset: TSDataset,
    plan: PlanSpec,
    on_fallback: Callable[[str, str, Exception], None] | None = None,
) -> tuple[T, str]:
    """Execute fit function with fallback ladder.

    Attempts to fit models in order (primary -> fallbacks) until one succeeds.

    Args:
        fit_func: Function that fits a model given (model_name, dataset)
        dataset: TSDataset to fit on
        plan: Execution plan with fallback chain
        on_fallback: Optional callback when fallback triggered (from_model, to_model, error)

    Returns:
        Tuple of (result, model_name_that_succeeded)

    Raises:
        EFallbackExhausted: If all models in the ladder fail
    """
    models = get_candidate_models(plan)
    last_error: Exception | None = None

    for i, model_name in enumerate(models):
        try:
            result = fit_func(model_name, dataset)
            return result, model_name
        except Exception as e:
            if not should_trigger_fallback(e):
                raise
            last_error = e

            # Trigger callback if provided
            if on_fallback and i < len(models) - 1:
                on_fallback(model_name, models[i + 1], e)

            # Continue to next fallback
            continue

    # All models failed
    error_msg = f"All models failed. Last error: {last_error}"
    raise EFallbackExhausted(
        error_msg,
        context={
            "models_attempted": models,
            "last_error": str(last_error),
        },
    )


class FallbackLadder:
    """Manages fallback chains for different scenarios.

    Provides predefined fallback ladders for common use cases.
    """

    # Standard fallback chains
    STANDARD_LADDER: list[str] = ["SeasonalNaive", "HistoricAverage", "Naive"]
    """Standard fallback: SeasonalNaive -> HistoricAverage -> Naive"""

    INTERMITTENT_LADDER: list[str] = ["Croston", "Naive"]
    """For intermittent demand: Croston -> Naive"""

    COLD_START_LADDER: list[str] = ["HistoricAverage", "Naive"]
    """For cold-start series: HistoricAverage -> Naive"""

    @classmethod
    def get_ladder(
        cls,
        is_intermittent: bool = False,
        is_cold_start: bool = False,
    ) -> list[str]:
        """Get appropriate fallback ladder for scenario.

        Args:
            is_intermittent: Whether series is intermittent
            is_cold_start: Whether series is cold-start

        Returns:
            Ordered list of fallback model names
        """
        if is_intermittent:
            return cls.INTERMITTENT_LADDER
        if is_cold_start:
            return cls.COLD_START_LADDER
        return cls.STANDARD_LADDER

    @classmethod
    def with_primary(
        cls,
        primary: str,
        fallbacks: list[str] | None = None,
        is_intermittent: bool = False,
        is_cold_start: bool = False,
    ) -> list[str]:
        """Create full model chain with primary and fallbacks.

        Args:
            primary: Primary model name
            fallbacks: Optional explicit fallback list
            is_intermittent: Whether series is intermittent
            is_cold_start: Whether series is cold-start

        Returns:
            List with primary first, then fallbacks
        """
        if fallbacks is None:
            fallbacks = cls.get_ladder(is_intermittent, is_cold_start)

        # Ensure primary isn't in fallbacks
        filtered_fallbacks = [f for f in fallbacks if f != primary]

        return [primary] + filtered_fallbacks


def fit_predict_with_fallback(
    dataset: TSDataset,
    plan: PlanSpec,
    task_spec: Any,
    fit_func: Callable[[TSDataset, PlanSpec], Any] | None = None,
    predict_func: Callable[..., pd.DataFrame] | None = None,
    covariates: Any | None = None,
    start_after: str | None = None,
    initial_error: Exception | None = None,
    on_fallback: Callable[[str, str, Exception], None] | None = None,
    reconciliation_method: str = "bottom_up",
) -> tuple[Any, pd.DataFrame]:
    """Fit and predict with fallback across remaining candidates.

    This function attempts to fit and predict with models from the plan's
    candidate list, starting after the specified model. If a model fails,
    it falls back to the next model in the list.

    Args:
        dataset: TSDataset with prepared data
        plan: Execution plan with candidate models
        task_spec: Task specification
        fit_func: Function to fit model: fit_func(dataset, plan)
        predict_func: Function to predict: predict_func(dataset, artifact, spec, covariates=None)
        covariates: Optional aligned covariates
        start_after: Model name to start fallback after
        initial_error: Initial error that triggered fallback
        on_fallback: Callback when fallback occurs (from_model, to_model, error)
        reconciliation_method: Reconciliation method for hierarchical forecasts

    Returns:
        Tuple of (model_artifact, forecast_df)

    Raises:
        EFallbackExhausted: If all models fail
    """
    from tsagentkit.models import fit as default_fit
    from tsagentkit.models import predict as default_predict
    from tsagentkit.utils.compat import call_with_optional_kwargs
    from tsagentkit.hierarchy import ReconciliationMethod, reconcile_forecasts

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
            artifact = call_with_optional_kwargs(
                fit_callable,
                dataset,
                plan_for_model,
                covariates=covariates,
            )
        except Exception as e:
            if not should_trigger_fallback(e):
                raise
            last_error = e
            if on_fallback and i < len(remaining) - 1:
                on_fallback(model_name, remaining[i + 1], e)
            continue

        try:
            forecast = call_with_optional_kwargs(
                predict_callable,
                dataset,
                artifact,
                task_spec,
                covariates=covariates,
            )

            if isinstance(forecast, ForecastResult):
                forecast = forecast.df

            # Apply reconciliation if hierarchical
            if dataset.is_hierarchical() and dataset.hierarchy:
                from tsagentkit.hierarchy import apply_reconciliation_if_needed

                forecast = apply_reconciliation_if_needed(
                    forecast=forecast,
                    hierarchy=dataset.hierarchy,
                    method=reconciliation_method,
                )

            forecast = normalize_quantile_columns(forecast)
            if {"unique_id", "ds"}.issubset(forecast.columns):
                forecast = forecast.sort_values(["unique_id", "ds"]).reset_index(drop=True)

            return artifact, forecast
        except Exception as e:
            if not should_trigger_fallback(e):
                raise
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
