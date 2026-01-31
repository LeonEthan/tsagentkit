"""Fallback ladder implementation.

Provides automatic model degradation when primary models fail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar

from tsagentkit.contracts.errors import EFallbackExhausted, EModelFitFailed

if TYPE_CHECKING:
    from tsagentkit.series import TSDataset
    from .plan import Plan

T = TypeVar("T")


def execute_with_fallback(
    fit_func: Callable[[str, TSDataset, dict[str, Any]], T],
    dataset: TSDataset,
    plan: Plan,
    on_fallback: Callable[[str, str, Exception], None] | None = None,
) -> tuple[T, str]:
    """Execute fit function with fallback ladder.

    Attempts to fit models in order (primary -> fallbacks) until one succeeds.

    Args:
        fit_func: Function that fits a model given (model_name, dataset, config)
        dataset: TSDataset to fit on
        plan: Execution plan with fallback chain
        on_fallback: Optional callback when fallback triggered (from_model, to_model, error)

    Returns:
        Tuple of (result, model_name_that_succeeded)

    Raises:
        EFallbackExhausted: If all models in the ladder fail
    """
    models = plan.get_all_models()
    last_error: Exception | None = None

    for i, model_name in enumerate(models):
        try:
            result = fit_func(model_name, dataset, plan.config)
            return result, model_name
        except Exception as e:
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
