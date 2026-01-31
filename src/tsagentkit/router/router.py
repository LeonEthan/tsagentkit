"""Model selection and routing logic.

Routes TSDatasets to appropriate models based on data characteristics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tsagentkit.contracts import TaskSpec

from .fallback import FallbackLadder
from .plan import Plan

if TYPE_CHECKING:
    from tsagentkit.qa import QAReport
    from tsagentkit.series import SparsityClass, SparsityProfile, TSDataset


def make_plan(
    dataset: TSDataset,
    task_spec: TaskSpec,
    qa: QAReport | None = None,
    strategy: str = "auto",
) -> Plan:
    """Create an execution plan for a dataset.

    Analyzes the dataset characteristics and selects appropriate models
    with fallback chains.

    Args:
        dataset: TSDataset to create plan for
        task_spec: Task specification
        qa: Optional QA report for data quality considerations
        strategy: Routing strategy ("auto", "baseline_only", "tsfm_first")

    Returns:
        Plan with model selection and fallback chain

    Raises:
        ValueError: If unknown strategy specified
    """
    # Get sparsity profile
    sparsity = dataset.sparsity_profile

    # Determine primary model and fallbacks based on strategy
    if strategy == "auto":
        return _make_auto_plan(dataset, task_spec, sparsity)
    elif strategy == "baseline_only":
        return _make_baseline_plan(task_spec, sparsity)
    elif strategy == "tsfm_first":
        # TSFM-first strategy (for v0.2+)
        return _make_tsfm_plan(dataset, task_spec, sparsity)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _make_auto_plan(
    dataset: TSDataset,
    task_spec: TaskSpec,
    sparsity: SparsityProfile | None,
) -> Plan:
    """Create plan with automatic model selection.

    For v0.1, this selects baseline models based on data characteristics.
    """
    # Analyze data characteristics
    has_intermittent = sparsity.has_intermittent() if sparsity else False
    has_cold_start = sparsity.has_cold_start() if sparsity else False

    # Get season length from task spec or default
    season_length = task_spec.season_length or 1

    # Select primary model based on data characteristics
    if has_intermittent:
        primary = "SeasonalNaive"  # v0.1: Use seasonal naive for intermittent
        fallback_chain = FallbackLadder.INTERMITTENT_LADDER[1:]  # Exclude primary
    elif has_cold_start:
        primary = "HistoricAverage"
        fallback_chain = FallbackLadder.COLD_START_LADDER[1:]
    else:
        # Regular series: Use SeasonalNaive as primary
        primary = "SeasonalNaive"
        fallback_chain = FallbackLadder.STANDARD_LADDER[1:]

    # Build config
    config = {
        "season_length": season_length,
        "horizon": task_spec.horizon,
    }

    if task_spec.quantiles:
        config["quantiles"] = task_spec.quantiles

    return Plan(
        primary_model=primary,
        fallback_chain=fallback_chain,
        config=config,
        strategy="auto",
    )


def _make_baseline_plan(
    task_spec: TaskSpec,
    sparsity: SparsityProfile | None,
) -> Plan:
    """Create plan using only baseline models."""
    has_intermittent = sparsity.has_intermittent() if sparsity else False
    has_cold_start = sparsity.has_cold_start() if sparsity else False

    season_length = task_spec.season_length or 1

    if has_intermittent:
        primary = "SeasonalNaive"
        fallback_chain = ["HistoricAverage", "Naive"]
    elif has_cold_start:
        primary = "HistoricAverage"
        fallback_chain = ["Naive"]
    else:
        primary = "SeasonalNaive"
        fallback_chain = ["HistoricAverage", "Naive"]

    config = {
        "season_length": season_length,
        "horizon": task_spec.horizon,
    }

    if task_spec.quantiles:
        config["quantiles"] = task_spec.quantiles

    return Plan(
        primary_model=primary,
        fallback_chain=fallback_chain,
        config=config,
        strategy="baseline_only",
    )


def _make_tsfm_plan(
    dataset: TSDataset,
    task_spec: TaskSpec,
    sparsity: SparsityProfile | None,
) -> Plan:
    """Create plan with TSFM-first strategy.

    For v0.1, this falls back to baseline since TSFM integration
    is planned for v0.2.
    """
    # v0.1: TSFM not yet integrated, use baseline
    # In v0.2, this would attempt TSFM models first
    return _make_baseline_plan(task_spec, sparsity)


def get_model_for_series(
    unique_id: str,
    sparsity: SparsityProfile | None,
    default_model: str = "SeasonalNaive",
) -> str:
    """Get recommended model for a specific series.

    Args:
        unique_id: Series identifier
        sparsity: Sparsity profile
        default_model: Default model if no specific recommendation

    Returns:
        Model name for the series
    """
    if sparsity is None:
        return default_model

    classification = sparsity.get_classification(unique_id)

    if classification.value == "intermittent":
        return "SeasonalNaive"
    elif classification.value == "cold_start":
        return "HistoricAverage"
    elif classification.value == "sparse":
        return "SeasonalNaive"
    else:
        return default_model
