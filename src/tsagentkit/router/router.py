"""Model selection and routing logic.

Routes TSDatasets to appropriate models based on data characteristics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tsagentkit.contracts import TaskSpec
from tsagentkit.hierarchy import ReconciliationMethod

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
    use_tsfm: bool = True,
    tsfm_preference: list[str] | None = None,
) -> Plan:
    """Create an execution plan for a dataset.

    Analyzes the dataset characteristics and selects appropriate models
    with fallback chains.

    Args:
        dataset: TSDataset to create plan for
        task_spec: Task specification
        qa: Optional QA report for data quality considerations
        strategy: Routing strategy ("auto", "baseline_only", "tsfm_first")
        use_tsfm: Whether to consider TSFMs (default: True)
        tsfm_preference: Ordered list of preferred TSFMs

    Returns:
        Plan with model selection and fallback chain

    Raises:
        ValueError: If unknown strategy specified
    """
    # Get sparsity profile
    sparsity = dataset.sparsity_profile

    # Determine primary model and fallbacks based on strategy
    if strategy == "auto":
        return _make_auto_plan(
            dataset, task_spec, sparsity, use_tsfm, tsfm_preference
        )
    elif strategy == "baseline_only":
        return _make_baseline_plan(task_spec, sparsity)
    elif strategy == "tsfm_first":
        # TSFM-first strategy (for v1.0)
        return _make_tsfm_plan(
            dataset, task_spec, sparsity, tsfm_preference
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _make_auto_plan(
    dataset: TSDataset,
    task_spec: TaskSpec,
    sparsity: SparsityProfile | None,
    use_tsfm: bool = True,
    tsfm_preference: list[str] | None = None,
) -> Plan:
    """Create plan with automatic model selection.

    For v1.0, this includes TSFM support and hierarchical forecasting.
    """
    # Analyze data characteristics
    has_intermittent = sparsity.has_intermittent() if sparsity else False
    has_cold_start = sparsity.has_cold_start() if sparsity else False

    # Get season length from task spec or default
    season_length = task_spec.season_length or 1

    # Check TSFM availability
    available_tsfms = []
    if use_tsfm:
        from tsagentkit.models.adapters import AdapterRegistry

        for tsfm_name in (tsfm_preference or ["chronos", "moirai", "timesfm"]):
            is_avail, _ = AdapterRegistry.check_availability(tsfm_name)
            if is_avail:
                available_tsfms.append(tsfm_name)

    # Hierarchy-aware planning
    if dataset.is_hierarchical():
        return _make_hierarchical_plan(
            dataset, task_spec, sparsity, available_tsfms
        )

    # Select primary model based on data characteristics
    if available_tsfms and not has_intermittent and not has_cold_start:
        # Use TSFM for regular series when available
        primary = f"tsfm-{available_tsfms[0]}"
        fallback_chain = [f"tsfm-{t}" for t in available_tsfms[1:]] + \
                         FallbackLadder.STANDARD_LADDER
    elif has_intermittent:
        primary = "SeasonalNaive"
        fallback_chain = FallbackLadder.INTERMITTENT_LADDER[1:]
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


def _make_hierarchical_plan(
    dataset: TSDataset,
    task_spec: TaskSpec,
    sparsity: SparsityProfile | None,
    available_tsfms: list[str],
) -> Plan:
    """Create plan for hierarchical forecasting.

    Args:
        dataset: Time series dataset with hierarchy
        task_spec: Task specification
        sparsity: Optional sparsity profile
        available_tsfms: List of available TSFM names

    Returns:
        Plan with hierarchical reconciliation strategy
    """
    hierarchy = dataset.hierarchy
    assert hierarchy is not None  # Checked by caller

    # Determine reconciliation strategy
    n_levels = hierarchy.get_num_levels()

    # For deep hierarchies, use MinT; for shallow, bottom-up may suffice
    if n_levels > 2:
        reconciliation_method = ReconciliationMethod.MIN_TRACE
    else:
        reconciliation_method = ReconciliationMethod.BOTTOM_UP

    # Build model selection
    if available_tsfms:
        primary = f"tsfm-{available_tsfms[0]}"
        fallback_chain = [f"tsfm-{t}" for t in available_tsfms[1:]] + \
                         ["theta", "seasonal_naive"]
    else:
        primary = "seasonal_naive"
        fallback_chain = ["theta", "historic_average"]

    config = {
        "hierarchical": True,
        "reconciliation_method": reconciliation_method.value,
        "hierarchy_levels": n_levels,
        "season_length": task_spec.season_length or 1,
        "horizon": task_spec.horizon,
    }

    if task_spec.quantiles:
        config["quantiles"] = task_spec.quantiles

    return Plan(
        primary_model=primary,
        fallback_chain=fallback_chain,
        config=config,
        strategy="hierarchical",
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
    tsfm_preference: list[str] | None = None,
) -> Plan:
    """Create plan with TSFM-first strategy.

    For v1.0, this attempts TSFM models first with fallback to baselines.
    """
    from tsagentkit.models.adapters import AdapterRegistry

    # Check available TSFMs
    available_tsfms = []
    for tsfm_name in (tsfm_preference or ["chronos", "moirai", "timesfm"]):
        is_avail, _ = AdapterRegistry.check_availability(tsfm_name)
        if is_avail:
            available_tsfms.append(tsfm_name)

    # Get season length
    season_length = task_spec.season_length or 1

    if available_tsfms:
        # Use first available TSFM as primary
        primary = f"tsfm-{available_tsfms[0]}"
        # Remaining TSFMs + baseline fallbacks
        fallback_chain = [f"tsfm-{t}" for t in available_tsfms[1:]]

        # Add baseline fallbacks based on data characteristics
        has_intermittent = sparsity.has_intermittent() if sparsity else False
        if has_intermittent:
            fallback_chain.extend(FallbackLadder.INTERMITTENT_LADDER)
        else:
            fallback_chain.extend(FallbackLadder.STANDARD_LADDER)
    else:
        # No TSFMs available, fall back to standard plan
        return _make_auto_plan(dataset, task_spec, sparsity, use_tsfm=False)

    config = {
        "season_length": season_length,
        "horizon": task_spec.horizon,
        "tsfm_models": available_tsfms,
    }

    if task_spec.quantiles:
        config["quantiles"] = task_spec.quantiles

    return Plan(
        primary_model=primary,
        fallback_chain=fallback_chain,
        config=config,
        strategy="tsfm_first",
    )


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
