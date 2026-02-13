"""Deterministic routing logic aligned to the PRD PlanSpec."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tsagentkit.contracts import (
    ETSFMRequiredUnavailable,
    PlanSpec,
    RouteDecision,
    RouterConfig,
    RouterThresholds,
    TaskSpec,
)

from .stats import compute_router_stats, compute_series_stats
from .tsfm_policy import (
    TSFMAvailabilityReport,
    build_candidate_list,
    inspect_tsfm_adapters,
    resolve_tsfm_availability,
    resolve_tsfm_policy,
)

if TYPE_CHECKING:
    from tsagentkit.qa import QAReport
    from tsagentkit.series import TSDataset


def _build_route_decision(
    stats: dict[str, float],
    buckets: list[str],
    plan: PlanSpec,
    availability: TSFMAvailabilityReport,
) -> RouteDecision:
    reasons = [
        f"selected_models: {plan.candidate_models}",
        f"buckets: {buckets}",
        f"tsfm_mode: {availability.mode}",
        f"tsfm_enabled: {availability.enabled}",
        f"tsfm_allowed_by_guardrail: {availability.allowed_by_guardrail}",
        f"tsfm_available: {bool(availability.available)}",
        f"allow_non_tsfm_fallback: {availability.allow_non_tsfm_fallback}",
    ]
    if availability.available:
        reasons.append(f"tsfm_models: {availability.available}")
    if availability.unavailable:
        reasons.append(f"tsfm_unavailable: {availability.unavailable}")

    return RouteDecision(
        stats=stats,
        buckets=buckets,
        selected_plan=plan,
        reasons=reasons,
    )


def make_plan(
    dataset: TSDataset,
    task_spec: TaskSpec,
    qa: QAReport | None = None,
    router_config: RouterConfig | None = None,
    use_tsfm: bool = True,
    tsfm_preference: list[str] | None = None,
) -> tuple[PlanSpec, RouteDecision]:
    """Create a deterministic PlanSpec and RouteDecision for a dataset."""
    _ = qa
    thresholds = (router_config or RouterConfig()).thresholds
    stats, buckets = compute_router_stats(dataset.df, task_spec, thresholds)
    availability = resolve_tsfm_availability(
        dataset=dataset,
        task_spec=task_spec,
        thresholds=thresholds,
        use_tsfm=use_tsfm,
        tsfm_preference=tsfm_preference,
    )

    if availability.mode == "required" and not availability.available:
        raise ETSFMRequiredUnavailable(
            "TSFM policy requires at least one available adapter, but none were found.",
            context={
                "mode": availability.mode,
                "preferred_adapters": availability.preferred,
                "unavailable": availability.unavailable,
                "allowed_by_guardrail": availability.allowed_by_guardrail,
            },
        )

    candidates = build_candidate_list(buckets, thresholds)
    tsfm_models = [f"tsfm-{name}" for name in availability.available]
    final_candidates, allow_baseline = resolve_tsfm_policy(
        mode=availability.mode,
        tsfm_models=tsfm_models,
        candidates=candidates,
        buckets=buckets,
        availability=availability,
    )

    plan = PlanSpec(
        plan_name="default",
        candidate_models=final_candidates,
        use_static=True,
        use_past=True,
        use_future_known=True,
        min_train_size=thresholds.min_train_size,
        max_train_size=thresholds.max_points_per_series_for_tsfm,
        interval_mode=task_spec.forecast_contract.interval_mode,
        levels=task_spec.forecast_contract.levels,
        quantiles=task_spec.forecast_contract.quantiles,
        allow_drop_covariates=True,
        allow_baseline=allow_baseline,
    )

    route_decision = _build_route_decision(stats, buckets, plan, availability)
    return plan, route_decision


def get_model_for_series(
    unique_id: str,
    dataset: TSDataset,
    task_spec: TaskSpec,
    thresholds: RouterThresholds | None = None,
) -> str:
    """Get recommended model for a specific series based on feature-driven analysis."""
    thresholds = thresholds or RouterThresholds()
    series_df = dataset.get_series(unique_id)
    _stats, buckets = compute_series_stats(series_df, task_spec, thresholds)

    if "intermittent" in buckets:
        return thresholds.intermittent_candidates[0] if thresholds.intermittent_candidates else "Croston"
    if "short_history" in buckets:
        return thresholds.short_history_candidates[0] if thresholds.short_history_candidates else "HistoricAverage"
    if "trend" in buckets:
        return thresholds.trend_candidates[0] if thresholds.trend_candidates else "Trend"
    if "seasonal_candidate" in buckets:
        return thresholds.seasonal_candidates[0] if thresholds.seasonal_candidates else "SeasonalNaive"
    if "high_frequency" in buckets:
        return thresholds.high_frequency_candidates[0] if thresholds.high_frequency_candidates else "RobustBaseline"
    return thresholds.default_candidates[0] if thresholds.default_candidates else "SeasonalNaive"


__all__ = ["make_plan", "get_model_for_series", "RouteDecision", "inspect_tsfm_adapters"]

