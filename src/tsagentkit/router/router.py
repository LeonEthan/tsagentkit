"""Deterministic routing logic aligned to the PRD PlanSpec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsagentkit.contracts import (
    ETSFMRequiredUnavailable,
    PlanSpec,
    RouteDecision,
    RouterConfig,
    RouterThresholds,
    TaskSpec,
)
from tsagentkit.time import normalize_pandas_freq

if TYPE_CHECKING:
    from tsagentkit.qa import QAReport
    from tsagentkit.series import TSDataset


@dataclass(frozen=True)
class TSFMAvailabilityReport:
    mode: str
    enabled: bool
    allowed_by_guardrail: bool
    preferred: list[str]
    available: list[str]
    unavailable: dict[str, str]
    allow_non_tsfm_fallback: bool


def inspect_tsfm_adapters(
    preferred: list[str] | None = None,
) -> dict[str, dict[str, str | bool]]:
    """Inspect TSFM adapter availability for explicit policy decisions."""
    from tsagentkit.models.adapters import AdapterRegistry

    report: dict[str, dict[str, str | bool]] = {}
    for name in (preferred or ["chronos", "moirai", "timesfm"]):
        is_available, reason = AdapterRegistry.check_availability(name)
        report[name] = {
            "available": is_available,
            "reason": reason,
        }
    return report


def make_plan(
    dataset: TSDataset,
    task_spec: TaskSpec,
    qa: QAReport | None = None,
    router_config: RouterConfig | None = None,
    use_tsfm: bool = True,
    tsfm_preference: list[str] | None = None,
) -> tuple[PlanSpec, RouteDecision]:
    """Create a deterministic PlanSpec and RouteDecision for a dataset.

    Returns:
        Tuple of (PlanSpec, RouteDecision) containing the execution plan
        and detailed routing decision information.
    """
    thresholds = (router_config or RouterConfig()).thresholds
    stats, buckets = _compute_router_stats(dataset, task_spec, thresholds)
    availability = _resolve_tsfm_availability(
        dataset=dataset,
        task_spec=task_spec,
        thresholds=thresholds,
        use_tsfm=use_tsfm,
        tsfm_preference=tsfm_preference,
    )

    if (
        availability.mode == "required"
        and not availability.available
    ):
        raise ETSFMRequiredUnavailable(
            "TSFM policy requires at least one available adapter, but none were found.",
            context={
                "mode": availability.mode,
                "preferred_adapters": availability.preferred,
                "unavailable": availability.unavailable,
                "allowed_by_guardrail": availability.allowed_by_guardrail,
            },
        )

    # Build candidate model list
    if "intermittent" in buckets:
        candidates = ["Croston", "Naive"]
    elif "short_history" in buckets:
        candidates = ["HistoricAverage", "Naive"]
    else:
        candidates = ["SeasonalNaive", "HistoricAverage", "Naive"]

    tsfm_models = [f"tsfm-{name}" for name in availability.available]

    if availability.mode == "disabled":
        allow_baseline = True
    elif availability.mode == "required":
        candidates = tsfm_models
        allow_baseline = False
    else:
        if tsfm_models and (
            "intermittent" not in buckets or not availability.allow_non_tsfm_fallback
        ):
            if availability.allow_non_tsfm_fallback:
                candidates = tsfm_models + candidates
            else:
                candidates = tsfm_models
        elif not tsfm_models and not availability.allow_non_tsfm_fallback:
            raise ETSFMRequiredUnavailable(
                "TSFM adapters unavailable and policy disallows non-TSFM fallback.",
                context={
                    "mode": availability.mode,
                    "preferred_adapters": availability.preferred,
                    "unavailable": availability.unavailable,
                    "allowed_by_guardrail": availability.allowed_by_guardrail,
                },
            )
        allow_baseline = availability.allow_non_tsfm_fallback

    plan = PlanSpec(
        plan_name="default",
        candidate_models=candidates,
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

    # Build RouteDecision for audit trail
    reasons = [
        f"selected_models: {candidates}",
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

    route_decision = RouteDecision(
        stats=stats,
        buckets=buckets,
        selected_plan=plan,
        reasons=reasons,
    )

    return plan, route_decision


def get_model_for_series(
    unique_id: str,
    dataset: TSDataset,
    task_spec: TaskSpec,
    thresholds: RouterThresholds | None = None,
) -> str:
    """Get recommended model for a specific series."""
    thresholds = thresholds or RouterThresholds()
    series_df = dataset.get_series(unique_id)
    stats, buckets = _compute_series_stats(series_df, task_spec, thresholds)

    if "intermittent" in buckets:
        return "Croston"
    if "short_history" in buckets:
        return "HistoricAverage"
    return "SeasonalNaive"


def _compute_router_stats(
    dataset: TSDataset,
    task_spec: TaskSpec,
    thresholds: RouterThresholds,
) -> tuple[dict[str, float], list[str]]:
    df = dataset.df
    stats: dict[str, float] = {}
    buckets: list[str] = []

    lengths = df.groupby("unique_id").size()
    min_len = int(lengths.min()) if not lengths.empty else 0
    stats["min_series_length"] = float(min_len)
    if min_len < thresholds.min_train_size:
        buckets.append("short_history")

    missing_ratio = _compute_missing_ratio(df, task_spec)
    stats["missing_ratio"] = float(missing_ratio)
    if missing_ratio > thresholds.max_missing_ratio:
        buckets.append("sparse")

    uid_col = task_spec.panel_contract.unique_id_col
    ds_col = task_spec.panel_contract.ds_col
    y_col = task_spec.panel_contract.y_col

    intermittency = _compute_intermittency(df, thresholds, uid_col, ds_col, y_col)
    stats.update(intermittency)
    if intermittency.get("intermittent_series_ratio", 0.0) > 0:
        buckets.append("intermittent")

    season_conf = _seasonality_confidence(df, task_spec, uid_col, y_col)
    stats["seasonality_confidence"] = float(season_conf)
    if season_conf >= thresholds.min_seasonality_conf:
        buckets.append("seasonal_candidate")

    return stats, buckets


def _compute_series_stats(
    series_df: pd.DataFrame,
    task_spec: TaskSpec,
    thresholds: RouterThresholds,
) -> tuple[dict[str, float], list[str]]:
    stats: dict[str, float] = {}
    buckets: list[str] = []

    length = len(series_df)
    stats["series_length"] = float(length)
    if length < thresholds.min_train_size:
        buckets.append("short_history")

    missing_ratio = _compute_missing_ratio(series_df, task_spec)
    stats["missing_ratio"] = float(missing_ratio)
    if missing_ratio > thresholds.max_missing_ratio:
        buckets.append("sparse")

    uid_col = task_spec.panel_contract.unique_id_col
    ds_col = task_spec.panel_contract.ds_col
    y_col = task_spec.panel_contract.y_col

    intermittency = _compute_intermittency(series_df, thresholds, uid_col, ds_col, y_col)
    stats.update(intermittency)
    if intermittency.get("intermittent_series_ratio", 0.0) > 0:
        buckets.append("intermittent")

    season_conf = _seasonality_confidence(series_df, task_spec, uid_col, y_col)
    stats["seasonality_confidence"] = float(season_conf)
    if season_conf >= thresholds.min_seasonality_conf:
        buckets.append("seasonal_candidate")

    return stats, buckets


def _compute_missing_ratio(df: pd.DataFrame, task_spec: TaskSpec) -> float:
    if df.empty:
        return 0.0
    uid_col = task_spec.panel_contract.unique_id_col
    ds_col = task_spec.panel_contract.ds_col

    ratios = []
    for uid in df[uid_col].unique():
        series = df[df[uid_col] == uid].sort_values(ds_col)
        if series.empty:
            continue
        full_range = pd.date_range(
            start=series[ds_col].min(),
            end=series[ds_col].max(),
            freq=normalize_pandas_freq(task_spec.freq),
        )
        missing = len(full_range) - len(series)
        ratio = missing / max(len(full_range), 1)
        ratios.append(ratio)
    return float(np.mean(ratios)) if ratios else 0.0


def _compute_intermittency(
    df: pd.DataFrame,
    thresholds: RouterThresholds,
    uid_col: str,
    ds_col: str,
    y_col: str,
) -> dict[str, float]:
    intermittent = 0
    total = 0

    for uid in df[uid_col].unique():
        series = df[df[uid_col] == uid].sort_values(ds_col)
        y = series[y_col].values
        total += 1

        non_zero_idx = np.where(y > 0)[0]
        if len(non_zero_idx) <= 1:
            adi = float("inf")
            cv2 = float("inf")
        else:
            intervals = np.diff(non_zero_idx)
            adi = float(np.mean(intervals)) if len(intervals) > 0 else float("inf")
            non_zero_vals = y[non_zero_idx]
            mean = np.mean(non_zero_vals) if len(non_zero_vals) > 0 else 0.0
            std = np.std(non_zero_vals) if len(non_zero_vals) > 0 else 0.0
            cv2 = float((std / mean) ** 2) if mean != 0 else float("inf")

        if adi >= thresholds.max_intermittency_adi and cv2 >= thresholds.max_intermittency_cv2:
            intermittent += 1

    ratio = intermittent / total if total > 0 else 0.0
    return {
        "intermittent_series_ratio": ratio,
        "intermittent_series_count": float(intermittent),
    }


def _seasonality_confidence(
    df: pd.DataFrame,
    task_spec: TaskSpec,
    uid_col: str,
    y_col: str,
) -> float:
    season_length = task_spec.season_length
    if season_length is None or season_length <= 1:
        return 0.0
    confs: list[float] = []
    for uid in df[uid_col].unique():
        series = df[df[uid_col] == uid][y_col].values
        if len(series) <= season_length:
            continue
        series = series - np.mean(series)
        denom = np.dot(series, series)
        if denom == 0:
            continue
        lagged = np.roll(series, season_length)
        corr = np.dot(series[season_length:], lagged[season_length:]) / denom
        confs.append(abs(float(corr)))
    return float(np.mean(confs)) if confs else 0.0


def _tsfm_allowed(dataset: TSDataset, thresholds: RouterThresholds) -> bool:
    if dataset.n_series > thresholds.max_series_count_for_tsfm:
        return False
    max_points = dataset.df.groupby("unique_id").size().max()
    return max_points <= thresholds.max_points_per_series_for_tsfm


def _resolve_tsfm_availability(
    dataset: TSDataset,
    task_spec: TaskSpec,
    thresholds: RouterThresholds,
    use_tsfm: bool,
    tsfm_preference: list[str] | None,
) -> TSFMAvailabilityReport:
    policy = task_spec.tsfm_policy
    mode = "disabled" if not use_tsfm else policy.mode
    preferred = list(tsfm_preference or policy.adapters)
    allow_non_tsfm_fallback = bool(policy.allow_non_tsfm_fallback)
    enabled = mode != "disabled"
    allowed_by_guardrail = _tsfm_allowed(dataset, thresholds)

    if not enabled:
        return TSFMAvailabilityReport(
            mode=mode,
            enabled=False,
            allowed_by_guardrail=allowed_by_guardrail,
            preferred=preferred,
            available=[],
            unavailable={},
            allow_non_tsfm_fallback=True,
        )

    if not allowed_by_guardrail:
        return TSFMAvailabilityReport(
            mode=mode,
            enabled=True,
            allowed_by_guardrail=False,
            preferred=preferred,
            available=[],
            unavailable={name: "routing_guardrail_blocked" for name in preferred},
            allow_non_tsfm_fallback=allow_non_tsfm_fallback,
        )

    report = inspect_tsfm_adapters(preferred)
    available = [name for name, details in report.items() if details.get("available")]
    unavailable = {
        name: str(details.get("reason") or "")
        for name, details in report.items()
        if not details.get("available")
    }
    return TSFMAvailabilityReport(
        mode=mode,
        enabled=True,
        allowed_by_guardrail=True,
        preferred=preferred,
        available=available,
        unavailable=unavailable,
        allow_non_tsfm_fallback=allow_non_tsfm_fallback,
    )


__all__ = ["make_plan", "get_model_for_series", "RouteDecision", "inspect_tsfm_adapters"]
