"""TSFM availability and candidate policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, cast

from tsagentkit.contracts import ETSFMRequiredUnavailable, RouterThresholds, TaskSpec

if TYPE_CHECKING:
    from tsagentkit.series import TSDataset


class AdapterAvailabilityEntry(TypedDict):
    available: bool
    reason: str


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
) -> dict[str, AdapterAvailabilityEntry]:
    """Inspect TSFM adapter availability for explicit policy decisions."""
    from tsagentkit.models.adapters import AdapterRegistry

    report: dict[str, AdapterAvailabilityEntry] = {}
    for name in preferred or ["chronos", "moirai", "timesfm"]:
        is_available, reason = AdapterRegistry.check_availability(name)
        report[name] = {
            "available": is_available,
            "reason": reason,
        }
    return report


def build_candidate_list(
    buckets: list[str],
    thresholds: RouterThresholds,
) -> list[str]:
    """Build merged candidate list from buckets preserving priority order."""
    candidate_sets: list[list[str]] = []

    if "intermittent" in buckets:
        candidate_sets.append(list(thresholds.intermittent_candidates))
    if "short_history" in buckets:
        candidate_sets.append(list(thresholds.short_history_candidates))
    if "seasonal_candidate" in buckets:
        candidate_sets.append(list(thresholds.seasonal_candidates))
    if "trend" in buckets:
        candidate_sets.append(list(thresholds.trend_candidates))
    if "high_frequency" in buckets:
        candidate_sets.append(list(thresholds.high_frequency_candidates))

    if not candidate_sets:
        candidate_sets.append(list(thresholds.default_candidates))

    seen: set[str] = set()
    candidates: list[str] = []
    for cset in candidate_sets:
        for candidate in cset:
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)

    return candidates


def resolve_tsfm_policy(
    mode: str,
    tsfm_models: list[str],
    candidates: list[str],
    buckets: list[str],
    availability: TSFMAvailabilityReport,
) -> tuple[list[str], bool]:
    """Apply TSFM policy to determine final candidates and baseline allowance."""
    if mode == "disabled":
        return candidates, True

    if mode == "required":
        if not tsfm_models:
            raise ETSFMRequiredUnavailable(
                "TSFM adapters unavailable but mode is 'required'.",
                context={
                    "mode": mode,
                    "preferred_adapters": availability.preferred,
                    "unavailable": availability.unavailable,
                    "allowed_by_guardrail": availability.allowed_by_guardrail,
                },
            )
        return tsfm_models + candidates, True

    if tsfm_models and (
        "intermittent" not in buckets or not availability.allow_non_tsfm_fallback
    ):
        if availability.allow_non_tsfm_fallback:
            return tsfm_models + candidates, True
        return tsfm_models, False

    if not tsfm_models and not availability.allow_non_tsfm_fallback:
        raise ETSFMRequiredUnavailable(
            "TSFM adapters unavailable and policy disallows non-TSFM fallback.",
            context={
                "mode": mode,
                "preferred_adapters": availability.preferred,
                "unavailable": availability.unavailable,
                "allowed_by_guardrail": availability.allowed_by_guardrail,
            },
        )

    return candidates, availability.allow_non_tsfm_fallback


def resolve_tsfm_availability(
    dataset: TSDataset,
    task_spec: TaskSpec,
    thresholds: RouterThresholds,
    use_tsfm: bool,
    tsfm_preference: list[str] | None,
) -> TSFMAvailabilityReport:
    """Resolve TSFM availability report from policy and runtime checks."""
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
            unavailable=cast(dict[str, str], dict.fromkeys(preferred, "routing_guardrail_blocked")),
            allow_non_tsfm_fallback=allow_non_tsfm_fallback,
        )

    report = inspect_tsfm_adapters(preferred)
    available = [name for name, details in report.items() if details["available"]]
    unavailable = {
        name: details["reason"]
        for name, details in report.items()
        if not details["available"]
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


def _tsfm_allowed(dataset: TSDataset, thresholds: RouterThresholds) -> bool:
    if dataset.n_series > thresholds.max_series_count_for_tsfm:
        return False
    max_points = dataset.df.groupby("unique_id").size().max()
    return max_points <= thresholds.max_points_per_series_for_tsfm


__all__ = [
    "TSFMAvailabilityReport",
    "build_candidate_list",
    "inspect_tsfm_adapters",
    "resolve_tsfm_availability",
    "resolve_tsfm_policy",
]
