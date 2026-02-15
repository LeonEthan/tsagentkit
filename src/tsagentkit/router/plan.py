"""Simplified plan representation.

Replaces PlanSpec, PlanGraphSpec, and PlanNodeSpec with a single Plan dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tsagentkit.core.data import TSDataset


@dataclass(frozen=True)
class ModelCandidate:
    """Single model candidate for forecasting."""

    name: str
    is_tsfm: bool = False
    adapter_name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Plan:
    """Simplified execution plan.

    Replaces the complex PlanSpec/PlanGraphSpec/PlanNodeSpec hierarchy
    with a single flat structure: primary + fallbacks.
    """

    primary: ModelCandidate
    fallbacks: list[ModelCandidate] = field(default_factory=list)
    backtest_enabled: bool = True

    def all_candidates(self) -> list[ModelCandidate]:
        """Return all candidates in order of preference."""
        return [self.primary] + self.fallbacks

    def execute(self, dataset: TSDataset) -> Any:
        """Execute plan with automatic fallback.

        Tries each candidate in order until one succeeds.
        """
        from tsagentkit.core.errors import EModelFailed

        errors = []

        for candidate in self.all_candidates():
            try:
                return self._fit_candidate(candidate, dataset)
            except Exception as e:
                errors.append((candidate.name, str(e)))
                continue

        raise EModelFailed(
            f"All candidates failed: {errors}",
            context={"attempted": [c.name for c in self.all_candidates()], "errors": errors},
        )

    def _fit_candidate(self, candidate: ModelCandidate, dataset: TSDataset) -> Any:
        """Fit a single candidate model."""
        if candidate.is_tsfm:
            from tsagentkit.models import fit_tsfm

            return fit_tsfm(dataset, candidate.adapter_name or candidate.name.replace("tsfm-", ""))
        else:
            from tsagentkit.models import fit

            return fit(dataset, candidate.name)


def inspect_tsfm_adapters() -> list[str]:
    """Check which TSFM adapters are available.

    Returns list of available adapter names (e.g., ['chronos', 'moirai', 'timesfm'])
    """
    adapters = []
    try:
        import importlib

        for name in ["chronos", "moirai", "timesfm"]:
            try:
                module = importlib.import_module(f"tsagentkit.models.adapters.{name}")
                if hasattr(module, f"{name.capitalize()}Adapter"):
                    adapters.append(name)
            except ImportError:
                pass
    except Exception:
        pass
    return adapters


def build_plan(
    dataset: TSDataset,
    tsfm_mode: str = "required",
    allow_fallback: bool = True,
) -> Plan:
    """Build simplified execution plan.

    Args:
        dataset: Time-series dataset
        tsfm_mode: 'required', 'preferred', or 'disabled'
        allow_fallback: Whether to include fallback candidates

    Returns:
        Plan with primary and fallback candidates
    """
    available_tsfm = inspect_tsfm_adapters()
    candidates = []

    # Build TSFM candidates
    if tsfm_mode != "disabled":
        for name in available_tsfm:
            candidates.append(ModelCandidate(name=f"tsfm-{name}", is_tsfm=True, adapter_name=name))

    # Build fallback candidates (statistical baselines)
    fallback_candidates = []
    if allow_fallback:
        fallback_candidates = [
            ModelCandidate(name="SeasonalNaive"),
            ModelCandidate(name="HistoricAverage"),
            ModelCandidate(name="Naive"),
        ]

    # Determine primary
    if candidates:
        primary = candidates[0]
        # Add remaining TSFM as fallbacks before statistical
        fallbacks = candidates[1:] + fallback_candidates
    else:
        if tsfm_mode == "required":
            from tsagentkit.core.errors import ETSFMRequired

            raise ETSFMRequired("TSFM required but no adapters available")
        primary = fallback_candidates[0] if fallback_candidates else ModelCandidate(name="Naive")
        fallbacks = fallback_candidates[1:] if len(fallback_candidates) > 1 else []

    return Plan(
        primary=primary,
        fallbacks=fallbacks,
        backtest_enabled=tsfm_mode != "quick",
    )


__all__ = [
    "Plan",
    "ModelCandidate",
    "build_plan",
    "inspect_tsfm_adapters",
]
