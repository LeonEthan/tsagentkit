"""Simplified plan representation using the model registry.

Replaces PlanSpec, PlanGraphSpec, and PlanNodeSpec with a single Plan dataclass
that works with the REGISTRY for model specifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

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
    """Ensemble execution plan.

    Replaces the fallback chain with an ensemble approach where ALL models
    participate in the final forecast via median/mean aggregation.
    """

    # TSFM models (always included if available)
    tsfm_models: list[ModelCandidate] = field(default_factory=list)
    # Statistical models (included based on plan configuration)
    statistical_models: list[ModelCandidate] = field(default_factory=list)
    # Ensemble configuration
    ensemble_method: Literal["median", "mean"] = "median"
    require_all_tsfm: bool = False
    min_models_for_ensemble: int = 1
    backtest_enabled: bool = True

    def all_models(self) -> list[ModelCandidate]:
        """Return all models in the ensemble."""
        return self.tsfm_models + self.statistical_models

    def execute(self, dataset: TSDataset) -> Any:
        """Execute ensemble plan - fit all models and aggregate predictions.

        Returns dict with fitted artifacts for all successful models.
        """
        from tsagentkit.core.errors import EInsufficient
        from tsagentkit.models.protocol import fit
        from tsagentkit.models.registry import get_spec

        artifacts = []
        errors = []

        for candidate in self.all_models():
            try:
                spec = get_spec(candidate.adapter_name or candidate.name)
                artifact = fit(spec, dataset)
                artifacts.append({"candidate": candidate, "artifact": artifact})
            except Exception as e:
                if candidate.is_tsfm and self.require_all_tsfm:
                    raise EInsufficient(
                        f"Required TSFM model '{candidate.name}' failed",
                        context={"error": str(e)},
                    )
                errors.append((candidate.name, str(e)))

        if len(artifacts) < self.min_models_for_ensemble:
            raise EInsufficient(
                f"Insufficient models succeeded: {len(artifacts)} < {self.min_models_for_ensemble}",
                context={"errors": errors, "succeeded": len(artifacts)},
            )

        return {"artifacts": artifacts, "errors": errors}


def inspect_tsfm_adapters() -> list[str]:
    """Check which TSFM adapters are available.

    Returns list of available adapter names (e.g., ['chronos', 'moirai', 'timesfm'])
    """
    from tsagentkit.models.registry import REGISTRY, check_available

    available = []
    for name, spec in REGISTRY.items():
        if spec.is_tsfm and check_available(spec):
            available.append(name)
    return available


def build_plan(
    dataset: TSDataset,
    tsfm_mode: str = "required",
    allow_fallback: bool = True,
    ensemble_method: str = "median",
    require_all_tsfm: bool = False,
) -> Plan:
    """Build ensemble execution plan.

    Args:
        dataset: Time-series dataset
        tsfm_mode: 'required', 'preferred', or 'disabled'
        allow_fallback: Whether to include statistical models
        ensemble_method: 'median' or 'mean' aggregation
        require_all_tsfm: If True, fail if any TSFM model fails

    Returns:
        Plan with all TSFM and statistical models for ensemble
    """
    from tsagentkit.core.errors import ENoTSFM
    from tsagentkit.models.registry import REGISTRY, check_available

    available_tsfm = inspect_tsfm_adapters()

    # Build TSFM candidates
    tsfm_candidates = []
    if tsfm_mode != "disabled":
        for name in available_tsfm:
            tsfm_candidates.append(ModelCandidate(name=f"tsfm-{name}", is_tsfm=True, adapter_name=name))

    # Build statistical candidates
    statistical_candidates = []
    if allow_fallback:
        # Check which baseline models are available
        for name in ["naive", "seasonal_naive"]:
            if name in REGISTRY and check_available(REGISTRY[name]):
                statistical_candidates.append(ModelCandidate(name=name))

    # Validate TSFM availability
    if tsfm_mode == "required" and not tsfm_candidates:
        raise ENoTSFM("TSFM required but no adapters available")

    return Plan(
        tsfm_models=tsfm_candidates,
        statistical_models=statistical_candidates,
        ensemble_method=ensemble_method,  # type: ignore[arg-type]
        require_all_tsfm=require_all_tsfm,
        min_models_for_ensemble=1,
        backtest_enabled=tsfm_mode != "quick",
    )


__all__ = [
    "Plan",
    "ModelCandidate",
    "build_plan",
    "inspect_tsfm_adapters",
]
