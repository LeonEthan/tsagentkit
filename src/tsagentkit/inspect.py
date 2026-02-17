"""Inspection utilities for tsagentkit.

Provides functions to list models, check health, and diagnose issues.
"""

from __future__ import annotations

from dataclasses import dataclass

from tsagentkit.models.registry import REGISTRY, check_available, list_models as registry_list_models


@dataclass(frozen=True)
class HealthReport:
    """Health check report for tsagentkit."""

    tsfm_available: list[str]
    tsfm_missing: list[str]
    baselines_available: bool
    all_ok: bool

    def __str__(self) -> str:
        lines = ["tsagentkit Health Report", "=" * 40]
        lines.append(f"TSFMs available: {', '.join(self.tsfm_available) or 'None'}")
        if self.tsfm_missing:
            lines.append(f"TSFMs missing: {', '.join(self.tsfm_missing)}")
        lines.append(f"Baselines available: {self.baselines_available}")
        lines.append(f"Overall: {'OK' if self.all_ok else 'Issues detected'}")
        return "\n".join(lines)


def list_models(tsfm_only: bool = False) -> list[str]:
    """List available models.

    Args:
        tsfm_only: If True, only return TSFM models

    Returns:
        List of model names with available dependencies
    """
    return registry_list_models(tsfm_only=tsfm_only, available_only=True)


def check_health() -> HealthReport:
    """Check tsagentkit health status.

    Returns:
        HealthReport with available/missing models
    """
    tsfm_available = []
    tsfm_missing = []

    for name, spec in REGISTRY.items():
        if not spec.is_tsfm:
            continue

        if check_available(spec):
            tsfm_available.append(name)
        else:
            missing = ", ".join(spec.requires)
            tsfm_missing.append(f"{name} (needs: {missing})")

    # Check baselines
    baseline_spec = REGISTRY.get("naive")
    baselines_available = check_available(baseline_spec) if baseline_spec else False

    # All OK if at least one TSFM or baselines are available
    all_ok = len(tsfm_available) > 0 or baselines_available

    return HealthReport(
        tsfm_available=tsfm_available,
        tsfm_missing=tsfm_missing,
        baselines_available=baselines_available,
        all_ok=all_ok,
    )


__all__ = ["list_models", "check_health", "HealthReport"]
