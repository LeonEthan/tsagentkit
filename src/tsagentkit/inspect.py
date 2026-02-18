"""Inspection utilities for tsagentkit.

Provides functions to list models, check health, and diagnose issues.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from tsagentkit.models.registry import REGISTRY
from tsagentkit.models.registry import list_models as registry_list_models


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
    """List registered models.

    Args:
        tsfm_only: If True, only return TSFM models

    Returns:
        List of model names from the registry
    """
    return registry_list_models(tsfm_only=tsfm_only)


def _packages_available(packages: list[str]) -> bool:
    """Return True when all package specs can be resolved."""
    return all(importlib.util.find_spec(package) is not None for package in packages)


def check_health() -> HealthReport:
    """Check tsagentkit health status.

    Returns:
        HealthReport with available/missing models
    """
    # TSFMs are mandatory dependencies in tsagentkit. Health checks expose
    # registry state, not optional dependency probing for TSFMs.
    tsfm_available = registry_list_models(tsfm_only=True)
    tsfm_missing: list[str] = []

    # Baselines are optional.
    baseline_spec = REGISTRY.get("naive")
    baselines_available = _packages_available(baseline_spec.requires) if baseline_spec else False

    # All OK when TSFM registry is populated (baselines are optional).
    all_ok = len(tsfm_available) > 0

    return HealthReport(
        tsfm_available=tsfm_available,
        tsfm_missing=tsfm_missing,
        baselines_available=baselines_available,
        all_ok=all_ok,
    )


__all__ = ["list_models", "check_health", "HealthReport"]
