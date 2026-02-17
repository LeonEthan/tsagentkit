"""Model registry - single source of truth for all models.

This module defines the REGISTRY dictionary containing all available models
and their specifications. Adding a new model requires only adding an entry here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a forecasting model.

    Attributes:
        name: Unique model identifier
        adapter_path: Import path to the adapter module (e.g., "tsagentkit.models.adapters.chronos")
        config_fields: Default configuration parameters for the model
        requires: List of required Python packages
        is_tsfm: Whether this is a Time-Series Foundation Model
    """

    name: str
    adapter_path: str
    config_fields: dict[str, Any]
    requires: list[str]
    is_tsfm: bool


# Single source of truth for all models
# To add a new model, simply add an entry here
REGISTRY: dict[str, ModelSpec] = {
    "chronos": ModelSpec(
        name="chronos",
        adapter_path="tsagentkit.models.adapters.tsfm.chronos",
        config_fields={"model_name": "amazon/chronos-2"},
        requires=["chronos", "torch"],  # pip install chronos-forecasting
        is_tsfm=True,
    ),
    "timesfm": ModelSpec(
        name="timesfm",
        adapter_path="tsagentkit.models.adapters.tsfm.timesfm",
        config_fields={},
        requires=["tsagentkit_timesfm", "torch"],  # pip install tsagentkit-timesfm
        is_tsfm=True,
    ),
    "moirai": ModelSpec(
        name="moirai",
        adapter_path="tsagentkit.models.adapters.tsfm.moirai",
        config_fields={"model_name": "Salesforce/moirai-2.0-R-small"},
        requires=["tsagentkit_uni2ts", "gluonts", "torch"],  # pip install tsagentkit-uni2ts
        is_tsfm=True,
    ),
    "patchtst_fm": ModelSpec(
        name="patchtst_fm",
        adapter_path="tsagentkit.models.adapters.tsfm.patchtst_fm",
        config_fields={"model_name": "ibm-research/patchtst-fm-r1"},
        requires=["tsfm_public", "torch"],  # pip install tsagentkit-patchtst-fm
        is_tsfm=True,
    ),
    "naive": ModelSpec(
        name="naive",
        adapter_path="tsagentkit.models.adapters.baseline.naive",
        config_fields={},
        requires=["statsforecast"],
        is_tsfm=False,
    ),
    "seasonal_naive": ModelSpec(
        name="seasonal_naive",
        adapter_path="tsagentkit.models.adapters.baseline.seasonal",
        config_fields={},
        requires=["statsforecast"],
        is_tsfm=False,
    ),
}


def list_models(tsfm_only: bool = False, available_only: bool = False) -> list[str]:
    """List models from the registry.

    Args:
        tsfm_only: If True, only return TSFM models
        available_only: If True, only return models with satisfied dependencies

    Returns:
        List of model names
    """
    names: list[str] = []
    for name, spec in REGISTRY.items():
        if tsfm_only and not spec.is_tsfm:
            continue
        if available_only and not check_available(spec):
            continue
        names.append(name)
    return names


def get_spec(name: str) -> ModelSpec:
    """Get model specification by name.

    Args:
        name: Model name

    Returns:
        ModelSpec for the named model

    Raises:
        KeyError: If model not found in registry
    """
    if name not in REGISTRY:
        available = ", ".join(list_models())
        raise KeyError(f"Model '{name}' not found. Available: {available}")
    return REGISTRY[name]


def check_available(spec: ModelSpec) -> bool:
    """Check if a model's dependencies are available.

    TSFM dependencies are now default, so they should always be available.
    This function is kept for backward compatibility and for checking
    optional dependencies in the future.

    Args:
        spec: Model specification to check

    Returns:
        True if all required packages can be imported
    """
    # TSFMs are now default dependencies - always return True
    if spec.is_tsfm:
        return True

    # Check non-TSFM models (baselines, etc.)
    for package in spec.requires:
        try:
            __import__(package)
        except ImportError:
            return False
    return True


def list_available(tsfm_only: bool = False) -> list[str]:
    """List models that have all dependencies available.

    Args:
        tsfm_only: If True, only check TSFM models

    Returns:
        List of model names with available dependencies
    """
    return list_models(tsfm_only=tsfm_only, available_only=True)


__all__ = [
    "ModelSpec",
    "REGISTRY",
    "list_models",
    "get_spec",
    "check_available",
    "list_available",
]
