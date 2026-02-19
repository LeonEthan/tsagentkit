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
        min_context_length: Minimum input length required (None = no minimum)
        max_context_length: Maximum input length supported (None = unlimited)
        max_prediction_length: Maximum horizon supported (None = unlimited)
        pad_to_min_context: Auto-pad if below min_context_length
        truncate_to_max_context: Auto-truncate if above max_context_length
    """

    name: str
    adapter_path: str
    config_fields: dict[str, Any]
    requires: list[str]
    is_tsfm: bool

    # Length limits (None = unlimited or model-determined)
    min_context_length: int | None = None
    max_context_length: int | None = None
    max_prediction_length: int | None = None

    # Padding behavior
    pad_to_min_context: bool = True
    truncate_to_max_context: bool = True


# Single source of truth for all models
# To add a new model, simply add an entry here
REGISTRY: dict[str, ModelSpec] = {
    "chronos": ModelSpec(
        name="chronos",
        adapter_path="tsagentkit.models.adapters.tsfm.chronos",
        config_fields={"model_name": "amazon/chronos-2"},
        requires=["chronos", "torch"],  # pip install chronos-forecasting
        is_tsfm=True,
        # Chronos 2 official specs: 8K context, 1K max prediction
        # https://github.com/amazon-science/chronos-forecasting/discussions/408
        min_context_length=None,
        max_context_length=8192,
        max_prediction_length=1024,
        pad_to_min_context=False,  # Chronos handles short contexts natively
        truncate_to_max_context=True,  # Auto-truncates internally
    ),
    "timesfm": ModelSpec(
        name="timesfm",
        adapter_path="tsagentkit.models.adapters.tsfm.timesfm",
        config_fields={},
        requires=["tsagentkit_timesfm", "torch"],  # pip install tsagentkit-timesfm
        is_tsfm=True,
        # TimesFM 2.5 official specs: 16K context limit, 1K direct prediction
        # https://github.com/google-research/timesfm
        # Note: context_limit = 16384, and max_context + max_horizon <= context_limit
        # So we use max_context=15360 (16384 - 1024) to support full 1K horizon
        # Note: min_context=993 is a workaround for NaN bug (issue #321)
        min_context_length=993,
        max_context_length=15360,  # 16384 - 1024 to fit horizon within context_limit
        max_prediction_length=1024,  # Direct prediction limit; AR for longer
        pad_to_min_context=True,
        truncate_to_max_context=True,
    ),
    "moirai": ModelSpec(
        name="moirai",
        adapter_path="tsagentkit.models.adapters.tsfm.moirai",
        config_fields={"model_name": "Salesforce/moirai-2.0-R-small"},
        requires=["tsagentkit_uni2ts", "gluonts", "torch"],  # pip install tsagentkit-uni2ts
        is_tsfm=True,
        # Moirai 2.0 specs: 4K context, unlimited via AR generation
        # https://github.com/SalesforceAIResearch/uni2ts
        min_context_length=None,
        max_context_length=4096,
        max_prediction_length=None,  # Any via AR generation
        pad_to_min_context=False,
        truncate_to_max_context=True,
    ),
    "patchtst_fm": ModelSpec(
        name="patchtst_fm",
        adapter_path="tsagentkit.models.adapters.tsfm.patchtst_fm",
        config_fields={"model_name": "ibm-research/patchtst-fm-r1"},
        requires=["tsfm_public", "torch"],  # pip install tsagentkit-patchtst-fm
        is_tsfm=True,
        # PatchTST-FM: limits come from model config dynamically
        # https://huggingface.co/ibm-research/patchtst-fm-r1
        min_context_length=None,
        max_context_length=None,  # Resolved from model.config.context_length
        max_prediction_length=None,  # Resolved dynamically
        pad_to_min_context=True,
        truncate_to_max_context=False,  # Will error if too long
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


def list_models(tsfm_only: bool = False) -> list[str]:
    """List models from the registry.

    Args:
        tsfm_only: If True, only return TSFM models

    Returns:
        List of model names
    """
    names: list[str] = []
    for name, spec in REGISTRY.items():
        if tsfm_only and not spec.is_tsfm:
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


__all__ = [
    "ModelSpec",
    "REGISTRY",
    "list_models",
    "get_spec",
]
